import numpy as np
import copy
import os
import json

from .tree_selector import TreeSelector

# For `OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.`
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def hard_sigmoid(x):
    y = 0.2 * x + 0.5
    return np.clip(y, 0.0, 1.0)


def inv_hard_sigmoid(y):
    x = 5.0 * y - 2.5
    return x


class Optimizer(object):
    # pi = {"mu": [...], "sigma": [...]}
    def __init__(self, pi, epsilon=1e-08):
        self.pi = pi
        self.dim = sum([len(x) for x in pi.values()])
        self.t = 0
        self.epsilon = epsilon

    def update(self, globalg):
        self.t += 1
        step = self._compute_step(globalg)
        theta = np.concatenate([x for x in self.pi.values()])
        ratio = np.linalg.norm(step) / (np.linalg.norm(theta) + self.epsilon)
        new_pi = theta + step
        idx = 0
        for key, value in self.pi.items():
            self.pi[key] = new_pi[idx:idx + len(value)].tolist()
            idx += len(value)
        assert idx == len(new_pi)
        return ratio

    def _compute_step(self, globalg):
        raise NotImplementedError


class Adam(Optimizer):
    def __init__(self, pi, stepsize, beta1=0.9, beta2=0.999):
        Optimizer.__init__(self, pi)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_step(self, globalg):
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step


class Trainer():
    def __init__(self, cfg, structure_edges_list, structure_weights_list, policy_weights_list):
        self.cfg = cfg
        self.elapsed = 0
        self.generation = 0
        # Basically create a list in order to dump in json
        # Only policy_weights is created in ndarray
        self.structure_edges_list = structure_edges_list
        self.structure_weights_list = structure_weights_list
        self.policy_weights_list = policy_weights_list

        structure_lr = cfg["structure_lr"]
        num_species = len(self.policy_weights_list)
        self.optimizers = []
        for species_id in range(num_species):
            # reference pass
            self.optimizers.append(Adam(self.structure_weights_list[species_id], structure_lr))

        self.best_reward = -999999
        self.best_structure_edges = copy.deepcopy(self.structure_edges_list[0])
        self.best_structure_weights = copy.deepcopy(self.structure_weights_list[0])
        self.best_policy_weights_list = copy.deepcopy(self.policy_weights_list[0])

        # structure_improvement related parameters
        self.best_reward_in_current_edges = [-999999] * num_species
        self.reward_stall_streaks = [0] * num_species
        self.prev_edges_selection_generations = [0] * num_species
        self.eval_reward_list_history = []
        self.pre_eliminated_ids = []
        self.pre_eval_contact_rate = np.array([1.0 for _ in range(cfg["max_num_limbs"] + 1)])

        self.tree_log = {}

        # Use tree_selector for discrete structure changes other than contact
        self.update_pivot = (cfg["do_edges_selection"] and cfg["edges_selection_criteria"] in ["fitting", "patient"])
        if cfg["do_edges_selection"] and cfg["edges_selection_criteria"] in ["fitting", "patient", "fitting_rand"]:
            self.tree_selector = TreeSelector(cfg["max_num_limbs"], cfg["use_2dof"],
                                              require_code_graph=self.update_pivot)
            if self.update_pivot:
                self.tree_selector.update_pivot(self.best_structure_edges)
            self.tree_codes = [""] * num_species
            self.tree_codes[0] = self.tree_selector.edges2code(self.best_structure_edges)
            self.__edges_selection(range(1, num_species), cfg)
        else:
            # No tree_selector is provided
            self.tree_selector = None
            self.tree_codes = [""] * num_species
            code = "10"
            for i in range(num_species):
                self.tree_codes[i] = copy.deepcopy(code)

    def ask(self):
        cfg = self.cfg
        num_individuals = len(self.policy_weights_list[0])

        structure_properties_list = []
        for structure_weights in self.structure_weights_list:
            mu = structure_weights["mu"]
            sigma = structure_weights["sigma"]
            structure_properties_list.append([])
            for i in range(num_individuals):
                if cfg["do_structure_improvement"]:
                    # Sample following to a normal distribution
                    epsilon = np.random.randn(len(sigma)) * sigma
                else:
                    epsilon = np.zeros(len(sigma))
                properties = hard_sigmoid(mu + epsilon)
                structure_properties_list[-1].append(properties)
        self.structure_properties_list = structure_properties_list

        return self.structure_edges_list, self.structure_weights_list, self.structure_properties_list, self.policy_weights_list

    # Only policy_weights is updated
    def tell(self, result_list):
        cfg = self.cfg

        num_species = len(self.policy_weights_list)
        num_individuals = len(self.policy_weights_list[0])
        rewards_list = np.array([[result_list[i][j][0] for j in range(num_individuals)]
                                 for i in range(num_species)])
        self.rewards_list = rewards_list
        self.contact_rate_list = np.array([[result_list[i][j][2] for j in range(num_individuals)]
                                           for i in range(num_species)])

        # Apply PPO results in self.policy_weights_list
        policy_weights_list = np.array([[x[j][1] for j in range(len(x))]for x in result_list])
        for i in range(num_species):
            if cfg["use_averaged_policy"]:
                averaged_policy_weights = np.mean(policy_weights_list[i], axis=0)
                for j in range(num_individuals):
                    self.policy_weights_list[i][j] = averaged_policy_weights
            else:
                for j in range(num_individuals):
                    self.policy_weights_list[i][j] = result_list[i][j][1]

        # Whether edges_selection is done or not, eval_reward should be created (because it is needed in history)
        # If using elite, take max, otherwise take mean (considering max and mean in individuals)
        if cfg["use_elite_in_eval_reward"]:
            eval_reward_list = np.max(rewards_list, axis=1)
        else:
            eval_reward_list = np.mean(rewards_list, axis=1)
        # EMA
        if len(self.eval_reward_list_history) > 0:
            pre_eval_reward_list = self.eval_reward_list_history[-1]
            for i in range(num_species):
                if i in self.pre_eliminated_ids:
                    if type(cfg["ema_init_value"]) is str:
                        pre_eval_reward_list[i] = eval_reward_list[i]
                    else:
                        pre_eval_reward_list[i] = cfg["ema_init_value"]
        else:
            if type(cfg["ema_init_value"]) is str:
                pre_eval_reward_list = eval_reward_list
            else:
                pre_eval_reward_list = np.array([cfg["ema_init_value"] for i in range(len(eval_reward_list))])
        num_ema = cfg["num_ema_in_edges_selection"]
        smoothed_eval_reward_list = pre_eval_reward_list + \
            (2 / (num_ema + 1)) * (eval_reward_list - pre_eval_reward_list)
        self.eval_reward_list_history.append(smoothed_eval_reward_list)

        # Find the best species and individuals
        if cfg["how_to_select_best_species"] == "smoothed_eval":
            best_species_id = int(np.argmax(smoothed_eval_reward_list))
            best_individual_id = int(np.argmax(rewards_list[best_species_id]))
        elif "mean":
            species_rewards = np.mean(rewards_list, axis=1)
            best_species_id = int(np.argmax(species_rewards))
            best_individual_id = int(np.argmax(rewards_list[best_species_id]))
        elif "max":
            species_rewards = np.max(rewards_list, axis=1)
            best_species_id = int(np.argmax(species_rewards))
            best_individual_id = int(np.argmax(rewards_list[best_species_id]))
        self.current_best_speceis_id = best_species_id
        self.current_best_individual_id = best_individual_id

        # Determine if the best has been created compared to previous generations
        if rewards_list[best_species_id][best_individual_id] > self.best_reward:
            self.best_reward = rewards_list[best_species_id][best_individual_id]
            self.best_structure_edges = copy.deepcopy(self.structure_edges_list[best_species_id])
            self.best_structure_weights = copy.deepcopy(self.structure_weights_list[best_species_id])
            self.best_policy_weights_list = copy.deepcopy(self.policy_weights_list[best_species_id])
            if self.update_pivot:
                self.tree_selector.update_pivot(self.best_structure_edges)

    def compute_next_params(self):
        cfg = self.cfg
        num_species = len(self.policy_weights_list)
        num_individuals = len(self.policy_weights_list[0])
        rewards_list = self.rewards_list

        # Update self.structure_weights_list with population-based REINFORCE
        if cfg["do_structure_improvement"]:
            sigma_max_change = cfg["structure_sigma_max_change"]
            sigma_limit = cfg["structure_sigma_limit"]
            for species_id in range(num_species):
                edges = self.structure_edges_list[species_id]
                properties = self.structure_properties_list[species_id]
                mu = np.array(self.structure_weights_list[species_id]["mu"])
                sigma = np.array(self.structure_weights_list[species_id]["sigma"])

                rewards = rewards_list[species_id]
                reward_average = np.mean(rewards)
                reward_std = np.std(rewards)

                mu_grad_list = []
                sigma_grad_list = []
                for individual_id in range(num_individuals):
                    # Use alpha*sigma^2
                    reward = rewards[individual_id]
                    r = (reward - reward_average) / (reward_std + 1e-08)
                    perturbed_mu = inv_hard_sigmoid(properties[individual_id])
                    mu_grad_list.append(r * (perturbed_mu - mu))
                    sigma_grad_list.append(r * (np.square(perturbed_mu - mu) - np.square(sigma)) / sigma)
                mu_grad = np.mean(mu_grad_list, axis=0)
                sigma_grad = np.mean(sigma_grad_list, axis=0)

                mu_diff = -mu_grad
                sigma_diff = -sigma_grad
                sigma_diff = np.minimum(sigma_diff, sigma_max_change * sigma)
                sigma_diff = np.maximum(sigma_diff, -sigma_max_change * sigma)

                # Set the amount of change to 0 for the elements corresponding to a rigid bodys that are not present
                is_vacant = [True] * len(mu)
                num_diff_params = 0
                for parent, child, dof in edges:
                    is_vacant[child * 9 + 0] = False
                    is_vacant[child * 9 + 1] = False
                    is_vacant[child * 9 + 2] = False
                    is_vacant[child * 9 + 7] = False
                    is_vacant[child * 9 + 8] = False
                    num_diff_params += 5
                    if dof >= 1:
                        is_vacant[child * 9 + 3] = False
                        is_vacant[child * 9 + 4] = False
                        num_diff_params += 2
                    if dof >= 2:
                        is_vacant[child * 9 + 5] = False
                        is_vacant[child * 9 + 6] = False
                        num_diff_params += 2
                mu_diff[is_vacant] = [0.0] * (len(mu) - num_diff_params)
                sigma_diff[is_vacant] = [0.0] * (len(mu) - num_diff_params)
                # print("mu_diff: ", mu_diff[:10])
                # print("sigma_diff: ", sigma_diff[:10])

                # * IMPORTANT: In the optimizer, self.structure_weights_list[species_id] is updated
                _ = self.optimizers[species_id].update(np.concatenate([mu_diff, sigma_diff]))
                self.structure_weights_list[species_id]["sigma"] = np.maximum(
                    self.structure_weights_list[species_id]["sigma"], np.array([sigma_limit] * len(sigma))).tolist()

        if cfg["do_policy_selection"] and self.generation % cfg["policy_selection_cycle"] == 0:
            for species_id in range(num_species):
                rewards = rewards_list[species_id]
                eliminated_id_list = np.argsort(rewards)[:cfg["num_eliminated_in_policy_selection"]]
                rewards_ranking = np.argsort(-rewards)
                cnt = 0
                for i in eliminated_id_list:
                    # Bring the required number of weights in order from best to worst
                    policy_parent_id = rewards_ranking[cnt]
                    cnt += 1
                    self.policy_weights_list[species_id][i] = copy.deepcopy(
                        self.policy_weights_list[species_id][policy_parent_id])

        # Update self.structure_edges_list
        if cfg["do_edges_selection"]:
            eliminated_ids = []
            edges_selection_criteria = cfg["edges_selection_criteria"]
            edges_cfg = cfg["edges_selection_params"][edges_selection_criteria]
            if edges_selection_criteria == "patient":
                assert cfg["num_species"] > 1
                for i in range(num_species):
                    eval_reward = self.eval_reward_list_history[-1][i]
                    if eval_reward > self.best_reward_in_current_edges[i]:
                        self.best_reward_in_current_edges[i] = eval_reward
                        self.reward_stall_streaks[i] = 0
                    else:
                        self.reward_stall_streaks[i] += 1
                    if edges_cfg["patience"] > 0 and self.reward_stall_streaks[i] > edges_cfg["patience"]:
                        # Evolution was identified as stalled
                        eliminated_ids.append(i)
                    elif (self.generation - self.prev_edges_selection_generations[i]) % cfg["edges_selection_cycle"] == 0:
                        # A certain number of generations have passed since the last evolution
                        eliminated_ids.append(i)
                # * Change the number of rigid bodies
                self.__edges_selection(eliminated_ids, cfg)
            elif edges_selection_criteria in ["fitting", "fitting_rand"]:
                # if edges_selection_criteria == "fitting":
                #     assert cfg["num_species"] > 1
                eval_reward_list_history = np.array(self.eval_reward_list_history)
                for i in range(num_species):
                    if self.generation - self.prev_edges_selection_generations[i] < edges_cfg["sight"]:
                        continue
                    f = eval_reward_list_history[-edges_cfg["sight"]:, i]
                    x = np.arange(len(f))
                    a, _ = np.polyfit(x, f, 1)
                    print(f"slope_of_{i}: {a}")
                    if a < edges_cfg["slope_threshold"]:
                        eliminated_ids.append(i)
                    elif (self.generation - self.prev_edges_selection_generations[i]) % cfg["edges_selection_cycle"] == 0:
                        # A certain number of generations have passed since the last evolution
                        eliminated_ids.append(i)
                # * Change the number of rigid bodies
                self.__edges_selection(eliminated_ids, cfg)
            elif edges_selection_criteria in ["contact", "contact_fitting"]:
                for i in range(num_species):
                    if self.generation - self.prev_edges_selection_generations[i] < edges_cfg["decrease_interval"]:
                        continue
                    edges = np.array(self.structure_edges_list[i])
                    parent_nodes = edges[:, 0]
                    child_nodes = edges[:, 1]
                    leaf_nodes = np.setdiff1d(child_nodes, parent_nodes)

                    contact_rate = np.mean(self.contact_rate_list[i], axis=0)
                    num_ema = edges_cfg["num_ema_in_contact"]
                    eval_contact_rate = self.pre_eval_contact_rate + \
                        (2 / (num_ema + 1)) * (contact_rate - self.pre_eval_contact_rate)
                    self.pre_eval_contact_rate = eval_contact_rate
                    print(eval_contact_rate)

                    removed_nodes = []
                    if edges_selection_criteria == "contact":
                        for k in leaf_nodes:
                            if eval_contact_rate[k] < edges_cfg["min_contact_rate"]:
                                removed_nodes.append(k)
                    elif edges_selection_criteria == "contact_fitting":
                        # If the contact rate is less than the threshold, remove that rigid body
                        # However, unless the slope is smaller than the threshold, it will not be deleted
                        eval_reward_list_history = np.array(self.eval_reward_list_history)
                        if self.generation - self.prev_edges_selection_generations[i] < edges_cfg["sight"]:
                            continue
                        f = eval_reward_list_history[-edges_cfg["sight"]:, i]
                        x = np.arange(len(f))
                        a, _ = np.polyfit(x, f, 1)
                        if a < edges_cfg["slope_threshold"]:
                            for k in leaf_nodes:
                                if eval_contact_rate[k] < edges_cfg["min_contact_rate"]:
                                    removed_nodes.append(k)
                    if len(removed_nodes) > 0:
                        new_edges = []
                        for e in edges:
                            if e[0] not in removed_nodes and e[1] not in removed_nodes:
                                new_edges.append([int(x) for x in e])
                        # If the number of rigid bodies is going to be less than 3, then that change is aborted
                        if len(new_edges) < 3:
                            print("Warning: Rigid evolution is abandoned because of minimum number of edges. ")
                            new_edges = edges.tolist()
                        else:
                            eliminated_ids.append(i)
                            self.structure_edges_list[i] = new_edges
                            self.best_reward_in_current_edges[i] = -999999
                            self.reward_stall_streaks[i] = 0
                            self.prev_edges_selection_generations[i] = self.generation
                            self.__reflesh_optimizer_params(i)
                            self.__save_current_tree(i)
            else:
                raise NotImplementedError
            self.pre_eliminated_ids = eliminated_ids

    def __edges_selection(self, eliminated_ids, cfg):
        num_individuals = cfg["num_individuals"]
        edges_selection_criteria = cfg["edges_selection_criteria"]
        for i in eliminated_ids:
            self.best_reward_in_current_edges[i] = -999999
            self.reward_stall_streaks[i] = 0
            self.prev_edges_selection_generations[i] = self.generation

            # Select a new discrete morphology
            if edges_selection_criteria == "fitting_rand":
                old_edges = np.array(self.structure_edges_list[i])
                new_edges = np.array(self.tree_selector.select_random_changed_edges(old_edges))
                # Set the previous robot as parent
                self.structure_edges_list[i] = [[int(x) for x in x1] for x1 in new_edges]
            else:
                old_edges = np.array(self.best_structure_edges)
                new_edges = np.array(self.tree_selector.select_next_edges())
                # Set the best robot ever as parent
                self.structure_edges_list[i] = [[int(x) for x in x1] for x1 in new_edges]
                # * Do not copy the entire self.structure_weights_list[i] so as not to break the optimizer's references
                self.structure_weights_list[i]["mu"] = copy.deepcopy(self.best_structure_weights["mu"])
                self.structure_weights_list[i]["sigma"] = copy.deepcopy(self.best_structure_weights["sigma"])
                for j in range(num_individuals):
                    self.policy_weights_list[i][j] = copy.deepcopy(self.best_policy_weights_list[j])
            removed_nodes = np.setdiff1d(old_edges[:, 1], new_edges[:, 1])
            added_nodes = np.setdiff1d(new_edges[:, 1], old_edges[:, 1])

            for removed_node in removed_nodes:
                # Restore the corresponding structure_weights_list to its initial value
                k = removed_node * 9
                self.structure_weights_list[i]["mu"][k:k + 9] = [-2.5, -2.5, -2.5, -2.5, -2.5,
                                                                 -2.5, -2.5, -2.5, -2.5]
                self.structure_weights_list[i]["sigma"][k: k + 9] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
            for added_node in added_nodes:
                # Select and bring the corresponding structure_weights_list from other rigid bodies
                weights_parent_node = np.random.choice(old_edges[:, 1])
                k1 = added_node * 9
                k2 = weights_parent_node * 9
                self.structure_weights_list[i]["mu"][k1: k1 + 9] = copy.deepcopy(
                    self.structure_weights_list[i]["mu"][k2:k2 + 9])
                self.structure_weights_list[i]["sigma"][k1:k1 + 9] = [0.1, 0.1, 0.1, 0.1, 0.1,
                                                                      0.1, 0.1, 0.1, 0.1]

            # Make policy_weights
            original_policy_weights_list = np.array(self.best_policy_weights_list)
            for j in range(num_individuals):
                # Select two parameters from original and make their average the new policy_weights
                policy_parent_ids = np.random.choice(range(num_individuals), 2)
                self.policy_weights_list[i][j] = np.mean(
                    original_policy_weights_list[policy_parent_ids], axis=0)

            self.tree_codes[i] = self.tree_selector.edges2code(new_edges)
            self.__reflesh_optimizer_params(i)
            self.__save_current_tree(i)

    def __reflesh_optimizer_params(self, i):
        # Reset sigma
        if self.cfg["reset_sigma_on_edges_selection"]:
            self.structure_weights_list[i]["sigma"] = [0.1] * len(self.structure_weights_list[i]["sigma"])

        # Reset optimizer
        if self.cfg["reset_optimizer_on_edges_selection"]:
            # call by reference
            self.optimizers[i] = Adam(self.structure_weights_list[i], self.cfg["structure_lr"])

    def __save_current_tree(self, i):
        self.tree_log[f"{self.generation}_{i}"] = self.structure_edges_list[i]
        if self.cfg["save_every_edges"]:
            filename = os.path.join(self.cfg["output_dirname"], "edges.json")
            with open(filename, "w") as f:
                json.dump(self.tree_log, f, indent=4)

    def get_current_best_ids(self):
        return self.current_best_speceis_id, self.current_best_individual_id

    def get_current_eval_rewards(self):
        return self.eval_reward_list_history[-1]

    def get_policy_weights(self, i, j):
        return self.policy_weights_list[i][j]
