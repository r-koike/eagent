import numpy as np
import time
import json  # noqa
import os
import copy

import torch
from stable_baselines3 import PPO, DDPG, SAC, HerReplayBuffer
import gym
from gym.wrappers import FilterObservation, FlattenObservation

from .trainer import hard_sigmoid


class Model:
    def __init__(self, cfg, initial_params, model_path_dict={}):
        # If both are not None, model_path_dict takes priority over initial_params
        self.cfg = cfg
        self.env_name = cfg["env_name"]

        params = copy.deepcopy(initial_params)
        if "params_jsonname" in model_path_dict.keys():
            params_jsonname = os.path.join(cfg["output_dirname"], model_path_dict["params_jsonname"])
            with open(params_jsonname, "r") as f:
                params = json.load(f)

        structure_edges = params["structure_edges"]
        structure_weights = params["structure_weights"]
        structure_properties = hard_sigmoid(np.array(structure_weights["mu"]))
        policy_weights = np.array(params["policy_weights"])

        env = gym.make(
            self.env_name,
            structure_edges=structure_edges,
            structure_properties=structure_properties,
            max_num_limbs=cfg["max_num_limbs"],
            env_cfg=cfg["env_specific_cfg"]
        )
        if "Hand" in self.env_name and cfg["rl_cfg"]["algorithm"] == "ppo":
            self.env = FlattenObservation(FilterObservation(env, ["observation", "desired_goal"]))
        else:
            self.env = env

        self.policy_model = self.__make_policy_model()
        # with open("output.json", "w") as f:
        #     json.dump(self.get_flat_policy_weights().tolist(), f)
        self.load_policy_model(model_path_dict)

        self.set_policy(policy_weights)

    def __del__(self):
        self.env.close()

    def __make_policy_model(self):
        cfg = self.cfg
        rl_cfg = cfg["rl_cfg"]
        policy_kwargs = cfg["policy_kwargs"].copy()
        if policy_kwargs["activation_fn"] == "tanh":
            policy_kwargs["activation_fn"] = torch.nn.Tanh
        elif policy_kwargs["activation_fn"] == "relu":
            policy_kwargs["activation_fn"] = torch.nn.ReLU
        else:
            policy_kwargs["activation_fn"] = torch.nn.Tanh
        if rl_cfg["tb_log"] == "None":
            rl_cfg["tb_log"] = None

        # Even if no log is sent out, only the dir setting should be done
        # Without this, tempfile.gettempdir() will be called many times in SB3 and it will slow down
        os.environ["SB3_LOGDIR"] = os.path.join("log", "temp")

        if rl_cfg["algorithm"] == "ppo":
            assert cfg["policy"] == "MlpPolicy"
            return PPO(
                policy="MlpPolicy",
                env=self.env,
                policy_kwargs=policy_kwargs,
                verbose=cfg["policy_verbose"],
            )
        elif rl_cfg["algorithm"] == "ddpg":
            assert cfg["policy"] == "MultiInputPolicy"
            return DDPG(
                policy=cfg["policy"],
                env=self.env,
                policy_kwargs=policy_kwargs,
                verbose=cfg["policy_verbose"],
            )
        elif rl_cfg["algorithm"] == "ddpg_her":
            assert cfg["policy"] == "MultiInputPolicy"
            assert rl_cfg["goal_selection_strategy"] in ["future", "final", "episode"]
            return DDPG(
                policy=cfg["policy"],
                env=self.env,
                batch_size=rl_cfg["batch_size"],
                learning_starts=rl_cfg["learning_starts"],
                gamma=rl_cfg["gamma"],
                tau=rl_cfg["tau"],
                learning_rate=rl_cfg["learning_rate"],
                policy_kwargs=policy_kwargs,
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=dict(
                    n_sampled_goal=4,
                    goal_selection_strategy=rl_cfg["goal_selection_strategy"],
                    online_sampling=rl_cfg["online_sampling"],
                ),
                tensorboard_log=rl_cfg["tb_log"],
                verbose=cfg["policy_verbose"],
            )
        elif rl_cfg["algorithm"] == "sac_her":
            assert cfg["policy"] == "MultiInputPolicy"
            assert rl_cfg["goal_selection_strategy"] in ["future", "final", "episode"]
            return SAC(
                policy=cfg["policy"],
                env=self.env,
                buffer_size=rl_cfg["buffer_size"],
                batch_size=rl_cfg["batch_size"],
                learning_starts=rl_cfg["learning_starts"],
                gamma=rl_cfg["gamma"],
                tau=rl_cfg["tau"],
                learning_rate=rl_cfg["learning_rate"],
                train_freq=(1, "episode"),  # Defaults to (1, "episode") in DDPG, but defaults to (1, "step") in SAC
                gradient_steps=-1,  # Defaults to -1 in DDPG, but defaults to 1 in SAC
                policy_kwargs=policy_kwargs,
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=dict(
                    n_sampled_goal=4,
                    goal_selection_strategy=rl_cfg["goal_selection_strategy"],
                    online_sampling=rl_cfg["online_sampling"],
                ),
                tensorboard_log=rl_cfg["tb_log"],
                verbose=cfg["policy_verbose"],
            )
        else:
            raise NotImplementedError

    def load_policy_model(self, model_path_dict):
        cfg = self.cfg
        if "model_zipname" in model_path_dict.keys():
            zipname = os.path.join(cfg["output_dirname"], model_path_dict["model_zipname"])
            self.policy_model = self.policy_model.load(zipname, env=self.env)
        if "replay_buffer_pklname" in model_path_dict.keys():
            pklname = os.path.join(cfg["output_dirname"], model_path_dict["replay_buffer_pklname"])
            self.policy_model.load_replay_buffer(pklname)

    def save_policy_model(self, model_path_dict):
        cfg = self.cfg
        if "model_zipname" in model_path_dict.keys():
            zipname = os.path.join(cfg["output_dirname"], model_path_dict["model_zipname"])
            self.policy_model.save(zipname)
        if "replay_buffer_pklname" in model_path_dict.keys():
            pklname = os.path.join(cfg["output_dirname"], model_path_dict["replay_buffer_pklname"])
            self.policy_model.save_replay_buffer(pklname)

    def learn_policy_model(self):
        cfg = self.cfg
        if cfg["rl_cfg"]["algorithm"] in ["ppo", "ddpg", "ddpg_her", "sac_her"]:
            self.policy_model.learn(
                total_timesteps=cfg["rl_cfg"]["num_steps_in_learn"],
            )
        else:
            raise NotImplementedError

    def set_params(self, structure_edges, structure_properties, policy_weights, ignore_edges_update=False):
        edges_were_changed = not (sorted(self.env.get_structure_edges()) == sorted(structure_edges))
        self.env.set_structure_params(structure_edges, structure_properties)
        self.env.reset()
        # * When structure_edges changes, we want to reset the parameters for learning PPO, so we rebuild policy_model
        if self.cfg["reset_ppo_on_edges_selection"] and edges_were_changed and not ignore_edges_update:
            # Because env has changed, policy_model.env needs to be updated after self.env.reset()
            self.policy_model = self.__make_policy_model()
        self.set_policy(policy_weights)

    # params is a dictionary obtained by model.policy_model.get_parameters()
    def set_policy(self, flat_weights, params=None):
        if params is None:
            params = self.policy_model.get_parameters()
        new_params = self.__insert_flat_weights_to_params_of_policy(flat_weights, params)
        self.policy_model.set_parameters(new_params)

    def __insert_flat_weights_to_params_of_policy(self, flat_weights, params):
        # layer is a mutable object, so it behaves like a reference pass
        # idx is an immutable object, so it behaves like a value pass
        def __insert_to_layer(layer, flat_weights, idx):
            size = torch.numel(layer)
            flat = flat_weights[idx:idx + size]
            idx += size
            layer.data = torch.from_numpy(np.reshape(flat, layer.shape)).clone().float()
            return idx

        idx = 0
        for key, layer in params["policy"].items():
            if key == "log_std":
                continue
            idx += torch.numel(layer)
        assert len(flat_weights) == idx, f"inserting weights: {len(flat_weights)}, required: {idx}"

        idx = 0
        for key, layer in params["policy"].items():
            if key == "log_std":
                continue
            idx = __insert_to_layer(layer, flat_weights, idx)
        return params

    def get_flat_policy_weights(self):
        # Be careful to keep consistency with self.__insert_flat_weights_to_params_of_policy()
        flat_weights_list = []
        params = self.policy_model.get_parameters()
        for key, layer in params["policy"].items():
            if key == "log_std":
                continue
            flat_weights_list.append(layer.data.to("cpu").detach().numpy().flatten())
        flat_weights = np.concatenate(flat_weights_list)
        return flat_weights

    def get_action(self, obs):
        act, _states = self.policy_model.predict(obs, deterministic=True)
        return act

    def evaluate(self, num_episodes, num_steps, use_elite=False):
        rewards = []
        contact_rates = []
        num_success = 0
        for _ in range(num_episodes):
            r, c, is_success = self.simulate_once(
                render_mode=False,
                num_steps=num_steps,
            )
            rewards.append(r)
            contact_rates.append(c)
            num_success += int(is_success)
        contact_rate = np.mean(contact_rates, axis=0)
        success_rate = num_success / num_episodes
        if use_elite:
            return np.array(rewards).max(), contact_rate, success_rate
        else:
            return np.array(rewards).mean(), contact_rate, success_rate

    def simulate_once(self, render_mode=True, seed=-1, num_steps=1000):
        if render_mode:
            print("----------")
            print("simulate")
            print(f"env_name: {self.env_name}")
            self.env.render("human")
        if (seed >= 0):
            np.random.seed(seed)
            self.env.seed(seed)

        obs = self.env.reset()

        total_reward = 0.0
        contact_states = []
        is_success = False
        for t in range(num_steps):
            if render_mode:
                self.env.render("human")
                time.sleep(0.01)

            act = self.get_action(obs)
            obs, reward, done, info = self.env.step(act)
            contact_state = self.env.get_contact_state()
            contact_states.append(contact_state)

            if "is_success" in info.keys() and info["is_success"]:
                is_success = True

            total_reward += reward

            if done:
                break
        contact_rate = np.mean(contact_states, axis=0)

        if render_mode:
            print("----------")
            print(f"reward: {total_reward}, timesteps: {t}")

        return total_reward, contact_rate, is_success
