import os
import subprocess
import sys
import argparse
import json
import time
import copy
import pickle
import numpy as np
from datetime import datetime
from enum import IntEnum, auto

from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

from eagent.model import Model
from eagent.trainer import Trainer
from eagent.config import cfg_dict

version = "0.8.8"


class Processor():
    def __init__(self, cfg, mpi_rank):
        self.mpi_rank = mpi_rank

        self.cfg = cfg
        print(self.cfg["initial_params_filename"])
        with open(self.cfg["initial_params_filename"], "r") as f:
            self.initial_params = json.load(f)

        # Determine if checkpoints are already in place
        if os.path.exists(os.path.join(cfg["output_dirname"], "checkpoint.json")):
            with open(os.path.join(cfg["output_dirname"], "checkpoint.json"), "r") as f:
                self.checkpoint = json.load(f)
        else:
            self.checkpoint = None

    class MpiTag(IntEnum):
        CONTINUE = auto()
        SAVE = auto()
        EXIT = auto()

    def follwer_process(self):
        cfg = self.cfg
        eval_averaged_policy = cfg["eval_averaged_policy"]

        # If there is a checkpoint, go read .pkl and other files
        if self.checkpoint is not None:
            model_path_dict = self.checkpoint[f"follower{self.mpi_rank}"]
            model = Model(cfg, self.initial_params, model_path_dict)
            print(f"{model_path_dict} is loaded")
        else:
            model = Model(cfg, self.initial_params)

        mpi_status = MPI.Status()
        while 1:
            packet = comm.recv(source=0, status=mpi_status)
            tag = mpi_status.Get_tag()
            if tag == self.MpiTag.EXIT:
                model.save_policy_model(packet)
                comm.send("done", dest=0)
                break
            elif tag == self.MpiTag.SAVE:
                model.save_policy_model(packet)
                comm.send("done", dest=0)
                continue
            structure_edges, structure_properties, policy_weights = packet
            model.set_params(structure_edges, structure_properties, policy_weights)
            # Run PPO in model and make the result
            model.learn_policy_model()

            # When evaluating averaged_policy, a round-trip communication with the leader occurs here
            if eval_averaged_policy:
                comm.send(model.get_flat_policy_weights(), dest=0)
                policy_weights = comm.recv(source=0, status=mpi_status)
                model.set_params(structure_edges, structure_properties, policy_weights)

            reward, contact_rate, success_rate = model.evaluate(
                num_episodes=self.cfg["num_episodes_in_eval"],
                num_steps=self.cfg["num_steps_in_eval"],
                use_elite=self.cfg["use_elite_in_eval"],
            )
            result = [reward, model.get_flat_policy_weights(), contact_rate, success_rate]
            result_packet = result
            comm.send(result_packet, dest=0)

    def leader_process(self):
        print("--------------------")
        print("Start leader_process")
        sys.stdout.flush()
        cfg = self.cfg

        # Keep cfg settings as local variables
        num_species = cfg["num_species"]
        num_individuals = cfg["num_individuals"]
        # num_substituted = cfg["num_substituted"]
        eval_averaged_policy = cfg["eval_averaged_policy"]

        # Save cfg
        output_dirname = cfg["output_dirname"]
        if not os.path.exists(output_dirname):
            os.makedirs(output_dirname)
        params_output_dirname = os.path.join(output_dirname, "params")
        if not os.path.exists(params_output_dirname):
            os.makedirs(params_output_dirname)
        with open(os.path.join(output_dirname, "cfg.json"), "w") as f:
            json.dump(cfg, f, indent=4)

        # If there is a checkpoint, go read .pkl and other files
        if self.checkpoint is not None:
            filename = os.path.join(output_dirname, self.checkpoint["leader"]["trainer"])
            with open(filename, "rb")as f:
                trainer = pickle.load(f)
            print(f"{filename} is loaded")

            filename = os.path.join(output_dirname, self.checkpoint["leader"]["history"])
            with open(filename, "r")as f:
                history = json.load(f)
            best_reward = history[-1]["best_reward"]
            best_generation = history[-1]["best_generation"]
        else:
            structure_edges_list = []
            structure_weights_list = []
            policy_weights_list = []
            for i in range(num_species):
                structure_edges_list.append(copy.deepcopy(self.initial_params["structure_edges"]))
                structure_weights_list.append(copy.deepcopy(self.initial_params["structure_weights"]))
                policy_weights_list.append([])
                for j in range(num_individuals):
                    # This initial value might be taken from followers.
                    policy_weights_list[i].append(copy.deepcopy(self.initial_params["policy_weights"]))
            trainer = Trainer(cfg, structure_edges_list, structure_weights_list, policy_weights_list)

            history = []
            best_reward = -99999.0
            best_generation = -1

        previous_time = time.time()
        while True:
            print("-----------------------------")
            trainer.generation += 1

            structure_edges_list, structure_weights_list, structure_properties_list, policy_weights_list = trainer.ask()

            # Send packets to follwer to perform mujoco simulation
            for i in range(num_species):
                for j in range(num_individuals):
                    follwer_rank = 1 + i * num_individuals + j
                    packet = [structure_edges_list[i], structure_properties_list[i][j], policy_weights_list[i][j]]
                    comm.send(packet, tag=int(self.MpiTag.CONTINUE), dest=follwer_rank)

            # When evaluating averaged_policy, another round-trip communication occurs here
            if eval_averaged_policy:
                for i in range(num_species):
                    policy_weights_list = []
                    for j in range(num_individuals):
                        follwer_rank = 1 + i * num_individuals + j
                        policy_weights = comm.recv(source=follwer_rank)
                        policy_weights_list.append(policy_weights)
                    averaged_policy_weights = np.mean(policy_weights_list, axis=0)
                    for j in range(num_individuals):
                        follwer_rank = 1 + i * num_individuals + j
                        comm.send(averaged_policy_weights, tag=int(self.MpiTag.CONTINUE), dest=follwer_rank)

            # Synchronously waits for the simulation to finish in the follwer and receives a packet
            result_list = []
            for i in range(num_species):
                result_list.append([])
                for j in range(num_individuals):
                    follwer_rank = 1 + i * num_individuals + j
                    result_packet = comm.recv(source=follwer_rank)
                    result = result_packet
                    result_list[i].append(result)

            # * IMPORTANT
            trainer.tell(result_list)

            # Storage and Display
            current_best_species_id, current_best_individual_id = trainer.get_current_best_ids()
            rewards_list = np.array([[result_list[i][j][0] for j in range(num_individuals)]
                                     for i in range(num_species)])
            current_eval_rewards = trainer.get_current_eval_rewards().tolist()
            current_best_eval_reward = current_eval_rewards[current_best_species_id]
            current_max_reward = float(np.max(rewards_list))
            current_mean_reward = float(np.mean(rewards_list))
            current_min_reward = float(np.min(rewards_list))
            current_best_structure_edges = structure_edges_list[current_best_species_id]
            current_best_structure_weights = structure_weights_list[current_best_species_id]
            current_best_policy_weights = trainer.get_policy_weights(
                current_best_species_id, current_best_individual_id)
            success_rate_list = [[result_list[i][j][3] for j in range(num_individuals)] for i in range(num_species)]
            # Save the current parameters at each generation
            with open(os.path.join(output_dirname, "parameter.json"), "w") as f:
                json.dump({
                    "structure_edges": current_best_structure_edges,
                    "structure_weights": current_best_structure_weights,
                    "policy_weights": current_best_policy_weights.tolist(),
                }, f)
            # Save the current parameters every few generations
            if trainer.generation % cfg["save_parameter_cycle"] == 0:
                with open(os.path.join(output_dirname, f"parameter_{trainer.generation}.json"), "w") as f:
                    json.dump({
                        "structure_edges": current_best_structure_edges,
                        "structure_weights": current_best_structure_weights,
                        "policy_weights": current_best_policy_weights.tolist(),
                    }, f)
                for i in range(num_species):
                    best_id = int(np.argmax(rewards_list[i]))
                    with open(os.path.join(params_output_dirname, f"parameter_{trainer.generation}_{i}.json"), "w") as f:
                        json.dump({
                            "structure_edges": structure_edges_list[i],
                            "structure_weights": structure_weights_list[i],
                            "policy_weights": trainer.get_policy_weights(i, best_id).tolist()
                        }, f)
            # If it's the highest score ever, save the parameters at that point
            if current_best_eval_reward > best_reward:
                best_reward = current_best_eval_reward
                best_generation = trainer.generation
                with open(os.path.join(output_dirname, "parameter_best.json"), "w") as f:
                    json.dump({
                        "structure_edges": current_best_structure_edges,
                        "structure_weights": current_best_structure_weights,
                        "policy_weights": current_best_policy_weights.tolist(),
                    }, f)

            # * IMPORTANT
            trainer.compute_next_params()

            # Viewing and saving logs
            # Basically, it displays local parameters, not the value in the trainer
            current_time = time.time()
            trainer.elapsed += current_time - previous_time
            previous_time = current_time
            info_dict = {
                "generation": trainer.generation,
                "best_reward": best_reward,
                "best_generation": best_generation,
                "elapsed": trainer.elapsed,
                "current_best_eval_reward": current_best_eval_reward,
                "current_best_reward": current_max_reward,
                "current_mean_reward": current_mean_reward,
                "current_min_reward": current_min_reward,
                "current_best_species_id": current_best_species_id,
                "current_best_individual_id": current_best_individual_id,
                "num_limbs": [len(x) for x in structure_edges_list],
                "structure_codes": trainer.tree_codes,
                "current_eval_rewards": current_eval_rewards,
                "current_mean_rewards": np.mean(rewards_list, axis=1).tolist(),
                "current_rewards": rewards_list.tolist(),
                "success_rate": success_rate_list,
            }
            for key, val in info_dict.items():
                print(f"- {key}: {val}")
            history.append(info_dict)
            # Save logs at every generation
            with open(os.path.join(output_dirname, "history.json"), "w") as f:
                json.dump(history, f, indent=4)

            sys.stdout.flush()

            completed = trainer.generation >= cfg["max_generation"]
            # Save each object once every few generations
            if trainer.generation % cfg["checkpoint_cycle"] == 0 or completed:
                checkpoint = {}

                pickles_dirname = os.path.join(output_dirname, "pickles")
                if not os.path.exists(pickles_dirname):
                    os.makedirs(pickles_dirname)
                zips_dirname = os.path.join(output_dirname, "zips")
                if not os.path.exists(zips_dirname):
                    os.makedirs(zips_dirname)

                follwer_rank = 1
                for i in range(num_species):
                    best_j = int(np.argmax(rewards_list[i]))
                    params_json_basename = f"parameter_{trainer.generation}_{i}.json"
                    with open(os.path.join(params_output_dirname, params_json_basename), "w") as f:
                        json.dump({
                            "structure_edges": structure_edges_list[i],
                            "structure_weights": structure_weights_list[i],
                            "policy_weights": result_list[i][best_j][1].tolist(),
                        }, f)
                    for j in range(num_individuals):
                        model_path_dict = {
                            "params_jsonname": f"params/{params_json_basename}",
                            "model_zipname": f"zips/follower{follwer_rank}_policy_model.zip"
                        }
                        if "her" in cfg["rl_cfg"]["algorithm"]:
                            model_path_dict["replay_buffer_pklname"] = f"pickles/follower{follwer_rank}_replay_buffer.pkl"

                        checkpoint[f"follower{follwer_rank}"] = model_path_dict

                        # Send policy_model_path_dict to the follwer to store the current model_policy for each
                        if completed:
                            tag = int(self.MpiTag.EXIT)
                        else:
                            tag = int(self.MpiTag.SAVE)
                        comm.send(model_path_dict, tag=tag, dest=follwer_rank)

                        # Synchronously wait for the follwer to send the model_policy save completion signal
                        _ = comm.recv(source=follwer_rank)

                        follwer_rank += 1

                basename = "pickles/leader_trainer.pkl"
                checkpoint["leader"] = {"trainer": basename}
                filename = os.path.join(output_dirname, f"trainer_{trainer.generation}.pkl")
                with open(os.path.join(output_dirname, basename), "wb") as f:
                    pickle.dump(trainer, f)

                checkpoint["leader"]["history"] = "history_temp.json"
                with open(os.path.join(output_dirname, "history_temp.json"), "w") as f:
                    json.dump(history, f)

                checkpoint["status"] = {"done": False}

                with open(os.path.join(output_dirname, "checkpoint.json"), "w") as f:
                    json.dump(checkpoint, f, indent=4)

                print("checkpoint is saved")

            if completed:
                return 0


def mpi_fork(n, output_dirname):
    # Execute n new sub-processes, excluding the currently running process
    # We make the other subprocesses wait with the first one as parent
    # Of the other n children, the one with a rank of 0 is used as the leader and the others as followers
    assert n > 1

    rank = comm.Get_rank()

    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        if os.name == "nt":
            # On Windows, use Microsoft MPI (https://www.microsoft.com/en-us/download/details.aspx?id=57467)
            cmd = ["mpiexec", "-l", "-np", str(n), sys.executable] \
                + ["-u"] + sys.argv + ["-o", output_dirname]
        else:
            # On Ubuntu, use mpich
            cmd = ["mpiexec", "--oversubscribe", "-np", str(n), sys.executable] \
                + ["-u"] + sys.argv + ["-o", output_dirname]
        print(cmd)
        # Wait for all sub-processes to terminate
        subprocess.check_call(cmd, env=env)
        return "parent", rank
    else:
        rank = comm.Get_rank()
        if rank == 0:
            status = "child_leader"
        else:
            status = "child_follwer"
        return status, rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg_filename", type=str, default=None)
    # If cfg.json exists in this directory, it takes priority
    parser.add_argument("-o", "--output_dirname", type=str, default=None)
    args = parser.parse_args()

    if args.cfg_filename is None:
        cfg = {}
    else:
        cfg = cfg_dict[args.cfg_filename]
    if args.output_dirname is None:
        cfg["output_dirname"] = os.path.join("log", version + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    else:
        cfg["output_dirname"] = args.output_dirname
    cfg_filename = os.path.join(cfg["output_dirname"], "cfg.json")
    if os.path.exists(cfg_filename):
        with open(cfg_filename, "r") as f:
            cfg.update(json.load(f))
    assert len(cfg.keys()) > 0

    num_follwer = cfg["num_species"] * cfg["num_individuals"]
    mpi_status, mpi_rank = mpi_fork(num_follwer + 1, cfg["output_dirname"])
    print(f"status: {mpi_status}, rank: {mpi_rank} (0-{num_follwer})")
    if (mpi_status == "parent"):
        print("Finished training")
    elif (mpi_status == "child_leader"):
        Processor(cfg, mpi_rank).leader_process()
    else:
        Processor(cfg, mpi_rank).follwer_process()


if __name__ == "__main__":
    main()
