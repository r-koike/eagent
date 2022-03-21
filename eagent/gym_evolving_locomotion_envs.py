import numpy as np
import tempfile
import os

from gym.envs.mujoco import mujoco_env
from gym import utils, spaces

from .evolving_tools import EvolvingTools
from .mujoco_xml_generater import WalkerXmlGenerater


class EvolvingWalkerEnv(mujoco_env.MujocoEnv, utils.EzPickle, EvolvingTools):
    def __init__(self, structure_edges, structure_properties, max_num_limbs, env_cfg, render=False):
        self.env_cfg = env_cfg
        self.max_num_limbs = max_num_limbs

        EvolvingTools.__init__(self, max_num_limbs, structure_edges, structure_properties)
        utils.EzPickle.__init__(self)
        self.__reset_env()

    def __reset_env(self):
        # Create an xml file and delete it immediately after loading
        xml_data = WalkerXmlGenerater(self.env_cfg).generate_xml(
            self.structure_tree,
            self.structure_properties,
            self.dofs,
        )
        fd, temp_fullname = tempfile.mkstemp(
            text=True,
            suffix=".xml",
            dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "mujoco_assets", "temp")
        )
        os.close(fd)
        with open(temp_fullname, "w") as f:
            f.write(xml_data)
        mujoco_env.MujocoEnv.__init__(self, temp_fullname, 5)
        os.remove(temp_fullname)

    def step(self, all_joint_action):
        xposbefore = self.get_body_com("torso")[0]
        a = self.__calc_action(all_joint_action)
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore) / self.dt
        # ! Calculating cost assuming that `a` means torque
        ctrl_cost = 0.5 * np.square(a).sum()
        # ! If no touch sensor is provided, this value would always be 0
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 0.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        # state = self.state_vector()
        # notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        # done = not notdone
        done = False
        ob = self._get_obs()
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low[0], high=high[0], shape=(self.max_num_limbs * 2,), dtype=np.float32)
        return self.action_space

    def __calc_action(self, all_joint_action):
        assert all_joint_action.shape == self.action_space.shape
        assert (np.isfinite(all_joint_action).all())

        action = np.zeros(len(self.sim.data.ctrl))
        ctrlrange = self.sim.model.actuator_ctrlrange
        actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.
        actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.

        for rigid_id in range(self.max_num_limbs):
            joint_ids = self.rigid_id_2_joint_ids[rigid_id]
            idx = 0
            for joint_id in joint_ids:
                action[joint_id] = all_joint_action[rigid_id * 2 + idx]
                idx += 1

        a = actuation_center + action * actuation_range
        return np.clip(a, ctrlrange[:, 0], ctrlrange[:, 1])

    def reset(self):
        self.__reset_env()

        # Copy from mujoco_env.py
        self.sim.reset()
        ob = self.reset_model()
        return ob

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    # Put the state [0.0,0.0] as a dummy where the rigid body does not exist
    def _get_obs(self):
        qpos = []
        qvel = []
        torso_qpos = None
        torso_qvel = None
        for name in self.sim.model.joint_names:
            if name == "root":
                # Exclude x-coordinates and y-coordinates before putting them into state
                torso_qpos = self.sim.data.get_joint_qpos(name)[2:]
                torso_qvel = self.sim.data.get_joint_qvel(name)
            else:
                qpos.append(self.sim.data.get_joint_qpos(name))
                qvel.append(self.sim.data.get_joint_qvel(name))
        assert torso_qpos is not None and torso_qvel is not None

        # The joint_state that was determined from the top of mjcf becomes all_joint_state according to the order of morphology here.
        # mujoco uses mjcf's interpretation of joints from top to bottom
        # It is more accurate to match self.sim.model.joint_names with a regular expression to get the exact joint_id
        # However, for reasons of computation time, this method is used.
        joint_state = [None] * (len(qpos) + len(qvel))
        joint_state[::2] = qpos
        joint_state[1::2] = qvel
        all_joint_state = [0.0] * self.max_num_limbs * 4
        joint_state_id = 0
        for child in self.structure_tree[-1]:
            joint_state_id = self._insert_joint_state(child, joint_state_id, all_joint_state, joint_state)

        return np.concatenate(
            [
                all_joint_state,
                torso_qpos,
                torso_qvel,
            ]
        )

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
