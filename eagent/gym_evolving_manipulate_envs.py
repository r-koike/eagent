# Related URLs:
# https://github.com/openai/gym/blob/master/gym/envs/robotics/hand/manipulate.py
# https://github.com/openai/gym/tree/v0.21.0/gym/envs/robotics
import os
import numpy as np
import copy
import tempfile

from gym import utils, error, spaces
from gym.envs.robotics import rotations, hand_env
# from gym.envs.robotics.utils import robot_get_obs

from .mujoco_xml_generater import HandXmlGenerater
from .evolving_tools import EvolvingTools

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e
        )
    )


def quat_from_angle_and_axis(angle, axis):
    assert axis.shape == (3,)
    axis /= np.linalg.norm(axis)
    quat = np.concatenate([[np.cos(angle / 2.0)], np.sin(angle / 2.0) * axis])
    quat /= np.linalg.norm(quat)
    return quat


class EvolvingManipulateEnv(hand_env.HandEnv):
    def __init__(
        self,
        target_position,
        target_rotation,
        target_position_range,
        reward_type,
        randomize_initial_position=True,
        randomize_initial_rotation=True,
        distance_threshold=0.01,
        rotation_threshold=0.1,
        relative_control=False,
        ignore_z_target_rotation=False,
    ):
        """Initializes a new Hand manipulation environment.
        Args:
            target_position (string): the type of target position:
                - ignore: target position is fully ignored, i.e. the object can be positioned arbitrarily
                - fixed: target position is set to the initial position of the object
                - random: target position is fully randomized according to target_position_range
            target_rotation (string): the type of target rotation:
                - ignore: target rotation is fully ignored, i.e. the object can be rotated arbitrarily
                - fixed: target rotation is set to the initial rotation of the object
                - xyz: fully randomized target rotation around the X, Y and Z axis
                - z: fully randomized target rotation around the Z axis
                - parallel: fully randomized target rotation around Z and axis-aligned rotation around X, Y
            ignore_z_target_rotation (boolean): whether or not the Z axis of the target rotation is ignored
            target_position_range (np.array of shape (3, 2)): range of the target_position randomization
            reward_type ("sparse" or "dense"): the reward type, i.e. sparse or dense
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            randomize_initial_position (boolean): whether or not to randomize the initial position of the object
            randomize_initial_rotation (boolean): whether or not to randomize the initial rotation of the object
            distance_threshold (float, in meters): the threshold after which the position of a goal is considered achieved
            rotation_threshold (float, in radians): the threshold after which the rotation of a goal is considered achieved
            n_substeps (int): number of substeps the simulation runs on every call to step
            relative_control (boolean): whether or not the hand is actuated in absolute joint positions or relative to the current state

        Note:
            superの__init__は継承していない
        """
        self.target_position = target_position
        self.target_rotation = target_rotation
        self.target_position_range = target_position_range
        self.parallel_quats = [
            rotations.euler2quat(r) for r in rotations.get_parallel_rotations()
        ]
        self.randomize_initial_rotation = randomize_initial_rotation
        self.randomize_initial_position = randomize_initial_position
        self.distance_threshold = distance_threshold
        self.rotation_threshold = rotation_threshold
        self.reward_type = reward_type
        self.ignore_z_target_rotation = ignore_z_target_rotation

        assert self.target_position in ["ignore", "fixed", "random", "fixed_random"]
        assert self.target_rotation in ["ignore", "fixed", "xyz", "z", "parallel"]

        # ----------------------------
        # Copy from HandEnv.__init__
        self.relative_control = relative_control

    def _get_achieved_goal(self):
        # Object position and rotation.
        object_qpos = self.sim.data.get_joint_qpos("object:joint")
        assert object_qpos.shape == (7,)
        return object_qpos

    def _goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        assert goal_a.shape[-1] == 7

        d_pos = np.zeros_like(goal_a[..., 0])
        d_rot = np.zeros_like(goal_b[..., 0])
        if self.target_position != "ignore":
            delta_pos = goal_a[..., :3] - goal_b[..., :3]
            d_pos = np.linalg.norm(delta_pos, axis=-1)

        if self.target_rotation != "ignore":
            quat_a, quat_b = goal_a[..., 3:], goal_b[..., 3:]

            if self.ignore_z_target_rotation:
                # Special case: We want to ignore the Z component of the rotation.
                # This code here assumes Euler angles with xyz convention. We first transform
                # to euler, then set the Z component to be equal between the two, and finally
                # transform back into quaternions.
                euler_a = rotations.quat2euler(quat_a)
                euler_b = rotations.quat2euler(quat_b)
                euler_a[2] = euler_b[2]
                quat_a = rotations.euler2quat(euler_a)

            # Subtract quaternions and extract angle between them.
            quat_diff = rotations.quat_mul(quat_a, rotations.quat_conjugate(quat_b))
            angle_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1.0, 1.0))
            d_rot = angle_diff
        assert d_pos.shape == d_rot.shape
        return d_pos, d_rot

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        if self.reward_type == "sparse":
            if not self._is_on_palm():
                return -2.0
            success = self._is_success(achieved_goal, goal).astype(np.float32)
            return success - 1.0
        else:
            d_pos, d_rot = self._goal_distance(achieved_goal, goal)
            # We weigh the difference in position to avoid that `d_pos` (in meters) is completely
            # dominated by `d_rot` (in radians).
            return -(10.0 * d_pos + d_rot)

    # RobotEnv methods
    # ----------------------------

    def _is_success(self, achieved_goal, desired_goal):
        d_pos, d_rot = self._goal_distance(achieved_goal, desired_goal)
        achieved_pos = (d_pos < self.distance_threshold).astype(np.float32)
        achieved_rot = (d_rot < self.rotation_threshold).astype(np.float32)
        achieved_both = achieved_pos * achieved_rot
        return achieved_both

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.sim.forward()

        initial_qpos = self.sim.data.get_joint_qpos("object:joint").copy()
        initial_pos, initial_quat = initial_qpos[:3], initial_qpos[3:]
        assert initial_qpos.shape == (7,)
        assert initial_pos.shape == (3,)
        assert initial_quat.shape == (4,)
        initial_qpos = None

        # Randomization initial rotation.
        if self.randomize_initial_rotation:
            if self.target_rotation == "z":
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = np.array([0.0, 0.0, 1.0])
                offset_quat = quat_from_angle_and_axis(angle, axis)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation == "parallel":
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = np.array([0.0, 0.0, 1.0])
                z_quat = quat_from_angle_and_axis(angle, axis)
                parallel_quat = self.parallel_quats[
                    self.np_random.integers(len(self.parallel_quats))
                ]
                offset_quat = rotations.quat_mul(z_quat, parallel_quat)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation in ["xyz", "ignore"]:
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = self.np_random.uniform(-1.0, 1.0, size=3)
                offset_quat = quat_from_angle_and_axis(angle, axis)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation == "fixed":
                pass
            else:
                raise error.Error(
                    f"Unknown target_rotation option \"{self.target_rotation}\"."
                )

        # Randomize initial position.
        if self.randomize_initial_position:
            if self.target_position != "fixed":
                initial_pos += self.np_random.normal(size=3, scale=0.005)

        initial_quat /= np.linalg.norm(initial_quat)
        initial_qpos = np.concatenate([initial_pos, initial_quat])
        self.sim.data.set_joint_qpos("object:joint", initial_qpos)

        # Run the simulation for a bunch of timesteps to let everything settle in.
        sim_steps = 10
        for _ in range(sim_steps):
            self._set_action(np.zeros(self.action_space.shape[0]))
            try:
                self.sim.step()
            except mujoco_py.MujocoException:
                return False
        self.sim.forward()
        return self._is_on_palm()

    def _is_on_palm(self):
        cube_middle_idx = self.sim.model.site_name2id("object:center")
        cube_middle_pos = self.sim.data.site_xpos[cube_middle_idx]
        is_on_palm = cube_middle_pos[2] > self.env_cfg["palm_drop_height"]
        return is_on_palm

    def _sample_goal(self):
        # Select a goal for the object position.
        target_pos = None
        if self.target_position == "random":
            assert self.target_position_range.shape == (3, 2)
            offset = self.np_random.uniform(
                self.target_position_range[:, 0], self.target_position_range[:, 1]
            )
            assert offset.shape == (3,)
            target_pos = self.sim.data.get_joint_qpos("object:joint")[:3] + offset
        elif self.target_position == "fixed_random":
            assert self.target_position_range.shape == (3, 2)
            offset = self.np_random.uniform(
                self.target_position_range[:, 0], self.target_position_range[:, 1]
            )
            assert offset.shape == (3,)
            target_pos = [1.0, 1.0, 0.3] + offset
        elif self.target_position in ["ignore", "fixed"]:
            target_pos = self.sim.data.get_joint_qpos("object:joint")[:3]
        else:
            raise error.Error(
                f"Unknown target_position option \"{self.target_position}\"."
            )
        assert target_pos is not None
        assert target_pos.shape == (3,)

        # Select a goal for the object rotation.
        target_quat = None
        if self.target_rotation == "z":
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = np.array([0.0, 0.0, 1.0])
            target_quat = quat_from_angle_and_axis(angle, axis)
        elif self.target_rotation == "parallel":
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = np.array([0.0, 0.0, 1.0])
            target_quat = quat_from_angle_and_axis(angle, axis)
            parallel_quat = self.parallel_quats[
                self.np_random.integers(len(self.parallel_quats))
            ]
            target_quat = rotations.quat_mul(target_quat, parallel_quat)
        elif self.target_rotation == "xyz":
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = self.np_random.uniform(-1.0, 1.0, size=3)
            target_quat = quat_from_angle_and_axis(angle, axis)
        elif self.target_rotation in ["ignore", "fixed"]:
            target_quat = self.sim.data.get_joint_qpos("object:joint")
        else:
            raise error.Error(
                f"Unknown target_rotation option \"{self.target_rotation}\"."
            )
        assert target_quat is not None
        assert target_quat.shape == (4,)

        target_quat /= np.linalg.norm(target_quat)  # normalized quaternion
        goal = np.concatenate([target_pos, target_quat])
        return goal

    def _render_callback(self):
        # Assign current state to target object but offset a bit so that the actual object
        # is not obscured.
        goal = self.goal.copy()
        assert goal.shape == (7,)
        if self.target_position == "ignore":
            # Move the object to the side since we do not care about it's position.
            goal[0] += 0.15
        self.sim.data.set_joint_qpos("target:joint", goal)
        self.sim.data.set_joint_qvel("target:joint", np.zeros(6))

        if "object_hidden" in self.sim.model.geom_names:
            hidden_id = self.sim.model.geom_name2id("object_hidden")
            self.sim.model.geom_rgba[hidden_id, 3] = 1.0
        self.sim.forward()

    # Put the state [0.0,0.0] as a dummy where the rigid body does not exist
    def _get_obs(self):
        if self.sim.data.qpos is not None and self.sim.model.joint_names:
            names = sorted([n for n in self.sim.model.joint_names if n.startswith("robot")])
            robot_qpos, robot_qvel = (
                np.array([self.sim.data.get_joint_qpos(name) for name in names]),
                np.array([self.sim.data.get_joint_qvel(name) for name in names]),
            )
            joint_state = [None] * (len(robot_qpos) + len(robot_qvel))
            joint_state[::2] = robot_qpos
            joint_state[1::2] = robot_qvel
        else:
            raise NotImplementedError
        # The joint_state was determined from the top of mjcf.
        # The all_joint_states created here are in the same order as the morphology
        all_joint_state = [0.0] * self.max_num_limbs * 4
        joint_state_id = 0
        for child in self.structure_tree[-1]:
            joint_state_id = self._insert_joint_state(child, joint_state_id, all_joint_state, joint_state)

        object_qvel = self.sim.data.get_joint_qvel("object:joint")
        achieved_goal = (
            self._get_achieved_goal().ravel()
        )  # this contains the object position + rotation
        observation = np.concatenate(
            [all_joint_state, object_qvel, achieved_goal]
        )
        return {
            "observation": observation.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.ravel().copy(),
        }


class EvolvingHandEnv(EvolvingManipulateEnv, utils.EzPickle, EvolvingTools):
    def __init__(
        self, structure_edges, structure_properties, max_num_limbs, env_cfg,
        target_position, target_rotation, reward_type, object_type, render=False
    ):
        assert object_type in ["block", "egg"], "invalid object type"
        self.env_cfg = env_cfg
        self.target_position = target_position
        self.target_rotation = target_rotation
        self.reward_type = reward_type
        self.max_num_limbs = max_num_limbs
        self.object_type = object_type

        EvolvingTools.__init__(self, max_num_limbs, structure_edges, structure_properties)
        utils.EzPickle.__init__(self, self.target_position, self.target_rotation, self.reward_type)
        self.__reset_env()

    def __reset_env(self):
        initial_qpos = None
        initial_qpos = initial_qpos or {}
        n_substeps = 20
        EvolvingManipulateEnv.__init__(
            self,
            target_position=self.target_position,
            target_rotation=self.target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.04, 0.04), (0, 0.06)]),
            reward_type=self.reward_type
        )

        # Create an xml file and delete it immediately after loading
        xml_data = HandXmlGenerater(self.env_cfg).generate_xml(
            self.structure_tree,
            self.structure_properties,
            self.dofs,
            self.object_type,
        )
        fd, temp_fullname = tempfile.mkstemp(
            text=True,
            suffix=".xml",
            dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "mujoco_assets", "temp")
        )
        os.close(fd)
        with open(temp_fullname, "w") as f:
            f.write(xml_data)
        # ----------------------------
        # This area is roughly a copy of RobotEnv.__init__
        model = mujoco_py.load_model_from_path(temp_fullname)
        os.remove(temp_fullname)

        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": int(np.round(1.0 / self.dt))
        }

        self.seed()
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())
        self.goal = self._sample_goal()

        obs = self._get_obs()
        self.action_space = spaces.Box(-1., 1., shape=(self.max_num_limbs * 2,), dtype="float32")
        # self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(obs),), dtype="float32")
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"),
            observation=spaces.Box(-np.inf, np.inf, shape=obs["observation"].shape, dtype="float32"),
        ))

    # Copy from robot_env.py
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()
        done = False
        achieved_goal = self._get_achieved_goal().ravel().copy()
        info = {
            "is_success": self._is_success(achieved_goal, self.goal),
        }
        reward = self.compute_reward(achieved_goal, self.goal, info)
        return obs, reward, done, info

    # Copy from hand_env.py
    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id("torso")
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 0.5
        self.viewer.cam.azimuth = 55.
        self.viewer.cam.elevation = -25.

    # Copy from hand_env.py
    # act is input with the size of self.max_num_limbs,
    # and the input to the place where the rigid body does not exist is discarded
    def _set_action(self, all_joint_action):
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

        self.sim.data.ctrl[:] = actuation_center + action * actuation_range
        self.sim.data.ctrl[:] = np.clip(self.sim.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])

    # Copy from robot_env.py and core.py
    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        # Enforce that each GoalEnv uses a Goal-compatible observation space.

        # reset
        self.__reset_env()

        # GoalEnv.reset()
        if not isinstance(self.observation_space, spaces.Dict):
            raise error.Error("GoalEnv requires an observation space of type gym.spaces.Dict")
        for key in ["observation", "achieved_goal", "desired_goal"]:
            if key not in self.observation_space.spaces:
                raise error.Error(
                    "GoalEnv requires the '{}' key to be part of the observation dictionary.".format(key))

        # RobotEnv.reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        return obs
