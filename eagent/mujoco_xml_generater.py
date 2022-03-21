import numpy as np
import copy


def hard_sigmoid(x):
    y = 0.2 * x + 0.5
    return np.clip(y, 0.0, 1.0)


def inv_hard_sigmoid(y):
    x = 5.0 * y - 2.5
    return x


def polar_to_cartesian(phi, theta, r):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


class XmlGenerater():
    def __init__(self, env_cfg=None):
        self.actuators_xml = []
        self.min_z_from_torso = 999.9
        if env_cfg is not None and "robot_cfg" in env_cfg.keys():
            self.robot_cfg.update(copy.deepcopy(env_cfg["robot_cfg"]))

    def _generate_nested_xml(self, current, body_pos, tree, dofs, params_list, depth, z_from_torso=0):
        scale = self.robot_cfg["scale"]
        fromto_phi = params_list[current][0] * np.pi * 2.0
        fromto_theta = params_list[current][1] * np.pi * 2.0
        fromto_r = abs(params_list[current][2]) * scale / 0.08
        fromto_r = max(0.001, fromto_r)
        axis1_phi = params_list[current][3] * np.pi * 2.0
        axis1_theta = params_list[current][4] * np.pi * 2.0
        axis2_phi = params_list[current][5] * np.pi * 2.0
        axis2_theta = params_list[current][6] * np.pi * 2.0
        dof = dofs[current]
        fromto = polar_to_cartesian(fromto_phi, fromto_theta, fromto_r)
        axises = []
        axises.append(polar_to_cartesian(axis1_phi, axis1_theta, 1.0))
        axises.append(polar_to_cartesian(axis2_phi, axis2_theta, 1.0))

        if self.robot_cfg["self_collision"]:
            conaffinity = 3
        else:
            conaffinity = 1

        rigid_body_name = f"{current}"
        xml_data_list = []
        tab = " " * (8 + 4 * depth)
        xml_data_list.append(f"""
{tab}<body pos="{body_pos[0]} {body_pos[1]} {body_pos[2]}" name="body_{rigid_body_name}">
{tab}    <geom conaffinity="{conaffinity}" fromto="0.0 0.0 0.0 {fromto[0]} {fromto[1]} {fromto[2]}" name="geom_{rigid_body_name}" class="robot0:geom" size="{scale}" type="capsule"/>""")

        # [<motor> = torque control](https://roboti.us/forum/index.php?threads/control-input-of-actuator-motor-direct-drive-type.3745/)
        for i in range(np.round(dof).astype(int)):
            id = i + 1
            [r0, r1] = self.robot_cfg["joint_range"]
            [cr0, cr1] = self.robot_cfg["ctrlrange"]
            [fr0, fr1] = self.robot_cfg["forcerange"]
            xml_data_list.append(f"""
{tab}    <joint axis="{axises[i][0]} {axises[i][1]} {axises[i][2]}" pos="0.0 0.0 0.0" range="{r0} {r1}" type="hinge" name="robot0:joint_{rigid_body_name}_{id}"/>""")
            if self.robot_cfg["actuator"] == "motor":
                gear = self.robot_cfg["gear"]
                self.actuators_xml.append(f"""
        <motor ctrllimited="true" ctrlrange="{cr0} {cr1}" forcerange="{fr0} {fr1}" joint="robot0:joint_{rigid_body_name}_{id}" gear="{gear}"/>""")
            elif self.robot_cfg["actuator"] == "position":
                kp = self.robot_cfg["kp"]
                self.actuators_xml.append(f"""
        <position ctrllimited="true" ctrlrange="{cr0} {cr1}" forcerange="{fr0} {fr1}" kp="{kp}" joint="robot0:joint_{rigid_body_name}_{id}"/>""")
            else:
                raise NotImplementedError

        z_from_torso += fromto[2]
        self.min_z_from_torso = min(self.min_z_from_torso, z_from_torso)

        for child in tree[current]:
            xml_data_list.append(self._generate_nested_xml(
                child, fromto, tree, dofs, params_list, depth + 1, z_from_torso))
        xml_data_list.append(f"""\n{tab}</body>""")
        return "".join(xml_data_list)


class WalkerXmlGenerater(XmlGenerater):
    def __init__(self, env_cfg=None):
        self.robot_cfg = {
            "actuator": "motor",
            "torso_radius": 0.1,
            "scale": 0.08,
            "armature": 1,
            "damping": 1,
            "joint_range": [-0.7, 0.7],
            "ctrlrange": [-1.0, 1.0],
            "forcerange": [-1.0, 1.0],
            "gear": 150,
            "self_collision": True,
        }
        XmlGenerater.__init__(self, env_cfg)

    def generate_xml(self, tree, params, dofs):
        params_list = np.array(params).reshape([-1, 9]).tolist()

        torso_radius = self.robot_cfg["torso_radius"] * self.robot_cfg["scale"]

        xml_data_list = []
        xml_data_list.append(f"""
<mujoco model="ant">
    <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>
    <option integrator="RK4" timestep="0.01"/>
    <default>
        <joint armature="{self.robot_cfg["armature"]}" damping="{self.robot_cfg["damping"]}" limited="true" />
        <geom contype="1" conaffinity="1" condim="3" density="5.0" friction="1 0.5 0.5" margin="0.01" rgba="0.8 0.6 0.4 1"/>
        <default class="robot0:geom">
            <geom contype="2" conaffinity="1"/>
        </default>
    </default>
    <asset>
        <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
        <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
        <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
        <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
        <material name="geom" texture="texgeom" texuniform="true"/>
    </asset>
    <worldbody>
        <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
        <geom condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
        <body name="torso" pos="0 0 <placeholder>">
            <camera name="track" mode="trackcom" pos="0 -4 1.5" xyaxes="1 0 0 0 0.5 1"/>
            <geom name="torso_geom" pos="0 0 0" size="{torso_radius}" type="sphere"/>
            <joint armature="0" damping="0" limited="false" margin="0.01" name="root" pos="0 0 0" type="free"/>""")
        for child in tree[-1]:
            phi = params_list[child][7] * np.pi * 2.0
            theta = params_list[child][8] * np.pi * 2.0
            body_pos = polar_to_cartesian(phi, theta, torso_radius)
            xml_data_list.append(self._generate_nested_xml(child, body_pos, tree, dofs, params_list, 1, body_pos[2]))

        xml_data_list.append("""
        </body>
    </worldbody>
    <actuator>""")
        xml_data_list.extend(self.actuators_xml)
        xml_data_list.append("""
    </actuator>
</mujoco>""")

        xml_data = "".join(xml_data_list)
        # The size of the capsule is self.robot_cfg["scale"], so add that
        initial_z = max(abs(self.min_z_from_torso), torso_radius) + self.robot_cfg["scale"] + 0.05
        xml_data = xml_data.replace("<placeholder>", str(initial_z))

        return xml_data


class HandXmlGenerater(XmlGenerater):
    def __init__(self, env_cfg=None):
        self.robot_cfg = {
            "actuator": "position",
            "torso_radius": 0.1,
            "scale": 0.012,
            "armature": 0.001,
            "damping": 0.1,
            "joint_range": [-0.4, 0.4],
            "ctrlrange": [-0.4, 0.4],
            "forcerange": [-1.0, 1.0],
            "kp": 1,
            "hand_position": "below",
            "self_collision": True,
        }
        XmlGenerater.__init__(self, env_cfg)

    def generate_xml(self, tree, params, dofs, object_type):
        params_list = np.array(params).reshape([-1, 9]).tolist()

        torso_radius = self.robot_cfg["torso_radius"] * self.robot_cfg["scale"]
        if self.robot_cfg["hand_position"] == "behind":
            torso_position = "1 1.15 0.3"
        elif self.robot_cfg["hand_position"] == "below":
            torso_position = "1 1 0.3"
        else:
            raise NotImplementedError
        xml_data_list = []
        xml_data_list.append(f"""
<?xml version="1.0" encoding="utf-8"?>
<mujoco>
    <compiler angle="radian" coordinate="local" meshdir="../stls/hand" texturedir="../textures"></compiler>
    <option timestep="0.002" iterations="20" apirate="200">
        <flag warmstart="enable"></flag>
    </option>
    <size njmax="500" nconmax="100" nuser_jnt="1" nuser_site="1" nuser_tendon="1" nuser_sensor="1" nuser_actuator="16" nstack="600000"></size>
    <visual>
        <map fogstart="3" fogend="5" force="0.1"></map>
        <quality shadowsize="4096"></quality>
    </visual>
    <default>
        <joint armature="{self.robot_cfg["armature"]}" damping="{self.robot_cfg["damping"]}" limited="true" />
        <default class="robot0:geom">
            <geom contype="2" conaffinity="1" condim="3" density="5.0" friction="1.5 0.1 0.1" margin="0.001" rgba="0.8 0.6 0.4 1" />
        </default>
    </default>
    <asset>
        <include file="../hand/shared_asset.xml"></include>
        <texture name="texture:object" file="block.png" gridsize="3 4" gridlayout=".U..LFRB.D.."></texture>
        <texture name="texture:hidden" file="block_hidden.png" gridsize="3 4" gridlayout=".U..LFRB.D.."></texture>
        <material name="material:object" texture="texture:object" specular="1" shininess="0.3" reflectance="0"></material>
        <material name="material:hidden" texture="texture:hidden" specular="1" shininess="0.3" reflectance="0"></material>
        <material name="material:target" texture="texture:object" specular="1" shininess="0.3" reflectance="0" rgba="1 1 1 0.5"></material>
    </asset>
    <worldbody>
        <geom name="floor0" pos="1 1 0" size="1 1 1" type="plane" condim="3" material="floor_mat"></geom>
        <body name="floor0" pos="1 1 0"></body>
        <body name="torso" pos="{torso_position}">
            <geom name="torso_geom" class="robot0:geom" pos="0 0 0" size="{torso_radius}" type="sphere" />""")

        for child in tree[-1]:
            phi = params_list[child][7] * np.pi * 2.0
            theta = params_list[child][8] * np.pi * 2.0
            body_pos = polar_to_cartesian(phi, theta, torso_radius)
            xml_data_list.append(self._generate_nested_xml(child, body_pos, tree, dofs, params_list, 1, body_pos[2]))

        xml_data_list.append("""
        </body>""")
        if object_type == "block":
            xml_data_list.append("""
        <body name="object" pos="1 1 0.4">
            <geom name="object" type="box" size="0.025 0.025 0.025" material="material:object" condim="4" density="567"></geom>
            <geom name="object_hidden" type="box" size="0.024 0.024 0.024" material="material:hidden" condim="4" contype="0" conaffinity="0" mass="0"></geom>
            <site name="object:center" pos="0 0 0" rgba="1 0 0 0" size="0.01 0.01 0.01"></site>
            <joint name="object:joint" type="free" damping="0.01" armature="0" limited="false"></joint>
        </body>
        <body name="target" pos="1 1 0.4">
            <geom name="target" type="box" size="0.025 0.025 0.025" material="material:target" condim="4" group="2" contype="0" conaffinity="0"></geom>
            <site name="target:center" pos="0 0 0" rgba="1 0 0 0" size="0.01 0.01 0.01"></site>
            <joint name="target:joint" type="free" damping="0.01" armature="0" limited="false"></joint>
        </body>""")
        elif object_type == "egg":
            xml_data_list.append("""
        <body name="object" pos="1 1 0.4">
            <geom name="object" type="ellipsoid" size="0.03 0.03 0.04" material="material:object" condim="4"></geom>
            <geom name="object_hidden" type="ellipsoid" size="0.029 0.029 0.03" material="material:hidden" condim="4" contype="0" conaffinity="0" mass="0"></geom>
            <site name="object:center" pos="0 0 0" rgba="1 0 0 0" size="0.01 0.01 0.01"></site>
            <joint name="object:joint" type="free" damping="0.01" armature="0" limited="false"></joint>
        </body>
        <body name="target" pos="1 1 0.4">
            <geom name="target" type="ellipsoid" size="0.03 0.03 0.04" material="material:target" condim="4" group="2" contype="0" conaffinity="0"></geom>
            <site name="target:center" pos="0 0 0" rgba="1 0 0 0" size="0.01 0.01 0.01"></site>
            <joint name="target:joint" type="free" damping="0.01" armature="0" limited="false"></joint>
        </body>""")
        xml_data_list.append("""
        <light directional="true" ambient="0.2 0.2 0.2" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3" castshadow="false" pos="0 1 4" dir="0 0 -1" name="light0"></light>
    </worldbody>
    <actuator>""")
        xml_data_list.extend(self.actuators_xml)
        xml_data_list.append("""
    </actuator>
</mujoco>""")

        xml_data = "".join(xml_data_list)
        return xml_data


class WalkerTester():
    def __init__(self):
        pass

    def test_walker40(self):
        max_num_limbs = 40
        rigid_id = 0
        edges = []
        params = []
        for i in range(8):
            # params.append(inv_hard_sigmoid(np.random.random(9)))
            params.append(inv_hard_sigmoid(np.array([i / 8, 1 / 4, 0.5, i / 8 + 1 / 4, 1 / 4, 0.5, 0.5, 0.5, 0.5])))
            edges.append([-1, rigid_id, 1])
            i0 = rigid_id
            rigid_id += 1
            for j in range(2):
                # params.append(inv_hard_sigmoid(np.random.random(9)))
                a0 = i / 8 + 1 / 5 * j - 1 / 10
                if a0 < 0:
                    a0 += 1.0
                params.append(inv_hard_sigmoid(np.array([a0, 1 / 3, 0.28, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])))
                edges.append([i0, rigid_id, 1])
                j0 = rigid_id
                rigid_id += 1
                for k in range(1):
                    # params.append(inv_hard_sigmoid(np.random.random(9)))
                    a1 = i / 8
                    a2 = a1 + 1 / 4
                    if a2 > 1.0:
                        a2 -= 1.0
                    params.append(inv_hard_sigmoid(np.array([a1, 2 / 5, 0.25, a2, 1 / 4, 0.5, 0.5, 0.5, 0.5])))
                    edges.append([j0, rigid_id, 1])
                    rigid_id += 1
        print(edges)

        # From -2.5 to 2.5 at this point
        temp_params = params.copy()
        while len(temp_params) < max_num_limbs:
            temp_params.append(inv_hard_sigmoid(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])))
        print(np.array(temp_params).flatten().tolist())
        for i in range(len(params)):
            params[i] = hard_sigmoid(params[i])
        # From 0 to 1 at this point

        tree = [[] for i in range(max_num_limbs + 1)]
        dofs = np.array([2] * max_num_limbs)
        for parent, child, dof in edges:
            tree[parent].append(child)
            dofs[child] = dof
        xml_data = WalkerXmlGenerater().generate_xml(
            tree,
            params,
            dofs
        )
        with open("eagent/mujoco_assets/test.xml", "w") as f:
            f.write(xml_data)

    def test(self):
        max_num_limbs = 20
        edges = [
            [-1, 0, 1],
            [-1, 1, 1],
            [-1, 2, 1],
            [-1, 3, 1],
            [0, 4, 1],
            [1, 5, 1],
            [2, 6, 1],
            [3, 7, 1]
        ]
        # edges = [
        #     [-1, 0, 1],
        #     [-1, 1, 1],
        #     [-1, 2, 1]
        # ]
        # edges = [
        #     [-1, 0, 1],
        #     [-1, 1, 1],
        #     [-1, 2, 1],
        #     [-1, 3, 1],
        #     [-1, 4, 1],
        #     [-1, 5, 1],
        # ]

        params = []
        # params.append(inv_hard_sigmoid(np.array([0 / 4, 1 / 4, 0.45, 0.5, 0.5, 0.5, 0.5, 0 / 4, 1 / 4])))
        # params.append(inv_hard_sigmoid(np.array([1 / 4, 1 / 4, 0.45, 0.5, 0.5, 0.5, 0.5, 1 / 4, 1 / 4])))
        # params.append(inv_hard_sigmoid(np.array([2 / 4, 1 / 4, 0.45, 0.5, 0.5, 0.5, 0.5, 2 / 4, 1 / 4])))
        # params.append(inv_hard_sigmoid(np.array([3 / 4, 1 / 4, 0.45, 0.5, 0.5, 0.5, 0.5, 3 / 4, 1 / 4])))
        params.append(inv_hard_sigmoid(np.array([0 / 4, 1 / 5, 0.45, 0.5, 0.5, 0.5, 0.5, 0 / 4, 1 / 4])))
        params.append(inv_hard_sigmoid(np.array([1 / 4, 1 / 5, 0.45, 0.5, 0.5, 0.5, 0.5, 1 / 4, 1 / 4])))
        params.append(inv_hard_sigmoid(np.array([2 / 4, 1 / 5, 0.45, 0.5, 0.5, 0.5, 0.5, 2 / 4, 1 / 4])))
        params.append(inv_hard_sigmoid(np.array([3 / 4, 1 / 5, 0.45, 0.5, 0.5, 0.5, 0.5, 3 / 4, 1 / 4])))
        params.append(inv_hard_sigmoid(np.array([0 / 4, 2 / 5, 0.45, 1 / 4, 1 / 4, 0.5, 0.5, 0.0, 0.0])))
        params.append(inv_hard_sigmoid(np.array([1 / 4, 2 / 5, 0.45, 2 / 4, 1 / 4, 0.5, 0.5, 0.0, 0.0])))
        params.append(inv_hard_sigmoid(np.array([2 / 4, 2 / 5, 0.45, 3 / 4, 1 / 4, 0.5, 0.5, 0.0, 0.0])))
        params.append(inv_hard_sigmoid(np.array([3 / 4, 2 / 5, 0.45, 4 / 4, 1 / 4, 0.5, 0.5, 0.0, 0.0])))

        # params.append(inv_hard_sigmoid(np.array([1 / 8, 1 / 4, 0.45, 0.5, 0.5, 0.5, 0.5, 1 / 8, 1 / 4])))
        # params.append(inv_hard_sigmoid(np.array([3 / 8, 1 / 4, 0.45, 0.5, 0.5, 0.5, 0.5, 3 / 8, 1 / 4])))
        # params.append(inv_hard_sigmoid(np.array([5 / 8, 1 / 4, 0.45, 0.5, 0.5, 0.5, 0.5, 5 / 8, 1 / 4])))
        # params.append(inv_hard_sigmoid(np.array([7 / 8, 1 / 4, 0.45, 0.5, 0.5, 0.5, 0.5, 7 / 8, 1 / 4])))
        # params.append(inv_hard_sigmoid(np.array([1 / 8, 2 / 5, 0.45, 3 / 8, 1 / 4, 0.5, 0.5, 0.0, 0.0])))
        # params.append(inv_hard_sigmoid(np.array([3 / 8, 2 / 5, 0.45, 5 / 8, 1 / 4, 0.5, 0.5, 0.0, 0.0])))
        # params.append(inv_hard_sigmoid(np.array([5 / 8, 2 / 5, 0.45, 7 / 8, 1 / 4, 0.5, 0.5, 0.0, 0.0])))
        # params.append(inv_hard_sigmoid(np.array([7 / 8, 2 / 5, 0.45, 1 / 8, 1 / 4, 0.5, 0.5, 0.0, 0.0])))

        # params.append(inv_hard_sigmoid(np.array([0.125, 0.75, 0.65, 0.5, 0.5, 0.5, 0.5, 0.125, 0.75])))
        # params.append(inv_hard_sigmoid(np.array([0.375, 0.75, 0.65, 0.5, 0.5, 0.5, 0.5, 0.375, 0.75])))
        # params.append(inv_hard_sigmoid(np.array([0.625, 0.75, 0.65, 0.5, 0.5, 0.5, 0.5, 0.625, 0.75])))
        # params.append(inv_hard_sigmoid(np.array([0.875, 0.75, 0.65, 0.5, 0.5, 0.5, 0.5, 0.875, 0.75])))
        # params.append(inv_hard_sigmoid(np.array([0.125, 0.6, 0.65, 0.375, 0.75, 0.5, 0.5, 0.0, 0.0])))
        # params.append(inv_hard_sigmoid(np.array([0.375, 0.6, 0.65, 0.625, 0.75, 0.5, 0.5, 0.0, 0.0])))
        # params.append(inv_hard_sigmoid(np.array([0.625, 0.6, 0.65, 0.375, 0.75, 0.5, 0.5, 0.0, 0.0])))
        # params.append(inv_hard_sigmoid(np.array([0.875, 0.6, 0.65, 0.625, 0.75, 0.5, 0.5, 0.0, 0.0])))

        # # starfish6_1dof
        # params.append(inv_hard_sigmoid(np.array([0 / 6, 1 / 3, 0.5, 0 / 6, 1 / 12, 0.5, 0.5, 0 / 6, 1 / 3])))
        # params.append(inv_hard_sigmoid(np.array([1 / 6, 1 / 3, 0.5, 1 / 6, 1 / 12, 0.5, 0.5, 1 / 6, 1 / 3])))
        # params.append(inv_hard_sigmoid(np.array([2 / 6, 1 / 3, 0.5, 2 / 6, 1 / 12, 0.5, 0.5, 2 / 6, 1 / 3])))
        # params.append(inv_hard_sigmoid(np.array([3 / 6, 1 / 3, 0.5, 3 / 6, 1 / 12, 0.5, 0.5, 3 / 6, 1 / 3])))
        # params.append(inv_hard_sigmoid(np.array([4 / 6, 1 / 3, 0.5, 4 / 6, 1 / 12, 0.5, 0.5, 4 / 6, 1 / 3])))
        # params.append(inv_hard_sigmoid(np.array([5 / 6, 1 / 3, 0.5, 5 / 6, 1 / 12, 0.5, 0.5, 5 / 6, 1 / 3])))

        # # ant2
        # params.append(inv_hard_sigmoid(np.array([0 / 4, 1 / 4, 0.45, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0])))
        # params.append(inv_hard_sigmoid(np.array([0 / 4, 1 / 4, 0.45, 1 / 4, 1 / 4, 0.5, 0.5, 0.0, 0.0])))

        # From -2.5 to 2.5 at this point
        temp_params = params.copy()
        while len(temp_params) < max_num_limbs:
            temp_params.append(inv_hard_sigmoid(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])))
        print(np.array(temp_params).flatten().tolist())
        for i in range(len(params)):
            params[i] = hard_sigmoid(params[i])
        # From 0 to 1 at this point

        tree = [[] for i in range(max_num_limbs + 1)]
        dofs = np.array([2] * max_num_limbs)
        for parent, child, dof in edges:
            tree[parent].append(child)
            dofs[child] = dof
        xml_data = WalkerXmlGenerater().generate_xml(
            tree,
            params,
            dofs
        )
        with open("eagent/mujoco_assets/test.xml", "w") as f:
            f.write(xml_data)


class HandTester():
    def __init__(self):
        pass

    def test_hand25(self):
        max_num_limbs = 25
        rigid_id = 0
        edges = []
        params = []
        for i in range(6):
            # params.append(inv_hard_sigmoid(np.random.random(9)))
            a0 = i / 6
            params.append(inv_hard_sigmoid(np.array([a0, 3 / 16, 0.5, a0 + 1 / 4, 1 / 4, 0.5, 0.5, 0.5, 0.5])))
            edges.append([-1, rigid_id, 1])
            i0 = rigid_id
            rigid_id += 1
            for j in range(1):
                # params.append(inv_hard_sigmoid(np.random.random(9)))
                params.append(inv_hard_sigmoid(np.array([a0, 1 / 16, 0.28, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])))
                edges.append([i0, rigid_id, 1])
                j0 = rigid_id
                rigid_id += 1
                for k in range(2):
                    # params.append(inv_hard_sigmoid(np.random.random(9)))
                    a1 = a0 + 1 / 8 * k - 1 / 16
                    if a1 < 0:
                        a1 += 1.0
                    if a1 > 1.0:
                        a1 -= 1.0
                    a2 = a1 + 1 / 4
                    if a2 > 1.0:
                        a2 -= 1.0
                    params.append(inv_hard_sigmoid(np.array([a1, 3 / 4, 0.25, a2, 1 / 4, 0.5, 0.5, 0.5, 0.5])))
                    edges.append([j0, rigid_id, 1])
                    rigid_id += 1
        print(rigid_id)
        print(edges)

        # From -2.5 to 2.5 at this point
        temp_params = params.copy()
        while len(temp_params) < max_num_limbs:
            temp_params.append(inv_hard_sigmoid(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])))
        print(np.array(temp_params).flatten().tolist())
        for i in range(len(params)):
            params[i] = hard_sigmoid(params[i])
        # From 0 to 1 at this point

        tree = [[] for i in range(max_num_limbs + 1)]
        dofs = np.array([2] * max_num_limbs)
        for parent, child, dof in edges:
            tree[parent].append(child)
            dofs[child] = dof
        xml_data = HandXmlGenerater(env_cfg={
            "robot_cfg": {
                # "actuator": "position",
                # "torso_radius": 0.00001,
                # "armature": 0.001,
                # "damping": 0.1,
                # "joint_range": [-0.4, 0.4],
                # "ctrlrange": [-0.4, 0.4],
                # "forcerange": [-1.0, 1.0],
                # "kp": 1,
                # "hand_position": "below",
                # "self_collision": True,
            }
        }).generate_xml(
            tree,
            params,
            dofs,
            object_type="egg"
        )
        with open("eagent/mujoco_assets/test.xml", "w") as f:
            f.write(xml_data)

    def test(self):
        max_num_limbs = 20
        # edges = [
        #     [-1, 0, 1],
        #     [-1, 1, 1],
        #     [-1, 2, 1],
        # ]
        edges = [
            [-1, 0, 1],
            [-1, 1, 1],
            [-1, 2, 1],
            [-1, 3, 1],
            [-1, 4, 1],
            # [0, 5, 1],
            # [1, 6, 1],
            # [2, 7, 1],
            # [3, 8, 1],
            # [4, 9, 1],
            # [5, 10, 1],
            # [6, 11, 1],
            # [7, 12, 1],
            # [8, 13, 1],
            # [9, 14, 1],
        ]
        # edges = [
        #     [-1, 0, 1],
        #     [-1, 1, 1],
        #     [-1, 2, 1],
        #     [-1, 3, 1],
        #     [0, 4, 1],
        #     [1, 5, 1],
        #     [2, 6, 1],
        #     [3, 7, 1],
        # ]

        params = []
        # # hand3
        # params.append(inv_hard_sigmoid(
        #     np.array([17 / 24, 1 / 4, 0.8, 0, 1 / 4, 0.5, 0.5, 1 / 4, 3 / 4])))
        # params.append(inv_hard_sigmoid(
        #     np.array([18 / 24, 1 / 4, 0.8, 0, 1 / 4, 0.5, 0.5, 1 / 4, 3 / 4])))
        # params.append(inv_hard_sigmoid(
        #     np.array([19 / 24, 1 / 4, 0.8, 0, 1 / 4, 0.5, 0.5, 1 / 4, 3 / 4])))

        # # hand10
        # params.append(inv_hard_sigmoid(
        #     np.array([16 / 24, 1 / 4, 0.6, 0, 1 / 4, 0.5, 0.5, 1 / 4, 3 / 4])))
        # params.append(inv_hard_sigmoid(
        #     np.array([17 / 24, 1 / 4, 0.6, 0, 1 / 4, 0.5, 0.5, 1 / 4, 3 / 4])))
        # params.append(inv_hard_sigmoid(
        #     np.array([18 / 24, 1 / 4, 0.6, 0, 1 / 4, 0.5, 0.5, 1 / 4, 3 / 4])))
        # params.append(inv_hard_sigmoid(
        #     np.array([19 / 24, 1 / 4, 0.6, 0, 1 / 4, 0.5, 0.5, 1 / 4, 3 / 4])))
        # params.append(inv_hard_sigmoid(
        #     np.array([20 / 24, 1 / 4, 0.6, 0, 1 / 4, 0.5, 0.5, 1 / 4, 3 / 4])))
        # params.append(inv_hard_sigmoid(
        #     np.array([16 / 24, 1 / 4, 0.6, 0, 1 / 4, 0.5, 0.5, 1 / 4, 3 / 4])))
        # params.append(inv_hard_sigmoid(
        #     np.array([17 / 24, 1 / 4, 0.6, 0, 1 / 4, 0.5, 0.5, 1 / 4, 3 / 4])))
        # params.append(inv_hard_sigmoid(
        #     np.array([18 / 24, 1 / 4, 0.6, 0, 1 / 4, 0.5, 0.5, 1 / 4, 3 / 4])))
        # params.append(inv_hard_sigmoid(
        #     np.array([19 / 24, 1 / 4, 0.6, 0, 1 / 4, 0.5, 0.5, 1 / 4, 3 / 4])))
        # params.append(inv_hard_sigmoid(
        #     np.array([20 / 24, 1 / 4, 0.6, 0, 1 / 4, 0.5, 0.5, 1 / 4, 3 / 4])))

        # # hand15
        # params.append(inv_hard_sigmoid(np.array([16 / 24, 1 / 4, 0.6, 0, 1 / 8, 0.5, 0.5, 1 / 4, 3 / 4])))
        # params.append(inv_hard_sigmoid(np.array([17 / 24, 1 / 4, 0.6, 0, 1 / 8, 0.5, 0.5, 1 / 4, 3 / 4])))
        # params.append(inv_hard_sigmoid(np.array([18 / 24, 1 / 4, 0.6, 0, 1 / 4, 0.5, 0.5, 1 / 4, 3 / 4])))
        # params.append(inv_hard_sigmoid(np.array([19 / 24, 1 / 4, 0.6, 0, 7 / 8, 0.5, 0.5, 1 / 4, 3 / 4])))
        # params.append(inv_hard_sigmoid(np.array([20 / 24, 1 / 4, 0.6, 0, 7 / 8, 0.5, 0.5, 1 / 4, 3 / 4])))
        # params.append(inv_hard_sigmoid(np.array([16 / 24, 3 / 16, 0.2, 0, 1 / 4, 0.5, 0.5, 0.5, 0.5])))
        # params.append(inv_hard_sigmoid(np.array([17 / 24, 3 / 16, 0.2, 0, 1 / 4, 0.5, 0.5, 0.5, 0.5])))
        # params.append(inv_hard_sigmoid(np.array([18 / 24, 3 / 16, 0.2, 0, 1 / 4, 0.5, 0.5, 0.5, 0.5])))
        # params.append(inv_hard_sigmoid(np.array([19 / 24, 3 / 16, 0.2, 0, 1 / 4, 0.5, 0.5, 0.5, 0.5])))
        # params.append(inv_hard_sigmoid(np.array([20 / 24, 3 / 16, 0.2, 0, 1 / 4, 0.5, 0.5, 0.5, 0.5])))
        # params.append(inv_hard_sigmoid(np.array([16 / 24, 2 / 16, 0.2, 0, 1 / 4, 0.5, 0.5, 0.5, 0.5])))
        # params.append(inv_hard_sigmoid(np.array([17 / 24, 2 / 16, 0.2, 0, 1 / 4, 0.5, 0.5, 0.5, 0.5])))
        # params.append(inv_hard_sigmoid(np.array([18 / 24, 2 / 16, 0.2, 0, 1 / 4, 0.5, 0.5, 0.5, 0.5])))
        # params.append(inv_hard_sigmoid(np.array([19 / 24, 2 / 16, 0.2, 0, 1 / 4, 0.5, 0.5, 0.5, 0.5])))
        # params.append(inv_hard_sigmoid(np.array([20 / 24, 2 / 16, 0.2, 0, 1 / 4, 0.5, 0.5, 0.5, 0.5])))

        # hand15_below
        params.append(inv_hard_sigmoid(np.array([0 / 5, 3 / 16, 0.45, 5 / 20, 1 / 4, 0.5, 0.5, 0 / 5, 1 / 4])))
        params.append(inv_hard_sigmoid(np.array([1 / 5, 3 / 16, 0.45, 9 / 20, 1 / 4, 0.5, 0.5, 1 / 5, 1 / 4])))
        params.append(inv_hard_sigmoid(np.array([2 / 5, 3 / 16, 0.45, 13 / 20, 1 / 4, 0.5, 0.5, 2 / 5, 1 / 4])))
        params.append(inv_hard_sigmoid(np.array([3 / 5, 3 / 16, 0.45, 17 / 20, 1 / 4, 0.5, 0.5, 3 / 5, 1 / 4])))
        params.append(inv_hard_sigmoid(np.array([4 / 5, 3 / 16, 0.45, 1 / 20, 1 / 4, 0.5, 0.5, 4 / 5, 1 / 4])))
        # params.append(inv_hard_sigmoid(np.array([0 / 5, 0, 0.25, 0 / 5, 3 / 16, 0.5, 0.5, 0 / 5, 1 / 4])))
        # params.append(inv_hard_sigmoid(np.array([1 / 5, 0, 0.25, 1 / 5, 3 / 16, 0.5, 0.5, 1 / 5, 1 / 4])))
        # params.append(inv_hard_sigmoid(np.array([2 / 5, 0, 0.25, 2 / 5, 3 / 16, 0.5, 0.5, 2 / 5, 1 / 4])))
        # params.append(inv_hard_sigmoid(np.array([3 / 5, 0, 0.25, 3 / 5, 3 / 16, 0.5, 0.5, 3 / 5, 1 / 4])))
        # params.append(inv_hard_sigmoid(np.array([4 / 5, 0, 0.25, 4 / 5, 3 / 16, 0.5, 0.5, 4 / 5, 1 / 4])))
        # params.append(inv_hard_sigmoid(np.array([0 / 5, 13 / 16, 0.12, 5 / 20, 1 / 4, 0.5, 0.5, 0 / 5, 1 / 4])))
        # params.append(inv_hard_sigmoid(np.array([1 / 5, 13 / 16, 0.12, 9 / 20, 1 / 4, 0.5, 0.5, 1 / 5, 1 / 4])))
        # params.append(inv_hard_sigmoid(np.array([2 / 5, 13 / 16, 0.12, 13 / 20, 1 / 4, 0.5, 0.5, 2 / 5, 1 / 4])))
        # params.append(inv_hard_sigmoid(np.array([3 / 5, 13 / 16, 0.12, 17 / 20, 1 / 4, 0.5, 0.5, 3 / 5, 1 / 4])))
        # params.append(inv_hard_sigmoid(np.array([4 / 5, 13 / 16, 0.12, 1 / 20, 1 / 4, 0.5, 0.5, 4 / 5, 1 / 4])))

        # # hand8_below
        # params.append(inv_hard_sigmoid(np.array([0 / 4, 3 / 16, 0.43, 1 / 4, 1 / 4, 0.5, 0.5, 0 / 4, 1 / 6])))
        # params.append(inv_hard_sigmoid(np.array([1 / 4, 3 / 16, 0.43, 2 / 4, 1 / 4, 0.5, 0.5, 1 / 4, 1 / 6])))
        # params.append(inv_hard_sigmoid(np.array([2 / 4, 3 / 16, 0.43, 3 / 4, 1 / 4, 0.5, 0.5, 2 / 4, 1 / 6])))
        # params.append(inv_hard_sigmoid(np.array([3 / 4, 3 / 16, 0.43, 4 / 4, 1 / 4, 0.5, 0.5, 3 / 4, 1 / 6])))
        # params.append(inv_hard_sigmoid(np.array([0 / 4, 15 / 16, 0.25, 0 / 4, 3 / 16, 0.5, 0.5, 0.5, 0.5])))
        # params.append(inv_hard_sigmoid(np.array([1 / 4, 15 / 16, 0.25, 1 / 4, 3 / 16, 0.5, 0.5, 0.5, 0.5])))
        # params.append(inv_hard_sigmoid(np.array([2 / 4, 15 / 16, 0.25, 2 / 4, 3 / 16, 0.5, 0.5, 0.5, 0.5])))
        # params.append(inv_hard_sigmoid(np.array([3 / 4, 15 / 16, 0.25, 3 / 4, 3 / 16, 0.5, 0.5, 0.5, 0.5])))

        # # hand10_below
        # params.append(inv_hard_sigmoid(np.array([0 / 5, 1 / 8, 0.3, 5 / 20, 1 / 4, 0.5, 0.5, 0 / 5, 1 / 6])))
        # params.append(inv_hard_sigmoid(np.array([1 / 5, 1 / 8, 0.3, 9 / 20, 1 / 4, 0.5, 0.5, 1 / 5, 1 / 6])))
        # params.append(inv_hard_sigmoid(np.array([2 / 5, 1 / 8, 0.3, 13 / 20, 1 / 4, 0.5, 0.5, 2 / 5, 1 / 6])))
        # params.append(inv_hard_sigmoid(np.array([3 / 5, 1 / 8, 0.3, 17 / 20, 1 / 4, 0.5, 0.5, 3 / 5, 1 / 6])))
        # params.append(inv_hard_sigmoid(np.array([4 / 5, 1 / 8, 0.3, 1 / 20, 1 / 4, 0.5, 0.5, 4 / 5, 1 / 6])))
        # params.append(inv_hard_sigmoid(np.array([0 / 5, 15 / 16, 0.25, 0 / 5, 3 / 16, 0.5, 0.5, 0.0, 0.0])))
        # params.append(inv_hard_sigmoid(np.array([1 / 5, 15 / 16, 0.25, 1 / 5, 3 / 16, 0.5, 0.5, 0.0, 0.0])))
        # params.append(inv_hard_sigmoid(np.array([2 / 5, 15 / 16, 0.25, 2 / 5, 3 / 16, 0.5, 0.5, 0.0, 0.0])))
        # params.append(inv_hard_sigmoid(np.array([3 / 5, 15 / 16, 0.25, 3 / 5, 3 / 16, 0.5, 0.5, 0.0, 0.0])))
        # params.append(inv_hard_sigmoid(np.array([4 / 5, 15 / 16, 0.25, 4 / 5, 3 / 16, 0.5, 0.5, 0.0, 0.0])))

        # From -2.5 to 2.5 at this point
        temp_params = params.copy()
        while len(temp_params) < max_num_limbs:
            temp_params.append(inv_hard_sigmoid(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])))
        print(np.array(temp_params).flatten().tolist())
        for i in range(len(params)):
            params[i] = hard_sigmoid(params[i])
        # From 0 to 1 at this point

        tree = [[] for i in range(max_num_limbs + 1)]
        dofs = np.array([2] * max_num_limbs)
        for parent, child, dof in edges:
            tree[parent].append(child)
            dofs[child] = dof
        xml_data = HandXmlGenerater(env_cfg={
            "robot_cfg": {
            }
        }).generate_xml(
            tree,
            params,
            dofs,
            object_type="egg"
        )
        with open("eagent/mujoco_assets/test.xml", "w") as f:
            f.write(xml_data)


if __name__ == "__main__":
    WalkerTester().test()
    # WalkerTester().test_walker40()
    # HandTester().test()
    # HandTester().test_hand25()
