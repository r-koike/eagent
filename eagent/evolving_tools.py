import numpy as np
import re


class EvolvingTools():
    def __init__(self, max_num_limbs, structure_edges, structure_properties):
        self.max_num_limbs = max_num_limbs
        self.set_structure_params(structure_edges, structure_properties)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __set_rigid_id_2_joint_ids(self, current, idx):
        for child in self.structure_tree[current]:
            if self.dofs[child] == 1:
                self.rigid_id_2_joint_ids[child].extend([idx])
                idx += 1
            elif self.dofs[child] == 2:
                self.rigid_id_2_joint_ids[child].extend([idx, idx + 1])
                idx += 2
            idx = self.__set_rigid_id_2_joint_ids(child, idx)
        return idx

    def set_structure_params(self, structure_edges, structure_properties):
        self.structure_edges = structure_edges
        self.structure_properties = structure_properties

        # Update foot_list and tree according to edges
        self.foot_list = []
        self.structure_tree = [[] for i in range(self.max_num_limbs + 1)]
        self.dofs = np.zeros(self.max_num_limbs)
        for parent, child, dof in self.structure_edges:
            self.structure_tree[parent].append(child)
            self.dofs[child] = dof
            self.foot_list.append(f"body_{child}")
        self.rigid_id_2_joint_ids = [[] for i in range(self.max_num_limbs + 1)]
        self.__set_rigid_id_2_joint_ids(-1, 0)

    def _insert_joint_state(self, current, joint_state_id, all_joint_state, joint_state):
        if self.dofs[current] == 1:
            all_joint_state[current * 4:current * 4 + 2] = joint_state[joint_state_id * 2:joint_state_id * 2 + 2]
            joint_state_id += 1
        if self.dofs[current] == 2:
            all_joint_state[current * 4:current * 4 + 4] = joint_state[joint_state_id * 2:joint_state_id * 2 + 4]
            joint_state_id += 2
        for child in self.structure_tree[current]:
            joint_state_id = self._insert_joint_state(child, joint_state_id, all_joint_state, joint_state)
        return joint_state_id

    def get_structure_edges(self):
        return self.structure_edges

    def get_contact_state(self):
        contact_state = [0.0 for i in range(self.max_num_limbs + 1)]
        geom_pattern = re.compile(r"geom_(\d+)")

        sim = self.sim
        for i in range(sim.data.ncon):
            # Note that the contact array has more than `ncon` entries,
            # so be careful to only read the valid entries.
            contact = sim.data.contact[i]
            geom_name1 = sim.model.geom_id2name(contact.geom1)
            geom_name2 = sim.model.geom_id2name(contact.geom2)
            match1 = geom_pattern.match(geom_name1)
            match2 = geom_pattern.match(geom_name2)
            if match1 is not None:
                body_match = match1
                obj_name = geom_name2
            elif match2 is not None:
                body_match = match2
                obj_name = geom_name1
            else:
                continue
            if obj_name not in ["object", "floor"]:
                continue

            limb_id = int(body_match.group(1))
            contact_state[limb_id] = 1.0

        return contact_state
