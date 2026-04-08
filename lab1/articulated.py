from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from rigidbody import create_rigid_body


@dataclass
class Joint:
    joint_type: str
    parent: int
    child: int
    anchor_parent_local: np.ndarray
    anchor_child_local: np.ndarray
    axis_parent_local: np.ndarray | None
    axis_child_local: np.ndarray | None


class Chain:
    def __init__(self, num_links: int, joints: List[Joint]):
        self.num_links = num_links
        self.joints = joints
        self.parent = [-1] * num_links
        self.children = [[] for _ in range(num_links)]
        self.joint_by_child = {}

        for joint in joints:
            if joint.parent == joint.child:
                raise ValueError("Joint parent and child must be different")
            if joint.child < 0 or joint.child >= num_links:
                raise ValueError("Joint child index out of bounds")
            if joint.parent < 0 or joint.parent >= num_links:
                raise ValueError("Joint parent index out of bounds")
            if self.parent[joint.child] != -1:
                raise ValueError("Each link must have at most one parent")

            self.parent[joint.child] = joint.parent
            self.children[joint.parent].append(joint.child)
            self.joint_by_child[joint.child] = joint

        roots = [i for i, p in enumerate(self.parent) if p == -1]
        if len(roots) != 1:
            raise ValueError("Articulated chain must have exactly one root")
        self.root = roots[0]

        state = [0] * num_links

        def dfs(u):
            state[u] = 1
            for v in self.children[u]:
                if state[v] == 1:
                    raise ValueError("Cycle detected in articulated chain")
                if state[v] == 0:
                    dfs(v)
            state[u] = 2

        dfs(self.root)
        if any(s == 0 for s in state):
            raise ValueError("Articulated chain must be fully connected")


class ArticulatedBody:
    type_name = "articulated_base"

    def __init__(self, cfg):
        root_position = cfg["position"]
        self.root = np.array(root_position, dtype=np.float32)
        self.links = []
        self.link_name_to_local_id = {}
        self.chain = None

        link_specs = self._build_link_specs(cfg)
        joint_specs = self._build_joint_specs(cfg)
        self._build_links(link_specs)
        self._build_chain(joint_specs)

    def _build_link_specs(self, cfg):
        raise NotImplementedError

    def _build_joint_specs(self, cfg):
        raise NotImplementedError

    def _build_links(self, link_specs):
        for local_idx, spec in enumerate(link_specs):
            link_cfg = dict(spec)
            link_name = link_cfg.pop("name")
            local_offset = np.array(link_cfg["local_offset"], dtype=np.float32)

            link_cfg.setdefault("rotation_deg", [0.0, 0.0, 0.0])
            link_cfg.setdefault("velocity", [0.0, 0.0, 0.0])
            link_cfg.setdefault("angular_velocity", [0.0, 0.0, 0.0])
            link_cfg.setdefault("dyn_type", "free")

            link_cfg["position"] = (self.root + local_offset).tolist()
            self.links.append(create_rigid_body(link_cfg))
            self.link_name_to_local_id[link_name] = local_idx

    def _build_chain(self, joint_specs):
        return NotImplementedError

    def get_joint_constraints(self, link_ids: List[int]):
        constraints = []
        for joint in self.chain.joints:
            constraints.append(
                {
                    "joint_type": joint.joint_type,
                    "parent": int(link_ids[joint.parent]),
                    "child": int(link_ids[joint.child]),
                    "anchor_parent_local": joint.anchor_parent_local,
                    "anchor_child_local": joint.anchor_child_local,
                    "axis_parent_local": joint.axis_parent_local,
                    "axis_child_local": joint.axis_child_local,
                }
            )
        return constraints


class ArticulatedRevoluteChain(ArticulatedBody):
    type_name = "articulated_revolute_chain"

    def _build_link_specs(self, cfg):
        size = float(cfg["size"])
        color = np.array(cfg["color"], dtype=np.float32)
        return [
            {
                "name": "base",
                "type": "cuboid",
                "size": [0.24 * size, 0.40 * size, 0.24 * size],
                "mass": 2.0,
                "local_offset": [0.0, 0.0, 0.0],
                "color": (color * 0.6).tolist(),
            },
            {
                "name": "link1",
                "type": "cuboid",
                "size": [0.16 * size, 0.50 * size, 0.16 * size],
                "mass": 1.2,
                "local_offset": [0.0, 0.45 * size, 0.0],
                "color": (color * 0.8).tolist(),
            },
            {
                "name": "link2",
                "type": "cuboid",
                "size": [0.14 * size, 0.44 * size, 0.14 * size],
                "mass": 0.9,
                "local_offset": [0.0, 0.92 * size, 0.0],
                "color": color.tolist(),
            },
        ]

    def _build_joint_specs(self, cfg):
        scale = float(cfg["size"])
        return [
            {
                "type": "revolute",
                "parent": "base",
                "child": "link1",
                "anchor_parent_local": [0.0, 0.20 * scale, 0.0],
                "anchor_child_local": [0.0, -0.25 * scale, 0.0],
                "axis_parent_local": [0.0, 0.0, 1.0],
                "axis_child_local": [0.0, 0.0, 1.0],
            },
            {
                "type": "revolute",
                "parent": "link1",
                "child": "link2",
                "anchor_parent_local": [0.0, 0.25 * scale, 0.0],
                "anchor_child_local": [0.0, -0.22 * scale, 0.0],
                "axis_parent_local": [0.0, 0.0, 1.0],
                "axis_child_local": [0.0, 0.0, 1.0],
            },
        ]

    def _build_chain(self, joint_spec):
        joints = []
        for spec in joint_spec:
            parent_name = spec["parent"]
            child_name = spec["child"]
            joint_type = spec["type"]
            joints.append(
                Joint(
                    joint_type,
                    self.link_name_to_local_id[parent_name],
                    self.link_name_to_local_id[child_name],
                    np.array(spec["anchor_parent_local"], np.float32),
                    np.array(spec["anchor_child_local"], np.float32),
                    np.array(spec["axis_parent_local"], np.float32),
                    np.array(spec["axis_parent_local"], np.float32),
                )
            )
        self.chain = Chain(len(self.links), joints)


class ArticulatedBallChain(ArticulatedBody):
    type_name = "articulated_ball_chain"

    def _build_link_specs(self, cfg):
        size = float(cfg["size"])
        color = np.array(cfg["color"], dtype=np.float32)
        return [
            {
                "name": "p0",
                "type": "sphere",
                "size": 0.14 * size,
                "mass": 1.4,
                "local_offset": [0.0, 0.0, 0.0],
                "color": (color * 0.6).tolist(),
            },
            {
                "name": "p1",
                "type": "sphere",
                "size": 0.12 * size,
                "mass": 1.1,
                "local_offset": [0.0, -0.30 * size, 0.0],
                "color": (color * 0.8).tolist(),
            },
            {
                "name": "p2",
                "type": "sphere",
                "size": 0.10 * size,
                "mass": 0.9,
                "local_offset": [0.0, -0.56 * size, 0.0],
                "color": color.tolist(),
            },
        ]

    def _build_joint_specs(self, cfg):
        scale = float(cfg.get("scale", 1.0))
        return [
            {
                "type": "ball",
                "parent": "p0",
                "child": "p1",
                "anchor_parent_local": [0.0, -0.14 * scale, 0.0],
                "anchor_child_local": [0.0, 0.12 * scale, 0.0],
            },
            {
                "type": "ball",
                "parent": "p1",
                "child": "p2",
                "anchor_parent_local": [0.0, -0.12 * scale, 0.0],
                "anchor_child_local": [0.0, 0.10 * scale, 0.0],
            },
        ]

    def _build_chain(self, joint_spec):
        joints = []
        for spec in joint_spec:
            parent_name = spec["parent"]
            child_name = spec["child"]
            joint_type = spec["type"]
            joints.append(
                Joint(
                    joint_type,
                    self.link_name_to_local_id[parent_name],
                    self.link_name_to_local_id[child_name],
                    np.array(spec["anchor_parent_local"], np.float32),
                    np.array(spec["anchor_child_local"], np.float32),
                    None,
                    None,
                )
            )
        self.chain = Chain(len(self.links), joints)


ARTICULATED_TYPE_TO_CLASS = {
    ArticulatedRevoluteChain.type_name: ArticulatedRevoluteChain,
    ArticulatedBallChain.type_name: ArticulatedBallChain,
}


def is_articulated_type(type_name):
    return type_name in ARTICULATED_TYPE_TO_CLASS


def create_articulated_body(cfg) -> ArticulatedBody:
    body_type = cfg["type"]
    return ARTICULATED_TYPE_TO_CLASS[body_type](cfg)
