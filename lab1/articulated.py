from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from rigidbody import create_rigid_body


def _to_vec3(value, default):
    v = np.array(default if value is None else value, dtype=np.float32)
    if v.shape != (3,):
        raise ValueError(f"Expected 3D vector, got shape {v.shape}")
    return v


@dataclass
class Joint:
    joint_type: str
    parent: int
    child: int
    anchor_parent_local: np.ndarray
    anchor_child_local: np.ndarray
    axis_parent_local: np.ndarray
    axis_child_local: np.ndarray


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

        link_specs = self._build_link_specs(cfg)
        joint_specs = self._build_joint_specs(cfg)

        self.links = []
        self.link_name_to_local_id = {}
        for local_idx, spec in enumerate(link_specs):
            link_cfg = dict(spec)
            link_name = link_cfg.pop("name")
            local_offset = _to_vec3(link_cfg.pop("local_offset", None), [0.0, 0.0, 0.0])

            link_cfg.setdefault("rotation_deg", [0.0, 0.0, 0.0])
            link_cfg.setdefault("velocity", [0.0, 0.0, 0.0])
            link_cfg.setdefault("angular_velocity", [0.0, 0.0, 0.0])
            link_cfg.setdefault("dyn_type", "free")
            link_cfg.setdefault("color", [0.65, 0.65, 0.85])
            if link_cfg["dyn_type"] == "free":
                link_cfg.setdefault("mass", 1.0)

            link_cfg["position"] = (
                (self.root + local_offset).astype(np.float32).tolist()
            )
            self.links.append(create_rigid_body(link_cfg))
            self.link_name_to_local_id[link_name] = local_idx

        joints = []
        for spec in joint_specs:
            parent_name = spec["parent"]
            child_name = spec["child"]
            if parent_name not in self.link_name_to_local_id:
                raise ValueError(f"Unknown parent link name: {parent_name}")
            if child_name not in self.link_name_to_local_id:
                raise ValueError(f"Unknown child link name: {child_name}")

            joint_type = spec["type"]
            if joint_type not in ("revolute", "ball"):
                raise NotImplementedError(f"Unsupported joint type: {joint_type}")

            joints.append(
                Joint(
                    joint_type=joint_type,
                    parent=self.link_name_to_local_id[parent_name],
                    child=self.link_name_to_local_id[child_name],
                    anchor_parent_local=_to_vec3(
                        spec.get("anchor_parent_local"), [0.0, 0.0, 0.0]
                    ),
                    anchor_child_local=_to_vec3(
                        spec.get("anchor_child_local"), [0.0, 0.0, 0.0]
                    ),
                    axis_parent_local=_to_vec3(
                        spec.get("axis_parent_local"), [0.0, 0.0, 1.0]
                    ),
                    axis_child_local=_to_vec3(
                        spec.get("axis_child_local"), [0.0, 0.0, 1.0]
                    ),
                )
            )

        self.chain = Chain(len(self.links), joints)

    def _build_link_specs(self, cfg):
        raise NotImplementedError

    def _build_joint_specs(self, cfg):
        raise NotImplementedError

    def get_joint_constraints(self, link_ids: List[int]):
        constraints = []
        for joint in self.chain.joints:
            constraints.append(
                {
                    "joint_type": joint.joint_type,
                    "parent": int(link_ids[joint.parent]),
                    "child": int(link_ids[joint.child]),
                    "anchor_parent_local": np.array(
                        joint.anchor_parent_local, dtype=np.float32
                    ),
                    "anchor_child_local": np.array(
                        joint.anchor_child_local, dtype=np.float32
                    ),
                    "axis_parent_local": np.array(
                        joint.axis_parent_local, dtype=np.float32
                    ),
                    "axis_child_local": np.array(
                        joint.axis_child_local, dtype=np.float32
                    ),
                }
            )
        return constraints


class ArticulatedRevoluteChain(ArticulatedBody):
    type_name = "articulated_revolute_chain"

    def _build_link_specs(self, cfg):
        scale = float(cfg.get("scale", 1.0))
        return [
            {
                "name": "base",
                "type": "cuboid",
                "size": [0.24 * scale, 0.40 * scale, 0.24 * scale],
                "mass": 2.0,
                "local_offset": [0.0, 0.0, 0.0],
                "color": cfg["color"],
            },
            {
                "name": "link1",
                "type": "cuboid",
                "size": [0.16 * scale, 0.50 * scale, 0.16 * scale],
                "mass": 1.2,
                "local_offset": [0.0, 0.45 * scale, 0.0],
                "color": cfg["color"],
            },
            {
                "name": "link2",
                "type": "cuboid",
                "size": [0.14 * scale, 0.44 * scale, 0.14 * scale],
                "mass": 0.9,
                "local_offset": [0.0, 0.92 * scale, 0.0],
                "color": cfg["color"],
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


class ArticulatedBallChain(ArticulatedBody):
    type_name = "articulated_ball_chain"

    def _build_link_specs(self, cfg):
        size = float(cfg["size"])
        return [
            {
                "name": "p0",
                "type": "sphere",
                "size": 0.14 * size,
                "mass": 1.4,
                "local_offset": [0.0, 0.0, 0.0],
                "color": cfg["color"],
            },
            {
                "name": "p1",
                "type": "sphere",
                "size": 0.12 * size,
                "mass": 1.1,
                "local_offset": [0.0, -0.30 * size, 0.0],
                "color": cfg["color"],
            },
            {
                "name": "p2",
                "type": "sphere",
                "size": 0.10 * size,
                "mass": 0.9,
                "local_offset": [0.0, -0.56 * size, 0.0],
                "color": cfg["color"],
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


ARTICULATED_TYPE_TO_CLASS = {
    ArticulatedRevoluteChain.type_name: ArticulatedRevoluteChain,
    ArticulatedBallChain.type_name: ArticulatedBallChain,
}


def is_articulated_type(type_name):
    return type_name in ARTICULATED_TYPE_TO_CLASS


def create_articulated_body(cfg) -> ArticulatedBody:
    body_type = cfg["type"]
    return ARTICULATED_TYPE_TO_CLASS[body_type](cfg)
