import taichi as ti

ti.init(arch=ti.gpu)

import hydra
from omegaconf import OmegaConf
from scene import Scene
from simulator import ImpulseSimulator, ConstraintSimulator


@hydra.main(config_path="cfg", config_name="base", version_base=None)
def main(cfg):
    scene_cfg = OmegaConf.to_container(cfg.scene, resolve=True)
    sim_cfg = OmegaConf.to_container(cfg.sim, resolve=True)
    scene = Scene(scene_cfg)

    sim_type = sim_cfg["type"]
    sim_cfg["video"] = cfg.video
    if sim_type == "constraint":
        simulator = ConstraintSimulator(sim_cfg, scene)
    elif sim_type == "impulse":
        simulator = ImpulseSimulator(sim_cfg, scene)
    else:
        raise NotImplementedError(f"Unsupported simulator type: {sim_type}")

    simulator.run()


if __name__ == "__main__":
    main()
