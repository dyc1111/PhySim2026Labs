import taichi as ti

ti.init(arch=ti.gpu)

import hydra
from omegaconf import OmegaConf
from scene import Scene
from simulator import build_simulator


@hydra.main(config_path="../cfg", config_name="base", version_base=None)
def main(cfg):
    scene_cfg = OmegaConf.to_container(cfg.scene, resolve=True)
    sim_cfg = OmegaConf.to_container(cfg.sim, resolve=True)
    sim_cfg["video"] = cfg.video

    scene = Scene(scene_cfg)
    simulator = build_simulator(sim_cfg, scene)
    simulator.run()


if __name__ == "__main__":
    main()
