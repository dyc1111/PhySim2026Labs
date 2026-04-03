import taichi as ti

ti.init(arch=ti.gpu)

import hydra
from omegaconf import OmegaConf
from scene import Scene
from simulator import Simulator


@hydra.main(config_path="cfg", config_name="base", version_base=None)
def main(cfg):
    scene_cfg = OmegaConf.to_container(cfg.scene, resolve=True)
    sim_cfg = OmegaConf.to_container(cfg.sim, resolve=True)
    scene = Scene(scene_cfg)
    simulator = Simulator(sim_cfg, scene)
    simulator.run()


if __name__ == "__main__":
    main()
