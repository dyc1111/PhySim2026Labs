import taichi as ti

ti.init(arch=ti.gpu)

import hydra
from omegaconf import OmegaConf
from scene import Scene
from simulator import Simulator


@hydra.main(config_path="cfg", config_name="single", version_base=None)
def main(cfg):
    scene_cfg = OmegaConf.to_container(cfg.scene, resolve=True)
    scene = Scene(scene_cfg)
    simulator = Simulator(scene)

    steps = scene_cfg["steps"]
    simulator.run(steps)


if __name__ == "__main__":
    main()
