from .advection import *
from .collision import *
from .density import *
from .divergence import *
from .separation import *
from .transfer import *
from dataclasses import dataclass


@dataclass
class FluidStrategies:
    advection: AdvectionStrategyBase
    collision: CollisionStrategyBase
    separation: SeparationStrategyBase
    transfer: TransferStrategyBase
    density: DensityStrategyBase
    divergence: DivergenceStrategyBase
