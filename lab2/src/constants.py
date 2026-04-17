from enum import Enum


class CellType(Enum):
    CELL_AIR = 0
    CELL_WATER = 1
    CELL_SOLID = 2


particle_offset = (
    (0.25, 0.25, 0.25),
    (0.75, 0.25, 0.25),
    (0.25, 0.75, 0.25),
    (0.75, 0.75, 0.25),
    (0.25, 0.25, 0.75),
    (0.75, 0.25, 0.75),
    (0.25, 0.75, 0.75),
    (0.75, 0.75, 0.75),
)
