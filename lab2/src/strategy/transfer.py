from abc import ABC, abstractmethod
from scene import Scene


class TransferStrategyBase(ABC):
    @abstractmethod
    def handle_transfer(self, is_p2g):
        """Perform velocity transfer between particles and grid."""
        return NotImplementedError


class FlipTransferStrategy(TransferStrategyBase):
    def __init__(self, scene: Scene, flip_ratio):
        self.scene = scene
        self.flip_ratio = flip_ratio

    def handle_transfer(self, is_p2g):
        pass
