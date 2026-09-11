import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Union
from robots.hands.dh3_hand.dh3_hand import DH3Hand
from robots.hands.hand_collision_base import HANDCOLLISION, HandCollisionBase, create_hand_collision
from robots.hands.hand_base import create_hand_model


@HANDCOLLISION.register()
class DH3HandCollision(HandCollisionBase):
    """Collision model for DH3 Hand.

    Args:
        device (Union[str, torch.device]): device to load the model on.
        margin (float): margin for the collision model.
    """

    def __init__(
        self,
        device: Union[str, torch.device] = "cpu",
        margin: float = 0.0,
    ):
        """Configure right-hand collision spheres on the requested device with a metric margin."""
        super().__init__(
            urdf_path=DH3Hand.URDF_PATH,
            device=device,
            margin=margin,
        )
        self.hand_name = "allegro_hand"


if __name__ == "__main__":

    hand = create_hand_model("DH3Hand", use_cache=True)
    hand_collision = create_hand_collision("DH3HandCollision", device="cpu", margin=0.0)

    joints = hand.get_predefined_joints(1, 8)
    joints = torch.tensor(joints)

    # example 1: visualize sampled point cloud from collision model
    hand_collision.visualize_sampled_pc(
        joints=joints, translations=[0.2, 0.1, 0.1], rotations=[0.0, 0.0, 10.0], num_points=8192
    )

    # # example 2: visualize collision spheres
    # hand_collision.visualize_transformed_spheres(joints)
