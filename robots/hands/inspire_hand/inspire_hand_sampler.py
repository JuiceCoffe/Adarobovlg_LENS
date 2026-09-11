import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Optional, Union
from robots.hands.hand_sampler_base import HandSamplerBase, HANDSAMPLER, create_hand_sampler
from robots.hands.inspire_hand.inspire_hand import InspireHand
from robots.hands.hand_base import create_hand_model


@HANDSAMPLER.register()
class InspireHandSampler(HandSamplerBase):
    """Sampler for Inspire Hand."""

    def __init__(
        self,
        device: Union[str, torch.device] = "cpu",
        num_fixed_points: Optional[int] = None,
        use_cache: bool = False,
        max_points: int = 100000,
    ):
        """Initialize the Inspire Hand Sampler.

        Args:
            urdf_path (str): The file path to the URDF of the Inspire hand.
            device [torch.device or str]: The device to be used for computation (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.
            num_fixed_points (Optional[int], optional): The number of fixed points to use. Defaults to None.
            use_cache (bool, optional): Whether to use caching for internal operations. Defaults to False.
            max_points (int, optional): The maximum number of points allowed. Defaults to 100000.
        """
        super().__init__(
            urdf_path=InspireHand.URDF_PATH,
            device=device,
            num_fixed_points=num_fixed_points,
            use_cache=use_cache,
            pointcloud_cache=InspireHand.POINTCLOUD_CACHE,
            max_points=max_points,
        )
        self.hand_name = "inspire_hand"


if __name__ == "__main__":

    hand = create_hand_model("InspireHand", use_cache=True)
    hand_sampler = create_hand_sampler(
        hand_name="InspireHandSampler",
        device="cpu",
        num_fixed_points=32768,
        use_cache=True,
    )
    joints = hand.get_predefined_joints(1, 8)
    hand.vis_predefined_grasp_type(1, 8)

    # # example 1
    # joints = hand.get_predefined_joints(1, 8)
    # joints = torch.cat(
    #     [torch.zeros(6), torch.tensor(joints)], dim=0
    # ).to(torch.float32)
    # hand_sampler.visualize_sampled_pc(joints, num_points=8192)

    # example 2
    # joints = hand_sampler.get_canonical_joints()
    hand_sampler.visualize_sampled_pc(
        joints=joints,
        translations=[0.2, 0.3, 0.5],
        rotations=[0.0, 0.0, 0.0],
        num_points=32768,
        visualize_normals=False,
    )
