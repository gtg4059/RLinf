"""Right-edge overlay text stays off the left side of the frame."""

import numpy as np

from rlinf.envs.utils import put_info_on_image


def test_right_overlay_is_white_and_sits_on_the_right():
    image = np.zeros((80, 320, 3), dtype=np.uint8)
    drawn = put_info_on_image(image, {"CRI": 1.25, "dist_cm": 8.5}, align="right")
    left = drawn[:, :40]
    right = drawn[:, -80:]
    assert int(left.sum()) == 0
    assert int(right.max()) == 255
    assert int(right.sum()) > 0
