import math

import jax_dataclasses as jdc
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp
from scipy.spatial.transform import Rotation as Rot


@jdc.pytree_dataclass
class GaussianState:
    """
    Attributes:
        x (jnp.ndarray): State Vector
        P (jnp.ndarray): Covariance Matrix
    """

    x: jnp.ndarray
    P: jnp.ndarray


def rot_mat_2d(angle: float) -> np.ndarray:
    """Create 2D rotation matrix from an angle
    Parameters.
    """
    return Rot.from_euler("z", angle).as_matrix()[0:2, 0:2]


def plot_covariance_ellipse(xy: np.ndarray, variance_xy: np.ndarray) -> None:  # pragma: no cover
    p_xy = variance_xy[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(p_xy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    a = math.sqrt(eigval[bigind])
    b = math.sqrt(eigval[smallind])
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eigvec[1, bigind], eigvec[0, bigind])
    fx = rot_mat_2d(angle) @ (np.array([x, y]))
    px = np.array(fx[0, :] + xy[0, 0]).flatten()
    py = np.array(fx[1, :] + xy[1, 0]).flatten()
    plt.plot(px, py, "--r")
