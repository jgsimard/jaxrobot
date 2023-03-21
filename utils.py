import dataclasses
import math
import sys

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
        cov (jnp.ndarray): Covariance Matrix
    """

    x: jnp.ndarray
    cov: jnp.ndarray


def rot_mat_2d(angle: float) -> np.ndarray:
    """Create 2D rotation matrix from an angle
    Parameters.
    """
    return Rot.from_euler("z", angle).as_matrix()[0:2, 0:2]


def plot_covariance_ellipse(xy: np.ndarray, variance_xy: np.ndarray) -> None:  # pragma: no cover
    p_xy = variance_xy[0:2, 0:2]
    eigval, eigvec = np.linalg.eigh(p_xy)

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


@dataclasses.dataclass
class History:
    x_est: np.ndarray
    x_true: np.ndarray
    x_dead_rekon: np.ndarray
    z: np.ndarray

    def push(
        self,
        x_est: np.ndarray,
        x_dead_rekon: np.ndarray,
        x_true: np.ndarray,
        z: np.ndarray,
    ) -> None:
        self.x_est = np.hstack((self.x_est, x_est))
        self.x_dead_rekon = np.hstack((self.x_dead_rekon, x_dead_rekon))
        self.x_true = np.hstack((self.x_true, x_true))
        self.z = np.hstack((self.z, z))


def plot(history: History, gs: GaussianState) -> None:
    plt.cla()
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        "key_release_event",
        lambda event: [sys.exit(0) if event.key == "escape" else None],
    )
    plt.plot(history.z[0, :], history.z[1, :], ".g")
    plt.plot(history.x_true[0, :].flatten(), history.x_true[1, :].flatten(), "-b")
    plt.plot(history.x_dead_rekon[0, :].flatten(), history.x_dead_rekon[1, :].flatten(), "-k")
    plt.plot(history.x_est[0, :].flatten(), history.x_est[1, :].flatten(), "-r")
    plot_covariance_ellipse(gs.x[:2].reshape(2, 1), gs.cov)
    plt.axis("equal")
    plt.grid()
    plt.pause(0.001)
