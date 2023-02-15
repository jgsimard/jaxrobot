import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def rot_mat_2d(angle):
    """Create 2D rotation matrix from an angle
    Parameters.

    ----------
    angle :
    Returns
    -------
    A 2D rotation matrix
    Examples
    --------
    >>> angle_mod(-4.0).
    """
    return Rot.from_euler("z", angle).as_matrix()[0:2, 0:2]


def plot_covariance_ellipse(xy, variance_xy):  # pragma: no cover
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
