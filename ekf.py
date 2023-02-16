import time
from collections.abc import Callable
from functools import partial

import jax
import jax_dataclasses as jdc
import numpy as np
from jax import numpy as jnp

from utils import plot_covariance_ellipse


@jdc.pytree_dataclass
class GaussianState:
    """
    Attributes:
        x (jnp.ndarray): State Vector
        P (jnp.ndarray): Covariance Matrix
    """

    x: jnp.ndarray
    P: jnp.ndarray


@jdc.pytree_dataclass
class ExtendedKalmanFilter:
    """
    Attributes:
        R (jnp.ndarray): Observation Covariance Matrix
        Q (jnp.ndarray): Process Covariance Matrix
        F (Callable): State Transition Function
        H (Callable): Observation Model
    """

    R: jnp.ndarray
    Q: jnp.ndarray
    F: Callable[[jnp.ndarray, jnp.ndarray, float], jnp.ndarray]
    H: Callable[[jnp.ndarray], jnp.ndarray]

    def predict(self, estimate: GaussianState, u: jnp.ndarray, dt: float) -> GaussianState:
        return _predict(estimate, u, dt, self.F, self.Q)

    def update(self, prediction: GaussianState, z: jnp.ndarray) -> GaussianState:
        return _update(prediction, z, self.H, self.R)


@partial(jax.jit, static_argnums=(3,))
def _predict(
    estimate: GaussianState,
    u: jnp.ndarray,
    dt: float,
    f: Callable[[jnp.ndarray, jnp.ndarray, float], jnp.ndarray],
    q: jnp.ndarray,
) -> GaussianState:
    x_pred = f(estimate.x, u, dt)
    jf = jax.jacobian(f, argnums=0)(x_pred, u, dt).squeeze()
    p_pred = jf @ estimate.P @ jf.T + q
    return GaussianState(x_pred, p_pred)


@partial(jax.jit, static_argnums=(2,))
def _update(
    prediction: GaussianState,
    z: jnp.ndarray,
    h: Callable[[jnp.ndarray], jnp.ndarray],
    r: jnp.ndarray,
) -> GaussianState:
    z_pred = h(prediction.x)
    y = z - z_pred
    jh = jax.jacobian(h, argnums=0)(prediction.x).squeeze()
    s = jh @ prediction.P @ jh.T + r
    kalman_gain = prediction.P @ jh.T @ jnp.linalg.inv(s)
    return GaussianState(
        x=prediction.x + kalman_gain @ y,
        P=(jnp.eye(len(prediction.x)) - kalman_gain @ jh) @ prediction.P,
    )


if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt

    def motion_model(x: jnp.ndarray, u: jnp.ndarray, dt: float):
        # State Vector [x y yaw v]
        # Input Vector [new_v yaw_dot]
        # x_{t+1} = x_t + cos(yaw) * v_t * dt
        # y_{t+1} = y_t + sin(yaw) * v_t * dt
        # yaw_{t+1} = yaw_t + yaw_dot * dt
        # v_{t+1} = new_v
        x_t, y_t, yaw_t, v_t = x[:, 0]
        new_v, yaw_dot = u[:, 0]
        # fmt: off
        return jnp.array(
            [[x_t + jnp.cos(yaw_t) * v_t * dt],
             [y_t + jnp.sin(yaw_t) * v_t * dt],
             [yaw_t + yaw_dot * dt],
             [new_v]]
        )
        # fmt: on

    def observation_model(x: jnp.ndarray):
        return x[:2, 0].reshape(2, 1)

    # Covariance for EKF simulation
    # State Vector [x y yaw v]
    Q = jnp.diag(jnp.array([0.1, 0.1, np.deg2rad(1.0), 1.0])) ** 2
    R = jnp.diag(jnp.array([1.0, 1.0])) ** 2  # Observation x,y position covariance
    ekf = ExtendedKalmanFilter(R, Q, motion_model, observation_model)

    #  Simulation parameter
    INPUT_NOISE = jnp.array(np.diag([1.0, np.deg2rad(30.0)]) ** 2)
    GPS_NOISE = jnp.array(np.diag([0.5, 0.5]) ** 2)
    DT = 0.1  # time tick [s]
    SIM_TIME = 50.0  # simulation time [s]
    seed = 1234
    rng = jax.random.PRNGKey(seed)

    show_animation = False

    @jax.jit
    def observation(xTrue, xd, u, rng):
        xTrue = motion_model(xTrue, u, DT)

        key_obs, key_u = jax.random.split(rng, 2)
        # add noise to gps x-y
        z = observation_model(xTrue) + GPS_NOISE @ jax.random.normal(key_obs, (2, 1))

        # add noise to input
        ud = u + INPUT_NOISE @ jax.random.normal(key_u, (2, 1))

        xd = motion_model(xd, ud, DT)

        return xTrue, z, xd, ud

    print(__file__ + " start!!")

    # State Vector [x y yaw v]'
    xTrue = jnp.zeros((4, 1))
    gs_est = GaussianState(jnp.zeros((4, 1)), jnp.eye(4))
    xDR = np.zeros((4, 1))  # Dead reckoning

    # history
    hxEst = gs_est.x
    hxTrue = xTrue
    hxDR = xTrue
    hz = jnp.zeros((2, 1))
    u = jnp.array([[1.0], [0.1]])

    t0 = time.time()
    sim_time = 0.0
    while sim_time <= SIM_TIME:
        # skip compilation step in time tracking : ~500 ms
        if sim_time == DT:
            t0 = time.time()
        sim_time += DT

        rng, obs_key = jax.random.split(rng, 2)
        xTrue, z, xDR, ud = observation(xTrue, xDR, u, obs_key)

        gs_pred = ekf.predict(gs_est, ud, DT)
        gs_est = ekf.update(gs_pred, z)

        # store data history
        hxEst = np.hstack((hxEst, gs_est.x))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        hz = np.hstack((hz, z))

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                "key_release_event", lambda event: [sys.exit(0) if event.key == "escape" else None]
            )
            plt.plot(hz[0, :], hz[1, :], ".g")
            plt.plot(hxTrue[0, :].flatten(), hxTrue[1, :].flatten(), "-b")
            plt.plot(hxDR[0, :].flatten(), hxDR[1, :].flatten(), "-k")
            plt.plot(hxEst[0, :].flatten(), hxEst[1, :].flatten(), "-r")
            plot_covariance_ellipse(gs_est.x, gs_est.P)
            plt.axis("equal")
            plt.grid()
            plt.pause(0.01)
    print(f"{(time.time() - t0) * 1000:.3f} ms")
