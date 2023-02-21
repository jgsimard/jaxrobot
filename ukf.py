import math
from collections.abc import Callable
from functools import partial

import jax
import jax_dataclasses as jdc
import numpy as np
from jax import numpy as jnp

from utils import GaussianState


@jdc.pytree_dataclass
class UnscentedKalmanFilter:
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
    mean_weights: jnp.ndarray = jdc.field(init=False)
    cov_weights: jnp.ndarray = jdc.field(init=False)
    gamma: float = jdc.field(init=False)
    alpha: float = 0.1
    beta: float = 2.0
    kappa: float = 0.0

    def __post_init__(self):
        mean_weights, cov_weights, gamma = self._sigma_weights(len(Q[0]))
        object.__setattr__(self, "mean_weights", mean_weights)
        object.__setattr__(self, "cov_weights", cov_weights)
        object.__setattr__(self, "gamma", gamma)

    def _sigma_weights(self, n: int) -> (jnp.ndarray, jnp.ndarray):
        _lambda = self.alpha**2 * (n + self.kappa) - n
        v = 1.0 / (2 * (n + _lambda))
        mean_weights = np.full(2 * n + 1, v)
        cov_weights = np.full(2 * n + 1, v)
        # special cases
        mean_weights[0] = _lambda / (n + _lambda)
        cov_weights[0] = _lambda / (n + _lambda) + (1 - self.alpha**2 + self.beta)

        gamma = math.sqrt(n + _lambda)

        return jnp.array(mean_weights), jnp.array(cov_weights), gamma

    def estimate(self, state, u, z, dt):
        return estimation(
            state,
            u,
            z,
            self.R,
            self.Q,
            self.F,
            self.H,
            dt,
            self.mean_weights,
            self.cov_weights,
            self.gamma,
        )


def generate_sigma_points(state: GaussianState, gamma) -> jnp.ndarray:
    n = len(state.x)
    mean = state.x.squeeze()
    Psigma = jax.scipy.linalg.sqrtm(state.P)
    plus = jnp.array([mean + gamma * Psigma[:, i : i + 1].squeeze() for i in range(n)])
    minus = jnp.array([mean - gamma * Psigma[:, i : i + 1].squeeze() for i in range(n)])
    sigma_points = jnp.concatenate((mean[None, :], plus, minus))[:, :, None]
    return sigma_points


def compute_cov(a, b, weights):
    return jnp.einsum("ni,nj,n->ij", a, b, weights)


def unscented_update(
    points: jnp.ndarray,
    mean_weights: jnp.ndarray,
    cov_weights: jnp.ndarray,
    base_cov: jnp.ndarray,
) -> GaussianState:
    mean = (points.squeeze() * mean_weights[:, None]).sum(axis=0)
    dx = (points.squeeze() - mean[None, :]).squeeze()
    cov = base_cov + compute_cov(
        dx, dx, cov_weights
    )  # jnp.einsum("ni,nj,n->ij", dx, dx, cov_weights)
    return GaussianState(x=mean, P=cov)


@partial(
    jax.jit,
    static_argnums=(
        5,  # F
        6,  # H
    ),
)
def estimation(
    state,
    u,
    z,
    R,
    Q,
    F,
    H,
    dt,
    mean_weights,
    cov_weights,
    gamma,
) -> GaussianState:
    # mean_weights, cov_weights = sigma_weights(len(state.x))

    # predict
    sigma_points = generate_sigma_points(state, gamma)
    vF = jax.vmap(F, (0, None, None), 0)  # vectorize
    state_sigma_points = vF(sigma_points, u, dt)
    x_pred = unscented_update(state_sigma_points, mean_weights, cov_weights, base_cov=Q)

    # update
    x_pred_sigma_points = generate_sigma_points(x_pred, gamma)
    vH = jax.vmap(H, (0,), 0)
    z_sigma_points = vH(sigma_points)
    z_pred = unscented_update(z_sigma_points, mean_weights, cov_weights, base_cov=R)

    cov_ = compute_cov(
        x_pred_sigma_points.squeeze() - x_pred.x[None, :],
        z_sigma_points.squeeze() - z_pred.x[None, :],
        cov_weights,
    )

    kalman_gain = cov_ @ jnp.linalg.inv(z_pred.P)
    y = z.squeeze() - z_pred.x
    return GaussianState(
        x=x_pred.x + kalman_gain @ y,
        P=x_pred.P - kalman_gain @ z_pred.P @ kalman_gain.T,
    )


if __name__ == "__main__":
    import sys
    import time

    import matplotlib.pyplot as plt

    from utils import plot_covariance_ellipse

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
    ukf = UnscentedKalmanFilter(R=R, Q=Q, F=motion_model, H=observation_model)
    # mean_weights, cov_weights, gamma = sigma_weights(4)

    #  Simulation parameter
    INPUT_NOISE = jnp.array(np.diag([1.0, np.deg2rad(30.0)]) ** 2)
    GPS_NOISE = jnp.array(np.diag([0.5, 0.5]) ** 2)
    DT = 0.1  # time tick [s]
    SIM_TIME = 50.0  # simulation time [s]
    seed = 1234
    rng = jax.random.PRNGKey(seed)

    show_animation = True
    # show_animation = False

    @jax.jit
    def observation(xTrue, xd, command, rng):
        xTrue = motion_model(xTrue, command, DT)
        key_obs, key_u = jax.random.split(rng, 2)
        # add noise to gps x-y
        obs = observation_model(xTrue) + GPS_NOISE @ jax.random.normal(key_obs, (2, 1))
        # add noise to input
        command_noisy = command + INPUT_NOISE @ jax.random.normal(key_u, (2, 1))
        xd = motion_model(xd, command_noisy, DT)
        return xTrue, obs, xd, command_noisy

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
        # skip compilation step in time tracking
        if sim_time == DT:
            t0 = time.time()
        sim_time += DT

        rng, obs_key = jax.random.split(rng, 2)
        xTrue, z, xDR, ud = observation(xTrue, xDR, u, obs_key)

        gs_est = ukf.estimate(gs_est, u, z, DT)

        # store data history
        hxEst = np.hstack((hxEst, gs_est.x[:, None]))
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
            plot_covariance_ellipse(gs_est.x[:, None], gs_est.P)
            plt.axis("equal")
            plt.grid()
            plt.pause(0.01)
    print(f"{(time.time() - t0) * 1000:.3f} ms")
