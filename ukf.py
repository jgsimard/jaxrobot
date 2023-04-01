import math
from functools import partial

import jax
import jax_dataclasses as jdc
import numpy as np
from jax import numpy as jnp

from models import MeasurementModel, MotionModel
from utils import GaussianState


@jdc.pytree_dataclass
class UnscentedKalmanFilter:
    """
    Attributes:
        R (jnp.ndarray): Observation Covariance Matrix
        Q (jnp.ndarray): Process Covariance Matrix
        motion (MotionModel): State Transition Function
        measurement (MeasurementModel): Observation Model
    """

    R: jnp.ndarray
    Q: jnp.ndarray
    motion: MotionModel
    measurement: MeasurementModel
    mean_weights: jnp.ndarray = jdc.field(init=False)
    cov_weights: jnp.ndarray = jdc.field(init=False)
    gamma: float = jdc.field(init=False)
    alpha: float = 0.1
    beta: float = 2.0
    kappa: float = 0.0

    def __post_init__(self) -> None:
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

    def estimate(
        self,
        state: GaussianState,
        u: jnp.ndarray,
        z: jnp.ndarray,
        dt: float,
    ) -> GaussianState:
        return estimation(
            state,
            u,
            z,
            self.R,
            self.Q,
            self.motion,
            self.measurement,
            dt,
            self.mean_weights,
            self.cov_weights,
            self.gamma,
        )


def generate_sigma_points(state: GaussianState, gamma: float) -> jnp.ndarray:
    n = len(state.x)
    mean = state.x.squeeze()
    cov_sqrt = jax.scipy.linalg.sqrtm(state.cov)
    plus = jnp.array([mean + gamma * cov_sqrt[:, i : i + 1].squeeze() for i in range(n)])
    minus = jnp.array([mean - gamma * cov_sqrt[:, i : i + 1].squeeze() for i in range(n)])
    return jnp.concatenate((mean[None, :], plus, minus))[:, :, None]


@partial(jax.jit, static_argnums=(5, 6))
def estimation(
    state: GaussianState,
    u: jnp.ndarray,
    z: jnp.ndarray,
    r: jnp.ndarray,
    q: jnp.ndarray,
    motion: MotionModel,
    measurement: MeasurementModel,
    dt: float,
    mean_weights: jnp.ndarray,
    cov_weights: jnp.ndarray,
    gamma: float,
) -> GaussianState:
    # predict
    sigma_points = generate_sigma_points(state, gamma)
    vf = jax.vmap(motion.predict, (0, None, None), 0)  # vectorize
    state_sigma_points = vf(sigma_points, u, dt)
    x_pred = unscented_update(state_sigma_points, mean_weights, cov_weights, base_cov=q)

    # update
    x_pred_sigma_points = generate_sigma_points(x_pred, gamma)
    vh = jax.vmap(measurement.predict, (0,), 0)
    z_sigma_points = vh(sigma_points)
    z_pred = unscented_update(z_sigma_points, mean_weights, cov_weights, base_cov=r)

    cov_ = compute_cov(
        x_pred_sigma_points.squeeze() - x_pred.x[None, :],
        z_sigma_points.squeeze() - z_pred.x[None, :],
        cov_weights,
    )

    kalman_gain = cov_ @ jnp.linalg.inv(z_pred.cov)
    y = z.squeeze() - z_pred.x
    return GaussianState(
        x=x_pred.x + kalman_gain @ y,
        cov=x_pred.cov - kalman_gain @ z_pred.cov @ kalman_gain.T,
    )


def compute_cov(a: jnp.ndarray, b: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
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
        dx,
        dx,
        cov_weights,
    )
    return GaussianState(x=mean, cov=cov)


if __name__ == "__main__":
    import time

    from models import SimpleProblemMeasurementModel, SimpleProblemMotionModel
    from utils import History, plot

    # Covariance for UKF simulation
    # State Vector [x y yaw v]
    Q = jnp.diag(jnp.array([0.1, 0.1, np.deg2rad(1.0), 1.0])) ** 2
    R = jnp.diag(jnp.array([1.0, 1.0])) ** 2  # Observation x,y position covariance
    ukf = UnscentedKalmanFilter(
        R=R,
        Q=Q,
        motion=SimpleProblemMotionModel(),
        measurement=SimpleProblemMeasurementModel(),
    )

    #  Simulation parameter
    INPUT_NOISE = jnp.array(np.diag([1.0, np.deg2rad(30.0)]) ** 2)
    GPS_NOISE = jnp.array(np.diag([0.5, 0.5]) ** 2)
    DT = 0.1  # time tick [s]
    SIM_TIME = 50.0  # simulation time [s]
    seed = 1234
    rng = jax.random.PRNGKey(seed)

    show_animation = False

    @jax.jit
    def observation(
        x_true: jnp.ndarray,
        xd: jnp.ndarray,
        command: jnp.ndarray,
        rng: jnp.ndarray,
    ) -> (jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray):
        x_true = SimpleProblemMotionModel().predict(x_true, command, DT)
        key_obs, key_u = jax.random.split(rng, 2)
        # add noise to gps x-y
        obs = SimpleProblemMeasurementModel().predict(x_true) + GPS_NOISE @ jax.random.normal(
            key_obs,
            (2, 1),
        )
        # add noise to input
        command_noisy = command + INPUT_NOISE @ jax.random.normal(key_u, (2, 1))
        xd = SimpleProblemMotionModel().predict(xd, command_noisy, DT)
        return x_true, obs, xd, command_noisy

    print(__file__ + " start!!")  # noqa: T201

    # State Vector [x y yaw v]'
    x_true = jnp.zeros((4, 1))
    gs_est = GaussianState(jnp.zeros((4, 1)), jnp.eye(4))
    x_dead_rekon = np.zeros((4, 1))  # Dead reckoning

    # history
    history = History(gs_est.x, x_true, x_true, np.zeros((2, 1)))
    u = jnp.array([[1.0], [0.1]])

    t0 = time.time()
    sim_time = 0.0
    while sim_time <= SIM_TIME:
        # skip compilation step in time tracking
        if sim_time == DT:
            t0 = time.time()
        sim_time += DT

        rng, obs_key = jax.random.split(rng, 2)
        x_true, z, x_dead_rekon, ud = observation(x_true, x_dead_rekon, u, obs_key)

        gs_est = ukf.estimate(gs_est, u, z, DT)

        # store data history
        history.push(gs_est.x[:, None], x_dead_rekon, x_true, z)

        if show_animation:
            plot(history, gs_est)
    print(f"{(time.time() - t0) * 1000:.3f} ms")  # noqa: T201
