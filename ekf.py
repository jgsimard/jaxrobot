from functools import partial

import jax
import jax_dataclasses as jdc
import numpy as np
from jax import numpy as jnp

from models import MeasurementModel, MotionModel
from utils import GaussianState


@jdc.pytree_dataclass
class ExtendedKalmanFilter:
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

    def predict(self, estimate: GaussianState, u: jnp.ndarray, dt: float) -> GaussianState:
        return _predict(estimate, u, dt, self.motion, self.Q)

    def update(self, prediction: GaussianState, z: jnp.ndarray) -> GaussianState:
        return _update(prediction, z, self.measurement, self.R)


@partial(jax.jit, static_argnums=(3,))
def _predict(
    estimate: GaussianState,
    u: jnp.ndarray,
    dt: float,
    motion: MotionModel,
    q: jnp.ndarray,
) -> GaussianState:
    x_pred = motion.predict(estimate.x, u, dt)
    jf = jax.jacobian(motion.predict, argnums=0)(x_pred, u, dt).squeeze()
    p_pred = jf @ estimate.cov @ jf.T + q
    return GaussianState(x_pred, p_pred)


@partial(jax.jit, static_argnums=(2,))
def _update(
    prediction: GaussianState,
    z: jnp.ndarray,
    measurement: MeasurementModel,
    r: jnp.ndarray,
) -> GaussianState:
    z_pred = measurement.predict(prediction.x)
    y = z - z_pred
    jh = jax.jacobian(measurement.predict, argnums=0)(prediction.x).squeeze()
    s = jh @ prediction.cov @ jh.T + r
    kalman_gain = prediction.cov @ jh.T @ jnp.linalg.inv(s)
    return GaussianState(
        x=prediction.x + kalman_gain @ y,
        cov=(jnp.eye(len(prediction.x)) - kalman_gain @ jh) @ prediction.cov,
    )


if __name__ == "__main__":
    import time

    from models import SimpleProblemMeasurementModel, SimpleProblemMotionModel
    from utils import History, plot

    # Covariance for EKF simulation
    # State Vector [x y yaw v]
    Q = jnp.diag(jnp.array([0.1, 0.1, np.deg2rad(1.0), 1.0])) ** 2
    R = jnp.diag(jnp.array([1.0, 1.0])) ** 2  # Observation x,y position covariance
    ekf = ExtendedKalmanFilter(
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
    x_dead_rekon = np.zeros((4, 1))

    history = History(gs_est.x, x_true, x_true, np.zeros((2, 1)))

    u = jnp.array([[1.0], [0.1]])

    t0 = time.time()
    sim_time = 0.0
    while sim_time <= SIM_TIME:
        # skip compilation step in time tracking : ~500 ms
        if sim_time == DT:
            t0 = time.time()
        sim_time += DT

        rng, obs_key = jax.random.split(rng, 2)
        x_true, z, x_dead_rekon, ud = observation(x_true, x_dead_rekon, u, obs_key)

        gs_pred = ekf.predict(gs_est, ud, DT)
        gs_est = ekf.update(gs_pred, z)

        # store data history
        history.push(gs_est.x, x_dead_rekon, x_true, z)

        if show_animation:
            plot(history, gs_est)

    print(f"{(time.time() - t0) * 1000:.3f} ms")  # noqa: T201
