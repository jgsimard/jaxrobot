from abc import ABC, abstractmethod

import jax.random
import jax_dataclasses as jdc
from jax import numpy as jnp

from utils import wrap


class MeasurementModel(ABC):
    @abstractmethod
    def predict(self, x: jnp.ndarray, landmark: jnp.ndarray | None) -> jnp.ndarray:
        raise NotImplementedError


class SimpleProblemMeasurementModel(MeasurementModel):
    def predict(self, x: jnp.ndarray, _landmark: jnp.ndarray | None = None) -> jnp.ndarray:
        return x[:2, 0].reshape(2, 1)


class MotionModel(ABC):
    @abstractmethod
    def predict(self, x: jnp.ndarray, u: jnp.ndarray, dt: float) -> jnp.ndarray:
        raise NotImplementedError

    @abstractmethod
    def sample(self, x: jnp.ndarray, u: jnp.ndarray, dt: float, rng: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    @abstractmethod
    def noise_control_space(self, u: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError


class SimpleProblemMotionModel(MotionModel):
    def predict(self, x: jnp.ndarray, u: jnp.ndarray, dt: float) -> jnp.ndarray:
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
             [new_v]],
        )
        # fmt: on

    def sample(self, x: jnp.ndarray, u: jnp.ndarray, dt: float) -> jnp.ndarray:
        raise NotImplementedError

    def noise_control_space(self, u: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError


@jdc.pytree_dataclass
class Velocity(MotionModel):
    a1: float
    a2: float
    a3: float
    a4: float
    a5: float
    a6: float

    def predict(self, x: jnp.ndarray, u: jnp.ndarray, dt: float) -> jnp.ndarray:
        theta = x[2]
        # control
        v = u[0]
        w = u[1]

        return jnp.ndarray(
            [
                [x[0] + v / w * (-jnp.sin(theta) + jnp.sin(theta + w * dt))],
                [x[1] + v / w * (jnp.cos(theta) - jnp.cos(theta + w * dt))],
                [wrap(x[2] + w * dt)],
            ],
        )

    def sample(self, x: jnp.ndarray, u: jnp.ndarray, dt: float, rng: jnp.ndarray) -> jnp.ndarray:
        theta = x[2]
        # control
        v = u[0]
        w = u[1]

        v2 = v**2
        w2 = w**2

        key_v, key_w, key_gamma = jax.random.split(rng, 3)
        v_noisy = v + jnp.sqrt(self.a1 * v2 + self.a2 * w2) * jax.random.normal(key_v)
        w_noisy = w + jnp.sqrt(self.a3 * v2 + self.a4 * w2) * jax.random.normal(key_w)
        gamma_noisy = jnp.sqrt(self.a5 * v2 + self.a6 * w2) * jax.random.normal(key_gamma)

        return jnp.ndarray(
            [
                [x[0] + v_noisy / w_noisy * (-jnp.sin(theta) + jnp.sin(theta + w_noisy * dt))],
                [x[1] + v_noisy / w_noisy * (jnp.cos(theta) - jnp.cos(theta + w_noisy * dt))],
                [wrap(x[2] + w_noisy * dt + gamma_noisy * dt)],
            ],
        )

    def noise_control_space(self, u: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError


class RangeBearing(MeasurementModel):
    def predict(self, x: jnp.ndarray, landmark: jnp.ndarray) -> jnp.ndarray:
        # state
        x_x = x[0]
        x_y = x[1]
        x_theta = x[2]
        # landmark
        l_x = landmark[0]
        l_y = landmark[1]

        q = (l_x - x_x) ** 2 + (l_y - x_y) ** 2
        range_ = q.sqrt()

        bearing = jnp.arctan2(l_y - x_y, l_x - x_x) - x_theta
        return jnp.array([range_, bearing])
