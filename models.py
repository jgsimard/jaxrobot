from abc import ABC, abstractmethod

import jax_dataclasses as jdc
from jax import numpy as jnp


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
    def sample(self, x: jnp.ndarray, u: jnp.ndarray, dt: float) -> jnp.ndarray:
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

    @abstractmethod
    def predict(self, x: jnp.ndarray, u: jnp.ndarray, dt: float) -> jnp.ndarray:
        raise NotImplementedError

    @abstractmethod
    def sample(self, x: jnp.ndarray, u: jnp.ndarray, dt: float) -> jnp.ndarray:
        raise NotImplementedError

    @abstractmethod
    def noise_control_space(self, u: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError


class RangeBearing(MeasurementModel):
    def predict(self, x: jnp.ndarray, landmark: jnp.ndarray | None) -> jnp.ndarray:
        raise NotImplementedError
