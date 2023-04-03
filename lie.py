from abc import ABC, abstractmethod
from typing import TypeVar

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np

LieGroupType = TypeVar("LieGroupType", bound="LieGroup")


class LieGroup(ABC):
    def __matmul__(
        self: "LieGroup",
        other: LieGroupType | jnp.ndarray | np.ndarray,
    ) -> jnp.ndarray | LieGroupType:
        if isinstance(other, jnp.ndarray | np.ndarray):
            return self.apply(point=other)
        if isinstance(other, LieGroup):
            return self.multiply(other=other)
        msg = "sacre bleu"
        raise NotImplementedError(msg)

    @abstractmethod
    def as_matrix(self) -> jnp.ndarray:
        """Get transformation as a matrix. Homogeneous for SE groups."""

    @abstractmethod
    def as_vec(self) -> jnp.ndarray:
        """Get transformation as a vec"""

    @staticmethod
    @abstractmethod
    def from_matrix(matrix: jnp.ndarray) -> "LieGroup":
        """Get lie group member from matrix representation"""

    @abstractmethod
    def inverse(self: "LieGroup") -> "LieGroup":
        """Computes the inverse of our transform."""

    @abstractmethod
    def apply(self, point: jnp.ndarray) -> jnp.ndarray:
        """Applies lie group action to a point"""

    @abstractmethod
    def multiply(self: "LieGroup", other: "LieGroup") -> "LieGroup":
        """Composes this transformation with another"""


@jdc.pytree_dataclass
class SO2(LieGroup):
    """SO(2) : 2D Special Orthogonal Group : Rotation"""

    vals: jnp.ndarray  # cos, sin

    @staticmethod
    def from_theta(theta: float) -> "SO2":
        return SO2(np.ndarray([jnp.cos(theta), jnp.sin(theta)]))

    @staticmethod
    def from_matrix(matrix: jnp.ndarray) -> "SO2":
        return SO2(matrix[:, 0])

    @jax.jit
    def inverse(self) -> "SO2":
        return SO2(self.vals * jnp.array([1, -1]))

    @jax.jit
    def as_matrix(self) -> jnp.ndarray:
        cos, sin = self.vals
        return jnp.array(
            [
                [cos, -sin],
                [sin, cos],
            ],
        )

    def as_vec(self) -> jnp.ndarray:
        raise NotImplementedError

    @jax.jit
    def derivative(self) -> jnp.ndarray:
        cos, sin = self.vals
        return jnp.array(
            [
                [-sin, -cos],
                [cos, -sin],
            ],
        )

    @jax.jit
    def apply(self, point: jnp.ndarray) -> jnp.ndarray:
        return self.as_matrix() @ point

    @jax.jit
    def multiply(self: "SO2", other: "SO2") -> "SO2":
        return SO2(self.as_matrix() @ other.vals)


@jdc.pytree_dataclass
class SE2(LieGroup):
    """SE(2) : 2D Special Euclidian Group : Rotation  + Translation"""

    vals: jnp.ndarray

    @staticmethod
    def from_xy_theta(x: float, y: float, theta: float) -> "SE2":
        return SE2(jnp.array([jnp.cos(theta), jnp.sin(theta), x, y]))

    @staticmethod
    def from_rotation_and_translation(rotation: SO2, translation: jnp.ndarray) -> "SE2":
        return SE2(jnp.concatenate([rotation.vals, translation]))

    @jax.jit
    def rotation(self) -> "SO2":
        return SO2(self.vals[:2])

    @jax.jit
    def translation(self) -> jnp.ndarray:
        return self.vals[2:]

    @jax.jit
    def as_matrix(self) -> jnp.ndarray:
        cos, sin, x, y = self.vals
        return jnp.array(
            [
                [cos, -sin, x],
                [sin, cos, y],
                [0.0, 0.0, 1.0],
            ],
        )

    @staticmethod
    @jax.jit
    def from_vec(v: jnp.ndarray) -> "SE2":
        x, y, theta = v
        return SE2.from_xy_theta(x, y, theta)

    @jax.jit
    def as_vec(self) -> jnp.ndarray:
        """Get transformation as a vec"""
        cos, sin, x, y = self.vals
        return jnp.array([x, y, jnp.arctan2(sin, cos)])

    @staticmethod
    def from_matrix(matrix: jnp.ndarray) -> "SE2":
        return SE2.from_rotation_and_translation(SO2.from_matrix(matrix[:2, :2]), matrix[:2, 2])

    @jax.jit
    def inverse(self) -> "SE2":
        r_inv = self.rotation().inverse()
        return SE2.from_rotation_and_translation(
            rotation=r_inv,
            translation=-(r_inv @ self.translation()),
        )

    @jax.jit
    def apply(self, point: jnp.ndarray) -> jnp.ndarray:
        return self.rotation() @ point + self.translation()

    @jax.jit
    def multiply(self: "SE2", other: "SE2") -> "SE2":
        return SE2.from_rotation_and_translation(
            rotation=self.rotation() @ other.rotation(),
            translation=self.rotation() @ other.translation() + self.translation(),
        )
