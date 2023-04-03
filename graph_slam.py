from dataclasses import dataclass
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from jax import numpy as jnp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from lie import SE2


@dataclass
class Edge:
    edge_type: str
    from_node: int
    to_node: int
    measurement: SE2 | Array
    information: Array


# TODO: use rustworkx instead
@dataclass
class Graph:
    x: np.ndarray
    nodes: dict[int, np.ndarray]
    edges: list[Edge]
    lut: dict[int, int]


def read_graph_g2o(filename: str) -> Graph:
    """This function reads the g2o text file as the graph class

    Parameters
    ----------
    filename : string
        path to the g2o file

    Returns
    -------
    graph: Graph contaning information for SLAM

    """
    edges = []
    nodes = {}
    with Path.open(filename) as file:
        for line in file:
            data = line.split()

            if data[0] == "VERTEX_SE2":
                node_id = int(data[1])
                pose = np.array(data[2:5], dtype=jnp.float32)
                nodes[node_id] = pose

            elif data[0] == "VERTEX_XY":
                node_id = int(data[1])
                loc = np.array(data[2:4], dtype=jnp.float32)
                nodes[node_id] = loc

            elif data[0] == "EDGE_SE2":
                edge_type = "P"
                from_node = int(data[1])
                to_node = int(data[2])
                measurement = np.array(data[3:6], dtype=jnp.float32)
                measurement = SE2.from_vec(measurement)
                uppertri = jnp.array(data[6:12], dtype=jnp.float32)
                information = np.array(
                    [
                        [uppertri[0], uppertri[1], uppertri[2]],
                        [uppertri[1], uppertri[3], uppertri[4]],
                        [uppertri[2], uppertri[4], uppertri[5]],
                    ],
                )
                edge = Edge(edge_type, from_node, to_node, measurement, information)
                edges.append(edge)

            elif data[0] == "EDGE_SE2_XY":
                edge_type = "L"
                from_node = int(data[1])
                to_node = int(data[2])
                measurement = np.array(data[3:5], dtype=jnp.float32).reshape((2, 1))
                uppertri = np.array(data[5:8], dtype=jnp.float32)
                information = jnp.array([[uppertri[0], uppertri[1]], [uppertri[1], uppertri[2]]])
                edge = Edge(edge_type, from_node, to_node, measurement, information)
                edges.append(edge)

            else:
                print("VERTEX/EDGE type not defined")

    # compute state vector and lookup table
    lut = {}
    x = []
    offset = 0
    for node_id in nodes:
        lut.update({node_id: offset})
        offset = offset + len(nodes[node_id])
        x.append(nodes[node_id])
    x = np.concatenate(x, axis=0)

    # collect nodes, edges and lookup in graph structure
    graph = Graph(x, nodes, edges, lut)
    print(f"Loaded graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")

    return graph


def plot_graph(g: Graph) -> None:
    # initialize figure
    plt.figure(1)
    plt.clf()

    # get a list of all poses and landmarks
    poses, landmarks = get_poses_landmarks(g)

    # plot robot poses
    if len(poses) > 0:
        poses = np.stack(poses, axis=0)
        plt.plot(poses[:, 0], poses[:, 1], "bo")

    # plot landmarks
    if len(landmarks) > 0:
        landmarks = np.stack(landmarks, axis=0)
        plt.plot(landmarks[:, 0], landmarks[:, 1], "r*")

    # plot edges/constraints
    pose_edges_p1 = []
    pose_edges_p2 = []
    landmark_edges_p1 = []
    landmark_edges_p2 = []

    for edge in g.edges:
        from_idx = g.lut[edge.from_node]
        to_idx = g.lut[edge.to_node]
        if edge.edge_type == "P":
            pose_edges_p1.append(g.x[from_idx : from_idx + 3])
            pose_edges_p2.append(g.x[to_idx : to_idx + 3])

        elif edge.edge_type == "L":
            landmark_edges_p1.append(g.x[from_idx : from_idx + 2])
            landmark_edges_p2.append(g.x[to_idx : to_idx + 2])

    pose_edges_p1 = np.stack(pose_edges_p1, axis=0)
    pose_edges_p2 = np.stack(pose_edges_p2, axis=0)
    plt.plot(
        np.concatenate((pose_edges_p1[:, 0], pose_edges_p2[:, 0])),
        np.concatenate((pose_edges_p1[:, 1], pose_edges_p2[:, 1])),
        "r",
    )

    plt.draw()
    plt.pause(1)


def get_poses_landmarks(g: Graph) -> (list, list):
    poses = []
    landmarks = []

    for node_id in g.nodes:
        dimension = len(g.nodes[node_id])
        offset = g.lut[node_id]

        if dimension == 3:
            pose = g.x[offset : offset + 3]
            poses.append(pose)
        elif dimension == 2:
            landmark = g.x[offset : offset + 2]
            landmarks.append(landmark)

    return poses, landmarks


def run_graph_slam(g: Graph, num_iterations: int) -> list[float]:
    tolerance = 1e-4
    norm_dx_all = []
    # perform optimization
    for i in range(num_iterations):
        # compute the incremental update dx of the state vector
        dx = linearize_and_solve(g)
        # apply the solution to the state vector g.x
        g.x += dx
        # plot graph
        plot_graph(g)
        # compute and print global error
        norm_dx = np.linalg.norm(dx)
        print(f"|dx| for step {i} : {norm_dx}\n")
        norm_dx_all.append(norm_dx)

        # terminate procedure if change is less than 10e-4
        if norm_dx < tolerance:
            break
    return norm_dx_all


def compute_global_error(g: Graph) -> float:
    """This function computes the total error for the graph.

    Parameters
    ----------
    g : Graph class

    Returns
    -------
    error: scalar
        Total error for the graph
    """
    error = 0.0
    for edge in g.edges:
        # get node state for the current edge
        from_idx = g.lut[edge.from_node]
        to_idx = g.lut[edge.to_node]

        # pose-pose constraint
        if edge.edge_type == "P":
            # get node state for the current edge
            x1 = SE2.from_vec(g.x[from_idx : from_idx + 3])
            x2 = SE2.from_vec(g.x[to_idx : to_idx + 3])

            # get measurement  and information matrix for the edge
            z = edge.measurement

            error += pose_pose_constraint_error(x1, x2, z)

        # pose-landmark constraint
        elif edge.edge_type == "L":
            # get node state for the current edge
            x = SE2.from_vec(g.x[from_idx : from_idx + 3])
            landmark = g.x[to_idx : to_idx + 2]

            # get measurement  and information matrix for the edge
            z = edge.measurement

            error += pose_landmark_error(x, landmark, z)

    return error


@jax.jit
def terms(e: Array, A: Array, B: Array, omega: Array) -> (Array, Array, Array, Array, Array, Array):
    b_i = A.T @ omega @ e
    b_j = B.T @ omega @ e

    h_ii = A.T @ omega @ A
    h_ij = A.T @ omega @ B
    h_ji = h_ij.T
    h_jj = B.T @ omega @ B

    return b_i, b_j, h_ii, h_ij, h_ji, h_jj


def linearize_and_solve(g: Graph) -> Array:
    """This function solves the least-squares problem for one iteration
        by linearizing the constraints

    Parameters
    ----------
    g : Graph class

    Returns
    -------
    dx : Nx1 vector
         change in the solution for the unknowns x
    """

    # initialize the sparse H and the vector b
    H = np.zeros((len(g.x), len(g.x)))
    b = np.zeros(len(g.x))

    # set flag to fix gauge
    need_to_add_prior = True

    # compute the addend term to H and b for each of our constraints
    print("linearize and build system")

    for edge in g.edges:
        # pose-pose constraint
        if edge.edge_type == "P":
            # compute idx for nodes using lookup table
            from_idx = g.lut[edge.from_node]
            to_idx = g.lut[edge.to_node]

            # get node state for the current edge
            x_i = SE2.from_vec(g.x[from_idx : from_idx + 3])
            x_j = SE2.from_vec(g.x[to_idx : to_idx + 3])
            z_ij = edge.measurement
            omega = edge.information

            # compute the error and the Jacobians
            e, A, B = linearize_pose_pose_constraint(x_i, x_j, z_ij)

            # compute the terms
            b_i, b_j, h_ii, h_ij, h_ji, h_jj = terms(e, A, B, omega)

            # add the terms to H matrix and b
            # Update H
            H[from_idx : from_idx + 3, from_idx : from_idx + 3] += h_ii
            H[from_idx : from_idx + 3, to_idx : to_idx + 3] += h_ij
            H[to_idx : to_idx + 3, from_idx : from_idx + 3] += h_ji
            H[to_idx : to_idx + 3, to_idx : to_idx + 3] += h_jj

            # Update b
            b[from_idx : from_idx + 3] += b_i
            b[to_idx : to_idx + 3] += b_j

            # Add the prior for one pose of this edge
            # This fixes one node to remain at its current location
            if need_to_add_prior:
                H[from_idx : from_idx + 3, from_idx : from_idx + 3] = H[
                    from_idx : from_idx + 3,
                    from_idx : from_idx + 3,
                ] + 1000 * np.eye(3)
                need_to_add_prior = False

        # pose-pose constraint
        elif edge.edge_type == "L":
            # compute idx for nodes using lookup table
            from_idx = g.lut[edge.from_node]
            to_idx = g.lut[edge.to_node]

            # get node states for the current edge
            x = SE2.from_vec(g.x[from_idx : from_idx + 3])
            landmark = g.x[to_idx : to_idx + 2].reshape(2, 1)
            omega = edge.information

            # compute the error and the Jacobians
            e, A, B = linearize_pose_landmark_constraint(x, landmark, edge.measurement)

            # compute the terms
            b_i, b_j, h_ii, h_ij, h_ji, h_jj = terms(e, A, B, omega)

            # add the terms to H matrix and b
            # Update H
            H[from_idx : from_idx + 3, from_idx : from_idx + 3] += h_ii
            H[from_idx : from_idx + 3, to_idx : to_idx + 2] += h_ij
            H[to_idx : to_idx + 2, from_idx : from_idx + 3] += h_ji
            H[to_idx : to_idx + 2, to_idx : to_idx + 2] += h_jj

            # Update b
            b[from_idx : from_idx + 3] += b_i.squeeze()
            b[to_idx : to_idx + 2] += b_j.squeeze()

    # # solve system

    # Instead of above we transform to a sparse one
    # Transformation to sparse matrix form
    h_sparse = csr_matrix(H)
    # Solve sparse system
    dx = spsolve(h_sparse, -b)
    dx = dx.squeeze()
    return dx


# (TODO) use jax autodiff instead
@jax.jit
def pose_pose_constraint(x1: SE2, x2: SE2, z: SE2) -> SE2:
    """Compute pose-pose constraint"""
    return z.inverse() @ x1.inverse() @ x2


@jax.jit
def pose_pose_constraint_error(x1: SE2, x2: SE2, z: SE2) -> Array:
    """Compute the error for pose-pose constraint"""
    e = pose_pose_constraint(x1, x2, z)
    e = jnp.linalg.norm(e.as_vec())
    return e


@jax.jit
def pose_landmark_constraint(x: SE2, landmark: Array, z: Array) -> Array:
    """Compute pose-pose constraint"""
    return x.rotation().as_matrix().T @ (landmark - x.translation().reshape(2, 1)) - z


@jax.jit
def pose_landmark_error(x: SE2, landmark: Array, z: Array) -> Array:
    """Compute the error for pose-landmark constraint"""
    e = pose_landmark_constraint(x, landmark, z)
    e = jnp.linalg.norm(e)
    return e


@jax.jit
def linearize_pose_pose_constraint(x1: SE2, x2: SE2, z: SE2) -> (Array, Array, Array):
    """Compute the error and the Jacobian for pose-pose constraint

    Parameters
    ----------
    x1 : SE2
         first robot pose
    x2 : SE2
         second robot pose
    z :  SE2
         measurement

    Returns
    -------
    e  : 3x1
         error of the constraint
    A  : 3x3
         Jacobian wrt x1
    B  : 3x3
         Jacobian wrt x2
    """
    e = pose_pose_constraint(x1, x2, z).as_vec()

    # Jacobians
    # todo : replace by jax.jacobian

    # d e / d x1 : not sure of this one
    z_r = z.rotation().as_matrix()
    x1_r = x1.rotation().as_matrix()
    a_11 = -(z_r.T @ x1_r.T)
    d_r1 = x1.rotation().derivative()
    a_12 = (z_r.T @ d_r1.T @ (x2.translation() - x1.translation())).reshape((2, 1))
    a_21_22 = jnp.array([0, 0, -1])
    a = jnp.vstack([jnp.hstack([a_11, a_12]), a_21_22])

    # d e / d x2 : ref set translation to 0, but why?
    b_11 = (z.inverse() @ x1.inverse()).rotation().as_matrix()
    b_12 = jnp.zeros((2, 1), dtype=np.float32)
    b_21_22 = jnp.array([0, 0, 1])
    b = jnp.vstack([jnp.hstack([b_11, b_12]), b_21_22])
    return e, a, b


@jax.jit
def linearize_pose_landmark_constraint(x: SE2, landmark: Array, z: Array) -> (Array, Array, Array):
    """Compute the error and the Jacobian for pose-landmark constraint

    Parameters
    ----------
    x : SE2
        the robot pose
    landmark : 2x1 vector
        (x,y) of the landmark
    z : 2x1 vector
        (x,y) of the measurement

    Returns
    -------
    e : 2x1 vector
        error for the constraint
    A : 2x3 Jacobian wrt x
    B : 2x2 Jacobian wrt l
    """
    e = x.rotation().as_matrix().T @ (landmark - x.translation().reshape(2, 1)) - z

    xrd = x.rotation().derivative()
    a = jnp.hstack(
        [-x.rotation().as_matrix().T, xrd.T @ (landmark - x.translation().reshape(2, 1))],
    )

    b = x.rotation().as_matrix().T

    return e, a, b


if __name__ == "__main__":
    print("hello world!")
