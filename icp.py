import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from jax import numpy as jnp


# icp_known_corresp: performs icp given that the input datasets
# are aligned so that Line1(:, QInd(k)) corresponds to Line2(:, PInd(k))
def icp_known_corresp(line1: Array, line2: Array, q_index: Array, p_index: Array):
    q = line1[:, q_index]
    p = line2[:, p_index]

    mean_q = compute_mean(q)
    mean_p = compute_mean(p)

    w = compute_w(q, p, mean_q, mean_p)

    [r, t] = compute_r_t(w, mean_q, mean_p)

    # Compute the new positions of the points after
    # applying found rotation and translation to them
    new_line = r @ p + t

    error = compute_error(q, new_line)

    return new_line, error


# compute_W: compute matrix W to use in SVD
def compute_w(q: Array, p: Array, mean_q: Array, mean_p: Array) -> Array:
    return (q - mean_q) @ (p - mean_p).T


# compute_R_t: compute rotation matrix and translation vector
# based on the SVD as presented in the lecture
def compute_r_t(w: Array, mean_q: Array, mean_p: Array) -> (Array, Array):
    u, s, vt = jnp.linalg.svd(w)
    r = u @ vt
    t = mean_q - r @ mean_p
    return r, t


# compute_mean: compute mean value for a [M x N] matrix
def compute_mean(m: Array) -> Array:
    return np.array([m.mean(axis=1)]).T


# compute_error: compute the icp error
def compute_error(q: Array, optimized_points: Array) -> Array:
    return jnp.sum(jnp.linalg.norm(q - optimized_points) ** 2)  # l2 loss


# simply show the two lines
def show_figure(line1, line2):
    plt.figure()
    plt.scatter(line1[0], line1[1], marker="o", s=2, label="Line 1")
    plt.scatter(line2[0], line2[1], s=1, label="Line 2")

    plt.xlim([-8, 8])
    plt.ylim([-8, 8])
    plt.legend()

    plt.show()


# initialize figure
def init_figure():
    fig = plt.gcf()
    fig.show()
    fig.canvas.draw()

    line1_fig = plt.scatter([], [], marker="o", s=2, label="Line 1")
    line2_fig = plt.scatter([], [], marker="o", s=1, label="Line 2")
    plt.xlim([-8, 8])
    plt.ylim([-8, 8])
    plt.legend()

    return fig, line1_fig, line2_fig


# update_figure: show the current state of the lines
def update_figure(fig, line1_fig, line2_fig, line1, line2, hold=False):
    line1_fig.set_offsets(line1.T)
    line2_fig.set_offsets(line2.T)
    if hold:
        plt.show()
    else:
        fig.canvas.flush_events()
        fig.canvas.draw()
        plt.pause(0.5)


if __name__ == "__main__":
    data = np.load("dataset/new_slam_course/icp_data.npz")
    line1 = data["LineGroundTruth"]
    line2 = data["LineMovedCorresp"]

    # Show the initial positions of the lines
    show_figure(line1, line2)

    # We assume that the there are 1 to 1 correspondences for this data
    q_index = np.arange(len(line1[0]))
    p_index = np.arange(len(line2[0]))

    # Perform icp given the correspondences
    line2_final, error = icp_known_corresp(line1, line2, q_index, p_index)

    # Show the adjusted positions of the lines
    show_figure(line1, line2_final)

    # print the error
    print(f"Error value is: {error}")

    from sklearn.neighbors import NearestNeighbors

    max_iter = 100
    epsilon = 0.4
    error = np.inf

    # KNN for Line1
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(line1.T)

    for i in range(max_iter):
        # point with index QInd(1, k) from Line1 corresponds to
        # point with index PInd(1, k) from Line2
        # Find correspondences using KNN
        q_ind = np.array([i[0] for i in neigh.kneighbors(line2.T, return_distance=False)])
        p_ind = np.arange(line2.shape[1])

        # update Line2 and error
        # Now that you know the correspondences, use your implementation
        # of icp with known correspondences and perform an update
        error_old = error
        line2, error = icp_known_corresp(line1, line2, q_ind, p_ind)

        if i % 1 == 0:
            show_figure(line1, line2)
            print(f"Error value on {i} iteration is: {error}")

        if error < epsilon:
            break
