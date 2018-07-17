# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import bezier
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.qhull
import shapely.geometry

import plot_utils


ALPHA = 0.375
NODES_X = np.array(
    [
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
    ]
)
NODES_Y = np.array(
    [
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -0.5,
        -0.5,
        -0.5,
        -0.5,
        -0.5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
)
TRIANGLES = np.array(
    [
        [0, 1, 6],
        [0, 6, 5],
        [1, 2, 7],
        [1, 7, 6],
        [2, 3, 8],
        [2, 8, 7],
        [3, 4, 9],
        [3, 9, 8],
        [5, 6, 11],
        [5, 11, 10],
        [6, 7, 12],
        [6, 12, 11],
        [7, 8, 13],
        [7, 13, 12],
        [8, 9, 14],
        [8, 14, 13],
        [10, 11, 16],
        [10, 16, 15],
        [11, 12, 17],
        [11, 17, 16],
        [12, 13, 18],
        [12, 18, 17],
        [13, 14, 19],
        [13, 19, 18],
        [15, 16, 21],
        [15, 21, 20],
        [16, 17, 22],
        [16, 22, 21],
        [17, 18, 23],
        [17, 23, 22],
        [18, 19, 24],
        [18, 24, 23],
    ],
    dtype=np.int32,
)
# NOTE: These must be in order along the edge.
BOUNDARY_INDICES = (0, 1, 2, 3, 4, 9, 14, 19, 24, 23, 22, 21, 20, 15, 10, 5)


def point_on_characteristic(xv, yv, t):
    yt = yv + t
    xt = xv + (yt * yt * yt - yv * yv * yv) / 3
    return xt, yt


def get_title(t):
    if t == int(t):
        return "$t = {:d}.0$".format(int(t))
    else:
        return "$t = {:g}$".format(t)


def plot_exterior(
    internal_x, internal_y, external_x, external_y, ax, custom_tris=None
):
    N1 = len(internal_x)
    N2 = len(external_x)
    nodes = np.empty((N1 + N2, 2))
    nodes[:N1, 0] = internal_x
    nodes[:N1, 1] = internal_y
    nodes[N1:, 0] = external_x
    nodes[N1:, 1] = external_y
    tessellation = scipy.spatial.qhull.Delaunay(nodes)
    # Remove any triangles that cross the boundary.
    to_keep = []
    polygon1 = shapely.geometry.Polygon(nodes[:N1, :])
    for i, tri in enumerate(tessellation.simplices):
        polygon2 = shapely.geometry.Polygon(nodes[tri, :])
        intersection = polygon1.intersection(polygon2)
        if intersection.area == 0.0:
            to_keep.append(i)

    triangles = tessellation.simplices[to_keep, :]
    if custom_tris is not None:
        triangles = np.vstack([triangles, custom_tris])
    ax.triplot(
        nodes[:, 0],
        nodes[:, 1],
        triangles,
        color=plot_utils.GREEN,
        alpha=ALPHA,
    )


def plot_distorted(filename, exterior=False):
    external_x = np.array([-1.25, 1.125, 3.5, 3.5, 3.5, 1.125, -1.25, -1.25])
    external_y = np.array([-1.25, -1.25, -1.25, 0.5, 2.25, 2.25, 2.25, 0.5])

    figure, all_axes = plt.subplots(3, 3)
    all_axes = all_axes.flatten()
    for index in range(9):
        ax = all_axes[index]
        t = 0.125 * index
        xt, yt = point_on_characteristic(NODES_X, NODES_Y, t)
        ax.triplot(xt, yt, TRIANGLES, color=plot_utils.BLUE)

        if exterior:
            custom_tris = None
            curr_ex = external_x
            curr_ey = external_y
            if index in (0, 1):
                curr_ex = np.append(curr_ex, -1.25)
                curr_ey = np.append(curr_ey, -0.375)
            elif index in (7, 8):
                curr_ex = np.append(curr_ex, 2.3125)
                curr_ey = np.append(curr_ey, 2.25)
            if index == 8:
                custom_tris = np.array([[8, 20, 9]], dtype=np.int32)
            plot_exterior(
                xt[BOUNDARY_INDICES,],
                yt[BOUNDARY_INDICES,],
                curr_ex,
                curr_ey,
                ax,
                custom_tris=custom_tris,
            )
        title = get_title(t)
        ax.set_title(title)
        # Set the axis.
        ax.axis("scaled")
        ax.set_xlim(-1.35, 3.6)
        ax.set_ylim(-1.35, 2.35)

    all_axes = all_axes.reshape(3, 3)
    for ax in all_axes[:, 0]:
        ax.set_yticks([-1.0, 0.5, 2.0])
    for col in (1, 2):
        for ax in all_axes[:, col]:
            ax.set_yticklabels([])
    for row in (0, 1):
        for ax in all_axes[row, :]:
            ax.set_xticklabels([])
    for ax in all_axes[2, :]:
        ax.set_xticks([-1.0, 1.0, 3.0])
        ax.set_xticklabels(["$-1.0$", "$1.0$", "$3.0$"])

    figure.set_size_inches(10.75, 8.61)
    figure.subplots_adjust(
        left=0.04, bottom=0.02, right=0.98, top=0.98, wspace=0.04, hspace=0.1
    )
    path = plot_utils.get_path("curved-mesh", filename)
    figure.savefig(path, bbox_inches="tight")
    print("Saved {}".format(filename))
    plt.close(figure)


def distort_cubic_tri():
    node1 = np.array([-0.75, 0.0])
    node2 = np.array([2.25, -1.5])
    node3 = np.array([1.5, 1.5])
    control_points = np.array(
        [
            node1,
            0.5 * (node1 + node2),
            node2,
            0.5 * (node1 + node3),
            0.5 * (node2 + node3),
            node3,
        ]
    )

    figure, all_axes = plt.subplots(2, 3)
    min_y = -1.65
    max_y = 2.8
    control_x = control_points[:, 0]
    control_y = control_points[:, 1]
    bezier_nodes = np.empty((2, len(control_x)), order="F")

    # First add characteristic curves to the top row of axes.
    for i, xv in enumerate(control_x):
        yv = control_y[i]
        min_t = min_y - yv
        max_t = max_y - yv
        t_vals = np.linspace(min_t, max_t, 100)
        to_plot = point_on_characteristic(xv, yv, t_vals)
        for ax in all_axes[0, :]:
            ax.plot(
                to_plot[0], to_plot[1], color=plot_utils.GREEN, alpha=ALPHA
            )

    for index, ax_top in enumerate(all_axes[0, :]):
        t = 0.5 * index
        xt, yt = point_on_characteristic(control_x, control_y, t)

        corner_x = xt[(0, 2, 5, 0),]
        corner_y = yt[(0, 2, 5, 0),]
        ax_top.plot(corner_x, corner_y)

        title = get_title(t)
        ax_top.set_title(title)

        # Now plot the curved element in the "below" axis".
        ax_below = all_axes[1, index]
        # NOTE: This assumes quadratic nodes.
        bezier_nodes[:, 0] = xt[0], yt[0]
        bezier_nodes[:, 1] = (
            2.0 * xt[1] - 0.5 * xt[0] - 0.5 * xt[2],
            2.0 * yt[1] - 0.5 * yt[0] - 0.5 * yt[2],
        )
        bezier_nodes[:, 2] = xt[2], yt[2]
        bezier_nodes[:, 3] = (
            2.0 * xt[3] - 0.5 * xt[0] - 0.5 * xt[5],
            2.0 * yt[3] - 0.5 * yt[0] - 0.5 * yt[5],
        )
        bezier_nodes[:, 4] = (
            2.0 * xt[4] - 0.5 * xt[2] - 0.5 * xt[5],
            2.0 * yt[4] - 0.5 * yt[2] - 0.5 * yt[5],
        )
        bezier_nodes[:, 5] = xt[5], yt[5]
        surface = bezier.Surface.from_nodes(bezier_nodes)
        surface.plot(256, ax=ax_below)

        # Add "nodes" to both plots.
        for ax in (ax_top, ax_below):
            ax.plot(
                xt,
                yt,
                color="black",
                marker="o",
                linestyle="none",
                markersize=6,
            )
        # Add shadow "nodes" to top row for "next" plots.
        for next_index in range(index + 1, 3):
            ax = all_axes[0, next_index]
            ax.plot(
                xt,
                yt,
                color="black",
                alpha=0.5 - 0.25 * (next_index - index - 1),
                marker="o",
                linestyle="none",
                markersize=6,
            )

    for ax in all_axes.flatten():
        ax.axis("scaled")
        ax.set_xlim(-1.0, 5.9)
        ax.set_ylim(min_y, max_y)
    for ax in all_axes[:, 1:].flatten():
        ax.set_yticklabels([])
    for ax in all_axes[0, :].flatten():
        ax.set_xticklabels([])
    for ax in all_axes[:, 0]:
        ax.set_yticks([-1.5, -0.5, 0.5, 1.5, 2.5])
        ax.set_yticklabels(["$-1.5$", "$-0.5$", "$0.5$", "$1.5$", "$2.5$"])
    for ax in all_axes[1, :]:
        ax.set_xticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        ax.set_xticklabels(
            ["$0.0$", "$1.0$", "$2.0$", "$3.0$", "$4.0$", "$5.0$"]
        )

    figure.set_size_inches(13.0, 6.37)
    figure.subplots_adjust(
        left=0.03, bottom=0.05, right=1.0, top=0.95, wspace=0.03, hspace=-0.03
    )
    filename = "element_distortion.pdf"
    path = plot_utils.get_path("curved-mesh", filename)
    figure.savefig(path, bbox_inches="tight")
    print("Saved {}".format(filename))
    plt.close(figure)


def remesh():
    figure, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)

    # Update to 1-second ahead in time.
    xt, yt = point_on_characteristic(NODES_X, NODES_Y, 1.0)
    nodes = np.empty((len(NODES_X), 2))
    nodes[:, 0] = xt
    nodes[:, 1] = yt
    for ax in (ax1, ax2):
        ax.triplot(nodes[:, 0], nodes[:, 1], TRIANGLES, color=plot_utils.BLUE)

    # Do a Delaunay triangulation and discard exterior triangles.
    tessellation = scipy.spatial.qhull.Delaunay(nodes)
    polygon1 = shapely.geometry.Polygon(nodes[BOUNDARY_INDICES, :])
    to_keep = []
    for i, tri in enumerate(tessellation.simplices):
        polygon2 = shapely.geometry.Polygon(nodes[tri, :])
        intersection = polygon1.intersection(polygon2)
        int_area = intersection.area
        if int_area == polygon2.area:
            to_keep.append(i)
        elif int_area != 0:
            raise NotImplementedError

    triangles_new = tessellation.simplices[to_keep, :]
    for ax in (ax2, ax3):
        ax.triplot(
            nodes[:, 0], nodes[:, 1], triangles_new, color=plot_utils.GREEN
        )

    ax1.set_yticks([-0.5, 1.0, 2.5])
    for ax in (ax1, ax2, ax3):
        ax.axis("scaled")
        ax.set_xlim(-1.3, 3.6)
        ax.set_ylim(-0.75, 2.75)
        ax.set_xticks([-1.0, 1.0, 3.0])
        ax.set_xticklabels(["$-1.0$", "$1.0$", "$3.0$"])
    ax1.set_title("Before Remeshing", fontsize=24)
    ax3.set_title("After Remeshing", fontsize=24)

    figure.set_size_inches(16.3, 3.85)
    figure.subplots_adjust(
        left=0.03, bottom=0.0, right=0.99, top=1.0, wspace=0.04, hspace=0.2
    )
    filename = "distortion_remesh.pdf"
    path = plot_utils.get_path("curved-mesh", filename)
    figure.savefig(path, bbox_inches="tight")
    print("Saved {}".format(filename))
    plt.close(figure)


def main():
    plot_distorted("mesh_distortion.pdf")
    plot_distorted("mesh_distortion_ext.pdf", exterior=True)
    distort_cubic_tri()
    remesh()


if __name__ == "__main__":
    plot_utils.set_styles()
    main()
