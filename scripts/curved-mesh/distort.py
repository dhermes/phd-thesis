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

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.qhull
import shapely.geometry

import plot_utils


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
        alpha=0.375,
    )


def plot_distorted(filename, exterior=False):
    nodes_x = np.array(
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
    nodes_y = np.array(
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
    external_x = np.array([-1.25, 1.125, 3.5, 3.5, 3.5, 1.125, -1.25, -1.25])
    external_y = np.array([-1.25, -1.25, -1.25, 0.5, 2.25, 2.25, 2.25, 0.5])
    triangles = np.array(
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
    boundary_indices = (
        0,
        1,
        2,
        3,
        4,
        5,
        9,
        14,
        19,
        24,
        23,
        22,
        21,
        20,
        15,
        10,
        5,
    )

    figure, all_axes = plt.subplots(3, 3)
    all_axes = all_axes.flatten()
    for index in range(9):
        ax = all_axes[index]
        t = 0.125 * index
        yt = nodes_y + t
        xt = nodes_x + (yt * yt * yt - nodes_y * nodes_y * nodes_y) / 3
        ax.triplot(xt, yt, triangles, color=plot_utils.BLUE)

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
                xt[boundary_indices,],
                yt[boundary_indices,],
                curr_ex,
                curr_ey,
                ax,
                custom_tris=custom_tris,
            )
        if t == int(t):
            title = "$t = {:d}.0$".format(int(t))
        else:
            title = "$t = {:g}$".format(t)
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


def main():
    plot_distorted("mesh_distortion.pdf")
    plot_distorted("mesh_distortion_ext.pdf", exterior=True)


if __name__ == "__main__":
    plot_utils.set_styles()
    main()
