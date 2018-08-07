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

"""This is mostly copied from elsewhere.

In particular: ``scripts/solution-transfer/polygon_vs_curved.py``.
"""

import fractions

import bezier
import matplotlib.patches
import matplotlib.path
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry

import plot_utils


F = fractions.Fraction


def shoelace_for_area(nodes):
    _, num_nodes = nodes.shape
    if num_nodes == 3:
        shoelace = ((2, 0, 1), (1, 0, 2), (2, 1, 2))
        scale_factor = 6.0
    else:
        raise NotImplementedError

    result = 0.0
    for multiplier, index1, index2 in shoelace:
        result += multiplier * (
            nodes[0, index1] * nodes[1, index2]
            - nodes[1, index1] * nodes[0, index2]
        )

    return result / scale_factor


def compute_area(*edges):
    area = 0.0
    for edge in edges:
        area += shoelace_for_area(edge.nodes)
    return area


def edges_to_polygon(edge1, edge2, edge3, s_vals):
    # NOTE: This assumes ``s_vals[0] == 0.0`` and ``s_vals[-1] < 1.0``.
    N = len(s_vals)
    polygon_nodes = np.empty((2, 3 * N), order="F")
    polygon_nodes[:, :N] = edge1.evaluate_multi(s_vals)
    polygon_nodes[:, N : 2 * N] = edge2.evaluate_multi(s_vals)
    polygon_nodes[:, 2 * N :] = edge3.evaluate_multi(s_vals)
    return polygon_nodes


def bezier_triangle_area(slide_num):
    nodes = np.asfortranarray(
        [
            [0.0, 0.5, 1.0, 0.125, 0.375, 0.25],
            [0.0, 0.0, 0.25, 0.5, 0.375, 1.0],
        ]
    )
    surface = bezier.Surface.from_nodes(nodes)
    exact_area = F(37, 96)
    edge1, edge2, edge3 = surface.edges

    figure, all_axes = plt.subplots(2, 3)
    all_axes = all_axes.flatten()
    surface.plot(256, ax=all_axes[0])
    all_axes[0].set_title("Curved", fontsize=plot_utils.TEXT_SIZE)

    error_vals = []
    for n in range(1, 20 + 1):
        N = 2 ** n
        s_vals = np.linspace(0.0, 1.0, N + 1)[:-1]
        polygon_nodes = edges_to_polygon(edge1, edge2, edge3, s_vals)
        polygon = shapely.geometry.Polygon(polygon_nodes.T)
        # Compute the relative error.
        poly_area = F(polygon.area)
        rel_error = abs(poly_area - exact_area) / exact_area
        error_vals.append((N, float(rel_error)))

        if n in (1, 2, 3, 4):
            ax = all_axes[n]
            # Wrap-around the first node so the polygon is closed.
            polygon_nodes = np.hstack([polygon_nodes, polygon_nodes[:, :1]])
            patch = matplotlib.patches.PathPatch(
                matplotlib.path.Path(polygon_nodes.T), alpha=0.625
            )
            ax.add_patch(patch)
            ax.plot(
                polygon_nodes[0, :],
                polygon_nodes[1, :],
                marker="o",
                markersize=4,
            )
            ax.set_title("$N = {:d}$".format(N), fontsize=plot_utils.TEXT_SIZE)

    for ax in all_axes[:5]:
        ax.axis("equal")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
    for ax in all_axes[:3]:
        ax.set_xticklabels([])
    for ax in all_axes[(1, 2, 4),]:
        ax.set_yticklabels([])

    error_vals = np.array(error_vals)
    ax = all_axes[5]
    line1, = ax.loglog(
        error_vals[:, 0],
        error_vals[:, 1],
        basex=2,
        color=plot_utils.BLUE,
        marker="o",
        markersize=4,
        label="Polygonal",
    )
    if slide_num < 5:
        line1.set_visible(False)
        ax.loglog(
            error_vals[:slide_num, 0],
            error_vals[:slide_num, 1],
            basex=2,
            color=plot_utils.BLUE,
            marker="o",
            markersize=4,
            label="Polygonal",
        )

    surface_area = F(compute_area(edge1, edge2, edge3))
    curved_rel_error = float(abs(exact_area - surface_area) / exact_area)
    line2, = ax.loglog(
        [error_vals[0, 0], error_vals[-1, 0]],
        [curved_rel_error, curved_rel_error],
        basex=2,
        color="black",
        linestyle="dashed",
        label="Curved",
    )
    if slide_num != 6:
        line2.set_visible(False)

    ax.set_title("Area Estimates", fontsize=plot_utils.TEXT_SIZE)
    ax.set_xlabel(
        "Line Segments per Side ($N$)", fontsize=plot_utils.TEXT_SIZE
    )
    ax.set_ylabel("Relative Error", fontsize=plot_utils.TEXT_SIZE)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    all_axes[0].yaxis.set_tick_params(labelsize=plot_utils.TICK_SIZE)
    all_axes[3].yaxis.set_tick_params(labelsize=plot_utils.TICK_SIZE)
    all_axes[5].yaxis.set_tick_params(labelsize=plot_utils.TICK_SIZE)
    all_axes[3].xaxis.set_tick_params(labelsize=plot_utils.TICK_SIZE)
    all_axes[4].xaxis.set_tick_params(labelsize=plot_utils.TICK_SIZE)
    all_axes[5].xaxis.set_tick_params(labelsize=plot_utils.TICK_SIZE)

    all_axes[3].set_xticks([0.1, 0.5, 0.9])
    all_axes[4].set_xticks([0.1, 0.5, 0.9])
    all_axes[5].set_xticks(
        [2.0 ** 4, 2.0 ** 8, 2.0 ** 12, 2.0 ** 16, 2.0 ** 20]
    )

    if slide_num < 6:
        all_axes[0].set_visible(False)
    if slide_num < 4:
        all_axes[4].set_visible(False)
    if slide_num < 3:
        all_axes[3].set_visible(False)
    if slide_num < 2:
        all_axes[2].set_visible(False)

    figure.set_size_inches(6.0, 4.2)
    figure.subplots_adjust(
        left=0.06, bottom=0.11, right=0.9, top=0.95, wspace=0.03, hspace=0.15
    )
    filename = "polygon_vs_curved{}.pdf".format(slide_num)
    path = plot_utils.get_path("slides", filename)
    figure.savefig(path)
    print("Saved {}".format(filename))
    plt.close(figure)


def main():
    bezier_triangle_area(1)
    bezier_triangle_area(2)
    bezier_triangle_area(3)
    bezier_triangle_area(4)
    bezier_triangle_area(5)
    bezier_triangle_area(6)


if __name__ == "__main__":
    plot_utils.set_styles()
    main()
