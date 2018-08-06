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

"""This is mostly copied from ``scripts/solution-transfer/distort.py``."""

import bezier
import matplotlib.pyplot as plt
import numpy as np

import plot_utils


ALPHA = 0.375


def point_on_characteristic(xv, yv, t):
    yt = yv + t
    xt = xv + (yt * yt * yt - yv * yv * yv) / 3
    return xt, yt


def get_title(t):
    if t == int(t):
        return "$t = {:d}.0$".format(int(t))
    else:
        return "$t = {:g}$".format(t)


def distort_cubic_tri(num_columns):
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

    figure, all_axes = plt.subplots(2, 3, sharex=True, sharey=True)
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
        for index, ax in enumerate(all_axes[0, :]):
            if index == num_columns:
                break
            ax.plot(
                to_plot[0], to_plot[1], color=plot_utils.GREEN, alpha=ALPHA
            )

    for index, ax_top in enumerate(all_axes[0, :]):
        if index == num_columns:
            break
        t = 0.5 * index
        xt, yt = point_on_characteristic(control_x, control_y, t)

        corner_x = xt[(0, 2, 5, 0),]
        corner_y = yt[(0, 2, 5, 0),]
        ax_top.plot(corner_x, corner_y)

        title = get_title(t)
        ax_top.set_title(title, fontsize=plot_utils.TEXT_SIZE)

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
                markersize=4,
            )
        # Add shadow "nodes" to top row for "next" plots.
        for next_index in range(index + 1, 3):
            if next_index == num_columns:
                break
            ax = all_axes[0, next_index]
            ax.plot(
                xt,
                yt,
                color="black",
                alpha=0.5 - 0.25 * (next_index - index - 1),
                marker="o",
                linestyle="none",
                markersize=4,
            )

    for ax in all_axes.flatten():
        ax.axis("scaled")
    # One axis, all axes (since sharex/sharey).
    ax1 = all_axes[0, 0]
    ax1.set_xlim(-1.0, 5.9)
    ax1.set_ylim(min_y, max_y)
    ax1.set_xticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    ax1.set_xticklabels(["$0.0$", "$1.0$", "$2.0$", "$3.0$", "$4.0$", "$5.0$"])
    ax1.set_yticks([-1.5, -0.5, 0.5, 1.5, 2.5])
    ax1.set_yticklabels(["$-1.5$", "$-0.5$", "$0.5$", "$1.5$", "$2.5$"])
    ax1.yaxis.set_tick_params(labelsize=plot_utils.TICK_SIZE)
    all_axes[1, 0].yaxis.set_tick_params(labelsize=plot_utils.TICK_SIZE)
    for ax in all_axes[1, :]:
        ax.xaxis.set_tick_params(labelsize=plot_utils.TICK_SIZE)

    for ax in all_axes[:, num_columns:].flatten():
        ax.axis("off")

    figure.set_size_inches(6.0, 2.9)
    figure.subplots_adjust(
        left=0.07, bottom=0.05, right=0.99, top=0.97, wspace=0.04, hspace=-0.1
    )
    filename = "element_distortion{}.pdf".format(num_columns)
    path = plot_utils.get_path("slides", filename)
    figure.savefig(path)
    print("Saved {}".format(filename))
    plt.close(figure)


def main():
    distort_cubic_tri(1)
    distort_cubic_tri(2)
    distort_cubic_tri(3)


if __name__ == "__main__":
    plot_utils.set_styles()
    main()
