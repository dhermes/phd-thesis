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
import bezier._helpers
import matplotlib.patches
import matplotlib.path
import matplotlib.pyplot as plt
import numpy as np

import plot_utils


def add_patch(triangle, ax, color):
    # NOTE: This is mostly copy-pasta from ``subdivision.py``.
    left, right, bottom, top = bezier._helpers.bbox(triangle.nodes)
    polygon = np.array(
        [
            [left, bottom],
            [right, bottom],
            [right, top],
            [left, top],
            [left, bottom],
        ]
    )
    patch = matplotlib.patches.PathPatch(
        matplotlib.path.Path(polygon), facecolor=color, alpha=0.625
    )
    ax.add_patch(patch)


def image1():
    nodes = np.asfortranarray(
        [[0.0, 1.5, 3.0, 0.75, 2.25, 0.0], [0.0, -0.5, 0.0, 1.0, 1.5, 2.0]]
    )
    triangle = bezier.Surface(nodes, degree=2, _copy=False)
    point = triangle.evaluate_cartesian(0.25, 0.125)
    xv, yv = point.flatten()
    sub_triangles = triangle.subdivide()
    edges = triangle.edges

    figure, all_axes = plt.subplots(2, 2, sharex=True, sharey=True)
    all_axes = all_axes.flatten()
    for ax, sub_triangle in zip(all_axes, sub_triangles):
        # Add the bounding box for the sub triangle.
        add_patch(sub_triangle, ax, plot_utils.GREEN)
        # Add the triangle boundary to each subplot.
        for edge in edges:
            edge.plot(256, ax=ax, color=plot_utils.BLUE)
        # Add the sub triangle.
        sub_triangle.plot(256, ax=ax, color=plot_utils.BLUE)
        # Add the point to be found.
        ax.plot(
            [xv],
            [yv],
            color="black",
            marker="o",
            markersize=3,
            linestyle="none",
        )

    for ax in all_axes:
        ax.set_aspect("equal")
    # One axis sets all axis
    all_axes[0].set_xticklabels([])
    all_axes[0].set_yticklabels([])

    figure.set_size_inches(4.0, 3.0)
    figure.subplots_adjust(
        left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.03, hspace=0.04
    )
    filename = "locate_in_triangle.pdf"
    path = plot_utils.get_path("data-transfer", filename)
    figure.savefig(path)
    print("Saved {}".format(filename))
    plt.close(figure)


def main():
    image1()


if __name__ == "__main__":
    plot_utils.set_styles()
    main()
