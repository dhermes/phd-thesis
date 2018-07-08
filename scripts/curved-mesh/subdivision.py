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
import bezier._geometric_intersection
import bezier._helpers
import matplotlib.patches
import matplotlib.path
import matplotlib.pyplot as plt
import numpy as np

import plot_utils


def simple_axis(ax):
    ax.axis("scaled")
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def image1():
    filename = "subdivide_curve.pdf"
    figure, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)
    nodes = np.asfortranarray([[0.0, 1.0, 2.0, 4.0], [0.0, 4.0, 0.0, 3.0]])
    curve = bezier.Curve(nodes, degree=2)
    left, right = curve.subdivide()
    curve.plot(256, ax=ax1, alpha=0.25, color="black")
    left.plot(256, ax=ax1)
    curve.plot(256, ax=ax2)
    curve.plot(256, ax=ax3, alpha=0.25, color="black")
    right.plot(256, ax=ax3)
    ax1.text(
        2.5,
        0.25,
        r"$\left[0, \frac{1}{2}\right]$",
        horizontalalignment="center",
        verticalalignment="center",
    )
    ax2.text(
        2.5,
        0.25,
        r"$\left[0, 1\right]$",
        horizontalalignment="center",
        verticalalignment="center",
    )
    ax3.text(
        2.5,
        0.25,
        r"$\left[\frac{1}{2}, 1\right]$",
        horizontalalignment="center",
        verticalalignment="center",
    )

    for ax in (ax1, ax2, ax3):
        simple_axis(ax)

    path = plot_utils.get_path("curved-mesh", filename)
    figure.savefig(path, bbox_inches="tight")
    print("Saved {}".format(filename))
    plt.close(figure)


def add_patch(curve, ax, color):
    left, right, bottom, top = bezier._helpers.bbox(curve.nodes)
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


def plot_with_bbox(curve, ax, color=None):
    curve.plot(256, color=color, ax=ax)
    if color is None:
        line = ax.lines[-1]
        color = line.get_color()

    add_patch(curve, ax, color)
    return color


def bbox_intersect(curve1, curve2):
    enum_val = bezier._geometric_intersection.bbox_intersect(
        curve1.nodes, curve2.nodes
    )
    return enum_val != 2


def refine_candidates(left, right):
    new_left = []
    for curve in left:
        new_left.extend(curve.subdivide())

    new_right = []
    for curve in right:
        new_right.extend(curve.subdivide())

    keep_left = []
    keep_right = []
    for curve1 in new_left:
        for curve2 in new_right:
            if bbox_intersect(curve1, curve2):
                keep_left.append(curve1)
                if curve2 not in keep_right:
                    keep_right.append(curve2)

    return keep_left, keep_right


def image2():
    filename = "subdivision_process.pdf"
    nodes15 = np.asfortranarray([[0.25, 0.625, 1.0], [0.625, 0.25, 1.0]])
    curve15 = bezier.Curve(nodes15, degree=2)
    nodes25 = np.asfortranarray([[0.0, 0.25, 0.75, 1.0], [0.5, 1.0, 1.5, 0.5]])
    curve25 = bezier.Curve(nodes25, degree=3)

    figure, all_axes = plt.subplots(2, 3, sharex=True, sharey=True)
    ax1, ax2, ax3, ax4, ax5, ax6 = all_axes.flatten()

    color1 = plot_with_bbox(curve15, ax1)
    color2 = plot_with_bbox(curve25, ax1)

    left, right = refine_candidates([curve15], [curve25])
    for curve in left:
        plot_with_bbox(curve, ax2, color=color1)
    for curve in right:
        plot_with_bbox(curve, ax2, color=color2)

    for ax in (ax3, ax4, ax5, ax6):
        left, right = refine_candidates(left, right)
        curve15.plot(256, color=color1, alpha=0.5, ax=ax)
        for curve in left:
            plot_with_bbox(curve, ax, color=color1)
        curve25.plot(256, color=color2, alpha=0.5, ax=ax)
        for curve in right:
            plot_with_bbox(curve, ax, color=color2)

    for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
        simple_axis(ax)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0.4, 1.15)

    figure.set_size_inches(6.4, 4.8)
    figure.subplots_adjust(
        left=0.12, bottom=0.11, right=0.90, top=0.88, wspace=0.05, hspace=-0.55
    )
    path = plot_utils.get_path("curved-mesh", filename)
    figure.savefig(path, bbox_inches="tight")
    print("Saved {}".format(filename))
    plt.close(figure)


def image3():
    filename = "bbox_check.pdf"
    figure, (ax1, ax2, ax3) = plt.subplots(1, 3)

    control_pts1a = np.asfortranarray([[0.0, 0.375, 1.0], [0.0, 0.5, 0.125]])
    curve1a = bezier.Curve(control_pts1a, degree=2)
    control_pts1b = np.asfortranarray(
        [[0.25, -0.125, 0.5], [-0.125, 0.375, 1.0]]
    )
    curve1b = bezier.Curve(control_pts1b, degree=2)
    plot_with_bbox(curve1a, ax1)
    plot_with_bbox(curve1b, ax1)

    control_pts2a = np.asfortranarray([[0.0, 0.75, 1.0], [1.0, 0.75, 0.0]])
    curve2a = bezier.Curve(control_pts2a, degree=2)
    control_pts2b = np.asfortranarray(
        [[0.375, 0.625, 1.375], [1.375, 0.625, 0.375]]
    )
    curve2b = bezier.Curve(control_pts2b, degree=2)
    plot_with_bbox(curve2a, ax2)
    plot_with_bbox(curve2b, ax2)

    control_pts3a = np.asfortranarray([[0.0, 0.25, 1.0], [-0.25, 0.25, -0.75]])
    curve3a = bezier.Curve(control_pts3a, degree=2)
    control_pts3b = np.asfortranarray([[1.0, 1.5, 2.0], [-1.0, -1.5, -1.0]])
    curve3b = bezier.Curve(control_pts3b, degree=2)
    plot_with_bbox(curve3a, ax3)
    plot_with_bbox(curve3b, ax3)

    for ax in (ax1, ax2, ax3):
        simple_axis(ax)

    ax1.set_xlim(-0.2, 1.1)
    ax1.set_ylim(-0.2, 1.1)
    ax1.set_title("MAYBE")
    ax2.set_xlim(-0.1, 1.5)
    ax2.set_ylim(-0.1, 1.5)
    ax2.set_title("MAYBE")
    ax3.set_xlim(-0.1, 2.1)
    ax3.set_ylim(-1.7, 0.5)
    ax3.set_title("NO")

    path = plot_utils.get_path("curved-mesh", filename)
    figure.savefig(path, bbox_inches="tight")
    print("Saved {}".format(filename))
    plt.close(figure)


def image4():
    filename = "subdivision_linearized.pdf"
    figure, all_axes = plt.subplots(2, 7)
    all_axes = all_axes.flatten()

    nodes15 = np.asfortranarray([[0.25, 0.625, 1.0], [0.625, 0.25, 1.0]])
    curve15 = bezier.Curve(nodes15, degree=2)
    nodes25 = np.asfortranarray([[0.0, 0.25, 0.75, 1.0], [0.5, 1.0, 1.5, 0.5]])
    curve25 = bezier.Curve(nodes25, degree=3)

    curve15.plot(256, ax=all_axes[0])
    color1 = all_axes[0].lines[-1].get_color()
    curve25.plot(256, ax=all_axes[0])
    color2 = all_axes[0].lines[-1].get_color()

    choices1 = [1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1]
    choices2 = [1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1]
    first = curve15
    second = curve25
    for i in range(13):
        ax = all_axes[i + 1]
        index1 = choices1[i]
        index2 = choices2[i]
        first = first.subdivide()[index1]
        second = second.subdivide()[index2]
        first.plot(256, ax=ax)
        second.plot(256, ax=ax)
        # After splitting, put the bounding box on the previous axis.
        prev_ax = all_axes[i]
        add_patch(first, prev_ax, color1)
        add_patch(second, prev_ax, color2)

    for ax in all_axes:
        ax.axis("equal")
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    figure.set_size_inches(15.29, 4.8)
    figure.subplots_adjust(
        left=0.01, bottom=0.02, right=0.99, top=0.98, wspace=0.06, hspace=0.06
    )
    path = plot_utils.get_path("curved-mesh", filename)
    figure.savefig(path, bbox_inches="tight")
    print("Saved {}".format(filename))
    plt.close(figure)


def main():
    image1()
    image2()
    image3()
    image4()


if __name__ == "__main__":
    plot_utils.set_styles()
    main()
