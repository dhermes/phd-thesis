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

import plot_utils


LABEL_SIZE = 14.0
FONT_SIZE = 20.0


def image1():
    figure, (ax1, ax2) = plt.subplots(1, 2)
    nodes1 = np.asfortranarray([[0.0, 3.0, 7.0], [5.0, 0.0, 8.0]])
    triangle1 = bezier.Surface(nodes1, degree=1)
    triangle1.plot(256, ax=ax1)

    nodes2 = np.asfortranarray(
        [[0.0, 1.0, 2.0, 2.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0, 2.0, 2.0]]
    )
    triangle2 = bezier.Surface(nodes2, degree=2)
    triangle2.plot(256, ax=ax2)

    params = np.asfortranarray([[0.125, 0.125], [0.125, 0.75]])
    points1 = triangle1.evaluate_cartesian_multi(params)
    ax1.plot(points1[0, :], points1[1, :], marker="o", color="black")
    points2 = triangle2.evaluate_cartesian_multi(params)
    ax2.plot(points2[0, :], points2[1, :], marker="o", color="black")

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=LABEL_SIZE, which="both")
        ax.axis("equal")

    ax1.set_title("Convex", fontsize=FONT_SIZE)
    ax2.set_title("Not (Necessarily) Convex", fontsize=FONT_SIZE)

    figure.set_size_inches(8.74, 4.8)
    figure.subplots_adjust(
        left=0.05, bottom=0.06, right=0.99, top=0.93, wspace=0.12, hspace=0.2
    )
    filename = "not_convex.pdf"
    path = plot_utils.get_path("slides", filename)
    figure.savefig(path)
    print("Saved {}".format(filename))
    plt.close(figure)


def image2():
    figure, (ax1, ax2) = plt.subplots(1, 2)

    nodes1a = np.asfortranarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    nodes2a = np.asfortranarray([[-0.125, 1.0, 0.125], [-0.0625, 0.5, 0.375]])
    nodes1b = np.asfortranarray(
        [[0.0, 0.375, 1.0, 0.25, 0.75, 0.5], [0.0, 0.375, 0.0, 0.5, 0.5, 1.0]]
    )
    nodes2b = np.asfortranarray(
        [
            [1.0, 0.625, 0.0, 0.75, 0.25, 0.5],
            [0.375, -0.125, 0.375, -0.1875, -0.1875, -0.75],
        ]
    )
    info = ((nodes1a, nodes2a, ax1), (nodes1b, nodes2b, ax2))

    for nodes1, nodes2, ax in info:
        triangle1 = bezier.Surface.from_nodes(nodes1)
        triangle2 = bezier.Surface.from_nodes(nodes2)
        intersections = triangle1.intersect(triangle2)

        triangle1.plot(256, ax=ax, color=plot_utils.BLUE)
        triangle2.plot(256, ax=ax, color=plot_utils.GREEN)
        for intersection in intersections:
            intersection.plot(256, ax=ax, color=plot_utils.RED)

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=LABEL_SIZE, which="both")
        ax.axis("equal")

    ax1.set_title("Convex Intersection", fontsize=FONT_SIZE)
    ax2.set_title("Multiple Intersections", fontsize=FONT_SIZE)

    figure.set_size_inches(8.74, 4.8)
    figure.subplots_adjust(
        left=0.06, bottom=0.06, right=0.99, top=0.93, wspace=0.18, hspace=0.2
    )
    filename = "split_intersection.pdf"
    path = plot_utils.get_path("slides", filename)
    figure.savefig(path)
    print("Saved {}".format(filename))
    plt.close(figure)


def main():
    image1()
    image2()


if __name__ == "__main__":
    plot_utils.set_styles()
    main()
