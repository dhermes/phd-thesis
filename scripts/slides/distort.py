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


LABEL_SIZE = 14.0
FONT_SIZE = 20.0
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


def point_on_characteristic(xv, yv, t):
    yt = yv + t
    xt = xv + (yt * yt * yt - yv * yv * yv) / 3
    return xt, yt


def get_title(t):
    if t == int(t):
        return "$t = {:d}.0$".format(int(t))
    else:
        return "$t = {:g}$".format(t)


def plot_distorted():
    figure, all_axes = plt.subplots(2, 3, sharex=True, sharey=True)
    all_axes = all_axes.flatten()
    for index in range(6):
        ax = all_axes[index]
        t = index / 5.0
        xt, yt = point_on_characteristic(NODES_X, NODES_Y, t)
        ax.triplot(xt, yt, TRIANGLES, color=plot_utils.BLUE)

        title = get_title(t)
        ax.set_title(title, fontsize=FONT_SIZE)
        # Set the axis.
        ax.axis("scaled")
        ax.set_xlim(-1.35, 3.6)
        ax.set_ylim(-1.35, 2.35)

    for ax in all_axes:
        ax.tick_params(labelsize=LABEL_SIZE, which="both")

    all_axes[0].set_yticks([-1.0, 0.5, 2.0])
    all_axes[0].set_xticks([-1.0, 1.0, 3.0])
    all_axes[0].set_xticklabels(["$-1.0$", "$1.0$", "$3.0$"])

    figure.set_size_inches(8.74, 4.8)
    figure.subplots_adjust(
        left=0.06, bottom=0.06, right=0.99, top=0.93, wspace=0.0, hspace=0.2
    )
    filename = "mesh_distortion.pdf"
    path = plot_utils.get_path("slides", filename)
    figure.savefig(path)
    print("Saved {}".format(filename))
    plt.close(figure)


def main():
    plot_distorted()


if __name__ == "__main__":
    plot_utils.set_styles()
    main()
