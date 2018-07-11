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

import plot_utils


def main():
    filename = "mesh_deformation.pdf"

    # NOTE: This was generated via
    #           random_mesh(5, np.random.RandomState(seed=7230931))
    #       where ``random_mesh`` comes from the curved mesh project.
    nodes_x = np.array(
        [
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -0.6,
            -0.19999999999999996,
            0.20000000000000018,
            0.6000000000000001,
            -0.6,
            -0.19999999999999996,
            0.20000000000000018,
            0.6000000000000001,
            -0.19349006859052886,
            -0.41424360753592776,
            0.4604249207295301,
            0.10091468023596761,
            -0.03762926676455603,
            0.24054774626668213,
            -0.3623135655163526,
            0.6478850481938815,
            0.5600774600085,
            -0.4511045169829932,
            0.3269886858565265,
            0.3066965624530354,
            0.3319243238454705,
            -0.5309451470129688,
            0.2706975232733759,
            0.6189062528070872,
        ]
    )
    nodes_y = np.array(
        [
            -1.0,
            -0.6,
            -0.19999999999999996,
            0.20000000000000018,
            0.6000000000000001,
            1.0,
            -1.0,
            -0.6,
            -0.19999999999999996,
            0.20000000000000018,
            0.6000000000000001,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            0.0736248612304714,
            0.08817602246010336,
            -0.04617878507066997,
            -0.24867264850826343,
            -0.5798432722897898,
            0.5362608529745471,
            -0.24241940650019664,
            0.13035429545886779,
            -0.19374517234381208,
            0.49667185660255964,
            -0.6998830631694057,
            -0.396693906250825,
            -0.16563392390106765,
            -0.6078277671901492,
            0.04165930100991923,
            -0.504605776399067,
        ]
    )
    triangles = np.array(
        [
            [25, 29, 20],
            [29, 25, 13],
            [1, 33, 2],
            [33, 16, 17],
            [16, 1, 0],
            [1, 16, 33],
            [33, 26, 2],
            [9, 10, 27],
            [10, 25, 27],
            [7, 19, 6],
            [26, 23, 20],
            [25, 14, 13],
            [29, 4, 3],
            [12, 29, 13],
            [4, 12, 5],
            [12, 4, 29],
            [21, 3, 2],
            [26, 21, 2],
            [21, 26, 20],
            [29, 21, 20],
            [21, 29, 3],
            [8, 9, 27],
            [18, 19, 30],
            [25, 34, 27],
            [34, 25, 20],
            [23, 34, 20],
            [24, 26, 33],
            [24, 23, 26],
            [24, 33, 17],
            [18, 24, 17],
            [24, 18, 30],
            [15, 14, 25],
            [15, 10, 11],
            [10, 15, 25],
            [28, 8, 27],
            [34, 22, 27],
            [22, 28, 27],
            [8, 35, 7],
            [28, 35, 8],
            [19, 35, 30],
            [35, 19, 7],
            [22, 32, 28],
            [32, 34, 23],
            [32, 22, 34],
            [31, 35, 28],
            [32, 31, 28],
            [35, 31, 30],
            [31, 32, 23],
            [31, 24, 30],
            [24, 31, 23],
        ],
        dtype=np.int32,
    )

    figure, all_axes = plt.subplots(3, 3)
    all_axes = all_axes.flatten()
    for index in range(9):
        ax = all_axes[index]
        t = 0.125 * index
        yt = nodes_y + t
        xt = nodes_x + (yt * yt * yt - nodes_y * nodes_y * nodes_y) / 3
        ax.triplot(xt, yt, triangles)
        if t == int(t):
            title = "$t = {:d}.0$".format(int(t))
        else:
            title = "$t = {:g}$".format(t)
        ax.set_title(title)
        # Set the axis.
        ax.axis("scaled")
        ax.set_xlim(-1.22, 3.55)
        ax.set_ylim(-1.15, 2.15)

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
        left=0.04, bottom=0.02, right=0.98, top=0.98, wspace=0.04, hspace=0.0
    )
    path = plot_utils.get_path("curved-mesh", filename)
    figure.savefig(path, bbox_inches="tight")
    print("Saved {}".format(filename))
    plt.close(figure)


if __name__ == "__main__":
    plot_utils.set_styles()
    main()
