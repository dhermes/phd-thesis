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
import matplotlib.patches
import matplotlib.path
import matplotlib.pyplot as plt
import numpy as np

import plot_utils


ALPHA = 0.375


def main():
    nodes = np.asfortranarray(
        [[1.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]]
    )
    bez_triangle = bezier.Surface(nodes, degree=2, _copy=False)
    # b(s, t) = [(1 - s - t)^2 + t^2, s^2 + t^2]
    figure, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
        2, 3, sharex=True, sharey=True
    )
    for ax in (ax1, ax2, ax3):
        ax.plot([0, 1, 0, 0], [0, 0, 1, 0], color="black")

    # The "left" edge b(0, t) = [(1 - t)^2 + t^2, t^2] lies on the algebraic
    # curve 4x = (1 + x - y)^2. Plugging b(s, t) into this algebraic curve
    # we find interior points as well.
    sv = np.linspace(0.0, 2.0 / 5.0, 250)
    sqrt_part = np.sqrt((2.0 - 5.0 * sv) / (2.0 - sv))
    poly1 = np.empty((501, 2))
    poly1[:250, 0] = sv
    poly1[:250, 1] = 0.5 * (1.0 - sv + sqrt_part)
    poly1[250:499, 0] = sv[-2::-1]
    poly1[250:499, 1] = 0.5 * (1.0 - sv[-2::-1] - sqrt_part[-2::-1])
    poly1[499, :] = 1.0, 0.0
    poly1[500, :] = 0.0, 1.0
    patch = matplotlib.patches.PathPatch(
        matplotlib.path.Path(poly1), color=plot_utils.GREEN
    )
    ax1.add_patch(patch)
    for ax in (ax2, ax3):
        patch = matplotlib.patches.PathPatch(
            matplotlib.path.Path(poly1), color=plot_utils.GREEN, alpha=ALPHA
        )
        ax.add_patch(patch)

    for ax in (ax4, ax5, ax6):
        bez_triangle.plot(256, ax=ax, color=plot_utils.GREEN)
        for edge in bez_triangle.edges:
            edge.plot(256, ax=ax, color="black")
    ax4.patches[-1].set_alpha(1.0)
    ax5.patches[-1].set_alpha(ALPHA)
    ax6.patches[-1].set_alpha(ALPHA)

    # det(J) = 4[t^2 + (s - 1) t + (s - s^2)]
    sv = np.linspace(0.0, 1.0 / 5.0, 250)
    sqrt_part = np.sqrt((1.0 - sv) * (1.0 - 5.0 * sv))
    poly2 = np.empty((500, 2))
    poly2[:250, 0] = sv
    poly2[:250, 1] = 0.5 * (1.0 - sv - sqrt_part)
    poly2[250:499, 0] = sv[-2::-1]
    poly2[250:499, 1] = 0.5 * (1.0 - sv[-2::-1] + sqrt_part[-2::-1])
    poly2[499, :] = 0.0, 0.0
    for ax in (ax1, ax2, ax3):
        ax.plot(
            poly2[:499, 0], poly2[:499, 1], color="black", linestyle="dashed"
        )
    patch = matplotlib.patches.PathPatch(
        matplotlib.path.Path(poly2), color=plot_utils.RED
    )
    ax3.add_patch(patch)
    for ax in (ax1, ax2):
        patch = matplotlib.patches.PathPatch(
            matplotlib.path.Path(poly2), color="black", alpha=ALPHA
        )
        ax.add_patch(patch)

    # Shared slice in between two curves.
    poly3 = np.empty((997, 2))
    poly3[:499, :] = poly2[498::-1, :]
    poly3[499:, :] = poly1[497::-1, :]
    patch = matplotlib.patches.PathPatch(
        matplotlib.path.Path(poly3), color=plot_utils.BLUE
    )
    ax2.add_patch(patch)
    for ax in (ax1, ax3):
        patch = matplotlib.patches.PathPatch(
            matplotlib.path.Path(poly3), color="black", alpha=ALPHA
        )
        ax.add_patch(patch)

    # Now, compute the image of the det(J) = 0 curve under b(s, t).
    poly4 = np.empty((598, 2))
    sv = poly2[:499, 0][::-1]
    tv = poly2[:499, 1][::-1]
    poly4[:499, 0] = (1.0 - sv - tv) * (1.0 - sv - tv) + sv * sv
    poly4[:499, 1] = sv * sv + tv * tv
    for ax in (ax4, ax5, ax6):
        ax.plot(
            poly4[:499, 0], poly4[:499, 1], color="black", linestyle="dashed"
        )
    # Combine this with the image b(0, t)
    tv = np.linspace(0.0, 1.0, 100)[1:]
    poly4[499:, 0] = (1.0 - tv) * (1.0 - tv)
    poly4[499:, 1] = tv * tv
    patch = matplotlib.patches.PathPatch(
        matplotlib.path.Path(poly4), color="black", alpha=ALPHA
    )
    ax4.add_patch(patch)
    for ax, color in ((ax5, plot_utils.BLUE), (ax6, plot_utils.RED)):
        patch = matplotlib.patches.PathPatch(
            matplotlib.path.Path(poly4), color=color
        )
        ax.add_patch(patch)

    for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
        ax.set_aspect("equal")
    # One axis sets all axis
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    ax2.text(
        0.3,
        0.35,
        r"$+$",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=16,
        color="black",
    )
    ax2.add_patch(
        matplotlib.patches.Circle(
            (0.3, 0.35), radius=0.06, fill=False, edgecolor="black"
        )
    )
    ax3.text(
        0.1,
        0.42,
        r"$-$",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=16,
        color="black",
    )
    ax3.add_patch(
        matplotlib.patches.Circle(
            (0.1, 0.42), radius=0.06, fill=False, edgecolor="black"
        )
    )
    figure.set_size_inches(7.45, 5.19)
    figure.subplots_adjust(
        left=0.03, bottom=0.07, right=1.0, top=0.98, wspace=0.0, hspace=0.06
    )
    filename = "inverted_element.pdf"
    path = plot_utils.get_path("curved-mesh", filename)
    figure.savefig(path, bbox_inches="tight")
    print("Saved {}".format(filename))
    plt.close(figure)


if __name__ == "__main__":
    plot_utils.set_styles()
    main()
