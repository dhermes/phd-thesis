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

In particular, ``scripts/compensated-newton/tangent_intersection.py``.

The only change is ``b_1 --> b_0`` and ``b_2 --> b_1``.
"""

import fractions

import bezier
import matplotlib.pyplot as plt
import numpy as np

import newton_bezier
import plot_utils


COEFFS1 = np.asfortranarray([[-2.0, -2.0, 6.0], [2.0, 0.0, 2.0]])
COEFFS2 = np.asfortranarray([[-4.0, -4.0, 12.0], [5.0, -3.0, 5.0]])
F = fractions.Fraction
U = 0.5 ** 53


def image1():
    figure = plt.figure()
    ax = figure.gca()

    curve1 = bezier.Curve(COEFFS1, degree=2, _copy=False)
    curve2 = bezier.Curve(COEFFS2, degree=2, _copy=False)

    curve1.plot(256, ax=ax, color=plot_utils.BLUE)
    ax.lines[-1].set_label("$b_0(s)$")
    curve2.plot(256, ax=ax, color=plot_utils.GREEN)
    ax.lines[-1].set_label("$b_1(t)$")
    ax.plot(
        [0.0],
        [1.0],
        marker="o",
        markersize=4.0,
        color="black",
        linestyle="none",
    )

    ax.legend(fontsize=plot_utils.TEXT_SIZE)
    ax.tick_params(labelsize=plot_utils.TICK_SIZE)
    ax.axis("scaled")

    figure.set_size_inches(5.4, 1.6)
    figure.subplots_adjust(
        left=0.05, bottom=0.09, right=0.99, top=0.99, wspace=0.2, hspace=0.2
    )
    filename = "tangent_intersection.pdf"
    path = plot_utils.get_path("slides", filename)
    figure.savefig(path)
    print("Saved {}".format(filename))
    plt.close(figure)


def main():
    image1()


if __name__ == "__main__":
    plot_utils.set_styles()
    main()
