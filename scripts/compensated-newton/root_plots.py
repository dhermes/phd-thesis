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

r"""Make a scatter plot of complex roots.

This will display the roots of :math:`p(s) = (1 - 5s)^n + 2^{d} (1 - 3s)^n`
:math:`d = 30` and :math:`n` odd.
"""

import matplotlib.pyplot as plt
import mpmath

import plot_utils


def add_plot(ax, ctx, n, d):
    # (1 - 5s) = 2^d (3s - 1) = (1 + w) (3s - 1)
    w_vals = [ctx.root(2.0 ** d, n, k=k) - 1 for k in range(n)]
    x_vals = []
    y_vals = []
    for w in w_vals:
        root = (2 + w) / (8 + 3 * w)
        x_vals.append(plot_utils.to_float(root.real))
        y_vals.append(plot_utils.to_float(root.imag))

    ax.plot(x_vals, y_vals, marker="o", markersize=2.0, linestyle="none")
    ax.set_title("$n = {}$".format(n), fontsize=plot_utils.TEXT_SIZE)


def main():
    figure, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ctx = mpmath.MPContext()
    ctx.prec = 500
    add_plot(ax1, ctx, 5, 30)
    add_plot(ax2, ctx, 15, 30)
    add_plot(ax3, ctx, 25, 30)

    for ax in (ax1, ax2, ax3):
        ax.tick_params(labelsize=plot_utils.TICK_SIZE)

    filename = "root_plots.pdf"
    figure.set_size_inches(6.4, 2.4)
    figure.subplots_adjust(
        left=0.08, bottom=0.11, right=0.99, top=0.92, wspace=0.29, hspace=0.2
    )
    path = plot_utils.get_path("compensated-newton", filename)
    figure.savefig(path)
    print("Saved {}".format(filename))
    plt.close(figure)


if __name__ == "__main__":
    plot_utils.set_styles()
    main()
