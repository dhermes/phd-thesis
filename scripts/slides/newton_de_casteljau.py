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

In particular, ``scripts/compensated-newton/newton_de_casteljau.py``.

This takes a painfully long time (i.e. 5-6 minutes) to run.
"""

import fractions

import matplotlib.pyplot as plt
import mpmath
import numpy as np

import de_casteljau
import plot_utils


U = fractions.Fraction(1, 2 ** 53)
RHS = 2.0 ** 30
ALPHA = 0.25


def get_coeffs(n):
    r"""Get the coefficients of a specific polynomial.

    When :math:`p(s) = (1 - 5s)^n + 2^{30} (1 - 3s)^n`, the coefficients
    are :math:`b_j = (-4)^j + 2^{30} (-2)^j` since
    :math:`1 - 5s = (1 - s) - 4s` and :math:`1 - 2s = (1 - s) - 2s`.

    It's worth noting that these terms can be represented exactly in
    floating point when :math:`|j - 30| <= 52`.
    """
    coeffs = []
    for j in range(n + 1):
        coeff = (-4.0) ** j + RHS * (-2.0) ** j
        coeffs.append(coeff)

    return tuple(coeffs)


def find_best_solution(n, ctx):
    """Finds :math:`(1 + 2^{n/d}) / (5 + 3 (2^{n/d}))` to highest precision."""
    highest = 500
    w = ctx.root(RHS, n) - 1
    return (2 + w) / (8 + 3 * w)


def condition_number(n, root):
    r"""Compute the condition number of :math:`p(s)` at a root.

    When :math:`p(s) = (1 - 5s)^n + 2^{30} (1 - 3s)^n` is written in the
    Bernstein basis, we have :math:`\widetilde{p}(s) = (1 + 3s)^n +
    2^{30} (1 + s)^n`.
    """
    if not 0 <= root <= 1:
        raise ValueError("Expected root in unit interval.", root)
    p_tilde = (1 + 3 * root) ** n + RHS * (1 + root) ** n
    dp = -5 * n * (1 - 5 * root) ** (n - 1) - RHS * 3 * n * (1 - 3 * root) ** (
        n - 1
    )
    return plot_utils.to_float(abs(p_tilde / (root * dp)))


def root_info(n, ctx):
    coeffs = get_coeffs(n)
    root = find_best_solution(n, ctx)
    cond = condition_number(n, root)
    # We know the root is in [1/4, 1/3] so this starts outside that interval.
    s0 = 0.5
    s1_converged = de_casteljau.basic_newton(s0, coeffs)
    s2_converged = de_casteljau.accurate_newton(s0, coeffs)
    s3_converged = de_casteljau.full_newton(s0, coeffs)

    rel_error1 = plot_utils.to_float(abs((s1_converged - root) / root))
    rel_error2 = plot_utils.to_float(abs((s2_converged - root) / root))
    rel_error3 = plot_utils.to_float(abs((s3_converged - root) / root))
    if rel_error1 == 0:
        raise RuntimeError(
            "Unexpected error for basic Newton.", n, root, s1_converged
        )
    if rel_error2 == 0:
        raise RuntimeError(
            "Unexpected error for accurate Newton.", n, root, s2_converged
        )
    if rel_error3 == 0:
        raise RuntimeError(
            "Unexpected error for full Newton.", n, root, s3_converged
        )

    return cond, rel_error1, rel_error2, rel_error3


def get_bounds(values):
    # NOTE: This assumes ``values`` is positive.
    min_exp = np.log10(min(values))
    max_exp = np.log10(max(values))
    min_result = np.floor(min_exp)
    if min_exp - min_result < 0.5:
        min_result -= 1.0
    max_result = np.ceil(max_exp)
    if max_result - max_exp < 0.5:
        max_result += 1.0

    return 10.0 ** min_result, 10.0 ** max_result


def generate_data():
    ctx = mpmath.MPContext()
    ctx.prec = 500

    cond_nums = []
    rel_errors_basic = []
    rel_errors_accurate = []
    rel_errors_full = []
    for n in range(1, 71 + 2, 2):
        cond, rel_error1, rel_error2, rel_error3 = root_info(n, ctx)
        cond_nums.append(cond)
        rel_errors_basic.append(rel_error1)
        rel_errors_accurate.append(rel_error2)
        rel_errors_full.append(rel_error3)

    return cond_nums, rel_errors_basic, rel_errors_accurate, rel_errors_full


def error_plot(slide_num, data):
    cond_nums, rel_errors_basic, rel_errors_accurate, rel_errors_full = data

    figure = plt.figure()
    ax = figure.gca()
    ax.loglog(
        cond_nums,
        rel_errors_basic,
        marker="d",
        linestyle="none",
        markersize=7,
        color=plot_utils.BLUE,
        zorder=2,
        label=r"$\mathtt{DNewtonBasic}$",
    )
    line2, = ax.loglog(
        cond_nums,
        rel_errors_accurate,
        marker="o",
        linestyle="none",
        markersize=3,
        color="black",
        zorder=2,
        label=r"$\mathtt{DNewtonAccurate}$",
    )
    # H/T: (http://widu.tumblr.com/post/43624348228/
    #       making-unfilled-hollow-markers-in-matplotlib)
    line3, = ax.loglog(
        cond_nums,
        rel_errors_full,
        marker="o",
        linestyle="none",
        markersize=6,
        markeredgewidth=1,
        markerfacecolor="none",
        color=plot_utils.GREEN,
        zorder=2,
        label=r"$\mathtt{DNewtonFull}$",
    )
    # Add the error lines.
    ax.loglog(
        [1e-3, 1e30],
        [U * 1e-3, U * 1e30],
        color="black",
        linestyle="dashed",
        alpha=ALPHA,
        zorder=1,
    )
    ax.loglog(
        [1e10, 1e40],
        [U ** 2 * 1e10, U ** 2 * 1e40],
        color="black",
        linestyle="dashed",
        alpha=ALPHA,
        zorder=1,
    )
    # Add the ``x = 1/u^k`` vertical lines.
    min_x, max_x = get_bounds(cond_nums)
    min_y, max_y = get_bounds(
        rel_errors_basic + rel_errors_accurate + rel_errors_full
    )
    delta_y = max_y - min_y
    for exponent in (1, 2, 3):
        u_inv = 1.0 / float(U) ** exponent
        ax.loglog(
            [u_inv, u_inv],
            [min_y - 0.05 * delta_y, max_y + 0.05 * delta_y],
            color="black",
            linestyle="dashed",
            alpha=ALPHA,
            zorder=1,
        )
    # Add the ``y = 1`` and ``y = u`` horizontal lines.
    for exponent in (0, 1):
        u_pow = float(U) ** exponent
        ax.loglog(
            [min_x, max_x],
            [u_pow, u_pow],
            color="black",
            linestyle="dashed",
            alpha=ALPHA,
            zorder=1,
        )
    # Set the axis limits.
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    # Set "nice" ticks.
    ax.set_xticks([10.0 ** n for n in range(0, 50 + 10, 10)])
    ax.set_yticks([10.0 ** n for n in range(-16, 0 + 4, 4)])
    # Set special ``xticks`` for ``1/u^k``.
    u_xticks = [1.0 / float(U), 1.0 / float(U) ** 2, 1.0 / float(U) ** 3]
    u_xticklabels = [
        r"$1/\mathbf{u}$",
        r"$1/\mathbf{u}^2$",
        r"$1/\mathbf{u}^3$",
    ]
    ax.set_xticks(u_xticks, minor=True)
    ax.set_xticklabels(u_xticklabels, minor=True)
    ax.tick_params(
        axis="x",
        which="minor",
        direction="out",
        top=1,
        bottom=0,
        labelbottom=0,
        labeltop=1,
    )
    # Set special ``yticks`` for ``u`` and ``1``.
    ax.set_yticks([float(U), 1.0], minor=True)
    ax.set_yticklabels([r"$\mathbf{u}$", "$1$"], minor=True)
    ax.tick_params(
        axis="y",
        which="minor",
        direction="out",
        left=0,
        right=1,
        labelleft=0,
        labelright=1,
    )
    # Label the axes.
    ax.set_xlabel("Condition Number", fontsize=plot_utils.TEXT_SIZE)
    ax.set_ylabel("Relative Forward Error", fontsize=plot_utils.TEXT_SIZE)
    if slide_num < 2:
        line2.set_visible(False)
        # This is to fool the legend.
        line2.set_alpha(0.0)
    if slide_num < 3:
        line3.set_visible(False)
        # This is to fool the legend.
        line3.set_alpha(0.0)

    # Add the legend.
    legend = ax.legend(
        loc="lower right",
        framealpha=1.0,
        frameon=True,
        fontsize=plot_utils.TEXT_SIZE,
    )
    # Leave the text intact but make it "invisible".
    if slide_num < 2:
        _, text2, _ = legend.get_texts()
        text2.set_alpha(0.0)
    if slide_num < 3:
        _, _, text3 = legend.get_texts()
        text3.set_alpha(0.0)

    ax.tick_params(labelsize=plot_utils.TICK_SIZE)
    ax.tick_params(labelsize=plot_utils.TEXT_SIZE, which="minor")

    figure.set_size_inches(5.4, 4.0)
    figure.subplots_adjust(
        left=0.12, bottom=0.11, right=0.95, top=0.92, wspace=0.2, hspace=0.2
    )
    filename = "newton_de_casteljau{}.pdf".format(slide_num)
    path = plot_utils.get_path("slides", filename)
    figure.savefig(path)
    print("Saved {}".format(filename))
    plt.close(figure)


def main():
    data = generate_data()
    error_plot(1, data)
    error_plot(2, data)
    error_plot(3, data)


if __name__ == "__main__":
    plot_utils.set_styles()
    main()
