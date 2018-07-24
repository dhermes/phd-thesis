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

r"""Perform Newton's iteration to find roots.

This uses Horner's algorithm to evaluate both :math:`p(s)` and
:math:`p'(s)`.

This script in particular uses :math:`p(x) = (x - 1)^n - 2^{-31}` which
has :math:`\widetilde{p}(s) = (x + 1)^n - (-1)^n 2^{-31}`.
"""

import collections
import fractions

import matplotlib.pyplot as plt
import mpmath

import horner
import plot_utils


U = fractions.Fraction(1, 2 ** 53)
ALPHA = 0.25


def get_coeffs(n):
    r"""Get the coefficients of a specific polynomial.

    We have :math:`p(x) = (x - 1)^n - 2^{-31}` and the coefficients
    are :math:`a_0 = (-1)^{n} - 2^{-31}` and
    :math:`a_j = \binom{n}{j} (-1)^{n - j}`.
    """
    coeffs = []
    for j in range(n + 1):
        coeff = plot_utils.binomial(n, j) * (-1.0) ** j
        if j == n:
            coeff -= 0.5 ** 31
        coeffs.append(coeff)

    return tuple(coeffs)


def find_best_solution(n):
    """Finds :math:`1 + 2^{-31/n}` to highest precision."""
    highest = 500
    root = None
    counter = collections.Counter()
    for precision in range(100, highest + 20, 20):
        ctx = mpmath.MPContext()
        ctx.prec = precision

        value = 1 + ctx.root(0.5 ** 31, n)
        if precision == highest:
            root = value
        value = plot_utils.to_float(value)
        counter[value] += 1

    if len(counter) != 1:
        raise ValueError("Expected only one value.")
    return root


def condition_number(n, root):
    r"""Compute the condition number of :math:`p(x)` at a root.

    When :math:`p(x) = (x - 1)^n - 2^{-31}` is written in the monomial basis,
    we have :math:`\widetilde{p}(x) = (x + 1)^n - (-1)^n 2^{-31}`.
    """
    if root <= 0:
        raise ValueError("Expected positive root.", root)
    p_tilde = (root + 1) ** n - (-1) ** n * 0.5 ** 31
    dp = n * (root - 1) ** (n - 1)
    return plot_utils.to_float(abs(p_tilde / (root * dp)))


def root_info(n):
    coeffs = get_coeffs(n)
    root = find_best_solution(n)
    cond = condition_number(n, root)
    # We know the root is in [1, 2) so this starts outside that interval.
    x1_converged = horner.basic_newton(2.0, coeffs)
    x2_converged = horner.accurate_newton(2.0, coeffs)
    x3_converged = horner.full_newton(2.0, coeffs)

    rel_error1 = plot_utils.to_float(abs((x1_converged - root) / root))
    rel_error2 = plot_utils.to_float(abs((x2_converged - root) / root))
    rel_error3 = plot_utils.to_float(abs((x3_converged - root) / root))
    if rel_error1 == 0:
        raise RuntimeError("Unexpected error for basic Newton.")
    if rel_error2 == 0:
        raise RuntimeError("Unexpected error for accurate Newton.")
    if rel_error3 == 0:
        raise RuntimeError("Unexpected error for full Newton.")

    return cond, rel_error1, rel_error2, rel_error3


def main(filename=None):
    cond_nums = []
    rel_errors_basic = []
    rel_errors_accurate = []
    rel_errors_full = []
    bounds_vals1 = []
    bounds_vals2 = []
    for n in range(2, 55 + 1):
        cond, rel_error1, rel_error2, rel_error3 = root_info(n)
        cond_nums.append(cond)
        rel_errors_basic.append(rel_error1)
        rel_errors_accurate.append(rel_error2)
        rel_errors_full.append(rel_error3)
        gamma2n = (2 * n * U) / (1 - 2 * n * U)
        bounds_vals1.append(float(10 * gamma2n * cond))
        bounds_vals2.append(float(6 * gamma2n ** 2 * cond))

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
        label=r"$\mathtt{HNewtonBasic}$",
    )
    ax.loglog(
        cond_nums,
        rel_errors_accurate,
        marker="o",
        linestyle="none",
        markersize=3,
        color="black",
        zorder=2,
        label=r"$\mathtt{HNewtonAccurate}$",
    )
    # H/T: (http://widu.tumblr.com/post/43624348228/
    #       making-unfilled-hollow-markers-in-matplotlib)
    ax.loglog(
        cond_nums,
        rel_errors_full,
        marker="o",
        linestyle="none",
        markersize=6,
        markeredgewidth=1,
        markerfacecolor="none",
        color=plot_utils.GREEN,
        zorder=2,
        label=r"$\mathtt{HNewtonFull}$",
    )
    # Add the error lines.
    ax.loglog(
        cond_nums,
        bounds_vals1,
        color="black",
        linestyle="dashed",
        alpha=ALPHA,
        zorder=1,
    )
    ax.loglog(
        cond_nums,
        bounds_vals2,
        color="black",
        linestyle="dashed",
        alpha=ALPHA,
        zorder=1,
    )
    # Add the ``x = 1/u^k`` vertical lines.
    min_x, max_x = 1e4, 1e33
    min_y, max_y = 1e-19, 1e2
    delta_y = max_y - min_y
    for exponent in (1, 2):
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
    # Set the major x- and y-ticks.
    ax.set_xticks([1e5, 1e10, 1e15, 1e20, 1e25, 1e30])
    ax.set_yticks(
        [1e-18, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1]
    )
    # Set special ``xticks`` for ``1/u`` and ``1/u^2``.
    u_xticks = [1.0 / float(U), 1.0 / float(U) ** 2]
    u_xticklabels = [r"$1/\mathbf{u}$", r"$1/\mathbf{u}^2$"]
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
    # Add the legend.
    ax.legend(
        loc="lower right",
        framealpha=1.0,
        frameon=True,
        fontsize=plot_utils.TEXT_SIZE,
    )

    ax.xaxis.set_tick_params(labelsize=plot_utils.TICK_SIZE)
    ax.xaxis.set_tick_params(labelsize=plot_utils.TEXT_SIZE, which="minor")
    ax.yaxis.set_tick_params(labelsize=plot_utils.TICK_SIZE)
    ax.yaxis.set_tick_params(labelsize=plot_utils.TEXT_SIZE, which="minor")

    figure.set_size_inches(5.4, 4.0)
    figure.subplots_adjust(
        left=0.12, bottom=0.11, right=0.95, top=0.92, wspace=0.2, hspace=0.2
    )
    filename = "newton_jghplus13.pdf"
    path = plot_utils.get_path("compensated-newton", filename)
    figure.savefig(path)
    print("Saved {}".format(filename))
    plt.close(figure)


if __name__ == "__main__":
    plot_utils.set_styles()
    main()
