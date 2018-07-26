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
import mpmath
import numpy as np

import newton_bezier
import plot_utils


U = 0.5 ** 53
ALPHA = 0.25


def kappa(r, ctx):
    sqrt_r = ctx.sqrt(r)
    mu1 = 9 + 4 * sqrt_r + 2 * r - r * sqrt_r - 0.5 * r * r
    mu2 = 2 / r + 2 + 2 * r

    v1_v1 = 5.0 / (64.0 * r)
    v1_v2 = -3.0 / (32.0 * r) - 5.0 / (32.0 * r * sqrt_r)
    v2_v2 = (5 + 6 * sqrt_r + 2 * r) / (16 * r * r)

    numerator = (
        mu1 * mu1 * v1_v1 + 2 * mu1 * mu2 * abs(v1_v2) + mu2 * mu2 * v2_v2
    )
    alpha = 0.5 * (1 + sqrt_r)
    beta = 0.25 * (2 + sqrt_r)
    denominator = alpha * alpha + beta * beta
    kappa_sq = numerator / denominator

    return ctx.sqrt(kappa_sq)


def main():
    ctx = mpmath.MPContext()
    ctx.prec = 500

    s0 = 1.0
    t0 = 1.0

    cond_nums = np.empty((49,))
    rel_errors1 = np.empty((49,))
    rel_errors2 = np.empty((49,))
    for n in range(2, 50 + 1):
        r = 0.5 ** n
        r_inv = 1.0 / r
        cond_num = kappa(r, ctx)

        # Compute the coefficients.
        coeffs1 = np.asfortranarray(
            [[-2.0 - r, -2.0 - r, 6.0 - r], [2.0 + r_inv, r_inv, 2.0 + r_inv]]
        )
        coeffs2 = np.asfortranarray(
            [[-4.0, -4.0, 12.0], [5.0 + r_inv, -3.0 + r_inv, 5.0 + r_inv]]
        )
        # Use Newton's method to find the intersection.
        iterates1 = newton_bezier.newton(
            s0, coeffs1, t0, coeffs2, newton_bezier.standard_residual
        )
        iterates2 = newton_bezier.newton(
            s0, coeffs1, t0, coeffs2, newton_bezier.compensated_residual
        )
        # Just keep the final iterate and discard the rest.
        s1, t1 = iterates1[-1]
        s2, t2 = iterates2[-1]
        # Compute the relative error in the 2-norm.
        sqrt_r = ctx.sqrt(r)
        alpha = 0.5 * (1 + sqrt_r)
        beta = 0.25 * (2 + sqrt_r)
        size = ctx.norm([alpha, beta], p=2)
        rel_error1 = ctx.norm([alpha - s1, beta - t1], p=2) / size
        rel_error2 = ctx.norm([alpha - s2, beta - t2], p=2) / size
        # Convert the errors to floats and store.
        cond_nums[n - 2] = plot_utils.to_float(cond_num)
        rel_errors1[n - 2] = plot_utils.to_float(rel_error1)
        rel_errors2[n - 2] = plot_utils.to_float(rel_error2)

    # Make sure all of the non-compensated errors are non-zero and
    # at least one of the compensated errors is zero.
    if rel_errors1.min() <= 0.0:
        raise ValueError("Unexpected minimum error (non-compensated).")
    if rel_errors2.min() != 0.0:
        raise ValueError("Unexpected minimum error (compensated).")

    figure = plt.figure()
    ax = figure.gca()
    # Add all of the non-compensated errors.
    ax.loglog(
        cond_nums,
        rel_errors1,
        marker="v",
        linestyle="none",
        color=plot_utils.BLUE,
        label="Standard",
    )
    # Add the nonzero compensated errors.
    nz_inds = np.where(rel_errors2 > 0.0)
    ax.loglog(
        cond_nums[nz_inds],
        rel_errors2[nz_inds],
        marker="d",
        linestyle="none",
        color=plot_utils.GREEN,
        label="Compensated",
    )
    # Add the zero compensated errors at a "fake"
    # value below all other.
    min_nz = min(rel_errors1.min(), rel_errors2[nz_inds].min())
    below_min = min_nz / 1024.0
    zero_inds = np.where(rel_errors2 == 0.0)
    ax.loglog(
        cond_nums[zero_inds],
        np.full_like(rel_errors2[zero_inds], below_min),
        marker="d",
        linestyle="none",
        color=plot_utils.GREEN,
    )

    # Plot the lines of the a priori error bounds.
    min_x = 1.5
    max_x = 5.0e+32
    for coeff in (U, U ** 2):
        start_x = min_x
        start_y = coeff * start_x
        if start_y < 128.0 * below_min:
            start_y = 128.0 * below_min
            start_x = start_y / coeff
        ax.loglog(
            [start_x, max_x],
            [start_y, coeff * max_x],
            color="black",
            alpha=ALPHA,
            zorder=1,
        )
    # Add the ``x = 1/U`` and ``x = 1/U^2`` vertical lines.
    min_y = 1.5e-21
    max_y = 5.0
    for x_val in (1.0 / U, 1.0 / U ** 2):
        ax.loglog(
            [x_val, x_val],
            [min_y, max_y],
            color="black",
            linestyle="dashed",
            alpha=ALPHA,
            zorder=1,
        )
    # Add the ``y = 0``, ``y = u`` and ``y = 1`` horizontal lines.
    for y_val in (below_min, U, 1.0):
        ax.loglog(
            [min_x, max_x],
            [y_val, y_val],
            color="black",
            linestyle="dashed",
            alpha=ALPHA,
            zorder=1,
        )

    # Set "nice" ticks.
    ax.set_xticks([10.0 ** n for n in range(4, 28 + 8, 8)])
    ax.set_yticks([10.0 ** n for n in range(-16, 0 + 4, 4)])
    # Set special ``xticks`` for ``1/u`` and ``1/u^2``.
    ax.set_xticks([1.0 / U, 1.0 / U ** 2], minor=True)
    ax.set_xticklabels([r"$1/\mathbf{u}$", r"$1/\mathbf{u}^2$"], minor=True)
    ax.tick_params(
        axis="x",
        which="minor",
        direction="out",
        top=1,
        bottom=0,
        labelbottom=0,
        labeltop=1,
    )
    # Set special ``yticks`` for ``0``, ``u`` and ``1``.
    ax.set_yticks([below_min, U, 1.0], minor=True)
    ax.set_yticklabels(["$0$", r"$\mathbf{u}$", "$1$"], minor=True)
    ax.tick_params(
        axis="y",
        which="minor",
        direction="out",
        left=0,
        right=1,
        labelleft=0,
        labelright=1,
    )
    # Add a mark to show the "broken" axis in the y-scale.
    ax.plot(
        [-1.0 / 64, 1.0 / 64],
        [0.125 - 1.0 / 64, 0.125 + 1.0 / 64],
        transform=ax.transAxes,
        color="black",
        clip_on=False,
    )
    # Label the axes.
    ax.set_xlabel("Condition Number", fontsize=plot_utils.TEXT_SIZE)
    ax.set_ylabel("Relative Forward Error", fontsize=plot_utils.TEXT_SIZE)
    # Make sure the ticks are sized appropriately.
    ax.tick_params(labelsize=plot_utils.TICK_SIZE)
    ax.tick_params(labelsize=plot_utils.TEXT_SIZE, which="minor")

    # Set axis limits.
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    # Add the legend.
    ax.legend(
        loc="upper left",
        framealpha=1.0,
        frameon=True,
        fontsize=plot_utils.TEXT_SIZE,
    )

    figure.set_size_inches(6.0, 4.5)
    figure.subplots_adjust(
        left=0.11, bottom=0.09, right=0.96, top=0.94, wspace=0.2, hspace=0.2
    )
    filename = "almost_tangent.pdf"
    path = plot_utils.get_path("compensated-newton", filename)
    figure.savefig(path)
    print("Saved {}".format(filename))
    plt.close(figure)


if __name__ == "__main__":
    plot_utils.set_styles()
    main()
