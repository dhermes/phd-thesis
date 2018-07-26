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

import fractions

import bezier
import matplotlib.pyplot as plt
import numpy as np

import de_casteljau
import eft
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
    ax.lines[-1].set_label("$b_1(s)$")
    curve2.plot(256, ax=ax, color=plot_utils.GREEN)
    ax.lines[-1].set_label("$b_2(t)$")
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
    path = plot_utils.get_path("compensated-newton", filename)
    figure.savefig(path)
    print("Saved {}".format(filename))
    plt.close(figure)


def standard_residual(s, coeffs1, t, coeffs2):
    x1 = de_casteljau.basic(s, coeffs1[0, :])
    y1 = de_casteljau.basic(s, coeffs1[1, :])
    x2 = de_casteljau.basic(t, coeffs2[0, :])
    y2 = de_casteljau.basic(t, coeffs2[1, :])
    return np.array([[x1 - x2], [y1 - y2]])


def compensated_residual(s, coeffs1, t, coeffs2):
    x1, dx1 = de_casteljau._compensated_k(s, coeffs1[0, :], 2)
    y1, dy1 = de_casteljau._compensated_k(s, coeffs1[1, :], 2)
    x2, dx2 = de_casteljau._compensated_k(t, coeffs2[0, :], 2)
    y2, dy2 = de_casteljau._compensated_k(t, coeffs2[1, :], 2)

    dx, sigma = eft.add_eft(x1, -x2)
    tau = (dx1 - dx2) + sigma
    dx += tau
    dy, sigma = eft.add_eft(y1, -y2)
    tau = (dy1 - dy2) + sigma
    dy += tau

    return np.array([[dx], [dy]])


def newton(s0, coeffs1, t0, coeffs2, residual):
    max_iter = 50
    tol = 1e-15
    s = s0
    t = t0

    iterates = []
    for _ in range(max_iter):
        F = residual(s, coeffs1, t, coeffs2)
        # Compute the standard Jacobian.
        dx1 = de_casteljau.derivative(s, coeffs1[0, :])
        dy1 = de_casteljau.derivative(s, coeffs1[1, :])
        dx2 = de_casteljau.derivative(t, coeffs2[0, :])
        dy2 = de_casteljau.derivative(t, coeffs2[1, :])
        J = np.array([[dx1, -dx2], [dy1, -dy2]])
        # Solve for the updates.
        ds, dt = np.linalg.solve(J, F).flatten()
        # Apply the updates.
        s = s - ds
        t = t - dt
        iterates.append((s, t))
        # Return if the update is below the tolerance.
        if np.linalg.norm([ds, dt], ord=2) < tol:  # 2-norm
            break

    return iterates


def image2():
    expected_s = 0.5
    expected_t = 0.5
    s0 = 1.0 - 0.5 ** 40
    t0 = 0.75 + 0.5 ** 20

    iterates1 = newton(s0, COEFFS1, t0, COEFFS2, standard_residual)
    iterates2 = newton(s0, COEFFS1, t0, COEFFS2, compensated_residual)
    errors1 = []
    errors2 = []
    for iterates, errors in ((iterates1, errors1), (iterates2, errors2)):
        for n, (s_val, t_val) in enumerate(iterates):
            rel_error_s = F(s_val) / F(expected_s) - 1
            if rel_error_s <= 0:
                raise ValueError(s_val, rel_error_s)
            rel_error_t = F(t_val) / F(expected_t) - 1
            if rel_error_t <= 0:
                raise ValueError(t_val, rel_error_t)
            errors.append((n, float(rel_error_s), float(rel_error_t)))

    figure, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    min_x = -2.0
    max_x = 51.0
    errors1 = np.array(errors1)
    errors2 = np.array(errors2)
    for index, ax in ((1, ax1), (2, ax2)):
        ax.semilogy(
            errors1[:, 0],
            errors1[:, index],
            marker="o",
            linestyle="none",
            markersize=7,
            markeredgewidth=1,
            markerfacecolor="none",
            label="Standard",
        )
        ax.semilogy(
            errors2[:, 0],
            errors2[:, index],
            color="black",
            marker="o",
            linestyle="none",
            markersize=3,
            label="Compensated",
        )
        ax.semilogy(
            [min_x, max_x],
            [np.cbrt(U), np.cbrt(U)],
            linestyle="dashed",
            color="black",
        )
        ax.semilogy(
            [min_x, max_x],
            [np.cbrt(U * U), np.cbrt(U * U)],
            linestyle="dashed",
            color="black",
        )

        ax.set_yscale("log", basey=2)
        ax.set_xlabel("Iteration", fontsize=plot_utils.TEXT_SIZE)

    ax1.set_ylabel("Relative Error", fontsize=plot_utils.TEXT_SIZE)
    ax1.set_title("$s$", fontsize=plot_utils.TEXT_SIZE)
    ax1.set_xlim(min_x, max_x)
    ax2.set_title("$t$", fontsize=plot_utils.TEXT_SIZE)
    ax2.legend(loc="upper right", fontsize=plot_utils.TEXT_SIZE)

    ax2.set_yticks([np.cbrt(U), np.cbrt(U * U)], minor=True)
    ax2.set_yticklabels(
        [r"$\sqrt[3]{\mathbf{u}}$", r"$\sqrt[3]{\mathbf{u}^2}$"], minor=True
    )
    plt.setp(ax1.get_yticklabels(minor=True), visible=False)
    ax2.tick_params(
        axis="y",
        which="minor",
        direction="out",
        left=0,
        right=1,
        labelleft=0,
        labelright=1,
    )

    ax1.tick_params(labelsize=plot_utils.TICK_SIZE)
    ax2.tick_params(labelsize=plot_utils.TICK_SIZE)
    ax2.tick_params(labelsize=plot_utils.TEXT_SIZE, which="minor")

    figure.set_size_inches(6.4, 2.6)
    figure.subplots_adjust(
        left=0.09, bottom=0.16, right=0.93, top=0.9, wspace=0.04, hspace=0.2
    )
    filename = "newton_linear_converge.pdf"
    path = plot_utils.get_path("compensated-newton", filename)
    figure.savefig(path)
    print("Saved {}".format(filename))
    plt.close(figure)


def main():
    image1()
    image2()


if __name__ == "__main__":
    plot_utils.set_styles()
    main()
