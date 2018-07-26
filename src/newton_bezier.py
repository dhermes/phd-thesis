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

import numpy as np

import de_casteljau
import eft


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
