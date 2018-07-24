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

r"""Performs Horner's method.

Horner's method computes

.. math::

    p(x) = a_n x^n + \cdots a_1 x + a_0

via

.. math::

    \begin{align*}
    p_n &= a_n \\
    p_k &= p_{k + 1} x + a_k \\
    p(x) &= p_0
    \end{align*}

This module provides both the standard version and a compensated version.

.. note::

   This assumes throughout that ``coeffs`` is ordered from
   :math:`a_n` to :math:`a_0`.
"""

import eft


def basic(x, coeffs):
    if not coeffs:
        return 0.0

    p = coeffs[0]
    for coeff in coeffs[1:]:
        p = p * x + coeff

    return p


def _compensated(x, coeffs):
    if not coeffs:
        return 0.0, [], []

    p = coeffs[0]
    e_pi = []
    e_sigma = []
    for coeff in coeffs[1:]:
        prod, e1 = eft.multiply_eft(p, x)
        p, e2 = eft.add_eft(prod, coeff)
        e_pi.append(e1)
        e_sigma.append(e2)

    return p, e_pi, e_sigma


def compensated(x, coeffs):
    p, e_pi, e_sigma = _compensated(x, coeffs)

    # Compute the error via standard Horner's.
    e = 0.0
    for e1, e2 in zip(e_pi, e_sigma):
        e = x * e + (e1 + e2)

    return p + e


def compensated3(x, coeffs):
    h1, p2, p3 = _compensated(x, coeffs)
    h2, p4, p5 = _compensated(x, p2)
    h3, p6, p7 = _compensated(x, p3)

    # Use standard Horner from here.
    h4 = basic(x, p4)
    h5 = basic(x, p5)
    h6 = basic(x, p6)
    h7 = basic(x, p7)

    # Now use 3-fold summation.
    p = [h1, h2, h3, h4, h5, h6, h7]
    return eft.sum_k(p, 3)


def compensated_k(x, coeffs, k):
    h = {}
    p = {1: coeffs}

    # First, "filter" off the errors from the interior
    # polynomials.
    for i in range(1, 2 ** (k - 1)):
        h[i], p[2 * i], p[2 * i + 1] = _compensated(x, p[i])

    # Then use standard Horner for the leaf polynomials.
    for i in range(2 ** (k - 1), 2 ** k):
        h[i] = basic(x, p[i])

    # Now use K-fold summation on everything in ``h`` (but keep the
    # order).
    to_sum = [h[i] for i in range(1, 2 ** k)]
    return eft.sum_k(to_sum, k)


def basic_newton_update(x, coeffs):
    numerator = basic(x, coeffs)
    denominator = hd1(x, coeffs)
    return x - numerator / denominator


def hd1(x, coeffs):
    """Performs the ``HD`` algorithm when ``k = 1``.

    .. _JGH+13: https://dx.doi.org/10.1016/j.cam.2012.11.008

    See the `JGH+13`_ paper for more details on the ``HD`` algorithm and the
    ``CompHD`` algorithm.

    Here ``HD`` stands for "Horner derivative".
    """
    y1 = 0.0
    y2 = coeffs[0]

    for coeff in coeffs[1:-1]:
        # Update ``y1``.
        y1 = x * y1 + y2
        # Update ``y2``.
        y2 = x * y2 + coeff

    # Perform one last update of ``y1``.
    return x * y1 + y2


def compensated_hd1(x, coeffs):
    """Performs the compensated ``HD`` algorithm when ``k = 1``.

    .. _JGH+13: https://dx.doi.org/10.1016/j.cam.2012.11.008

    See the `JGH+13`_ paper for more details on the ``HD`` algorithm and the
    ``CompHD`` algorithm.

    Here ``HD`` stands for "Horner derivative".
    """
    y1 = 0.0
    y2 = coeffs[0]
    e1 = 0.0  # y1_hat = y1 + e1
    e2 = 0.0  # y2_hat = y2 + e2

    for coeff in coeffs[1:-1]:
        # Update ``y1`` and ``e1``.
        prod, pi = eft.multiply_eft(x, y1)
        y1, sigma = eft.add_eft(prod, y2)
        e1 = x * e1 + e2 + (pi + sigma)
        # Update ``y2`` and ``e2``.
        prod, pi = eft.multiply_eft(x, y2)
        y2, sigma = eft.add_eft(prod, coeff)
        e2 = x * e2 + (pi + sigma)

    # Perform one last update of ``y1`` and ``e1``.
    prod, pi = eft.multiply_eft(x, y1)
    y1, sigma = eft.add_eft(prod, y2)
    e1 = x * e1 + e2 + (pi + sigma)

    # Return the compensated form of ``y1``.
    return y1 + e1


def basic_newton(x0, coeffs, max_iter=100, tol=1e-15):
    """Perform Newton's method to find a root of a polynomial.

    This assumes ``coeffs`` are the coefficients of :math:`p(s)` in the
    monomial basis.
    """
    curr_x = x0
    for _ in range(max_iter):
        next_x = basic_newton_update(curr_x, coeffs)
        if abs(next_x - curr_x) < tol:
            return next_x
        curr_x = next_x

    return curr_x


def accurate_newton_update(x, coeffs):
    numerator = compensated(x, coeffs)
    denominator = hd1(x, coeffs)
    return x - numerator / denominator


def accurate_newton(x0, coeffs, max_iter=100, tol=1e-15):
    curr_x = x0
    for _ in range(max_iter):
        next_x = accurate_newton_update(curr_x, coeffs)
        if abs(next_x - curr_x) < tol:
            return next_x
        curr_x = next_x

    return curr_x


def full_newton_update(x, coeffs):
    numerator = compensated(x, coeffs)
    denominator = compensated_hd1(x, coeffs)
    return x - numerator / denominator


def full_newton(x0, coeffs, max_iter=100, tol=1e-15):
    curr_x = x0
    for _ in range(max_iter):
        next_x = full_newton_update(curr_x, coeffs)
        if abs(next_x - curr_x) < tol:
            return next_x
        curr_x = next_x

    return curr_x
