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
