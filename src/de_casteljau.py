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

r"""Performs de Casteljau's method.

de Casteljau's method evaluates a function in Bernstein-Bezier form

.. math::

    p(s) = p_0 (1 - s)^n + \cdots + p_j \binom{n}{j} (1 - s)^{n - j} s^j +
        \cdots + p_n s^n

by progressively reducing the control points

.. math::

    \begin{align*}
    p_j^{(0)} &= p_j \\
    p_j^{(k + 1)} &= (1 - s) p_j^{(k)} + p_{j + 1}^{(k)} \\
    p(s) &= p_0^{(n)}.
    \end{align*}

This module provides both the standard version and a compensated version.

.. note::

   This assumes throughout that ``coeffs`` is ordered from
   :math:`p_n` to :math:`p_0`.
"""


import eft


def basic(s, coeffs):
    """Performs the "standard" de Casteljau algorithm."""
    r = 1 - s

    degree = len(coeffs) - 1
    pk = list(coeffs)
    for k in range(degree):
        new_pk = []
        for j in range(degree - k):
            new_pk.append(r * pk[j] + s * pk[j + 1])
        # Update the "current" values.
        pk = new_pk

    return pk[0]


def derivative(s, coeffs):
    """Performs the "standard" de Casteljau algorithm."""
    r = 1 - s

    degree = len(coeffs) - 1
    pk = []
    for k in range(degree):
        pk.append(coeffs[k + 1] - coeffs[k])

    for k in range(degree - 1):
        new_pk = []
        for j in range(degree - 1 - k):
            new_pk.append(r * pk[j] + s * pk[j + 1])
        # Update the "current" values.
        pk = new_pk

    return degree * pk[0]


def local_error(errors, rho, delta_b):
    r"""Compute :math:`\ell` from a list of errors.

    This assumes, but does not check, that there are at least two
    ``errors``.
    """
    num_errs = len(errors)

    l_hat = errors[0] + errors[1]
    for j in range(2, num_errs):
        l_hat += errors[j]

    l_hat += rho * delta_b

    return l_hat


def local_error_eft(errors, rho, delta_b):
    r"""Perform an error-free transformation for computing :math:`\ell`.

    This assumes, but does not check, that there are at least two
    ``errors``.
    """
    num_errs = len(errors)
    new_errors = [None] * (num_errs + 1)

    l_hat, new_errors[0] = eft.add_eft(errors[0], errors[1])
    for j in range(2, num_errs):
        l_hat, new_errors[j - 1] = eft.add_eft(l_hat, errors[j])

    prod, new_errors[num_errs - 1] = eft.multiply_eft(rho, delta_b)
    l_hat, new_errors[num_errs] = eft.add_eft(l_hat, prod)

    return new_errors, l_hat


def _compensated_k(s, coeffs, K):
    r"""Performs a K-compensated de Casteljau.

    .. _JLCS10: https://doi.org/10.1016/j.camwa.2010.05.021

    Note that the order of operations exactly matches the `JLCS10`_ paper.
    For example, :math:`\widehat{\partial b}_j^{(k)}` is computed as

    .. math::

        \widehat{ell}_{1, j}^{(k)} \oplus \left(s \otimes
            \widehat{\partial b}_{j + 1}^{(k + 1)}\right) \oplus
            \left(\widehat{r} \otimes \widehat{\partial b}_j^{(k + 1)}\right)

    instead of "typical" order

    .. math::

        \left(\widehat{r} \otimes \widehat{\partial b}_j^{(k + 1)}\right)
            \oplus \left(s \otimes \widehat{\partial b}_{j + 1}^{(k + 1)}
            \right) \oplus \widehat{ell}_{1, j}^{(k)}.

    This is so that the term

    .. math::

        \widehat{r} \otimes \widehat{\partial b}_j^{(k + 1)}

    only has to be in one sum. We avoid an extra sum because
    :math:`\widehat{r}` already has round-off error.
    """
    r, rho = eft.add_eft(1.0, -s)

    degree = len(coeffs) - 1
    bk = {0: list(coeffs)}
    # NOTE: This will be shared, but is read only.
    all_zero = (0.0,) * (degree + 1)
    for F in range(1, K - 1 + 1):
        bk[F] = all_zero

    for k in range(degree):
        new_bk = {F: [] for F in range(K - 1 + 1)}

        for j in range(degree - k):
            # Update the "level 0" stuff.
            P1, pi1 = eft.multiply_eft(r, bk[0][j])
            P2, pi2 = eft.multiply_eft(s, bk[0][j + 1])
            S3, sigma3 = eft.add_eft(P1, P2)
            new_bk[0].append(S3)

            errors = [pi1, pi2, sigma3]
            delta_b = bk[0][j]

            for F in range(1, K - 2 + 1):
                new_errors, l_hat = local_error_eft(errors, rho, delta_b)
                P1, pi1 = eft.multiply_eft(s, bk[F][j + 1])
                S2, sigma2 = eft.add_eft(l_hat, P1)
                P3, pi3 = eft.multiply_eft(r, bk[F][j])
                S, sigma4 = eft.add_eft(S2, P3)
                new_bk[F].append(S)

                new_errors.extend([pi1, sigma2, pi3, sigma4])
                errors = new_errors
                delta_b = bk[F][j]

            # Update the "level 2" stuff.
            l_hat = local_error(errors, rho, delta_b)
            new_bk[K - 1].append(
                l_hat + s * bk[K - 1][j + 1] + r * bk[K - 1][j]
            )

        # Update the "current" values.
        bk = new_bk

    return tuple(bk[F][0] for F in range(K - 1 + 1))


def compensated(s, coeffs):
    b, db = _compensated_k(s, coeffs, 2)
    return eft.sum_k((b, db), 2)


def compensated3(s, coeffs):
    b, db, d2b = _compensated_k(s, coeffs, 3)
    return eft.sum_k((b, db, d2b), 3)


def compensated4(s, coeffs):
    b, db, d2b, d3b = _compensated_k(s, coeffs, 4)
    return eft.sum_k((b, db, d2b, d3b), 4)


def compensated5(s, coeffs):
    b, db, d2b, d3b, d4b = _compensated_k(s, coeffs, 5)
    return eft.sum_k((b, db, d2b, d3b, d4b), 5)


def pre_compensated_derivative(coeffs):
    degree = len(coeffs) - 1
    pk = []
    err_k = []
    for k in range(degree):
        delta_b, sigma = eft.add_eft(coeffs[k + 1], -coeffs[k])
        pk.append(delta_b)
        err_k.append(sigma)

    return pk, err_k


def _compensated_derivative(s, pk, err_k):
    r, rho = eft.add_eft(1.0, -s)
    degree = len(pk)
    for k in range(1, degree):
        new_pk = []
        new_err_k = []
        for j in range(degree - k):
            # new_pk.append(r * pk[j] + s * pk[j + 1])
            prod1, d_pi1 = eft.multiply_eft(pk[j], r)
            prod2, d_pi2 = eft.multiply_eft(pk[j + 1], s)
            new_dp, d_sigma = eft.add_eft(prod1, prod2)
            new_pk.append(new_dp)
            d_ell = d_pi1 + d_pi2 + d_sigma + pk[j] * rho
            new_err = d_ell + s * err_k[j + 1] + r * err_k[j]
            new_err_k.append(new_err)

        # Update the "current" values.
        pk = new_pk
        err_k = new_err_k

    return degree, pk[0], err_k[0]


def compensated_derivative(s, pk, err_k):
    """Compute :math:`p'(s)` with compensation.

    This assumes ``pk`` and ``err_k`` have been computed by
    :func:`pre_compensated_derivative`.
    """
    degree, p, e = _compensated_derivative(s, pk, err_k)
    return degree * (p + e)


def basic_newton_update(s, coeffs, d_coeffs):
    numerator = basic(s, coeffs)
    n = len(coeffs) - 1
    denominator = n * basic(s, d_coeffs)
    return s - numerator / denominator


def basic_newton(s0, coeffs, max_iter=100, tol=1e-15):
    r"""Perform Newton's method to find a root of a polynomial.

    When taking the derivative of :math:`p(s) = \sum_j b_j B_{j, n}(s)`,
    we have :math:`p'(s) = n \sum_j \Delta b_j B_{j, n - 1}(s)` where
    :math:`\Delta b_j = b_{j + 1} - b_j` is the first forward difference.

    Args:
        s0 (float): The starting value for the iteration.
        coeffs (Tuple[float, ...]): The coefficients of :math:`p(s)` in the
            Bernstein basis. Will be of the form :math:`(b_0, b_1, \ldots,
            b_n)`.
        max_iter (int): The maximum number of iterations.
        tol (float): The absolute error tolerance to converge.

    Returns:
        float: The value that Newton's method converged to.
    """
    curr_s = s0
    d_coeffs = tuple(val2 - val1 for val1, val2 in zip(coeffs, coeffs[1:]))
    for _ in range(max_iter):
        next_s = basic_newton_update(curr_s, coeffs, d_coeffs)
        if abs(next_s - curr_s) < tol:
            return next_s
        curr_s = next_s

    return curr_s


def accurate_newton_update(s, coeffs, d_coeffs):
    numerator = compensated(s, coeffs)
    n = len(coeffs) - 1
    denominator = n * basic(s, d_coeffs)
    return s - numerator / denominator


def accurate_newton(s0, coeffs, max_iter=100, tol=1e-15):
    curr_s = s0
    d_coeffs = tuple(val2 - val1 for val1, val2 in zip(coeffs, coeffs[1:]))
    for _ in range(max_iter):
        next_s = accurate_newton_update(curr_s, coeffs, d_coeffs)
        if abs(next_s - curr_s) < tol:
            return next_s
        curr_s = next_s

    return curr_s


def full_newton_update(s, coeffs, pk, err_k):
    numerator = compensated(s, coeffs)
    denominator = compensated_derivative(s, pk, err_k)
    if denominator == 0.0:
        # If there is a division-by-zero, just give up.
        return s

    return s - numerator / denominator


def full_newton(s0, coeffs, max_iter=100, tol=1e-15):
    curr_s = s0
    pk, err_k = pre_compensated_derivative(coeffs)
    for _ in range(max_iter):
        next_s = full_newton_update(curr_s, coeffs, pk, err_k)
        if abs(next_s - curr_s) < tol:
            return next_s
        curr_s = next_s

    return curr_s
