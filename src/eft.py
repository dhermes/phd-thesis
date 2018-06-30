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

"""Collection of error-free transforms."""


import fractions


def add_eft(val1, val2):
    # See: https://doi.org/10.1137/030601818
    sum_ = val1 + val2
    delta1 = sum_ - val1
    error = (val1 - (sum_ - delta1)) + (val2 - delta1)
    return sum_, error


def _split(val):
    # Helper for ``multiply_eft``.
    scaled = val * 134217729.0  # 134217729 == 2^{27} + 1.
    high_bits = scaled - (scaled - val)
    low_bits = val - high_bits
    return high_bits, low_bits


def _fma(val1, val2, val3):
    if (
        isinstance(val1, float)
        and isinstance(val2, float)
        and isinstance(val3, float)
    ):
        frac1 = fractions.Fraction(val1)
        frac2 = fractions.Fraction(val2)
        frac3 = fractions.Fraction(val3)
        return float(frac1 * frac2 + frac3)
    else:
        return val1.fma(val1, val2, val3)


def multiply_eft(val1, val2, use_fma=True):
    # See: https://doi.org/10.1109/TC.2008.215
    product = val1 * val2
    if use_fma:
        error = _fma(val1, val2, -product)
    else:
        high1, low1 = _split(val1)
        high2, low2 = _split(val2)
        error = low1 * low2 - (
            ((product - high1 * high2) - low1 * high2) - high1 * low2
        )

    return product, error


def _vec_sum(p):
    # See: https://doi.org/10.1137/030601818
    # Helper for ``sum_k``.
    # NOTE: This modifies ``p`` in place.
    n = len(p)
    for i in range(1, n):
        p[i], p[i - 1] = add_eft(p[i], p[i - 1])


def sum_k(p, k):
    # See: https://doi.org/10.1137/030601818
    p = list(p)  # Make a copy to be modified.

    for _ in range(k - 1):
        _vec_sum(p)

    result = p[0]
    for p_val in p[1:]:
        result += p_val

    return result
