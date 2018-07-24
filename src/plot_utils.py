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

"""Shared utilities and settings for plotting."""


import fractions
import math
import os

import seaborn


# As of ``0.9.0``, this palette has (BLUE, ORANGE, GREEN, RED, PURPLE, BROWN).
_COLORS = seaborn.color_palette(palette="deep", n_colors=6)
BLUE = _COLORS[0]
GREEN = _COLORS[2]
RED = _COLORS[3]
PURPLE = _COLORS[4]
del _COLORS
TEXT_SIZE = 10  # NOTE: Thesis text uses 12 point.
TICK_SIZE = 7


def set_styles():
    """Set the styles used for plotting."""
    seaborn.set(style="white")


def get_path(*parts):
    """Get a file path in the ``images/`` directory.

    This assumes the script is currently in the ``src/``
    directory.
    """
    curr_dir = os.path.abspath(os.path.dirname(__file__))
    root_dir = os.path.dirname(curr_dir)
    images_dir = os.path.join(root_dir, "images")
    return os.path.join(images_dir, *parts)


def binomial(n, k):
    numerator = math.factorial(n)
    denominator = math.factorial(k) * math.factorial(n - k)
    result = fractions.Fraction(numerator, denominator)
    if float(result) != result:
        raise ValueError("Cannot be represented exactly")
    return float(result)


def next_float(value, greater=True):
    """Gets the next (or previous) floating point value."""
    frac, exponent = math.frexp(value)
    if greater:
        if frac == -0.5:
            ulp = 0.5 ** 54
        else:
            ulp = 0.5 ** 53
    else:
        if frac == 0.5:
            ulp = -0.5 ** 54
        else:
            ulp = -0.5 ** 53

    return (frac + ulp) * 2.0 ** exponent


def to_float(v):
    """Converts an MPF (``mpmath`` float) to a ``float``."""
    f = float(v)
    if f == v:
        return f
    if f < v:
        low = f
        high = next_float(f, greater=True)
    else:
        low = next_float(f, greater=False)
        high = f

    d_low = v - low
    d_high = high - v
    if d_low < d_high:
        return low
    else:
        return high
