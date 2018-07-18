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


import os

import seaborn


# As of ``0.9.0``, this palette has (BLUE, ORANGE, GREEN, RED, PURPLE, BROWN).
_COLORS = seaborn.color_palette(palette="deep", n_colors=6)
BLUE = _COLORS[0]
GREEN = _COLORS[2]
RED = _COLORS[3]
PURPLE = _COLORS[4]
del _COLORS


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
