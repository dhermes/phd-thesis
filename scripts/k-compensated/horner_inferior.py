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

"""Compare how "smooth" plotted values look.

This will contrast three different implementations (``K = 1, 2, 3``)
and show that more accuracy produces smoother plots.
"""

import matplotlib.pyplot as plt
import numpy as np

import de_casteljau
import horner
import plot_utils

# p(s) = (2s - 1)^3 = (-(1 - s) + s)^3
POLY_COEFFS = (8.0, -12.0, 6.0, -1.0)
BEZIER_COEFFS = (-1.0, 1.0, -1.0, 1.0)
ROOT = 0.5
DELTA_S = 5e-6
NUM_POINTS = 401


def _main():
    s_vals = np.linspace(ROOT - DELTA_S, ROOT + DELTA_S, NUM_POINTS)

    horner1 = []
    de_casteljau1 = []

    for s in s_vals:
        horner1.append(horner.basic(s, POLY_COEFFS))
        de_casteljau1.append(de_casteljau.basic(s, BEZIER_COEFFS))

    figure, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    ax1.plot(s_vals, horner1)
    ax2.plot(s_vals, de_casteljau1)

    # Since ``sharex=True``, ticks only need to be set once.
    ax1.set_xticks(
        [
            ROOT - DELTA_S,
            ROOT - 0.5 * DELTA_S,
            ROOT,
            ROOT + 0.5 * DELTA_S,
            ROOT + DELTA_S,
        ]
    )

    ax1.set_title(r"$\mathtt{Horner}$", fontsize=plot_utils.TEXT_SIZE)
    ax2.set_title(r"$\mathtt{DeCasteljau}$", fontsize=plot_utils.TEXT_SIZE)

    ax1.tick_params(labelsize=plot_utils.TICK_SIZE, which="both")
    ax2.tick_params(labelsize=plot_utils.TICK_SIZE, which="both")

    filename = "horner_inferior.pdf"
    # NOTE: These are (intended to be) the same settings used in
    #       ``compensated_insufficient.py``, so they should probably be
    #       kept in sync.
    figure.set_size_inches(6.0, 2.9)
    figure.subplots_adjust(
        left=0.07, bottom=0.13, right=0.97, top=0.92, wspace=0.13, hspace=0.20
    )
    path = plot_utils.get_path("k-compensated", filename)
    figure.savefig(path)
    print("Saved {}".format(filename))
    plt.close(figure)


def main():
    mapping = {
        "xtick.labelsize": plot_utils.TICK_SIZE,
        "ytick.labelsize": plot_utils.TICK_SIZE,
    }
    with plt.style.context(mapping):
        _main()


if __name__ == "__main__":
    plot_utils.set_styles()
    main()
