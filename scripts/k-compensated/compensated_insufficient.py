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

"""Show an example where compensated de Casteljau is still not enough."""

import fractions

import matplotlib.pyplot as plt
import numpy as np

import de_casteljau
import plot_utils


F = fractions.Fraction
# p(s) = (2s - 1)^3 (s - 1)
BEZIER_COEFFS = (1.0, -0.75, 0.5, -0.25, 0.0)
ROOT = 0.5
DELTA_S = 1.5e-11
NUM_POINTS = 401


def main(filename=None):
    s_vals = np.linspace(ROOT - DELTA_S, ROOT + DELTA_S, NUM_POINTS)

    de_casteljau2 = []
    exact = []

    for s in s_vals:
        de_casteljau2.append(de_casteljau.compensated(s, BEZIER_COEFFS))

        exact_s = F(s)
        exact_p = (2 * exact_s - 1) ** 3 * (exact_s - 1)
        exact.append(float(exact_p))

    figure, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    ax1.plot(s_vals, de_casteljau2)
    ax2.plot(s_vals, exact)

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

    ax1.set_title(r"$\mathtt{CompDeCasteljau}$")
    ax2.set_title("$p(s)$")

    if filename is None:
        plt.show()
    else:
        # NOTE: These are (intended to be) the same settings used in
        #       ``horner_inferior.py``, so they should probably be
        #       kept in sync.
        figure.set_size_inches(9.87, 4.8)
        figure.subplots_adjust(
            left=0.06,
            bottom=0.12,
            right=0.97,
            top=0.92,
            wspace=0.13,
            hspace=0.20,
        )
        path = plot_utils.get_path("k-compensated", filename)
        figure.savefig(path)
        print("Saved {}".format(filename))
        plt.close(figure)


if __name__ == "__main__":
    plot_utils.set_styles()
    main(filename="compensated_insufficient.pdf")
