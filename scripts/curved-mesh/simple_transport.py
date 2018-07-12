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

import matplotlib.pyplot as plt
import numpy as np

import plot_utils


def main():
    # u_t + u_x = 0
    # u(x, t) = u(x - t, 0)
    min_x = -1.5
    max_x = 3.5
    x_vals = np.linspace(min_x, max_x, 2048 + 1)
    # u(x, 0) = x^3
    u0 = x_vals * x_vals * x_vals
    u2 = (x_vals - 2.0) * (x_vals - 2.0) * (x_vals - 2.0)

    figure = plt.figure()
    ax = figure.gca()

    ax.plot(x_vals, u0, label="$u(x, 0)$")
    ax.plot(x_vals, u2, label="$u(x, 2)$")
    ax.annotate(
        "",
        xy=(2.875, 0.75),
        xytext=(1.0, 0.75),
        arrowprops={"arrowstyle": "->", "linewidth": 2.0},
    )

    ax.legend(loc="upper left", fontsize=12)
    ax.axis("scaled")
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("$x$", fontsize=16)
    ax.set_ylabel("$u$", rotation=0, fontsize=16)

    filename = "simple_transport.pdf"
    path = plot_utils.get_path("curved-mesh", filename)
    figure.savefig(path, bbox_inches="tight")
    print("Saved {}".format(filename))
    plt.close(figure)


if __name__ == "__main__":
    plot_utils.set_styles()
    main()
