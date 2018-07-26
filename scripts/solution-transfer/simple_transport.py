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

    ax.plot(x_vals, u0, label="$u(x, 0)$", color=plot_utils.BLUE)
    ax.plot(x_vals, u2, label="$u(x, 2)$", color=plot_utils.GREEN)
    ax.annotate(
        "",
        xy=(2.875, 0.75),
        xytext=(1.0, 0.75),
        arrowprops={"arrowstyle": "->", "linewidth": 2.0},
    )

    ax.legend(loc="upper left", fontsize=plot_utils.TEXT_SIZE)
    ax.axis("scaled")
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("$x$", fontsize=plot_utils.TEXT_SIZE)
    ax.set_ylabel("$u$", rotation=0, fontsize=plot_utils.TEXT_SIZE)
    ax.xaxis.set_tick_params(labelsize=plot_utils.TICK_SIZE)
    ax.yaxis.set_tick_params(labelsize=plot_utils.TICK_SIZE)

    figure.set_size_inches(4.8, 2.4)
    figure.subplots_adjust(
        left=0.12, bottom=0.1, right=0.99, top=1.04, wspace=0.2, hspace=0.2
    )
    filename = "simple_transport.pdf"
    path = plot_utils.get_path("solution-transfer", filename)
    figure.savefig(path)
    print("Saved {}".format(filename))
    plt.close(figure)


if __name__ == "__main__":
    plot_utils.set_styles()
    main()
