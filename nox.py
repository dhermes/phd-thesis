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

"""This is a configuration file for running ``nox`` on this project.

To determine the supported actions run ``nox --list-sessions`` from the
project root.
"""

import os

import nox
import py.path


NOX_DIR = os.path.abspath(os.path.dirname(__file__))
SINGLE_INTERP = "python3.6"


def get_path(*names):
    return os.path.join(NOX_DIR, *names)


class Remove(object):
    def __init__(self, prefix, extensions):
        self.prefix = prefix
        self.extensions = extensions

    def __call__(self):
        for extension in self.extensions:
            path = "{}.{}".format(self.prefix, extension)
            os.remove(path)


def build_tex_file(
    session, base, new_id, extensions=(), with_bibtex=False, use_xelatex=False
):
    # NOTE: This assumes that ``session.chdir(get_path('doc'))``
    #       has been called.
    modify_id = get_path("scripts", "modify_pdf_id.py")

    if use_xelatex:
        session.run("xelatex", base)
        session.run("xelatex", base)
    elif with_bibtex:
        session.run("pdflatex", base)
        session.run("bibtex", base)
        session.run("pdflatex", base)
        session.run("bibtex", base)
        session.run("pdflatex", base)
        session.run("pdflatex", base)
    else:
        session.run("pdflatex", base)
        session.run("pdflatex", base)
        session.run("pdflatex", base)

    path = get_path("doc", base)
    remove = Remove(path, extensions)
    session.run(remove)

    if not use_xelatex:
        session.run("python", modify_id, "--base", path, "--id", new_id)


@nox.session
def build_tex(session):
    session.interpreter = SINGLE_INTERP

    if py.path.local.sysfind("pdflatex") is None:
        session.skip("`pdflatex` must be installed")

    if py.path.local.sysfind("xelatex") is None:
        session.skip("`xelatex` must be installed")

    if py.path.local.sysfind("bibtex") is None:
        session.skip("`bibtex` must be installed")

    # No need to create a virtualenv.
    session.virtualenv = False

    session.chdir(get_path("doc"))

    build_tex_file(
        session,
        "thesis",
        "55008F4EDC13ADFCEFED89CA0A359ACD",
        extensions=("aux", "bbl", "blg", "lof", "log", "lot", "out", "toc"),
        with_bibtex=True,
    )

    build_tex_file(
        session,
        "approval_page",
        "73E1E12D3FFF1C2BBD94B849733BF55A",
        extensions=("aux", "log", "out"),
    )

    extras = (
        "abstract",
        "algorithms",
        "bezier-intersection",
        "compensated-newton",
        "conclusion",
        "introduction",
        "k-compensated",
        "metadata",
        "preliminaries",
        "proofs",
        "solution-transfer",
    )
    for extra in extras:
        session.run(Remove(extra, ("aux",)))

    build_tex_file(
        session,
        "tikz_local_err",
        "61C99C3315FAE74F6F2E4EEAB3E4D3AA",
        extensions=("aux", "log", "out"),
    )

    build_tex_file(
        session,
        "tikz_filtration",
        "5AD16E27EBA16C57CF11C93F5CE4D079",
        extensions=("aux", "log", "out"),
    )

    build_tex_file(
        session,
        "tikz_shape_fns1",
        "B5234F7B23999E2560401FE20167B7C8",
        extensions=("aux", "log", "out"),
    )

    build_tex_file(
        session,
        "tikz_shape_fns2",
        "8C88A92F10A23D28C060D88CF0CE94A0",
        extensions=("aux", "log", "out"),
    )

    build_tex_file(
        session,
        "thesis_talk",
        "6A945FD7D33437399D0EB8EC77533E6C",
        extensions=("aux", "log", "nav", "out", "snm", "toc"),
        use_xelatex=True,
    )


@nox.session
def make_images(session):
    session.interpreter = SINGLE_INTERP
    # Install all dependencies.
    session.install("--requirement", "make-images-requirements.txt")
    # Run the script(s).
    # Make sure
    # - Custom ``matplotlibrc`` is used
    # - Code in ``src/`` is importable
    # - PDFs have deterministic ``CreationDate``
    env = {
        "MATPLOTLIBRC": get_path("images"),
        "PYTHONPATH": get_path("src"),
        "SOURCE_DATE_EPOCH": "0",
    }
    script_paths = (
        ("bezier-intersection", "locate_in_triangle.py"),
        ("bezier-intersection", "subdivision.py"),
        ("compensated-newton", "almost_tangent.py"),
        ("compensated-newton", "jghplus13.py"),
        ("compensated-newton", "newton_de_casteljau.py"),
        ("compensated-newton", "root_plots.py"),
        ("compensated-newton", "tangent_intersection.py"),
        ("k-compensated", "compensated_insufficient.py"),
        ("k-compensated", "error_against_cond.py"),
        ("k-compensated", "horner_inferior.py"),
        ("k-compensated", "smooth_drawing.py"),
        ("preliminaries", "inverted_element.py"),
        ("slides", "curved_vs_straight.py"),
        ("slides", "distort.py"),
        ("slides", "element_distortion.py"),
        ("slides", "error_against_cond.py"),
        ("slides", "inverted_element.py"),
        ("slides", "newton_de_casteljau.py"),
        ("slides", "polygon_vs_curved.py"),
        ("slides", "tangent_intersection.py"),
        ("solution-transfer", "distort.py"),
        ("solution-transfer", "polygon_vs_curved.py"),
        ("solution-transfer", "simple_transport.py"),
    )
    for segments in script_paths:
        script = get_path("scripts", *segments)
        session.run("python", script, env=env)


@nox.session
def update_requirements(session):
    session.interpreter = SINGLE_INTERP

    if py.path.local.sysfind("git") is None:
        session.skip("`git` must be installed")

    # Install all dependencies.
    session.install("pip-tools")

    # Update all of the requirements file(s).
    names = ("make-images",)
    for name in names:
        in_name = "{}-requirements.in".format(name)
        txt_name = "{}-requirements.txt".format(name)
        session.run(
            "pip-compile", "--upgrade", "--output-file", txt_name, in_name
        )
        session.run("git", "add", txt_name)
