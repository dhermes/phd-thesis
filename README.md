# UC Berkeley PhD Thesis

This is the repository for my thesis. If you'd just like to read the
[paper][1], feel free.

This repository is laid out in a manner described in
[Good Enough Practices in Scientific Computing][2].

## Installation

The code used to build the manuscript, generate images and verify
computations is written in Python. To run the code, Python 3.6
should be installed, along with ``nox-automation``:

```
python -m pip install --upgrade nox-automation
```

Once installed, the various build jobs can be listed. For example:

```
$ nox --list-sessions
Available sessions:
* build_tex
* make_images
* update_requirements
```

To run ``nox -s build_tex`` (i.e. to build the PDF), ``pdflatex`` and
``bibtex`` are required.

[1]: doc/thesis.pdf
[2]: https://arxiv.org/pdf/1609.00037.pdf
