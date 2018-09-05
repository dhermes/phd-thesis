# UC Berkeley PhD Thesis

This is the repository for my thesis. If you'd just like to read the
[paper][1], feel free.

This repository is laid out in a manner described in
[Good Enough Practices in Scientific Computing][2].

The content itself has been broken into a few standalone papers
and uploaded to the arXiv and / or submitted to journals:

- [``K``-Compensated de Casteljau][3] paper (GitHub [repo][4],
  submitted to [AMC][9])
- [2-Norm Condition Number for B&#xe9;zier Curve Intersection][5]
  (GitHub [repo][6])
- [A Curious Case of Curbed Condition][7] (GitHub [repo][8])
- High-order Solution Transfer between Curved Meshes (GitHub [repo][10]);
  yet to be released / submitted

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

To run ``nox -s build_tex`` (i.e. to build the PDFs), ``pdflatex``,
``xelatex`` and ``bibtex`` are required. In addition the ``metropolis``
Beamer theme should be installed, as well as the Fira font family.

[1]: doc/thesis.pdf
[2]: https://arxiv.org/abs/1609.00037
[3]: https://arxiv.org/abs/1808.10387
[4]: https://github.com/dhermes/k-compensated-de-casteljau/
[5]: https://arxiv.org/abs/1808.06126
[6]: https://github.com/dhermes/condition-number-bezier-curve-intersection
[7]: https://arxiv.org/abs/1806.05145
[8]: https://github.com/dhermes/curious-case-curbed-condition
[9]: https://www.journals.elsevier.com/applied-mathematics-and-computation
[10]: https://github.com/dhermes/solution-transfer-curved-meshes
