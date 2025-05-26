# mapQC

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/theislab/mapqc/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/mapqc

A metric for the evaluation of single-cell query-to-reference mappings

## Getting started

Please refer to the [documentation](https://mapqc.readthedocs.io/), in particular, the [API documentation](https://mapqc.readthedocs.io/en/latest/api/index.html) for detailed package documentation. For reproduction of the results in the paper, check out the [mapQC reproducibility repository](https://github.com/theislab/mapqc_reproducibility). <br><br>
Below a few notes on how and when to use MapQC:

### What does mapQC do?

MapQC evaluates the quality of a query-to-reference mapping, and outputs a cell-level mapQC score for every query cell. MapQC scores higher than 2 indicate a large distance of the query cell to the reference. Given a healthy/control reference, we expect query controls to have low mapQC scores, and query case/disease cells to have higher mapQC scores in the case of case-specific cellular phenotypes. You can thus use mapQC scores to assess, in a quantitative manner, if your mapping was successful.

![Overview of mapQC workflow](https://raw.githubusercontent.com/theislab/mapqc/main/docs/_static/images/mapQC_concept_figure.png)
<p style="margin-top: 0;"><small><i>Overview of mapQC's workflow</i></small></p>

### What are the data requirements for using mapQC?

In short, you need one AnnData object, including:
- A large scale reference, including only its healthy/control cells.
- A mapped query dataset, with healthy/control cells (must-have) and case/perturbed cells (if you have them).
- Metadata (query/reference status, study, sample, and optionally clustering and cell type annotations)
- A mapping-derived embedding including both the reference and the query

In the [quick-start tutorial notebook](./docs/notebooks/mapqc_quickstart.ipynb) we provide a more extensive description of the exact data requirements.


## Installation

You need to have Python 3.10 or newer installed on your system.

There are several alternative options to install mapqc:

1) Install the latest release of `mapqc` from [PyPI][]:

```bash
pip install mapqc
```

1. Install the latest development version:

```bash
pip install git+https://github.com/theislab/mapqc.git@main
```

## Release notes

See the [changelog][].

## Contact

I am happy to hear any comments, suggestions, or even bugs that you run into. I would like to make this package run as smoothly as possible! So for any of these, submit an issue on the mapQC GitHub page and I will be glad to help.

## Citation

> t.b.a

[mambaforge]: https://github.com/conda-forge/miniforge#mambaforge
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/theislab/mapqc/issues
[tests]: https://github.com/theislab/mapqc/actions/workflows/test.yml
[documentation]: https://mapqc.readthedocs.io
[changelog]: https://mapqc.readthedocs.io/en/latest/changelog.html
[api documentation]: https://mapqc.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/mapqc
