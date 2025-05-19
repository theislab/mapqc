# mapQC

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/theislab/mapqc/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/mapqc

A metric for the evaluation of single-cell query-to-reference mappings

## Getting started

Please refer to the [documentation](https://mapqc.readthedocs.io/), in particular, the [API documentation](https://mapqc.readthedocs.io/en/latest/api/index.html). A few notes on how and when to use MapQC.

### What does mapQC do?

MapQC evaluates the quality of an exisiting query-to-reference mapping, by quantifying the distance between query and reference samples. Rather than using a standard metric to calculate this distance, it compares the expected inter-sample distance (based on controls in the references) to the observed distance between query samples and reference samples, and outputs a normalized distance called the mapQC score for every query cell. MapQC calculates inter-sample distances in a local, per-neighborhood manner. (For more methodological details, check out the preprint.) <br>
The reference is expected to cover most of the diversity existing in the control population (e.g. young and old, low and high BMI, smokers and non-smokers, different ethnicities, etc. for human data), such that the controls in the query are expected to look similar to some of the samples in the reference. Therefore, mapQC works best if the reference is a large-scale reference including data from many individuals. Moreover, the query needs to includes control samples, such that we know for a subset of query samples how well they should integrate with the reference.<br>
MapQC scores can be regarded as a Z-scored distance to the reference score (Z-scored based on the inter-sample distances in the reference itself), such that a mapQC score of 2 for a given query cell represents a distance to the reference of two standard deviations above the expected distance (based on the reference). Therefore, mapQC scores > 2 are considered high, and indicate either remaining batch effects (if seen in control samples) or disease-specific cell states (if seen in case but not in control samples).

### What are the data requirements for using mapQC?

In short, you need one AnnData object, including:
- A large scale reference, including only its healthy/control cells.
- A mapped query dataset, with healthy/control cells (must-have) and case/perturbed cells (if you have them).
- Metadata (see below)
- A mapping-derived embedding of both the reference and the query

Below, the exact requirements are outlined in more detail.

1. **Reference**: MapQC is meant to evaluate the mapping of a given dataset to an existing, large-scale reference. It assumes the reference more or less covers the diversity of the control population (e.g. diversity among healthy individuals for the case of human data, or of unperturbed organoids generated with a wide array of protocols for an organoid dataset). Therefore, a mapping of a single dataset to another single dataset is likely to not fulfill these assumptions, and mapQC is not guaranteed to work well. MapQC runs on a scanpy AnnData object, that includes the *control* cells from the reference (i.e. no perturbed or diseased cells!) and *no perturbed/diseased/etc.* cells in the reference. Make sure to exclude these before running mapQC.

2. **Query**: The query (the dataset mapped to the reference) is expected to have both control and case samples. MapQC can also be run without case samples in the query, but it should always include controls. The query cells should be in the same AnnData object as the reference.

3. **Required Metadata**: Several metadata columns need to be present in your adata.obs:

   The following need a value for every cell from both the reference and the query. Column names can be set as wanted:
   - A "study" key, specifying from which study/dataset each cell in the reference and query came. The query is assumed to come from a single study. If the query includes multiple studies, map these separately and run mapQC on each of them separately.
   - A "sample" key, specifying from which biological sample a given cell came.
   - A reference versus query key, specifying for each cell whether it is from the reference or the query.

   And optionally:
   - A grouping of all your cells, e.g. a clustering run on your mapping embedding. If this is provided, mapQC will sample cells proportional to those groups instead of taking randomly sampled cells to choose its neighborhood sample cells. Providing a grouping might help better covering the full embedding space (especially helpful for rare cell types) when running mapQC.

   And for the query:
   - A "condition" key, specifying for the query what condition (case/control etc.) each cell belongs to, e.g. the disease of the patient from which the sample came or if it was a control.

4. **Embedding Data**: Your adata object needs to include the mapped embedding, including coordinates for both the reference and the query. These can be stored either in adata.X or in adata.obsm.

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

For questions and help requests, submit an issue on the mapQC GitHub page.

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
