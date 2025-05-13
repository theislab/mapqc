# mapqc

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/theislab/mapqc/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/mapqc

A metric for the evaluation of single-cell query-to-reference mappings

## Getting started

Please refer to the [documentation][], in particular, the [API documentation][]. A few notes on how and when to use MapQC.

### What are the data requirements to use mapQC?

1. **Reference**: MapQC is meant to evaluate the mapping of a given dataset to an existing, large-scale reference. It assumes the reference more or less covers the diversity of the control population (e.g. diversity among healthy individuals for the case of human data, or of unperturbed organoids generated with a wide array of protocols for an organoid dataset). Therefore, a mapping of a single dataset to another single dataset is likely to not fulfill these assumptions, and mapQC is not guaranteed to work well.

2. **Query**: The query is expected to have both control and case samples in the query. It can also be run without case samples, but should always include controls.

3. **Reference Data**: MapQC runs on a scanpy AnnData object, that includes the *control* cells from the reference (i.e. no perturbed or diseased cells!) and *no perturbed/diseased/etc.* cells in the reference. Make sure to exclude these before running mapQC.

4. **Query Data**: For the query, include both control and case/perturbed samples. The query cells should be in the same AnnData object as the reference.

5. **Required Metadata**: Several metadata columns need to be present in your adata.obs:

   The following need a value for every cell from both the reference and the query. Column names can be set as wanted:
   - A "study" key, specifying from which study/dataset each cell in the reference and query came. The query is assumed to come from a single study. If the query includes multiple studies, map these separately and run mapQC on each of them separately.
   - A "sample" key, specifying from which biological sample a given cell came.
   - A reference versus query key, specifying for each cell whether it is from the reference or the query.

   And optionally:
   - A grouping of all your cells, e.g. a clustering run on your mapping embedding. If this is provided, mapQC will sample cells proportional to those groups instead of taking randomly sampled cells to choose its neighborhood sample cells. Providing a grouping might help better covering the full embedding space (especially helpful for rare cell types) when running mapQC.

   And for the query:
   - A "condition" key, specifying for the query what condition (case/control etc.) each cell belongs to, e.g. the disease of the patient from which the sample came or if it was a control.

6. **Embedding Data**: Your adata object needs to include the mapped embedding, including coordinates for both the reference and the query. These can be stored either in adata.X or in adata.obsm.

## Installation

You need to have Python 3.10 or newer installed on your system.
If you don't have Python installed, we recommend installing [Mambaforge][].

There are several alternative options to install mapqc:

<!--
1) Install the latest release of `mapqc` from [PyPI][]:

```bash
pip install mapqc
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/theislab/mapqc.git@main
```

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

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
