"""MapQC package for quality control of spatial transcriptomics data."""

from importlib.metadata import version

from . import pl
from .evaluate import evaluate
from .run import run_mapqc

__version__ = version("mapqc")
__all__ = ["run_mapqc", "evaluate", "pl"]
