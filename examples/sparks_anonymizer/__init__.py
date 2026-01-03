"""
SparksAnonymizer - A Python package for anonymizing dataframes for LLM input.

This package allows users to take a dataframe and anonymize it for input into LLMs
without fear of data restrictions.
"""

__version__ = "0.1.0"
__author__ = "SkrapsMD"
__email__ = ""

# Main package imports will go here
# sparks_anonymizer/__init__.py

from .anonymizer import SparksAnonymizer, AnonymizeConfig
from .deanonymizer_helper import (
    rename_to_anonymized,
    rename_to_original,
)

__all__ = [
    "SparksAnonymizer",
    "AnonymizeConfig",
    "rename_to_anonymized",
    "rename_to_original",
]

__version__ = "0.1.0"

def anonymize_df(df, *, seed=0, **config_kwargs):
    """
    Convenience wrapper:
      anonymize_df(df, seed=123, string_strategy="stable_pseudonym")
    """
    cfg = AnonymizeConfig(seed=seed, **config_kwargs)
    return SparksAnonymizer(cfg).anonymize(df)
