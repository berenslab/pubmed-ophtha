"""
Package for extracting and processing fundus images from biomedical literature.

This package loads packages from PubMed Central Open Access (PMC-OA),
extracts relevant figures, and splits them into panels for further analysis.
"""

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
