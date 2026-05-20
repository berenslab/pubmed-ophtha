"""
Module for processing the ground truth labels for the PubMed Ophtha dataset.

The purpose of the module is to convert the ground truth labels into the PubMed-Ophtha
format.
Uses the ground-truth annotations from Label Studio.
For annotations that contain the full information, the samples are simply converted.
For all other samples with missing information, the module follows the steps in the
original processing pipeline while retaining the ground-truth annotations.

"""
