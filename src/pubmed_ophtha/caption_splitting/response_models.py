"""Module defining response models for caption splitting functionality."""

import pydantic


class SubCaptionName(pydantic.BaseModel):
    """Model the name of a sub caption."""

    name: str


class SubCaption(pydantic.BaseModel):
    """Model a sub caption."""

    text: str


class SubCaptionNames(pydantic.BaseModel):
    """Model the list of sub caption names."""

    names: list[SubCaptionName]


class SplitSubCaptions(pydantic.BaseModel):
    """Model the resulting split sub captions."""

    sub_captions: dict[str, SubCaption]
