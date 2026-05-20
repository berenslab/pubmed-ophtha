"""Defines the CLI of the project."""

import click

from pubmed_ophtha.logging_config import setup_logging

setup_logging()

import pubmed_ophtha.aggregation.cli  # noqa: E402
import pubmed_ophtha.caption_splitting.cli  # noqa: E402
import pubmed_ophtha.figure_splitting.cli  # noqa: E402
import pubmed_ophtha.figure_splitting.detectron.cli  # noqa: E402
import pubmed_ophtha.fill_null.cli  # noqa: E402
import pubmed_ophtha.filtering.cli  # noqa: E402
import pubmed_ophtha.panel_assembly.cli  # noqa: E402
import pubmed_ophtha.pipeline  # noqa: E402


@click.group()
def main():
    """CLI entry point."""
    pass


# ── dataset ───────────────────────────────────────────────────────────────────
@click.group()
def dataset():
    """Dataset utilities."""
    pass


dataset.add_command(pubmed_ophtha.fill_null.cli.cli.commands["fill"], name="fill-null")
dataset.add_command(
    pubmed_ophtha.figure_splitting.cli.cli.commands["pull-models"], name="pull-models"
)
main.add_command(dataset)


# ── pipeline ──────────────────────────────────────────────────────────────────
@click.group()
def stages():
    """Run individual pipeline stages."""
    pass


stages.add_command(pubmed_ophtha.filtering.cli.cli, name="filtering")
stages.add_command(pubmed_ophtha.figure_splitting.cli.cli, name="figure-splitting")
stages.add_command(pubmed_ophtha.caption_splitting.cli.cli, name="caption-splitting")
stages.add_command(pubmed_ophtha.panel_assembly.cli.cli, name="panel-assembly")
stages.add_command(pubmed_ophtha.aggregation.cli.cli, name="aggregation")

pubmed_ophtha.pipeline.cli.add_command(stages)
main.add_command(pubmed_ophtha.pipeline.cli, name="pipeline")


# ── train ─────────────────────────────────────────────────────────────────────
@click.group()
def train():
    """Model training commands."""
    pass


_detectron = pubmed_ophtha.figure_splitting.detectron.cli
train.add_command(_detectron.train_detectron, name="detectron")
train.add_command(_detectron.train_annotation_classifier, name="mark-status-classifier")
main.add_command(train)


if __name__ == "__main__":
    main()
