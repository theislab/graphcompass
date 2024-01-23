"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Graph-COMPASS."""


if __name__ == "__main__":
    main(prog_name="graphcompass")  # pragma: no cover
