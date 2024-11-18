import argparse
import sys

import biofit.cli.install_r as install_r
from biofit.utils import logging

logger = logging.get_logger(__name__)


def create_parser():
    parser = argparse.ArgumentParser(
        description="BIOFIT: General-purpose Omics Machine Learning framework"
    )

    # add a quiet option
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress all logging output except for errors",
    )
    subparsers = parser.add_subparsers(dest="subcommand")
    subparsers.required = True

    return parser, subparsers


def main():
    parser, subparsers = create_parser()
    subparsers = install_r.add_to_parser(subparsers)
    args = parser.parse_args()

    if args.quiet:
        logging.set_verbosity(logging.ERROR)
    else:
        logging.set_verbosity(logging.INFO)

    try:
        if args.subcommand == "install":
            install_r.run(args)
        else:
            parser.print_help()
            sys.exit(1)
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
