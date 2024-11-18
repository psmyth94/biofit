import os

from biofit.integration.R.r_caller import (
    R_PLOTTING_DEPENDENCIES,
    R_PREPROCESSING_DEPENDENCIES,
    RCaller,
)
from biofit.utils import logging

logger = logging.get_logger(__name__)


def add_to_parser(parser):
    # Register the 'install' subcommand
    install_r_parser = parser.add_parser(
        "install",
        help="Install R dependencies",
        description="Install various dependencies including R and its packages.",
    )

    install_r_parser.add_argument(
        "--all",
        action="store_true",
        help="Install all R dependencies for plotting and preprocessing",
    )
    install_r_parser.add_argument(
        "--plotting",
        action="store_true",
        help="Install R plotting dependencies",
    )
    install_r_parser.add_argument(
        "--preprocessing",
        action="store_true",
        help="Install R preprocessing dependencies",
    )
    install_r_parser.add_argument(
        "--cran",
        nargs="+",
        help="Install specific CRAN packages",
    )
    install_r_parser.add_argument(
        "--bioconductor",
        nargs="+",
        help="Install specific Bioconductor packages",
    )
    install_r_parser.add_argument(
        "--binary",
        action="store_true",
        help="Install with binaries only",
    )

    install_r_parser.add_argument(
        "--r-home",
        "-r",
        type=str,
        help="Path to the R installation directory which contains the library folder",
    )
    return parser


def run(args):
    if args.r_home:
        os.environ["R_HOME"] = args.r_home
    cran_deps = args.cran or []
    bioconductor_deps = args.bioconductor or []
    params = {}
    if args.binary:
        params["pkgType"] = "binary"
    if args.all or args.plotting:
        cran_deps += R_PLOTTING_DEPENDENCIES["cran"]
        bioconductor_deps += R_PLOTTING_DEPENDENCIES["bioconductor"]
    if args.all or args.preprocessing:
        cran_deps += R_PREPROCESSING_DEPENDENCIES["cran"]
        bioconductor_deps += R_PREPROCESSING_DEPENDENCIES["bioconductor"]

    if cran_deps:
        logger.info(f"Checking for CRAN packages: {', '.join(cran_deps)}")
        RCaller.verify_r_dependencies(
            cran_dependencies=cran_deps,
            bioconductor_dependencies=[],
            install_missing=True,
        )
        logger.info("CRAN dependencies installed successfully.")

    if bioconductor_deps:
        logger.info(f"Checking Bioconductor packages: {', '.join(bioconductor_deps)}")
        RCaller.verify_r_dependencies(
            cran_dependencies=[],
            bioconductor_dependencies=bioconductor_deps,
            install_missing=True,
        )
        logger.info("Bioconductor dependencies installed successfully.")

    if not cran_deps and not bioconductor_deps:
        logger.info("No R dependencies to install.")
