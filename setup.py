from setuptools import find_packages, setup

REQUIRED_PKGS = [
    # the library that biofit is built upon
    "biocore>=1.1.1",
    # For file locking
    "filelock",
    # We use numpy>=1.17 to have np.random.Generator (Dataset shuffling)
    "numpy>=1.17",
    # Backend and serialization.
    # Minimum 8.0.0 to be able to use .to_reader()
    "pyarrow>=8.0.0",
    # As long as we allow pyarrow < 14.0.1, to fix vulnerability CVE-2023-47248
    "pyarrow-hotfix",
    # For smart caching dataset processing
    "dill>=0.3.0,<0.3.8",  # tmp pin until dill has official support for determinism see https://github.com/uqfoundation/dill/issues/19
    # For performance gains with apache arrow
    "pandas",
    # for downloading datasets over HTTPS
    "requests>=2.19.0",
    # progress bars in download and scripts
    "tqdm>=4.62.1",
    # for fast hashing
    "xxhash",
    # for better multiprocessing
    "multiprocess",
    # to save datasets locally or on any filesystem
    # minimum 2023.1.0 to support protocol=kwargs in fsspec's `open`, `get_fs_token_paths`, etc.: see https://github.com/fsspec/filesystem_spec/pull/1143
    "fsspec[http]>=2023.1.0,<=2023.10.0",
    # Utilities from PyPA to e.g., compare versions
    "packaging",
    # To parse YAML metadata from dataset cards
    "pyyaml>=5.1",
    # for processing and transforming datasets
    "scikit-learn",
]

QUALITY_REQUIRE = ["ruff>=0.1.5"]

DOCS_REQUIRE = [
    # Might need to add doc-builder and some specific deps in the future
    "s3fs",
]

VISUALIZATION_REQUIRE = [
    "matplotlib",
    "seaborn",
    "psutil",  # for the start time of the biofit run
]


ML_REQUIRE = [
    "polars>=0.20.5",
    "timezones>=0.10.2",
    "optuna",
    "lightgbm",
    # "xgboost",
    # "catboost",
    "imbalanced-learn",
]

TESTS_REQUIRE = ["pytest", "pytest-timeout", "pytest-xdist"]


EXTRAS_REQUIRE = {
    "polars": ["polars>=0.20.5", "timezones>=0.10.2"],
    "rpy2": ["rpy2>=3.5.15", "rpy2-arrow>=0.0.8"],
    "ml": ML_REQUIRE,
    "apache-beam": ["apache-beam>=2.26.0,<2.44.0"],
    "vcf": ["cyvcf2>=0.30.0", "sgkit>=0.0.1"],
    "tensorflow": [
        "tensorflow>=2.2.0,!=2.6.0,!=2.6.1; sys_platform != 'darwin' or platform_machine != 'arm64'",
        "tensorflow-macos; sys_platform == 'darwin' and platform_machine == 'arm64'",
    ],
    "tensorflow_gpu": ["tensorflow-gpu>=2.2.0,!=2.6.0,!=2.6.1"],
    "torch": ["torch"],
    "jax": ["jax>=0.3.14", "jaxlib>=0.3.14"],
    "s3": ["s3fs"],
    "viz": VISUALIZATION_REQUIRE,
    "test": QUALITY_REQUIRE
    + TESTS_REQUIRE
    + DOCS_REQUIRE
    + VISUALIZATION_REQUIRE
    + ML_REQUIRE,
    "all": VISUALIZATION_REQUIRE + ML_REQUIRE + QUALITY_REQUIRE + DOCS_REQUIRE,
    "quality": QUALITY_REQUIRE,
    "docs": DOCS_REQUIRE,
}

setup(
    name="biofit",
    version="0.0.1",  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
    description="BioFit: Bioinformatics Machine Learning Framework",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Patrick Smyth",
    author_email="psmyth1994@gmail.com",
    url="https://github.com/psmyth94/biofit",
    download_url="https://github.com/psmyth94/biofit/tags",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    python_requires=">=3.8.0,<3.12.0",
    install_requires=REQUIRED_PKGS,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": [
            "biofit = biofit.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords="omics machine learning bioinformatics metrics",
    zip_safe=False,  # Required for mypy to find the py.typed file
)
