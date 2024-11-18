<p align="center">
    $${\Huge{\textbf{\textsf{\color{#2E8B57}Bio\color{red}fit}}}}$$
    <br/>
    <br/>
</p>
<p align="center">
    <a href="https://github.com/psmyth94/biofit/actions/workflows/ci_cd_pipeline.yml?query=branch%3Amain"><img alt="Build" src="https://github.com/psmyth94/biofit/actions/workflows/ci_cd_pipeline.yml/badge.svg?branch=main"></a>
    <a href="https://github.com/psmyth94/biofit/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/psmyth94/biofit.svg?color=blue"></a>
    <a href="https://github.com/psmyth94/biofit/tree/main/docs"><img alt="Documentation" src="https://img.shields.io/website/http/github/psmyth94/biofit/tree/main/docs.svg?down_color=red&down_message=offline&up_message=online"></a>
    <a href="https://github.com/psmyth94/biofit/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/psmyth94/biofit.svg"></a>
    <a href="CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg"></a>
    <!-- <a href="https://zenodo.org/records/14028772"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.14028772.svg" alt="DOI"></a> -->
</p>

**Biofit** is a machine learning library designed for bioinformatics datasets. It
provides tools for transforming, extracting, training, and evaluating machine learning
models on biomedical data. It also provides automatic data preprocessing, visualization,
and configurable processing pipelines. Here are some of the main features of Biofit:

- **Automatic Data Preprocessing:** Automatically preprocess biomedical datasets using
  built-in preprocessing steps.
- **Automatic Visualization:** Automatically visualize data using built-in visualization
  methods geared towards biomedical data.
- **Configurable Processing Pipelines:** Define and customize data processing pipelines.
- **Data Handling Flexibility:** Support for a wide range of data formats, including:
  - [Pandas](https://github.com/pandas-dev/pandas)
  - [Polars](https://github.com/pola-rs/polars)
  - [NumPy](https://github.com/numpy/numpy)
  - [CSR (SciPy)](https://github.com/scipy/scipy)
  - [Arrow](https://github.com/apache/arrow)
  - 🤗 [Datasets](https://github.com/huggingface/datasets)
  - [Biosets](https://github.com/psmyth94/biosets)
- **Machine Learning Models:** Supports a wide range of machine learning models, including:
  - [Scikit-learn](https://github.com/scikit-learn/scikit-learn)
    - [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
    - [Support Vector Machine](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
    - [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
    - [Lasso Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
  - [LightGBM](https://github.com/microsoft/LightGBM)
  - More to come!
- **Caching and Reuse:** Caches intermediate results using Apache Arrow for efficient reuse.
- **Batch Processing and Multiprocessing:** Utilize batch processing and multiprocessing for efficient handling of large-scale data.

## Installation

You can install Biofit via pip:

```bash
pip install biofit
```

## Quick Start

### Preprocessing Data

Biofit provides preprocessing capabilities tailored for omics data. You can use
built-in classes to load preprocessing steps based on the experiment type or create
custom preprocessing pipelines. The preprocessing pipeline in Biofit uses a syntax
similar to sklearn and supports distributed processing.

#### Using a Preprocessor

Biofit allows you to fit and transform your data in a few lines, similar to sklearn.
For example, you can use the LogTransformer to apply a log transformation to your data:

```python
from biofit.preprocessing import LogTransformer
import pandas as pd

dataset = pd.DataFrame({"feature1": [1, 2, 3, 4, 5]})
log_transformer = LogTransformer()
preprocessed_data = log_transformer.fit_transform(dataset)
# Applying log transformation: 100%|█████████████████████████████| 5/5 [00:00<00:00, 7656.63 examples/s]
print(preprocessed_data)
#    feature1
# 0  0.000000
# 1  0.693147
# 2  1.098612
# 3  1.386294
# 4  1.609438
```

#### Auto Preprocessing

You can automatically apply standard preprocessing steps by specifying the experiment
type. This allows you to load tailored preprocessing steps for the type of data you are
working with, such as "otu", "asv", "snp", or "maldi":

```python
from biofit.preprocessing import AutoPreprocessor

preprocessor = AutoPreprocessor.for_experiment("snp", [{"min_prevalence": 0.1}, None])
print(preprocessor)
# [('min_prevalence_row', MinPrevalencFilter(min_prevalence=0.1)),
#  ('min_prevalence', MinPrevalenceFeatureSelector(min_prevalence=0.01))]

# Fit and transform the dataset using the preprocessor
preprocessed_data = preprocessor.fit_transform(dataset)
```

Biofit is made with biofit in mind. You can pass the loaded dataset instead of a string
to load the preprocessors:

```python
from biofit import load_dataset

dataset = load_dataset("csv", data_files="my_file.csv", experiment_type="snp")

preprocessor = AutoPreprocessor.for_experiment(dataset)
print(preprocessor)
# [('min_prevalence_row', MinPrevalencFilter(min_prevalence=0.01)),
#  ('min_prevalence', MinPrevalenceFeatureSelector(min_prevalence=0.01))]
preprocessed_data = preprocessor.fit_transform(dataset)
```

#### Custom Preprocessing Pipeline

Biofit allows you to create custom preprocessing pipelines using the
`PreprocessorPipeline` class. This allows chaining multiple preprocessing steps from
`sklearn` and Biofit in a single operation:

```python
from biofit import load_dataset
from biofit.preprocessing import LogTransformer, PreprocessorPipeline
from sklearn.preprocessing import StandardScaler

# Load the dataset
dataset = load_dataset("csv", data_files="my_file.csv")

# Define a custom preprocessing pipeline
pipeline = PreprocessorPipeline(
    [("scaler", StandardScaler()), ("log_transformer", LogTransformer())]
)

# Fit and transform the dataset using the pipeline
preprocessed_data = pipeline.fit_transform(dataset.to_pandas())
```

For further details, check the [advance usage documentation](./docs/PREPROCESSING.md).

# License

Biofit is licensed under the Apache 2.0 License. See the [LICENSE](./LICENSE) file for
more information.

# Contributing

If you would like to contribute to Biofit, please read the
[CONTRIBUTING](./CONTRIBUTING.md) guidelines.
