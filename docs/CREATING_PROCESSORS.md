# Contents {#contents .TOC-Heading}

[BaseProcessor](#baseprocessor)

    [Fit](#fit)

    [Initialization and Input Handling](#initialization-and-input-handling)

    [Processing Fit (`_process_fit`)](#processing-fit-_process_fit)

    [Core Fitting Function (`_fit`)](#core-fitting-function-_fit)

    [Batch Processing](#batch-processing)

    [Finalizing Fit (`_process_fit_output` and State Update)](#finalizing-fit-_process_fit_output-and-state-update)

[Transform/Predict](#transformpredict)

[Initialization and Input Handling](#initialization-and-input-handling-1)

[Processing Transform (`_process_transform`)](#processing-transform-_process_transform)

[Core Transformation Function (`_transform`)](#core-transformation-function-_transform)

[Batch Processing](#batch-processing-1)

[Finalizing Transform](#finalizing-transform)

[Overview of ProcessorConfig](#overview-of-processorconfig)

[Required Parameters](#required-parameters)

[Other Important Parameters](#other-important-parameters)

[Configuration Management and Customization](#configuration-management-and-customization)

[Extensibility and Usage](#extensibility-and-usage)

[Overview of Classification vs. Regression Outputs](#overview-of-classification-vs.-regression-outputs)

[Overview of Classification vs. Regression Outputs](#_Toc183067150)

[1. Task Differentiation](#task-differentiation)

[2. Key Functions Handling Outputs](#key-functions-handling-outputs)

[3. Data Processing and Preprocessing](#data-processing-and-preprocessing)

[4. Visualization and Reporting](#visualization-and-reporting)

[5. Model Training](#model-training)
[6. Dataset Creation](#dataset-creation)

[7. Configuration Management](#configuration-management)

# BaseProcessor

The BaseProcessor class is an abstract class that provides ways to
support different input formats and parameters, and handles the running
process for batching and multiprocessing. This document outlines its
runtime behavior, focusing on the fit and transform methods, and their
design

## Fit

When the fit method is called in a concrete class (e.g. ColumnSumStat,
RandomForestModel, etc), the BaseProcessor class is responsible for
configuring and preparing the processor to handle data transformation.
The purpose of fit is to learn from the input data and store parameters
that will later be used to transform the data. The fitting process
involves multiple stages, including input validation, method selection,
data batching, and potential multiprocessing. Below is an overview of
how these components work together during runtime:

### 1. **Initialization and Input Handling**

The fit method handles basic setup and configuration. Consider the
following example:

```python
from dataclasses import dataclass
from biofit.processing import BaseProcessor, ProcessorConfig


@dataclass
class MyProcessorConfig(ProcessorConfig):
    some_val: int = 2


class MyProcessor(BaseProcessor):
    _config_class = MyProcessorConfig

    def _fit_pandas(self, X, y):
        # Fit logic for Pandas DataFrame
        pass

    def fit(
        self,
        X,
        y=None,
        input_columns=None,
        target_column=None,
        cache_dir=None,
        some_val=2,
    ):
        self.config._input_columns = self._set_input_columns_and_arity(
            input_columns, target_column
        )

        self._process_fit(X, y, cache_dir=cache_dir, some_val=some_val)
```

In **all** fit methods within concrete classes, the following should be
ensured:

-   **Correct Argument Ordering (`_process_fit`)**: The order of the
    first few arguments matters when passing arguments from fit to
    `_process_fit`. The initial arguments should match the input for the
    `_fit_*` methods (e.g., `_fit_pandas`) in the class:

    -   All non-keyword variable arguments (\*args) will be passed as
        data input into `_fit_*` methods in that order. With the
        exception of the first argument, these values can be passed as
        None.

    -   All keyword arguments will update the corresponding attribute in
        the config class of the processor.

    -   In the example above, the fit method requires the input data (X)
        and class labels (y) in that order. By passing X and y as
        variable arguments, they will be passed to `_fit_pandas`, while
        the keyword arguments will update the config class.

-   **Input Columns and Arity (`_set_input_columns_and_arity`)**: This
    method determines which columns from the provided input data to use
    and their arity (number of arguments).

    -   The first argument acts as both the master table and the first
        argument for `_fit_*` methods, meaning that all other input
        data can be derived from this master table.

    -   The order of the columns in `_set_input_columns_and_arity` should
        match the order provided in `_process_fit`.

    -   In the example above, if only X is passed to the fit function,
        where X contains both input data and labels (y) along with other
        irrelevant data (e.g sample metadata), we can specify the input
        columns and target column and leave y as None.

    -   In the backend, whenever an input data argument is None and its
        corresponding input column is specified, the program will
        retrieve the data from the first argument.

    -   In cases where input data cannot be derived from the first
        argument (e.g., if the first two arguments represent x1 before
        and x2 after a transformation), the validation logic should be
        implemented within the fit function itself.

    -   You would still use `_set_input_columns_and_arity` and
        `_process_fit` as usual, but ensure that x2 is not None before
        calling `_process_fit`.

### Processing Fit (`_process_fit`)

After the initial setup, the fit method transitions into a more detailed
preparation phase handled by `_process_fit` within the BaseProcessor
class. This method coordinates several operations:

-   **Retrieve Appropriate Methods (`_get_method`)**: Depending on the
    format of the data (e.g., Pandas DataFrame, Polars DataFrame),
    `_get_method` selects the appropriate internal function to be used.

    -   When a data format like a NumPy array is provided but no
        specific implementation exists for that format, the `_get_method`
        function selects the method that minimizes the overhead required
        to convert the data into the expected format for the fitting
        process.

        -   The reason for supporting multiple data formats is two
            folds:

            -   Conversions between certain formats can be
                computationally expensive, especially for large
                datasets. By selecting the most efficient format, we can
                avoid large overhead. The order of priority for data
                formats is as follows (from highest to lowest):

                1.  **Arrow**

                2.  **Polars** (if available)

                3.  **Pandas**

                4.  **NumPy** (Low on the list because it requires
                    contiguous memory)

                5.  **Native Python types** (e.g., list, dict, etc.)

            -   Many Processor classes are wrappers arround third party
                libraries and most don't support all the same format.
                This way, we can chain multiple different third-party
                libraries without much fuss.

    -   When batch_size is specified (not None), `_get_method`
        prioritizes methods with the `_partial_fit``_`\* prefix:

        -   The presence of a `_partial_fit``_`\* method indicates that
            the processor supports batch processing.

        -   If a `_pool_fit` method is also implemented and num_threads
            is greater than 1, it signifies support for multiprocessing:

            -   `_pool_fit` will receive list of individual processor
                instances, where each ran on different subsets of the
                data.

            -   It consolidates these instances into a single new
                processor class, aggregating information collected from
                each individual processor.

-   **Validation (`_validate_fit_params`)**: `_process_fit` ensures that
    the provided parameters match the processor\'s requirements. This
    check helps identify issues early, such as mismatched input columns
    or unsupported data types.

-   **Wrapper for Fitting (wrapper)**: `_process_fit` uses a wrapper
    function that manages additional setup, such as handling exceptions
    or managing the execution context before proceeding to the core
    fitting function (`_fit`).

### Core Fitting Function (`_fit`)

The `_fit` method is the core of the fitting process and handles multiple
tasks to prepare the processor for transforming data:

-   **Get Columns (`_get_columns`)**: This method identifies the relevant
    columns from the input data, either based on user specifications or
    default behavior. It ensures that the necessary features are
    available for fitting.

    -   The columns passed are the ones in self.config.`_input_columns`,
        which should be a value returned from
        `self._set_input_columns_and_arity`(...).

    -   When the input data is a Dataset class, the
        `_fit_input_feature_types` and `_fit_unused_feature_types`
        parameters are used in the function when the input columns
        passed to self.config.`_input_columns` are None. These attributes
        define which feature types (columns) are expected to be used or
        excluded during the fitting process.

    -   Features, retrieved via X.`_info`.features, are classes assigned
        to dataset columns to indicate the type of data each column
        contains.

    -   Examples of feature types:

        -   Sample: Columns holding sample IDs.

        -   Abundance: Columns holding abundance values.

    -   Features are classes defined by `datasets.features.features.Features`.

        -   The optional library `biosets` has predefined feature classes that
            are geared towards biological data.

        -   The `get_feature` function from `biofit.integration.biosets`
            can be used to retrieve a feature class by name. An `object` is
            returned if `biosets` is not installed.

    -   `_fit_input_feature_types` specifies which feature types are
        required for the processor and `_fit_unused_feature_types`
        specifies which feature types should be ignored during
        processing.

    -   These attributes must be defined as lists of tuples or types in
        the processor\'s configuration class, using
        `field(default_factory=list)`.

    -   For example:

```python
from dataclasses import dataclass, field

from biofit.integration.biosets import get_feature


@dataclass
class ProcessorConfig:
    _fit_input_feature_types: list = field(
        default_factory=lambda: [None, get_feature("ClassLabel")]
    )
    _fit_unused_feature_types: list = field(
        default_factory=lambda: [get_feature("METADATA_FEATURE_TYPES"), None]
    )
```

    -   **Here:**

        -   For the first input data, no specific features are used (None),
            but metadata-like features (METADATA_FEATURE_TYPES) are
            excluded.

        -   For the second input data, columns contain ClassLabel are
            required, and no specific features are excluded (None).

-   **Order and Behavior:**

    -   The order of elements in `_fit_input_feature_types` and
        `_fit_unused_feature_types` must match the order of the input
        arguments in `_process_fit`.

    -   If no constraints are needed for a specific input, None can be
        specified in either

-   **Prepare Fit Arguments (`_prepare_fit_kwargs`)**: Biofit is capable of
    batching and multiprocessing to handle large datasets.
    `_prepare_fit_kwargs` sets up keyword arguments that control how the
    data will be processed, including batch size, multiprocessing
    settings, and other parameters.

-   **Input Processing (`_process_fit_input`)**: This function allows for
    any context specific implementation for concrete classes on prepared
    input. For example, if you have a model and you need to decipher if
    target values are regression targets or labels, `_process_fit_input`
    will pass along the unbatched data so that you can check if number
    of unique labels is greater than 2 (i.e. multiclass classification).

-   **Run the Processing (run)**: The run method executes the core
    fitting function, potentially in batches and using multiprocessing.

-   **Prepare Runner (`_prepare_runner`)**: Before execution,
    `_prepare_runner` ensures that the runner function has the required
    context and metadata to run effectively.

-   **Runner Execution (runner)**: The runner function is responsible
    for applying the fitting logic over the data, which may involve
    using the map function for batch processing or `_pool_fit` for
    multiprocessing.

### Batch Processing

The fitting process often involves processing data in batches,
especially for large datasets:

-   **Mapping Over Data (map and `_map_single`)**: The runner function
    utilizes map to iterate over data in manageable chunks. `_map_single`
    is called for each batch, applying the fitting logic and managing
    the individual operations within each batch.

-   **Function Application (apply_function_on_filtered_inputs)**: This
    method filters the input data as needed and applies the selected
    function to it.

-   **Batch-Level Operations (`_process_batches`)**: The
    `_process_batches` function oversees the application of fitting logic
    to each batch. It orchestrates calls to `_process_fit_batch_input`,
    `_fit_polars` (if using Polars DataFrames), and
    `_process_fit_batch_output` to handle the end-to-end batch
    processing.

    -   **Batch Input Handling (`_process_fit_batch_input`)**: Prepares
        each batch for fitting, ensuring the data is in the correct
        format.

    -   **Fitting Logic (`_fit_polars`)**: If the input data is in a
        Polars DataFrame format, `_fit_polars` is used to perform the
        fitting. This method handles the core fitting logic for Polars
        data specifically.

    -   **Batch Output Processing (`_process_fit_batch_output`)**: Once a
        batch has been processed, this function handles any necessary
        updates or cleanup.

### Finalizing Fit (`_process_fit_output` and State Update)

Once all batches are processed and the core fitting logic has been
applied:

-   **Process Fit Output (`_process_fit_output`)**: The processor
    finalizes the fitting by consolidating results, updating internal
    states, and marking the processor as fitted.

-   **Caching and State Updates**: If caching is enabled, the processor
    saves the learned parameters to a cache file, allowing for faster
    loading in the future. The processor\'s state is updated to reflect
    that it has been fitted, and any necessary fingerprints (unique
    identifiers for data and configurations) are generated or updated.

# Transform/Predict

When the transform/predict method is called in a concrete class (e.g.
ColumnSumStat, MinPrevalenceFeatureSelector, RandomForestModel, etc.),
the BaseProcessor class is responsible for orchestrating the
transformation of data based on the learned parameters from the fit
method. The purpose of transform is to apply these learned parameters to
new data. The transformation process involves input validation, method
selection, handling batch operations, and managing output. The
preparation behind transformation behaves very similarly to fitting
functions. However, the most notable difference is that state management
is done directly on the class itself and not in the config class like
what happens during fit runtime. Anything that is not `self.config.*` are
transitory and are not stored or cached.

## Initialization and Input Handling

Much like the fit function, the transform method handles basic setup and
configuration. [The exact same requirements applies to
transformation,]{.underline} except:

-   **Input Columns and Arity (`_set_input_columns_and_arity`)**: In the
    fit method, the parameter set is self.config.`_input_columns`, which
    configures the columns at the configuration level. In contrast, in
    the transform method, the parameter set is `self._input_columns`,
    which is used for processing runtime transformations.

-   **Method Prefix (`_method_prefix`):** Another difference is that some
    classes, namely model classes, requires that the method prefix be
    specified using `self._method_prefix` attribute to select the method
    with the correct prefix. This is important for model classes where
    you have the option to use predict_proba or predict. For instance,
    the user calls the \`predict_proba\` method from the model class,
    inside predict_proba will have:

    ```python
    self._method_prefix = "_predict_proba"
    ```

The `_get_method` function will retrieve `_predict_proba``_`\* (i.e.
`_predict_proba_sklearn`)

-   By default `self.`_method_prefix`` equals `"`_transform`"`.

Another important aspect is

## Processing Transform (`_process_transform`)

After initial setup, the transform method proceeds with preparation
through `_process_transform` within the BaseProcessor class. This stage
involves several operations:

-   **Retrieve Appropriate Methods (`_get_method`)**: Exactly as fit
    runtime, `_get_method` selects the appropriate internal function to
    apply based on the format of the input data (e.g., Pandas DataFrame,
    Polars DataFrame).

    -   As mentioned above, `_get_method` uses self.`_method_prefix` to
        select appropriate functions.

    -   [By it's very nature, transformations are independent of other
        samples, which means all transform functions support batching
        and multiprocessing.]{.underline}

-   **Validation (`_validate_transform_params`)**: `_process_transform`
    validates that the provided parameters match the processor\'s
    requirements. This step helps to catch configuration or data type
    issues early.

-   **Wrapper for Transformation (wrapper)**: `_process_transform` uses a
    wrapper to manage additional setup, such as context handling or
    logging, before moving to the core transformation function
    (`_transform`).

## Core Transformation Function (`_transform`)

The `_transform` method is central to the transformation process and
performs multiple steps to modify the data:

-   **Check if Fitted (is_fitted)**: The method first checks if the
    processor has been fitted to the data. This ensures that the
    necessary parameters for transformation are available.

-   **Get Columns (`_get_columns`)**: As with fir methods, this method
    identifies the relevant columns from the input data based on user
    specifications or default behavior. It ensures that the necessary
    features are available for transformation.

    -   When the input is a Dataset and self.`_input_columns` is None for
        the corresponding data, `_get_columns` will use
        `_transform_input_feature_types` and
        `_transform_input_feature_types`, instead of
        `_fit_input_feature_types` and `_fit_input_feature_types`.

    -   Initialization and behavior of these attributes are the exact
        same as the ones for fit.

-   **Prepare Transformation Arguments (`_prepare_transform_kwargs`)**:
    This step sets up keyword arguments for batch processing,
    multiprocessing, and any additional transformation settings.

-   **Data Input Handling (`_process_transform_input`):** The
    `_process_transform_input` function sets up the configuration before
    running the batch-level operations. This is important for logging
    and keeping track of information. Say you need to log the number of
    rows that were filtered out from a filter transformation, you would
    use this function to store the number of rows in the original
    dataset. You would then use the `_process_transform_output` function
    to gather the final number of rows and out print the number of
    filtered rows.

-   **Run the Transformation (run)**: The run method executes the core
    transformation, potentially in batches and using multiprocessing.

    -   **Prepare Runner (`_prepare_runner`)**: `_prepare_runner` ensures
        that the runner function has the necessary context and metadata
        before executing the transformation.

    -   **Runner Execution (runner)**: The runner function applies the
        transformation logic to the data, using batch processing or
        multiprocessing as needed.

## Batch Processing

The transformation process supports batch processing to handle large
datasets. However, regardless of wether:

-   **Mapping Over Data (map and `_map_single`)**: The runner function
    uses map to iterate over the data in manageable chunks. `_map_single`
    is applied to each batch, ensuring consistency in the transformation
    logic.

-   **Batch-Level Operations (`_process_batches`)**: The
    `_process_batches` function coordinates the transformation for each
    batch. It calls methods like `_process_transform_input` and
    `_transform_polars` (if using Polars DataFrames) to handle the
    specific operations for each batch.

    -   **Batch Input Handling (`_process_transform_batch_input`)**:
        Prepares each batch of data for transformation, ensuring the
        data is formatted correctly.

    -   **Transformation Logic (`_transform_polars`)**: If the input data
        is in a Polars DataFrame format, `_transform_polars` applies the
        transformation logic specific to Polars.

    -   **Batch Output Processing (`_process_transform_batch_output`)**:
        After processing each batch, this function manages output
        formatting and any cleanup required.

## Finalizing Transform

After all batches are processed and the core transformation logic has
been applied:

-   **Process Transform Output (`_process_transform_output`)**: This
    method finalizes the transformation, consolidating results and
    updating internal states as needed. See the example that was given
    in `_process_transform_input`.

-   **Caching and State Updates**: If caching is enabled, the processor
    saves the transformed data for faster access in future operations.
    The processor\'s state is updated to indicate that the
    transformation is complete.

    -   Loading from cache is only possible with Dataset types.
        Internally, it uses the X.`_fingerprint` attribute to load the
        cache. The fingerprint is modified every time the data gets
        transformed.

# Overview of ProcessorConfig

The ProcessorConfig is responsible for storing and managing processor
configurations throughout the lifecycle of fitting and transforming
data. It defines attributes for input configuration, processing control,
caching behavior, and more.

## Required Parameters

The ProcessorConfig class plays a central role in configuring a
processor to handle specific tasks. Among the required attributes,
several stand out for their importance during runtime, especially when
determining how the data will be processed:

1.  **processor_name**:

-   **Purpose**: Represents the name of the processor. This is
    particularly useful for registering them in auto classes. This name
    should be unique from other processors so that they do not override
    the keys within the auto class registry.

-   **Example Usage**: The name could be something like
    MinMaxScalerConfig or RandomForestConfig, depending on the type of
    processor being used.

2.  **`_[fit/transform]_[input/unused]_feature_types`**:

-   **Purpose**: Define which features should be used or ignored during
    the fitting and transformation processes. These attributes are only
    used for Dataset types.

-   **Structure**: These attributes are lists of tuples or types that
    specify the feature types for each input argument.

-   **Usage Example**:

```python
from dataclasses import dataclass, field

from biofit.integration.biosets import get_feature


@dataclass
class ProcessorConfig:
    _fit_input_feature_types: list = field(
        default_factory=lambda: [None, get_feature("ClassLabel")]
    )

    _fit_unused_feature_types: list = field(
        default_factory=lambda: [get_feature("METADATA_FEATURE_TYPES"), None]
    )
```

-   In the above example, the first input dataset has no specific
    feature requirements (indicated by None), while metadata-like
    features are ignored. The second input dataset must contain
    ClassLabel features and has no unused feature requirements.

<!-- -->

-   **Order and Behavior**: The order of elements in
    \_fit_input_feature_types and \_fit_unused_feature_types must match
    the order of the input arguments in the fitting or transformation
    method. If no constraints are required, None can be specified.

3.  **\_n_features_out**

-   **Purpose**: The \_n_features_out parameter is used when the output
    of a transformation is not a one-to-one mapping with the input
    features. This typically occurs in cases where new features are
    generated or existing features are combined, such as feature
    extraction or dimensionality reduction.

-   **Example**: When using a feature extraction technique like PCA, the
    number of output features (n_features_out) is often different from
    the number of input features, as the transformation results in a
    reduced set of principal components.

-   **Usage**: This attribute is set during fitting or before
    transformation to indicate the number of output features when the
    transformation is not one-to-one. If the transformation results in a
    one-to-one mapping, \_n_features_out can be left as None.

### Other Important Parameters

1.  **\_\[fit/transform\]\_process_desc**:

-   **Purpose**: Provides a textual description of the process for
    progress tracking or logging purposes.

-   **Example**:

\_fit_process_desc: str = field(default=\"Fitting the processor to the
input data\", init=False, repr=False)

\_transform_process_desc: str = field(default=\"Transforming the input
data\", init=False, repr=False)

2.  **dataset_name** (specific to dataset configurations):

-   **Purpose**: Specifies the dataset that the processor is designed to
    handle. This is mainly used when extending a common processor
    configuration to work with a specific experiment type.

-   **Usage**: This attribute is typically set in classes that inherit
    the common configuration class and add specific functionality for a
    named dataset, often following the naming convention
    \...ConfigForSNP. This helps create dataset-specific configurations
    for automatic pipeline creation.

```python
@dataclass
class MinPrevalenceFeatureSelectorConfigForSNP(MinPrevalenceFeatureSelectorConfig):
    dataset_name: str = "snp"
    depth: int = 0
```

### Configuration Management and Customization

ProcessorConfig attributes are defined using the dataclasses library.
You can restrict how they are retrieved via the field function from
dataclasses:

-   **Initialization of Config classes**: All config classes are
    **dataclasses**. This means that you define attributes as **class
    attributes**, rather than instance attributes. One key thing is that
    you can differentiate different class attributes with the function
    **field**.

    -   **init=True/False**: You can define whether that class attribute
        can be initialized. For instance, MyClass(my_var=0), means that
        init is true.

    -   **repr=True/False**: This defines wether that attribute will be
        printed when using the function print on one of its instances.

    -   Note that regardless if either parameters being set to true or
        false, all parameters within config class are stored within
        cache.

-   **Attribute Retrieval with get_params**:

    -   The get_params method in ProcessorConfig can retrieve parameters
        based on whether they were set at initialization or should be
        included in a representation string.

    -   **show_init_only=True/False**: When set to True, get_params only
        returns parameters that were defined with init=True via
        dataclasses' field function. This can be used to differentiate
        between parameters explicitly provided by the user and those
        that were set internally.

    -   **show_repr_only=True/False**: When set to True, get_params only
        returns parameters that are intended to be shown in the string
        representation (repr=True). This is useful for retrieving class
        attributes that can only be set internally but are needed for
        other functions.

    -   By using both show_repr_only and show_init_only, you can choose
        which parameters will be passed to other functions, which can be
        ones set by the user (show_init_only=True) and which ones set
        internally (show_init_only=False, show_repr_only=True).

### Extensibility and Usage

-   **Dataset-Specific Configurations**:

    -   Specialized configurations can be created for certain datasets
        by extending the default config class and adding attributes like
        dataset_name. For instance:

-   **Interfacing with BaseProcessor**:

    -   During the fit or transform process, BaseProcessor relies on
        ProcessorConfig to retrieve and update the parameters necessary
        for data handling and caching.

        -   Learned parameters from fitting should be stored within the
            configuration (`self.config.*`), and runtime-specific
            information in the processor itself (`self.*`). This
            distinguishes what gets stored within cashed and what is
            transient.

        1.  You can associate parameters of the same name to fit
            functions and transform function via `self.config.*` and
            `self.*`, respectively. An example is the attribute,
            \_feature_names_in, where one specifies the input features
            for fit and transform separately. By comparing the input
            features of the current transformations against the ones
            used during fitting, you can check if there are some missing
            and/or extra columns and if they are in different order.

