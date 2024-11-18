import gc
import inspect
import io
import os
import re
import shutil
import subprocess
import sys
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, TextIO

import numpy as np
from biocore.utils.import_util import (
    is_rpy2_arrow_available,
    is_rpy2_available,
    requires_backends,
)

import biofit.config
from biofit import config
from biofit.utils import logging

if TYPE_CHECKING:
    from rpy2.robjects.conversion import Converter


class PackageNotInstalledError(ImportError):
    """Error occuring because the R package to import is not installed."""

    pass


logger = logging.get_logger(__name__)

_is_auto_install = None

R_PLOTTING_DEPENDENCIES = {
    "cran": [
        "ggplot2",
        "arrow",
        "circlize",
        "RColorBrewer",
        "scales",
        "forcats",
        "patchwork",
        "reshape2",
        "dplyr",
        "tools",
    ],
    "bioconductor": ["ComplexHeatmap"],
}
R_PREPROCESSING_DEPENDENCIES = {
    "cran": [],
    "bioconductor": ["edgeR"],
}


def is_auto_install():
    global _is_auto_install
    if _is_auto_install is None:
        _is_auto_install = not biofit.config.BIOFIT_SKIP_R_DEPENDENCIES
    return _is_auto_install


def enable_auto_install():
    global _is_auto_install
    _is_auto_install = True


def disable_auto_install():
    global _is_auto_install
    _is_auto_install = False


class ROutputCapture:
    def __init__(self, stdout: TextIO = io.StringIO(), stderr: TextIO = io.StringIO()):
        requires_backends("ROutputCapture", "rpy2")
        from rpy2.rinterface_lib import callbacks

        # Create StringIO buffers to capture output
        self.stdout = stdout
        self.stderr = stderr
        # Save original R console write functions
        self._original_consolewrite_print = callbacks.consolewrite_print
        self._original_consolewrite_warnerror = callbacks.consolewrite_warnerror

    def __enter__(self):
        from rpy2.rinterface_lib import callbacks

        # Define new console write functions
        def custom_consolewrite_print(output):
            self.stdout.write(output)

        def custom_consolewrite_warnerror(output):
            self.stderr.write(output)

        # Replace R console write functions with custom functions
        callbacks.consolewrite_print = custom_consolewrite_print
        callbacks.consolewrite_warnerror = custom_consolewrite_warnerror

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        from rpy2.rinterface_lib import callbacks

        # Restore original R console write functions
        callbacks.consolewrite_print = self._original_consolewrite_print
        callbacks.consolewrite_warnerror = self._original_consolewrite_warnerror

        # Reset the position of StringIO buffers to the beginning
        self.stdout.seek(0)
        self.stderr.seek(0)

    def get_stdout(self):
        return self.stdout.getvalue()

    def get_stderr(self):
        return self.stderr.getvalue()


def get_linux_distro_codename():
    """
    Retrieves the codename of the Linux distribution by parsing /etc/os-release.

    Returns:
        str: The codename of the distribution (e.g., 'focal', 'bullseye', 'centos7').
             Returns 'Unknown' if the codename cannot be determined.
    """
    os_release_path = "/etc/os-release"

    if not os.path.isfile(os_release_path):
        return "Unknown"

    os_info = {}

    try:
        with open(os_release_path, "r") as file:
            for line in file:
                # Remove any leading/trailing whitespace and skip empty lines
                line = line.strip()
                if not line or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                # Remove surrounding quotes if present
                value = value.strip('"').strip("'")
                os_info[key] = value
    except Exception as e:
        print(f"Error reading {os_release_path}: {e}")
        return None

    # Handle specific distributions
    distro_id = os_info.get("ID", "").lower()

    if distro_id in ["ubuntu", "debian"]:
        # For Ubuntu and Debian, extract codename from VERSION or VERSION_ID
        if "VERSION_CODENAME" in os_info:
            return os_info["VERSION_CODENAME"].lower()
        elif "VERSION" in os_info:
            # Example VERSION: "20.04.5 LTS (Focal Fossa)"
            match = re.search(r"\(([^)]+)\)", os_info["VERSION"])
            if match:
                return match.group(1).lower()
        elif "VERSION_ID" in os_info:
            return os_info["VERSION_ID"].lower()

    elif distro_id in ["centos", "rhel", "fedora"]:
        # For CentOS, RHEL, Fedora, use VERSION_ID or similar
        if "VERSION_ID" in os_info:
            return f"{distro_id}{os_info['VERSION_ID'].lower()}"
        elif "VERSION" in os_info:
            # Example VERSION: "7 (Core)"
            match = re.search(r"(\d+)", os_info["VERSION"])
            if match:
                return f"{distro_id}{match.group(1)}"

    # Attempt to get VERSION_CODENAME directly
    if "VERSION_CODENAME" in os_info:
        return os_info["VERSION_CODENAME"].lower()

    # Fallback: Try to extract codename from PRETTY_NAME or other fields
    for key in ["PRETTY_NAME", "NAME", "DESCRIPTION"]:
        if key in os_info:
            match = re.search(r"\b([a-z]+)\b", os_info[key], re.IGNORECASE)
            if match:
                return match.group(1).lower()

    return None


def get_cran_info():
    """
    Detects the R version using rpy2 and constructs the appropriate CRAN URL based on the operating system
    and, for Linux, the distribution codename.

    Returns:
        str: The constructed CRAN URL, or a default 'latest' URL if R version cannot be detected.
    """
    cran_url = None

    from rpy2.rinterface_lib.embedded import RRuntimeError

    try:
        if sys.platform in ["win32", "darwin"]:
            # For Windows, include R version in the CRAN URL
            cran_url = "https://packagemanager.posit.co/cran/latest"
        elif sys.platform.startswith("linux"):
            # For Linux and macOS
            dist = get_linux_distro_codename()
            if dist:
                cran_url = (
                    f"https://packagemanager.posit.co/cran/__linux__/{dist}/latest"
                )
            else:
                logger.warning(
                    "Linux distribution codename could not be determined. Using default CRAN URL."
                )
                return None
        else:
            logger.warning("Unknown operating system. Using default CRAN URL.")
            return None

    except RRuntimeError as e:
        logger.error(f"Error detecting R version: {e}. Using default CRAN URL.")

    return cran_url


def get_bioconductor_info():
    bioc_mirror = "https://packagemanager.posit.co/bioconductor/latest"
    bioconductor_config_file = f"{bioc_mirror}/config.yaml"
    logger.info(f"Using Bioconductor mirror: {bioc_mirror}")
    logger.info(f"Using Bioconductor config file: {bioconductor_config_file}")

    return bioconductor_config_file, bioc_mirror


class RCaller:
    r_code = None
    r_source = None
    _r_context = {}
    _global_vars = {}

    def _get_converter(self, obj, use_arrow=True) -> "Converter":
        """Check if rpy2 is available and retrieves the right converter.

        Args:
            obj (obj): The object (or name) calling this method.

        Raises:
            RuntimeError: If rpy2 is not available.

        Returns:
            Converter: The converter to convert R objects to Python objects.
        """
        if is_rpy2_arrow_available() and use_arrow:
            from rpy2.robjects import default_converter, numpy2ri, pandas2ri
            from rpy2_arrow.arrow import converter

            _converter = (
                default_converter + numpy2ri.converter + pandas2ri.converter + converter
            )
        elif is_rpy2_available():
            from rpy2.robjects import conversion, numpy2ri, pandas2ri

            _converter = (
                (
                    conversion.get_conversion()
                    if getattr(conversion, "get_conversion", None)
                    else conversion.converter
                )
                + numpy2ri.converter
                + pandas2ri.converter
            )
        else:
            # suggest installing rpy2_arrow if rpy2 is not available
            requires_backends(obj, "rpy2_arrow")
            _converter = None

        return _converter

    @staticmethod
    def get_r_home():
        """
        Retrieve the R_HOME directory across multiple systems, prioritizing local R installations,
        particularly those within an active Conda environment.

        Returns:
            str or None: The path to R_HOME if found, otherwise None.
        """
        # Step 1: Check R_HOME environment variable
        r_home_env = os.environ.get("R_HOME")
        if r_home_env and os.path.exists(r_home_env):
            return r_home_env

        # Step 2: Check for R in active Conda environment
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            if sys.platform == "win32":
                r_executable = os.path.join(conda_prefix, "Scripts", "R.exe")
            else:
                r_executable = os.path.join(conda_prefix, "bin", "R")

            if os.path.exists(r_executable):
                try:
                    r_home = subprocess.check_output(
                        [r_executable, "RHOME"], universal_newlines=True
                    ).strip()
                    if os.path.exists(r_home):
                        return r_home
                except subprocess.CalledProcessError:
                    pass  # R executable found but failed to get R_HOME

        # Step 3: Check system PATH for R executable
        r_executable = shutil.which("R")
        if r_executable:
            try:
                r_home = subprocess.check_output(
                    [r_executable, "RHOME"], universal_newlines=True
                ).strip()
                if os.path.exists(r_home):
                    return r_home
            except subprocess.CalledProcessError:
                pass  # R executable found but failed to get R_HOME

        # Step 4: Check common installation paths (additional step for thoroughness)
        potential_paths = []
        if sys.platform == "win32":
            # Windows common installation paths
            potential_paths.extend(glob("C:/Program Files/R/R-*/"))

            # Attempt to read R_HOME from Windows Registry
            try:
                import winreg

                reg_path = r"SOFTWARE\R-core\R"
                for root in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
                    try:
                        key = winreg.OpenKey(root, reg_path)
                        r_home, _ = winreg.QueryValueEx(key, "InstallPath")
                        if os.path.exists(r_home):
                            return r_home
                    except FileNotFoundError:
                        continue
            except ImportError:
                pass  # winreg not available
        else:
            # macOS and Linux common installation paths
            potential_paths.extend(
                [
                    "/usr/local/lib/R",
                    "/usr/lib/R",
                    "/Library/Frameworks/R.framework/Resources",  # macOS
                ]
            )

        for path in potential_paths:
            if os.path.exists(path):
                return path

        # R_HOME could not be determined
        return None

    @staticmethod
    def verify_r_dependencies(
        cran_dependencies: List[str] = R_PLOTTING_DEPENDENCIES["cran"]
        + R_PREPROCESSING_DEPENDENCIES["cran"],
        bioconductor_dependencies: List[str] = R_PREPROCESSING_DEPENDENCIES[
            "bioconductor"
        ]
        + R_PLOTTING_DEPENDENCIES["bioconductor"],
        install_missing: Optional[bool] = None,
        **kwargs,
    ):
        if install_missing is None:
            install_missing = is_auto_install()

        r_home = None
        try:
            import rpy2.situation

            r_home = rpy2.situation.get_r_home()
        except Exception:
            pass
        r_home = r_home or RCaller.get_r_home()
        if r_home is None:
            raise RuntimeError("R_HOME could not be determined.")

        import rpy2.robjects.packages as rpackages
        from rpy2.robjects import ListVector
        from rpy2.robjects.vectors import StrVector

        utils = rpackages.importr("utils")
        base = rpackages.importr("base")
        bioc_missing, cran_missing = [], []
        cran_url = get_cran_info()
        if cran_url is not None and "repos" not in kwargs:
            kwargs["repos"] = ListVector([("CRAN", cran_url)])
        if cran_dependencies:
            names_to_install = [
                x for x in cran_dependencies if not rpackages.isinstalled(x)
            ]
            if len(names_to_install) > 0 and install_missing:
                logger.info(f"Using R_HOME: {r_home}")
                logger.info(f"Installing missing CRAN dependencies: {names_to_install}")
                if "BiocManager" in names_to_install:
                    names_to_install.remove("BiocManager")
                    utils.chooseCRANmirror(ind=1)
                    utils.install_packages("BiocManager")

                if cran_url is not None:
                    logger.info("Using CRAN mirror: %s", cran_url)
                base.options(**kwargs)
                utils.install_packages(StrVector(names_to_install))
                # verify again to check if all dependencies are installed
                names_to_install = [
                    x for x in cran_dependencies if not rpackages.isinstalled(x)
                ]
                if len(names_to_install) > 0:
                    # conda package names
                    conda_packages = ["r-" + name.lower() for name in names_to_install]

                    conda_install_cmd = (
                        f"conda install -y -c conda-forge {' '.join(conda_packages)}"
                    )
                    logger.warning(
                        f"Failed to install the following CRAN dependencies: "
                        f"{names_to_install}. If you are using conda, you can try "
                        "installing the dependencies via:\n"
                        f"{conda_install_cmd}"
                    )

            else:
                cran_missing = names_to_install
        if bioconductor_dependencies:
            names_to_install = [
                x for x in bioconductor_dependencies if not rpackages.isinstalled(x)
            ]
            if len(names_to_install) > 0 and install_missing:
                logger.info(f"Using R_HOME: {r_home}")
                logger.info(
                    "Installing missing Bioconductor dependencies: "
                    f"{bioconductor_dependencies}"
                )
                if not rpackages.isinstalled("BiocManager"):
                    logger.info("Installing BiocManager")
                    utils.chooseCRANmirror(ind=1)
                    utils.install_packages("BiocManager")

                biocmanager = rpackages.importr("BiocManager")
                bioconductor_config_file, bioc_mirror = get_bioconductor_info()
                if cran_url is not None:
                    logger.info("Using CRAN mirror: %s", cran_url)
                if bioc_mirror is not None:
                    logger.info("Using Bioconductor mirror: %s", bioc_mirror)
                if bioc_mirror is not None and "BioC_mirror" not in kwargs:
                    kwargs["BioC_mirror"] = bioc_mirror
                if (
                    bioconductor_config_file is not None
                    and "BIOCONDUCTOR_CONFIG_FILE" not in kwargs
                ):
                    kwargs["BIOCONDUCTOR_CONFIG_FILE"] = bioconductor_config_file
                base.options(**kwargs)
                biocmanager.install(
                    StrVector(names_to_install), ask=False, update=False
                )
                # verify again to check if all dependencies are installed
                names_to_install = [
                    x for x in bioconductor_dependencies if not rpackages.isinstalled(x)
                ]
                if len(names_to_install) > 0:
                    conda_packages = [
                        "bioconductor-" + name.lower() for name in names_to_install
                    ]
                    conda_install_cmd = (
                        "conda install -y -c conda-forge -c bioconda "
                        f"{' '.join(conda_packages)}"
                    )
                    msg = (
                        f"Failed to install the following Bioconductor dependencies: "
                        f"{names_to_install}. If you are using conda, you can try "
                        "installing the dependencies via:\n"
                        f"conda install -y -c conda-forge {' '.join(conda_packages)}"
                    )
                    if sys.platform == "win32":
                        msg += (
                            "\nNote: Many of the Bioconductor packages are not "
                            "available on Windows. Please submit an issue on the BIOFIT "
                            "GitLab repository if you need help with installation."
                        )
                    logger.warning(msg)
            else:
                bioc_missing = names_to_install
        if (cran_missing or bioc_missing) and not install_missing:
            instructions = "Run `biofit.integration.R.RCaller.verify_r_dependencies("
            if cran_missing:
                instructions += f"cran_deps={cran_missing}, "
            if bioc_missing:
                instructions += f"bioc_deps={bioc_missing}, "
            instructions += "install_missing=True)` to install missing dependencies."
            raise PackageNotInstalledError(
                f"Missing R dependencies: {cran_missing + bioc_missing}. "
                f"{instructions}"
            )

    @classmethod
    def from_script(cls, r_code_or_path=None):
        self = cls()
        self._global_vars["R_SCRIPTS_PATH"] = (
            f"{config.R_SCRIPTS.resolve().as_posix()}/"
        )
        self.r_code = ""
        if r_code_or_path:
            if os.path.exists(r_code_or_path):
                self.r_source = r_code_or_path
                self._global_vars["CURRENT_DIR"] = (
                    f"{Path(self.r_source).resolve().parent.as_posix()}/"
                )
                with open(r_code_or_path, "r") as f:
                    self.r_code += f.read()
            else:
                self.r_code += r_code_or_path
        return self

    def _run_r(
        self,
        code_runner,
        r_code: str,
        env=None,
        add_globals=False,
        quiet=False,
    ) -> object:
        """
        Run R code using the provided code_runner function.

        Args:
            code_runner (Callable): The function to run the R code.
            r_code (str): The R code to run.
            env (SexpEnvironment, optional): The R environment to add global variables
                to. Defaults to globalenv.
            add_globals (bool, optional): Whether to add global variables to the R
                context. Defaults to False.
            quiet (bool, optional): Whether to suppress output from R. Defaults to
                False.

        Returns:
            The output from the R code or the converted output if convert is True.
        """
        from rpy2.rinterface_lib.embedded import RRuntimeError

        def run_code(env):
            if env is None:
                from rpy2.rinterface import globalenv

                env = globalenv
            if add_globals:
                for name, value in self._global_vars.items():
                    if value.endswith('"'):
                        value = value[:-1]
                    if value.startswith('"'):
                        value = value[1:]
                    env[name] = value
            results = code_runner(r_code)
            last_line = r_code.strip().split("\n")[-1].strip()
            # for some reason, arrow tables are not returned properly when letting
            # rpy2 handle the conversion. So we grab it directly from the R context.
            if last_line in env:
                results = env[last_line]
            return results

        if quiet:
            with ROutputCapture() as output_capture:
                try:
                    return run_code(env)
                except RRuntimeError as e:
                    # Attempt to capture the traceback from R.
                    r_stdout = output_capture.get_stdout()
                    r_stderr = output_capture.get_stderr()
                    logger.error(f"captured stdout from R: {r_stdout}")
                    logger.error(f"captured stderr from R: {r_stderr}")
                    try:
                        r_traceback = "\n".join(code_runner("unlist(traceback())"))
                    except Exception as traceback_exc:
                        r_traceback = f"Failed to capture R traceback: {traceback_exc}"
                    raise RuntimeError(f"Error in R code: {e}\n{r_traceback}") from e
        else:
            try:
                return run_code(env)
            except RRuntimeError as e:
                try:
                    r_traceback = "\n".join(code_runner("unlist(traceback())"))
                except Exception as traceback_exc:
                    r_traceback = f"Failed to capture R traceback: {traceback_exc}"
                raise RuntimeError(f"Error in R code: {e}\n{r_traceback}") from e

    def _convert_output(self, obj, code_runner, converter):
        # check if the result is a list
        is_list = converter.rpy2py(
            self._run_r(code_runner, "is.list", quiet=True)(obj)
        )[0]
        if not is_list:
            obj = [obj]

        names = self._run_r(code_runner, "names")(obj)
        has_no_names = converter.rpy2py(self._run_r(code_runner, "is.null")(names))[0]
        if has_no_names:
            names = [f"{i}" for i in range(len(obj))]
        out = {}

        for n, result in zip(names, obj):
            if converter.rpy2py(self._run_r(code_runner, "is.vector")(result))[0]:
                out[n] = np.array([r for r in result])
            elif converter.rpy2py(self._run_r(code_runner, "is.list")(result))[0]:
                out[n] = [r for r in result]
            elif result.__class__ in converter.rpy2py_nc_map and hasattr(
                result, "rclass"
            ):
                nc_map = converter.rpy2py_nc_map[result.__class__]
                for rclass in result.rclass:
                    if rclass in nc_map:
                        out[n] = nc_map[rclass](result)
                        break
                else:
                    out[n] = converter.rpy2py(result)
            else:
                out[n] = converter.rpy2py(result)

        if len(out) == 1:
            out = out[n]
        return out

    def _cleanup(self, code_runner):
        code_runner("rm(list=ls())")
        # garbage collection in Python doesn't always clean up after R automatically
        code_runner("gc()")
        gc.collect()

    def _init_vars(self, code_runner, env, converter, **kwargs):
        import rpy2.robjects as ro
        from rpy2.rinterface import Sexp

        def convert_to_r(arg):
            if isinstance(arg, Sexp):
                return arg
            elif arg is None:
                return ro.NULL
            elif isinstance(arg, (list, tuple)):
                return converter.py2rpy([convert_to_r(a) for a in arg])
            elif isinstance(arg, dict):
                return ro.ListVector(arg)
            else:
                return converter.py2rpy(arg)

        for n, arg in kwargs.items():
            env[n] = convert_to_r(arg)

    def get_method(self, name, enter_code="", exit_code="", convert=True, quiet=True):
        _converter = self._get_converter(name)
        import rpy2.rinterface
        import rpy2.robjects as ro

        with rpy2.rinterface.local_context() as r_context:
            self._run_r(ro.r, self.r_code, add_globals=True, quiet=quiet)
            func_args = list(_converter.rpy2py(r_context[name]).formals().names)
            self._cleanup(ro.r)

        def func(*args, context_kwargs: dict = None, **kwargs):
            if args:
                for n, arg in zip(func_args[: len(args)], args):
                    kwargs[n] = arg

            run_code = self.r_code + "\n" + enter_code + "\n"

            arg_str = ", ".join([f"{k} = {k}" for k in kwargs.keys() if k in func_args])

            # additional variables not in the function signature
            if context_kwargs:
                kwargs.update(context_kwargs)
            kwargs.update(self._global_vars)

            run_code += f"results <- {name}({arg_str})"

            run_code += "\n" + exit_code + "\nresults"

            out = self.run(run_code, convert=convert, quiet=quiet, **kwargs)

            return out

        parameters = [
            inspect.Parameter(name=arg, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for arg in func_args
            if arg != "..."
        ]
        parameters += (
            [inspect.Parameter(name="kwargs", kind=inspect.Parameter.VAR_KEYWORD)]
            if "..." in func_args
            else []
        )
        func.__signature__ = inspect.Signature(parameters=parameters)
        return func

    def run(self, r_code=None, convert=True, quiet=False, **kwargs):
        """A method to run the R code with the provided arguments.

        Args:
            r_code (str, optional): The R code to run. Defaults to the provided R code
                given in the `from_script` method.
            convert (bool, optional): Whether to convert the R output to Python objects.
                Defaults to True.
            **kwargs: The variables to pass to the R code
        """
        r_code = r_code or self.r_code
        if r_code:
            r_code = r_code.strip()
            _converter = self._get_converter(self.run)

            import rpy2.rinterface
            import rpy2.robjects as ro

            with rpy2.rinterface.local_context() as r_context:
                self._init_vars(ro.r, r_context, _converter, **kwargs)
                out = self._run_r(ro.r, r_code, env=r_context, quiet=quiet)
                if out is None:
                    logger.warning("No output from R code")
                if convert:
                    if out is None:
                        raise RuntimeError(
                            "Conversion requested but no output from R code. Either "
                            "provide an output by adding the variable name on the last "
                            "line of the R code or set `convert` to `False`."
                        )
                    out = self._convert_output(out, ro.r, _converter)
                    self._cleanup(ro.r)
                return out
        else:
            raise RuntimeError(
                "R code not provided, please provide R code or the path to the R script"
                " via `from_script` method."
            )
