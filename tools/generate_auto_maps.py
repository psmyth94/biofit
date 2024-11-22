import ast
import os
import re
from collections import defaultdict
from pathlib import Path

from biosets.packaged_modules import EXPERIMENT_TYPE_TO_OMIC_TYPE

OMIC_TYPE_TO_EXPERIMENT_TYPE = defaultdict(list)
ALL_EXPERIMENTS = set()
for experiment_name, omic_type in EXPERIMENT_TYPE_TO_OMIC_TYPE.items():
    OMIC_TYPE_TO_EXPERIMENT_TYPE[omic_type].append(experiment_name)
    ALL_EXPERIMENTS.add(experiment_name)


class ClassFinder(ast.NodeVisitor):
    def __init__(self, known_classes, module_prefix, target_names):
        self.found_classes = {}
        self.known_classes = known_classes
        self.module_prefix = module_prefix
        self.target_names = target_names

    def visit_ClassDef(self, node):
        # Check if any base of the current class is a known ProcessorConfig descendant
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in self.known_classes:
                target_vals = [None] * len(self.target_names)
                # Process each node in the class body
                for body_node in node.body:
                    if isinstance(body_node, (ast.AnnAssign, ast.Assign)):
                        target_name = self.get_target_var_name(body_node)
                        if target_name in self.target_names:
                            index = self.target_names.index(target_name)
                            target_vals[index] = self.extract_name_value(
                                body_node.value
                            )
                self.found_classes[node.name] = (
                    self.module_prefix,
                    base.id,
                    *target_vals,
                )

                # Assume this class is also a config now
                self.known_classes.append(node.name)
        self.generic_visit(node)

    def get_target_var_name(self, node):
        if isinstance(node, ast.AnnAssign):
            return node.target.id
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            return node.targets[0].id
        return None

    def extract_name_value(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Call):
            # Check if the first positional argument or a keyword argument 'default' is set
            if node.args:
                return self.safe_literal_eval(node.args[0])
            for kw in node.keywords:
                if kw.arg == "default":
                    return self.safe_literal_eval(kw.value)
        return None

    def safe_literal_eval(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        return None


def finalize_configs(classes):
    out = {}
    for key, value in classes.items():
        module_prefix, base_id, processor_type, processor_name, experiment_name = value
        if base_id in classes:
            _base_id = base_id
            while not processor_type:
                try:
                    _, _base_id, processor_type, _, _ = classes[_base_id]
                except KeyError:
                    break

            _base_id = base_id
            while not processor_name:
                try:
                    _, _base_id, _, processor_name, _ = classes[_base_id]
                except KeyError:
                    break
        out[key] = (module_prefix, processor_type, processor_name, experiment_name)
    return out


def finalize_processors(classes, config_classes):
    name2proc = {}
    name2config = {}
    name2category = {}
    name2type = {}

    for key, value in classes.items():
        module_prefix, base_id, config_class_name = value
        config_class = (
            config_classes.get(config_class_name, None) if config_class_name else None
        )
        if config_class:
            processor_type, processor_name, experiment_name = config_class[1:]
            if processor_name:
                name2proc[processor_name] = key
                name2config[processor_name] = config_class_name
                processor_category = module_prefix.split(".")[1]
                if processor_type == processor_category:
                    processor_type = None
                if processor_type:
                    name2type[processor_name] = processor_type
                name2category[processor_name] = processor_category

    dataset2config = defaultdict(dict)
    proc2config = defaultdict(list)
    for key, value in config_classes.items():
        module_prefix, processor_type, processor_name, experiment_name = value
        if processor_name:
            proc2config[processor_name].append(
                (module_prefix, key, processor_type, experiment_name)
            )
    for key, values in proc2config.items():
        _dataset2config = {value[3]: value for value in values if value[3]}
        generic_config = [value for value in values if not value[3]][0]
        for experiment_name in ALL_EXPERIMENTS:
            if experiment_name in _dataset2config:
                _, config_class, _, _ = _dataset2config[experiment_name]
                dataset2config[experiment_name][key] = config_class
            else:
                omic_type = EXPERIMENT_TYPE_TO_OMIC_TYPE.get(experiment_name, None)
                if omic_type in _dataset2config:
                    _, config_class, _, _ = _dataset2config[omic_type]
                    dataset2config[experiment_name][key] = config_class
                else:
                    _, config_class, _, _ = generic_config
                    dataset2config[experiment_name][key] = config_class

    return name2proc, name2config, name2category, name2type, dataset2config


def get_all_processor_configs(
    source_folder: str, module_name="biofit", known_classes="ProcessorConfig"
):
    if not isinstance(known_classes, list):
        known_classes = [known_classes]
    _processors = {}
    package_folder = Path(source_folder) / module_name.replace(".", "/")
    for root, dirs, files in os.walk(package_folder.as_posix()):
        relative_root = os.path.relpath(root, start=source_folder)
        module_prefix = relative_root.replace(os.sep, ".")
        for file in files:
            if file.endswith(".py"):
                # Construct the module path relative to the root
                module_path = os.path.join(root, file)
                with open(module_path, "r", encoding="utf-8") as file:
                    try:
                        # Parse the file content into an AST
                        tree = ast.parse(file.read(), filename=module_path)
                        # Initialize the finder and visit the AST
                        finder = ClassFinder(
                            known_classes,
                            module_prefix,
                            ["processor_type", "processor_name", "experiment_name"],
                        )
                        finder.visit(tree)
                        # Collect found processors
                        _processors.update(finder.found_classes)
                    except SyntaxError:
                        print(f"Syntax error in {module_path}, skipping.")

    return finalize_configs(_processors)


def get_all_processors(
    source_folder: str,
    module_name="biofit",
    known_classes="BaseProcessor",
    config_classes=None,
):
    if not isinstance(known_classes, list):
        known_classes = [known_classes]
    _processors = {}
    package_folder = Path(source_folder) / module_name.replace(".", "/")
    for root, dirs, files in os.walk(package_folder.as_posix()):
        relative_root = os.path.relpath(root, start=source_folder)
        module_prefix = relative_root.replace(os.sep, ".")
        for file in files:
            if file.endswith(".py"):
                # Construct the module path relative to the root
                module_path = os.path.join(root, file)
                with open(module_path, "r", encoding="utf-8") as file:
                    try:
                        # Parse the file content into an AST
                        tree = ast.parse(file.read(), filename=module_path)
                        # Initialize the finder and visit the AST
                        finder = ClassFinder(
                            known_classes, module_prefix, ["_config_class"]
                        )
                        finder.visit(tree)
                        # Collect found processors
                        _processors.update(finder.found_classes)
                    except SyntaxError:
                        print(f"Syntax error in {module_path}, skipping.")
    return finalize_processors(_processors, config_classes)


def create_config_mapping_constants(
    name2proc,
    name2config,
    name2category,
    name2type,
    dataset2config,
    name2plotter,
    name2pltconfig,
    dataset2pltconfig,
):
    processing_mapping_names_str = "PROCESSOR_MAPPING_NAMES = OrderedDict(\n    [\n"
    for key, value in name2proc.items():
        processing_mapping_names_str += f'        ("{key}", "{value}"),\n'
    processing_mapping_names_str += "    ]\n)"

    plotter_mapping_names_str = "PLOTTER_MAPPING_NAMES = OrderedDict(\n    [\n"
    for key, value in name2plotter.items():
        plotter_mapping_names_str += f'        ("{key}", "{value}"),\n'
    plotter_mapping_names_str += "    ]\n)"

    config_mapping_names_str = "CONFIG_MAPPING_NAMES = OrderedDict(\n    [\n"
    for key, value in name2config.items():
        config_mapping_names_str += f'        ("{key}", "{value}"),\n'
    config_mapping_names_str += "    ]\n)"

    pltconfig_mapping_names_str = "PLOTTER_CONFIG_MAPPING_NAMES = OrderedDict(\n    [\n"
    for key, value in name2pltconfig.items():
        pltconfig_mapping_names_str += f'        ("{key}", "{value}"),\n'
    pltconfig_mapping_names_str += "    ]\n)"

    processor_category_mapping_names_str = (
        "PROCESSOR_CATEGORY_MAPPING_NAMES = OrderedDict(\n    [\n"
    )
    for key, value in name2category.items():
        processor_category_mapping_names_str += f'        ("{key}", "{value}"),\n'
    processor_category_mapping_names_str += "    ]\n)"

    processor_type_mapping_names_str = (
        "PROCESSOR_TYPE_MAPPING_NAMES = OrderedDict(\n    [\n"
    )
    for key, value in name2type.items():
        processor_type_mapping_names_str += f'        ("{key}", "{value}"),\n'
    processor_type_mapping_names_str += "    ]\n)"

    head_str = processing_mapping_names_str
    head_str += "\n\n"
    head_str += plotter_mapping_names_str
    head_str += "\n\n"
    head_str += config_mapping_names_str
    head_str += "\n\n"
    head_str += pltconfig_mapping_names_str
    head_str += "\n\n"
    head_str += processor_category_mapping_names_str
    head_str += "\n\n"
    head_str += processor_type_mapping_names_str
    head_str += "\n\n"

    regex = re.compile(r"\W")

    for key, value in dataset2config.items():
        mapping_name = f"{regex.sub('_', key).upper()}_MAPPING_NAMES"
        head_str += f"{mapping_name} = OrderedDict(\n    [\n"
        for k, v in value.items():
            head_str += f'        ("{k}", "{v}"),\n'
        head_str += "    ]\n)\n\n"

    for key, value in dataset2pltconfig.items():
        mapping_name = f"{regex.sub('_', key).upper()}_PLOTTER_MAPPING_NAMES"
        head_str += f"{mapping_name} = OrderedDict(\n    [\n"
        for k, v in value.items():
            head_str += f'        ("{k}", "{v}"),\n'
        head_str += "    ]\n)"
        head_str += "\n\n"

    mapping_str = "CONFIG_MAPPING = _LazyConfigMapping(CONFIG_MAPPING_NAMES)\n"
    mapping_str += (
        "PLOTTER_CONFIG_MAPPING = _LazyConfigMapping(PLOTTER_CONFIG_MAPPING_NAMES)\n"
    )

    for key, value in dataset2config.items():
        mapping_name = f"{regex.sub('_', key).upper()}_MAPPING_NAMES"
        mapping_val = f"{regex.sub('_', key).upper()}_MAPPING"
        mapping_str += f"{mapping_val} = _LazyConfigMapping({mapping_name})\n"

    for key, value in dataset2pltconfig.items():
        mapping_name = f"{regex.sub('_', key).upper()}_PLOTTER_MAPPING_NAMES"
        mapping_val = f"{regex.sub('_', key).upper()}_PLOTTER_MAPPING"
        mapping_str += f"{mapping_val} = _LazyConfigMapping({mapping_name})\n"

    ds2mapper = "EXPERIMENT_CONFIG_MAPPING = {"
    for key, value in dataset2config.items():
        mapping_val = f"{regex.sub('_', key).upper()}_CONFIG_MAPPING"
        ds2mapper += f'"{key}": {mapping_val},'
    ds2mapper += "}"

    ds2mapper_names = "EXPERIMENT_CONFIG_MAPPING_NAMES = {"
    for key, value in dataset2config.items():
        mapping_name = f"{regex.sub('_', key).upper()}_CONFIG_MAPPING_NAMES"
        ds2mapper_names += f'"{key}": {mapping_name},'
    ds2mapper_names += "}"

    dsplt2mapper = "EXPERIMENT_PLOTTER_CONFIG_MAPPING = {"
    for key, value in dataset2pltconfig.items():
        mapping_val = f"{regex.sub('_', key).upper()}_PLOTTER_CONFIG_MAPPING"
        dsplt2mapper += f'"{key}": {mapping_val},'
    dsplt2mapper += "}"

    dsplt2mapper_names = "EXPERIMENT_PLOTTER_CONFIG_MAPPING_NAMES = {"
    for key, value in dataset2pltconfig.items():
        mapping_name = f"{regex.sub('_', key).upper()}_PLOTTER_CONFIG_MAPPING_NAMES"
        dsplt2mapper_names += f'"{key}": {mapping_name},'
    dsplt2mapper_names += "}"

    mapping_str += "\n\n"
    mapping_str += ds2mapper
    mapping_str += "\n\n"
    mapping_str += ds2mapper_names
    mapping_str += "\n\n"
    mapping_str += dsplt2mapper
    mapping_str += "\n\n"
    mapping_str += dsplt2mapper_names
    mapping_str += "\n\n"

    return head_str, mapping_str


def replace_mapping(head_str, mapping_str, file_path):
    with open(file_path, "r") as file:
        source_code = file.read()

    tree = ast.parse(source_code)

    head_start_point = None
    head_end_point = None
    mapping_start_point = None
    mapping_end_point = None

    _head_insert_point = 0
    # Find where to end head_str
    for node in ast.walk(tree):
        line = getattr(node, "lineno", None)
        if (
            not head_start_point
            and line is not None
            and not (
                isinstance(node, ast.ImportFrom)
                or isinstance(node, ast.Import)
                or (
                    isinstance(node, ast.If)
                    and isinstance(node.test, ast.Name)
                    and node.test.id == "TYPE_CHECKING"
                )
            )
        ):
            head_start_point = line - 1

        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "config_class_to_processor_name"
        ):
            head_end_point = node.lineno - 1

        if isinstance(node, ast.ClassDef) and node.name == "_LazyConfigMapping":
            mapping_start_point = node.end_lineno + 1

        if isinstance(node, ast.Assign):
            if (
                hasattr(node.targets[0], "id")
                and node.targets[0].id == "EXPERIMENT_PREPROCESSOR_MAPPING_NAMES"
            ):
                mapping_end_point = node.lineno - 1

    splitted_source_code = re.split(r"[\r\n]", source_code)
    new_source_code = (
        splitted_source_code[:head_start_point]
        + [f"\n{head_str}"]
        + splitted_source_code[head_end_point:mapping_start_point]
        + [f"\n{mapping_str}"]
        + splitted_source_code[mapping_end_point:]
    )

    new_source_code = "\n".join(new_source_code)

    with open(file_path, "w") as file:
        file.write(new_source_code)

    file_path = Path(file_path).resolve().as_posix()
    # use ruff to format the code
    os.system(f"ruff format {file_path}")


if __name__ == "__main__":
    config_classes = get_all_processor_configs("../src")
    classes = get_all_processors("../src", config_classes=config_classes)
    plotter_config_classes = get_all_processor_configs(
        "../src", known_classes="PlotterConfig"
    )
    plotter_classes = get_all_processors(
        "../src", config_classes=plotter_config_classes, known_classes="BasePlotter"
    )
    name2proc, name2config, _, _, dataset2config = plotter_classes
    head_str, mapping_str = create_config_mapping_constants(
        *classes, name2proc, name2config, dataset2config
    )

    file_path = "../src/biofit/auto/configuration_auto.py"
    replace_mapping(head_str, mapping_str, file_path)
