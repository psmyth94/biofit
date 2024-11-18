import importlib

from biocore.utils.import_util import is_biosets_available


class NotAvailable:
    def __init__(self, *args, **kwargs):
        pass


def get_feature(val):
    if is_biosets_available():
        return getattr(importlib.import_module("biosets.features"), val)
    return None
