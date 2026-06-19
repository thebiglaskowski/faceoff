# Auto-scan all _arch.py files and import them into the hat package registry.
# Based on the upstream HAT hat/archs/__init__.py.
# Vendored for local use; we keep only hat_arch.py here since the HAT repo
# inlines all architectural building blocks into that single file.

import importlib
from os import path as osp

from basicsr.utils import scandir

# automatically scan and import arch modules for registry
# scan all the files that end with '_arch.py' under the archs folder
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]
# import all the arch modules
_arch_modules = [importlib.import_module(f'processing.hat_models.{file_name}') for file_name in arch_filenames]
