# Copyright 2024 The AI Edge Model Explorer Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from importlib.metadata import version

from . import graph_builder, node_data_builder
from .adapter import Adapter, AdapterMetadata
from .apis import config, visualize, visualize_from_config, visualize_pytorch
from .consts import PACKAGE_NAME
from .types import ModelExplorerGraphs

# Default 'exports'.
__all__ = [
    'config',
    'visualize',
    'visualize_pytorch',
    'visualize_from_config',
    'Adapter',
    'AdapterMetadata',
    'ModelExplorerGraphs',
    'graph_builder',
    'node_data_builder',
]

__version__ = version(PACKAGE_NAME)
