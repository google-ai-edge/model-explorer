# Copyright 2026 The AI Edge Model Explorer Authors.
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

"""Scoped tensor snapshots and data flow for PyTorch eager inference."""

from ._provenance import Edge
from ._provenance import Op
from ._provenance import Port
from .collector import Capture
from .collector import CaptureError
from .collector import collect
from .collector import Record
from .graph import to_model_explorer
from .graph import unused_ops

__all__ = [
    "Capture",
    "CaptureError",
    "Edge",
    "Op",
    "Port",
    "Record",
    "collect",
    "to_model_explorer",
    "unused_ops",
]
