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

from .extension_class_processor import ExtensionClassProcessor
from .extension_matadata import ExtensionMetadata


class Extension(object, metaclass=ExtensionClassProcessor):
  """The base class for all extensions."""

  # Base class needs to override this class variable to provide its metadata.
  metadata = ExtensionMetadata()

  pass
