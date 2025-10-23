# Copyright 2025 The AI Edge Model Explorer Authors.
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

from typing import Literal, Union
from dataclasses import dataclass, field


@dataclass
class ConfigEditor:
  """Base class for all config editors."""

  # Editor type.
  type: str = ''

  # Editor unique id.
  id: str = ''

  # The label for this editor when shown in UI.
  #
  # `Id` will be used as fallback label if this field is not specified.
  label: str = ''

  # The description text, shown below the config editor.
  description: str = ''

  # The help text, shown in a popup when a "?" icon is hovered over.
  help: str = ''

  # Whether the config is required or not.
  required: bool = False

  # Default value of the editor.
  defaultValue: Union[str, bool, float, int, None, list[str]] = None


# --- Specialized Editor Configs ---


@dataclass
class TextInputConfigEditor(ConfigEditor):
  """Configuration for a simple text input editor."""

  type: Literal['text_input'] = 'text_input'
  number: bool = False


@dataclass
class TextAreaConfigEditor(ConfigEditor):
  """Configuration for a text area editor."""

  type: Literal['text_area'] = 'text_area'

  # The height of the text area.
  height: float = 40


@dataclass
class SlideToggleConfigEditor(ConfigEditor):
  """Configuration for a slide toggle editor."""

  type: Literal['slide_toggle'] = 'slide_toggle'


@dataclass
class ColorPickerConfigEditor(ConfigEditor):
  """Configuration for a color picker editor."""

  type: Literal['color_picker'] = 'color_picker'


@dataclass
class OptionItem:
  value: str = ''
  label: str = ''


@dataclass
class DropDownConfigEditor(ConfigEditor):
  """Configuration for a drop down editor."""

  type: Literal['drop_down'] = 'drop_down'
  options: list[OptionItem] = field(default_factory=list)


@dataclass
class ButtonToggleConfigEditor(ConfigEditor):
  """Configuration for a button toggle editor."""

  type: Literal['button_toggle'] = 'button_toggle'
  options: list[OptionItem] = field(default_factory=list)
  multiple: bool = False


@dataclass
class FileConfigEditor(ConfigEditor):
  """Configuration for a file upload editor."""

  type: Literal['file'] = 'file'
