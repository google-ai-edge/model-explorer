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

from model_explorer.config_editor import TextInputConfigEditor, TextAreaConfigEditor, SlideToggleConfigEditor, ColorPickerConfigEditor, OptionItem, DropDownConfigEditor, ButtonToggleConfigEditor, FileConfigEditor
from model_explorer import NodeDataProvider, NodeDataProviderMetadata, NodeDataProviderResult, GetConfigEditorsResult
from model_explorer.node_data_builder import GraphNodeData, NodeDataResult, GradientItem
import time
import json


class TestNodeDataProvider(NodeDataProvider):
  metadata = [
      NodeDataProviderMetadata(
          id="test-ndp",
          name="My test node data provider",
          description="A node data provider for testing purpose",
          # Filter example:
          #
          # - # Show this extension only for "tflite" model files.
          #   filter=NodeDataProviderFilter(modelFileExts=["tflite"]),
          # - # Show this extension only for "builtin_tflite_flatbuffer" adapter.
          #   filter=NodeDataProviderFilter(adapterIds=["builtin_tflite_flatbuffer"]),
      ),
      # You can have multiiple NodeDataProviderMetadata here.
  ]

  def get_config_editors(self, provider_id: str) -> GetConfigEditorsResult:
    if provider_id == "test-ndp":
      return GetConfigEditorsResult(
          configEditors=[
              TextInputConfigEditor(
                  id="text_input_1",
                  label="Text input 1",
                  defaultValue="defaul text",
              ),
              TextInputConfigEditor(
                  id="text_input_number",
                  label="Number only",
                  help="must be a number",
                  defaultValue="100",
                  number=True,
              ),
              TextAreaConfigEditor(
                  id="text_area_1", label="Text area 1", height=60
              ),
              SlideToggleConfigEditor(id="toggle", label="A boolean"),
              ColorPickerConfigEditor(id="start_color", label="Start color"),
              ColorPickerConfigEditor(id="end_color", label="End color"),
              DropDownConfigEditor(
                  id="drop_down",
                  label="A dropdown",
                  defaultValue="option_2",
                  options=[
                      OptionItem(label="Option 1", value="option_1"),
                      OptionItem(label="Option 2", value="option_2"),
                      OptionItem(label="Option 3", value="option_3"),
                  ],
              ),
              ButtonToggleConfigEditor(
                  id="button_toggle",
                  label="A button toggle",
                  defaultValue=["gpu"],
                  options=[
                      OptionItem(label="CPU", value="cpu"),
                      OptionItem(label="GPU", value="gpu"),
                      OptionItem(label="NPU", value="npu"),
                  ],
              ),
              ButtonToggleConfigEditor(
                  id="button_toggle_multiple",
                  label="A button toggle (multiple)",
                  defaultValue=["left", "right"],
                  options=[
                      OptionItem(label="Left", value="left"),
                      OptionItem(label="Middle", value="middle"),
                      OptionItem(label="Right", value="right"),
                  ],
                  multiple=True,
              ),
              FileConfigEditor(id="file1", label="File 1"),
          ],
      )
    else:
      return GetConfigEditorsResult(error="Unsupported provider id")

  def run(
      self, provider_id: str, model_path: str, config_values: dict
  ) -> NodeDataProviderResult:
    # Print out the config values user specified in the UI.
    print(json.dumps(config_values, indent=2))

    # Fake delay.
    time.sleep(3)

    if provider_id == "test-ndp":
      # Typically you would use `model_path` and `config_values` to calculate
      # node data. Here for demonstration purpose we just populate values for
      # node id 0-121 (the node ids in coco-ssd.json) with a gradient color.
      results = {}

      # Use config editor id to retrive value.
      gradient_from_color = config_values["start_color"]
      gradient_to_color = config_values["end_color"]
      for i in range(122):
        results[str(i)] = NodeDataResult(value=i)
      return NodeDataProviderResult(
          result=GraphNodeData(
              results=results,
              gradient=[
                  GradientItem(stop=0, bgColor=gradient_from_color),
                  GradientItem(stop=1, bgColor=gradient_to_color),
              ],
          )
      )
    else:
      return NodeDataProviderResult(error="Some error message")
