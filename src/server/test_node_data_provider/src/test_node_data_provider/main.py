from model_explorer.config_editor import ConfigEditor, TextInputConfigEditor, TextAreaConfigEditor, SlideToggleConfigEditor, ColorPickerConfigEditor, OptionItem, DropDownConfigEditor, ButtonToggleConfigEditor, FileConfigEditor
from model_explorer import NodeDataProvider, NodeDataProviderMetadata, NodeDataProviderResult, GetConfigEditorsResult
from model_explorer.node_data_builder import GraphNodeData, NodeDataResult, GradientItem
import time
import json


class TestNodeDataProvider(NodeDataProvider):
  metadata = [
      NodeDataProviderMetadata(
          id='test-ndp',
          name='My test node data provider',
          description='A node data provider for testing purpose',
          adapterExtensionIds=['tosa_flatbuffer_adapter']),
      NodeDataProviderMetadata(
          id='test-ndp2',
          icon='face',
          name='Another node data provider',
          description='hello node data provider for some other testing purpose',
      )
  ]

  def get_config_editors(self, extension_id: str) -> GetConfigEditorsResult:
    if extension_id == 'test-ndp':
      return GetConfigEditorsResult(
          configEditors=[
              TextInputConfigEditor(
                  id="text_input_1", label='Text input 1', defaultValue="defaul text"),
              TextInputConfigEditor(
                  id="text_input_number", label='Number only', help='must be a number',
                  defaultValue="100", number=True),
              TextAreaConfigEditor(
                  id="text_area_1", label="Text area 1", height=60),
              SlideToggleConfigEditor(id="toggle", label="A boolean"),
              ColorPickerConfigEditor(id='start_color', label="Start color"),
              ColorPickerConfigEditor(id='end_color', label="End color"),
              DropDownConfigEditor(id='drop_down', label="A dropdown",
                                   defaultValue='option_2', options=[
                                       OptionItem(
                                           label="Option 1", value="option_1"),
                                       OptionItem(
                                           label="Option 2", value="option_2"),
                                       OptionItem(
                                           label="Option 3", value="option_3"),
                                   ]),
              ButtonToggleConfigEditor(id='button_toggle', label="A button toggle",
                                       defaultValue=['gpu'], options=[
                                           OptionItem(
                                               label="CPU", value="cpu"),
                                           OptionItem(
                                               label="GPU", value="gpu"),
                                           OptionItem(
                                               label="NPU", value="npu"),
                                       ]),
              ButtonToggleConfigEditor(id='button_toggle_multiple', label="A button toggle (multiple)",
                                       defaultValue=['left', 'right'], options=[
                                           OptionItem(
                                               label="Left", value="left"),
                                           OptionItem(
                                               label="Middle", value="middle"),
                                           OptionItem(
                                               label="Right", value="right"),
                                       ], multiple=True)
          ]
      )
    elif extension_id == 'test-ndp2':
      return GetConfigEditorsResult(
          configEditors=[
              TextInputConfigEditor(id="text_input_1", label='Text input 1'),
              FileConfigEditor(id='file1', label='File 1'),
              FileConfigEditor(id='file2', label='File 2')
          ]
      )
    else:
      return GetConfigEditorsResult(
          error='Unsupported extension id'
      )

  def run(self, extension_id: str, model_path: str, config_values: dict) -> NodeDataProviderResult:
    print(json.dumps(config_values, indent=2))
    time.sleep(3)

    if extension_id == 'test-ndp':
      results = {}
      for i in range(122):
        results[str(i)] = NodeDataResult(value=i)
      return NodeDataProviderResult(
          result=GraphNodeData(
              results=results,
              gradient=[
                  GradientItem(stop=0, bgColor=config_values['start_color']),
                  GradientItem(stop=1, bgColor=config_values['end_color'])]))
    else:
      return NodeDataProviderResult(error='Some error message')
