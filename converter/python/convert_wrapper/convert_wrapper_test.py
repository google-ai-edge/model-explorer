# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for convert_wrapper package."""

import json

from google3.pyglib import resources
from google3.third_party.pybind11_abseil import status
from google3.third_party.tensorflow.compiler.mlir.lite.experimental.google.tooling.python.convert_wrapper import _pywrap_convert_wrapper as convert_wrapper
from google3.third_party.tensorflow.python.framework import test_util
from google3.third_party.tensorflow.python.platform import resource_loader
from google3.third_party.tensorflow.python.platform import test


class ConvertWrapperTest(test_util.TensorFlowTestCase):

  def testConvertSavedModelV1ToJsonSuccess(self):
    model_path = resource_loader.get_path_to_datafile(
        '../../tests/model2json/simple_add'
    )
    config = convert_wrapper.VisualizeConfig()
    actual_json = json.loads(
        convert_wrapper.ConvertSavedModelV1ToJson(config, model_path)
    )
    expected_json = json.loads(
        resources.GetResource(
            'google3/third_party/tensorflow/compiler/mlir/lite/experimental/google/tooling/tests/model2json/output/tf_simple_add_output.json'
        )
    )
    self.assertEqual(expected_json, actual_json)

  def testConvertSavedModelV1ToJsonModelNotFound(self):
    model_path = 'invalid_model_path'
    config = convert_wrapper.VisualizeConfig()
    with self.assertRaises(status.StatusNotOk) as error:
      convert_wrapper.ConvertSavedModelV1ToJson(config, model_path)
    self.assertEqual(error.exception.status.code(), status.StatusCode.NOT_FOUND)
    self.assertIn(
        'Could not find SavedModel',
        error.exception.status.message(),
    )

  def testConvertSavedModelV2ToJsonSuccess(self):
    model_path = resource_loader.get_path_to_datafile(
        '../../tests/model2json/simple_add'
    )
    config = convert_wrapper.VisualizeConfig()
    actual_json = json.loads(
        convert_wrapper.ConvertSavedModelV2ToJson(config, model_path)
    )
    expected_json = json.loads(
        resources.GetResource(
            'google3/third_party/tensorflow/compiler/mlir/lite/experimental/google/tooling/tests/model2json/output/tf_simple_add_v2_output.json'
        )
    )
    self.assertEqual(expected_json, actual_json)

  def testConvertSavedModelV2ToJsonModelNotFound(self):
    model_path = 'invalid_model_path'
    config = convert_wrapper.VisualizeConfig()
    with self.assertRaises(status.StatusNotOk) as error:
      convert_wrapper.ConvertSavedModelV2ToJson(config, model_path)
    self.assertEqual(error.exception.status.code(), status.StatusCode.NOT_FOUND)
    self.assertIn(
        'Could not find SavedModel',
        error.exception.status.message(),
    )

  def testConvertFlatbufferToJsonSuccess(self):
    model_path = resources.GetResourceFilename(
        'google3/third_party/tensorflow/compiler/mlir/lite/experimental/google/tooling/tests/model2json/fully_connected.tflite'
    )
    config = convert_wrapper.VisualizeConfig()
    actual_json = json.loads(
        convert_wrapper.ConvertFlatbufferToJson(config, model_path, True)
    )
    expected_json = json.loads(
        resources.GetResource(
            'google3/third_party/tensorflow/compiler/mlir/lite/experimental/google/tooling/tests/model2json/output/tfl_fully_connected_output.json'
        )
    )
    self.assertEqual(expected_json, actual_json)

  def testConvertFlatbufferToJsonModelNotFound(self):
    model_path = 'invalid_model_path'
    config = convert_wrapper.VisualizeConfig()
    with self.assertRaises(status.StatusNotOk) as error:
      convert_wrapper.ConvertFlatbufferToJson(config, model_path, True)
    self.assertEqual(error.exception.status.code(), status.StatusCode.NOT_FOUND)
    self.assertIn(
        'open failed',
        error.exception.status.message(),
    )


if __name__ == '__main__':
  test.main()
