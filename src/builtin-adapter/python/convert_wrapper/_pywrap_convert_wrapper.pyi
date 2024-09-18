# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

class VisualizeConfig:
    const_element_count_limit: int
    def __init__(self) -> None: ...

def ConvertFlatbufferDirectlyToJson(arg0: VisualizeConfig, arg1: str) -> str: ...
def ConvertFlatbufferToJson(arg0: VisualizeConfig, arg1: str, arg2: bool) -> str: ...
def ConvertGraphDefDirectlyToJson(arg0: VisualizeConfig, arg1: str) -> str: ...
def ConvertMlirToJson(arg0: VisualizeConfig, arg1: str) -> str: ...
def ConvertSavedModelDirectlyToJson(arg0: VisualizeConfig, arg1: str) -> str: ...
def ConvertSavedModelToJson(arg0: VisualizeConfig, arg1: str) -> str: ...
