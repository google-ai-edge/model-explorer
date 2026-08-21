// Copyright 2026 The AI Edge Model Explorer Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef C_ADAPTER_H_
#define C_ADAPTER_H_

#include <stdbool.h>
#include <stdint.h>

#if defined(_WIN32)
#define ADAPTER_C_API_EXPORT __declspec(dllexport)
#else
#define ADAPTER_C_API_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Status return codes
typedef enum {
  ADAPTER_STATUS_OK = 0,
  ADAPTER_STATUS_ERROR = 1,
  ADAPTER_STATUS_INVALID_ARGUMENT = 2,
  ADAPTER_STATUS_NOT_FOUND = 3,
} AdapterStatusCode;

// Configuration matching tooling::visualization_client::VisualizeConfig
typedef struct {
  int32_t const_element_count_limit;
  bool add_tensor_name_attribute;
} AdapterVisualizeConfig;

// Initialize default configuration
ADAPTER_C_API_EXPORT void adapter_get_default_config(
    AdapterVisualizeConfig* config);

// Free strings allocated by the C library (JSON outputs or error messages)
ADAPTER_C_API_EXPORT void adapter_free_string(char* str);

// Conversion functions
ADAPTER_C_API_EXPORT AdapterStatusCode adapter_convert_saved_model_to_json(
    const AdapterVisualizeConfig* config, const char* model_path,
    char** out_json, char** out_error_message);

ADAPTER_C_API_EXPORT AdapterStatusCode adapter_convert_flatbuffer_to_json(
    const AdapterVisualizeConfig* config, const char* model_path,
    bool is_modelpath, char** out_json, char** out_error_message);

ADAPTER_C_API_EXPORT AdapterStatusCode
adapter_convert_flatbuffer_directly_to_json(
    const AdapterVisualizeConfig* config, const char* model_path,
    char** out_json, char** out_error_message);

ADAPTER_C_API_EXPORT AdapterStatusCode
adapter_convert_saved_model_directly_to_json(
    const AdapterVisualizeConfig* config, const char* model_path,
    char** out_json, char** out_error_message);

ADAPTER_C_API_EXPORT AdapterStatusCode
adapter_convert_graph_def_directly_to_json(const AdapterVisualizeConfig* config,
                                           const char* model_path,
                                           char** out_json,
                                           char** out_error_message);

ADAPTER_C_API_EXPORT AdapterStatusCode adapter_convert_mlir_to_json(
    const AdapterVisualizeConfig* config, const char* model_path,
    char** out_json, char** out_error_message);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // C_ADAPTER_H_
