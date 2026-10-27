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

#ifndef MODEL_EXPLORER_BACKEND_C_ADAPTER_H_
#define MODEL_EXPLORER_BACKEND_C_ADAPTER_H_

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
  ADAPTER_STATUS_CANCELLED = 4,
} AdapterStatusCode;

// Model format selector for the unified adapter_convert entry point.
typedef enum {
  ADAPTER_FORMAT_UNKNOWN = 0,
  ADAPTER_FORMAT_FLATBUFFER = 1,
  ADAPTER_FORMAT_FLATBUFFER_DIRECT = 2,
  ADAPTER_FORMAT_SAVED_MODEL = 3,
  ADAPTER_FORMAT_SAVED_MODEL_DIRECT = 4,
  ADAPTER_FORMAT_GRAPH_DEF_DIRECT = 5,
  ADAPTER_FORMAT_MLIR = 6,
} AdapterModelFormat;

// Configuration matching model_explorer::adapter::VisualizeConfig
typedef struct {
  int32_t const_element_count_limit;
  bool add_tensor_name_attribute;
} AdapterVisualizeConfig;

// Structured progress report passed to the callback
typedef struct {
  // Size of this struct in bytes for ABI version verification.
  uint32_t struct_size;
  // Lifecycle stage: "starting", "reading_file", "parsing_container_header",
  // "parsing_model", "compiler_passes", "processing_subgraphs",
  // "processing_operations", "building_nodes_and_edges",
  // "transforming_namespaces", "subgraph_validation",
  // "postprocessing_subgraph", "serializing_json", "completed", "failed"
  const char* stage;
  // Numerical ID (0..13) matching LifecycleStage enum
  int32_t stage_id;
  // Step index within the current stage (0-based)
  int64_t current_step;
  // Total steps in the current stage (-1 if indeterminate)
  int64_t total_steps;
  // Overall normalized progress in range [0.0, 1.0] (-1.0 if indeterminate)
  double progress_fraction;
  // Human-readable status message
  const char* message;
  // Active subgraph name (or NULL)
  const char* subgraph_name;
  // Active op label (or NULL)
  const char* current_op;
  // Elapsed time since conversion initiation in milliseconds
  int64_t elapsed_time_ms;
  // Optional JSON metadata / diagnostics string (or NULL)
  const char* payload_json;
} AdapterProgressReport;

// Progress callback function pointer.
// Return 0 to continue conversion; return non-zero (e.g. 1) to request
// cancellation.
typedef int (*AdapterProgressCallback)(const AdapterProgressReport* report,
                                       void* user_data);

// Structured options for the unified adapter_convert function.
typedef struct {
  // Size of this struct in bytes: sizeof(AdapterConvertOptions) for ABI
  // verification.
  uint32_t struct_size;
  // Format / conversion pipeline to invoke.
  AdapterModelFormat format;
  // Path to the model file or directory.
  const char* model_path;
  // Visualization settings.
  AdapterVisualizeConfig config;
  // Optional progress and cancellation callback (or NULL).
  AdapterProgressCallback progress_callback;
  // User data pointer passed to progress_callback (or NULL).
  void* user_data;
  // Legacy option for flatbuffer conversion: true if model_path is a file path.
  bool is_modelpath;
} AdapterConvertOptions;

// Initialize default configuration
ADAPTER_C_API_EXPORT void adapter_get_default_config(
    AdapterVisualizeConfig* config);

// Free strings allocated by the C library (JSON outputs or error messages)
ADAPTER_C_API_EXPORT void adapter_free_string(char* str);

// Unified conversion entry point supporting all model formats, progress
// tracking, and cancellation.
ADAPTER_C_API_EXPORT AdapterStatusCode
adapter_convert(const AdapterConvertOptions* options, char** out_json,
                char** out_error_message);

// Legacy conversion functions.
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

#endif  // MODEL_EXPLORER_BACKEND_C_ADAPTER_H_
