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

#include "c/adapter.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "adapters/litert/direct_flatbuffer_to_json_graph_convert.h"
#include "adapters/mlir/model_json_graph_convert.h"
#include "adapters/tensorflow/direct_saved_model_to_json_graph_convert.h"
#include "common/status_reporter.h"
#include "common/visualize_config.h"

namespace {
inline void EnsureInitialized() {}
}  // namespace

namespace {

using ::model_explorer::adapter::VisualizeConfig;

VisualizeConfig ToCppConfig(const AdapterVisualizeConfig* c_config) {
  EnsureInitialized();
  VisualizeConfig config;
  if (c_config != nullptr) {
    config.const_element_count_limit = c_config->const_element_count_limit;
    config.add_tensor_name_attribute = c_config->add_tensor_name_attribute;
  }
  return config;
}

char* DuplicateString(absl::string_view str) {
  char* result = static_cast<char*>(malloc(str.size() + 1));
  if (result != nullptr) {
    memcpy(result, str.data(), str.size());
    result[str.size()] = '\0';
  }
  return result;
}

model_explorer::StatusReporter MakeReporter(
    AdapterProgressCallback progress_callback, void* user_data) {
  if (progress_callback == nullptr) {
    return model_explorer::StatusReporter();
  }
  return model_explorer::StatusReporter(
      [progress_callback, user_data](
          const model_explorer::ConversionProgressReport& report) -> int {
        std::string stage_str(
            model_explorer::LifecycleStageToString(report.stage));
        AdapterProgressReport c_report;
        c_report.struct_size = sizeof(AdapterProgressReport);
        c_report.stage = stage_str.c_str();
        c_report.stage_id = static_cast<int32_t>(report.stage);
        c_report.current_step = report.current_step;
        c_report.total_steps = report.total_steps;
        c_report.progress_fraction = report.progress_fraction;
        c_report.message = report.message.c_str();
        c_report.subgraph_name = report.subgraph_name.empty()
                                     ? nullptr
                                     : report.subgraph_name.c_str();
        c_report.current_op =
            report.current_op.empty() ? nullptr : report.current_op.c_str();
        c_report.elapsed_time_ms = report.elapsed_time_ms;
        c_report.payload_json =
            report.payload_json.empty() ? nullptr : report.payload_json.c_str();
        return progress_callback(&c_report, user_data);
      });
}

AdapterStatusCode HandleResult(
    const absl::StatusOr<std::string>& status_or_json, char** out_json,
    char** out_error_message) {
  EnsureInitialized();
  if (out_json == nullptr || out_error_message == nullptr) {
    return ADAPTER_STATUS_INVALID_ARGUMENT;
  }
  *out_json = nullptr;
  *out_error_message = nullptr;

  if (!status_or_json.ok()) {
    *out_error_message = DuplicateString(status_or_json.status().ToString());
    if (status_or_json.status().code() == absl::StatusCode::kCancelled) {
      return ADAPTER_STATUS_CANCELLED;
    }
    if (status_or_json.status().code() == absl::StatusCode::kNotFound) {
      return ADAPTER_STATUS_NOT_FOUND;
    }
    if (status_or_json.status().code() == absl::StatusCode::kInvalidArgument) {
      return ADAPTER_STATUS_INVALID_ARGUMENT;
    }
    return ADAPTER_STATUS_ERROR;
  }

  *out_json = DuplicateString(*status_or_json);
  return ADAPTER_STATUS_OK;
}

AdapterConvertOptions MakeOptions(AdapterModelFormat format,
                                  const AdapterVisualizeConfig* config,
                                  const char* model_path,
                                  AdapterProgressCallback progress_callback,
                                  void* user_data, bool is_modelpath = true) {
  AdapterConvertOptions options;
  options.struct_size = sizeof(AdapterConvertOptions);
  options.format = format;
  options.model_path = model_path;
  if (config != nullptr) {
    options.config = *config;
  } else {
    adapter_get_default_config(&options.config);
  }
  options.progress_callback = progress_callback;
  options.user_data = user_data;
  options.is_modelpath = is_modelpath;
  return options;
}

}  // namespace

extern "C" {

void adapter_get_default_config(AdapterVisualizeConfig* config) {
  if (config != nullptr) {
    VisualizeConfig default_config;
    config->const_element_count_limit =
        default_config.const_element_count_limit;
    config->add_tensor_name_attribute =
        default_config.add_tensor_name_attribute;
  }
}

void adapter_free_string(char* str) {
  if (str != nullptr) {
    free(str);
  }
}

// Unified conversion entry point supporting all formats, progress tracking,
// and cancellation.
AdapterStatusCode adapter_convert(const AdapterConvertOptions* options,
                                  char** out_json, char** out_error_message) {
  EnsureInitialized();
  if (options == nullptr || out_json == nullptr ||
      out_error_message == nullptr) {
    return ADAPTER_STATUS_INVALID_ARGUMENT;
  }
  if (options->struct_size != sizeof(AdapterConvertOptions)) {
    return ADAPTER_STATUS_INVALID_ARGUMENT;
  }
  if (options->model_path == nullptr) {
    return ADAPTER_STATUS_INVALID_ARGUMENT;
  }

  VisualizeConfig config = ToCppConfig(&options->config);
  auto reporter = MakeReporter(options->progress_callback, options->user_data);

  absl::StatusOr<std::string> result;
  switch (options->format) {
    case ADAPTER_FORMAT_FLATBUFFER: {
      result = ::model_explorer::adapter::ConvertFlatbufferToJson(
          config, options->model_path, options->is_modelpath);
      break;
    }
    case ADAPTER_FORMAT_FLATBUFFER_DIRECT: {
      result = ::model_explorer::adapter::ConvertFlatbufferDirectlyToJson(
          config, options->model_path, &reporter);
      break;
    }
    case ADAPTER_FORMAT_SAVED_MODEL: {
      result = ::model_explorer::adapter::ConvertSavedModelToJson(
          config, options->model_path);
      break;
    }
    case ADAPTER_FORMAT_SAVED_MODEL_DIRECT: {
      result = ::model_explorer::adapter::ConvertSavedModelDirectlyToJson(
          config, options->model_path);
      break;
    }
    case ADAPTER_FORMAT_GRAPH_DEF_DIRECT: {
      result = ::model_explorer::adapter::ConvertGraphDefDirectlyToJson(
          config, options->model_path);
      break;
    }
    case ADAPTER_FORMAT_MLIR: {
      result = ::model_explorer::adapter::ConvertMlirToJson(
          config, options->model_path);
      break;
    }
    default:
      return ADAPTER_STATUS_INVALID_ARGUMENT;
  }

  return HandleResult(result, out_json, out_error_message);
}

// Legacy conversion functions delegating directly to adapter_convert.
AdapterStatusCode adapter_convert_saved_model_to_json(
    const AdapterVisualizeConfig* config, const char* model_path,
    char** out_json, char** out_error_message) {
  AdapterConvertOptions options =
      MakeOptions(ADAPTER_FORMAT_SAVED_MODEL, config, model_path,
                  /*progress_callback=*/nullptr, /*user_data=*/nullptr);
  return adapter_convert(&options, out_json, out_error_message);
}

AdapterStatusCode adapter_convert_flatbuffer_to_json(
    const AdapterVisualizeConfig* config, const char* model_path,
    bool is_modelpath, char** out_json, char** out_error_message) {
  AdapterConvertOptions options = MakeOptions(
      ADAPTER_FORMAT_FLATBUFFER, config, model_path,
      /*progress_callback=*/nullptr, /*user_data=*/nullptr, is_modelpath);
  return adapter_convert(&options, out_json, out_error_message);
}

AdapterStatusCode adapter_convert_flatbuffer_directly_to_json(
    const AdapterVisualizeConfig* config, const char* model_path,
    char** out_json, char** out_error_message) {
  AdapterConvertOptions options =
      MakeOptions(ADAPTER_FORMAT_FLATBUFFER_DIRECT, config, model_path,
                  /*progress_callback=*/nullptr, /*user_data=*/nullptr);
  return adapter_convert(&options, out_json, out_error_message);
}

AdapterStatusCode adapter_convert_saved_model_directly_to_json(
    const AdapterVisualizeConfig* config, const char* model_path,
    char** out_json, char** out_error_message) {
  AdapterConvertOptions options =
      MakeOptions(ADAPTER_FORMAT_SAVED_MODEL_DIRECT, config, model_path,
                  /*progress_callback=*/nullptr, /*user_data=*/nullptr);
  return adapter_convert(&options, out_json, out_error_message);
}

AdapterStatusCode adapter_convert_graph_def_directly_to_json(
    const AdapterVisualizeConfig* config, const char* model_path,
    char** out_json, char** out_error_message) {
  AdapterConvertOptions options =
      MakeOptions(ADAPTER_FORMAT_GRAPH_DEF_DIRECT, config, model_path,
                  /*progress_callback=*/nullptr, /*user_data=*/nullptr);
  return adapter_convert(&options, out_json, out_error_message);
}

AdapterStatusCode adapter_convert_mlir_to_json(
    const AdapterVisualizeConfig* config, const char* model_path,
    char** out_json, char** out_error_message) {
  AdapterConvertOptions options =
      MakeOptions(ADAPTER_FORMAT_MLIR, config, model_path,
                  /*progress_callback=*/nullptr, /*user_data=*/nullptr);
  return adapter_convert(&options, out_json, out_error_message);
}

}  // extern "C"
