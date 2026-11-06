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

#include "model_debugger/runner/llm/litert_lm/litert_lm_runtime.h"

#include <dlfcn.h>

#include <atomic>
#include <cmath>
#include <condition_variable>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <system_error>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "google/protobuf/struct.pb.h"
#include "litert_lm/version.h"
#include "model_debugger/protocol/runner.pb.h"
#include "model_debugger/runner/llm/litert_lm/capture_files.h"
#include "model_debugger/runner/llm/litert_lm/fed_tokens.h"
#include "model_debugger/runner/llm/litert_lm/gpu_plugins.h"
#include "model_debugger/runner/llm/litert_lm/ml_drift_kv_cache.h"
#include "model_debugger/runner/llm/runtime.h"
#include "model_debugger/runner/util/json.h"
#include "third_party/LiteRT-LM/c/conversation.h"
#include "third_party/LiteRT-LM/c/engine.h"
#include "third_party/LiteRT-LM/c/error_reporter.h"
#include "third_party/LiteRT-LM/c/experimental.h"
#include "third_party/LiteRT-LM/c/model_info.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

namespace model_debugger::runner::llm {
namespace {

// The C API takes token ids as int.
static_assert(sizeof(int) == sizeof(int32_t));

// ---- The C API's status codes

// Returns OK for kLiteRtLmStatusOk, or else an Internal error: `what` failed,
// with the runtime's code and message. Internal whatever the runtime's code:
// the Runner's codes are the protocol's, where the runtime's would mean
// something else (InvalidArgument, a message refused with nothing changed).
// Called on the thread that made the call, whose last error the message is.
absl::Status Check(LiteRtLmStatusCode code, std::string_view what) {
  if (code == kLiteRtLmStatusOk) return absl::OkStatus();
  // The codes are absl's (error_reporter.h); a newer one is Unknown here.
  const absl::StatusCode canonical = code <= kLiteRtLmStatusUnauthenticated
                                         ? static_cast<absl::StatusCode>(code)
                                         : absl::StatusCode::kUnknown;
  const char* message = litert_lm_get_last_error_message();
  return absl::InternalError(
      absl::StrCat(what, " (", absl::StatusCodeToString(canonical), ": ",
                   message != nullptr ? message : "no message", ")."));
}

// ---- What this runtime declares

// The engine settings Load takes beyond device and contextLength, as a JSON
// Schema the server and the service validate LoadRequest.runtime_options
// against.
constexpr std::string_view kLoadOptionsSchema = R"({
  "type": "object",
  "properties": {
    "cpuThreads": {"type": "integer", "minimum": 1},
    "audioCpuThreads": {"type": "integer", "minimum": 1},
    "prefillChunkSize": {"type": "integer", "minimum": 1},
    "loraRank": {"type": "integer", "minimum": 0},
    "audioLoraRank": {"type": "integer", "minimum": 0},
    "enableYnnpack": {"type": "boolean"},
    "parallelFileSectionLoading": {"type": "boolean"},
    "singleThreadedExecution": {"type": "boolean"},
    "gpuMetalResidencySet": {"type": "boolean"},
    "visionBackend": {"enum": ["CPU", "GPU"]},
    "audioBackend": {"enum": ["CPU", "GPU"]},
    "activationDataType": {"enum": ["float32", "float16", "int16", "int8"]}
  },
  "additionalProperties": false
})";

// The session and conversation settings a chat takes beyond the fixed ones,
// as a JSON Schema the server and the service validate
// ChatOptions.runtime_options against.
constexpr std::string_view kChatOptionsSchema = R"({
  "type": "object",
  "properties": {
    "thinking": {"type": "boolean"},
    "thinking_token_budget": {"type": "integer", "minimum": -1},
    "apply_prompt_template": {"type": "boolean"},
    "speculative_decoding": {"type": "boolean"}
  },
  "additionalProperties": false
})";

// The runtime's version is the commit it was built from: its C API reports
// none, and the build wrote it down.
protocol::RuntimeInfo MakeInfo() {
  protocol::RuntimeInfo info;
  info.set_name(protocol::RuntimeInfo::LITERT_LM);
  info.set_version(litert_lm::kCommit);
  info.set_load_options_schema(std::string(kLoadOptionsSchema));
  info.set_chat_options_schema(std::string(kChatOptionsSchema));
  return info;
}

// ---- Runtime options, from a Struct

using google::protobuf::Struct;
using google::protobuf::Value;

absl::StatusOr<bool> AsBool(std::string_view name, const Value& value) {
  if (value.kind_case() != Value::kBoolValue) {
    return absl::InvalidArgumentError(
        absl::StrCat("Option ", name, " is not a boolean."));
  }
  return value.bool_value();
}

// Returns `value` as the int the C API takes: a whole number in range.
absl::StatusOr<int> AsInt(std::string_view name, const Value& value) {
  if (value.kind_case() != Value::kNumberValue) {
    return absl::InvalidArgumentError(
        absl::StrCat("Option ", name, " is not a number."));
  }
  const double number = value.number_value();
  if (std::trunc(number) != number ||
      number < std::numeric_limits<int>::min() ||
      number > std::numeric_limits<int>::max()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Option ", name, " is not a whole number in range."));
  }
  return static_cast<int>(number);
}

absl::StatusOr<std::string> AsString(std::string_view name,
                                     const Value& value) {
  if (value.kind_case() != Value::kStringValue) {
    return absl::InvalidArgumentError(
        absl::StrCat("Option ", name, " is not a string."));
  }
  return value.string_value();
}

// Returns the name LiteRT-LM gives `device` as a backend.
absl::StatusOr<const char*> BackendName(protocol::Device device) {
  switch (device) {
    case protocol::CPU:
      return "cpu";
    case protocol::GPU:
      return "gpu";
    default:
      return absl::InvalidArgumentError("The device is not CPU or GPU.");
  }
}

// Returns the device named by option `name`, as the protocol spells it.
absl::StatusOr<protocol::Device> DeviceOf(std::string_view name,
                                          const Value& value) {
  absl::StatusOr<std::string> text = AsString(name, value);
  if (!text.ok()) return text.status();
  if (*text == "CPU") return protocol::CPU;
  if (*text == "GPU") return protocol::GPU;
  return absl::InvalidArgumentError(
      absl::StrCat("Option ", name, " is ", *text, ", not CPU or GPU."));
}

absl::StatusOr<LiteRtLmActivationDataType> ActivationOf(std::string_view name,
                                                        const Value& value) {
  absl::StatusOr<std::string> text = AsString(name, value);
  if (!text.ok()) return text.status();
  if (*text == "float32") return kLiteRtLmActivationDataTypeFloat32;
  if (*text == "float16") return kLiteRtLmActivationDataTypeFloat16;
  if (*text == "int16") return kLiteRtLmActivationDataTypeInt16;
  if (*text == "int8") return kLiteRtLmActivationDataTypeInt8;
  return absl::InvalidArgumentError(
      absl::StrCat("Option ", name, " is ", *text,
                   ", not float32, float16, int16 or int8."));
}

// The engine settings of Load, parsed and checked.
struct EngineOptions {
  std::optional<int> cpu_threads;
  std::optional<int> audio_cpu_threads;
  std::optional<int> prefill_chunk_size;
  std::optional<int> lora_rank;
  std::optional<int> audio_lora_rank;
  std::optional<bool> enable_ynnpack;
  std::optional<bool> parallel_file_section_loading;
  std::optional<bool> single_threaded_execution;
  std::optional<bool> gpu_metal_residency_set;
  std::optional<protocol::Device> audio_backend;
  std::optional<protocol::Device> vision_backend;
  std::optional<LiteRtLmActivationDataType> activation_data_type;
};

// Returns a function that reads `value`, the option `name`, with a reader into
// a field, or records why not in `status`.
auto Taker(const std::string& name, const Value& value, absl::Status& status) {
  return [&](auto& into, auto&& reader) {
    auto read = reader(name, value);
    if (read.ok()) {
      into = *read;
    } else {
      status = read.status();
    }
  };
}

// Reads `options` into an EngineOptions, refusing a name this runtime does
// not have and a value of the wrong type: what kLoadOptionsSchema says.
absl::StatusOr<EngineOptions> ParseEngineOptions(const Struct& options) {
  EngineOptions parsed;
  for (const auto& field : options.fields()) {
    const std::string& name = field.first;
    const Value& value = field.second;
    absl::Status status;
    auto take = Taker(name, value, status);
    if (name == "cpuThreads") {
      take(parsed.cpu_threads, AsInt);
    } else if (name == "audioCpuThreads") {
      take(parsed.audio_cpu_threads, AsInt);
    } else if (name == "prefillChunkSize") {
      take(parsed.prefill_chunk_size, AsInt);
    } else if (name == "loraRank") {
      take(parsed.lora_rank, AsInt);
    } else if (name == "audioLoraRank") {
      take(parsed.audio_lora_rank, AsInt);
    } else if (name == "enableYnnpack") {
      take(parsed.enable_ynnpack, AsBool);
    } else if (name == "parallelFileSectionLoading") {
      take(parsed.parallel_file_section_loading, AsBool);
    } else if (name == "singleThreadedExecution") {
      take(parsed.single_threaded_execution, AsBool);
    } else if (name == "gpuMetalResidencySet") {
      take(parsed.gpu_metal_residency_set, AsBool);
    } else if (name == "audioBackend") {
      take(parsed.audio_backend, DeviceOf);
    } else if (name == "visionBackend") {
      take(parsed.vision_backend, DeviceOf);
    } else if (name == "activationDataType") {
      take(parsed.activation_data_type, ActivationOf);
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("LiteRT-LM has no load option ", name, "."));
    }
    if (!status.ok()) return status;
  }
  return parsed;
}

// Calls `setter` with `value`, the setting `name`, if there is one. The
// setting and the setter share one type T, so that an int setting cannot
// reach a setter of bool.
template <typename T>
absl::Status Set(LiteRtLmEngineSettings* to, std::string_view name,
                 const std::optional<T>& value,
                 LiteRtLmStatusCode (*setter)(LiteRtLmEngineSettings*, T)) {
  if (!value) return absl::OkStatus();
  return Check(setter(to, *value), absl::StrCat("Could not set ", name));
}

// Passes the request and the parsed options to `to`, in the order of the
// setters in third_party/LiteRT-LM/c/engine.h. Returns the first failure.
absl::Status Apply(const protocol::LoadRequest& request,
                   const std::string& data_dir, const EngineOptions& options,
                   LiteRtLmEngineSettings* to) {
  absl::Status status;
  status.Update(Set(to, "context_length",
                    std::optional<int>(request.context_length()),
                    litert_lm_engine_settings_set_max_num_tokens));
  status.Update(Set(to, "cpuThreads", options.cpu_threads,
                    litert_lm_engine_settings_set_num_threads));
  status.Update(Set(to, "audioCpuThreads", options.audio_cpu_threads,
                    litert_lm_engine_settings_set_audio_num_threads));
  status.Update(Set(
      to, "parallelFileSectionLoading", options.parallel_file_section_loading,
      litert_lm_engine_settings_set_parallel_file_section_loading));
  status.Update(Set(to, "singleThreadedExecution",
                    options.single_threaded_execution,
                    litert_lm_engine_settings_set_single_threaded_execution));
  // The runtime's cache directory is the Runner's data directory: its
  // compiled caches and its captures both land there.
  status.Update(Set(to, "the cache directory",
                    std::optional<const char*>(data_dir.c_str()),
                    litert_lm_engine_settings_set_cache_dir));
  status.Update(Set(to, "activationDataType", options.activation_data_type,
                    litert_lm_engine_settings_set_activation_data_type));
  status.Update(Set(to, "prefillChunkSize", options.prefill_chunk_size,
                    litert_lm_engine_settings_set_prefill_chunk_size));
  status.Update(Set(to, "enableYnnpack", options.enable_ynnpack,
                    litert_lm_engine_settings_set_enable_ynnpack));
  // Always on: what a turn cost (Usage) is read from its counters.
  status.Update(Check(litert_lm_engine_settings_enable_benchmark(to),
                      "Could not enable the benchmark"));
  status.Update(Set(to, "loraRank", options.lora_rank,
                    litert_lm_engine_settings_set_lora_rank));
  status.Update(Set(to, "audioLoraRank", options.audio_lora_rank,
                    litert_lm_engine_settings_set_audio_lora_rank));
  status.Update(
      Set(to, "gpuMetalResidencySet", options.gpu_metal_residency_set,
          litert_lm_engine_settings_set_gpu_enable_metal_residency_set));
  return status;
}

// The session and conversation settings of a chat, parsed and checked.
struct ChatRuntimeOptions {
  std::optional<bool> thinking;
  std::optional<int> thinking_token_budget;
  std::optional<bool> apply_prompt_template;
  // False unless given: the Debugger pairs one token with one step.
  bool speculative_decoding = false;
};

absl::StatusOr<ChatRuntimeOptions> ParseChatOptions(const Struct& options) {
  ChatRuntimeOptions parsed;
  for (const auto& field : options.fields()) {
    const std::string& name = field.first;
    const Value& value = field.second;
    absl::Status status;
    auto take = Taker(name, value, status);
    if (name == "thinking") {
      take(parsed.thinking, AsBool);
    } else if (name == "thinking_token_budget") {
      take(parsed.thinking_token_budget, AsInt);
    } else if (name == "apply_prompt_template") {
      take(parsed.apply_prompt_template, AsBool);
    } else if (name == "speculative_decoding") {
      take(parsed.speculative_decoding, AsBool);
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("LiteRT-LM has no chat option ", name, "."));
    }
    if (!status.ok()) return status;
  }
  return parsed;
}

// ---- Chat messages, as the conversation API takes them

absl::StatusOr<const char*> RoleName(protocol::ChatMessage::Role role) {
  switch (role) {
    case protocol::ChatMessage::SYSTEM:
      return "system";
    case protocol::ChatMessage::USER:
      return "user";
    case protocol::ChatMessage::ASSISTANT:
      return "assistant";
    default:
      return absl::InvalidArgumentError(
          "A chat message's role is not system, user or assistant.");
  }
}

// Returns `message` as the conversation API takes it.
absl::StatusOr<Json> MessageJson(const protocol::ChatMessage& message) {
  absl::StatusOr<const char*> role = RoleName(message.role());
  if (!role.ok()) return role.status();
  return Json{{"role", *role}, {"content", message.content()}};
}

// Returns `message` as the conversation API takes it, as JSON text.
absl::StatusOr<std::string> MessageText(const protocol::ChatMessage& message) {
  absl::StatusOr<Json> json = MessageJson(message);
  if (!json.ok()) return json.status();
  return Dump(*json);
}

// ---- A reply as it streams

// What the runtime's thread writes and Send waits for.
struct Stream {
  explicit Stream(const ChunkCallback& callback) : on_chunk(callback) {}

  const ChunkCallback& on_chunk;
  std::mutex mutex;  // NOLINT(build/c++11)
  std::condition_variable done;
  protocol::Completion completion;
  absl::Status status;
  bool finished = false;
};

// Gives `text` to the stream's callback as a chunk of the answer or of the
// reasoning, and keeps it for the completion.
void Deliver(Stream& stream, bool reasoning, std::string_view text) {
  if (text.empty()) return;
  protocol::Chunk chunk;
  if (reasoning) {
    stream.completion.mutable_reasoning_content()->append(text);
    chunk.set_reasoning_content(std::string(text));
  } else {
    stream.completion.mutable_content()->append(text);
    chunk.set_content(std::string(text));
  }
  if (stream.on_chunk) stream.on_chunk(chunk);
}

// Reads one chunk of a reply, a message as JSON text: `content` as a string or
// as parts of type "text", and the model's reasoning in `reasoning_content`.
//
// Returns Internal for text that is not JSON, and for a part of another type:
// it is output, and dropping it would be a piece missing from the evidence.
absl::Status Take(std::string_view json, Stream& stream) {
  absl::StatusOr<Json> message = ParseJson(json);
  if (!message.ok()) {
    return absl::InternalError(absl::StrCat("The runtime's reply is not JSON: ",
                                            message.status().message()));
  }
  if (const auto content = message->find("content");
      content != message->end()) {
    if (content->is_string()) {
      Deliver(stream, /*reasoning=*/false,
              content->get_ref<const std::string&>());
    } else if (content->is_array()) {
      for (const Json& part : *content) {
        if (!part.is_object()) {
          return absl::InternalError("A part of the reply is not an object.");
        }
        const auto type = part.find("type");
        if (type == part.end() || !type->is_string()) {
          return absl::InternalError("A part of the reply has no type.");
        }
        if (type->get_ref<const std::string&>() != "text") {
          return absl::InternalError(absl::StrCat(
              "A part of the reply is of type ",
              type->get_ref<const std::string&>(), ", which is not text."));
        }
        if (const auto text = part.find("text");
            text != part.end() && text->is_string()) {
          Deliver(stream, /*reasoning=*/false,
                  text->get_ref<const std::string&>());
        }
      }
    } else if (!content->is_null()) {
      return absl::InternalError("The reply's content is not text.");
    }
  }
  if (const auto reasoning = message->find("reasoning_content");
      reasoning != message->end() && reasoning->is_string()) {
    Deliver(stream, /*reasoning=*/true,
            reasoning->get_ref<const std::string&>());
  }
  return absl::OkStatus();
}

// The callback the runtime calls on its thread with each chunk.
void OnChunk(void* data, const LiteRtLmStreamChunk* chunk) {
  Stream& stream = *static_cast<Stream*>(data);
  std::lock_guard<std::mutex> lock(stream.mutex);  // NOLINT(build/c++11)
  if (stream.finished) return;
  const char* error = nullptr;
  const char* text = nullptr;
  bool last = false;
  stream.status.Update(Check(litert_lm_stream_chunk_get_error(chunk, &error),
                             "Could not read a chunk's error"));
  stream.status.Update(Check(litert_lm_stream_chunk_get_text(chunk, &text),
                             "Could not read a chunk's text"));
  stream.status.Update(Check(litert_lm_stream_chunk_is_final(chunk, &last),
                             "Could not read whether a chunk is the last"));
  if (error != nullptr) stream.status.Update(absl::InternalError(error));
  if (stream.status.ok() && text != nullptr) stream.status = Take(text, stream);
  if (last) {
    stream.finished = true;
    stream.done.notify_one();
  }
}

// ---- Tokens, through the engine

absl::StatusOr<std::vector<int32_t>> Tokenize(LiteRtLmEngine* engine,
                                              const std::string& text) {
  LiteRtLmTokenizeResult* result = nullptr;
  if (absl::Status status =
          Check(litert_lm_engine_tokenize(engine, text.c_str(), &result),
                "Could not tokenize the prompt");
      !status.ok()) {
    return status;
  }
  const int* tokens = nullptr;
  size_t count = 0;
  absl::Status status;
  status.Update(Check(litert_lm_tokenize_result_get_tokens(result, &tokens),
                      "Could not read the prompt's tokens"));
  status.Update(Check(litert_lm_tokenize_result_get_num_tokens(result, &count),
                      "Could not count the prompt's tokens"));
  if (status.ok() && tokens == nullptr && count > 0) {
    status = absl::InternalError("Runtime counted tokens it did not give.");
  }
  std::vector<int32_t> ids;
  if (status.ok() && tokens != nullptr) {
    ids.assign(tokens, tokens + count);
  }
  litert_lm_tokenize_result_delete(result);
  if (!status.ok()) return status;
  return ids;
}

absl::StatusOr<std::string> Detokenize(LiteRtLmEngine* engine, int32_t id) {
  LiteRtLmDetokenizeResult* result = nullptr;
  if (absl::Status status =
          Check(litert_lm_engine_detokenize(engine, &id, 1, &result),
                absl::StrCat("Could not detokenize token ", id));
      !status.ok()) {
    return status;
  }
  const char* text = nullptr;
  const absl::Status status =
      Check(litert_lm_detokenize_result_get_string(result, &text),
            absl::StrCat("Could not read the text of token ", id));
  std::string copy = text != nullptr ? text : "";
  litert_lm_detokenize_result_delete(result);
  if (!status.ok()) return status;
  return copy;
}

// Returns `token` as the protocol's StopToken has it: by its ids, or by its
// text if it is configured as text.
absl::StatusOr<protocol::StopToken> ReadToken(const LiteRtLmTokenUnion* token) {
  LiteRtLmTokenUnionType type;
  if (absl::Status status = Check(litert_lm_token_union_get_type(token, &type),
                                  "Could not read a token's type");
      !status.ok()) {
    return status;
  }
  protocol::StopToken read;
  if (type == kLiteRtLmTokenUnionTypeIds) {
    const int* ids = nullptr;
    size_t count = 0;
    if (absl::Status status =
            Check(litert_lm_token_union_get_ids(token, &ids, &count),
                  "Could not read a token's ids");
        !status.ok()) {
      return status;
    }
    if (ids == nullptr && count > 0) {
      return absl::InternalError("Runtime counted token ids it did not give.");
    }
    read.mutable_ids()->mutable_ids()->Assign(ids, ids + count);
    return read;
  }
  const char* text = nullptr;
  if (absl::Status status =
          Check(litert_lm_token_union_get_string(token, &text),
                "Could not read a token's text");
      !status.ok()) {
    return status;
  }
  if (text == nullptr) {
    return absl::InternalError(
        "Runtime named a token by text it did not give.");
  }
  read.set_text(text);
  return read;
}

// Reads the model's start and stop tokens into a LoadResponse.
absl::StatusOr<protocol::LoadResponse> ReadTokens(LiteRtLmEngine* engine) {
  protocol::LoadResponse response;
  // Null, for a model without one.
  LiteRtLmTokenUnion* start = nullptr;
  if (absl::Status status =
          Check(litert_lm_engine_get_start_token(engine, &start),
                "Could not read the start token");
      !status.ok()) {
    return status;
  }
  if (start != nullptr) {
    const absl::StatusOr<protocol::StopToken> token = ReadToken(start);
    litert_lm_token_union_delete(start);
    if (!token.ok()) return token.status();
    if (!token->has_ids() || token->ids().ids_size() != 1) {
      return absl::InternalError("Runtime start token is not a single id.");
    }
    response.set_start_token(token->ids().ids(0));
  }
  // Null, for a model without any.
  LiteRtLmTokenUnions* unions = nullptr;
  if (absl::Status status =
          Check(litert_lm_engine_get_stop_tokens(engine, &unions),
                "Could not read the stop tokens");
      !status.ok()) {
    return status;
  }
  if (unions == nullptr) return response;
  size_t count = 0;
  absl::Status status =
      Check(litert_lm_token_unions_get_num_tokens(unions, &count),
            "Could not count the stop tokens");
  for (size_t i = 0; i < count && status.ok(); ++i) {
    LiteRtLmTokenUnion* token = nullptr;
    status = Check(litert_lm_token_unions_get_token_at(unions, i, &token),
                   "Could not read a stop token");
    if (!status.ok()) break;
    absl::StatusOr<protocol::StopToken> stop = ReadToken(token);
    litert_lm_token_union_delete(token);
    if (stop.ok()) {
      *response.add_stop_tokens() = *std::move(stop);
    } else {
      status = stop.status();
    }
  }
  litert_lm_token_unions_delete(unions);
  if (!status.ok()) return status;
  return response;
}

// The sampling a chat gets for what its options leave unset. The defaults
// are LiteRT-LM's for a model that declares none, on CPU and GPU
// (runtime/engine/engine_settings.cc).
struct Sampling {
  int top_k = 1;
  float top_p = 0.95f;
  float temperature = 1.0f;
  // The C API does not say a model's seed; LiteRT-LM's is 0.
  int seed = 0;
};

// Returns the sampling the model at `model` sets for a chat that sets none,
// as the runtime reports it: a value it reports as 0 or unspecified, which the
// C API says the model does not set, is unset; nullopt where it reports none,
// such as for a model with no LLM metadata.
absl::StatusOr<std::optional<protocol::SamplerDefaults>> ReadSamplerDefaults(
    const std::filesystem::path& model) {
  LiteRtLmLoadedFile* file = nullptr;
  if (absl::Status status = Check(
          litert_lm_loaded_file_create(model.c_str(), &file),
          absl::StrCat("Could not read the metadata of ", model.string()));
      !status.ok()) {
    return status;
  }
  LiteRtLmSamplerType type = kLiteRtLmSamplerTypeUnspecified;
  int32_t top_k = 0;
  float top_p = 0;
  float temperature = 0;
  // NotFound: the model has no LLM metadata, and so declares no sampling.
  const LiteRtLmStatusCode typed =
      litert_lm_loaded_file_sampler_type(file, &type);
  absl::Status status;
  if (typed != kLiteRtLmStatusNotFound) {
    status.Update(Check(typed, "Could not read the model's sampler type"));
    status.Update(Check(litert_lm_loaded_file_sampler_top_k(file, &top_k),
                        "Could not read the model's top_k"));
    status.Update(Check(litert_lm_loaded_file_sampler_top_p(file, &top_p),
                        "Could not read the model's top_p"));
    status.Update(
        Check(litert_lm_loaded_file_sampler_temperature(file, &temperature),
              "Could not read the model's temperature"));
  }
  litert_lm_loaded_file_delete(file);
  if (!status.ok()) return status;
  if (typed == kLiteRtLmStatusNotFound) return std::nullopt;
  protocol::SamplerDefaults defaults;
  switch (type) {
    case kLiteRtLmSamplerTypeGreedy:
      defaults.set_type(protocol::SamplerDefaults::GREEDY);
      break;
    case kLiteRtLmSamplerTypeTopK:
      defaults.set_type(protocol::SamplerDefaults::TOP_K);
      break;
    case kLiteRtLmSamplerTypeTopP:
      defaults.set_type(protocol::SamplerDefaults::TOP_P);
      break;
    default:  // Unspecified.
      break;
  }
  if (top_k > 0) defaults.set_top_k(top_k);
  if (top_p > 0) defaults.set_top_p(top_p);
  if (temperature > 0) defaults.set_temperature(temperature);
  // Nothing set serializes to nothing.
  if (defaults.ByteSizeLong() == 0) return std::nullopt;
  return defaults;
}

// Returns the sampling a chat gets for what its options leave unset, from the
// model's own in `info`, or LiteRT-LM's where the model sets none.
Sampling SamplingOf(const protocol::LoadResponse& info) {
  Sampling sampling;
  if (!info.has_sampler_defaults()) return sampling;
  const protocol::SamplerDefaults& defaults = info.sampler_defaults();
  // The runtime maps every sampling onto its top-p sampler: a value the model
  // sets replaces the default, one it does not set does not. Greedy is a
  // top_k of 1; top-k alone is a top_p of 1, which cuts nothing.
  if (defaults.has_top_k()) sampling.top_k = defaults.top_k();
  if (defaults.has_top_p()) {
    sampling.top_p = defaults.top_p();
  } else if (defaults.type() == protocol::SamplerDefaults::TOP_K) {
    sampling.top_p = 1.0f;
  }
  if (defaults.has_temperature()) sampling.temperature = defaults.temperature();
  if (defaults.type() == protocol::SamplerDefaults::GREEDY) sampling.top_k = 1;
  return sampling;
}

// ---- A turn's capture

// The file the runtime appends each released token to, in the capture
// directory.
constexpr std::string_view kTokensFile = "generated_tokens.jsonl";

// The regular files of a capture directory, each with when it was last
// written.
using Listing = std::map<std::string, std::filesystem::file_time_type>;

// Lists `dir`; a directory not there yet holds nothing.
Listing ListFiles(const std::filesystem::path& dir) {
  Listing files;
  std::error_code error;
  for (const auto& entry : std::filesystem::directory_iterator(dir, error)) {
    if (!entry.is_regular_file(error)) continue;
    const std::filesystem::file_time_type written =
        entry.last_write_time(error);
    if (!error) files.emplace(entry.path().filename().string(), written);
  }
  return files;
}

// Returns the names of the files in `dir` written since `before` listed it:
// new ones, and ones written again, as a step's files are when a turn feeds
// a position again that a turn before it fed.
std::set<std::string> WrittenSince(const std::filesystem::path& dir,
                                   const Listing& before) {
  std::set<std::string> names;
  for (const auto& [name, written] : ListFiles(dir)) {
    const auto it = before.find(name);
    if (it == before.end() || it->second != written) names.insert(name);
  }
  return names;
}

// Returns the number of lines in `file`, 0 if it is not there.
int CountLines(const std::filesystem::path& file) {
  std::ifstream in(file);
  int lines = 0;
  std::string line;
  while (std::getline(in, line)) ++lines;
  return lines;
}

// Returns the ResourceExhausted of `message` for `needed` tokens where the
// context has `left`, with the payloads that say so.
absl::Status NoRoom(std::string_view message, int needed, int left) {
  absl::Status refused = absl::ResourceExhaustedError(message);
  refused.SetPayload(kTurnTokensPayload, absl::Cord(absl::StrCat(needed)));
  refused.SetPayload(kTokensLeftPayload, absl::Cord(absl::StrCat(left)));
  return refused;
}

// ---- Chat

class LiteRtLmChat final : public Chat {
 public:
  LiteRtLmChat(LiteRtLmConversation* handle, LiteRtLmEngine* engine,
               std::filesystem::path data_dir, int max_completion_tokens,
               int context_length, bool gpu, std::optional<int32_t> start_token,
               std::set<int32_t> stop_tokens)
      : handle_(handle),
        engine_(engine),
        data_dir_(std::move(data_dir)),
        max_completion_tokens_(max_completion_tokens),
        context_length_(context_length),
        gpu_(gpu),
        start_token_(start_token),
        stop_tokens_(std::move(stop_tokens)) {}
  ~LiteRtLmChat() override { litert_lm_conversation_delete(handle_); }

  absl::StatusOr<Turn> Send(const protocol::ChatMessage& message,
                            const ChunkCallback& on_chunk) override;
  void Cancel() override;

  // OK if the context has room for the chat's starting messages: the first
  // turn's prompt for an empty user message, which holds them in the chat
  // template. Else ResourceExhausted, with the payloads Send's has.
  absl::Status CheckRoomToStart() const;

 private:
  // Returns the prompt the model receives for `message`: what the chat has
  // not yet fed it, in the chat template, as the runtime tokenizes it.
  absl::StatusOr<std::vector<int32_t>> RenderPrompt(
      const std::string& message) const;

  // Returns the tokens the turn fed the model, read from its capture in
  // `capture_dir`: the files `names` and the token lines past the first
  // `lines_before`, with the texts of their ids.
  absl::StatusOr<FedTurn> FedTokens(const std::filesystem::path& capture_dir,
                                    const std::set<std::string>& names,
                                    int lines_before, const TurnFacts& facts);

  // Returns the number of tokens the last reply decoded: the runtime's
  // counter of its last decode.
  absl::StatusOr<int> CompletionTokens() const;

  // Returns the number of tokens in the context: everything prefilled and
  // decoded so far.
  absl::StatusOr<int> TokenCount() const;

  // Returns the directory the runtime writes this chat's captures to.
  absl::StatusOr<std::filesystem::path> CaptureDirectory() const;

  // Owned; deleted with litert_lm_conversation_delete.
  LiteRtLmConversation* const handle_;

  // The model's engine, for the tokenizer; owned by the model.
  LiteRtLmEngine* const engine_;

  // Where the runtime's capture directory, a relative path, is rooted.
  const std::filesystem::path data_dir_;

  // Passed with every message: the runtime takes it per message as well as
  // per session.
  const int max_completion_tokens_;

  // The model's context length: a reply stops when the context is full.
  const int context_length_;

  // Whether the model runs on the GPU, whose KV cache captures need
  // unpacking.
  const bool gpu_;

  // The model's start token, if it has one.
  const std::optional<int32_t> start_token_;

  // The model's stop tokens of one id.
  const std::set<int32_t> stop_tokens_;

  // The ids of the rows the chat's turns fed, as far as they are known.
  TokenRows rows_;

  // What the token the last turn left held can be, if known.
  std::optional<std::set<int32_t>> held_;

  // Completed turns: the number of the next turn, less one.
  int turns_ = 0;

  // True while a Send is in progress: what Cancel checks first.
  std::atomic<bool> sending_{false};

  // Set by Cancel, read and cleared by Send.
  std::atomic<bool> cancelled_{false};

  // True once a Send was cancelled: the runtime takes no further task.
  bool over_ = false;
};

absl::StatusOr<Turn> LiteRtLmChat::Send(const protocol::ChatMessage& message,
                                        const ChunkCallback& on_chunk) {
  if (over_) {
    return absl::FailedPreconditionError(
        "The chat was cancelled; the next turn needs a new one.");
  }
  if (message.role() != protocol::ChatMessage::USER) {
    return absl::InvalidArgumentError("A turn is a user message.");
  }
  absl::StatusOr<std::string> text = MessageText(message);
  if (!text.ok()) return text.status();

  // What the context and the capture hold before the turn, to tell the
  // turn's part of them apart afterwards.
  TurnFacts facts;
  facts.start_token = start_token_;
  facts.stop_tokens = stop_tokens_;
  // Known again only if this turn ends as it should.
  facts.held = std::exchange(held_, std::nullopt);
  absl::StatusOr<int> context_before = TokenCount();
  if (!context_before.ok()) return context_before.status();
  facts.context_before = *context_before;
  absl::StatusOr<std::filesystem::path> capture_dir = CaptureDirectory();
  if (!capture_dir.ok()) return capture_dir.status();
  const Listing files_before = ListFiles(*capture_dir);
  const int lines_before = CountLines(*capture_dir / kTokensFile);
  // The prompt before sending: rendering what the model is about to receive
  // moves the chat's template state along, so it is done on a copy.
  absl::StatusOr<std::vector<int32_t>> prompt = RenderPrompt(*text);
  if (!prompt.ok()) return prompt.status();
  facts.prompt = *std::move(prompt);

  // A turn the context has no room for is refused before the runtime starts
  // it: LiteRT-LM's own refusal, mid-prefill, leaves its conversation unable
  // to take another turn. The turn needs its prompt, after the first turn's
  // start token; the token held since the turn before is in the count already.
  const int needed =
      static_cast<int>(facts.prompt.size()) +
      (facts.context_before == 0 && start_token_.has_value() ? 1 : 0);
  const int left = context_length_ - facts.context_before;
  if (needed > left) {
    held_ = facts.held;
    return NoRoom(absl::StrCat("This turn needs ", needed,
                               " tokens; the context has ", left, " left."),
                  needed, left);
  }

  LiteRtLmConversationOptionalArgs* args = nullptr;
  if (absl::Status status =
          Check(litert_lm_conversation_optional_args_create(&args),
                "Could not create the message options");
      !status.ok()) {
    return status;
  }
  if (absl::Status status =
          Check(litert_lm_conversation_optional_args_set_max_output_tokens(
                    args, max_completion_tokens_),
                "Could not set max_completion_tokens");
      !status.ok()) {
    litert_lm_conversation_optional_args_delete(args);
    return status;
  }

  Stream stream(on_chunk);
  cancelled_ = false;
  sending_ = true;
  if (absl::Status status =
          Check(litert_lm_conversation_send_message_stream(
                    handle_, text->c_str(), /*extra_context=*/nullptr, args,
                    OnChunk, &stream),
                "Could not start the reply");
      !status.ok()) {
    sending_ = false;
    litert_lm_conversation_optional_args_delete(args);
    return status;
  }
  // Waits for the turn's tasks, as the runtime's own SendMessage does
  // (runtime/conversation/conversation.cc); with single-threaded execution
  // they run only during this wait. A wait gives up after ten minutes
  // (Engine::kDefaultTimeout), and a turn may take longer.
  LiteRtLmStatusCode waited;
  do {
    waited = litert_lm_conversation_wait_until_done(handle_);
  } while (waited == kLiteRtLmStatusDeadlineExceeded);
  const absl::Status wait = Check(waited, "Could not finish the reply");
  {
    // Nothing above is freed before the final chunk: the runtime can still
    // call back until then.
    std::unique_lock<std::mutex> lock(stream.mutex);  // NOLINT(build/c++11)
    stream.done.wait(lock, [&stream] { return stream.finished; });
  }
  sending_ = false;
  litert_lm_conversation_optional_args_delete(args);
  if (cancelled_.exchange(false)) {
    over_ = true;
    return absl::CancelledError("The reply was cancelled.");
  }
  // A turn that fits by its tokens can still fail here, before any chunk:
  // LiteRT-LM prefills in the blocks of its model file's prefill signatures
  // (prefill_8, prefill_128, ...), padding the last, and refuses a block past
  // the context's end.
  if (!stream.status.ok()) return stream.status;
  if (!wait.ok()) return wait;

  // The runtime's decode counter, and the context's size.
  absl::StatusOr<int> completion_tokens = CompletionTokens();
  if (!completion_tokens.ok()) return completion_tokens.status();
  facts.decode_steps = *completion_tokens;
  absl::StatusOr<int> context_after = TokenCount();
  if (!context_after.ok()) return context_after.status();
  facts.context_after = *context_after;

  // The files the turn wrote, by when they were written: a turn writes again
  // a file of a name an earlier turn wrote, as the GPU's cache kernel left
  // it, so each is unpacked anew.
  const std::set<std::string> written =
      WrittenSince(*capture_dir, files_before);
  if (absl::Status status = CheckWholeFiles(*capture_dir, written);
      !status.ok()) {
    return status;
  }
  if (gpu_) {
    if (absl::Status status = UnpackMlDriftKvCaches(*capture_dir, written);
        !status.ok()) {
      return absl::InternalError(status.message());
    }
  }
  absl::StatusOr<FedTurn> fed =
      FedTokens(*capture_dir, written, lines_before, facts);
  if (!fed.ok()) return fed.status();
  held_ = fed->held;

  // What the turn cost: the context the first decode step saw, and one token
  // per decode step. A rewrite after them, the reply written again without
  // its reasoning, leaves the context smaller.
  protocol::Completion& completion = stream.completion;
  protocol::Usage& usage = *completion.mutable_usage();
  usage.set_prompt_tokens(fed->first_decode_position + 1);
  usage.set_completion_tokens(*completion_tokens);
  usage.set_total_tokens(usage.prompt_tokens() + usage.completion_tokens());
  // Cut off by either limit: the reply's own, or the context's.
  completion.set_finish_reason(*completion_tokens >= max_completion_tokens_ ||
                                       usage.total_tokens() >= context_length_
                                   ? protocol::FINISH_REASON_LENGTH
                                   : protocol::FINISH_REASON_STOP);
  const int turn_number = turns_ + 1;
  protocol::ModelDebugger& debugger = *completion.mutable_model_debugger();
  debugger.set_turn(turn_number);
  debugger.set_context_token_count(*context_after);
  for (protocol::FedToken& token : fed->tokens) {
    *debugger.add_fed_tokens() = std::move(token);
  }

  Turn turn;
  turn.completion = std::move(completion);
  turn.capture_dir = *std::move(capture_dir);
  turns_ = turn_number;
  return turn;
}

absl::Status LiteRtLmChat::CheckRoomToStart() const {
  absl::StatusOr<int> context_before = TokenCount();
  if (!context_before.ok()) return context_before.status();
  absl::StatusOr<std::string> empty =
      Dump(Json{{"role", "user"}, {"content", ""}});
  if (!empty.ok()) return empty.status();
  absl::StatusOr<std::vector<int32_t>> prompt = RenderPrompt(*empty);
  if (!prompt.ok()) return prompt.status();
  // Counted as Send counts its turn.
  const int needed = static_cast<int>(prompt->size()) +
                     (*context_before == 0 && start_token_.has_value() ? 1 : 0);
  const int left = context_length_ - *context_before;
  if (needed <= left) return absl::OkStatus();
  return NoRoom(
      absl::StrCat("The starting messages need ", needed,
                   " tokens in the chat template; the context has ", left, "."),
      needed, left);
}

void LiteRtLmChat::Cancel() {
  if (!sending_) return;
  cancelled_ = true;
  // Refused only for a null conversation; a refused cancel leaves the reply
  // to finish.
  if (litert_lm_conversation_cancel_process(handle_) != kLiteRtLmStatusOk) {
    cancelled_ = false;
  }
}

absl::StatusOr<std::vector<int32_t>> LiteRtLmChat::RenderPrompt(
    const std::string& message) const {
  LiteRtLmConversation* copy = nullptr;
  if (absl::Status status =
          Check(litert_lm_conversation_clone(handle_, &copy),
                "Could not copy the conversation to render the prompt");
      !status.ok()) {
    return status;
  }
  // The rendered text is the copy's, until it is deleted.
  const char* rendered = nullptr;
  const absl::Status status =
      Check(litert_lm_conversation_render_message_to_string(
                copy, message.c_str(), &rendered),
            "Could not render the prompt");
  std::optional<std::string> prompt;
  if (status.ok() && rendered != nullptr) prompt = rendered;
  litert_lm_conversation_delete(copy);
  if (!status.ok()) return status;
  if (!prompt) return absl::InternalError("Runtime rendered no prompt.");

  return Tokenize(engine_, *prompt);
}

absl::StatusOr<FedTurn> LiteRtLmChat::FedTokens(
    const std::filesystem::path& capture_dir,
    const std::set<std::string>& names, int lines_before,
    const TurnFacts& facts) {
  absl::StatusOr<FedTurn> fed =
      ReadFedTokens(capture_dir, names, lines_before, facts, rows_);
  if (!fed.ok()) return fed.status();
  // A turn feeds few distinct ids, many times over.
  std::map<int32_t, std::string> texts;
  for (protocol::FedToken& token : fed->tokens) {
    if (!token.has_id()) continue;
    auto it = texts.find(token.id());
    if (it == texts.end()) {
      absl::StatusOr<std::string> text = Detokenize(engine_, token.id());
      if (!text.ok()) return text.status();
      it = texts.emplace(token.id(), *std::move(text)).first;
    }
    token.set_text(it->second);
  }
  return fed;
}

absl::StatusOr<int> LiteRtLmChat::CompletionTokens() const {
  LiteRtLmBenchmarkInfo* info = nullptr;
  if (absl::Status status =
          Check(litert_lm_conversation_get_benchmark_info(handle_, &info),
                "Could not read the turn's counters");
      !status.ok()) {
    return status;
  }
  int decodes = 0;
  int count = 0;
  absl::Status status =
      Check(litert_lm_benchmark_info_get_num_decode_turns(info, &decodes),
            "Could not count the decodes");
  if (status.ok() && decodes > 0) {
    status = Check(litert_lm_benchmark_info_get_decode_token_count_at(
                       info, decodes - 1, &count),
                   "Could not count the decoded tokens");
  }
  litert_lm_benchmark_info_delete(info);
  if (!status.ok()) return status;
  return count;
}

absl::StatusOr<int> LiteRtLmChat::TokenCount() const {
  int count = 0;
  if (absl::Status status =
          Check(litert_lm_conversation_get_token_count(handle_, &count),
                "Could not count the conversation's tokens");
      !status.ok()) {
    return status;
  }
  return count;
}

absl::StatusOr<std::filesystem::path> LiteRtLmChat::CaptureDirectory() const {
  // Null, without the debugger.
  LiteRtLmSessionDebugInfo* info = nullptr;
  if (absl::Status status =
          Check(litert_lm_experimental_conversation_get_session_debug_info(
                    handle_, &info),
                "Could not read the session's debug info");
      !status.ok()) {
    return status;
  }
  if (info == nullptr) {
    return absl::InternalError("Runtime gave no capture directory.");
  }
  const char* relative = nullptr;
  const absl::Status status =
      Check(litert_lm_experimental_session_debug_info_get_capture_dir(
                info, &relative),
            "Could not read the capture directory");
  std::optional<std::filesystem::path> result;
  if (status.ok() && relative != nullptr) result = data_dir_ / relative;
  litert_lm_experimental_session_debug_info_delete(info);
  if (!status.ok()) return status;
  if (!result) return absl::InternalError("Runtime gave no capture directory.");
  return *result;
}

// ---- Model

class LiteRtLmModel final : public Model {
 public:
  LiteRtLmModel(LiteRtLmEngine* handle, std::filesystem::path data_dir,
                int context_length, bool gpu, Sampling sampling,
                protocol::LoadResponse info)
      : handle_(handle),
        data_dir_(std::move(data_dir)),
        context_length_(context_length),
        gpu_(gpu),
        sampling_(sampling),
        info_(std::move(info)) {}
  ~LiteRtLmModel() override { litert_lm_engine_delete(handle_); }

  const protocol::LoadResponse& Info() const override { return info_; }
  absl::StatusOr<std::unique_ptr<Chat>> CreateChat(
      const protocol::CreateChatRequest& request) override;

 private:
  // Owned; deleted with litert_lm_engine_delete.
  LiteRtLmEngine* const handle_;

  const std::filesystem::path data_dir_;

  // The context length it was loaded with, the engine's max_num_tokens.
  const int context_length_;

  // Whether it was loaded on the GPU.
  const bool gpu_;

  // What a chat's sampling options leave unset is taken from here.
  const Sampling sampling_;

  const protocol::LoadResponse info_;
};

// Gives `session` the sampler `options` ask for. Sampling options left unset
// take their values from `sampling`: the runtime's sampler takes all four at
// once.
absl::Status SetSamplerParams(const protocol::ChatOptions& options,
                              const Sampling& sampling,
                              LiteRtLmSessionConfig* session) {
  LiteRtLmSamplerParams* sampler = nullptr;
  if (absl::Status status = Check(
          litert_lm_sampler_params_create(kLiteRtLmSamplerTypeTopP, &sampler),
          "Could not create the sampler params");
      !status.ok()) {
    return status;
  }
  absl::Status status;
  status.Update(Check(
      litert_lm_sampler_params_set_top_k(
          sampler, options.has_top_k() ? options.top_k() : sampling.top_k),
      "Could not set top_k"));
  status.Update(Check(
      litert_lm_sampler_params_set_top_p(
          sampler, options.has_top_p() ? options.top_p() : sampling.top_p),
      "Could not set top_p"));
  status.Update(
      Check(litert_lm_sampler_params_set_temperature(
                sampler, options.has_temperature() ? options.temperature()
                                                   : sampling.temperature),
            "Could not set temperature"));
  status.Update(
      Check(litert_lm_sampler_params_set_seed(
                sampler, options.has_seed() ? options.seed() : sampling.seed),
            "Could not set seed"));
  if (status.ok()) {
    status =
        Check(litert_lm_session_config_set_sampler_params(session, sampler),
              "Could not set the sampler params");
  }
  litert_lm_sampler_params_delete(sampler);
  return status;
}

// Returns the session config `options` ask for, owned by the caller.
absl::StatusOr<LiteRtLmSessionConfig*> SessionConfig(
    const protocol::ChatOptions& options, const ChatRuntimeOptions& runtime,
    const Sampling& sampling) {
  LiteRtLmSessionConfig* session = nullptr;
  if (absl::Status status = Check(litert_lm_session_config_create(&session),
                                  "Could not create the session config");
      !status.ok()) {
    return status;
  }
  absl::Status status;
  status.Update(Check(litert_lm_session_config_set_max_output_tokens(
                          session, options.max_completion_tokens()),
                      "Could not set max_completion_tokens"));
  if (runtime.apply_prompt_template) {
    status.Update(Check(litert_lm_session_config_set_apply_prompt_template(
                            session, *runtime.apply_prompt_template),
                        "Could not set apply_prompt_template"));
  }
  status.Update(Check(litert_lm_session_config_set_enable_speculative_decoding(
                          session, runtime.speculative_decoding),
                      "Could not set speculative_decoding"));
  if (status.ok() && (options.has_top_k() || options.has_top_p() ||
                      options.has_temperature() || options.has_seed())) {
    status = SetSamplerParams(options, sampling, session);
  }
  if (!status.ok()) {
    litert_lm_session_config_delete(session);
    return status;
  }
  return session;
}

// Gives `config` the thinking settings of `runtime`.
absl::Status SetThinkingConfig(const ChatRuntimeOptions& runtime,
                               LiteRtLmConversationConfig* config) {
  LiteRtLmThinkingConfig* thinking = nullptr;
  if (absl::Status status = Check(litert_lm_thinking_config_create(&thinking),
                                  "Could not create the thinking config");
      !status.ok()) {
    return status;
  }
  absl::Status status;
  if (runtime.thinking) {
    status.Update(Check(litert_lm_thinking_config_set_enable_thinking(
                            thinking, *runtime.thinking),
                        "Could not set thinking"));
  }
  if (runtime.thinking_token_budget) {
    status.Update(Check(litert_lm_thinking_config_set_thinking_token_budget(
                            thinking, *runtime.thinking_token_budget),
                        "Could not set thinking_token_budget"));
  }
  if (status.ok()) {
    status = Check(
        litert_lm_conversation_config_set_thinking_config(config, thinking),
        "Could not set the thinking config");
  }
  litert_lm_thinking_config_delete(thinking);
  return status;
}

absl::StatusOr<std::unique_ptr<Chat>> LiteRtLmModel::CreateChat(
    const protocol::CreateChatRequest& request) {
  const protocol::ChatOptions& options = request.options();
  absl::StatusOr<ChatRuntimeOptions> runtime =
      ParseChatOptions(options.runtime_options());
  if (!runtime.ok()) return runtime.status();

  // The texts first: an error in them costs no runtime object. A system
  // message is the first message or none: the runtime takes it apart.
  std::optional<std::string> system_message;
  Json history = Json::array();
  for (int i = 0; i < request.messages_size(); ++i) {
    const protocol::ChatMessage& message = request.messages(i);
    if (message.role() == protocol::ChatMessage::SYSTEM) {
      if (i != 0) {
        return absl::InvalidArgumentError(
            "A system message may only be the first message.");
      }
      absl::StatusOr<std::string> text = Dump(Json(message.content()));
      if (!text.ok()) return text.status();
      system_message = *std::move(text);
    } else {
      absl::StatusOr<Json> json = MessageJson(message);
      if (!json.ok()) return json.status();
      history.push_back(*std::move(json));
    }
  }
  std::optional<std::string> messages;
  if (!history.empty()) {
    absl::StatusOr<std::string> text = Dump(history);
    if (!text.ok()) return text.status();
    messages = *std::move(text);
  }

  absl::StatusOr<LiteRtLmSessionConfig*> session =
      SessionConfig(options, *runtime, sampling_);
  if (!session.ok()) return session.status();
  LiteRtLmConversationConfig* config = nullptr;
  if (absl::Status status = Check(litert_lm_conversation_config_create(&config),
                                  "Could not create the conversation config");
      !status.ok()) {
    litert_lm_session_config_delete(*session);
    return status;
  }
  absl::Status status =
      Check(litert_lm_conversation_config_set_session_config(config, *session),
            "Could not set the session config");
  if (status.ok() && (runtime->thinking || runtime->thinking_token_budget)) {
    status = SetThinkingConfig(*runtime, config);
  }
  if (status.ok() && system_message) {
    status = Check(litert_lm_conversation_config_set_system_message(
                       config, system_message->c_str()),
                   "Could not set the system message");
  }
  if (status.ok() && messages) {
    status = Check(
        litert_lm_conversation_config_set_messages(config, messages->c_str()),
        "Could not set the messages");
  }
  LiteRtLmConversation* handle = nullptr;
  if (status.ok()) {
    status = Check(litert_lm_conversation_create(handle_, config, &handle),
                   "Could not create the conversation");
  }
  litert_lm_conversation_config_delete(config);
  litert_lm_session_config_delete(*session);
  if (!status.ok()) return status;
  std::set<int32_t> stop_tokens;
  for (const protocol::StopToken& token : info_.stop_tokens()) {
    if (token.has_ids() && token.ids().ids_size() == 1) {
      stop_tokens.insert(token.ids().ids(0));
    }
  }
  auto chat = std::make_unique<LiteRtLmChat>(
      handle, handle_, data_dir_, options.max_completion_tokens(),
      context_length_, gpu_,
      info_.has_start_token() ? std::optional<int32_t>(info_.start_token())
                              : std::nullopt,
      std::move(stop_tokens));
  // A chat whose first turn could not fit is refused now, not at that turn:
  // a chat without starting messages has room for them.
  if (request.messages_size() > 0) {
    if (absl::Status room = chat->CheckRoomToStart(); !room.ok()) return room;
  }
  return std::unique_ptr<Chat>(std::move(chat));
}

// The directory the library was loaded from, links resolved: where whoever
// built it put the GPU plugins beside it.
absl::StatusOr<std::filesystem::path> LibraryDirectory() {
  Dl_info info;
  if (dladdr(reinterpret_cast<const void*>(&litert_lm_engine_create), &info) ==
          0 ||
      info.dli_fname == nullptr) {
    return absl::InternalError(
        "The loader does not say where the LiteRT-LM library is.");
  }
  std::error_code error;
  const std::filesystem::path library =
      std::filesystem::canonical(info.dli_fname, error);
  if (error) {
    return absl::InternalError(
        absl::StrCat("Cannot resolve ", info.dli_fname, ": ", error.message()));
  }
  return library.parent_path();
}

}  // namespace

// ---- Runtime

LiteRtLmRuntime::LiteRtLmRuntime() : info_(MakeInfo()) {}

absl::StatusOr<std::unique_ptr<Model>> LiteRtLmRuntime::Load(
    const std::filesystem::path& model, const std::filesystem::path& data_dir,
    const protocol::LoadRequest& request) {
  absl::StatusOr<const char*> backend = BackendName(request.device());
  if (!backend.ok()) return backend.status();
  absl::StatusOr<EngineOptions> options =
      ParseEngineOptions(request.runtime_options());
  if (!options.ok()) return options.status();
  // The error_code overloads: the ones without throw, and nothing catches.
  std::error_code error;
  if (!std::filesystem::is_regular_file(model, error)) {
    return absl::NotFoundError(
        absl::StrCat("No model at ", model.string(), "."));
  }
  if (!std::filesystem::is_directory(data_dir, error)) {
    return absl::FailedPreconditionError(
        absl::StrCat("No data directory at ", data_dir.string(), "."));
  }
  if (request.device() == protocol::GPU ||
      options->vision_backend == protocol::GPU ||
      options->audio_backend == protocol::GPU) {
    if (absl::Status gpu = CanRunOnGpu(); !gpu.ok()) return gpu;
    absl::StatusOr<std::filesystem::path> library = LibraryDirectory();
    if (!library.ok()) return library.status();
    if (absl::Status loaded = LoadLiteRtGpuPlugins(library->string());
        !loaded.ok()) {
      return loaded;
    }
  }

  // Process-wide: the runtime has no level per engine.
  if (absl::Status status =
          Check(litert_lm_set_min_log_level(kLiteRtLmLogSeverityWarning),
                "Could not set the log level");
      !status.ok()) {
    return status;
  }
  const std::string data_dir_text = data_dir.string();
  LiteRtLmEngineSettings* settings = nullptr;
  if (absl::Status status = Check(
          litert_lm_engine_settings_create(
              model.c_str(), *backend,
              options->vision_backend ? *BackendName(*options->vision_backend)
                                      : nullptr,
              options->audio_backend ? *BackendName(*options->audio_backend)
                                     : nullptr,
              &settings),
          "Could not create the engine settings");
      !status.ok()) {
    return status;
  }
  absl::Status status = Apply(request, data_dir_text, *options, settings);
  LiteRtLmEngine* handle = nullptr;
  if (status.ok()) {
    status = Check(litert_lm_engine_create(settings, &handle),
                   "Could not load the model");
  }
  litert_lm_engine_settings_delete(settings);
  if (!status.ok()) return status;
  absl::StatusOr<protocol::LoadResponse> info = ReadTokens(handle);
  if (!info.ok()) {
    litert_lm_engine_delete(handle);
    return info.status();
  }
  // The engine keeps the metadata it read to itself, so the file's is read
  // again, its metadata only.
  absl::StatusOr<std::optional<protocol::SamplerDefaults>> defaults =
      ReadSamplerDefaults(model);
  if (!defaults.ok()) {
    litert_lm_engine_delete(handle);
    return defaults.status();
  }
  if (*defaults) *info->mutable_sampler_defaults() = **std::move(defaults);
  const Sampling sampling = SamplingOf(*info);
  return std::unique_ptr<Model>(new LiteRtLmModel(
      handle, data_dir, request.context_length(),
      request.device() == protocol::GPU, sampling, *std::move(info)));
}

absl::Status LiteRtLmRuntime::CanRunOnGpu() const {
#if defined(__APPLE__) && TARGET_OS_SIMULATOR
  // The plugins would load there, but the model does not: the simulator's
  // Metal has no argument buffers, and binds textures only up to index 30,
  // where LiteRT-LM's Metal kernels use 31.
  return absl::FailedPreconditionError(
      "The iOS simulator's Metal lacks what LiteRT-LM's GPU backend needs "
      "(argument buffers, and textures bound past index 30); load the model "
      "on the CPU.");
#else
  absl::StatusOr<std::filesystem::path> library = LibraryDirectory();
  if (!library.ok()) return library.status();
  if (!HasLiteRtGpuPlugins(library->string())) {
    return absl::FailedPreconditionError(
        absl::StrCat("The GPU libraries are not beside the LiteRT-LM library, "
                     "in ",
                     library->string(), "."));
  }
  return absl::OkStatus();
#endif
}

}  // namespace model_debugger::runner::llm
