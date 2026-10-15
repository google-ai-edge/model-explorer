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
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "google/protobuf/struct.pb.h"
#include "litert_lm/version.h"
#include "model_debugger/runner/llm/litert_lm/gpu_plugins.h"
#include "model_debugger/runner/llm/litert_lm/ml_drift_kv_cache.h"
#include "model_debugger/runner/llm/runtime.h"
#include "model_debugger/runner/util/json.h"
#include "model_debugger/v1/runner.pb.h"
#include "third_party/LiteRT-LM/c/conversation.h"
#include "third_party/LiteRT-LM/c/engine.h"
#include "third_party/LiteRT-LM/c/experimental.h"
#include "third_party/LiteRT-LM/c/model_info.h"

namespace model_debugger::runner::llm {
namespace {

// The C API takes token ids as int.
static_assert(sizeof(int) == sizeof(int32_t));

// ---- What this runtime declares

// The engine settings Load takes beyond device and contextLength, as a JSON
// Schema the server validates LoadRequest.runtime_options against.
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
// as a JSON Schema the server validates ChatOptions.runtime_options against.
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
v1::RuntimeInfo MakeInfo() {
  v1::RuntimeInfo info;
  info.set_name(v1::RuntimeInfo::LITERT_LM);
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
absl::StatusOr<const char*> BackendName(v1::Device device) {
  switch (device) {
    case v1::CPU:
      return "cpu";
    case v1::GPU:
      return "gpu";
    default:
      return absl::InvalidArgumentError("The device is not CPU or GPU.");
  }
}

// Returns the device named by option `name`, as the protocol spells it.
absl::StatusOr<v1::Device> DeviceOf(std::string_view name, const Value& value) {
  absl::StatusOr<std::string> text = AsString(name, value);
  if (!text.ok()) return text.status();
  if (*text == "CPU") return v1::CPU;
  if (*text == "GPU") return v1::GPU;
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
  std::optional<v1::Device> audio_backend;
  std::optional<v1::Device> vision_backend;
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

// Calls `setter` with `value` if there is one. The setting and the setter
// share one type T, so that an int setting cannot reach a setter of bool.
template <typename T>
void Set(LiteRtLmEngineSettings* to, const std::optional<T>& value,
         void (*setter)(LiteRtLmEngineSettings*, T)) {
  if (value) {
    setter(to, *value);
  }
}

// Passes the request and the parsed options to `to`, in the order of the
// setters in third_party/LiteRT-LM/c/engine.h.
void Apply(const v1::LoadRequest& request, const std::string& data_dir,
           const EngineOptions& options, LiteRtLmEngineSettings* to) {
  litert_lm_engine_settings_set_max_num_tokens(to, request.context_length());
  Set(to, options.cpu_threads, litert_lm_engine_settings_set_num_threads);
  Set(to, options.audio_cpu_threads,
      litert_lm_engine_settings_set_audio_num_threads);
  Set(to, options.parallel_file_section_loading,
      litert_lm_engine_settings_set_parallel_file_section_loading);
  Set(to, options.single_threaded_execution,
      litert_lm_engine_settings_set_single_threaded_execution);
  // The runtime's cache directory is the Runner's data directory: its
  // compiled caches and its captures both land there.
  litert_lm_engine_settings_set_cache_dir(to, data_dir.c_str());
  if (options.activation_data_type) {
    litert_lm_engine_settings_set_activation_data_type(
        to, *options.activation_data_type);
  }
  Set(to, options.prefill_chunk_size,
      litert_lm_engine_settings_set_prefill_chunk_size);
  Set(to, options.enable_ynnpack, litert_lm_engine_settings_set_enable_ynnpack);
  // Always on: what a turn cost (Usage) is read from its counters.
  litert_lm_engine_settings_enable_benchmark(to);
  Set(to, options.lora_rank, litert_lm_engine_settings_set_lora_rank);
  Set(to, options.audio_lora_rank,
      litert_lm_engine_settings_set_audio_lora_rank);
  Set(to, options.gpu_metal_residency_set,
      litert_lm_engine_settings_set_gpu_enable_metal_residency_set);
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

absl::StatusOr<const char*> RoleName(v1::ChatMessage::Role role) {
  switch (role) {
    case v1::ChatMessage::SYSTEM:
      return "system";
    case v1::ChatMessage::USER:
      return "user";
    case v1::ChatMessage::ASSISTANT:
      return "assistant";
    default:
      return absl::InvalidArgumentError(
          "A chat message's role is not system, user or assistant.");
  }
}

// Returns `message` as the conversation API takes it, as JSON text.
absl::StatusOr<std::string> MessageText(const v1::ChatMessage& message) {
  absl::StatusOr<const char*> role = RoleName(message.role());
  if (!role.ok()) return role.status();
  return Dump(Json{{"role", *role}, {"content", message.content()}});
}

// ---- A reply as it streams

// What the runtime's thread writes and Send waits for.
struct Stream {
  explicit Stream(const ChunkCallback& callback) : on_chunk(callback) {}

  const ChunkCallback& on_chunk;
  absl::Mutex mutex;
  v1::Completion completion ABSL_GUARDED_BY(mutex);
  absl::Status status ABSL_GUARDED_BY(mutex);
  bool finished ABSL_GUARDED_BY(mutex) = false;
};

// Gives `text` to the stream's callback as a chunk of the answer or of the
// reasoning, and keeps it for the completion.
void Deliver(Stream& stream, bool reasoning, std::string_view text)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(stream.mutex) {
  if (text.empty()) return;
  v1::Chunk chunk;
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
absl::Status Take(std::string_view json, Stream& stream)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(stream.mutex) {
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
  absl::MutexLock lock(stream.mutex);
  if (stream.finished) return;
  if (const char* error = litert_lm_stream_chunk_get_error(chunk)) {
    stream.status = absl::InternalError(error);
  }
  if (stream.status.ok()) {
    if (const char* text = litert_lm_stream_chunk_get_text(chunk)) {
      stream.status = Take(text, stream);
    }
  }
  if (litert_lm_stream_chunk_is_final(chunk)) {
    stream.finished = true;
  }
}

// ---- Tokens, through the engine

absl::StatusOr<std::vector<int32_t>> Tokenize(LiteRtLmEngine* engine,
                                              const std::string& text) {
  LiteRtLmTokenizeResult* result =
      litert_lm_engine_tokenize(engine, text.c_str());
  if (result == nullptr) {
    return absl::InternalError("Runtime could not tokenize the prompt.");
  }
  const int* tokens = litert_lm_tokenize_result_get_tokens(result);
  const size_t count = litert_lm_tokenize_result_get_num_tokens(result);
  std::vector<int32_t> ids;
  if (tokens != nullptr) {
    ids.assign(tokens, tokens + count);
  }
  litert_lm_tokenize_result_delete(result);
  if (tokens == nullptr && count > 0) {
    return absl::InternalError("Runtime counted tokens it did not give.");
  }
  return ids;
}

absl::StatusOr<std::string> Detokenize(LiteRtLmEngine* engine, int32_t id) {
  LiteRtLmDetokenizeResult* result =
      litert_lm_engine_detokenize(engine, &id, 1);
  if (result == nullptr) {
    return absl::InternalError(
        absl::StrCat("Runtime has no text for token ", id, "."));
  }
  const char* text = litert_lm_detokenize_result_get_string(result);
  std::string copy = text != nullptr ? text : "";
  litert_lm_detokenize_result_delete(result);
  return copy;
}

// Returns the ids of `token`, or nullopt if it is configured as text.
std::optional<std::vector<int32_t>> IdsOf(const LiteRtLmTokenUnion* token) {
  const int* ids = nullptr;
  size_t count = 0;
  if (litert_lm_token_union_get_ids(token, &ids, &count) != 0 ||
      ids == nullptr) {
    return std::nullopt;
  }
  return std::vector<int32_t>(ids, ids + count);
}

// Reads the model's start and stop tokens into a LoadResponse.
absl::StatusOr<v1::LoadResponse> ReadTokens(LiteRtLmEngine* engine) {
  v1::LoadResponse response;
  if (LiteRtLmTokenUnion* start = litert_lm_engine_get_start_token(engine)) {
    const std::optional<std::vector<int32_t>> ids = IdsOf(start);
    litert_lm_token_union_delete(start);
    if (!ids || ids->size() != 1) {
      return absl::InternalError("Runtime start token is not a single id.");
    }
    response.set_start_token(ids->front());
  }
  LiteRtLmTokenUnions* unions = litert_lm_engine_get_stop_tokens(engine);
  if (unions == nullptr) return response;
  const size_t count = litert_lm_token_unions_get_num_tokens(unions);
  absl::Status status;
  for (size_t i = 0; i < count && status.ok(); ++i) {
    LiteRtLmTokenUnion* token = litert_lm_token_unions_get_token_at(unions, i);
    if (token == nullptr) {
      status =
          absl::InternalError("Runtime named a stop token it did not give.");
      break;
    }
    v1::StopToken* stop = response.add_stop_tokens();
    if (litert_lm_token_union_get_type(token) == kLiteRtLmTokenUnionTypeIds) {
      if (std::optional<std::vector<int32_t>> ids = IdsOf(token)) {
        stop->mutable_ids()->mutable_ids()->Assign(ids->begin(), ids->end());
      } else {
        status = absl::InternalError(
            "Runtime named a stop token by ids it did not give.");
      }
    } else if (const char* text = litert_lm_token_union_get_string(token)) {
      stop->set_text(text);
    } else {
      status = absl::InternalError(
          "Runtime named a stop token by text it did not give.");
    }
    litert_lm_token_union_delete(token);
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

// Returns the sampling the model at `model` declares, as the runtime uses it
// for a chat that sets none, or LiteRT-LM's own if it declares none.
absl::StatusOr<Sampling> ReadSampling(const std::filesystem::path& model) {
  LiteRtLmLoadedFile* file = litert_lm_loaded_file_create(model.c_str());
  if (file == nullptr) {
    return absl::InternalError(
        absl::StrCat("Could not read the metadata of ", model.string(), "."));
  }
  // The runtime maps every sampling onto its top-p sampler: a value the model
  // declares replaces the default, one it leaves at 0 does not. Greedy is a
  // top_k of 1; top-k alone is a top_p of 1, which cuts nothing.
  Sampling sampling;
  const LiteRtLmSamplerType type = litert_lm_loaded_file_sampler_type(file);
  if (const int top_k = litert_lm_loaded_file_sampler_top_k(file); top_k > 0) {
    sampling.top_k = top_k;
  }
  if (const float top_p = litert_lm_loaded_file_sampler_top_p(file);
      top_p > 0) {
    sampling.top_p = top_p;
  } else if (type == kLiteRtLmSamplerTypeTopK) {
    sampling.top_p = 1.0f;
  }
  if (const float temperature = litert_lm_loaded_file_sampler_temperature(file);
      temperature > 0) {
    sampling.temperature = temperature;
  }
  if (type == kLiteRtLmSamplerTypeGreedy) sampling.top_k = 1;
  litert_lm_loaded_file_delete(file);
  return sampling;
}

// ---- Chat

class LiteRtLmChat final : public Chat {
 public:
  LiteRtLmChat(LiteRtLmConversation* handle, LiteRtLmEngine* engine,
               std::filesystem::path data_dir, int max_completion_tokens,
               int context_length, bool gpu)
      : handle_(handle),
        engine_(engine),
        data_dir_(std::move(data_dir)),
        max_completion_tokens_(max_completion_tokens),
        context_length_(context_length),
        gpu_(gpu) {}
  ~LiteRtLmChat() override { litert_lm_conversation_delete(handle_); }

  absl::StatusOr<Turn> Send(const v1::ChatMessage& message,
                            const ChunkCallback& on_chunk) override;
  void Cancel() override;

 private:
  // Adds to `into`'s prompt tokens the prompt the model receives for
  // `message`: what the chat has not yet fed it, in the chat template, as the
  // runtime tokenizes it.
  absl::Status RenderPrompt(const std::string& message,
                            v1::ModelDebugger& into) const;

  // Returns the number of tokens in the context: everything prefilled and
  // decoded so far.
  absl::StatusOr<int> TokenCount() const;

  // Returns the directory the runtime writes this chat's captures to.
  absl::StatusOr<std::filesystem::path> CaptureDirectory() const;

  // Unpacks the K and V cache files in `capture_dir` not unpacked before, if
  // ML Drift's cache kernel wrote them: on the GPU, LiteRT-LM gives it the
  // cache of a model with a param tensor (llm_executor_settings_utils.cc,
  // single_kv_cache_buffer), which the capture shows as param_tensor files.
  // See UnpackMlDriftKvCache.
  absl::Status UnpackKvCaches(const std::filesystem::path& capture_dir);

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

  // The KV cache files UnpackKvCaches has rewritten: each is unpacked once.
  std::set<std::string> unpacked_;

  // Completed turns: the number of the next turn, less one.
  int turns_ = 0;

  // True while a Send is in progress: what Cancel checks first.
  std::atomic<bool> sending_{false};

  // Set by Cancel, read and cleared by Send.
  std::atomic<bool> cancelled_{false};

  // True once a Send was cancelled: the runtime takes no further task.
  bool over_ = false;
};

absl::StatusOr<Turn> LiteRtLmChat::Send(const v1::ChatMessage& message,
                                        const ChunkCallback& on_chunk) {
  if (over_) {
    return absl::FailedPreconditionError(
        "The chat was cancelled; the next turn needs a new one.");
  }
  if (message.role() != v1::ChatMessage::USER) {
    return absl::InvalidArgumentError("A turn is a user message.");
  }
  absl::StatusOr<std::string> text = MessageText(message);
  if (!text.ok()) return text.status();

  // The prompt before sending: rendering what the model is about to receive
  // moves the chat's template state along, so it is done on a copy.
  v1::ModelDebugger debugger;
  if (absl::Status status = RenderPrompt(*text, debugger); !status.ok()) {
    return status;
  }

  LiteRtLmConversationOptionalArgs* args =
      litert_lm_conversation_optional_args_create();
  if (args == nullptr) {
    return absl::InternalError("Could not create the message options.");
  }
  litert_lm_conversation_optional_args_set_max_output_tokens(
      args, max_completion_tokens_);

  Stream stream(on_chunk);
  cancelled_ = false;
  sending_ = true;
  const int started = litert_lm_conversation_send_message_stream(
      handle_, text->c_str(), /*extra_context=*/nullptr, args, OnChunk,
      &stream);
  if (started != 0) {
    sending_ = false;
    litert_lm_conversation_optional_args_delete(args);
    return absl::InternalError(
        absl::StrCat("Could not start the reply (", started, ")."));
  }
  absl::Status stream_status;
  v1::Completion completion;
  {
    // Nothing above is freed before the final chunk: the runtime can still
    // call back until then.
    absl::MutexLock lock(stream.mutex);
    stream.mutex.Await(absl::Condition(&stream.finished));
    stream_status = stream.status;
    completion = std::move(stream.completion);
  }
  sending_ = false;
  litert_lm_conversation_optional_args_delete(args);
  if (cancelled_.exchange(false)) {
    over_ = true;
    return absl::CancelledError("The reply was cancelled.");
  }
  if (!stream_status.ok()) return stream_status;

  // What the turn cost: the runtime's decode counter, and the context's size.
  LiteRtLmBenchmarkInfo* info =
      litert_lm_conversation_get_benchmark_info(handle_);
  if (info == nullptr) {
    return absl::InternalError("Runtime gave no counters for the turn.");
  }
  const int decodes = litert_lm_benchmark_info_get_num_decode_turns(info);
  const int completion_tokens =
      decodes > 0 ? litert_lm_benchmark_info_get_decode_token_count_at(
                        info, decodes - 1)
                  : 0;
  litert_lm_benchmark_info_delete(info);
  absl::StatusOr<int> context = TokenCount();
  if (!context.ok()) return context.status();

  // Cut off by either limit: the reply's own, or the context's.
  completion.set_finish_reason(completion_tokens >= max_completion_tokens_ ||
                                       *context >= context_length_
                                   ? v1::FINISH_REASON_LENGTH
                                   : v1::FINISH_REASON_STOP);
  v1::Usage& usage = *completion.mutable_usage();
  usage.set_prompt_tokens(*context - completion_tokens);
  usage.set_completion_tokens(completion_tokens);
  usage.set_total_tokens(*context);
  const int turn_number = turns_ + 1;
  debugger.set_turn(turn_number);
  debugger.set_context_token_count(*context);
  *completion.mutable_model_debugger() = std::move(debugger);

  absl::StatusOr<std::filesystem::path> capture_dir = CaptureDirectory();
  if (!capture_dir.ok()) return capture_dir.status();
  if (gpu_) {
    if (absl::Status status = UnpackKvCaches(*capture_dir); !status.ok()) {
      return status;
    }
  }
  Turn turn;
  turn.completion = std::move(completion);
  turn.capture_dir = *std::move(capture_dir);
  turns_ = turn_number;
  return turn;
}

void LiteRtLmChat::Cancel() {
  if (!sending_) return;
  cancelled_ = true;
  litert_lm_conversation_cancel_process(handle_);
}

absl::Status LiteRtLmChat::RenderPrompt(const std::string& message,
                                        v1::ModelDebugger& into) const {
  LiteRtLmConversation* copy = litert_lm_conversation_clone(handle_);
  if (copy == nullptr) {
    return absl::InternalError("Could not copy the conversation to render.");
  }
  const char* rendered =
      litert_lm_conversation_render_message_to_string(copy, message.c_str());
  std::optional<std::string> prompt;
  if (rendered != nullptr) prompt = rendered;
  litert_lm_conversation_delete(copy);
  if (!prompt) return absl::InternalError("Could not render the prompt.");

  absl::StatusOr<std::vector<int32_t>> ids = Tokenize(engine_, *prompt);
  if (!ids.ok()) return ids.status();
  for (const int32_t id : *ids) {
    absl::StatusOr<std::string> text = Detokenize(engine_, id);
    if (!text.ok()) return text.status();
    v1::PromptToken* token = into.add_prompt_tokens();
    token->set_id(id);
    token->set_text(*std::move(text));
  }
  return absl::OkStatus();
}

absl::StatusOr<int> LiteRtLmChat::TokenCount() const {
  const int count = litert_lm_conversation_get_token_count(handle_);
  if (count < 0) {
    return absl::InternalError("Could not count the conversation's tokens.");
  }
  return count;
}

absl::StatusOr<std::filesystem::path> LiteRtLmChat::CaptureDirectory() const {
  LiteRtLmSessionDebugInfo* info =
      litert_lm_experimental_conversation_get_session_debug_info(handle_);
  if (info == nullptr) {
    return absl::InternalError("Runtime gave no capture directory.");
  }
  const char* relative =
      litert_lm_experimental_session_debug_info_get_capture_dir(info);
  std::optional<std::filesystem::path> result;
  if (relative != nullptr) result = data_dir_ / relative;
  litert_lm_experimental_session_debug_info_delete(info);
  if (!result) return absl::InternalError("Runtime gave no capture directory.");
  return *result;
}

absl::Status LiteRtLmChat::UnpackKvCaches(
    const std::filesystem::path& capture_dir) {
  // Listed first: unpacking writes into the directory.
  std::vector<std::string> names;
  bool param_tensor = false;
  std::error_code error;
  for (const auto& entry :
       std::filesystem::directory_iterator(capture_dir, error)) {
    const std::string name = entry.path().filename().string();
    if (!entry.is_regular_file(error)) continue;
    param_tensor = param_tensor || absl::StrContains(name, "_param_tensor_");
    if ((absl::StrContains(name, "_kv_cache_k_") ||
         absl::StrContains(name, "_kv_cache_v_")) &&
        absl::EndsWith(name, ".safetensors") && unpacked_.count(name) == 0) {
      names.push_back(name);
    }
  }
  if (error) {
    return absl::InternalError(absl::StrCat(
        "Cannot list ", capture_dir.string(), ": ", error.message()));
  }
  if (!param_tensor) return absl::OkStatus();
  for (const std::string& name : names) {
    if (absl::Status status = UnpackMlDriftKvCache(capture_dir / name);
        !status.ok()) {
      return absl::InternalError(status.message());
    }
    unpacked_.insert(name);
  }
  return absl::OkStatus();
}

// ---- Model

class LiteRtLmModel final : public Model {
 public:
  LiteRtLmModel(LiteRtLmEngine* handle, std::filesystem::path data_dir,
                int context_length, bool gpu, Sampling sampling,
                v1::LoadResponse info)
      : handle_(handle),
        data_dir_(std::move(data_dir)),
        context_length_(context_length),
        gpu_(gpu),
        sampling_(sampling),
        info_(std::move(info)) {}
  ~LiteRtLmModel() override { litert_lm_engine_delete(handle_); }

  const v1::LoadResponse& Info() const override { return info_; }
  absl::StatusOr<std::unique_ptr<Chat>> CreateChat(
      const v1::CreateChatRequest& request) override;

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

  const v1::LoadResponse info_;
};

// Returns the session config `options` ask for, or nullptr. Sampling options
// left unset take their values from `sampling`: the runtime's sampler takes
// all four at once.
LiteRtLmSessionConfig* SessionConfig(const v1::ChatOptions& options,
                                     const ChatRuntimeOptions& runtime,
                                     const Sampling& sampling) {
  LiteRtLmSessionConfig* session = litert_lm_session_config_create();
  if (session == nullptr) return nullptr;
  litert_lm_session_config_set_max_output_tokens(
      session, options.max_completion_tokens());
  if (runtime.apply_prompt_template) {
    litert_lm_session_config_set_apply_prompt_template(
        session, *runtime.apply_prompt_template);
  }
  litert_lm_session_config_set_enable_speculative_decoding(
      session, runtime.speculative_decoding);
  if (options.has_top_k() || options.has_top_p() || options.has_temperature() ||
      options.has_seed()) {
    LiteRtLmSamplerParams* sampler =
        litert_lm_sampler_params_create(kLiteRtLmSamplerTypeTopP);
    if (sampler == nullptr) {
      litert_lm_session_config_delete(session);
      return nullptr;
    }
    litert_lm_sampler_params_set_top_k(
        sampler, options.has_top_k() ? options.top_k() : sampling.top_k);
    litert_lm_sampler_params_set_top_p(
        sampler, options.has_top_p() ? options.top_p() : sampling.top_p);
    litert_lm_sampler_params_set_temperature(
        sampler, options.has_temperature() ? options.temperature()
                                           : sampling.temperature);
    litert_lm_sampler_params_set_seed(
        sampler, options.has_seed() ? options.seed() : sampling.seed);
    litert_lm_session_config_set_sampler_params(session, sampler);
    litert_lm_sampler_params_delete(sampler);
  }
  return session;
}

absl::StatusOr<std::unique_ptr<Chat>> LiteRtLmModel::CreateChat(
    const v1::CreateChatRequest& request) {
  const v1::ChatOptions& options = request.options();
  if (options.max_completion_tokens() <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "max_completion_tokens is ", options.max_completion_tokens(), "."));
  }
  absl::StatusOr<ChatRuntimeOptions> runtime =
      ParseChatOptions(options.runtime_options());
  if (!runtime.ok()) return runtime.status();

  // The texts first: an error in them costs no runtime object. A system
  // message is the first message or none: the runtime takes it apart.
  std::optional<std::string> system_message;
  Json history = Json::array();
  for (int i = 0; i < request.messages_size(); ++i) {
    const v1::ChatMessage& message = request.messages(i);
    if (message.role() == v1::ChatMessage::SYSTEM) {
      if (i != 0) {
        return absl::InvalidArgumentError(
            "A system message may only be the first message.");
      }
      absl::StatusOr<std::string> text = Dump(Json(message.content()));
      if (!text.ok()) return text.status();
      system_message = *std::move(text);
    } else {
      absl::StatusOr<const char*> role = RoleName(message.role());
      if (!role.ok()) return role.status();
      history.push_back(Json{{"role", *role}, {"content", message.content()}});
    }
  }
  std::optional<std::string> messages;
  if (!history.empty()) {
    absl::StatusOr<std::string> text = Dump(history);
    if (!text.ok()) return text.status();
    messages = *std::move(text);
  }

  LiteRtLmSessionConfig* session = SessionConfig(options, *runtime, sampling_);
  if (session == nullptr) {
    return absl::InternalError("Could not create the session config.");
  }
  LiteRtLmConversationConfig* config = litert_lm_conversation_config_create();
  if (config == nullptr) {
    litert_lm_session_config_delete(session);
    return absl::InternalError("Could not create the conversation config.");
  }
  litert_lm_conversation_config_set_session_config(config, session);
  if (runtime->thinking || runtime->thinking_token_budget) {
    LiteRtLmThinkingConfig* thinking = litert_lm_thinking_config_create();
    if (thinking == nullptr) {
      litert_lm_conversation_config_delete(config);
      litert_lm_session_config_delete(session);
      return absl::InternalError("Could not create the thinking config.");
    }
    if (runtime->thinking) {
      litert_lm_thinking_config_set_enable_thinking(thinking,
                                                    *runtime->thinking);
    }
    if (runtime->thinking_token_budget) {
      litert_lm_thinking_config_set_thinking_token_budget(
          thinking, *runtime->thinking_token_budget);
    }
    litert_lm_conversation_config_set_thinking_config(config, thinking);
    litert_lm_thinking_config_delete(thinking);
  }
  if (system_message) {
    litert_lm_conversation_config_set_system_message(config,
                                                     system_message->c_str());
  }
  if (messages) {
    litert_lm_conversation_config_set_messages(config, messages->c_str());
  }
  LiteRtLmConversation* handle = litert_lm_conversation_create(handle_, config);
  litert_lm_conversation_config_delete(config);
  litert_lm_session_config_delete(session);
  if (handle == nullptr) {
    return absl::InternalError(
        "Could not create the conversation. Check the native logs.");
  }
  return std::unique_ptr<Chat>(new LiteRtLmChat(handle, handle_, data_dir_,
                                                options.max_completion_tokens(),
                                                context_length_, gpu_));
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
    const v1::LoadRequest& request) {
  if (request.context_length() <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("context_length is ", request.context_length(), "."));
  }
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
  if (request.device() == v1::GPU || options->vision_backend == v1::GPU ||
      options->audio_backend == v1::GPU) {
    absl::StatusOr<std::filesystem::path> library = LibraryDirectory();
    if (!library.ok()) return library.status();
    if (absl::Status loaded = LoadLiteRtGpuPlugins(library->string());
        !loaded.ok()) {
      return loaded;
    }
  }

  // Process-wide: the runtime has no level per engine.
  litert_lm_set_min_log_level(kLiteRtLmLogSeverityWarning);
  const std::string data_dir_text = data_dir.string();
  LiteRtLmEngineSettings* settings = litert_lm_engine_settings_create(
      model.c_str(), *backend,
      options->vision_backend ? *BackendName(*options->vision_backend)
                              : nullptr,
      options->audio_backend ? *BackendName(*options->audio_backend) : nullptr);
  if (settings == nullptr) {
    return absl::InternalError("Could not create engine settings.");
  }
  Apply(request, data_dir_text, *options, settings);
  LiteRtLmEngine* handle = litert_lm_engine_create(settings);
  litert_lm_engine_settings_delete(settings);
  if (handle == nullptr) {
    return absl::InternalError(
        "Model initialization failed. Check device memory and native logs.");
  }
  absl::StatusOr<v1::LoadResponse> tokens = ReadTokens(handle);
  if (!tokens.ok()) {
    litert_lm_engine_delete(handle);
    return tokens.status();
  }
  absl::StatusOr<Sampling> sampling = ReadSampling(model);
  if (!sampling.ok()) {
    litert_lm_engine_delete(handle);
    return sampling.status();
  }
  return std::unique_ptr<Model>(new LiteRtLmModel(
      handle, data_dir, request.context_length(), request.device() == v1::GPU,
      *sampling, *std::move(tokens)));
}

bool LiteRtLmRuntime::CanRunOnGpu() const {
  absl::StatusOr<std::filesystem::path> library = LibraryDirectory();
  return library.ok() && HasLiteRtGpuPlugins(library->string());
}

}  // namespace model_debugger::runner::llm
