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

// What the Runner needs of an LLM runtime, in the protocol's own types: a
// runtime loads a model, a model creates the chat, the chat takes turns.
// LiteRT-LM is one implementation (litert_lm/litert_lm_runtime.h).

#ifndef MODEL_DEBUGGER_RUNNER_LLM_RUNTIME_H_
#define MODEL_DEBUGGER_RUNNER_LLM_RUNTIME_H_

#include <filesystem>
#include <functional>
#include <memory>

#include "absl/status/statusor.h"
#include "model_debugger/v1/runner.pb.h"

namespace model_debugger::runner::llm {

// Receives each piece of a reply as the model produces it, before Send
// returns, on a thread of the runtime's choosing, which may be the caller's. It
// may call Cancel on the chat and nothing else of it.
using ChunkCallback = std::function<void(const v1::Chunk& chunk)>;

// What a turn leaves behind: the reply, and where the runtime wrote the turn's
// tensors. The directory is the Runner's to read and never leaves its machine.
struct Turn {
  v1::Completion completion;
  std::filesystem::path capture_dir;
};

// A chat on a model: a prompt template, a context, and the messages so far.
//
// Not thread-safe, with one exception: Cancel may be called from any thread.
// A Chat must not outlive its Model.
class Chat {
 public:
  virtual ~Chat() = default;

  // Sends `message` as the user's next turn and returns the whole reply, after
  // giving each piece of it to `on_chunk` as it was produced. Returns when the
  // model has finished, and not before. The completion's model_debugger.turn
  // counts the turns completed on this chat, this one included, from 1. Its
  // finish_reason is LENGTH if the reply was cut off by max_completion_tokens
  // or by a full context.
  //
  // Returns Cancelled if Cancel was called while this Send was in progress:
  // what `on_chunk` received stays received. The chat is then over, and only
  // its destructor may follow. Returns InvalidArgument if `message` is not a
  // user message in UTF-8, FailedPrecondition if the chat is over, and
  // Internal if the runtime fails or its reply cannot be read: never
  // InvalidArgument for what the model produced.
  virtual absl::StatusOr<Turn> Send(const v1::ChatMessage& message,
                                    const ChunkCallback& on_chunk) = 0;

  // Stops the Send in progress, if there is one, which ends the chat; does
  // nothing otherwise. May be called from any thread.
  virtual void Cancel() = 0;
};

// A loaded model.
//
// Not thread-safe. A Model must not outlive its Runtime, and must outlive its
// Chat.
class Model {
 public:
  virtual ~Model() = default;

  // What is known only once the model is loaded: its start and stop tokens.
  virtual const v1::LoadResponse& Info() const = 0;

  // Creates the chat with `request`'s starting messages and options.
  //
  // Returns InvalidArgument if a system message is not first or a text is not
  // UTF-8, Unimplemented if `request` asks for what this runtime does not do,
  // and Internal if the runtime refuses.
  virtual absl::StatusOr<std::unique_ptr<Chat>> CreateChat(
      const v1::CreateChatRequest& request) = 0;
};

// An LLM runtime. Every runtime here captures tensors: the Runner exists to
// read them.
//
// Not thread-safe. A Runtime must outlive its Models.
class Runtime {
 public:
  virtual ~Runtime() = default;

  // What this runtime is, known before any model: its name and version, and
  // the JSON Schemas of the options its Load and its chats take beyond the
  // fixed ones.
  virtual const v1::RuntimeInfo& Info() const = 0;

  // Whether this build of the runtime can run a model on the GPU: what Load
  // needs for it is where Load looks.
  virtual bool CanRunOnGpu() const = 0;

  // Loads the model at `model` with `request`'s device, context length and
  // runtime options. The runtime may write under `data_dir`, and nowhere
  // else; captures land somewhere beneath it.
  //
  // Returns InvalidArgument if `request.context_length` is not positive or a
  // runtime option is missing, unknown or of the wrong type; NotFound if
  // `model` is missing, and for nothing else; FailedPrecondition if
  // `data_dir` is missing or this build lacks what `request.device` needs,
  // such as its GPU plugins; Unimplemented if this runtime does not run on
  // `request.device`; Unavailable if the device is not there to use;
  // Internal if the runtime refuses, in which case its log says why.
  virtual absl::StatusOr<std::unique_ptr<Model>> Load(
      const std::filesystem::path& model, const std::filesystem::path& data_dir,
      const v1::LoadRequest& request) = 0;
};

}  // namespace model_debugger::runner::llm

#endif  // MODEL_DEBUGGER_RUNNER_LLM_RUNTIME_H_
