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
// runtime loads a model, a model creates a chat, the chat takes turns.
// LiteRT-LM is one implementation (litert_lm/litert_lm_runtime.h).

#ifndef MODEL_DEBUGGER_RUNNER_LLM_RUNTIME_H_
#define MODEL_DEBUGGER_RUNNER_LLM_RUNTIME_H_

#include <filesystem>  // NOLINT(build/c++17)
#include <functional>
#include <memory>
#include <string_view>

#include "model_debugger/protocol/runner.pb.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace model_debugger::runner::llm {

// Receives each piece of a reply as the model produces it, before Send
// returns, on a thread of the runtime's choosing, which may be the caller's. It
// may call Cancel on the chat and nothing else of it.
using ChunkCallback = std::function<void(const protocol::Chunk& chunk)>;

// What a turn leaves behind: the reply, and where the runtime wrote the turn's
// tensors. The directory is the Runner's to read and never leaves its machine.
struct Turn {
  protocol::Completion completion;
  std::filesystem::path capture_dir;
};

// The payloads of the ResourceExhausted a Send returns for a turn the context
// has no room for, and CreateChat for starting messages it has no room for,
// by type URL, each a decimal number: the tokens the turn, or the start of
// the chat, needs, and the tokens the context has left.
inline constexpr std::string_view kTurnTokensPayload =
    "model_debugger.runner/turn_tokens";
inline constexpr std::string_view kTokensLeftPayload =
    "model_debugger.runner/tokens_left";

// A chat on a model: a prompt template, a context, and the messages so far.
//
// Not thread-safe, with one exception: Cancel may be called from any thread.
// A Chat must not outlive its Model.
class Chat {
 public:
  virtual ~Chat() = default;

  // Sends `message`, the user's next turn, and returns the whole reply, after
  // giving each piece of it to `on_chunk` as it was produced. Returns when the
  // model has finished, and not before. The completion's model_debugger.turn
  // counts the turns completed on this chat, this one included, from 1. Its
  // finish_reason is LENGTH if the reply was cut off by max_completion_tokens
  // or by a full context.
  //
  // Returns Cancelled if Cancel was called while this Send was in progress:
  // what `on_chunk` received stays received. The chat is then over, and only
  // its destructor may follow. Returns InvalidArgument if `message` is not a
  // user message in UTF-8; ResourceExhausted if the turn needs more tokens
  // than the context has left, its payloads kTurnTokensPayload and
  // kTokensLeftPayload saying how many; FailedPrecondition if the chat is
  // over; and Internal if the runtime fails or its reply cannot be read: never
  // InvalidArgument for what the model produced. InvalidArgument and
  // ResourceExhausted change nothing: the chat takes the next turn as if this
  // one had not been sent.
  virtual absl::StatusOr<Turn> Send(const protocol::ChatMessage& message,
                                    const ChunkCallback& on_chunk) = 0;

  // Stops the Send in progress, if there is one, which ends the chat; does
  // nothing otherwise. May be called from any thread.
  virtual void Cancel() = 0;
};

// A loaded model, with one chat at a time.
//
// Not thread-safe. A Model must not outlive its Runtime, and must outlive its
// Chats.
class Model {
 public:
  virtual ~Model() = default;

  // What is known only once the model is loaded: its start and stop tokens,
  // and the sampling it sets for a chat that sets none.
  virtual const protocol::LoadResponse& Info() const = 0;

  // Creates a chat with `request`'s starting messages and options, from an
  // empty context. The caller has deleted the chat before, if any, so that
  // two contexts never take memory at once, and has checked the fixed
  // options against the protocol's ranges (ChatOptions),
  // max_completion_tokens at least 1 among them.
  //
  // Returns InvalidArgument if a system message is not first, a role is
  // unspecified or unknown, a text is not UTF-8, or a runtime option is
  // missing, unknown or of the wrong type; ResourceExhausted if the starting
  // messages, rendered as the first turn's prompt with an empty user message,
  // need more tokens than the context has, its payloads kTurnTokensPayload
  // and kTokensLeftPayload saying how many; Internal if the runtime refuses.
  virtual absl::StatusOr<std::unique_ptr<Chat>> CreateChat(
      const protocol::CreateChatRequest& request) = 0;
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
  virtual const protocol::RuntimeInfo& Info() const = 0;

  // OK if this build of the runtime can run a model on the GPU here, as far
  // as can be told without loading one: what Load needs for it is where Load
  // looks. Otherwise FailedPrecondition saying why, or Internal if the
  // runtime cannot tell. A Load on the GPU then fails the same way.
  virtual absl::Status CanRunOnGpu() const = 0;

  // Loads the model at `model` with `request`'s device, context length and
  // runtime options. The caller has checked that the context length is
  // positive and the device is CPU or GPU. The runtime may write under
  // `data_dir`, and nowhere else; captures land somewhere beneath it.
  //
  // Returns InvalidArgument if a runtime option is missing, unknown or of the
  // wrong type; NotFound if `model` is missing, and for nothing else;
  // FailedPrecondition if `data_dir` is missing or this build cannot run on
  // the GPU here, as CanRunOnGpu says, or its GPU plugins do not load;
  // Internal if the runtime refuses, in which case its log says why. Never
  // Unavailable: the protocol keeps that for a runner that cannot be
  // reached.
  virtual absl::StatusOr<std::unique_ptr<Model>> Load(
      const std::filesystem::path& model, const std::filesystem::path& data_dir,
      const protocol::LoadRequest& request) = 0;
};

}  // namespace model_debugger::runner::llm

#endif  // MODEL_DEBUGGER_RUNNER_LLM_RUNTIME_H_
