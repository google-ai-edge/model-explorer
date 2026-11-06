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

// The tokens LiteRT-LM fed the model during a turn, read from the turn's
// capture. No call of its C API returns them, and a model that takes
// embeddings is captured without ids: the positions and phases are measured,
// and each id comes from what the runtime reports or from the row's bytes,
// or stays unset where a check fails:
// - the start token: the model's, as LoadResponse.start_token reports it;
// - the prompt's tokens, up to the first decode step's: LiteRT-LM reports
//   none, so the chat renders and tokenizes the message as the runtime does,
//   and the ids are taken only if the prompt fed one row for each, at the
//   positions the runtime's token counts say, rows of equal bytes being
//   given equal ids;
// - each later decode step's: the token the step before sampled, as the
//   runtime released it, under the same check. Where a row of the prompt and
//   one of the reply disagree, neither is taken;
// - the rest, the token held since the turn before and every row of a
//   rewrite: read from the row itself, which holds either the id
//   (FillInputBufferWithToken) or the bytes of a row whose id is known. The
//   token held must also be what the turn before sampled last, as far as the
//   runtime reports it: its last token released, or else one of the model's
//   stop tokens. A held row whose bytes name another id contradicts them for
//   the whole chat, before this turn's other rows are named.
//
// A rewrite is a prefill after the decode steps, from the position decoding
// began: after a reply with reasoning, LiteRT-LM writes the reply into the
// context again without it, and what the context held there is gone.

#ifndef MODEL_DEBUGGER_RUNNER_LLM_LITERT_LM_FED_TOKENS_H_
#define MODEL_DEBUGGER_RUNNER_LLM_LITERT_LM_FED_TOKENS_H_

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "model_debugger/protocol/runner.pb.h"
#include "absl/status/statusor.h"

namespace model_debugger::runner::llm {

// What a chat knows of a turn beside its capture.
struct TurnFacts {
  // The model's start token, if it has one: what the first turn feeds first.
  std::optional<int32_t> start_token;
  // The prompt the turn feeds after its first token, as the conversation
  // renders and tokenizes it.
  std::vector<int32_t> prompt;
  // The conversation's token count before the Send, and after it.
  int context_before = 0;
  int context_after = 0;
  // The runtime's count of the turn's decode steps.
  int decode_steps = 0;
  // The model's stop tokens of one id: what a reply that stopped sampled
  // last, never released.
  std::set<int32_t> stop_tokens;
  // What the token held since the turn before can be, as that turn's
  // FedTurn::held says; unset where the chat does not know.
  std::optional<std::set<int32_t>> held;
};

// The ids of the input rows a chat's turns fed, by the rows' bytes: the
// runtime looks an id up the same way each time, so equal bytes are an equal
// id. Kept for the whole chat, which may feed in one turn a row learned in
// another.
class TokenRows {
 public:
  // The id the row of `key` was learned as, if any.
  std::optional<int32_t> Find(size_t key) const;

  // Learns that the row of `key` is `id`, unless it was learned or
  // contradicted before: the first id learned stays.
  void Learn(size_t key, int32_t id);

  // Forgets the row of `key` for the rest of the chat: two sources named it
  // differently, and which of them was wrong cannot be told.
  void Contradict(size_t key);

 private:
  std::map<size_t, int32_t> ids_;
  std::set<size_t> contradicted_;
};

// A turn's fed tokens, their texts left empty.
struct FedTurn {
  std::vector<protocol::FedToken> tokens;
  // The position the first decode step fed.
  int first_decode_position = 0;
  // What the token the turn leaves held can be, if the turn ended by
  // decoding: the id its last decode step sampled, which is the last one
  // released, or one of TurnFacts::stop_tokens where that was not released.
  // Unset after a rewrite, whose last token only its row names, and where the
  // counters disagree with the capture.
  std::optional<std::set<int32_t>> held;
};

// Reads which tokens the turn fed from its capture in `dir`: the tensor files
// `names`, and the lines of generated_tokens.jsonl past the first
// `lines_before`. Names each fed id as FedToken.id says, from `facts` and the
// row's bytes, learning and consulting `rows`; an id that nothing names, or
// that a check contradicts, stays unset. The prompt's ids, rendered rather
// than reported, are taken only if the prompt has a row for each, at the
// positions the counters say; where they and the reply's disagree on a row's
// bytes, neither is taken.
//
// Returns Internal if a file cannot be read, the turn has no decode step, or
// a call's input does not hold one row for each of its positions.
absl::StatusOr<FedTurn> ReadFedTokens(const std::filesystem::path& dir,
                                      const std::set<std::string>& names,
                                      int lines_before, const TurnFacts& facts,
                                      TokenRows& rows);

}  // namespace model_debugger::runner::llm

#endif  // MODEL_DEBUGGER_RUNNER_LLM_LITERT_LM_FED_TOKENS_H_
