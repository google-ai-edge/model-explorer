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

#include "model_debugger/runner/llm/litert_lm/fed_tokens.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "model_debugger/protocol/runner.pb.h"
#include "model_debugger/runner/util/json.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"

namespace model_debugger::runner::llm {
namespace {

// The RuntimeDebugger's names: a call's input tensor is captured as
// <signature>_pre_<input>_step_<step>.safetensors, its released tokens are
// appended to generated_tokens.jsonl.
constexpr std::string_view kPositionsInfix = "_pre_input_pos_step_";
constexpr std::string_view kSuffix = ".safetensors";
constexpr std::string_view kTokensFile = "generated_tokens.jsonl";

// ---- Safetensors

// The one tensor of a capture file.
struct Tensor {
  std::string dtype;
  std::string data;
};

// Reads the one tensor of `file`, a safetensors file as the RuntimeDebugger
// writes it.
absl::StatusOr<Tensor> ReadTensor(const std::filesystem::path& file) {
  std::ifstream in(file, std::ios::binary);
  std::string bytes((std::istreambuf_iterator<char>(in)),
                    std::istreambuf_iterator<char>());
  if (!in.is_open() || in.bad()) {
    return absl::InternalError(absl::StrCat("Cannot read ", file.string()));
  }
  const auto refuse = [&file](std::string_view why) {
    return absl::InternalError(
        absl::StrCat("Cannot read ", file.filename().string(), ": ", why));
  };
  // The header's size as a little-endian uint64, which every platform of the
  // Runner is, then the header, then the data.
  uint64_t header_size = 0;
  if (bytes.size() < sizeof(header_size)) return refuse("not safetensors.");
  std::memcpy(&header_size, bytes.data(), sizeof(header_size));
  if (header_size > bytes.size() - sizeof(header_size)) {
    return refuse("not safetensors.");
  }
  const size_t data_start = sizeof(header_size) + header_size;
  absl::StatusOr<Json> header = ParseJson(
      std::string_view(bytes).substr(sizeof(header_size), header_size));
  if (!header.ok() || !header->is_object()) {
    return refuse("no safetensors header.");
  }
  const Json* tensor = nullptr;
  for (const auto& [key, value] : header->items()) {
    if (key == "__metadata__") continue;
    if (tensor != nullptr) return refuse("more than one tensor.");
    tensor = &value;
  }
  if (tensor == nullptr || !tensor->is_object() || !tensor->contains("dtype") ||
      !tensor->at("dtype").is_string() || !tensor->contains("data_offsets")) {
    return refuse("a malformed header.");
  }
  const Json& offsets = tensor->at("data_offsets");
  if (!offsets.is_array() || offsets.size() != 2 ||
      !offsets[0].is_number_unsigned() || !offsets[1].is_number_unsigned()) {
    return refuse("a malformed header.");
  }
  const size_t begin = offsets[0].get<size_t>();
  const size_t end = offsets[1].get<size_t>();
  if (begin > end || end > bytes.size() - data_start) {
    return refuse("data past the end of the file.");
  }
  Tensor read;
  read.dtype = tensor->at("dtype").get<std::string>();
  read.data = bytes.substr(data_start + begin, end - begin);
  return read;
}

// ---- The calls of a turn

// One input row of a call, at one position.
struct Row {
  int position = 0;
  // Set where the runtime fed the id itself in place of an embedding: the
  // row is the id as an int32, then zeros (FillInputBufferWithToken).
  std::optional<int32_t> raw;
  // Otherwise the row's bytes, hashed; unset if no embeddings were captured.
  std::optional<size_t> key;
};

// One forward pass, its rows those of the positions it fed.
struct Call {
  bool decode = false;
  std::vector<Row> rows;
};

// Returns the id `row` holds itself, if it is an id and not an embedding.
std::optional<int32_t> RawId(std::string_view row) {
  int32_t id = 0;
  if (row.size() < sizeof(id)) return std::nullopt;
  std::memcpy(&id, row.data(), sizeof(id));
  if (id <= 0) return std::nullopt;
  for (const char byte : row.substr(sizeof(id))) {
    if (byte != 0) return std::nullopt;
  }
  return id;
}

// Splits the data of `tensor`, read from `file`, into `count` rows of equal
// size: one for each position of its call, padding included.
absl::StatusOr<std::vector<std::string_view>> Rows(const Tensor& tensor,
                                                   size_t count,
                                                   std::string_view file) {
  if (tensor.data.empty() || tensor.data.size() % count != 0) {
    return absl::InternalError(absl::StrCat(
        file, " does not hold one row for each of its ", count, " positions."));
  }
  const size_t size = tensor.data.size() / count;
  std::vector<std::string_view> rows;
  for (size_t i = 0; i < count; ++i) {
    rows.push_back(std::string_view(tensor.data).substr(i * size, size));
  }
  return rows;
}

// Reads the call that captured `positions_file` in `dir` as
// <signature>_pre_input_pos_step_<step>, and its embeddings if `names` has
// them.
absl::StatusOr<Call> ReadCall(const std::filesystem::path& dir,
                              const std::set<std::string>& names,
                              const std::string& positions_file) {
  const size_t infix = positions_file.find(kPositionsInfix);
  const std::string signature = positions_file.substr(0, infix);
  const std::string step = positions_file.substr(
      infix + kPositionsInfix.size(),
      positions_file.size() - infix - kPositionsInfix.size() - kSuffix.size());
  absl::StatusOr<Tensor> positions = ReadTensor(dir / positions_file);
  if (!positions.ok()) return positions.status();
  if (positions->dtype != "I32" || positions->data.empty() ||
      positions->data.size() % sizeof(int32_t) != 0) {
    return absl::InternalError(
        absl::StrCat(positions_file, " does not hold int32 positions."));
  }
  std::vector<int32_t> all(positions->data.size() / sizeof(int32_t));
  std::memcpy(all.data(), positions->data.data(), positions->data.size());
  // The positions the call fed are the leading run; padding follows at 0.
  size_t fed = 1;
  while (fed < all.size() && all[fed] == all[0] + static_cast<int32_t>(fed)) {
    ++fed;
  }

  Call call;
  call.decode = signature == "decode";
  call.rows.resize(fed);
  for (size_t i = 0; i < fed; ++i) call.rows[i].position = all[i];
  const auto input = [&](std::string_view name) {
    return absl::StrCat(signature, "_pre_", name, "_step_", step, kSuffix);
  };
  const std::string embeddings_file = input("embeddings");
  if (names.count(embeddings_file) == 0) return call;
  absl::StatusOr<Tensor> embeddings = ReadTensor(dir / embeddings_file);
  if (!embeddings.ok()) return embeddings.status();
  absl::StatusOr<std::vector<std::string_view>> embedding_rows =
      Rows(*embeddings, all.size(), embeddings_file);
  if (!embedding_rows.ok()) return embedding_rows.status();
  // Part of the key where captured: the per-layer lookup is by id as well.
  const std::string per_layer_file = input("per_layer_embeddings");
  std::optional<Tensor> per_layer;
  std::vector<std::string_view> per_layer_rows(all.size());
  if (names.count(per_layer_file) != 0) {
    absl::StatusOr<Tensor> read = ReadTensor(dir / per_layer_file);
    if (!read.ok()) return read.status();
    per_layer = *std::move(read);
    absl::StatusOr<std::vector<std::string_view>> split =
        Rows(*per_layer, all.size(), per_layer_file);
    if (!split.ok()) return split.status();
    per_layer_rows = *std::move(split);
  }
  for (size_t i = 0; i < fed; ++i) {
    call.rows[i].raw = RawId((*embedding_rows)[i]);
    if (!call.rows[i].raw) {
      call.rows[i].key = absl::HashOf((*embedding_rows)[i], per_layer_rows[i]);
    }
  }
  return call;
}

// Returns the ids the turn released, in order: the lines of `file` past the
// first `lines_before`, each a record of RuntimeDebugger::WriteTokensToFile.
// No file is none released.
absl::StatusOr<std::vector<int32_t>> ReadReleased(
    const std::filesystem::path& file, int lines_before) {
  std::vector<int32_t> released;
  std::ifstream in(file);
  std::string line;
  for (int i = 0; std::getline(in, line); ++i) {
    if (i < lines_before) continue;
    absl::StatusOr<Json> record = ParseJson(line);
    if (!record.ok() || !record->is_object() ||
        !record->contains("token_ids") || !record->at("token_ids").is_array() ||
        record->at("token_ids").empty() ||
        !record->at("token_ids")[0].is_array()) {
      return absl::InternalError(
          absl::StrCat("Line ", i + 1, " of ", file.filename().string(),
                       " is not a record of released tokens."));
    }
    for (const Json& id : record->at("token_ids")[0]) {
      if (!id.is_number_integer()) {
        return absl::InternalError(
            absl::StrCat("Line ", i + 1, " of ", file.filename().string(),
                         " has a token id that is not an integer."));
      }
      released.push_back(id.get<int32_t>());
    }
  }
  if (in.bad()) {
    return absl::InternalError(absl::StrCat("Cannot read ", file.string()));
  }
  return released;
}

// ---- Naming the rows

// Returns the id the bytes of `row` name: the id it holds itself, or the one
// a row of the same bytes was learned as.
std::optional<int32_t> FromBytes(const Row& row, const TokenRows& rows) {
  if (row.raw) return row.raw;
  if (row.key) return rows.Find(*row.key);
  return std::nullopt;
}

// Rows a source names, and the ids it names them as.
struct Group {
  std::vector<const Row*> rows;
  // One for each row; unset where the source names none.
  std::vector<std::optional<int32_t>> ids;
  // Whether the ids stand: false once a check contradicts one.
  bool named = true;
};

// Checks the ids of `groups` against what the rows' bytes say, a row learned
// in an earlier turn included, and against each other. A group whose row
// holds another id, or has the bytes of a row named otherwise, by `rows` or
// by any of `groups`, is no longer named, and neither is a group that named
// that other row: which of them is wrong cannot be told. So too for `held`,
// the row of the token held since the turn before, if its bytes name an id
// outside `may_be`: no group names its bytes either. Forgets the key of each
// row so contradicted, then learns the rows of the groups still named.
void CheckAndLearn(const std::vector<Group*>& groups, const Row* held,
                   const std::optional<std::set<int32_t>>& may_be,
                   TokenRows& rows) {
  // Each key named, with the id first named and who named it.
  struct Claim {
    int32_t id = 0;
    bool contradicted = false;
    std::vector<Group*> groups;
  };
  std::map<size_t, Claim> claims;
  for (Group* group : groups) {
    if (!group->named) continue;
    for (size_t i = 0; i < group->rows.size(); ++i) {
      const std::optional<int32_t> id = group->ids[i];
      if (!id) continue;
      const Row& row = *group->rows[i];
      const std::optional<int32_t> named = FromBytes(row, rows);
      if (!row.key) {
        if (named && *named != *id) group->named = false;
        continue;
      }
      const auto [it, inserted] = claims.try_emplace(*row.key);
      Claim& claim = it->second;
      if (inserted) claim.id = *id;
      if (claim.id != *id || (named && *named != *id)) {
        claim.contradicted = true;
      }
      claim.groups.push_back(group);
    }
  }
  for (const auto& [key, claim] : claims) {
    if (!claim.contradicted) continue;
    rows.Contradict(key);
    for (Group* group : claim.groups) group->named = false;
  }
  // The held row's bytes name what an earlier turn learned, or else what a
  // group still named claims for them now.
  if (held != nullptr && held->key && may_be) {
    const auto it = claims.find(*held->key);
    std::optional<int32_t> id = rows.Find(*held->key);
    if (!id && it != claims.end() &&
        std::any_of(it->second.groups.begin(), it->second.groups.end(),
                    [](const Group* group) { return group->named; })) {
      id = it->second.id;
    }
    if (id && may_be->count(*id) == 0) {
      rows.Contradict(*held->key);
      if (it != claims.end()) {
        for (Group* group : it->second.groups) group->named = false;
      }
    }
  }
  for (const auto& [key, claim] : claims) {
    for (const Group* group : claim.groups) {
      if (group->named) {
        rows.Learn(key, claim.id);
        break;
      }
    }
  }
}

// Whether the rows of `group` are at consecutive positions.
bool Contiguous(const std::vector<const Row*>& group) {
  for (size_t i = 1; i < group.size(); ++i) {
    if (group[i]->position != group[i - 1]->position + 1) return false;
  }
  return true;
}

// Appends the rows of `calls`, in order, to `group`.
void Append(const std::vector<const Call*>& calls,
            std::vector<const Row*>& group) {
  for (const Call* call : calls) {
    for (const Row& row : call->rows) group.push_back(&row);
  }
}

}  // namespace

std::optional<int32_t> TokenRows::Find(size_t key) const {
  const auto it = ids_.find(key);
  if (it == ids_.end()) return std::nullopt;
  return it->second;
}

void TokenRows::Learn(size_t key, int32_t id) {
  if (contradicted_.count(key) == 0) ids_.emplace(key, id);
}

void TokenRows::Contradict(size_t key) {
  ids_.erase(key);
  contradicted_.insert(key);
}

absl::StatusOr<FedTurn> ReadFedTokens(const std::filesystem::path& dir,
                                      const std::set<std::string>& names,
                                      int lines_before, const TurnFacts& facts,
                                      TokenRows& rows) {
  std::vector<Call> calls;
  for (const std::string& name : names) {
    if (name.find(kPositionsInfix) == std::string::npos ||
        !absl::EndsWith(name, kSuffix)) {
      continue;
    }
    absl::StatusOr<Call> call = ReadCall(dir, names, name);
    if (!call.ok()) return call.status();
    calls.push_back(*std::move(call));
  }
  absl::StatusOr<std::vector<int32_t>> released =
      ReadReleased(dir / kTokensFile, lines_before);
  if (!released.ok()) return released.status();

  // The calls by phase, each by the first position it fed. A prefill after
  // the first decode step starts where decoding did, or after: it rewrites.
  std::vector<const Call*> decodes;
  for (const Call& call : calls) {
    if (call.decode) decodes.push_back(&call);
  }
  const auto by_position = [](const Call* a, const Call* b) {
    return a->rows[0].position < b->rows[0].position;
  };
  std::sort(decodes.begin(), decodes.end(), by_position);
  if (decodes.empty()) {
    return absl::InternalError(absl::StrCat(
        "The turn's capture in ", dir.string(), " has no decode step."));
  }
  const int first_decode = decodes[0]->rows[0].position;
  std::vector<const Call*> prompt_calls;
  std::vector<const Call*> rewrite_calls;
  for (const Call& call : calls) {
    if (call.decode) continue;
    (call.rows[0].position < first_decode ? prompt_calls : rewrite_calls)
        .push_back(&call);
  }
  std::sort(prompt_calls.begin(), prompt_calls.end(), by_position);
  std::sort(rewrite_calls.begin(), rewrite_calls.end(), by_position);

  // The prompt: its prefill and the first decode step, which feeds the
  // prompt's last token. Then the reply, then the rewrite.
  std::vector<const Row*> prompt;
  Append(prompt_calls, prompt);
  std::vector<const Row*> steps;
  Append(decodes, steps);
  prompt.push_back(steps[0]);
  const std::vector<const Row*> reply(steps.begin() + 1, steps.end());
  std::vector<const Row*> rewrite;
  Append(rewrite_calls, rewrite);

  FedTurn turn;
  turn.first_decode_position = first_decode;
  const auto add = [&turn](const std::vector<const Row*>& group,
                           protocol::FedToken::Phase phase) {
    for (const Row* row : group) {
      protocol::FedToken& token = turn.tokens.emplace_back();
      token.set_position(row->position);
      token.set_phase(phase);
    }
  };
  add(std::vector<const Row*>(prompt.begin(), prompt.end() - 1),
      protocol::FedToken::PREFILL);
  add({prompt.back()}, protocol::FedToken::DECODE);
  add(reply, protocol::FedToken::DECODE);
  add(rewrite, protocol::FedToken::PREFILL);

  // The turn as the counters tell it, or no id at all: the rows would not be
  // the tokens they are matched with.
  const int start = facts.context_before == 0 ? 0 : facts.context_before - 1;
  const int last =
      rewrite.empty() ? steps.back()->position : rewrite.back()->position;
  if (facts.decode_steps != static_cast<int>(steps.size()) ||
      prompt[0]->position != start || !Contiguous(prompt) ||
      !Contiguous(steps) || !Contiguous(rewrite) ||
      facts.context_after != last + 2) {
    return turn;
  }

  // The prompt's ids: after the first turn's start token, or the token held
  // since the turn before, which only its row names, the ids the prompt was
  // rendered as. They are taken only if the prompt has a row for each, at the
  // positions checked above.
  const bool held = facts.context_before > 0;
  Group prompt_group;
  prompt_group.rows = prompt;
  if (held) {
    prompt_group.ids.push_back(std::nullopt);
  } else if (facts.start_token) {
    prompt_group.ids.push_back(*facts.start_token);
  }
  prompt_group.ids.insert(prompt_group.ids.end(), facts.prompt.begin(),
                          facts.prompt.end());
  prompt_group.named = prompt_group.ids.size() == prompt.size();
  // Each decode step after the first feeds what the step before sampled, as
  // the runtime released it; a reply that stopped did not release its last.
  Group reply_group;
  reply_group.rows = reply;
  reply_group.named =
      released->size() + 1 == steps.size() || released->size() == steps.size();
  if (reply_group.named) {
    reply_group.ids.assign(released->begin(), released->begin() + reply.size());
  }
  // Checked together: a row of the prompt and one of the reply with the
  // same bytes and different ids leave both unnamed, not only the reply; and
  // so do bytes of the held token's row that name what the turn before
  // cannot have left held.
  CheckAndLearn({&prompt_group, &reply_group}, held ? prompt[0] : nullptr,
                facts.held, rows);

  std::vector<std::optional<int32_t>> ids;
  if (prompt_group.named) {
    if (held) {
      // The held token: as its row names it, if that is what the turn before
      // left held. Bytes that name another were forgotten above.
      std::optional<int32_t> lead = FromBytes(*prompt[0], rows);
      if (lead && facts.held && facts.held->count(*lead) == 0) lead.reset();
      prompt_group.ids[0] = lead;
    }
    ids.insert(ids.end(), prompt_group.ids.begin(), prompt_group.ids.end());
  } else {
    ids.resize(prompt.size());
  }
  if (reply_group.named) {
    ids.insert(ids.end(), reply_group.ids.begin(), reply_group.ids.end());
  } else {
    ids.resize(ids.size() + reply.size());
  }
  for (const Row* row : rewrite) ids.push_back(FromBytes(*row, rows));
  for (size_t i = 0; i < ids.size(); ++i) {
    if (ids[i]) turn.tokens[i].set_id(*ids[i]);
  }

  // What the turn leaves held, if decoding ended it: the token the last step
  // sampled.
  if (rewrite.empty()) {
    if (released->size() == steps.size()) {
      turn.held = std::set<int32_t>{released->back()};
    } else if (released->size() + 1 == steps.size() &&
               !facts.stop_tokens.empty()) {
      turn.held = facts.stop_tokens;
    }
  }
  return turn;
}

}  // namespace model_debugger::runner::llm
