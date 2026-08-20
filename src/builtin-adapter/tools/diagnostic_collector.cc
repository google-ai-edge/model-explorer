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

#include "tools/diagnostic_collector.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"

namespace tooling {
namespace visualization_client {

namespace {

constexpr size_t kMaxSamples = 3;

void AddSample(std::vector<std::string>& samples, std::string sample) {
  if (samples.size() < kMaxSamples) {
    samples.push_back(std::move(sample));
  }
}

void IncrementCounter(absl::flat_hash_map<std::string, int>& map,
                      absl::string_view key, int* total_counter = nullptr) {
  map[key]++;
  if (total_counter != nullptr) {
    (*total_counter)++;
  }
}

}  // namespace

void DiagnosticCollector::RecordMissingOpDef(absl::string_view op_label) {
  IncrementCounter(missing_op_defs_, op_label, &total_missing_op_def_nodes_);
}

void DiagnosticCollector::RecordQuantizationMismatch(
    absl::string_view tensor_name, size_t scale_size, size_t zero_point_size) {
  quantization_mismatches_++;
  AddSample(sample_quant_mismatches_,
            absl::StrFormat("tensor '%s': scale(%zu) != zp(%zu)", tensor_name,
                            scale_size, zero_point_size));
}

void DiagnosticCollector::RecordIncompleteEdge(int tensor_index,
                                               absl::string_view details) {
  incomplete_edges_count_++;
  AddSample(sample_incomplete_edges_,
            absl::StrFormat("tensor %d: %s", tensor_index, details));
}

void DiagnosticCollector::RecordOptionError(absl::string_view op_label,
                                            absl::string_view error) {
  IncrementCounter(option_errors_, op_label, &total_option_errors_);
  AddSample(sample_option_errors_,
            absl::StrFormat("op '%s': %s", op_label, error));
}

void DiagnosticCollector::RecordShardyEdgeFailure(absl::string_view reason) {
  IncrementCounter(shardy_edge_failures_, reason);
}

void DiagnosticCollector::RecordMissingTensorName(absl::string_view op_name) {
  IncrementCounter(missing_tensor_names_, op_name,
                   &total_missing_tensor_names_);
}

bool DiagnosticCollector::HasDiagnostics() const {
  return total_missing_op_def_nodes_ > 0 || quantization_mismatches_ > 0 ||
         incomplete_edges_count_ > 0 || total_option_errors_ > 0 ||
         !shardy_edge_failures_.empty() || total_missing_tensor_names_ > 0;
}

void DiagnosticCollector::EmitSummary(absl::string_view context_name) const {
  if (!HasDiagnostics()) {
    return;
  }

  if (total_missing_op_def_nodes_ > 0) {
    std::vector<std::string> op_breakdown;
    op_breakdown.reserve(missing_op_defs_.size());
    for (const auto& [op, count] : missing_op_defs_) {
      op_breakdown.push_back(absl::StrFormat("%s: %d", op, count));
    }
    std::sort(op_breakdown.begin(), op_breakdown.end());
    LOG(INFO) << absl::StrFormat(
        "[%s] %d nodes omitted tensor argument tags across %d unique op "
        "defs: [%s]",
        context_name, total_missing_op_def_nodes_, missing_op_defs_.size(),
        absl::StrJoin(op_breakdown, ", "));
  }

  if (quantization_mismatches_ > 0) {
    LOG(WARNING) << absl::StrFormat(
        "[%s] Detected %d quantization parameter size mismatches. Samples: "
        "[%s]",
        context_name, quantization_mismatches_,
        absl::StrJoin(sample_quant_mismatches_, "; "));
  }

  if (incomplete_edges_count_ > 0) {
    VLOG(1) << absl::StrFormat(
        "[%s] Detected %d incomplete edges. Samples: [%s]", context_name,
        incomplete_edges_count_, absl::StrJoin(sample_incomplete_edges_, "; "));
  }

  if (total_option_errors_ > 0) {
    VLOG(1) << absl::StrFormat(
        "[%s] Failed to extract options for %d nodes. Samples: [%s]",
        context_name, total_option_errors_,
        absl::StrJoin(sample_option_errors_, "; "));
  }

  if (!shardy_edge_failures_.empty()) {
    VLOG(1) << absl::StrFormat(
        "[%s] Failed to extract %d Shardy propagation edge references.",
        context_name, shardy_edge_failures_.size());
  }

  if (total_missing_tensor_names_ > 0) {
    VLOG(1) << absl::StrFormat(
        "[%s] %d operations omitted tensor name debug locations across %d "
        "unique op types.",
        context_name, total_missing_tensor_names_,
        missing_tensor_names_.size());
  }
}

}  // namespace visualization_client
}  // namespace tooling
