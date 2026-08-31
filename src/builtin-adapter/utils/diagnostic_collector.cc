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

#include "utils/diagnostic_collector.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_log.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

namespace model_explorer {
namespace adapter {

namespace {

std::string FormatSamples(const std::vector<std::string>& samples) {
  return absl::StrJoin(samples, "; ");
}

std::string FormatSortedBreakdown(
    const absl::flat_hash_map<std::string, int>& breakdown) {
  std::vector<std::string> items;
  items.reserve(breakdown.size());
  for (const auto& [key, count] : breakdown) {
    items.push_back(absl::StrFormat("%s: %d", key, count));
  }
  std::sort(items.begin(), items.end());
  return absl::StrJoin(items, ", ");
}

}  // namespace

void DiagnosticCollector::RecordMissingOpDef(absl::string_view op_label) {
  missing_op_defs_.Record(op_label);
}

void DiagnosticCollector::RecordQuantizationMismatch(
    absl::string_view tensor_name, size_t scale_size, size_t zero_point_size) {
  if (quant_mismatches_.NeedsSample()) {
    quant_mismatches_.RecordSample(
        absl::StrFormat("tensor '%s': scale(%zu) != zp(%zu)", tensor_name,
                        scale_size, zero_point_size));
  } else {
    quant_mismatches_.total++;
  }
}

void DiagnosticCollector::RecordIncompleteEdge(int tensor_index,
                                               absl::string_view details) {
  if (incomplete_edges_.NeedsSample()) {
    incomplete_edges_.RecordSample(
        absl::StrFormat("tensor %d: %s", tensor_index, details));
  } else {
    incomplete_edges_.total++;
  }
}

void DiagnosticCollector::RecordOptionError(absl::string_view op_label,
                                            absl::string_view error) {
  if (option_errors_.NeedsSample()) {
    option_errors_.Record(op_label,
                          absl::StrFormat("op '%s': %s", op_label, error));
  } else {
    option_errors_.Record(op_label);
  }
}

void DiagnosticCollector::RecordShardyEdgeFailure(absl::string_view reason) {
  shardy_edge_failures_.Record(reason);
}

void DiagnosticCollector::RecordMissingTensorName(absl::string_view op_name) {
  missing_tensor_names_.Record(op_name);
}

bool DiagnosticCollector::HasDiagnostics() const {
  return !missing_op_defs_.empty() || !quant_mismatches_.empty() ||
         !incomplete_edges_.empty() || !option_errors_.empty() ||
         !shardy_edge_failures_.empty() || !missing_tensor_names_.empty();
}

void DiagnosticCollector::EmitSummary(absl::string_view context_name) const {
  if (!HasDiagnostics()) {
    return;
  }

  if (!missing_op_defs_.empty()) {
    ABSL_LOG(WARNING) << absl::StrFormat(
        "[%s] %d nodes omitted tensor argument tags across %d unique op "
        "defs: [%s]",
        context_name, missing_op_defs_.total, missing_op_defs_.breakdown.size(),
        FormatSortedBreakdown(missing_op_defs_.breakdown));
  }

  if (!quant_mismatches_.empty()) {
    ABSL_LOG(WARNING) << absl::StrFormat(
        "[%s] Detected %d quantization parameter size mismatches. Samples: "
        "[%s]",
        context_name, quant_mismatches_.total,
        FormatSamples(quant_mismatches_.samples));
  }

  if (!incomplete_edges_.empty()) {
    ABSL_VLOG(1) << absl::StrFormat(
        "[%s] Detected %d incomplete edges. Samples: [%s]", context_name,
        incomplete_edges_.total, FormatSamples(incomplete_edges_.samples));
  }

  if (!option_errors_.empty()) {
    ABSL_VLOG(1) << absl::StrFormat(
        "[%s] Failed to extract options for %d nodes. Samples: [%s]",
        context_name, option_errors_.total,
        FormatSamples(option_errors_.samples));
  }

  if (!shardy_edge_failures_.empty()) {
    ABSL_VLOG(1) << absl::StrFormat(
        "[%s] Failed to extract %d Shardy propagation edge references.",
        context_name, shardy_edge_failures_.total);
  }

  if (!missing_tensor_names_.empty()) {
    ABSL_VLOG(1) << absl::StrFormat(
        "[%s] %d operations omitted tensor name debug locations across %d "
        "unique op types.",
        context_name, missing_tensor_names_.total,
        missing_tensor_names_.breakdown.size());
  }
}

std::string DiagnosticCollector::ToJson() const {
  if (!HasDiagnostics()) {
    return "";
  }
  int warning_count =
      total_missing_op_def_nodes() + quantization_mismatches() +
      incomplete_edges_count() + total_option_errors() +
      total_shardy_edge_failures() + total_missing_tensor_names();
  if (warning_count == 0) {
    return "";
  }

  llvm::json::Object diag_obj;
  diag_obj["warningCount"] = warning_count;
  diag_obj["quantMismatches"] = quantization_mismatches();
  diag_obj["missingOpDefs"] = total_missing_op_def_nodes();
  diag_obj["incompleteEdges"] = incomplete_edges_count();
  diag_obj["optionErrors"] = total_option_errors();
  diag_obj["shardyEdgeFailures"] = total_shardy_edge_failures();
  diag_obj["missingTensorNames"] = total_missing_tensor_names();

  llvm::json::Object root;
  root["diagnostics"] = std::move(diag_obj);
  std::string json_str;
  llvm::raw_string_ostream os(json_str);
  os << llvm::json::Value(std::move(root));
  return json_str;
}

}  // namespace adapter
}  // namespace model_explorer
