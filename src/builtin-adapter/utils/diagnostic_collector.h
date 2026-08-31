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

#ifndef MODEL_EXPLORER_BACKEND_UTILS_DIAGNOSTIC_COLLECTOR_H_
#define MODEL_EXPLORER_BACKEND_UTILS_DIAGNOSTIC_COLLECTOR_H_

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"

namespace model_explorer {
namespace adapter {

// Collects, deduplicates, and aggregates non-fatal conversion diagnostics,
// preventing log spam in high-frequency conversion loops.
class DiagnosticCollector {
 public:
  static constexpr size_t kMaxSamples = 3;

  DiagnosticCollector() = default;

  // Records a missing op def for tensor port tagging.
  void RecordMissingOpDef(absl::string_view op_label);

  // Records a quantization scale/zero_point vector size mismatch.
  void RecordQuantizationMismatch(absl::string_view tensor_name,
                                  size_t scale_size, size_t zero_point_size);

  // Records an incomplete edge detected during subgraph validation.
  void RecordIncompleteEdge(int tensor_index, absl::string_view details);

  // Records an op option / subgraph index extraction error.
  void RecordOptionError(absl::string_view op_label, absl::string_view error);

  // Records Shardy propagation edge lookup failure.
  void RecordShardyEdgeFailure(absl::string_view reason);

  // Records missing Location / debug info for tensor names in MLIR conversion.
  void RecordMissingTensorName(absl::string_view op_name);

  // Returns true if any diagnostic issues were recorded.
  bool HasDiagnostics() const;

  // Emits a single consolidated summary log at conversion conclusion.
  void EmitSummary(absl::string_view context_name) const;

  // Accessors for inspection and unit testing.
  int total_missing_op_def_nodes() const { return missing_op_defs_.total; }
  const absl::flat_hash_map<std::string, int>& missing_op_defs() const {
    return missing_op_defs_.breakdown;
  }
  int quantization_mismatches() const { return quant_mismatches_.total; }
  const std::vector<std::string>& sample_quant_mismatches() const {
    return quant_mismatches_.samples;
  }
  int incomplete_edges_count() const { return incomplete_edges_.total; }
  const std::vector<std::string>& sample_incomplete_edges() const {
    return incomplete_edges_.samples;
  }
  int total_option_errors() const { return option_errors_.total; }
  const absl::flat_hash_map<std::string, int>& option_errors() const {
    return option_errors_.breakdown;
  }
  const std::vector<std::string>& sample_option_errors() const {
    return option_errors_.samples;
  }
  int total_shardy_edge_failures() const { return shardy_edge_failures_.total; }
  const absl::flat_hash_map<std::string, int>& shardy_edge_failures() const {
    return shardy_edge_failures_.breakdown;
  }
  int total_missing_tensor_names() const { return missing_tensor_names_.total; }
  const absl::flat_hash_map<std::string, int>& missing_tensor_names() const {
    return missing_tensor_names_.breakdown;
  }

 private:
  struct BreakdownCategory {
    int total = 0;
    absl::flat_hash_map<std::string, int> breakdown;
    bool empty() const { return total == 0; }
    void Record(absl::string_view key) {
      total++;
      breakdown[key]++;
    }
  };

  struct SampledCategory {
    int total = 0;
    std::vector<std::string> samples;
    bool empty() const { return total == 0; }
    bool NeedsSample() const { return samples.size() < kMaxSamples; }
    void RecordSample(std::string sample) {
      total++;
      samples.push_back(std::move(sample));
    }
  };

  struct BreakdownAndSampledCategory {
    int total = 0;
    absl::flat_hash_map<std::string, int> breakdown;
    std::vector<std::string> samples;
    bool empty() const { return total == 0; }
    bool NeedsSample() const { return samples.size() < kMaxSamples; }
    void Record(absl::string_view key, std::string sample = "") {
      total++;
      breakdown[key]++;
      if (!sample.empty() && NeedsSample()) {
        samples.push_back(std::move(sample));
      }
    }
  };

  BreakdownCategory missing_op_defs_;
  SampledCategory quant_mismatches_;
  SampledCategory incomplete_edges_;
  BreakdownAndSampledCategory option_errors_;
  BreakdownCategory shardy_edge_failures_;
  BreakdownCategory missing_tensor_names_;
};

}  // namespace adapter
}  // namespace model_explorer

#endif  // MODEL_EXPLORER_BACKEND_UTILS_DIAGNOSTIC_COLLECTOR_H_
