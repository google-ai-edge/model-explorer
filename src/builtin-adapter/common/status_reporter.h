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

#ifndef MODEL_EXPLORER_BACKEND_COMMON_STATUS_REPORTER_H_
#define MODEL_EXPLORER_BACKEND_COMMON_STATUS_REPORTER_H_

#include <atomic>
#include <chrono>  // NOLINT(build/c++11)
#include <cstdint>
#include <functional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"

namespace model_explorer {

// Canonical conversion lifecycle stages.
enum class LifecycleStage : int32_t {
  kStarting = 0,
  kReadingFile = 1,
  kParsingContainerHeader = 2,
  kParsingModel = 3,
  kCompilerPasses = 4,
  kProcessingSubgraphs = 5,
  kProcessingOperations = 6,
  kBuildingNodesAndEdges = 7,
  kTransformingNamespaces = 8,
  kSubgraphValidation = 9,
  kPostprocessingSubgraph = 10,
  kSerializingJson = 11,
  kCompleted = 12,
  kFailed = 13,
};

// Returns a human-readable identifier string for the given lifecycle stage.
inline absl::string_view LifecycleStageToString(LifecycleStage stage) {
  switch (stage) {
    case LifecycleStage::kStarting:
      return "starting";
    case LifecycleStage::kReadingFile:
      return "reading_file";
    case LifecycleStage::kParsingContainerHeader:
      return "parsing_container_header";
    case LifecycleStage::kParsingModel:
      return "parsing_model";
    case LifecycleStage::kCompilerPasses:
      return "compiler_passes";
    case LifecycleStage::kProcessingSubgraphs:
      return "processing_subgraphs";
    case LifecycleStage::kProcessingOperations:
      return "processing_operations";
    case LifecycleStage::kBuildingNodesAndEdges:
      return "building_nodes_and_edges";
    case LifecycleStage::kTransformingNamespaces:
      return "transforming_namespaces";
    case LifecycleStage::kSubgraphValidation:
      return "subgraph_validation";
    case LifecycleStage::kPostprocessingSubgraph:
      return "postprocessing_subgraph";
    case LifecycleStage::kSerializingJson:
      return "serializing_json";
    case LifecycleStage::kCompleted:
      return "completed";
    case LifecycleStage::kFailed:
      return "failed";
  }
  return "unknown";
}

// Structured progress report passed to progress callbacks.
struct ConversionProgressReport {
  LifecycleStage stage = LifecycleStage::kStarting;
  int64_t current_step = 0;
  int64_t total_steps = -1;
  double progress_fraction = -1.0;
  std::string message;
  std::string subgraph_name;
  std::string current_op;
  int64_t elapsed_time_ms = 0;
  std::string payload_json;
};

// Callback type for progress reporting.
// Returns 0 to continue, non-zero (e.g. 1) to request cancellation.
using ProgressCallback =
    std::function<int(const ConversionProgressReport& report)>;

// StatusReporter provides rate-limited, cancellation-aware progress tracking
// during model conversions.
//
// Thread-safety model:
// - Single Producer: Progress reporting methods (`Report`, `ReportOp`,
//   `ReportStage`) mutate internal state and must only be invoked from the
//   active conversion thread.
// - Thread-safe Cancellation: `RequestCancellation()` and `IsCancelled()` are
//   atomic and may be invoked concurrently from any thread (e.g. server
//   thread).
//
// Rate limiting is implemented via a dual-tier strategy:
// 1. Cadence check: In inner loops, clock queries only occur every
//    `step_cadence` steps (default N=64).
// 2. Time throttle: Callbacks within the same stage are throttled to at most
//    once per `interval` (default 100ms / 10 Hz).
// 3. Stage transition bypass: Any change in `LifecycleStage` immediately
//    flushes progress, bypassing time and cadence checks.
class StatusReporter {
 public:
  static constexpr int64_t kDefaultStepCadence = 64;
  static constexpr std::chrono::milliseconds kDefaultInterval =
      std::chrono::milliseconds(100);

  explicit StatusReporter(ProgressCallback callback = nullptr,
                          std::chrono::milliseconds interval = kDefaultInterval,
                          int64_t step_cadence = kDefaultStepCadence)
      : callback_(std::move(callback)),
        interval_(interval),
        step_cadence_(step_cadence > 0 ? step_cadence : 1),
        start_time_(std::chrono::steady_clock::now()),
        last_report_time_(std::chrono::steady_clock::time_point::min()) {}

  // Disallow copy/assignment.
  StatusReporter(const StatusReporter&) = delete;
  StatusReporter& operator=(const StatusReporter&) = delete;

  // Move constructor and assignment.
  StatusReporter(StatusReporter&&) = default;
  StatusReporter& operator=(StatusReporter&&) = default;

  // Reports progress with dual-tier rate limiting.
  // Returns true to continue conversion; returns false if cancelled.
  bool Report(LifecycleStage stage, int64_t current_step, int64_t total_steps,
              absl::string_view message, absl::string_view subgraph_name = "",
              absl::string_view current_op = "",
              absl::string_view payload_json = "") {
    if (cancelled_.load(std::memory_order_relaxed)) {
      return false;
    }

    bool stage_changed = !has_reported_ || (stage != current_stage_);
    has_reported_ = true;
    current_stage_ = stage;
    current_step_ = current_step;
    total_steps_ = total_steps;

    // Step 1: Immediate bypass on stage transition or first report.
    if (stage_changed) {
      steps_since_last_check_ = 0;
      last_report_time_ = std::chrono::steady_clock::now();
      return Flush(message, subgraph_name, current_op, payload_json);
    }

    // Step 2: Cadence check in tight loops (evaluate clock only every N steps).
    if ((++steps_since_last_check_ % step_cadence_) != 0) {
      return true;
    }

    // Step 3: Time-based rate limiting (e.g. 10 Hz = 100ms).
    auto now = std::chrono::steady_clock::now();
    if (now - last_report_time_ >= interval_) {
      last_report_time_ = now;
      return Flush(message, subgraph_name, current_op, payload_json);
    }
    return true;
  }

  // Reports a stage transition immediately without step counters.
  bool ReportStage(LifecycleStage stage, absl::string_view message,
                   absl::string_view subgraph_name = "",
                   absl::string_view payload_json = "") {
    return Report(stage, /*current_step=*/0, /*total_steps=*/-1, message,
                  subgraph_name, /*current_op=*/"", payload_json);
  }

  // Reports progress with dual-tier rate limiting and returns an absl::Status.
  // Returns absl::OkStatus() if conversion should continue; returns
  // absl::CancelledError if cancellation has been requested.
  absl::Status ReportStatus(LifecycleStage stage, int64_t current_step,
                            int64_t total_steps, absl::string_view message,
                            absl::string_view subgraph_name = "",
                            absl::string_view current_op = "",
                            absl::string_view payload_json = "") {
    if (!Report(stage, current_step, total_steps, message, subgraph_name,
                current_op, payload_json)) {
      return absl::CancelledError("Conversion cancelled by user");
    }
    return absl::OkStatus();
  }

  // Reports a stage transition immediately and returns an absl::Status.
  absl::Status ReportStageStatus(LifecycleStage stage,
                                 absl::string_view message,
                                 absl::string_view subgraph_name = "",
                                 absl::string_view payload_json = "") {
    if (!ReportStage(stage, message, subgraph_name, payload_json)) {
      return absl::CancelledError("Conversion cancelled by user");
    }
    return absl::OkStatus();
  }

  // Null-safe static helper that reports progress if reporter is non-null.
  // Returns absl::OkStatus() if reporter is null or if conversion should
  // continue; returns absl::CancelledError if cancellation was requested.
  static absl::Status Report(StatusReporter* reporter, LifecycleStage stage,
                             int64_t current_step, int64_t total_steps,
                             absl::string_view message,
                             absl::string_view subgraph_name = "",
                             absl::string_view current_op = "",
                             absl::string_view payload_json = "") {
    if (reporter == nullptr) {
      return absl::OkStatus();
    }
    return reporter->ReportStatus(stage, current_step, total_steps, message,
                                  subgraph_name, current_op, payload_json);
  }

  // Null-safe static helper that reports a stage transition if reporter is
  // non-null. Returns absl::OkStatus() if reporter is null or if conversion
  // should continue; returns absl::CancelledError if cancellation was
  // requested.
  static absl::Status ReportStage(StatusReporter* reporter,
                                  LifecycleStage stage,
                                  absl::string_view message,
                                  absl::string_view subgraph_name = "",
                                  absl::string_view payload_json = "") {
    if (reporter == nullptr) {
      return absl::OkStatus();
    }
    return reporter->ReportStageStatus(stage, message, subgraph_name,
                                       payload_json);
  }

  // Forces an immediate progress report bypassing step cadence and time
  // interval checks.
  bool ReportImmediate(LifecycleStage stage, int64_t current_step,
                       int64_t total_steps, absl::string_view message,
                       absl::string_view subgraph_name = "",
                       absl::string_view current_op = "",
                       absl::string_view payload_json = "") {
    if (cancelled_.load(std::memory_order_relaxed)) {
      return false;
    }
    has_reported_ = true;
    current_stage_ = stage;
    current_step_ = current_step;
    total_steps_ = total_steps;
    steps_since_last_check_ = 0;
    last_report_time_ = std::chrono::steady_clock::now();
    return Flush(message, subgraph_name, current_op, payload_json);
  }

  // Requests cooperative cancellation. Subsequent calls to Report() return
  // false.
  void RequestCancellation() {
    cancelled_.store(true, std::memory_order_relaxed);
  }

  // Returns true if cancellation has been requested.
  bool IsCancelled() const {
    return cancelled_.load(std::memory_order_relaxed);
  }

  // Elapsed time in milliseconds since StatusReporter construction.
  int64_t elapsed_time_ms() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now -
                                                                 start_time_)
        .count();
  }

  LifecycleStage current_stage() const { return current_stage_; }
  int64_t current_step() const { return current_step_; }
  int64_t total_steps() const { return total_steps_; }

 private:
  bool Flush(absl::string_view message, absl::string_view subgraph_name,
             absl::string_view current_op, absl::string_view payload_json) {
    if (!callback_) {
      return true;
    }

    double fraction = -1.0;
    if (total_steps_ > 0 && current_step_ >= 0) {
      fraction = static_cast<double>(current_step_) / total_steps_;
      if (fraction > 1.0) fraction = 1.0;
      if (fraction < 0.0) fraction = 0.0;
    }

    ConversionProgressReport report{
        /*stage=*/current_stage_,
        /*current_step=*/current_step_,
        /*total_steps=*/total_steps_,
        /*progress_fraction=*/fraction,
        /*message=*/std::string(message),
        /*subgraph_name=*/std::string(subgraph_name),
        /*current_op=*/std::string(current_op),
        /*elapsed_time_ms=*/elapsed_time_ms(),
        /*payload_json=*/std::string(payload_json),
    };

    int ret = callback_(report);
    if (ret != 0) {
      cancelled_.store(true, std::memory_order_relaxed);
      return false;
    }
    return true;
  }

  ProgressCallback callback_;
  std::chrono::milliseconds interval_;
  int64_t step_cadence_;
  std::atomic<bool> cancelled_{false};
  bool has_reported_ = false;
  LifecycleStage current_stage_ = LifecycleStage::kStarting;
  int64_t current_step_ = 0;
  int64_t total_steps_ = -1;
  int64_t steps_since_last_check_ = 0;
  std::chrono::steady_clock::time_point start_time_;
  std::chrono::steady_clock::time_point last_report_time_;
};

}  // namespace model_explorer

#endif  // MODEL_EXPLORER_BACKEND_COMMON_STATUS_REPORTER_H_
