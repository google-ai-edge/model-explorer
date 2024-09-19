// Copyright 2024 The AI Edge Model Explorer Authors.
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

#include "tools/namespace_heuristics.h"

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace tooling {
namespace visualization_client {
namespace {

// Gets the edit distance between two strings with the optimized memory usage.
int EditDistance(absl::string_view shorter_str, absl::string_view longer_str) {
  // Ensure that 'shorter_str' is indeed the shorter of the two strings
  // This helps optimize space usage in the DP table
  const int short_len = shorter_str.size();
  const int long_len = longer_str.size();
  if (short_len > long_len) {
    return EditDistance(longer_str, shorter_str);
  }

  // 'prev_diag' stores the value from the previous diagonal in the DP table.
  int prev_diag;

  // 'curr_row' represents the current row in the DP table. Initialize it with
  // increasing values from 0 to 'short_len', representing the edit distance
  // when the longer string is empty.
  std::vector<int> curr_row(short_len + 1, 0);
  for (int j = 0; j <= short_len; j++) {
    curr_row[j] = j;
  }

  for (int i = 1; i <= long_len; i++) {
    prev_diag = curr_row[0];
    // The first element in each row represents the edit distance when the
    // shorter string is empty.
    curr_row[0] = i;
    for (int j = 1; j <= short_len; j++) {
      int temp = curr_row[j];
      // If the characters match, the edit distance is the same as the value on
      // the previous diagonal.
      if (longer_str[i - 1] == shorter_str[j - 1]) {
        curr_row[j] = prev_diag;
      } else {
        // If the characters don't match, the edit distance is 1 plus the
        // minimum of:
        // 1. Insertion: 'curr_row[j]' (current cell)
        // 2. Deletion: 'prev_diag' (top-left diagonal)
        // 3. Substitution: 'curr_row[j - 1]' (left cell)
        curr_row[j] = 1 + std::min({curr_row[j - 1], prev_diag, curr_row[j]});
      }
      prev_diag = temp;
    }
  }
  // The final edit distance is stored in the last element of 'curr_row'.
  return curr_row[short_len];
}

// Preprocesses the candidate name by obtaining the last chunk of the substring
// separated by '/', removing the non-alphabetic characters and converting to
// lower case.
std::string PreprocessCandidateName(absl::string_view name) {
  const int start_pos = name.find_last_of('/');
  std::string last_substr;
  if (start_pos != std::string::npos) {
    last_substr = name.substr(start_pos + 1, name.size());
  } else {
    last_substr = name;
  }
  // Removes the non-alphabetic characters and converts to lower case.
  last_substr.erase(
      std::remove_if(last_substr.begin(), last_substr.end(),
                     [](unsigned char c) { return !absl::ascii_isalpha(c); }),
      last_substr.end());
  last_substr = absl::AsciiStrToLower(last_substr);
  return last_substr;
}

}  // namespace

std::string TfliteNodeNamespaceHeuristic(
    absl::string_view node_label,
    absl::Span<const std::string> candidate_names) {
  if (candidate_names.empty()) return "";
  if (candidate_names.size() == 1) {
    return candidate_names[0];
  }

  // Removes any underscores in `node_label`.
  const std::string node_label_substr =
      absl::StrReplaceAll(node_label, {{"_", ""}});

  // Default the name to the first candidate name.
  std::string result_name = candidate_names[0];
  int min_distance = std::numeric_limits<int>::max();
  // Sets the max distance threshold to be three times the length of the node
  // label substring. If the distance is larger than the threshold, it's
  // considered as irrelevant.
  const int max_distance_threshold = 3 * node_label_substr.length();
  // Iterates backwards is critical in finding a better match.
  for (auto name_it = std::rbegin(candidate_names);
       name_it != std::rend(candidate_names); ++name_it) {
    const std::string last_substr = PreprocessCandidateName(*name_it);
    // Skips the empty string to avoid false matching.
    if (last_substr.empty()) {
      continue;
    }
    int cur_distance = EditDistance(node_label_substr, last_substr);
    if (cur_distance > max_distance_threshold) {
      continue;
    }
    if (cur_distance < min_distance) {
      min_distance = cur_distance;
      result_name = *name_it;
    }
  }
  return result_name;
}

}  // namespace visualization_client
}  // namespace tooling
