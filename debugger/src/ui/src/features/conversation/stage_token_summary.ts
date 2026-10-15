/**
 * @license
 * Copyright 2026 The AI Edge Model Explorer Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

import type {TokenPair} from './token_alignment';

export interface StageTokenInput {
  turn: number;
  stage: 'prefill' | 'decode';
  pairs: readonly TokenPair[];
  complete: boolean;
}

/** Chronological, position-weighted summaries. Unknown positions never count as matches. */
export function summarizeStageTokens(stages: readonly StageTokenInput[]) {
  let total = 0;
  let matched = 0;
  let complete = true;
  let diverged = false;
  return stages.map((stage) => {
    const historyMatch = diverged ? false : complete && total > 0 ? true : null;
    const count = stage.pairs.length;
    const stageMatched = stage.pairs.filter((pair) => pair.match === true).length;
    const stageComplete = stage.complete && stage.pairs.every((pair) => pair.match !== null);
    const mismatch = stage.pairs.some((pair) => pair.match === false);
    const firstDivergence = mismatch && !diverged;
    total += count;
    matched += stageMatched;
    complete &&= stageComplete;
    diverged ||= mismatch;
    return {
      turn: stage.turn,
      stage: stage.stage,
      historyMatch,
      matchRate: stageComplete && count > 0 ? stageMatched / count : null,
      cumulativeMatchRate: complete && total > 0 ? matched / total : null,
      // The first observed mismatch; missing earlier captures cannot prove where an unseen one was.
      firstDivergence: firstDivergence ? true : stageComplete || diverged ? false : null,
    };
  });
}
