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

import type {CapturedBatch, CapturedToken} from '../../data/contracts/capture';

/** Native invocation IDs belong to one Run; paired batch IDs belong to the UI. */
export function batchForwardId(
  batch: CapturedBatch,
  run: string,
): number | null | undefined {
  return batch.forward_ids ? batch.forward_ids[run] : batch.forward_id;
}
export function tokenMatchesBatch(
  token: CapturedToken,
  batch: CapturedBatch,
  run: string,
): boolean {
  return (
    token.batch === batch.batch &&
    (token.source_forward_id == null ||
      batchForwardId(batch, run) === token.source_forward_id)
  );
}
