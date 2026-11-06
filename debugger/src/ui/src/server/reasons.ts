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

/** What the UI does with the server's reasons: it decides on their codes, and shows their text. */

import type {Reason} from './contracts/metrics';

/** The reason codes the UI decides on. Each is in the server's catalog (GET /api/metrics), which
 *  `metric_catalog.spec.ts` checks against the fixture the server writes. */
export const REASON_CODES = {
  contextDifferent: 'context_different',
  contextUnknown: 'context_unknown',
  vocabularyMismatch: 'vocabulary_mismatch',
  divergenceInfinite: 'divergence_infinite',
  tokenNotCaptured: 'token_not_captured',
  logitsNotCaptured: 'logits_not_captured',
  activationsNotCaptured: 'activations_not_captured',
  logitsNotPaired: 'logits_not_paired',
  forwardsNotPaired: 'forwards_not_paired',
  batchSelectionRequired: 'batch_selection_required',
} as const;

/** Codes of evidence a capture does not hold, rather than of evidence that is not comparable. */
const NOT_CAPTURED: ReadonlySet<string> = new Set([
  REASON_CODES.tokenNotCaptured,
  REASON_CODES.logitsNotCaptured,
  REASON_CODES.activationsNotCaptured,
  REASON_CODES.logitsNotPaired,
  REASON_CODES.forwardsNotPaired,
]);

/** Whether a missing value is missing from the capture: no reasons, or one that says so. */
export function notCaptured(reasons: readonly Reason[] | null | undefined): boolean {
  return !reasons?.length || reasons.some((reason) => NOT_CAPTURED.has(reason.code));
}

/** The server's text of its reasons as one line, a side's reason after its side: '' for none. */
export function reasonText(reasons: readonly Reason[] | null | undefined): string {
  return (reasons ?? [])
    .map((reason) => (reason.side ? `${reason.side}: ${reason.text}` : reason.text))
    .join('; ');
}

/** The server's text of one side's own reasons, without the side: '' for none. */
export function sideReasonText(reasons: readonly Reason[] | null | undefined): string {
  return (reasons ?? []).map((reason) => reason.text).join('; ');
}
