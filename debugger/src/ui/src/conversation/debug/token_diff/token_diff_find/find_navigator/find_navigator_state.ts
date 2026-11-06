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

/** Enabled state and count label shared by every find-result toolbar. */
export interface FindNavigatorInput {
  index: number;
  total: number;
  busy?: boolean;
  /** The count is still being computed (numeric filters); navigation waits. */
  pending?: boolean;
}
export interface FindNavigatorState {
  firstDisabled: boolean;
  previousDisabled: boolean;
  nextDisabled: boolean;
  label: string;
}
export function findNavigatorState({
  index,
  total,
  busy = false,
  pending = false,
}: FindNavigatorInput): FindNavigatorState {
  const blocked = busy || pending;
  return {
    firstDisabled: blocked || !total || index === 0,
    previousDisabled: blocked || index <= 0,
    nextDisabled: blocked || !total || index === total - 1,
    label: pending ? '… / …' : `${index < 0 ? '—' : index + 1} / ${total}`,
  };
}
