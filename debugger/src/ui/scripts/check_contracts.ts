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

import type {
  CapturedBatch,
  CapturedPhase,
  CapturedTurn,
  Session,
} from '../src/data/contracts/capture';
import type {TensorRecord} from '../src/data/contracts/types';
import preview from '../src/fixtures/report-preview.json';

// The fixture conforms to the production contract. It never defines the contract.
preview satisfies Session;

// Actual capture_importer.py states that cannot be represented by the numeric demo.
({
  n: 1,
  prefill_tokens: null,
  decode_tokens: null,
  step_start: null,
  step_end: null,
}) satisfies CapturedTurn;
({
  batch: 0,
  turn: 1,
  phase: 'unknown',
  index: 1,
  step: null,
  forward_id: 3,
}) satisfies CapturedBatch;
({
  batch: 1,
  turn: 1,
  phase: 'prefill',
  index: 1,
  step: 0,
  forward_ids: {ref: 2, target: 3},
}) satisfies CapturedBatch;
({id: 'unknown', label: 'Unknown'}) satisfies CapturedPhase;
({
  id: 'captured-output',
  run: 'ref',
  graph: 'forward-3',
  node: 'layer',
  output: '0',
  layer: 0,
  batch: 0,
  sample: null,
  shape: [1, 2],
  dtype: 'float32',
}) satisfies TensorRecord;
