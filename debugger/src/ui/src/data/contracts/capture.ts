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

import type {RuntimeOptions} from './session_summary';
import type {Anchor} from './types';

/** Saved evidence returned by /api/session; configuration alone is not capture evidence. */
export interface CapturedRun extends RuntimeOptions {
  id: string;
  runtime: string;
  backend: string;
  precision: string;
  artifact: string;
  artifact_name?: string;
  device?: string;
  runnerBuild?: string;
  provenance?: Record<string, unknown>;
}
export interface CapturedTurn {
  n: number;
  prefill_tokens: number | null;
  decode_tokens: number | null;
  step_start: number | null;
  step_end: number | null;
}
/** Batch IDs are UI capture coordinates. Native forward IDs remain scoped to a Run. */
export interface CapturedBatch {
  batch: number;
  turn: number;
  phase: string;
  index: number;
  step: number | null;
  runtime?: string;
  forward_id?: number | null;
  forward_ids?: Record<string, number | null>;
  signature?: string;
  graph?: string;
  comparison_basis?: string;
}
export interface CapturedPhase {
  id: string;
  label: string;
  captured?: boolean;
}
export interface CapturedLayer {
  index: number;
  label: string;
  hidden: number | null;
}
export interface Session {
  model: string;
  runs: CapturedRun[];
  turns: CapturedTurn[];
  phases: CapturedPhase[];
  batches: CapturedBatch[];
  layers: CapturedLayer[];
  anchors: Anchor[];
  notice?: string;
  name?: string;
  created_at?: string;
  capture_id?: string;
  generation?: Record<string, Record<string, string | number | null>>;
  conversation?: CapturedConversation[];
}

/** Optional recorded text; never inferred from tensor or batch counts. */
export interface CapturedToken {
  kind?: 'text' | 'template' | 'special';
  text: string | null;
  /** Where a control token's text came from when the runtime released none (e.g. `saved tokenizer`). */
  text_basis?: string;
  /** False for a sampled stop token the runtime filtered from the text; it still has its own logits. */
  released?: boolean;
  stop?: boolean;
  id?: number;
  step: number;
  batch?: number;
  source_forward_id?: number;
  phase?: 'thinking' | 'response';
}
export interface CapturedConversation {
  stop_reason?: string;
  alignments?: {content?: {ref: number | null; target: number | null}[]};
  turn: number;
  run: string;
  input?: string;
  input_token_count?: number;
  /** The admitted input as the runtime tokenized it (chat template included), when the Runner reported it. */
  input_tokens?: CapturedToken[];
  serialized_input?: string;
  messages?: {role: 'system' | 'user' | 'assistant'; content: string}[];
  thinking?: string;
  output?: string;
  tokens?: CapturedToken[];
}
