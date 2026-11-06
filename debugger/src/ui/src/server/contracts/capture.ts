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

/** GET /api/session: the evidence of one capture record. */

import type {Reason} from './metrics';
import type {RuntimeOptions} from './sessions';

/** Saved evidence returned by /api/session; configuration alone is not capture evidence. */
export interface CapturedRun extends RuntimeOptions {
  id: string;
  runtime: string;
  backend: string;
  artifact: string;
  device?: string;
  runnerBuild?: string;
  provenance?: Record<string, unknown>;
}
/** One turn in the saved capture index. */
export interface CapturedTurn {
  n: number;
  step_start: number | null;
  /** The user message of a rejected turn, which has no `conversation` entries to carry it. */
  prompt?: string;
  /** Present when the capture index rejected the turn: its counts are null and it has no tokens. */
  rejected?: CapturedTurnRejection;
  /** What the server makes of the turn's two sides; absent for a rejected turn. */
  comparison?: TurnComparison;
  /** What each stage offers, and where it is on each side, as the server reads the capture. */
  stages?: {prefill: TurnStage; decode: TurnStage};
}
/** The analyses "Add analysis" offers, by their ids in the metric catalog. */
export type StageAnalysisId = 'final_logits' | 'sampling' | 'kv' | 'intermediate_tensors';
/** Whether an analysis can be added to a stage; if not, the server's reasons why. */
export interface StageAvailability {
  available: boolean;
  reasons: Reason[];
}
/** A side's context positions of a stage: [start, end). */
export interface StageSpan {
  start: number;
  end: number;
}
/** Per-stage capture metadata and analysis availability for a turn. */
export interface TurnStage {
  /** Each side's own positions of the stage; null for a rejected turn. */
  positions: {ref: StageSpan | null; target: StageSpan | null};
  /**
   * The stage's KV snapshot: its moment (after Prefill, at the end of generation) and its
   * kv-analysis context, null when no side captured one.
   */
  kv: {moment: string; context_id: string | null};
  analyses: Record<StageAnalysisId, StageAvailability>;
}
/** One aligned row: each side's token, as an index of its `input_tokens` or `tokens`, or null. */
export interface ComparisonRow {
  ref: number | null;
  target: number | null;
  /** Whether the two tokens are the same; null when a token's text was not captured. */
  same: boolean | null;
}
/**
 * From which column on an analysis's two sides read different tokens, after the turn's
 * divergence, and why its values still compare, in the server's words.
 */
export interface AnalysisContext {
  start: number;
  reason: Reason;
}
/**
 * Which of a stage's columns an analysis has, [start, end), as the server decides them; why it
 * has no more (`omitted`), what its values cannot show (`notes`), and from which column on its
 * two sides read different tokens (`context`, null where none does), in the server's words.
 */
export interface AnalysisColumns {
  start: number;
  end: number;
  omitted: Reason[];
  notes: Reason[];
  context?: AnalysisContext | null;
}
/**
 * The context positions a stage's rows name: row k is column `start + k`, read on each side at
 * its own position (null where the row has no token of that side); the KV and activation
 * analyses chart these columns, each only those of them `columns` gives it.
 */
export interface StagePositions {
  start: number;
  pairs: Array<[number | null, number | null]>;
  columns?: Partial<Record<StageAnalysisId, AnalysisColumns>>;
}
/** One message of the chat template over a turn's input: indices of its tokens or rows. */
export interface InputMessage {
  role: 'system' | 'user' | 'model' | 'unknown';
  /** Template and special tokens before the text (the start token, the opener, a newline). */
  leading: number[];
  text: number[];
  /** Template tokens after the text, the prompt for the reply among them. */
  trailing: number[];
}
/** Whether both sides had fed the same tokens before a stage; null when nothing came before. */
export type HistoryComparison = 'same' | 'different' | 'unknown' | null;
/** The first generation step where the sides differ. */
export interface FirstDivergence {
  /** The row that holds it. */
  row: number;
  /** Each side's token of that step, as an index of its `tokens`. */
  ref: number;
  target: number;
  /** The side the row leaves empty and the step fills in; null when the row holds both. */
  filled: 'ref' | 'target' | null;
  /** The row the filled-in token is drawn in, or null. */
  partner: number | null;
}
/** What a turn's generations did relative to each other. */
export interface DecodeConclusion {
  /** `unknown`: a token's text is missing, or the rows do not prove which step differs first. */
  kind: 'same' | 'diverged' | 'unknown' | 'not_captured';
  /** The Target's first differing step, for `diverged`. */
  step: number | null;
  /** Leading rows that are the same. */
  prefix: number;
}
/** How much of one stage is the same, under one alignment. */
export interface StageComparison {
  match_rate: number | null;
  /** From the chat's first stage through this one. */
  cumulative_match_rate: number | null;
  /** Whether this stage holds the chat's first different row; null while unproven. */
  first_mismatch: boolean | null;
}
/** One alignment of a turn's generated tokens. */
export interface DecodeComparison {
  rows: ComparisonRow[];
  /** The first row that is different, or null. */
  first_mismatch: number | null;
  first_divergence: FirstDivergence | null;
  conclusion: DecodeConclusion;
  stages: {prefill: StageComparison; decode: StageComparison};
  positions: StagePositions;
}
/** A turn's two inputs, aligned by content and grouped into messages over the rows. */
export interface PrefillComparison {
  rows: ComparisonRow[];
  messages: InputMessage[];
  /** `row`: the first row that is not the same. */
  conclusion: {kind: 'same' | 'different' | 'unknown' | 'not_captured'; row: number | null};
  positions: StagePositions;
}
/** A turn's stages, and the alignments a stage's rows may be of: a Prefill's are by content. */
export type ComparisonStage = 'prefill' | 'decode';
/** The alignment modes for decode token comparisons. */
export type ComparisonAlignment = 'content' | 'steps';
/**
 * Which rows of the server's comparison an analysis request names: the server looks up the
 * tokens, steps and positions they stand for.
 */
export interface RowAddress {
  turn: number;
  stage: ComparisonStage;
  alignment: ComparisonAlignment;
}
/** Per-turn prefill and decode alignment comparisons across Reference and Target. */
export interface TurnComparison {
  history: {prefill: HistoryComparison; decode: HistoryComparison};
  prefill: PrefillComparison;
  alignments: {content: DecodeComparison; steps: DecodeComparison};
}
/** Why the capture index rejected a turn, in the server's words; the UI shows `text` as is. */
export interface CapturedTurnRejection {
  /** `turn_rejected` as the turn was added, `turn_unreadable` as the capture was read. */
  code: string;
  /** The side whose files or completion were rejected; null when an earlier turn was. */
  side: 'ref' | 'target' | null;
  /** What to tell the user; analyses of the turn answer 409 with the same text. */
  text: string;
}
/** Batch IDs are UI capture coordinates. */
export interface CapturedBatch {
  batch: number;
  turn: number;
  phase: string;
  index: number;
  step: number | null;
  runtime?: string;
  signature?: string;
  graph?: string;
  comparison_basis?: string;
}
/** Saved session capture payload returned by GET /api/session. */
export interface Session {
  model: string;
  runs: CapturedRun[];
  turns: CapturedTurn[];
  batches: CapturedBatch[];
  notice?: string;
  name?: string;
  created_at?: string;
  capture_id?: string;
  /**
   * The chat of the Session this is the capture of; its `generation` and `notice` are the
   * chat's.
   */
  chat?: number;
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
  /** Absent when the runner did not name the token's ID. */
  id?: number;
  step: number;
  /**
   * The token's global context position on its own side. A turn's `step_start` is the Reference
   * side's; the Target's tokens sit elsewhere when the two histories differ in length.
   */
  position?: number;
  batch?: number;
  source_forward_id?: number;
  phase?: 'thinking' | 'response';
  /**
   * A greedy step whose logits name no one argmax: the text to show, and why (of the token's
   * side, code `greedy_unconfirmed`).
   */
  cannot_confirm?: {text: string; reason: Reason};
  /**
   * The server's text of the step that produced the token, which Final Logits and the
   * intermediate tensors read at its column, e.g. `Decode step 20 · reads "," · produces "blue"`.
   */
  produced_text?: string;
  /**
   * The server's text of the step that wrote the token's K and V, which the KV cache shows at its
   * column, e.g. `Written at decode step 21, when the model reads "blue"`.
   */
  written_text?: string;
}
/** One turn's recorded conversation entry for one side (`ref` or `target`). */
export interface CapturedConversation {
  /** Why this side's reply ended: 'eos', 'max_output_tokens' or 'context_full'. */
  stop_reason?: string;
  /** The server's text for why this side's reply ended, e.g. 'Context full (1024 tokens)'. */
  stop_text?: string;
  turn: number;
  run: string;
  /** What this side was sent, as its capture keeps it. */
  input?: string;
  input_token_count?: number;
  /** The admitted input as the runtime tokenized it (chat template included), when the Runner reported it. */
  input_tokens?: CapturedToken[];
  /** `input_tokens` grouped into the messages of the chat template, by the server. */
  input_messages?: InputMessage[];
  /** Turn 1 only: the messages the chat opened with (its system message), which no turn's `input` or `output` holds. */
  messages?: {role: 'system' | 'user' | 'assistant'; content: string}[];
  thinking?: string;
  output?: string;
  tokens?: CapturedToken[];
}
