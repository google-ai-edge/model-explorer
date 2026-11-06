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

import type {RuntimeModel} from './runtime';

/** GET /api/jobs/{id}, POST /api/jobs/{id}/cancel and the GET /api/jobs/{id}/events stream. */

/** Mirrors model_debugger_contracts.jobs.TERMINAL_STATUSES on the server. */
export const TERMINAL_STATUSES = ['completed', 'failed', 'cancelled'] as const;
export type TerminalStatus = (typeof TERMINAL_STATUSES)[number];
export function isTerminal(status: string): status is TerminalStatus {
  return (TERMINAL_STATUSES as readonly string[]).includes(status);
}

export interface RuntimeJob {
  /** What a generate's turn asks; '' for other jobs. */
  prompt?: string;
  id: string;
  session_id: string;
  /** 'chat' starts a new chat of a Session; 'generate' sends a turn to its live chat. */
  operation: 'initialize' | 'chat' | 'generate' | 'download';
  status: string;
  error: string;
  sequence: number;
  /** Each side's reply so far, the 'content' deltas joined. */
  output: Record<string, string>;
  /** Each side's thinking so far, the 'reasoning' deltas joined. */
  reasoning: Record<string, string>;
  turn: number;
  /** The chat a 'generate' sends its turn to, or the number the chat a 'chat' starts gets. */
  chat?: number | null;
  /** The latest progress line. A generate has one only once its reply ended, while both sides'
   *  captures arrive: 'Retrieving the capture', then with how much arrived, e.g.
   *  'Retrieving the capture · 1.2 GB so far'. */
  progress?: string;
  /** What the job's end means for its turn, shown with its output; '' for none. */
  note?: string;
  /** The label of the job's status, such as 'Generation stopped'; '' for none. */
  status_text?: string;
  /** The model a completed download brought to the server. */
  model?: RuntimeModel;
  /** Of a generate whose turn neither side had room for in its context, which then did not
   *  happen, each side's counts: the tokens the turn needs and those its context has left. The
   *  chat goes on; `error` says what to do. */
  does_not_fit?: Record<string, {tokens: number; left: number}>;
  /** Of a failed generate, the sides that failed, when the server knows them: only those show
   *  its status text. */
  failed_sides?: string[];
}
/** One SSE frame from GET /api/jobs/{id}/events (jobs.py _event/_emit). */
export interface RuntimeJobEvent {
  sequence: number;
  jobId: string;
  sessionId: string;
  turnId?: number;
  type: string;
  status?: string;
  error?: string;
  note?: string;
  status_text?: string;
  failed_sides?: string[];
  error_code?: string | null;
  message?: string;
  runId?: string;
  /** A 'delta''s channel: the reply ('content') or the model's thinking ('reasoning'). */
  channel?: 'content' | 'reasoning';
  text?: string;
  output?: string;
}
