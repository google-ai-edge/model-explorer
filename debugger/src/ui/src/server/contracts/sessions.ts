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

import type {HuggingFaceSource} from './runtime';

/** GET /api/sessions, the POST /api/sessions/{operation} management requests (a create starts
 * the Session's Model Server too, or creates nothing), and the POST
 * /api/sessions/{id}/{close,chats,turns} lifecycle requests (chats and turns answer a RuntimeJob,
 * see jobs.ts). */

/** Requested run options; these fields do not prove accelerator residency. */
/** Session-level options as saved: text values, '' for the runtime default (session_rules.py RUN_FIELDS). */
export interface RuntimeOptions {
  backend?: string;
  /** PyTorch's dtype of the model's floating-point weights; '' for the model's own. */
  dtype?: string;
  cpuThreads?: string;
  audioBackend?: string;
  audioCpuThreads?: string;
  visionBackend?: string;
  activationDataType?: string;
  prefillChunkSize?: string;
  contextLength?: string;
  enableYnnpack?: string;
  speculativeDecoding?: string;
  parallelFileSectionLoading?: string;
  singleThreadedExecution?: string;
  useRingbuffersLocalAttention?: string;
  gpuDecodeStepsPerSync?: string;
  gpuWaitForWeightUploads?: string;
  gpuMetalResidencySet?: string;
  loraRank?: string;
  audioLoraRank?: string;
  /** What a PyTorch run captures each turn: 'anchors' (every layer's four anchors) or ''. */
  observe?: string;
  /** A PyTorch run's snapshot budget for one turn, in GiB; '' for its runner's default. */
  observeMemoryGiB?: string;
}
export type SessionExecutionPhase = 'inactive' | 'starting' | 'active' | 'ending' | 'ended';
/** Whether the user may do something now and, if not, why: the very refusal its request would
 *  get, word for word (session_listing.py). */
export interface Permission {
  allowed: boolean;
  reason: string;
}
/** What the user may do with a record now (session_listing.py `_actions`). */
export interface SessionActions {
  /** Open it (GET /api/session): allowed while its Model Server is on, even with no chat yet, or
   *  once any chat kept a turn; refused for a Session that ended with no turn, and while its
   *  export is being sent. */
  enter: Permission;
  delete: Permission;
  /** Start a new chat (POST /api/sessions/{id}/chats): while its Model Server is on, no turn or
   *  chat start runs, and both runners are connected. It ends the live chat, if any. */
  new_chat: Permission;
  /** Save it as one file (GET /api/sessions/{id}/export): once it has ended, and not while its
   *  export is being sent. */
  export: Permission;
}
/** What the user may do with one chat now (session_listing.py). */
export interface ChatActions {
  /** Send a turn to it (POST /api/sessions/{id}/turns): only to the live chat, with no job running
   *  and both runners connected; an ended chat's reason says it has ended. */
  send: Permission;
  /** Read it in Debug mode: once it kept a turn and its capture reads. */
  debug: Permission;
}
/** The settings each side of a chat runs with, by side ('ref', 'target'). */
export type ChatGeneration = Record<string, Record<string, string | number | null>>;
/** One chat of a Session. A Session holds several over its life, one live at a time: starting a
 *  new chat ends the live one, and an ended chat stays readable, in Chat and in Debug. */
export interface ChatSummary {
  /** Its number in the Session, from 1. */
  n: number;
  /** Its name, e.g. 'Chat 2'. */
  title: string;
  /** Its settings in words, e.g. 'Greedy · no system prompt', or 'Reference: … / Target: …' when
   *  the sides differ. */
  summary: string;
  /** What each side runs with. */
  generation: ChatGeneration;
  state: 'live' | 'ended';
  /** Why it ended, e.g. 'A new chat was started.'; '' while it is live. */
  notice: string;
  /** Whether it kept a turn. */
  has_capture: boolean;
  created_at: string;
  actions: ChatActions;
}
/** The one execution question the UI answers per Session: is its Model Server on? A Session
 *  starts once, as it is created: one no longer starting or on has 'ended', and never starts
 *  again. */
export type ModelServerState = 'starting' | 'on' | 'ended';
/** A Session's Model Server switch as the server shows it. */
export interface ModelServerView {
  state: ModelServerState;
  /** `state` in words: 'On', 'Starting', or, once ended, 'Ended' or 'Ended · no turn'. */
  text: string;
  /** Why a Session that ended with no turn ended (its start failed, say); '' otherwise. */
  error: string;
  /** What clicking the switch does while it starts or is on, a close (POST
   *  /api/sessions/{id}/close): 'Cancel' or 'Turn off'; null once it has ended. */
  button: (Permission & {label: string}) | null;
}
export interface SessionRunner {
  role: string;
  deviceId: string;
  deviceName?: string;
  buildId?: string;
  phase: string;
  /** `phase` in words, e.g. 'Loading model'. */
  phaseText: string;
  /** The side's latest progress line in the running job, e.g. 'Uploading model · 45%'; '' if none. */
  progress: string;
  connected: boolean | null;
}
export interface SessionExecution {
  phase: SessionExecutionPhase;
  runners: SessionRunner[];
  activeJobId?: string | null;
  activeOperation?: string | null;
}
export interface SessionSummary {
  id: string;
  name: string;
  created_at: string | null;
  model: string;
  /** The record's status in words: 'Starting', 'Active', 'Ending', 'Ended' or
   *  'Ended · no turn'. */
  status_text: string;
  /** Whether its export is being sent now: until then it is not opened, deleted or exported. */
  exporting?: boolean;
  model_server: ModelServerView;
  actions: SessionActions;
  execution?: SessionExecution;
  /** Whether each LiteRT-LM side captures every decoder layer's anchors of its model, by side,
   *  the default; `tap_points` are the points chosen beside them. A PyTorch side captures them by
   *  its own `observe` option. */
  tap_anchors?: Record<string, boolean>;
  tap_profile?: string;
  tap_points?: Record<string, string[]>;
  /** The operators inside the layers captured beside the anchors, one list for both sides, in
   *  every layer that has each (the server's GET /api/models/ops ids, e.g. 'attn/q_proj'). */
  tap_ops?: string[];
  /** Its preparation, as the server says: the copies of its models that observe the tensors
   *  chosen are being written ('preparing'), are ('prepared'), are still due ('needed'), or it
   *  observes none ('none'); `text` says so. */
  tap_preparation?: {
    state: 'none' | 'needed' | 'preparing' | 'prepared';
    text: string;
  };
  /** The last job the server ran for this record, which the page attached to it follows. */
  job_id?: string;
  /** Whether any of its chats kept a turn. */
  has_capture: boolean;
  /** Its chats, oldest first. */
  chats: ChatSummary[];
  /** The number of its live chat, which takes turns; null when none is. */
  live_chat: number | null;
  /** Where a Session imported from another computer's export came from: its id there, and when
   *  it was exported and imported. It has ended: Debug reads it; it never starts. */
  imported?: {session_id: string; exported_at: string | null; imported_at: string};
  runs: (RuntimeOptions & {
    id: string;
    runtime?: string;
    backend?: string;
    artifact?: string;
    source?: string;
    device?: string;
    /** The runner version this side chose, on a device that runs the one chosen: a whole
     *  LiteRT-LM commit, or 'dev' for the development build. */
    runnerBuild?: string;
    repository?: string;
    revision?: string;
    sourceUrl?: string;
    /** Where the model came from, when `source` is 'huggingface': the server's, never sent. */
    huggingface?: HuggingFaceSource;
  })[];
  /** Why the Session itself ended (its start failed, or 'Runner closed: …'); '' otherwise. A
   *  chat's own end is its `notice`. */
  notice: string;
  /** What deleting it deletes beside it, in the server's words: the models no other Session
   *  uses, and the room that frees; '' when none. */
  delete_note?: string;
}
export interface SessionListing {
  sessions: SessionSummary[];
  /** What a new Session's configuration starts from. */
  configuration_template: SessionConfig;
  unavailable_reason: string;
}

/** What the editor posts: the server names the two models itself (the summary's `model`). */
export type SessionConfig = Pick<
  SessionSummary,
  'name' | 'runs' | 'tap_anchors' | 'tap_profile' | 'tap_points' | 'tap_ops'
>;
