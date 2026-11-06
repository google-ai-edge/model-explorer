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

import type {OptionDescription, RuntimeCapability} from '../contracts/runtime';
import type {ChatSummary, SessionListing, SessionSummary} from '../contracts/sessions';

/** A promise handle whose resolution and rejection are controlled by the test. */
export interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (error: unknown) => void;
}

/** A promise the test resolves or rejects itself. */
export function deferred<T = unknown>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((yes, no) => ([resolve, reject] = [yes, no]));
  return {promise, resolve, reject};
}

/** Lets the promises queued so far settle, and the work they queue in turn, `rounds` deep. */
export async function flushMicrotasks(rounds = 8): Promise<void> {
  for (let i = 0; i < rounds; i++) {
    await Promise.resolve();
  }
}

/** Permissions as the server words them. */
export const allowed = {allowed: true, reason: ''};
/** Returns a refused action state with `reason`. */
export function refused(reason: string): {allowed: boolean; reason: string} {
  return {allowed: false, reason};
}

/**
 * A Session of the listing as the server sends it: one just created, whose Model Server is
 * starting, `extra` laid over it. Fields a case does not name are the server's defaults.
 */
export function session(id = 'a', extra: object = {}): SessionSummary {
  return {
    id,
    name: 'Original',
    model: 'test',
    created_at: null,
    status_text: 'Starting',
    model_server: {
      state: 'starting',
      text: 'Starting',
      error: '',
      button: {label: 'Cancel', ...allowed},
    },
    actions: {
      enter: refused('The Model Server is starting.'),
      delete: refused('Turn the Model Server off first.'),
      new_chat: refused('The Model Server is starting.'),
      export: refused('This Session has no turn to export.'),
    },
    has_capture: false,
    chats: [],
    live_chat: null,
    runs: [
      {id: 'ref', runtime: 'LiteRT-LM', source: 'path', artifact: 'model.litertlm'},
      {id: 'target', runtime: 'LiteRT-LM', source: 'path', artifact: 'model.litertlm'},
    ],
    notice: '',
    ...extra,
  } as SessionSummary;
}

/**
 * Chat `n` of a Session as the server lists it: live, with no turn yet, `extra` laid over it. An
 * ended chat (`state: 'ended'`) takes no turn; one that kept a turn (`has_capture`) is read in
 * Debug too.
 */
export function chat(n = 1, extra: Partial<ChatSummary> = {}): ChatSummary {
  const ended = extra.state === 'ended';
  return {
    n,
    title: 'Chat ' + n,
    summary: 'Greedy · no system prompt',
    generation: {ref: {samplerType: 'greedy'}, target: {samplerType: 'greedy'}},
    state: 'live',
    notice: ended ? 'A new chat was started.' : '',
    has_capture: false,
    created_at: '2026-10-27T00:00:00Z',
    ...extra,
    actions: {
      send: ended ? refused('This chat has ended.') : allowed,
      debug: extra.has_capture ? allowed : refused('This chat has no turn to inspect.'),
      ...extra.actions,
    },
  };
}

/** GET /api/sessions with these Sessions; a new one starts from the first's configuration. */
export function listing(sessions = [session()]): SessionListing {
  return {sessions, configuration_template: sessions[0], unavailable_reason: ''};
}

/** An option as the server describes it, `extra` laid over a settable text option. */
function option(key: string, label: string, extra: Partial<OptionDescription> = {}) {
  return {
    key,
    label,
    help: '',
    group: 'Runtime',
    type: 'text',
    default: null,
    available: true,
    ...extra,
  } as OptionDescription;
}

/**
 * PyTorch's RuntimeCapability as GET /api/runtime/capabilities describes it: its own Session
 * options, LiteRT-LM's Chat settings with those it does not take unavailable, and its tensor
 * capture: every layer's anchors, by its own two options.
 */
export function pytorchCapability(): RuntimeCapability {
  const unavailable = {available: false, reason: 'PyTorch does not take this setting.'};
  return {
    id: 'PyTorch',
    available: true,
    reason: '',
    backends: ['CPU', 'GPU'],
    session_options: [
      option('backend', 'Text Backend', {type: 'choice', choices: ['CPU', 'GPU'], default: 'CPU'}),
      option('cpuThreads', 'Text CPU threads', {
        type: 'integer',
        minimum: 1,
        requires: {key: 'backend', value: 'CPU'},
      }),
      option('dtype', 'Model dtype', {
        type: 'choice',
        choices: ['', 'float32', 'float16', 'bfloat16'],
      }),
      option('contextLength', 'Context length', {type: 'integer', minimum: 1, default: 4096}),
    ],
    chat_options: [
      option('systemPrompt', 'System prompt', {group: 'Prompt'}),
      option('thinking', 'Thinking', {group: 'Prompt', type: 'switch', ...unavailable}),
    ],
    tap: {
      available: true,
      reason: '',
      kind: 'anchors',
      text: "PyTorch captures every layer's four anchors.",
      options: [
        option('observe', 'Capture the anchors', {
          group: 'Tensor capture',
          type: 'choice',
          choices: ['', 'anchors'],
        }),
        option('observeMemoryGiB', 'Snapshot budget (GiB)', {
          group: 'Tensor capture',
          type: 'integer',
          minimum: 1,
          maximum: 1024,
          requires: {key: 'observe', value: 'anchors'},
        }),
      ],
    },
  };
}
