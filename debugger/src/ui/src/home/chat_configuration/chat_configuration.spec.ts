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

import {ElementRef, signal} from '@angular/core';
import {TestBed} from '@angular/core/testing';
import {WorkspaceStateService} from '../../state/workspace_state_service';
import type {RuntimeJob} from '../../server/contracts/jobs';
import type {RuntimeCapability} from '../../server/contracts/runtime';
import type {ChatGeneration, SessionSummary} from '../../server/contracts/sessions';
import {SessionsApi} from '../../server/sessions_api';
import {allowed, chat, deferred, refused, session} from '../../server/testing/session_fixtures';
import {ChatConfiguration} from './chat_configuration';

const SOURCE = "the Model Debugger's own, as the model has none; saved with these settings";
// The Chat settings as the server describes them (option_catalog.py), cut to the sampling ones.
const CAPABILITY: RuntimeCapability = {
  id: 'LiteRT-LM',
  available: true,
  reason: '',
  backends: ['CPU'],
  chat_options: [
    {
      key: 'samplerType',
      label: 'Sampler',
      help: '',
      group: 'Sampling',
      type: 'choice',
      choices: ['greedy', 'topK', 'topP'],
      choice_labels: {greedy: 'Greedy', topK: 'Top-K', topP: 'Top-K + Top-P'},
      default: 'greedy',
      available: true,
    },
    {
      key: 'temperature',
      label: 'Temperature',
      help: '',
      group: 'Sampling',
      type: 'number',
      samplers: ['topK', 'topP'],
      default: 1,
      fixed: {greedy: '1 · fixed by Greedy'},
      available: true,
    },
    {
      key: 'topK',
      label: 'Top K',
      help: '',
      group: 'Sampling',
      type: 'integer',
      samplers: ['topK', 'topP'],
      default: null,
      defaults: {topP: {value: 64, source: SOURCE}},
      fixed: {greedy: '1 · fixed by Greedy'},
      available: true,
    },
    {
      key: 'topP',
      label: 'Top P',
      help: '',
      group: 'Sampling',
      type: 'number',
      samplers: ['topP'],
      default: null,
      fixed: {greedy: '1 · fixed by Greedy', topK: '1 · fixed by Top-K'},
      available: true,
    },
  ],
};

const RUNS = [
  {id: 'ref', runtime: 'LiteRT-LM', device: 'local'},
  {id: 'target', runtime: 'LiteRT-LM', device: 'local'},
];

/** The slice of WorkspaceStateService the form reads and drives, over `record` as the list has
 *  it; the chats started are kept. */
function fakeWorkspace(record: SessionSummary, capability = CAPABILITY) {
  const started: ChatGeneration[] = [];
  const workspace = {
    started,
    session: signal(record),
    jobs: {job: signal<RuntimeJob | null>(null)},
    runtime: {
      capability: () => capability,
      busy: signal(false),
      submitting: signal(false),
      error: signal(''),
      startChat: async (generation: ChatGeneration) => {
        started.push(generation);
        return true;
      },
    },
  };
  return workspace;
}

function create(
  generation: Record<string, Record<string, string | number>>,
  workspace = fakeWorkspace(
    session('s', {
      runs: RUNS,
      chats: [chat(1, {generation})],
      live_chat: 1,
      actions: {...session().actions, new_chat: allowed},
    }),
  ),
) {
  TestBed.configureTestingModule({
    providers: [
      {provide: WorkspaceStateService, useValue: workspace},
      {
        provide: SessionsApi,
        useValue: {chatOptions: () => Promise.resolve({ref: CAPABILITY, target: CAPABILITY})},
      },
      {provide: ElementRef, useValue: {nativeElement: {querySelectorAll: () => []}}},
    ],
  });
  return TestBed.runInInjectionContext(() => new ChatConfiguration());
}

describe('new chat form', () => {
  const generation = {ref: {samplerType: 'topK', topK: 40}, target: {samplerType: 'greedy'}};
  const record = (extra: Partial<SessionSummary>) =>
    session('s', {runs: RUNS, actions: {...session().actions, new_chat: allowed}, ...extra});

  it("starts from the last chat's settings, and says that starting ends it only while it is live", () => {
    const live = fakeWorkspace(
      record({chats: [chat(1, {state: 'ended'}), chat(2, {generation})], live_chat: 2}),
    );
    const form = create({}, live);
    expect(form.values['ref']).toEqual({samplerType: 'topK', topK: 40});
    expect(form.endsLive()).toBe('Starting it ends Chat 2; Chat 2 stays readable.');
    expect(form.lastEnded()).toBe('');
    // Ended since, as the list shows it now: nothing is ended by starting.
    live.session.set(
      record({
        chats: [chat(1, {state: 'ended'}), chat(2, {generation, state: 'ended', notice: 'Gone.'})],
        live_chat: null,
      }),
    );
    expect(form.endsLive()).toBe('');
    expect(form.lastEnded()).toBe('Chat 2 ended: Gone.');
    // A Session with no chat yet starts from each side's defaults.
    TestBed.resetTestingModule();
    const fresh = create({}, fakeWorkspace(record({chats: [], live_chat: null})));
    expect(fresh.values).toEqual({ref: {}, target: {}});
    expect(fresh.endsLive()).toBe('');
  });

  it('starts the chat with the settings each side may set', async () => {
    const workspace = fakeWorkspace(record({chats: [chat(1, {generation})], live_chat: 1}));
    const form = create({}, workspace);
    form.setSampler('target', 'Top-K');
    form.values['target']['topK'] = 8;
    form.values['ref']['unknownSetting'] = 3;
    await form.start();
    expect(workspace.started).toStrictEqual([
      {ref: {samplerType: 'topK', topK: 40}, target: {samplerType: 'topK', topK: 8}},
    ]);
  });

  it('is disabled while a turn or a chat start runs, and while the server refuses a chat', async () => {
    const workspace = fakeWorkspace(record({chats: [chat(1)], live_chat: 1}));
    const form = create({}, workspace);
    expect(form.action().allowed).toBe(true);
    workspace.runtime.busy.set(true);
    expect(form.action()).toStrictEqual(refused('Wait for the current task to finish.'));
    await form.start();
    expect(workspace.started).toStrictEqual([]);
    workspace.runtime.busy.set(false);
    workspace.jobs.job.set({
      id: 'j',
      session_id: 's',
      operation: 'chat',
      status: 'running',
      error: '',
      sequence: 1,
      output: {},
      reasoning: {},
      turn: 0,
      chat: 2,
      progress: 'Starting chat',
    });
    expect(form.saving()).toBe(true);
    expect(form.starting()?.progress).toBe('Starting chat');
    workspace.jobs.job.set(null);
    workspace.session.set(
      record({
        chats: [chat(1)],
        live_chat: 1,
        actions: {...session().actions, new_chat: refused('This Session has ended.')},
      }),
    );
    expect(form.action()).toStrictEqual(refused('This Session has ended.'));
    await form.start();
    expect(workspace.started).toStrictEqual([]);
  });
});

describe('chat configuration sampler', () => {
  it('names each sampler as the server labels it', () => {
    const dialog = create({ref: {samplerType: 'topP', topK: 40, topP: 0.9}, target: {}});
    expect(dialog.samplerOptions('ref')).toEqual(['Default', 'Greedy', 'Top-K', 'Top-K + Top-P']);
    expect(dialog.samplerLabel('ref')).toBe('Top-K + Top-P');
    expect(dialog.samplerLabel('target')).toBe('Default');
  });

  it('saves the sampler the chosen label names, by its id', () => {
    const dialog = create({ref: {samplerType: 'topP', topK: 40, topP: 0.9}, target: {}});
    dialog.setSampler('target', 'Top-K + Top-P');
    expect(dialog.values['target']['samplerType']).toBe('topP');
    // Top-K uses top_k as Top-K + Top-P does, and not top_p: that one goes back to the default.
    dialog.setSampler('ref', 'Top-K');
    expect(dialog.values['ref']).toEqual({samplerType: 'topK', topK: 40, topP: null});
    dialog.setSampler('ref', 'Default');
    expect(dialog.values['ref']['samplerType']).toBeNull();
  });

  it("shows the side's sampler default and where the server says it comes from", () => {
    const dialog = create({ref: {samplerType: 'topP'}, target: {}});
    expect(dialog.placeholder('ref', 'topK')).toBe('Default (64)');
    expect(dialog.defaultSource('ref', 'topK')).toBe('Default 64: ' + SOURCE);
    expect(dialog.placeholder('ref', 'temperature')).toBe('Default (1)');
    expect(dialog.defaultSource('ref', 'temperature')).toBe('');
    // Greedy, the default sampler, has no default of its own for top_k: it fixes it.
    expect(dialog.placeholder('target', 'topK')).toBe('Default');
    expect(dialog.defaultSource('target', 'topK')).toBe('');
    expect(dialog.fixedValue('target', 'topK')).toBe('1 · fixed by Greedy');
  });

  it('requires a sampling value the sampler uses and has no default for', () => {
    const dialog = create({ref: {samplerType: 'topP'}, target: {samplerType: 'topK'}});
    // Top-K + Top-P needs Top P; it takes Top K from the model or the table.
    expect(dialog.required('ref', 'topP')).toBe(true);
    expect(dialog.placeholder('ref', 'topP')).toBe('Required');
    expect(dialog.note('ref', 'topP')).toBe('Top-K + Top-P needs a Top P.');
    // Once given, it says so no more.
    dialog.values['ref']['topP'] = 0.9;
    expect(dialog.note('ref', 'topP')).toBe('');
    dialog.values['ref']['topP'] = null;
    expect(dialog.required('ref', 'topK')).toBe(false);
    expect(dialog.note('ref', 'topK')).toBe('Default 64: ' + SOURCE);
    // Top-K needs Top K, and fixes Top P.
    expect(dialog.required('target', 'topK')).toBe(true);
    expect(dialog.required('target', 'topP')).toBe(false);
    expect(dialog.required('target', 'temperature')).toBe(false);
  });
});

describe('chat configuration limits', () => {
  // The chat settings read what the server describes for each side: its longest reply (its
  // context) and the system prompt's longest length, and what it says of a build never run.
  it('chat settings read each side limits and notices from the server', async () => {
    const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
    const offer = (maximum: number, notice?: string): RuntimeCapability =>
      ({
        id: 'LiteRT-LM',
        available: true,
        reason: '',
        backends: ['CPU'],
        chat_options: [
          {key: 'systemPrompt', type: 'text', maximum: 16000, available: true, default: null},
          {
            key: 'maxOutputTokens',
            type: 'integer',
            minimum: 1,
            maximum,
            available: true,
            default: 32,
          },
        ],
        ...(notice ? {notice} : {}),
      }) as RuntimeCapability;
    const table = offer(2147483647);
    const record = session('s', {
      runs: [
        {id: 'ref', runtime: 'LiteRT-LM', device: 'local', contextLength: '2048'},
        {id: 'target', runtime: 'LiteRT-LM', device: 'local', contextLength: ''},
      ],
      chats: [chat(1, {generation: {ref: {maxOutputTokens: 300}, target: {}}})],
    });
    const asked: string[] = [];
    let answer = deferred<Record<string, RuntimeCapability>>();
    TestBed.configureTestingModule({
      providers: [
        {provide: WorkspaceStateService, useValue: fakeWorkspace(record, table)},
        {
          provide: SessionsApi,
          useValue: {
            chatOptions: (id: string) => {
              asked.push(id);
              return answer.promise;
            },
          },
        },
        {provide: ElementRef, useValue: {nativeElement: {}}},
      ],
    });
    const dialog = TestBed.runInInjectionContext(() => new ChatConfiguration());
    expect(asked).toStrictEqual(['s']);
    // Before the server answers: its table.
    expect(dialog.range('ref', 'maxOutputTokens').max).toBe(2147483647);
    answer.resolve({ref: offer(2048), target: offer(4096, 'Not run yet.')});
    await tick();
    expect(dialog.range('ref', 'maxOutputTokens').max, "the ref side's context").toBe(2048);
    expect(dialog.range('target', 'maxOutputTokens').max, "the target's default").toBe(4096);
    expect(dialog.textLimit('ref', 'systemPrompt')).toBe(16000);
    expect(dialog.notice('ref')).toBe('');
    expect(dialog.notice('target')).toBe('Not run yet.');
    expect(dialog.values['ref']['maxOutputTokens']).toBe(300);
    // A failed read keeps the table, and says so.
    answer = deferred<Record<string, RuntimeCapability>>();
    const failed = TestBed.runInInjectionContext(() => new ChatConfiguration());
    answer.reject(new Error('offline'));
    await tick();
    expect(failed.range('ref', 'maxOutputTokens').max).toBe(2147483647);
    expect(failed.error()).toMatch(/could not be read.*offline/);
  });
});
