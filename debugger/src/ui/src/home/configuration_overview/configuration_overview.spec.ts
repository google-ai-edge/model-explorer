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

import {signal} from '@angular/core';
import {TestBed} from '@angular/core/testing';
import {MatDialog} from '@angular/material/dialog';
import {WorkspaceStateService} from '../../state/workspace_state_service';
import type {OptionDescription} from '../../server/contracts/runtime';
import {ConfigurationOverview} from './configuration_overview';

const CHAT_OPTIONS: OptionDescription[] = [
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
    key: 'topK',
    label: 'Top K',
    help: '',
    group: 'Sampling',
    type: 'integer',
    default: null,
    available: true,
  },
];

describe('configuration overview', () => {
  it("shows the shown chat's sampler by the server's label, and other values as they are", () => {
    const shown = signal<object | null>({
      generation: {ref: {samplerType: 'topP', topK: 2147483647}, target: {}},
    });
    TestBed.configureTestingModule({
      providers: [
        {
          provide: WorkspaceStateService,
          useValue: {
            session: () => ({id: 's'}),
            chat: shown,
            runtime: {capability: () => ({chat_options: CHAT_OPTIONS})},
          },
        },
        {provide: MatDialog, useValue: {}},
      ],
    });
    const overview = TestBed.runInInjectionContext(() => new ConfigurationOverview());
    expect(overview.value('ref', 'samplerType')).toBe('Top-K + Top-P');
    expect(overview.value('target', 'samplerType')).toBe('Default');
    expect(overview.value('ref', 'topK')).toBe('2147483647');
    expect(overview.differs('samplerType')).toBe(true);
    // The New chat form shows no chat: nothing is recorded, and nothing differs.
    shown.set(null);
    expect(overview.value('ref', 'samplerType')).toBe('Not recorded');
    expect(overview.differs('samplerType')).toBe(false);
  });
});
