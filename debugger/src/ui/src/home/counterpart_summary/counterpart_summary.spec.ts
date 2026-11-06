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

import {TestBed} from '@angular/core/testing';
import {ArtifactsApi} from '../../server/artifacts_api';
import type {TapCounterparts} from '../../server/contracts/artifacts';
import {flushMicrotasks} from '../../server/testing/session_fixtures';
import {CounterpartSummary} from './counterpart_summary';

describe('CounterpartSummary', () => {
  it('lists the modules captured for the outputs chosen, and why the rest have none', async () => {
    const asked: unknown[] = [];
    const found: TapCounterparts = {
      status: 'found',
      text: '',
      pytorch_text: 'Beside its anchors, gemma also captures 1 module.',
      points: [
        {
          id: 'decode:1:0',
          paired: 'module',
          layer: 0,
          place: 'attn/q_proj',
          label: 'q_proj',
          module: 'q',
          edge: 'out',
          text: 'Layer 0 q_proj: paired with PyTorch q, what it returns.',
        },
        {
          id: 'decode:2:0',
          paired: '',
          layer: 0,
          place: '',
          label: '',
          module: '',
          edge: '',
          text: 'Not its output.',
        },
        {
          id: 'decode:3:0',
          paired: 'anchor',
          layer: 0,
          place: 'output',
          label: 'Output',
          module: '',
          edge: '',
          text: 'An anchor.',
        },
      ],
    };
    TestBed.configureTestingModule({
      providers: [
        {
          provide: ArtifactsApi,
          useValue: {
            tapCounterparts: async (...args: unknown[]) => {
              asked.push(args);
              return found;
            },
          },
        },
      ],
    });
    const fixture = TestBed.createComponent(CounterpartSummary);
    fixture.componentRef.setInput('artifact', 'model.litertlm');
    fixture.componentRef.setInput('pytorch', '/models/gemma');
    fixture.componentRef.setInput('points', ['decode:1:0', 'decode:2:0', 'decode:3:0']);
    fixture.detectChanges();
    await flushMicrotasks();
    fixture.detectChanges();
    expect(asked).toEqual([
      ['model.litertlm', '/models/gemma', ['decode:1:0', 'decode:2:0', 'decode:3:0']],
    ]);
    const text = (fixture.nativeElement as HTMLElement).textContent ?? '';
    expect(text).toContain('Beside its anchors, gemma also captures 1 module.');
    expect(text).toContain('paired with PyTorch q');
    expect(text).toContain('Not its output.');
    expect(text, 'an anchor is captured as one already').not.toContain('An anchor.');
  });
});
