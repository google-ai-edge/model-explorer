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
import type {LayerOps} from '../../server/contracts/artifacts';
import {flushMicrotasks} from '../../server/testing/session_fixtures';
import {LayerOpsPicker} from './layer_ops_picker';

function op(id: string, name: string, fields: Partial<LayerOps['blocks'][number]['ops'][number]>) {
  return {
    id,
    name,
    kind: 'Linear',
    label: `Attention ${name}`,
    available: true,
    reason: '',
    size: '1 KiB',
    size_detail: 'Reference 512 B · Target 512 B per generated token',
    ...fields,
  };
}

const LIST: LayerOps = {
  status: 'found',
  text: '',
  title: 'Inside each layer',
  description: 'Tick an operator to capture it in every layer.',
  blocks: [
    {
      name: 'Attention',
      ops: [
        op('attn/q_proj', 'q_proj', {}),
        op('attn/k_proj', 'k_proj', {
          reason: "not in layers 15–34 (they reuse layer 13/14's K and V)",
        }),
        op('attn/q_rope', 'q_rope', {available: false, reason: 'no tensor of it'}),
      ],
    },
  ],
  notes: ['In prefill, gemma computes only layers 0–14.'],
  total: {bytes: 1024, text: 'Ticked: 1 operator, 1 KiB more per generated token.'},
  refusal: '',
};

describe('LayerOpsPicker', () => {
  it('lists the server operators by block; ticking one asks again with it', async () => {
    const asked: unknown[] = [];
    let answer = LIST;
    TestBed.configureTestingModule({
      providers: [
        {
          provide: ArtifactsApi,
          useValue: {
            layerOps: async (...args: unknown[]) => {
              asked.push(args);
              return answer;
            },
          },
        },
      ],
    });
    const fixture = TestBed.createComponent(LayerOpsPicker);
    const chosen: string[][] = [];
    const ready: boolean[] = [];
    fixture.componentInstance.selectionChange.subscribe((ops) => chosen.push(ops));
    fixture.componentInstance.readiness.subscribe((value) => ready.push(value));
    fixture.componentRef.setInput('ref', 'a.litertlm');
    fixture.componentRef.setInput('target', 'b');
    fixture.componentRef.setInput('selected', ['attn/q_proj']);
    fixture.detectChanges();
    await flushMicrotasks();
    fixture.detectChanges();
    expect(asked).toEqual([['a.litertlm', 'b', ['attn/q_proj']]]);
    const element = fixture.nativeElement as HTMLElement;
    const text = element.textContent ?? '';
    expect(text).toContain('Inside each layer');
    expect(text).toContain("not in layers 15–34 (they reuse layer 13/14's K and V)");
    expect(text).toContain('In prefill, gemma computes only layers 0–14.');
    expect(text).toContain('Ticked: 1 operator, 1 KiB more per generated token.');
    const boxes = element.querySelectorAll<HTMLInputElement>('input[type=checkbox]');
    expect(Array.from(boxes).map((box) => [box.checked, box.disabled])).toEqual([
      [true, false],
      [false, false],
      [false, true],
    ]);
    expect(ready).toEqual([true]);
    // Ticked in the list's order, whatever order they were ticked in.
    fixture.componentInstance.toggle('attn/k_proj', true);
    expect(chosen).toEqual([['attn/q_proj', 'attn/k_proj']]);
    answer = {...LIST, refusal: 'a.litertlm would observe 1,100 tensor outputs.'};
    fixture.componentRef.setInput('selected', ['attn/q_proj', 'attn/k_proj']);
    fixture.detectChanges();
    await flushMicrotasks();
    fixture.detectChanges();
    expect(ready).toEqual([true, false]);
    expect(element.querySelector('[role=alert]')?.textContent).toContain('1,100 tensor outputs');
  });

  it('says why the list is not known', async () => {
    TestBed.configureTestingModule({
      providers: [
        {
          provide: ArtifactsApi,
          useValue: {
            layerOps: async () => ({
              ...LIST,
              status: 'pending',
              text: "The operators inside a's layers are found in its graph.",
              blocks: [],
            }),
          },
        },
      ],
    });
    const fixture = TestBed.createComponent(LayerOpsPicker);
    fixture.componentRef.setInput('ref', 'a');
    fixture.componentRef.setInput('target', 'a');
    fixture.detectChanges();
    await flushMicrotasks();
    fixture.detectChanges();
    const element = fixture.nativeElement as HTMLElement;
    expect(element.querySelector('[data-ops-status=pending]')?.textContent).toContain(
      'found in its graph',
    );
    expect(element.querySelectorAll('input').length).toBe(0);
  });
});
