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
import type {TapCounterparts, TapPoint, TapScan} from '../../server/contracts/artifacts';
import {TapPicker} from './tap_picker';

function point(id: string, name: string): TapPoint {
  return {
    id,
    signature: 'decode',
    op: Number(id.split(':')[1]),
    output: 0,
    tensor: 1,
    tensor_name: name,
    shape: [1, 1, 2048],
    dtype: 'float32',
    operator: 'DEQUANTIZE',
    selectable: true,
    reason: '',
    note: '',
  };
}

const SCAN: TapScan = {
  status: 'completed',
  points: [
    point('decode:241:0', 'x/layer_3/layer_3.pre_qkv/q_einsum/composite1'),
    point('decode:239:0', 'x/layer_3/layer_3.pre_qkv/q_einsum/reshape'),
  ],
} as TapScan;

const PAIRING: TapCounterparts = {
  status: 'found',
  text: '',
  points: [
    {
      id: 'decode:241:0',
      paired: 'module',
      layer: 3,
      place: 'attn/q_proj',
      label: 'Attention q_proj',
      module: 'model.language_model.layers.3.self_attn.q_proj',
      edge: 'out',
      text: 'Layer 3 Attention q_proj: paired with PyTorch model.language_model.layers.3.self_attn.q_proj, what it returns.',
    },
    {
      id: 'decode:239:0',
      paired: '',
      layer: 3,
      place: '',
      label: '',
      module: '',
      edge: '',
      text: 'PyTorch has no counterpart for decode:239:0; only the anchors are captured there.',
    },
  ],
};

describe('TapPicker', () => {
  it('shows how the server pairs each output chosen with the PyTorch model', async () => {
    const asked: [string, string, string[]][] = [];
    TestBed.configureTestingModule({
      providers: [
        {
          provide: ArtifactsApi,
          useValue: {
            startTapScan: async () => SCAN,
            tapScan: async () => SCAN,
            tapCounterparts: async (id: string, pytorch: string, points: string[]) => {
              asked.push([id, pytorch, points]);
              return PAIRING;
            },
          },
        },
      ],
    });
    const fixture = TestBed.createComponent(TapPicker);
    fixture.componentRef.setInput('artifact', 'model.litertlm');
    fixture.componentRef.setInput('label', 'Target · model');
    fixture.componentRef.setInput('enabled', true);
    fixture.componentRef.setInput('selected', ['decode:241:0', 'decode:239:0']);
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();
    expect(asked, 'no PyTorch side: nothing asked').toEqual([]);
    expect(fixture.nativeElement.querySelector('small.pairing')).toBeNull();

    fixture.componentRef.setInput('pairedWith', '/models/gemma');
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();

    expect(asked).toEqual([['model.litertlm', '/models/gemma', ['decode:241:0', 'decode:239:0']]]);
    const notes = [...fixture.nativeElement.querySelectorAll('small.pairing')] as HTMLElement[];
    expect(notes.map((n) => n.textContent?.trim())).toEqual(PAIRING.points.map((p) => p.text));
    expect(notes.map((n) => n.classList.contains('unpaired'))).toEqual([false, true]);
  });
});
