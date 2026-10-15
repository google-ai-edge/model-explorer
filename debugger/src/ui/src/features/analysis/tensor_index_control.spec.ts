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
import {TensorIndexControl} from './tensor_index_control';

describe('TensorIndexControl', () => {
  it('supports named dimensions beyond layers and heads and emits only valid coordinates', () => {
    const fixture = TestBed.createComponent(TensorIndexControl),
      control = fixture.componentInstance;
    fixture.componentRef.setInput('dimensions', [
      {id: 'layer', label: 'Layer', value: '2', options: ['2', '3']},
      {id: 'sync', label: 'Sync point', value: 'attention', options: ['attention', 'mlp']},
    ]);
    const changes: unknown[] = [];
    control.valueChange.subscribe((change) => changes.push(change));
    expect(control.summary()).toBe('[Layer: 2, Sync point: attention]');
    control.choose('sync', 'mlp');
    control.choose('head', '0');
    control.choose('layer', '99');
    expect(changes).toEqual([{dimension: 'sync', value: 'mlp'}]);
    fixture.componentRef.setInput('dimensions', [
      {id: 'sync', label: 'Sync point', value: null, options: ['mlp'], disabled: true},
    ]);
    control.choose('sync', 'mlp');
    expect(changes).toHaveLength(1);
    expect(control.summary()).toBe('[Sync point: -]');
  });
});
