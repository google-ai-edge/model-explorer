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
import {ValueRangeControl} from './value_range_control';
describe('scatter numerical range input', () => {
  it('commits signed decimals with Enter and rejects inverted and empty ranges', () => {
    const f = TestBed.createComponent(ValueRangeControl);
    f.componentRef.setInput('range', {start: -2, end: 2});
    f.detectChanges();
    const changes: unknown[] = [];
    f.componentInstance.rangeChange.subscribe((r) => changes.push(r));
    const input = f.nativeElement.querySelector('input') as HTMLInputElement;
    input.value = '-0.25';
    input.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', bubbles: true}));
    expect(changes).toEqual([{start: -0.25, end: 2}]);
    input.value = '3';
    input.dispatchEvent(new Event('change'));
    expect(input.value).toBe('-2');
    input.value = '';
    input.dispatchEvent(new Event('change'));
    expect(input.value).toBe('-2');
    expect(changes).toHaveLength(1);
    f.destroy();
  });
});
