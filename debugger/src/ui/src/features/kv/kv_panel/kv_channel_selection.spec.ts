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
import {KvHeadDetails} from './kv_head_details';
import type {KvHeadEvidence} from '../../../data/contracts/kv';

describe('tensor channel selection readout', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    Reflect.deleteProperty(Range.prototype, 'getBoundingClientRect');
  });
  function setup() {
    vi.stubGlobal(
      'ResizeObserver',
      class {
        observe() {}
        disconnect() {}
      },
    );
    Object.defineProperty(Range.prototype, 'getBoundingClientRect', {
      configurable: true,
      value: () => new DOMRect(),
    });
    const fixture = TestBed.createComponent(KvHeadDetails);
    fixture.componentRef.setInput('plotOnly', true);
    fixture.componentRef.setInput('hidePlot', true);
    fixture.componentRef.setInput('channelRange', {start: 0, end: 2});
    fixture.componentRef.setInput('channelFull', {start: 0, end: 4});
    fixture.componentRef.setInput('evidence', {
      channels: [0, 1, 2, 3].map((channel) => ({
        channel,
        ref: channel,
        target: channel + 2,
        delta: 2,
        abs_delta: 2,
        status: 'ok',
      })),
      largest_channel: 3,
      status: 'ok',
    } as KvHeadEvidence);
    fixture.detectChanges();
    return fixture;
  }
  it('starts without an automatic Head dimension and keeps selection separate from the displayed range', () => {
    const fixture = setup(),
      details = fixture.componentInstance;
    expect(details.channel()).toBeNull();
    expect(details.selected()).toBeUndefined();
    expect(fixture.nativeElement.querySelector('.channel-navigation')).toBeNull();
    expect(fixture.nativeElement.querySelector('.channel-readout')).toBeNull();
    const zoom = vi.fn();
    details.channelZoom.subscribe(zoom);
    details.selectChannel(3);
    expect(details.channel()).toBe(3);
    expect(details.selectionOutsideRange()).toBe(true);
    expect(zoom).not.toHaveBeenCalled();
    details.selectChannel(1);
    expect(details.selectionOutsideRange()).toBe(false);
    expect(details.scalarRows().map((row) => row.value)).toEqual([1, 3, 2, 2]);
    fixture.destroy();
  });
  it('shows bin reductions instead of stale single-head dimension values and restores exact readouts', () => {
    const fixture = setup(),
      details = fixture.componentInstance;
    fixture.componentRef.setInput('selectedBin', {start: 0, end: 2});
    fixture.detectChanges();
    expect(details.selectionLabel()).toBe('Head dimensions 0–1');
    expect(details.scalarRows().map((row) => row.value)).toEqual([0.5, 2.5]);
    fixture.componentRef.setInput('channelReduction', 'max');
    fixture.detectChanges();
    expect(details.scalarRows().map((row) => row.value)).toEqual([1, 3]);
    fixture.componentRef.setInput('selectedBin', null);
    details.selectChannel(1);
    fixture.detectChanges();
    expect(details.selectionLabel()).toBe('Head dimension 1');
    expect(details.scalarRows().map((row) => row.value)).toEqual([1, 3, 2, 2]);
    fixture.destroy();
  });
  it('preserves a parent-supplied channel selection in plotOnly mode across evidence updates', () => {
    const fixture = setup();
    const details = fixture.componentInstance;
    const channelChanges: (number | null)[] = [];
    details.channel.subscribe((value) => channelChanges.push(value));
    fixture.componentRef.setInput('channel', 2);
    fixture.detectChanges();
    expect(details.channel()).toBe(2);
    fixture.componentRef.setInput('evidence', {
      ...details.evidence()!,
      largest_channel: 1,
    });
    fixture.detectChanges();
    expect(details.channel()).toBe(2);
    expect(channelChanges).not.toContain(null);
    fixture.destroy();
  });
});
