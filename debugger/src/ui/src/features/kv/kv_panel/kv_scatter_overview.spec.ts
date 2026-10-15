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
import {KvScatterOverview} from './kv_scatter_overview';

describe('scatter numerical overview', () => {
  it('bounds the SVG path for a full vocabulary without changing its full-domain window', () => {
    const fixture = TestBed.createComponent(KvScatterOverview);
    fixture.componentRef.setInput('full', {start: 0, end: 512});
    fixture.componentRef.setInput('range', {start: 128, end: 256});
    fixture.componentRef.setInput(
      'rows',
      Array.from({length: 262144}, (_, channel) => ({
        channel,
        ref: channel % 512,
        target: Math.floor(channel / 512),
        delta: null,
        abs_delta: null,
        status: 'ok',
      })),
    );
    fixture.detectChanges();
    expect((fixture.componentInstance.points().match(/M/g) ?? []).length).toBeLessThanOrEqual(4096);
    expect(fixture.componentInstance.box()).toEqual({low: 27.5, high: 50});
    fixture.destroy();
  });
  function setup() {
    const fixture = TestBed.createComponent(KvScatterOverview);
    fixture.componentRef.setInput('full', {start: -10, end: 10});
    fixture.componentRef.setInput('range', {start: -2, end: 2});
    fixture.detectChanges();
    const svg = fixture.nativeElement.querySelector('svg') as SVGSVGElement;
    svg.setPointerCapture = vi.fn();
    svg.hasPointerCapture = () => true;
    svg.releasePointerCapture = vi.fn();
    vi.spyOn(svg, 'getBoundingClientRect').mockReturnValue({width: 100} as DOMRect);
    const changes: {start: number; end: number}[] = [];
    fixture.componentInstance.zoom.subscribe((range) => {
      changes.push(range);
      fixture.componentRef.setInput('range', range);
    });
    const pointer = (action: string, type: string, x: number, y: number) => {
      const target = svg.querySelector(`[data-action="${action}"]`)!;
      target.dispatchEvent(
        Object.assign(new MouseEvent(type, {bubbles: true, button: 0, clientX: x, clientY: y}), {
          pointerId: 1,
        }),
      );
      fixture.detectChanges();
    };
    return {fixture, svg, pointer, changes};
  }
  it('pans a shared numerical interval, clamps to full extent, and resets by keyboard', () => {
    const {fixture, svg, pointer, changes} = setup();
    pointer('pan', 'pointerdown', 50, 50);
    pointer('pan', 'pointermove', 59, 41);
    expect(changes.at(-1)).toEqual({start: 0, end: 4});
    pointer('pan', 'pointermove', 500, -400);
    expect(changes.at(-1)).toEqual({start: 6, end: 10});
    pointer('pan', 'pointerup', 500, -400);
    expect(fixture.componentInstance.draft()).toBeNull();
    svg.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
    expect(changes.at(-1)).toEqual({start: -10, end: 10});
    fixture.destroy();
  });
  it('resizes one numerical bound and restores the initial window on cancellation', () => {
    const {fixture, pointer, changes} = setup();
    pointer('start', 'pointerdown', 40, 60);
    pointer('start', 'pointermove', 31, 69);
    expect(changes.at(-1)).toEqual({start: -4, end: 2});
    pointer('start', 'pointercancel', 31, 69);
    expect(changes.at(-1)).toEqual({start: -2, end: 2});
    pointer('end', 'pointerdown', 60, 40);
    pointer('end', 'pointermove', 69, 31);
    expect(changes.at(-1)).toEqual({start: -2, end: 4});
    pointer('end', 'pointerup', 69, 31);
    fixture.destroy();
  });
});
