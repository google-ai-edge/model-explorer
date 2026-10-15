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

import {Component, inject} from '@angular/core';
import {TestBed} from '@angular/core/testing';
import {ChartBackground} from './chart_background';

@Component({
  hostDirectives: [{directive: ChartBackground, outputs: ['backgroundClick']}],
  template:
    '<div class="plot"></div><button>Control</button><div class="rangeslider-container"></div>',
})
class ChartHost {
  readonly background = inject(ChartBackground);
}

describe('chart background inspection', () => {
  afterEach(() => vi.useRealTimers());
  function setup() {
    vi.useFakeTimers();
    const fixture = TestBed.createComponent(ChartHost);
    fixture.detectChanges();
    const host: HTMLElement = fixture.nativeElement;
    const background = fixture.componentInstance.background;
    const inspect = vi.fn();
    background.backgroundClick.subscribe(inspect);
    const fire = (type: string, target = host, options = {}) =>
      target.dispatchEvent(
        new MouseEvent(type, {bubbles: true, clientX: 10, clientY: 10, ...options}),
      );
    return {fixture, host, background, inspect, fire};
  }
  it('opens background inspection but waits for late Plotly data clicks', () => {
    const {fixture, background, inspect, fire} = setup();
    fire('pointerdown');
    fire('click');
    expect(inspect).not.toHaveBeenCalled();
    vi.runAllTimers();
    expect(inspect).toHaveBeenCalledTimes(1);
    fire('pointerdown');
    fire('click');
    background.handled();
    vi.runAllTimers();
    expect(inspect).toHaveBeenCalledTimes(1);
    fixture.destroy();
  });
  it('preserves inspection during zoom gestures and control clicks', () => {
    const {fixture, host, inspect, fire} = setup();
    fire('pointerdown');
    fire('pointermove', host, {buttons: 1, clientX: 40});
    fire('click');
    vi.runAllTimers();
    for (const selector of ['button', '.rangeslider-container']) {
      const target = host.querySelector(selector) as HTMLElement;
      fire('pointerdown', target);
      fire('click', target);
      vi.runAllTimers();
    }
    expect(inspect).not.toHaveBeenCalled();
    fixture.destroy();
  });
  it('cancels delayed inspection when the chart is removed', () => {
    const {fixture, inspect, fire} = setup();
    fire('pointerdown');
    fire('click');
    fixture.destroy();
    vi.runAllTimers();
    expect(inspect).not.toHaveBeenCalled();
  });
});
