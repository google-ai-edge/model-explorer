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

import {Component, signal} from '@angular/core';
import {TestBed} from '@angular/core/testing';
import {ResizeHandle} from './resize_handle';

@Component({
  imports: [ResizeHandle],
  template: `<div
    resizeHandle
    [size]="size()"
    [min]="100"
    [max]="500"
    [resetSize]="300"
    (sizeChange)="size.set($event)"
    (resizingChange)="resizing.push($event)"
  ></div>`,
})
class Host {
  readonly size = signal(200);
  readonly resizing: boolean[] = [];
}

describe('ResizeHandle', () => {
  function render() {
    const fixture = TestBed.createComponent(Host);
    fixture.detectChanges();
    const handle: HTMLElement =
      fixture.nativeElement.querySelector('[resizeHandle]');
    const key = (key: string, shiftKey = false) => {
      handle.dispatchEvent(
        new KeyboardEvent('keydown', {key, shiftKey, bubbles: true}),
      );
      fixture.detectChanges();
    };
    return {fixture, host: fixture.componentInstance, handle, key};
  }

  it('exposes the separator ARIA state from its inputs', () => {
    const view = render();
    expect(view.handle.getAttribute('role')).toBe('separator');
    expect(view.handle.getAttribute('tabindex')).toBe('0');
    expect(view.handle.getAttribute('aria-orientation')).toBe('vertical');
    expect(view.handle.getAttribute('aria-valuenow')).toBe('200');
    expect(view.handle.getAttribute('aria-valuemin')).toBe('100');
    expect(view.handle.getAttribute('aria-valuemax')).toBe('500');
  });

  it('steps with the arrow keys, fine steps with Shift, and clamps at Home and End', () => {
    const view = render();
    view.key('ArrowRight');
    expect(view.host.size()).toBe(220);
    view.key('ArrowLeft', true);
    expect(view.host.size()).toBe(215);
    view.key('End');
    expect(view.host.size()).toBe(500);
    view.key('ArrowRight');
    expect(view.host.size()).toBe(500);
    view.key('Home');
    expect(view.host.size()).toBe(100);
    view.key('Tab');
    expect(view.host.size()).toBe(100);
    expect(view.handle.getAttribute('aria-valuenow')).toBe('100');
  });

  it('restores the reset size on double click', () => {
    const view = render();
    view.handle.dispatchEvent(new MouseEvent('dblclick', {bubbles: true}));
    view.fixture.detectChanges();
    expect(view.host.size()).toBe(300);
  });
});
