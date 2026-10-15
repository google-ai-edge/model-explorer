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
import {ExplorerInfoValue} from './explorer_info_value';

describe('Explorer inline value text action', () => {
  const rangeRect = Object.getOwnPropertyDescriptor(Range.prototype, 'getBoundingClientRect');
  let textWidth = 120;
  let resized: () => void;
  beforeEach(() => {
    textWidth = 120;
    vi.stubGlobal(
      'ResizeObserver',
      class {
        constructor(callback: () => void) {
          resized = callback;
        }
        observe() {}
        disconnect() {}
      },
    );
    vi.spyOn(Element.prototype, 'getBoundingClientRect').mockReturnValue({
      x: 0,
      y: 0,
      top: 0,
      left: 0,
      right: 60,
      bottom: 20,
      width: 60,
      height: 20,
      toJSON() {},
    });
    Object.defineProperty(Range.prototype, 'getBoundingClientRect', {
      configurable: true,
      value: () => ({width: textWidth}),
    });
  });
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    if (rangeRect) Object.defineProperty(Range.prototype, 'getBoundingClientRect', rangeRect);
    else Reflect.deleteProperty(Range.prototype, 'getBoundingClientRect');
  });
  function create(textAction?: boolean, inline = true) {
    const fixture = TestBed.createComponent(ExplorerInfoValue);
    fixture.componentRef.setInput('value', '-12345.678901');
    fixture.componentRef.setInput('label', 'Maximum');
    fixture.componentRef.setInput('inline', inline);
    fixture.componentRef.setInput('reserveAction', true);
    if (textAction !== undefined) fixture.componentRef.setInput('textAction', textAction);
    fixture.detectChanges();
    fixture.detectChanges();
    const value = fixture.nativeElement.querySelector('.value') as HTMLElement;
    return {fixture, value, view: fixture.componentInstance};
  }

  it('toggles the clipped text in place on click without consuming a separate action slot', () => {
    const {fixture, value, view} = create(true);
    expect(view.truncated()).toBe(true);
    expect(fixture.nativeElement.classList.contains('reserve-action')).toBe(false);
    expect(fixture.nativeElement.querySelector('.read-value')).toBeNull();
    expect(value.getAttribute('role')).toBe('button');
    expect(value.tabIndex).toBe(0);
    expect(value.getAttribute('aria-expanded')).toBe('false');
    value.focus();
    value.click();
    fixture.detectChanges();
    expect(view.expanded()).toBe(true);
    expect(fixture.nativeElement.querySelector('.value')).toBe(value);
    expect(fixture.nativeElement.querySelector('.expanded-value')).toBeNull();
    expect(value.textContent?.trim()).toBe('-12345.678901');
    expect(getComputedStyle(value).visibility).toBe('visible');
    expect(value.getAttribute('aria-expanded')).toBe('true');
    expect(document.activeElement).toBe(value);
    value.click();
    fixture.detectChanges();
    expect(view.expanded()).toBe(false);
    expect(value.getAttribute('aria-expanded')).toBe('false');
    expect(document.activeElement).toBe(value);
    fixture.destroy();
  });

  it('expands and collapses with Enter and Space while retaining the focused value', () => {
    const {fixture, value, view} = create(true);
    value.focus();
    for (const [key, expanded] of [
      ['Enter', true],
      [' ', false],
      [' ', true],
      ['Enter', false],
    ] as const) {
      const event = new KeyboardEvent('keydown', {key, bubbles: true, cancelable: true});
      value.dispatchEvent(event);
      fixture.detectChanges();
      expect(event.defaultPrevented).toBe(true);
      expect(view.expanded()).toBe(expanded);
      expect(value.getAttribute('aria-expanded')).toBe(String(expanded));
      expect(document.activeElement).toBe(value);
      expect(fixture.nativeElement.querySelector('.value')).toBe(value);
    }
    fixture.destroy();
  });

  it('does not add a control or toggle text that already fits', () => {
    textWidth = 30;
    const {fixture, value, view} = create(true);
    expect(view.truncated()).toBe(false);
    expect(value.getAttribute('role')).toBeNull();
    expect(value.getAttribute('tabindex')).toBeNull();
    expect(value.getAttribute('aria-expanded')).toBeNull();
    value.click();
    value.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', bubbles: true}));
    value.dispatchEvent(new KeyboardEvent('keydown', {key: ' ', bubbles: true}));
    fixture.detectChanges();
    expect(view.expanded()).toBe(false);
    expect(fixture.nativeElement.querySelector('button')).toBeNull();
    fixture.destroy();
  });

  it('retains the separate expand and collapse button for existing inline values by default', () => {
    const {fixture, value, view} = create();
    expect(fixture.nativeElement.classList.contains('reserve-action')).toBe(true);
    expect(value.getAttribute('role')).toBeNull();
    const button = fixture.nativeElement.querySelector('.read-value') as HTMLButtonElement;
    expect(button.getAttribute('aria-label')).toBe('Expand Maximum');
    button.click();
    fixture.detectChanges();
    expect(view.expanded()).toBe(true);
    expect(fixture.nativeElement.querySelector('.expanded-value')?.textContent).toBe(
      '-12345.678901',
    );
    expect(button.getAttribute('aria-label')).toBe('Collapse Maximum');
    button.click();
    fixture.detectChanges();
    expect(view.expanded()).toBe(false);
    expect(fixture.nativeElement.querySelector('.expanded-value')).toBeNull();
    fixture.destroy();
  });

  it('ignores textAction for non-inline values and keeps their full-text action', () => {
    const {fixture, value, view} = create(true, false);
    Object.defineProperties(value, {scrollWidth: {value: 120}, clientWidth: {value: 60}});
    resized();
    fixture.detectChanges();
    expect(view.truncated()).toBe(true);
    expect(fixture.nativeElement.classList.contains('text-action')).toBe(false);
    expect(value.getAttribute('role')).toBeNull();
    expect(value.getAttribute('aria-expanded')).toBeNull();
    expect(fixture.nativeElement.querySelector('.read-value')?.getAttribute('aria-label')).toBe(
      'Read full Maximum',
    );
    value.dispatchEvent(new KeyboardEvent('keydown', {key: ' ', bubbles: true}));
    expect(view.expanded()).toBe(false);
    fixture.destroy();
  });
});
