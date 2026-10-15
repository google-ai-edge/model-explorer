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

import {vi} from 'vitest';
import {TestBed} from '@angular/core/testing';
import {ExplorerHelpTerm} from './explorer_help_term';

describe('Explorer help interaction', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'ResizeObserver',
      class {
        observe() {}
        disconnect() {}
      },
    );
    Object.defineProperty(Range.prototype, 'getBoundingClientRect', {
      configurable: true,
      value: () => ({width: 0}),
    });
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    delete (Range.prototype as unknown as Record<string, unknown>)['getBoundingClientRect'];
  });
  function create(text: string) {
    const fixture = TestBed.createComponent(ExplorerHelpTerm);
    fixture.componentRef.setInput('text', text);
    fixture.componentRef.setInput('help', {title: text, rows: [{label: 'Count', value: '32'}]});
    fixture.detectChanges();
    return fixture;
  }
  it('pins a help panel, replaces it with another, and closes on Escape', () => {
    const a = create('Token position'),
      b = create('Head');
    a.componentInstance.toggle(new Event('click'));
    a.detectChanges();
    expect(document.querySelector('[role="dialog"]')?.textContent).toContain('32');
    expect(document.querySelector('[role="dialog"] dt')?.textContent).toBe('Count');
    expect(document.querySelector('[role="dialog"] .body')).toBeNull();
    b.componentInstance.show();
    expect(a.componentInstance.open()).toBe(false);
    expect(a.componentInstance.pinned()).toBe(false);
    expect(b.componentInstance.open()).toBe(true);
    b.componentInstance.onOverlayKey(new KeyboardEvent('keydown', {key: 'Escape'}));
    expect(b.componentInstance.open()).toBe(false);
  });
  it('keeps pinned help open on pointer leave and closes on a second click', async () => {
    const fixture = create('Layer'),
      view = fixture.componentInstance;
    view.toggle(new Event('click'));
    view.scheduleClose();
    await new Promise((resolve) => setTimeout(resolve, 180));
    expect(view.open()).toBe(true);
    view.toggle(new Event('click'));
    expect(view.open()).toBe(false);
  });
  it('shows literal token text without metadata rows and keeps keyboard dismissal', () => {
    const fixture = create('65');
    const token = '  <img src=x onerror=alert(1)>\n"token\\n"  ';
    fixture.componentRef.setInput('help', {title: 'Token', text: token, rows: []});
    fixture.detectChanges();
    const trigger = fixture.nativeElement.querySelector('button') as HTMLButtonElement;
    trigger.click();
    fixture.detectChanges();
    const panel = document.querySelector('[role="dialog"]')!;
    expect(panel.getAttribute('aria-label')).toBe('Token');
    expect(panel.querySelector('.title')?.textContent).toBe('Token');
    expect(panel.querySelector('.body')?.textContent).toBe(token);
    expect(panel.querySelector('dl')).toBeNull();
    expect(panel.querySelector('img')).toBeNull();
    trigger.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
    fixture.detectChanges();
    expect(document.querySelector('[role="dialog"]')).toBeNull();
    expect(fixture.componentInstance.open()).toBe(false);
  });
});
