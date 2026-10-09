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
import {FindNavigator} from './find_navigator';

@Component({
  imports: [FindNavigator],
  template: `<find-navigator
    [index]="index()"
    [total]="total()"
    [busy]="busy()"
    [countAction]="true"
    countLabel="Toggle results"
    (first)="log.push('first')"
    (previous)="log.push('previous')"
    (next)="log.push('next')"
    (count)="log.push('count')"
  />`,
})
class Host {
  readonly index = signal(-1);
  readonly total = signal(0);
  readonly busy = signal(false);
  readonly log: string[] = [];
}

describe('FindNavigator', () => {
  function render() {
    const fixture = TestBed.createComponent(Host);
    fixture.detectChanges();
    const element: HTMLElement = fixture.nativeElement;
    return {
      fixture,
      host: fixture.componentInstance,
      buttons: () =>
        Array.from(element.querySelectorAll<HTMLButtonElement>('button')),
      group: () => element.querySelector('find-navigator')!,
    };
  }

  it('renders one ARIA group with first, previous, count and next controls', () => {
    const view = render();
    expect(view.group().getAttribute('role')).toBe('group');
    expect(view.group().getAttribute('aria-label')).toBe('Find navigation');
    expect(view.buttons().map((b) => b.getAttribute('aria-label'))).toEqual([
      'First result',
      'Previous result',
      'Toggle results',
      'Next result',
    ]);
    expect(view.buttons()[2].textContent?.trim()).toBe('— / 0');
    expect(view.buttons().map((b) => b.disabled)).toEqual([
      true,
      true,
      false,
      true,
    ]);
  });

  it('enables the right directions for a middle result and emits on click', () => {
    const view = render();
    view.host.total.set(3);
    view.host.index.set(1);
    view.fixture.detectChanges();
    expect(view.buttons().map((b) => b.disabled)).toEqual([
      false,
      false,
      false,
      false,
    ]);
    expect(view.buttons()[2].textContent?.trim()).toBe('2 / 3');
    for (const button of view.buttons()) button.click();
    expect(view.host.log).toEqual(['first', 'previous', 'count', 'next']);
    view.host.busy.set(true);
    view.fixture.detectChanges();
    expect(view.buttons().map((b) => b.disabled)).toEqual([
      true,
      true,
      false,
      true,
    ]);
  });
});
