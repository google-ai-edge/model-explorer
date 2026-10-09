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

import {OverlayModule} from '@angular/cdk/overlay';
import {Component, signal} from '@angular/core';
import {TestBed} from '@angular/core/testing';
import {OverlayDialog, OverlayPanel} from './overlay_dialog';

@Component({
  imports: [OverlayModule, OverlayDialog, OverlayPanel],
  template: `<button cdkOverlayOrigin #origin="cdkOverlayOrigin" (click)="open.set(true)">
      Open
    </button>
    <ng-template overlayDialog [origin]="origin" [open]="open()" (closed)="closed()">
      <section overlayPanel aria-label="Sample dialog"><button id="inside">Inside</button></section>
    </ng-template>`,
})
class Host {
  readonly open = signal(false);
  closes = 0;
  closed() {
    this.closes++;
    this.open.set(false);
  }
}

describe('OverlayDialog', () => {
  afterEach(() =>
    document
      .querySelectorAll('.cdk-overlay-container')
      .forEach((el) => el.remove()),
  );

  function render() {
    const fixture = TestBed.createComponent(Host);
    fixture.detectChanges();
    return {fixture, host: fixture.componentInstance};
  }
  const dialog = () =>
    document.querySelector<HTMLElement>(
      '[role="dialog"][aria-label="Sample dialog"]',
    );

  it('opens an ARIA dialog with a focus trap when the host signal turns on', async () => {
    const view = render();
    expect(dialog()).toBeNull();
    view.host.open.set(true);
    view.fixture.detectChanges();
    await view.fixture.whenStable();
    expect(dialog()).not.toBeNull();
    expect(document.querySelector('.cdk-overlay-backdrop')).not.toBeNull();
    // The CDK wraps the trapped element with two sibling anchors inside the overlay pane.
    expect(
      document.querySelectorAll('.cdk-overlay-pane .cdk-focus-trap-anchor')
        .length,
    ).toBe(2);
  });

  it('reports Escape and backdrop clicks as user closes but not programmatic closes', async () => {
    const view = render();
    view.host.open.set(true);
    view.fixture.detectChanges();
    await view.fixture.whenStable();
    document.body.dispatchEvent(
      new KeyboardEvent('keydown', {key: 'Escape', keyCode: 27, bubbles: true}),
    );
    view.fixture.detectChanges();
    expect(view.host.closes).toBe(1);
    expect(view.host.open()).toBe(false);
    view.host.open.set(true);
    view.fixture.detectChanges();
    await view.fixture.whenStable();
    document.querySelector<HTMLElement>('.cdk-overlay-backdrop')!.click();
    view.fixture.detectChanges();
    expect(view.host.closes).toBe(2);
    view.host.open.set(true);
    view.fixture.detectChanges();
    await view.fixture.whenStable();
    view.host.open.set(false);
    view.fixture.detectChanges();
    await view.fixture.whenStable();
    expect(view.host.closes).toBe(2);
  });
});
