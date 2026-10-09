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
import {MatDialogRef} from '@angular/material/dialog';
import {AboutDialog} from './about_dialog';

describe('AboutDialog', () => {
  it('lists every bundled resource with a reachable notice link', () => {
    TestBed.configureTestingModule({
      providers: [{provide: MatDialogRef, useValue: {close() {}}}],
    });
    const fixture = TestBed.createComponent(AboutDialog);
    fixture.detectChanges();
    const element: HTMLElement = fixture.nativeElement;
    const links = Array.from(
      element.querySelectorAll<HTMLAnchorElement>('a'),
    ).map((a) => a.getAttribute('href'));
    for (const expected of [
      'licenses/model-explorer.txt',
      'graph-execution/upstream/model-explorer/PROVENANCE.json',
      'fonts/LICENSE-material-icons.txt',
      '3rdpartylicenses.txt',
    ]) {
      expect(links).toContain(expected);
    }
  });
});
