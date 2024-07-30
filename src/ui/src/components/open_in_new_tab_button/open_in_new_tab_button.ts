/**
 * @license
 * Copyright 2024 The Model Explorer Authors. All Rights Reserved.
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

import {CommonModule} from '@angular/common';
import {Component} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';

/** A button to open model explorer (in colab) into a new tab. */
@Component({
  standalone: true,
  selector: 'open-in-new-tab-button',
  imports: [CommonModule, MatButtonModule, MatIconModule],
  template: `
@if (showOpenInNewTab) {
  <button mat-flat-button color="primary" class="btn-open-in-tab"
      (click)="handleClickOpenInNewTab()">
    <mat-icon>open_in_new</mat-icon>
    Open in new tab
  </button>
}`,
  styles: ``,
})
export class OpenInNewTabButton {
  readonly showOpenInNewTab: boolean;

  readonly isChrome = /Chrome/.test(navigator.userAgent);

  constructor() {
    // Get the url parameter that determines whether to show this button or not.
    const urlParams = new URLSearchParams(window.location.search);
    this.showOpenInNewTab =
      urlParams.get('show_open_in_new_tab') === '1' && this.isChrome;
  }

  handleClickOpenInNewTab() {
    // Use the current url without the show_open_in_new_tab parameter.
    const url = new URL(window.location.href);
    const params = new URLSearchParams(url.search);
    params.delete('show_open_in_new_tab');

    url.search = params.toString();
    window.open(url.toString(), '_blank');
  }
}
