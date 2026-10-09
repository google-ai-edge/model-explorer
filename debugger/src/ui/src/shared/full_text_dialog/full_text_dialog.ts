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
import {MatButtonModule} from '@angular/material/button';
import {
  MAT_DIALOG_DATA,
  MatDialogModule,
  MatDialogRef,
} from '@angular/material/dialog';
@Component({
  selector: 'full-text-dialog',
  imports: [MatDialogModule, MatButtonModule],
  template: `<h2 mat-dialog-title>{{ data.title }}</h2>
    <mat-dialog-content>
      <textarea
        readonly
        aria-label="Full captured text"
        [value]="data.text"
        spellcheck="false"
      ></textarea></mat-dialog-content
    ><mat-dialog-actions align="end"
      ><button mat-button (click)="ref.close()">Close</button></mat-dialog-actions
    >`,
  styles: [
    `
      :host {
        display: block;
        color: var(--me-on-surface-color);
        background: var(--me-surface-container-low-color);
      }
      h2 {
        font-family: var(--me-font-family);
      }
      textarea {
        box-sizing: border-box;
        width: 100%;
        height: 55vh;
        resize: vertical;
        border: 1px solid var(--me-outline-variant-color);
        border-radius: 4px;
        padding: 12px;
        background: var(--me-surface-color);
        color: inherit;
        font: 14px/24px var(--me-font-family);
        white-space: pre-wrap;
        overflow-wrap: anywhere;
      }
      button {
        font-family: var(--me-font-family);
      }
    `,
  ],
})
export class FullTextDialog {
  readonly data = inject<{title: string; text: string}>(MAT_DIALOG_DATA);
  readonly ref = inject(MatDialogRef<FullTextDialog>);
}
