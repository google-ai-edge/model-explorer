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
import {MatDialogModule, MAT_DIALOG_DATA} from '@angular/material/dialog';
import type {SessionSummary} from '../../server/contracts/sessions';

/**
 * Turning a Model Server off ends its Session and its live chat: the runtime cannot resume that
 * context later, and a Session starts once.
 */
@Component({
  selector: 'model-server-off',
  imports: [MatDialogModule, MatButtonModule],
  template: `<h2 mat-dialog-title>Turn off the Model Server?</h2>
    <mat-dialog-content>
      <p>
        This ends <strong>{{ session.name }}</strong> and its live chat, and releases both devices.
      </p>
      <p>Every chat's captured turns are kept. No chat can be continued or started later.</p>
    </mat-dialog-content>
    <mat-dialog-actions align="end">
      <button mat-button mat-dialog-close cdkFocusInitial>Cancel</button>
      <button mat-button [mat-dialog-close]="true">Turn off</button>
    </mat-dialog-actions>`,
})
export class ModelServerOff {
  readonly session = inject<SessionSummary>(MAT_DIALOG_DATA);
}
