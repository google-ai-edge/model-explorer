/**
 * @license
 * Copyright 2026 The Model Explorer Authors. All Rights Reserved.
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
import {MAT_DIALOG_DATA, MatDialogModule} from '@angular/material/dialog';
import {MatIconModule} from '@angular/material/icon';

import {type ApiTokenRejectionReason} from '../../common/api_auth';

/** The data passed to `ApiTokenDialog`. */
export interface ApiTokenDialogData {
  reason: ApiTokenRejectionReason;
}

/**
 * A dialog shown when the Model Explorer server rejects the page's access
 * token (missing or invalid). It explains how to open the page with the
 * current token, and how to run the server without one.
 */
@Component({
  selector: 'api-token-dialog',
  standalone: true,
  imports: [MatButtonModule, MatDialogModule, MatIconModule],
  templateUrl: './api_token_dialog.ng.html',
  styleUrls: ['./api_token_dialog.scss'],
})
export class ApiTokenDialog {
  private readonly data = inject<ApiTokenDialogData>(MAT_DIALOG_DATA);

  /** Whether the page sent a token that the server rejected. */
  readonly invalidToken = this.data.reason === 'invalid';
}
