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
import {Component, Injectable, signal} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';

import {IS_EXTERNAL} from '../../common/flags';
import {Bubble} from '../bubble/bubble';

const CHECK_NEW_VERSION = '/api/v1/check_new_version';

/** Response from check_new_version api. */
export declare interface CheckNewVersionResponse {
  // Empty means no new version.
  version: string;
  runningVersion: string;
}

/** Service to check new version. */
@Injectable({providedIn: 'root'})
export class NewVersionService {
  info = signal<CheckNewVersionResponse>({
    version: '',
    runningVersion: '',
  });

  constructor() {
    if (IS_EXTERNAL) {
      this.checkNewVersion();
    }
  }

  private async checkNewVersion() {
    try {
      const resp = await fetch(CHECK_NEW_VERSION);
      if (resp.ok) {
        const json = (await resp.json()) as CheckNewVersionResponse;
        this.info.set(json);
      }
    } catch (e) {
      // Ignore.
    }
  }
}

/**
 * A chip to show when there is new version available.
 *
 * Only for external version.
 */
@Component({
  standalone: true,
  selector: 'new-version-chip',
  imports: [Bubble, CommonModule, MatIconModule],
  template: `
@if (info().version !== '') {
  <div class="container"
      [bubble]="upgrade"
      [hoverDelayMs]="100">
    <mat-icon>upgrade</mat-icon>
    New version available
  </div>
  <ng-template #upgrade>
    <div class="model-explorer-upgrade-popup">
      Model Explorer <span class="bold">v{{info().version}}</span> is available.
      You are running <span class="bold">v{{info().runningVersion}}</span>.
      <br><br>
      Run the following command in your console to upgrade:
      <div class="code">
        pip install -U ai-edge-model-explorer
      </div>
    </div>
  </ng-template>
}
`,
  styles: `
.container {
  font-size: 12px;
  display: flex;
  align-items: center;
  color: #ab6c17;
  background-color: #ffefd9;
  line-height: 18px;
  border-radius: 99px;
  font-weight: 500;
  padding: 2px 10px 2px 4px;
  cursor: pointer;

  mat-icon {
    font-size: 16px;
    width: 16px;
    height: 16px;
  }
}

::ng-deep .model-explorer-upgrade-popup {
  padding: 8px;
  font-size: 12px;
  background-color: white;
  line-height: normal;

  .bold {
    font-weight: 500;
  }

  .code {
    background-color: #f1f1f1;
    font-family: monospace;
    margin-top: 8px;
    padding: 4px;
    font-size: 11px;
  }
}
`,
})
export class NewVersionChip {
  readonly info;

  constructor(private readonly newVersionService: NewVersionService) {
    this.info = this.newVersionService.info;
  }
}
