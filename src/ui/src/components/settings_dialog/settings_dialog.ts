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

import {ConnectedPosition, OverlaySizeConfig} from '@angular/cdk/overlay';
import {CommonModule} from '@angular/common';
import {Component} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatDialogModule} from '@angular/material/dialog';
import {MatIconModule} from '@angular/material/icon';
import {MatSlideToggleModule} from '@angular/material/slide-toggle';

import {
  ALL_SETTINGS,
  Setting,
  SettingsService,
  SettingType,
} from '../../services/settings_service';
import {Bubble} from '../bubble/bubble';

/**
 * A dialog showing app level settings.
 */
@Component({
  selector: 'settings-dialog',
  standalone: true,
  imports: [
    Bubble,
    CommonModule,
    MatButtonModule,
    MatDialogModule,
    MatIconModule,
    MatSlideToggleModule,
  ],
  templateUrl: './settings_dialog.ng.html',
  styleUrls: ['./settings_dialog.scss'],
})
export class SettingsDialog {
  readonly SettingType = SettingType;

  readonly helpPopupSize: OverlaySizeConfig = {
    minWidth: 0,
    minHeight: 0,
    maxWidth: 340,
  };

  readonly helpPopupPosition: ConnectedPosition[] = [
    {
      originX: 'end',
      originY: 'top',
      overlayX: 'start',
      overlayY: 'top',
      offsetX: 4,
    },
  ];

  constructor(readonly settingsService: SettingsService) {}

  /** All settings. */
  readonly allSettings: Setting[] = ALL_SETTINGS;

  handleClickResetToDefaultText(setting: Setting) {
    this.settingsService.saveStringValue(
      setting.defaultValue as string,
      setting.key,
    );
  }

  handleClickResetToDefaultNumber(setting: Setting) {
    this.settingsService.saveNumberValue(
      setting.defaultValue as number,
      setting.key,
    );
  }
}
