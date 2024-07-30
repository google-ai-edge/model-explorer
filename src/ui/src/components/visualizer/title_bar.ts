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
import {
  ChangeDetectionStrategy,
  Component,
  EventEmitter,
  Output,
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';

import {OpenInNewTabButton} from '../open_in_new_tab_button/open_in_new_tab_button';

import {AppService} from './app_service';
import {GraphSelector} from './graph_selector';
import {Logo} from './logo';
import {NewVersionChip} from './new_version_chip';
import {NodeDataProviderDropdown} from './node_data_provider_dropdown';
import {NodeStyler} from './node_styler';

/** The title bar component. */
@Component({
  standalone: true,
  selector: 'title-bar',
  imports: [
    CommonModule,
    GraphSelector,
    Logo,
    MatButtonModule,
    MatIconModule,
    MatTooltipModule,
    NewVersionChip,
    NodeDataProviderDropdown,
    NodeStyler,
    OpenInNewTabButton,
  ],
  templateUrl: './title_bar.ng.html',
  styleUrls: ['./title_bar.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class TitleBar {
  @Output() readonly titleClicked = new EventEmitter<void>();

  constructor(private readonly appService: AppService) {}

  get disableTitleTooltip(): boolean {
    return this.appService.testMode;
  }
}
