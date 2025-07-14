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

import {OverlaySizeConfig} from '@angular/cdk/overlay';
import {CommonModule} from '@angular/common';
import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  ElementRef,
  ViewChild,
  computed,
  inject,
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';
import {MatRadioModule} from '@angular/material/radio';
import {MatSnackBar} from '@angular/material/snack-bar';
import {MatTooltipModule} from '@angular/material/tooltip';

import {Bubble} from '../bubble/bubble';
import {BubbleClick} from '../bubble/bubble_click';

import {AppService} from './app_service';
import {
  SYNC_NAVIGATION_MODE_LABELS,
  SyncNavigationMode,
} from './common/sync_navigation';
import {LocalStorageService} from './local_storage_service';
import {SyncNavigationService} from './sync_navigation_service';

const LOCAL_STORAGE_KEY_MATCH_NODE_ID_HIGHLIGHT_DIFFS =
  'sync_navigation_match_node_id_highlight_diffs';

/** The button to manage sync navigation. */
@Component({
  standalone: true,
  selector: 'sync-navigation-button',
  imports: [
    Bubble,
    BubbleClick,
    CommonModule,
    MatButtonModule,
    MatIconModule,
    MatProgressSpinnerModule,
    MatRadioModule,
    MatTooltipModule,
  ],
  templateUrl: 'sync_navigation_button.ng.html',
  styleUrls: ['./sync_navigation_button.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SyncNavigationButton {
  @ViewChild(BubbleClick) dropdown?: BubbleClick;

  private readonly appService = inject(AppService);
  private readonly changeDetectorRef = inject(ChangeDetectorRef);
  private readonly localStorageService = inject(LocalStorageService);
  private readonly syncNavigationService = inject(SyncNavigationService);
  private readonly snackBar = inject(MatSnackBar);

  readonly SyncNavigationMode = SyncNavigationMode;
  readonly allSyncModes: SyncNavigationMode[];
  readonly syncMode = this.syncNavigationService.mode;
  readonly syncEnabled = computed(() => {
    return this.syncMode() !== SyncNavigationMode.DISABLED;
  });
  readonly syncIcon = computed(() =>
    this.syncMode() === SyncNavigationMode.DISABLED &&
    !this.syncNavigationService.loadingFromCns()
      ? 'sync_disabled'
      : 'sync',
  );
  readonly loadingFromCns = this.syncNavigationService.loadingFromCns;
  readonly matchNodeIdHighlightDiffs =
    this.syncNavigationService.matchNodeIdHighlightDiffs;

  readonly helpPopupSize: OverlaySizeConfig = {
    minWidth: 0,
    minHeight: 0,
  };

  readonly dropdownSize: OverlaySizeConfig = {
    minWidth: 0,
    minHeight: 0,
    maxHeight: 500,
  };

  uploadedFileName = '';

  constructor() {
    if (!this.appService.testMode) {
      this.syncNavigationService.matchNodeIdHighlightDiffs.set(
        this.localStorageService.getItem(
          LOCAL_STORAGE_KEY_MATCH_NODE_ID_HIGHLIGHT_DIFFS,
        ) === 'true',
      );
    }

    // Populate sync modes.
    //
    // Show "component input" mode only if there is sync navigation data passed
    // through visualizer config..
    const syncNavigationDataFromVisConfig =
      this.appService.config()?.syncNavigationData;
    this.allSyncModes = syncNavigationDataFromVisConfig
      ? [
          SyncNavigationMode.DISABLED,
          SyncNavigationMode.MATCH_NODE_ID,
          SyncNavigationMode.VISUALIZER_CONFIG,
          SyncNavigationMode.UPLOAD_MAPPING_FROM_COMPUTER,
        ]
      : [
          SyncNavigationMode.DISABLED,
          SyncNavigationMode.MATCH_NODE_ID,
          SyncNavigationMode.UPLOAD_MAPPING_FROM_COMPUTER,
        ];

    // If there is sync navigation data passed through visualizer config, set
    // the sync navigation data for the "visualizer config" mode and select the
    // mode by default.
    if (syncNavigationDataFromVisConfig) {
      this.syncNavigationService.mode.set(SyncNavigationMode.VISUALIZER_CONFIG);
      this.syncNavigationService.updateSyncNavigationData(
        SyncNavigationMode.VISUALIZER_CONFIG,
        syncNavigationDataFromVisConfig,
      );
    }
  }

  setSyncMode(mode: SyncNavigationMode) {
    this.syncNavigationService.mode.set(mode);

    switch (mode) {
      case SyncNavigationMode.DISABLED:
      case SyncNavigationMode.MATCH_NODE_ID:
        this.syncNavigationService.syncNavigationModeChanged$.next({
          mode,
        });
        break;
      default:
        break;
    }
  }

  getModeLabel(mode: SyncNavigationMode): string {
    return SYNC_NAVIGATION_MODE_LABELS[mode];
  }

  handleClickUpload(input: HTMLInputElement) {
    this.syncNavigationService.mode.set(
      SyncNavigationMode.UPLOAD_MAPPING_FROM_COMPUTER,
    );
    input.click();
  }

  handleUploadedFileChanged(input: HTMLInputElement) {
    const files = input.files;
    if (!files || files.length === 0) {
      return;
    }
    const file = files[0];
    this.uploadedFileName = '';

    const fileReader = new FileReader();
    fileReader.onload = (event) => {
      const error = this.syncNavigationService.processJsonData(
        event.target?.result as string,
        SyncNavigationMode.UPLOAD_MAPPING_FROM_COMPUTER,
      );
      if (!error) {
        this.uploadedFileName = file.name;
        this.changeDetectorRef.markForCheck();
      }
    };
    fileReader.readAsText(file);
  }

  handleToggleMatchNodeIdHighlightDiffs(checked: boolean) {
    this.syncNavigationService.matchNodeIdHighlightDiffs.set(checked);
    if (!this.appService.testMode) {
      this.localStorageService.setItem(
        LOCAL_STORAGE_KEY_MATCH_NODE_ID_HIGHLIGHT_DIFFS,
        `${checked}`,
      );
    }
  }

  private showError(message: string) {
    console.error(message);
    this.snackBar.open(message, 'Dismiss', {
      duration: 5000,
    });
  }
}
