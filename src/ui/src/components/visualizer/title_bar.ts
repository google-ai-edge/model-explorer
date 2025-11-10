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
  inject,
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatDialog} from '@angular/material/dialog';
import {MatIconModule} from '@angular/material/icon';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';
import {MatSnackBar} from '@angular/material/snack-bar';
import {MatTooltipModule} from '@angular/material/tooltip';

import {ConfigValue, NodeDataProviderExtension} from '../../common/types';
import {ExtensionService} from '../../services/extension_service';
import {OpenInNewTabButton} from '../open_in_new_tab_button/open_in_new_tab_button';

import {AppService} from './app_service';
import {GraphSelector} from './graph_selector';
import {Logo} from './logo';
import {NewVersionChip} from './new_version_chip';
import {NodeDataProviderDropdown} from './node_data_provider_dropdown';
import {NodeDataProviderExtensionService} from './node_data_provider_extension_service';
import {NodeStyler} from './node_styler';
import {
  RunNdpExtensionDialog,
  RunNdpExtensionDialogData,
  RunNdpExtensionDialogResult,
} from './run_ndp_extension_dialog';

let ndpId = 0;

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
    MatProgressSpinnerModule,
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

  private readonly appService = inject(AppService);
  private readonly dialog = inject(MatDialog);
  private readonly nodeDataProviderExtensionService = inject(
    NodeDataProviderExtensionService,
  );
  private readonly extensionService = inject(ExtensionService);
  private readonly snackBar = inject(MatSnackBar);

  readonly processing = this.appService.processing;

  handleOpenNdpExtensionDialogClicked(extension: NodeDataProviderExtension) {
    // Show run task dialog.
    const data: RunNdpExtensionDialogData = {
      extension,
      extensionService: this.extensionService,
    };
    const dialogRef = this.dialog.open(RunNdpExtensionDialog, {
      width: '400px',
      data,
    });

    // Process result.
    dialogRef
      .afterClosed()
      .subscribe((result?: RunNdpExtensionDialogResult) => {
        if (result == null) {
          return;
        }

        this.runNdpExtension(extension, result.runName, result.configValues);
      });
  }

  private async runNdpExtension(
    extension: NodeDataProviderExtension,
    runName: string,
    configValues: Record<string, ConfigValue>,
  ) {
    const modelGraph = this.appService.getModelGraphFromSelectedPane();
    if (modelGraph == null) {
      return;
    }
    // Kick off ndp run.
    const runId = `ndp_run_${ndpId++}`;
    this.nodeDataProviderExtensionService.setRunStatus(
      extension.id,
      runId,
      true,
    );
    const {cmdResp, otherError} = await this.extensionService.runNdpExtension(
      extension /* extension */,
      modelGraph.modelPath ?? '' /* model path */,
      configValues /* config */,
    );
    // Show error if any.
    const error = cmdResp?.error ?? otherError ?? '';
    if (error !== '') {
      console.error(error);
      this.snackBar.open(error, 'Dismiss', {
        duration: 5000,
      });
    }
    // Add ndp if no error.
    else if (cmdResp?.result != null) {
      this.nodeDataProviderExtensionService.addRun(
        runId,
        runName,
        extension.id,
        modelGraph,
        {[modelGraph.id]: cmdResp.result},
      );
    }
    this.nodeDataProviderExtensionService.setRunStatus(
      extension.id,
      runId,
      false,
    );
  }

  get disableTitleTooltip(): boolean {
    return this.appService.testMode;
  }
}
