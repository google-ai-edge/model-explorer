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
  computed,
  inject,
  output,
  Signal,
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';
import {MatTooltipModule} from '@angular/material/tooltip';

import {MatDialog} from '@angular/material/dialog';
import {MatSlideToggleModule} from '@angular/material/slide-toggle';
import {
  ExtensionType,
  NodeDataProviderExtension,
  RunNdpExtensionData,
} from '../../common/types';
import {ExtensionService} from '../../services/extension_service';
import {AppService} from './app_service';
import {NodeDataProviderExtensionService} from './node_data_provider_extension_service';
import {
  RunNdpExtensionDialog,
  RunNdpExtensionDialogData,
  RunNdpExtensionDialogResult,
} from './run_ndp_extension_dialog';

/** The drop down menu for add per-node data. */
@Component({
  standalone: true,
  selector: 'ndp-extensions-panel',
  imports: [
    CommonModule,
    MatButtonModule,
    MatIconModule,
    MatProgressSpinnerModule,
    MatSlideToggleModule,
    MatTooltipModule,
  ],
  templateUrl: './ndp_extensions_panel.ng.html',
  styleUrls: ['./ndp_extensions_panel.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class NodeDataProviderExtensionsPanel {
  readonly onRunNdpExtension = output<RunNdpExtensionData>();

  protected readonly extensions: Signal<NodeDataProviderExtension[]> = computed(
    () => {
      // Filter the extensions based on the model file type and adapter id if
      // specified.
      const modelGraph = this.appService.getModelGraphFromSelectedPane();
      const adapterId = modelGraph?.adapterId ?? '';
      const modelFileExt = (modelGraph?.modelPath ?? '').split('.').pop() ?? '';
      const ndpExtensions = this.extensionService.extensions.filter(
        (ext) => ext.type === ExtensionType.NODE_DATA_PROVIDER,
      ) as NodeDataProviderExtension[];
      return ndpExtensions.filter((ext) => {
        if (ext.filter == null) {
          return true;
        }
        const modelFileExtsConditionMet =
          ext.filter.modelFileExts == null ||
          ext.filter.modelFileExts.includes(modelFileExt);
        const adapterIdsConditionMet =
          ext.filter.adapterIds == null ||
          ext.filter.adapterIds.includes(adapterId);
        return modelFileExtsConditionMet && adapterIdsConditionMet;
      });
    },
  );

  private readonly appService = inject(AppService);
  private readonly extensionService = inject(ExtensionService);
  private readonly nodeDataProviderExtensionService = inject(
    NodeDataProviderExtensionService,
  );
  private readonly dialog = inject(MatDialog);

  handleClickExtension(extension: NodeDataProviderExtension) {
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
      .subscribe(async (result?: RunNdpExtensionDialogResult) => {
        if (result == null) {
          return;
        }

        this.onRunNdpExtension.emit({
          extension,
          runName: result.runName,
          configValues: result.configValues,
        });
      });
  }

  isExtensionRunning(extensionId: string): boolean {
    const status = this.nodeDataProviderExtensionService.runStatus();
    if (status[extensionId] == null) {
      return false;
    }
    return Object.values(status[extensionId]).some((running) => running);
  }
}
