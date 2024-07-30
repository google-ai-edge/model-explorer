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
  ViewChild,
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';
import {MatTooltipModule} from '@angular/material/tooltip';

import {Bubble} from '../bubble/bubble';
import {BubbleClick} from '../bubble/bubble_click';

import {AppService} from './app_service';
import {ModelGraph} from './common/model_graph';
import {NodeDataProviderData} from './common/types';
import {genUid} from './common/utils';
import {Extension} from './extension_service';
import {LocalStorageService} from './local_storage_service';
import {NodeDataProviderExtensionService} from './node_data_provider_extension_service';

/** The drop down menu for add per-node data. */
@Component({
  standalone: true,
  selector: 'node-data-provider-dropdown',
  imports: [
    Bubble,
    BubbleClick,
    CommonModule,
    MatButtonModule,
    MatIconModule,
    MatProgressSpinnerModule,
    MatTooltipModule,
  ],
  templateUrl: 'node_data_provider_dropdown.ng.html',
  styleUrls: ['./node_data_provider_dropdown.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class NodeDataProviderDropdown {
  @ViewChild(BubbleClick) dropdown?: BubbleClick;

  extensions: Extension[] = [];
  loadingExtensions = true;

  readonly helpPopupSize: OverlaySizeConfig = {
    minWidth: 0,
    minHeight: 0,
  };

  readonly dropdownSize: OverlaySizeConfig = {
    minWidth: 0,
    minHeight: 0,
    maxHeight: 500,
  };

  readonly remoteSourceLoading;

  constructor(
    private readonly appService: AppService,
    private readonly changeDetectorRef: ChangeDetectorRef,
    private readonly localStorageService: LocalStorageService,
    private readonly nodeDataProviderExtensionService: NodeDataProviderExtensionService,
  ) {
    this.remoteSourceLoading =
      this.nodeDataProviderExtensionService.remoteSourceLoading;
  }

  handleClickUpload(input: HTMLInputElement) {
    const files = input.files;
    if (!files) {
      return;
    }
    const modelGraph = this.appService.getModelGraphFromSelectedPane();
    if (!modelGraph) {
      return;
    }

    for (const file of Array.from(files)) {
      const fileReader = new FileReader();
      fileReader.onload = (event) => {
        const runId = genUid();
        try {
          const data = this.getNodeDataProviderData(
            event.target?.result as string,
            modelGraph,
          );

          this.nodeDataProviderExtensionService.addRun(
            runId,
            file.name,
            '',
            modelGraph,
            data,
          );
        } catch (e) {
          this.nodeDataProviderExtensionService.addRun(
            runId,
            file.name,
            '',
            modelGraph,
          );
          this.nodeDataProviderExtensionService.updateRunResults(
            runId,
            {[modelGraph.id]: {results: {}}},
            modelGraph,
            `Failed to process JSON file. ${e}`,
          );
        } finally {
          this.dropdown?.closeDialog();
        }
      };
      fileReader.readAsText(file);
    }
    input.value = '';
  }

  private getNodeDataProviderData(str: string, modelGraph: ModelGraph) {
    // tslint:disable-next-line:no-any Allow arbitrary types.
    const jsonObj = JSON.parse(str) as any;
    let data: NodeDataProviderData = {};

    // The json file doesn't have graph id as key. Use the current graph
    // as the only key.
    if (jsonObj['results'] != null && jsonObj['results']['results'] == null) {
      if (modelGraph) {
        data[modelGraph.id] = jsonObj;
      }
    }
    // File with graph id as keys.
    else {
      data = jsonObj;
    }

    return data;
  }
}
