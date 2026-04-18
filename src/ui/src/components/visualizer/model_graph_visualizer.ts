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
import {ChangeDetectionStrategy, Component} from '@angular/core';

import {LOCAL_STORAGE_SERVICE_INJECTION_TOKEN} from '../../common/local_storage_service_interface';
import {AppService} from './app_service';
import {BenchmarkRunner} from './benchmark_runner';
import {LocalStorageService} from './local_storage_service';
import {ModelGraphVisualizerBase} from './model_graph_visualizer_base';
import {NodeDataProviderExtensionService} from './node_data_provider_extension_service';
import {NodeStylerService} from './node_styler_service';
import {SplitPanesContainer} from './split_panes_container';
import {SyncNavigationService} from './sync_navigation_service';
import {TitleBar} from './title_bar';
import {UiStateService} from './ui_state_service';
import {VisualizerThemeService} from './visualizer_theme_service';
import {WorkerService} from './worker_service';

/** The main model graph visualizer component. */
@Component({
  standalone: true,
  selector: 'model-graph-visualizer',
  imports: [BenchmarkRunner, CommonModule, TitleBar, SplitPanesContainer],
  templateUrl: './model_graph_visualizer.ng.html',
  styleUrls: ['./model_graph_visualizer.scss'],
  host: {
    '[attr.data-metheme]': 'theme()',
  },
  providers: [
    AppService,
    NodeDataProviderExtensionService,
    NodeStylerService,
    SyncNavigationService,
    UiStateService,
    VisualizerThemeService,
    WorkerService,
    {
      provide: LOCAL_STORAGE_SERVICE_INJECTION_TOKEN,
      useClass: LocalStorageService,
    },
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ModelGraphVisualizer extends ModelGraphVisualizerBase {}
