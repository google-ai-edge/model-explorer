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
  AfterViewInit,
  Component,
  computed,
  effect,
  HostListener,
  Inject,
  Signal,
  signal,
  ViewChild,
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatDialog, MatDialogModule} from '@angular/material/dialog';
import {MatIconModule} from '@angular/material/icon';
import {MatMenuModule} from '@angular/material/menu';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';
import {MatSnackBar, MatSnackBarModule} from '@angular/material/snack-bar';
import {MatTooltipModule} from '@angular/material/tooltip';
import {ActivatedRoute, Router} from '@angular/router';

import {type ModelLoaderServiceInterface} from '../../common/model_loader_service_interface';
import {ExtensionService} from '../../services/extension_service';
import {GaEventType, GaService} from '../../services/ga_service';
import {ServerDirectorService} from '../../services/server_director_service';
import {
  SETTING_ARTIFACIAL_LAYER_NODE_COUNT_THRESHOLD,
  SETTING_DISALLOW_VERTICAL_EDGE_LABELS,
  SETTING_EDGE_COLOR,
  SETTING_EDGE_LABEL_FONT_SIZE,
  SETTING_HIDE_EMPTY_NODE_DATA_ENTRIES,
  SETTING_HIDE_OP_NODES_WITH_LABELS,
  SETTING_HIGHLIGHT_LAYER_NODE_INPUTS_OUTPUTS,
  SETTING_KEEP_LAYERS_WITH_A_SINGLE_CHILD,
  SETTING_MAX_CONST_ELEMENT_COUNT_LIMIT,
  SETTING_SHOW_OP_NODE_OUT_OF_LAYER_EDGES_WITHOUT_SELECTING,
  SETTING_SHOW_SIDE_PANEL_ON_NODE_SELECTION,
  SettingKey,
  SettingsService,
} from '../../services/settings_service';
import {UrlService} from '../../services/url_service';
import {ModelSourceInput} from '../model_source_input/model_source_input';
import {OpenInNewTabButton} from '../open_in_new_tab_button/open_in_new_tab_button';
import {OpenSourceLibsDialog} from '../open_source_libs_dialog/open_source_libs_dialog';
import {SettingsDialog} from '../settings_dialog/settings_dialog';
import {
  ModelGraphProcessedEvent,
  SyncNavigationModeChangedEvent,
} from '../visualizer/common/types';
import {VisualizerConfig} from '../visualizer/common/visualizer_config';
import {VisualizerUiState} from '../visualizer/common/visualizer_ui_state';
import {Logo} from '../visualizer/logo';
import {ModelGraphVisualizer} from '../visualizer/model_graph_visualizer';
import {
  NewVersionChip,
  NewVersionService,
} from '../visualizer/new_version_chip';
import {ThreejsService} from '../visualizer/threejs_service';
import {WelcomeCard} from '../welcome_card/welcome_card';

/**
 * The component for the home page.
 */
@Component({
  selector: 'home-page',
  standalone: true,
  imports: [
    CommonModule,
    Logo,
    MatButtonModule,
    MatDialogModule,
    MatIconModule,
    MatMenuModule,
    MatProgressSpinnerModule,
    MatSnackBarModule,
    MatTooltipModule,
    ModelGraphVisualizer,
    ModelSourceInput,
    NewVersionChip,
    OpenInNewTabButton,
    WelcomeCard,
  ],
  templateUrl: './home_page.ng.html',
  styleUrls: ['./home_page.scss'],
})
export class HomePage implements AfterViewInit {
  @ViewChild('modelSourceInput', {static: false})
  modelSourceInput!: ModelSourceInput;
  @ViewChild('modelGraphVisualizer', {static: false})
  modelGraphVisualizer?: ModelGraphVisualizer;

  readonly loadingExtensions;
  readonly loadedGraphCollections;
  runningVersion = computed(() => this.newVersionService.info().runningVersion);

  initialUiState?: VisualizerUiState;
  dismissWelcomeCard = false;
  dragOver = false;
  benchmark = false;
  remoteNodeDataPaths: string[] = [];
  remoteNodeDataTargetModels: string[] = [];
  syncNavigation?: SyncNavigationModeChangedEvent;
  hasUploadedModels = signal<boolean>(false);
  shareButtonTooltip: Signal<string> = signal<string>('');

  private readonly remoteProcessedNodeDataTargetModels = new Set<string>();

  constructor(
    private readonly dialog: MatDialog,
    private readonly extensionService: ExtensionService,
    private readonly gaService: GaService,
    @Inject('ModelLoaderService')
    private readonly modelLoaderService: ModelLoaderServiceInterface,
    private readonly newVersionService: NewVersionService,
    private readonly route: ActivatedRoute,
    private readonly router: Router,
    private readonly serverDirectorService: ServerDirectorService,
    private readonly settingsService: SettingsService,
    private readonly snackBar: MatSnackBar,
    readonly threejsService: ThreejsService,
    private readonly urlService: UrlService,
  ) {
    this.serverDirectorService.init();

    this.loadingExtensions = this.extensionService.loading;
    this.loadedGraphCollections =
      this.modelLoaderService.loadedGraphCollections;
    this.initialUiState = this.urlService.getUiState();

    effect(() => {
      const loading = this.extensionService.loading();
      if (!loading) {
        setTimeout(() => {
          this.handleExtensionLoaded();
        });
      }
    });

    effect(() => {
      // Push a dummy history state so that we can handle user clicking the
      // back button.
      if (this.loadedGraphCollections() != null) {
        window.history.pushState({ts: Date.now()}, '');
      }
    });

    // Check if we are in benchmark mode.
    const params = new URLSearchParams(document.location.search);
    this.benchmark = params.get('benchmark') === '1';

    // Remote node data paths encoded in the url.
    this.remoteNodeDataPaths = this.urlService.getNodeDataSources();
    this.remoteNodeDataTargetModels = this.urlService.getNodeDataTargets();

    // Sync navigation.
    this.syncNavigation = this.urlService.getSyncNavigation();
  }

  ngAfterViewInit() {
    if (this.modelSourceInput) {
      this.hasUploadedModels = this.modelSourceInput.hasUploadedModels;
      this.shareButtonTooltip = computed<string>(() =>
        this.hasUploadedModels()
          ? 'Share is not available for uploaded models'
          : 'Share',
      );
    }
  }

  @HostListener('window:popstate', ['$event'])
  handlePopState(event: Event) {
    // Set loaded graph collection to null when user clicks the back button
    // in browser, which will destroy the model graph visualizer component
    // and show the home page.
    this.loadedGraphCollections.set(undefined);

    // Clear ui state.
    this.initialUiState = undefined;

    // Clear query parameters.
    setTimeout(() => {
      this.router.navigate([], {
        queryParams: {},
        // Use '' as the params handling method so that the whole query params
        // string will be replaced by the current content of 'queryParams'.
        queryParamsHandling: '',
        replaceUrl: true,
      });
    });
  }

  handleDragOver(event: Event) {
    if (this.loadedGraphCollections() == null) {
      this.dragOver = true;
    }
    event.preventDefault();
  }

  handleDragLeave() {
    this.dragOver = false;
  }

  handleDrop(event: DragEvent) {
    event.preventDefault();
    this.dragOver = false;

    const files: File[] = [];
    if (event.dataTransfer?.items) {
      // Use DataTransferItemList interface to access the file(s)
      Array.from(event.dataTransfer.items).forEach((item, i) => {
        // If dropped items aren't files, reject them
        if (item.kind === 'file') {
          const file = item.getAsFile();
          if (file) {
            files.push(file);
          }
        }
      });
    } else {
      // Use DataTransfer interface to access the file(s)
      files.push(...Array.from(event.dataTransfer?.files || []));
    }
    this.modelSourceInput.addFiles(files);
  }

  handleClickTitle(refresh = false) {
    if (refresh) {
      this.router.navigate(['/']).then(() => {
        window.location.reload();
      });
    } else {
      // Back to the home page. See handlePopState above.
      window.history.back();
    }
  }

  handleClickSettings() {
    this.dialog.open(SettingsDialog, {});
  }

  handleClickDismissWelcomeCard() {
    this.settingsService.saveBooleanValue(false, SettingKey.SHOW_WELCOME_CARD);
  }

  handleUiStateChanged(uiState: VisualizerUiState) {
    this.urlService.setUiState(uiState);
  }

  handleModelGraphProcessed(event: ModelGraphProcessedEvent) {
    const modelName = event.modelGraph.collectionLabel;
    const processed = this.remoteProcessedNodeDataTargetModels.has(modelName);
    if (
      this.remoteNodeDataPaths &&
      this.remoteNodeDataPaths.length > 0 &&
      !processed
    ) {
      const curNodeDataPaths: string[] = [];
      for (let i = 0; i < this.remoteNodeDataPaths.length; i++) {
        const path = this.remoteNodeDataPaths[i];
        const targetModel = this.remoteNodeDataTargetModels[i] || '';
        if (targetModel === '' || targetModel === modelName) {
          curNodeDataPaths.push(path);
        }
      }
      this.modelGraphVisualizer?.loadRemoteNodeDataPaths(
        curNodeDataPaths,
        event.modelGraph,
      );
      this.remoteProcessedNodeDataTargetModels.add(modelName);
    }

    if (this.syncNavigation) {
      this.modelGraphVisualizer?.syncNavigationService.loadSyncNavigationDataFromEvent(
        this.syncNavigation,
      );
    }
  }

  handleRemoteNodeDataPathsChanged(paths: string[]) {
    this.urlService.setNodeDataSources(paths);
  }

  handleSyncNavigationModeChanged(event: SyncNavigationModeChangedEvent) {
    this.urlService.setSyncNavigation(event);
  }

  handleClickShowThirdPartyLibraries() {
    this.dialog.open(OpenSourceLibsDialog, {});
  }

  get showWelcomeCard(): boolean {
    const setting = this.settingsService.getSettingByKey(
      SettingKey.SHOW_WELCOME_CARD,
    );
    if (setting) {
      return this.settingsService.getBooleanValue(setting);
    }
    return true;
  }

  get curConfig(): VisualizerConfig {
    const showTfliteConsts =
      this.route.snapshot.queryParams['show_tflite_consts'] === '1';
    return {
      nodeLabelsToHide: this.settingsService
        .getStringValue(SETTING_HIDE_OP_NODES_WITH_LABELS)
        .split(',')
        .map((s) => s.trim())
        .filter(
          (s) =>
            s !== '' &&
            (!showTfliteConsts ||
              (showTfliteConsts &&
                s !== 'pseudo_const' &&
                s !== 'pseudo_qconst')),
        ),
      artificialLayerNodeCountThreshold: this.settingsService.getNumberValue(
        SETTING_ARTIFACIAL_LAYER_NODE_COUNT_THRESHOLD,
      ),
      edgeLabelFontSize: this.settingsService.getNumberValue(
        SETTING_EDGE_LABEL_FONT_SIZE,
      ),
      edgeColor: this.settingsService.getStringValue(SETTING_EDGE_COLOR),
      maxConstValueCount: this.settingsService.getNumberValue(
        SETTING_MAX_CONST_ELEMENT_COUNT_LIMIT,
      ),
      disallowVerticalEdgeLabels: this.settingsService.getBooleanValue(
        SETTING_DISALLOW_VERTICAL_EDGE_LABELS,
      ),
      enableSubgraphSelection: this.urlService.enableSubgraphSelection,
      enableExportToResource: this.urlService.enableExportToResource,
      enableExportSelectedNodes: this.urlService.enableExportSelectedNodes,
      exportSelectedNodesButtonLabel:
        this.urlService.exportSelectedNodesButtonLabel,
      exportSelectedNodesButtonIcon:
        this.urlService.exportSelectedNodesButtonIcon,
      keepLayersWithASingleChild: this.settingsService.getBooleanValue(
        SETTING_KEEP_LAYERS_WITH_A_SINGLE_CHILD,
      ),
      showOpNodeOutOfLayerEdgesWithoutSelecting:
        this.settingsService.getBooleanValue(
          SETTING_SHOW_OP_NODE_OUT_OF_LAYER_EDGES_WITHOUT_SELECTING,
        ),
      highlightLayerNodeInputsOutputs: this.settingsService.getBooleanValue(
        SETTING_HIGHLIGHT_LAYER_NODE_INPUTS_OUTPUTS,
      ),
      hideEmptyNodeDataEntries: this.settingsService.getBooleanValue(
        SETTING_HIDE_EMPTY_NODE_DATA_ENTRIES,
      ),
      showSidePanelOnNodeSelection: this.settingsService.getBooleanValue(
        SETTING_SHOW_SIDE_PANEL_ON_NODE_SELECTION,
      ),
      nodeAttrsToHide: this.urlService.nodeAttributesToHide,
    };
  }

  private handleExtensionLoaded() {
    // Load models encoded in url.
    const models = this.urlService.getModels();
    if (models != null && models.length > 0) {
      this.modelSourceInput.startProcessingModelSource(models);
    } else {
      this.initialUiState = undefined;
    }
  }
}
