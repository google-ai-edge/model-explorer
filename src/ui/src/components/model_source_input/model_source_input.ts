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

import {
  ConnectedPosition,
  Overlay,
  OverlayConfig,
  OverlayRef,
  OverlaySizeConfig,
} from '@angular/cdk/overlay';
import {ComponentPortal} from '@angular/cdk/portal';
import {CommonModule} from '@angular/common';
import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  computed,
  DestroyRef,
  ElementRef,
  Inject,
  signal,
  ViewChild,
  ViewContainerRef,
} from '@angular/core';
import {takeUntilDestroyed} from '@angular/core/rxjs-interop';
import {FormControl, ReactiveFormsModule} from '@angular/forms';
import {
  MatAutocompleteModule,
  MatAutocompleteSelectedEvent,
  MatAutocompleteTrigger,
} from '@angular/material/autocomplete';
import {MatButtonModule} from '@angular/material/button';
import {MatCheckboxModule} from '@angular/material/checkbox';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatIconModule} from '@angular/material/icon';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';
import {MatSelectModule} from '@angular/material/select';
import {MatTooltipModule} from '@angular/material/tooltip';

import {
  DATA_NEXUS_MODEL_SOURCE_PREFIX,
  GRAPHS_MODEL_SOURCE_PREFIX,
} from '../../common/consts';
import {IS_EXTERNAL} from '../../common/flags';
import {type ModelLoaderServiceInterface} from '../../common/model_loader_service_interface';
import {
  InternalAdapterExtId,
  ModelItem,
  ModelItemStatus,
  ModelItemType,
} from '../../common/types';
import {
  getElectronApi,
  INTERNAL_COLAB,
  isInternalStoragePath,
} from '../../common/utils';
import {AdapterExtensionService} from '../../services/adapter_extension_service';
import {ExtensionService} from '../../services/extension_service';
import {ModelSource, UrlService} from '../../services/url_service';
import {Bubble} from '../bubble/bubble';
import {LocalStorageService} from '../visualizer/local_storage_service';

import {AdapterSelectorPanel} from './adapter_selector_panel';
import {getAdapterCandidates} from './utils';

interface SavedModelPath {
  path: string;
  ts: number;
}

const MAX_MODELS_COUNT = 10;
const SAVED_MODEL_PATHS_KEY = 'model_explorer_model_paths';
const MAX_SAVED_MODEL_PATHS_COUNT = 50;

/**
 * The component where users enter the source of the model.
 */
@Component({
  selector: 'model-source-input',
  standalone: true,
  imports: [
    Bubble,
    CommonModule,
    MatAutocompleteModule,
    MatButtonModule,
    MatCheckboxModule,
    MatFormFieldModule,
    MatProgressSpinnerModule,
    MatSelectModule,
    MatTooltipModule,
    MatIconModule,
    ReactiveFormsModule,
  ],
  templateUrl: './model_source_input.ng.html',
  styleUrls: ['./model_source_input.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ModelSourceInput {
  @ViewChild('modelPathInput') modelPathInput!: ElementRef<HTMLInputElement>;
  @ViewChild(MatAutocompleteTrigger)
  matAutocompleteTrigger?: MatAutocompleteTrigger;

  curFilePath = new FormControl<string>('');

  modelItems: ModelItem[] = [];

  modelInputAutocompleteOptions: SavedModelPath[] = [];

  filteredModelInputAutocompleteOptions: string[] = [];

  readonly ModelItemStatus = ModelItemStatus;

  readonly modelFormatHelpPopupSize: OverlaySizeConfig = {
    maxWidth: 400,
    minHeight: 0,
  };

  readonly errorInfoPopupSize: OverlaySizeConfig = {
    minHeight: 0,
  };

  readonly adapterHelpPopupPosition: ConnectedPosition[] = [
    {
      originX: 'start',
      originY: 'top',
      overlayX: 'start',
      overlayY: 'bottom',
      offsetY: -4,
    },
  ];

  readonly loading = signal<boolean>(false);
  readonly hasUploadedModels = signal<boolean>(false);
  readonly internalColab = INTERNAL_COLAB;
  readonly customExtensions = computed(() => {
    if (this.extensionService.loading()) {
      return [];
    }
    return this.extensionService.getCustomExtensions();
  });

  private portal: ComponentPortal<AdapterSelectorPanel> | null = null;

  constructor(
    private readonly changeDetectorRef: ChangeDetectorRef,
    private readonly adapterExtensionService: AdapterExtensionService,
    private readonly destroyRef: DestroyRef,
    private readonly extensionService: ExtensionService,
    private readonly localStorageService: LocalStorageService,
    @Inject('ModelLoaderService')
    private readonly modelLoaderService: ModelLoaderServiceInterface,
    private readonly overlay: Overlay,
    private readonly urlService: UrlService,
    private readonly viewContainerRef: ViewContainerRef,
  ) {
    // Filter autocomplete options based on user's input.
    this.curFilePath.valueChanges
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((value) => {
        this.updateFilteredAutocompleteOptions();
      });

    // Load saved model paths from local storage.
    this.modelInputAutocompleteOptions =
      this.loadSavedModelPathsForAutocomplete();
    this.updateFilteredAutocompleteOptions();
  }

  /** Called by homepage to start processing model sources restored from URL. */
  startProcessingModelSource(modelSources: ModelSource[]) {
    // Populate model items with the data in modelSources.
    this.modelItems = modelSources.map((modelSource) => {
      // This model is the processed json graphs loaded from server
      // (external only).
      if (modelSource.url.startsWith(GRAPHS_MODEL_SOURCE_PREFIX)) {
        const ext = this.adapterExtensionService.getExtensionById(
          InternalAdapterExtId.JSON_LOADER,
        );
        const adapterCandidates = ext == null ? [] : [ext];
        return {
          path: modelSource.url,
          type: ModelItemType.GRAPH_JSONS_FROM_SERVER,
          status: signal<ModelItemStatus>(ModelItemStatus.NOT_STARTED),
          selected: adapterCandidates.length > 0,
          adapterCandidates,
          selectedAdapter: ext,
        };
      }
      // Data nexus model source.
      else if (modelSource.url.startsWith(DATA_NEXUS_MODEL_SOURCE_PREFIX)) {
        const ext = this.adapterExtensionService.getExtensionById(
          InternalAdapterExtId.DATA_NEXUS,
        );
        const adapterCandidates = ext == null ? [] : [ext];
        return {
          path: modelSource.url,
          type: ModelItemType.DATA_NEXUS,
          status: signal<ModelItemStatus>(ModelItemStatus.NOT_STARTED),
          selected: true,
          adapterCandidates,
          selectedAdapter: ext,
        };
      }
      // Other typical use cases.
      else {
        const adapterCandidates = getAdapterCandidates(
          modelSource.url,
          this.adapterExtensionService,
          IS_EXTERNAL,
        );

        // Try to get the adapter extension from the id encoded in the url.
        let selectedAdapter = this.adapterExtensionService.getExtensionById(
          modelSource.converterId || modelSource.adapterId || '',
        );
        if (!selectedAdapter) {
          selectedAdapter =
            adapterCandidates.length > 0 ? adapterCandidates[0] : undefined;
        }
        return {
          path: modelSource.url,
          type: IS_EXTERNAL ? ModelItemType.FILE_PATH : ModelItemType.REMOTE,
          status: signal<ModelItemStatus>(ModelItemStatus.NOT_STARTED),
          selected: adapterCandidates.length > 0,
          adapterCandidates,
          selectedAdapter,
        };
      }
    });
    this.changeDetectorRef.detectChanges();

    // Start processing.
    if (this.modelItems.some((item) => item.selected)) {
      this.handleClickViewSelectedModels();
    }
  }

  /**
   * Called by homepage when user starts external ME server with url parameters.
   */
  startWithUrlEncodedData(hasLoadGraphsJson: boolean, modelPaths: string[]) {
    // Add a special item for graphs json loaded from server.
    if (hasLoadGraphsJson) {
      const ext = this.adapterExtensionService.getExtensionById(
        InternalAdapterExtId.JSON_LOADER,
      );
      if (!ext) {
        return;
      }
      this.addModelItems([
        {
          path: '<Graphs imported from server>',
          type: ModelItemType.GRAPH_JSONS_FROM_SERVER,
          status: signal<ModelItemStatus>(ModelItemStatus.NOT_STARTED),
          selected: true,
          adapterCandidates: [ext],
          selectedAdapter: ext,
        },
      ]);
    }
    if (modelPaths.length > 0) {
      const modelItems: ModelItem[] = modelPaths.map((url) => {
        const adapterCandidates = getAdapterCandidates(
          url,
          this.adapterExtensionService,
          IS_EXTERNAL,
        );
        return {
          path: url,
          type: ModelItemType.FILE_PATH,
          status: signal<ModelItemStatus>(ModelItemStatus.NOT_STARTED),
          selected: adapterCandidates.length > 0,
          adapterCandidates,
          selectedAdapter:
            adapterCandidates.length > 0 ? adapterCandidates[0] : undefined,
        };
      });
      this.addModelItems(modelItems);

      // Add the paths to autocomplete history.
      this.addPathsToAutocompleteHistory(modelPaths);
      this.updateFilteredAutocompleteOptions();
    }
    this.changeDetectorRef.detectChanges();

    // Start processing immediately if we only have graphs json.
    if (hasLoadGraphsJson && modelPaths.length === 0) {
      this.handleClickViewSelectedModels();
    }
  }

  async handleClickAddEnteredModelPath() {
    // This is needed to make things work when clicking the "add" button.
    await new Promise((resolve) => {
      setTimeout(resolve);
    });

    const modelPath = this.curFilePath.value;
    if (modelPath == null) {
      return;
    }
    const modelItems: ModelItem[] = modelPath
      .trim()
      .split(',')
      .filter((url) => url.trim() !== '')
      .map((url) => {
        const adapterCandidates = getAdapterCandidates(
          url,
          this.adapterExtensionService,
          IS_EXTERNAL,
        );
        return {
          path: url,
          type: this.isInternal
            ? ModelItemType.REMOTE
            : ModelItemType.FILE_PATH,
          status: signal<ModelItemStatus>(ModelItemStatus.NOT_STARTED),
          selected: adapterCandidates.length > 0,
          adapterCandidates,
          // TODO: store the adapter selection in local storage and load
          // it here.
          selectedAdapter:
            adapterCandidates.length > 0 ? adapterCandidates[0] : undefined,
        };
      });
    this.addModelItems(modelItems);

    // Save to autocomplete history.
    this.addPathsToAutocompleteHistory(
      modelItems.map((modelItem) => modelItem.path),
    );

    // Clear input and close autocomplete.
    this.curFilePath.setValue('');
    setTimeout(() => {
      this.matAutocompleteTrigger?.closePanel();
      this.modelPathInput.nativeElement.blur();
    });
  }

  handleAutocompleteOptionSelected(event: MatAutocompleteSelectedEvent) {
    if (this.disableAddEnteredModelPathButton) {
      return;
    }
    event.option.deselect();
    this.handleClickAddEnteredModelPath();
  }

  handleModelSelectionChanged(item: ModelItem, checked: boolean) {
    item.selected = checked;
  }

  handleDeleteModel(index: number) {
    this.modelItems.splice(index, 1);
    this.changeDetectorRef.markForCheck();
  }

  handleDeselectAllModels() {
    for (const item of this.modelItems) {
      item.selected = false;
    }
  }

  handleEditAutocompleteModelPath(event: MouseEvent, index: number) {
    event.stopPropagation();

    this.curFilePath.setValue(this.modelInputAutocompleteOptions[index].path);
    this.modelPathInput.nativeElement.focus();
  }

  handleClickDeleteAutocompleteModelPath(event: MouseEvent, index: number) {
    event.stopPropagation();

    this.modelInputAutocompleteOptions.splice(index, 1);
    this.updateFilteredAutocompleteOptions();
    this.localStorageService.setItem(
      SAVED_MODEL_PATHS_KEY,
      JSON.stringify(this.modelInputAutocompleteOptions),
    );
  }

  handleClickUpload(input: HTMLInputElement) {
    const filesList = input.files;
    if (!filesList) {
      return;
    }
    const files: File[] = [];
    for (let i = 0; i < filesList.length; i++) {
      const file = filesList.item(i);
      if (!file) {
        continue;
      }
      files.push(file);
    }
    this.addFiles(files);
    input.value = '';
  }

  addFiles(files: File[]) {
    const modelItems: ModelItem[] = [];
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const meElectronApi = getElectronApi();
      let filePath = '';
      // When running in electron app, a `pathForFile` api is exposed from its
      // preload script to the renderer process (here) that can be used to get
      // the absolute path of a file.
      if (meElectronApi) {
        const pathForFileFn = meElectronApi['pathForFile'];
        filePath = pathForFileFn?.(file) || '';
      }
      const adapterCandidates = getAdapterCandidates(
        file.name,
        this.adapterExtensionService,
        IS_EXTERNAL,
      );
      if (filePath !== '') {
        modelItems.push({
          path: filePath,
          type: this.isInternal
            ? ModelItemType.REMOTE
            : ModelItemType.FILE_PATH,
          status: signal<ModelItemStatus>(ModelItemStatus.NOT_STARTED),
          selected: adapterCandidates.length > 0,
          adapterCandidates,
          // TODO: store the adapter selection in local storage and load
          // it here.
          selectedAdapter:
            adapterCandidates.length > 0 ? adapterCandidates[0] : undefined,
        });
      } else {
        modelItems.push({
          path: file.name,
          type: ModelItemType.LOCAL,
          status: signal<ModelItemStatus>(ModelItemStatus.NOT_STARTED),
          selected: adapterCandidates.length > 0,
          file,
          adapterCandidates,
          // TODO: store the adapter selection in local storage and load it here.
          selectedAdapter:
            adapterCandidates.length > 0 ? adapterCandidates[0] : undefined,
        });
      }
    }
    this.addModelItems(modelItems);
  }

  handleClickOpenAdapterDropdown(modelItem: ModelItem, anchor: HTMLElement) {
    const curSelectedAdapterId = modelItem.selectedAdapter?.id;
    const overlayRef = this.createOverlay(anchor);
    const ref = overlayRef.attach(this.portal!);
    ref.instance.selectedAdapter = modelItem.selectedAdapter;
    ref.instance.candidates = modelItem.adapterCandidates || [];
    ref.instance.onClose.subscribe((selectedCandidate) => {
      overlayRef.dispose();

      // Clear error and select the item when the item previously has errors and
      // user changes the adapter.
      if (
        selectedCandidate?.id !== curSelectedAdapterId &&
        modelItem.errorMessage != null
      ) {
        modelItem.status.set(ModelItemStatus.NOT_STARTED);
        modelItem.errorMessage = undefined;
        modelItem.selected = true;
      }
      modelItem.selectedAdapter = selectedCandidate;
      this.changeDetectorRef.markForCheck();
    });
  }

  handleClickViewSelectedModels() {
    this.loading.set(true);
    const selectedModelItems = this.modelItems.filter((item) => item.selected);
    this.modelLoaderService.loadModels(selectedModelItems).then(() => {
      this.loading.set(false);
    });

    this.urlService.setUiState(undefined);
    this.urlService.setModels(
      selectedModelItems
        .filter(
          (modelItem) =>
            modelItem.type === ModelItemType.REMOTE ||
            modelItem.type === ModelItemType.GRAPH_JSONS_FROM_SERVER ||
            modelItem.type === ModelItemType.FILE_PATH,
        )
        .map((modelItem) => {
          return {
            url: modelItem.path,
            adapterId: modelItem.selectedAdapter?.id,
          };
        }),
    );

    this.hasUploadedModels.set(
      selectedModelItems.some((item) => item.type === ModelItemType.LOCAL),
    );
  }

  trackByModelData(index: number, item: ModelItem): string {
    return `${item.path}_${item.file?.size}_${item.file?.lastModified}`;
  }

  isNotStarted(item: ModelItem): boolean {
    return item.status() === ModelItemStatus.NOT_STARTED;
  }

  getSelectedAdapterName(modelItem: ModelItem): string {
    if (modelItem.selectedAdapter == null) {
      return '?';
    }
    const isDefault =
      (modelItem.adapterCandidates || []).indexOf(modelItem.selectedAdapter) ===
      0;
    return isDefault ? 'Default' : modelItem.selectedAdapter?.name || '?';
  }

  hasSupportedAdapter(modelItem: ModelItem): boolean {
    return (modelItem.adapterCandidates || []).length > 0;
  }

  hasMultipleSupportedAdapters(modelItem: ModelItem): boolean {
    return (modelItem.adapterCandidates || []).length > 1;
  }

  showSpinner(modelItem: ModelItem): boolean {
    return (
      modelItem.status() === ModelItemStatus.PROCESSING ||
      modelItem.status() === ModelItemStatus.UPLOADING
    );
  }

  hasError(modelItem: ModelItem): boolean {
    return modelItem.status() === ModelItemStatus.ERROR;
  }

  getModelItemStatusString(modelItem: ModelItem): string {
    const status = modelItem.status();
    if (IS_EXTERNAL && status === ModelItemStatus.UPLOADING) {
      return 'Processing';
    }
    return status;
  }

  linkifyErrorMessage(modelItem: ModelItem): string {
    const errorMessage: string = modelItem.errorMessage || '';
    const parts = errorMessage.split(' ');
    return parts
      .map((part) => {
        // TODO(jingjin): Add support for other types of links.
        if (part.startsWith('go/')) {
          return `<a href='http://${part}' target='_blank'>${part}</a>`;
        } else {
          return part;
        }
      })
      .join(' ');
  }

  get disableAddEnteredModelPathButton(): boolean {
    if (this.hasReachedMaxModelsCount) {
      return true;
    }

    if (this.isInternal) {
      const path = (this.curFilePath.value || '').toLowerCase().trim();
      return (
        (path !== '' &&
          !isInternalStoragePath(path) &&
          !path.startsWith('http')) ||
        path === ''
      );
    } else {
      const path = (this.curFilePath.value || '').trim();
      return (
        path === '' ||
        (path !== '' && !path.startsWith('/') && !path.startsWith('~'))
      );
    }
  }

  get isInternal(): boolean {
    return !IS_EXTERNAL;
  }

  get isExternal(): boolean {
    return IS_EXTERNAL;
  }

  get selectedModelsCount(): number {
    return this.modelItems.filter((item) => item.selected).length;
  }

  get hasReachedMaxModelsCount(): boolean {
    return this.modelItems.length === MAX_MODELS_COUNT;
  }

  get modelPathInputPlaceholder(): string {
    return 'Absolute file paths (recommended for large models)';
  }

  private updateFilteredAutocompleteOptions() {
    const value = this.curFilePath.value;
    const filterValue = (value || '').toLowerCase();
    this.filteredModelInputAutocompleteOptions =
      this.modelInputAutocompleteOptions
        .filter((item) => item.path.toLowerCase().includes(filterValue))
        .map((item) => item.path);
    this.changeDetectorRef.markForCheck();
  }

  private addPathsToAutocompleteHistory(paths: string[]) {
    // Store in local storage.
    //
    // Get the current paths.
    const curSavedPaths = this.loadSavedModelPathsForAutocomplete();
    for (const path of paths) {
      // Find the currently entered model path.
      const index = curSavedPaths.findIndex((item) => item.path === path);
      // If it is in the current saved paths, Move it to the top.
      if (index >= 0) {
        const item = curSavedPaths.splice(index, 1);
        if (item && item.length > 0) {
          item[0].ts = Date.now();
          curSavedPaths.unshift(item[0]);
        }
      }
      // If not, add it to the top.
      else {
        curSavedPaths.unshift({
          path,
          ts: Date.now(),
        });
      }
    }

    // Keep list under a pre-defined size by removing items from the end.
    if (curSavedPaths.length > MAX_SAVED_MODEL_PATHS_COUNT) {
      curSavedPaths.splice(MAX_SAVED_MODEL_PATHS_COUNT);
    }

    // Save the list back to local storage.
    this.localStorageService.setItem(
      SAVED_MODEL_PATHS_KEY,
      JSON.stringify(curSavedPaths),
    );

    // Update current autocomplete options.
    this.modelInputAutocompleteOptions = curSavedPaths;
  }

  private addModelItems(modelItems: ModelItem[]) {
    // Dedup first.
    const filteredModelItems = modelItems.filter(
      (item) =>
        this.modelItems.find(
          (curItem) =>
            curItem.path === item.path &&
            curItem.file?.size === item.file?.size &&
            curItem.file?.lastModified === item.file?.lastModified,
        ) == null,
    );
    this.modelItems.push(...filteredModelItems);

    // Keep the number of models under the threshold.
    if (this.modelItems.length > MAX_MODELS_COUNT) {
      this.modelItems.splice(MAX_MODELS_COUNT);
    }

    this.changeDetectorRef.markForCheck();
  }

  private loadSavedModelPathsForAutocomplete(): SavedModelPath[] {
    return JSON.parse(
      this.localStorageService.getItem(SAVED_MODEL_PATHS_KEY) || '[]',
    ) as SavedModelPath[];
  }

  private createOverlay(ele: HTMLElement): OverlayRef {
    const config = new OverlayConfig({
      positionStrategy: this.overlay
        .position()
        .flexibleConnectedTo(ele)
        .withPositions([
          {
            originX: 'start',
            originY: 'bottom',
            overlayX: 'start',
            overlayY: 'top',
          },
        ]),
      hasBackdrop: true,
      backdropClass: 'cdk-overlay-transparent-backdrop',
      scrollStrategy: this.overlay.scrollStrategies.reposition(),
      maxWidth: '380px',
      panelClass: 'graph-selector-panel',
    });
    const overlayRef = this.overlay.create(config);
    this.portal = new ComponentPortal(
      AdapterSelectorPanel,
      this.viewContainerRef,
    );
    overlayRef.backdropClick().subscribe(() => {
      overlayRef.dispose();
    });
    return overlayRef;
  }
}
