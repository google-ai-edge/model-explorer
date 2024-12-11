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

import {Injectable, signal} from '@angular/core';

import {GRAPHS_MODEL_SOURCE_PREFIX} from '../common/consts';
import {
  type AdapterConvertResponse,
  type AdapterExecuteResponse,
  type AdapterOverrideResponse,
  type AdapterStatusCheckResponse,
  type ExtensionCommand,
  type ExtensionResponse,
} from '../common/extension_command';
import {ModelLoaderServiceInterface, type ChangesPerGraphAndNode, type ChangesPerNode } from '../common/model_loader_service_interface';
import {
  InternalAdapterExtId,
  ModelItem,
  ModelItemStatus,
  ModelItemType,
} from '../common/types';
import {processJson, processUploadedJsonFile} from '../common/utils';
import {
  Graph,
  GraphCollection,
} from '../components/visualizer/common/input_graph';
import {processErrorMessage} from '../components/visualizer/common/utils';

import {ExtensionService} from './extension_service';
import {SettingsService} from './settings_service';

const UPLOAD_API_PATH = '/apipost/v1/upload';
const LOAD_GRAPHS_JSON_API_PATH = '/api/v1/load_graphs_json';
const READ_TEXT_FILE_API_PATH = '/api/v1/read_text_file';

declare interface UploadResponse {
  path: string;
}

declare interface ReadTextFileResponse {
  content: string;
  error?: string;
}

/**
 * A service to manage model loading related tasks.
 */
@Injectable({
  providedIn: 'root',
})
export class ModelLoaderService implements ModelLoaderServiceInterface {
  // The loaded graph collections
  readonly loadedGraphCollections = signal<GraphCollection[] | undefined>(
    undefined,
  );

  readonly models = signal<ModelItem[]>([]);

  readonly changesToUpload = signal<ChangesPerGraphAndNode>({});

  readonly graphErrors = signal<string[] | undefined>(undefined);

  readonly selectedOptimizationPolicy = signal<string>('');

  constructor(
    private readonly settingsService: SettingsService,
    readonly extensionService: ExtensionService,
  ) {}

  get hasChangesToUpload() {
    return Object.keys(this.changesToUpload()).length > 0;
  }

  getOptimizationPolicies(extensionId: string): string[] {
    return this.extensionService.extensionSettings.get(extensionId)?.optimizationPolicies ?? [];
  }

  async executeModel(modelItem: ModelItem) {
    modelItem.status.set(ModelItemStatus.PROCESSING);
    let updatedPath = modelItem.path;
    let result: boolean = false;

    // User-entered file path.
    if (modelItem.type === ModelItemType.FILE_PATH) {
      result = await this.sendExecuteRequest(
        modelItem,
        updatedPath,
        {
          optimizationPolicy: this.selectedOptimizationPolicy()
        }
      );
    }
    // Upload or graph jsons from server.
    else if (
      modelItem.type === ModelItemType.LOCAL ||
      modelItem.type === ModelItemType.GRAPH_JSONS_FROM_SERVER
    ) {
      const file = modelItem.file!;

      // Upload the file
      modelItem.status.set(ModelItemStatus.UPLOADING);
      const {path, error: uploadError} = await this.uploadModelFile(file);
      if (uploadError) {
        modelItem.selected = false;
        modelItem.status.set(ModelItemStatus.ERROR);
        modelItem.errorMessage = uploadError;
        return false;
      }

      updatedPath = path;

      // Send request to backend for processing.
      result = await this.sendExecuteRequest(
        modelItem,
        updatedPath,
        {
          optimizationPolicy: this.selectedOptimizationPolicy()
        }
      );
    }

    return result;
  }

  async overrideModel(modelItem: ModelItem, graphCollection: GraphCollection, fieldsToUpdate: ChangesPerNode) {
    modelItem.status.set(ModelItemStatus.PROCESSING);
    let result = false;
    let updatedPath = modelItem.path;

    // User-entered file path.
    if (modelItem.type === ModelItemType.FILE_PATH) {
      result = await this.sendOverrideRequest(
        modelItem,
        updatedPath,
        graphCollection,
        fieldsToUpdate,
      );
    }
    // Upload or graph jsons from server.
    else if (
      modelItem.type === ModelItemType.LOCAL ||
      modelItem.type === ModelItemType.GRAPH_JSONS_FROM_SERVER
    ) {
      const file = modelItem.file!;

      // Upload the file
      modelItem.status.set(ModelItemStatus.UPLOADING);
      const {path, error: uploadError} = await this.uploadModelFile(file);
      if (uploadError) {
        modelItem.selected = false;
        modelItem.status.set(ModelItemStatus.ERROR);
        modelItem.errorMessage = uploadError;
        return false;
      }

      updatedPath = path;

      // Send request to backend for processing.
      result = await this.sendOverrideRequest(
        modelItem,
        updatedPath,
        graphCollection,
        fieldsToUpdate,
      );

      if (modelItem.status() !== ModelItemStatus.ERROR) {
        this.models.update((curModels) => {
          curModels.push({
            ...modelItem,
            path: updatedPath ?? modelItem.path,
          });

          return curModels;
        });

        modelItem.status.set(ModelItemStatus.DONE);
      }
    }

    return result;
  }

  async loadModels(modelItems: ModelItem[]) {
    // Create tasks for loading models in the given model items.
    const tasks: Array<Promise<GraphCollection[]>> = [];
    for (const modelItem of modelItems) {
      if (
        modelItem.type === ModelItemType.LOCAL ||
        modelItem.type === ModelItemType.GRAPH_JSONS_FROM_SERVER ||
        modelItem.type === ModelItemType.FILE_PATH
      ) {
        tasks.push(this.loadModel(modelItem));
      }
    }

    // Run tasks in parallel and gather results.
    const graphCollectionsList: GraphCollection[][] = await Promise.all(tasks);

    // Process error messages.
    for (const modelItem of modelItems) {
      if (modelItem.errorMessage != null) {
        modelItem.errorMessage = processErrorMessage(modelItem.errorMessage);
      }
    }

    // Only set the result if all tasks return non-empty collection list.
    if (
      graphCollectionsList.find((collections) => collections.length === 0) ==
      null
    ) {
      this.loadedGraphCollections.set(graphCollectionsList.flat());
    }
  }

  async loadModel(modelItem: ModelItem): Promise<GraphCollection[]> {
    modelItem.status.set(ModelItemStatus.PROCESSING);
    let result: GraphCollection[] = [];
    let updatedPath: string | undefined;

    // User-entered file path.
    if (modelItem.type === ModelItemType.FILE_PATH) {
      switch (modelItem.selectedAdapter?.id) {
        // Built-in json adapter.
        case InternalAdapterExtId.JSON_LOADER:
          try {
            const fileContent = await this.readTextFile(modelItem.path);
            const fileName = modelItem.path.split('/').pop() || 'untitled';
            const graphs = JSON.parse(fileContent) as Graph[];
            const jsonResult = processJson(fileName, graphs);
            if (jsonResult.error) {
              throw new Error(`Failed to process file: ${jsonResult.error})`);
            }
            if (jsonResult.graphCollections) {
              result = jsonResult.graphCollections;
            }
            modelItem.status.set(ModelItemStatus.DONE);
          } catch (e) {
            modelItem.selected = false;
            modelItem.status.set(ModelItemStatus.ERROR);
            modelItem.errorMessage = e as string;
          }
          break;

        // Other adapters. Send request to backend.
        default:
          const filePath = modelItem.path;
          const fileName = filePath.split('/').pop() || 'untitled';
          result = await this.sendConvertRequest(
            modelItem,
            filePath,
            fileName,
            false,
          );
          break;
      }
    }
    // Upload or graph jsons from server.
    else {
      const file = modelItem.file!;
      switch (modelItem.selectedAdapter?.id) {
        // This adapter processes json file in browser.
        case InternalAdapterExtId.JSON_LOADER:
          try {
            // Special handling for graphs json specified through backend
            // server.
            if (modelItem.type === ModelItemType.GRAPH_JSONS_FROM_SERVER) {
              // Load the json from backend.
              result = await this.loadGraphsFromBackendGraphsJson(
                modelItem.path,
              );
              modelItem.status.set(ModelItemStatus.DONE);
            }
            // Typical use cases where users pick a json file.
            else {
              result = await processUploadedJsonFile(file);
              modelItem.status.set(ModelItemStatus.DONE);
            }
          } catch (e) {
            modelItem.selected = false;
            modelItem.status.set(ModelItemStatus.ERROR);
            modelItem.errorMessage = e as string;
          }
          break;

        // For other adapters
        default:
          // Upload the file
          modelItem.status.set(ModelItemStatus.UPLOADING);
          const {path, error: uploadError} = await this.uploadModelFile(file);
          if (uploadError) {
            modelItem.selected = false;
            modelItem.status.set(ModelItemStatus.ERROR);
            modelItem.errorMessage = uploadError;
            return [];
          }

          updatedPath = path;

          // Send request to backend for processing.
          result = await this.sendConvertRequest(
            modelItem,
            path,
            file.name,
            true,
          );
          break;
      }
    }

    this.models.update((curModels) => {
      const filteredModels = curModels.filter(({ label }) => label === modelItem.label);

      return [
        ...filteredModels,
        {
          ...modelItem,
          path: updatedPath ?? modelItem.path,
        }
      ];
    });

    return result;
  }

  async checkExecutionStatus(modelItem: ModelItem, modelPath: string) {
    const result = await this.sendExtensionRequest<AdapterStatusCheckResponse>('status_check', modelItem, modelPath);

    if (!result || modelItem.status() === ModelItemStatus.ERROR) {
      return {
        isDone: true,
        progress: -1,
        error: modelItem.errorMessage ?? 'An error has occured'
      };
    }

    return this.processAdapterStatusCheckResponse(result) ?? {
      isDone: false,
      progress: -1,
      error: 'Empty response'
    };
  }

  private async readTextFile(path: string): Promise<string> {
    const resp = await fetch(`${READ_TEXT_FILE_API_PATH}?path=${path}`);
    const jsonObj = (await resp.json()) as ReadTextFileResponse;
    if (jsonObj.error) {
      throw new Error(`Failed to read file: ${jsonObj.error}`);
    }
    return jsonObj.content;
  }

  private async loadGraphsFromBackendGraphsJson(
    graphPath: string,
  ): Promise<GraphCollection[]> {
    // Get graphs index.
    //
    // graphPath is in the form of graph://{model_name}/{index}. Note that
    // {model_name} might contain "/".
    const partsStr = graphPath.replace(GRAPHS_MODEL_SOURCE_PREFIX, '');
    const lastSlashIndex = partsStr.lastIndexOf('/');
    const name = partsStr.substring(0, lastSlashIndex);
    const index = Number(partsStr.substring(lastSlashIndex + 1));
    const resp = await fetch(
      `${LOAD_GRAPHS_JSON_API_PATH}?graph_index=${index}`,
    );
    const json = (await resp.json()) as AdapterConvertResponse;
    return this.processAdapterConvertResponse(json, name);
  }

  private async uploadModelFile(
    file: File,
  ): Promise<{path: string; error?: string}> {
    const data = new FormData();
    data.append('file', file, file.name);
    const uploadResp = await fetch(UPLOAD_API_PATH, {
      method: 'POST',
      body: data,
    });
    if (!uploadResp.ok) {
      console.error(await uploadResp.text());
      return {
        path: '',
        error: 'Failed to upload model. Check console for details',
      };
    } else {
      const path = (JSON.parse(await uploadResp.text()) as UploadResponse).path;
      return {path};
    }
  }

  private async sendExtensionRequest<T extends ExtensionResponse<any[], any[]>>(
    command: string,
    modelItem: ModelItem,
    path: string,
    settings?: Record<string, any>,
    deleteAfterConversion: boolean = false,
  ) {
    try {
      modelItem.status.set(ModelItemStatus.PROCESSING);
      const convertCommand: ExtensionCommand = {
        cmdId: command,
        extensionId: modelItem.selectedAdapter?.id || '',
        modelPath: path,
        settings: settings ?? {},
        deleteAfterConversion,
      };

      const { cmdResp, otherError: cmdError } = await this.extensionService.sendCommandToExtension<T>(convertCommand);

      if (cmdError) {
        throw new Error(cmdError);
      }

      if (!cmdResp) {
        throw new Error(`Command "${command}" didn't return any response`);
      }

      if (cmdResp.error) {
        throw new Error(cmdResp.error);
      }

      modelItem.status.set(ModelItemStatus.DONE);
      return cmdResp;
    } catch (err) {
      modelItem.selected = false;
      modelItem.errorMessage = (err as Partial<Error>)?.message ?? err?.toString() ?? `An error has occured when running command "${command}"`;
      modelItem.status.set(ModelItemStatus.ERROR);

      return undefined;
    }
  }

  private async sendConvertRequest(
    modelItem: ModelItem,
    path: string,
    fileName: string,
    deleteAfterConversion: boolean,
  ): Promise<GraphCollection[]> {
    const result = await this.sendExtensionRequest<AdapterConvertResponse>('convert', modelItem, path, this.settingsService.getAllSettingsValues(), deleteAfterConversion);

    if (!result || modelItem.status() === ModelItemStatus.ERROR) {
      return [];
    }

    return this.processAdapterConvertResponse(result, fileName);
  }

  private async sendExecuteRequest(
    modelItem: ModelItem,
    path: string,
    settings: Record<string, any> = {}
  ) {
    const result = await this.sendExtensionRequest<AdapterExecuteResponse>('execute', modelItem, path, settings);

    if (!result || modelItem.status() === ModelItemStatus.ERROR) {
      return false;
    }

    return this.processAdapterExecuteResponse(result);
  }

  private async sendOverrideRequest(
    modelItem: ModelItem,
    path: string,
    graphCollection: GraphCollection,
    fieldsToUpdate: Record<string, any>
  ) {

    const result = await this.sendExtensionRequest<AdapterOverrideResponse>('override', modelItem, path, {
      graphs: graphCollection.graphs,
      changes: fieldsToUpdate,
    });

    if (!result || modelItem.status() === ModelItemStatus.ERROR) {
      return false;
    }

    return this.processAdapterOverrideResponse(result);
  }

  private processAdapterConvertResponse(
    resp: AdapterConvertResponse,
    fileName: string,
  ): GraphCollection[] {
    if (resp.graphs) {
      return [{label: fileName, graphs: resp.graphs }];
    } else if (resp.graphCollections) {
      return resp.graphCollections?.map((item) => {
        return {
          label: item.label === '' ? fileName : `${fileName} (${item.label})`,
          graphs: item.graphs,
          perf_data: item.perf_data
        };
      }) ?? [];
    }
    return [];
  }

  private processAdapterOverrideResponse(
    resp: AdapterOverrideResponse,
  ): boolean {
    return resp?.graphs?.[0].success ?? false;
  }

  private processAdapterStatusCheckResponse(
    resp: AdapterStatusCheckResponse
  ) {
      return resp?.graphs?.[0];
  }

  private processAdapterExecuteResponse(
    resp: AdapterExecuteResponse
  ) {
    return resp.graphs?.length === 0;
  }
}
