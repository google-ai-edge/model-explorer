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
  AdapterConvertCommand,
  AdapterConvertResponse,
} from '../common/extension_command';
import {ModelLoaderServiceInterface} from '../common/model_loader_service_interface';
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

  constructor(
    private readonly settingsService: SettingsService,
    readonly extensionService: ExtensionService,
  ) {}

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

    return result;
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

  private async sendConvertRequest(
    modelItem: ModelItem,
    path: string,
    fileName: string,
    deleteAfterConversion: boolean,
  ): Promise<GraphCollection[]> {
    let result: GraphCollection[] = [];
    modelItem.status.set(ModelItemStatus.PROCESSING);
    const convertCommand: AdapterConvertCommand = {
      cmdId: 'convert',
      extensionId: modelItem.selectedAdapter?.id || '',
      modelPath: path,
      settings: this.settingsService.getAllSettingsValues(),
      deleteAfterConversion,
    };
    const {cmdResp, otherError: cmdError} =
      await this.extensionService.sendCommandToExtension<AdapterConvertResponse>(
        convertCommand,
      );
    const error = cmdResp?.error || cmdError;
    if (error) {
      modelItem.selected = false;
      modelItem.status.set(ModelItemStatus.ERROR);
      modelItem.errorMessage = error;
      return [];
    } else if (cmdResp) {
      result = this.processAdapterConvertResponse(cmdResp, fileName);
    }
    modelItem.status.set(ModelItemStatus.DONE);
    return result;
  }

  private processAdapterConvertResponse(
    resp: AdapterConvertResponse,
    fileName: string,
  ): GraphCollection[] {
    if (resp.graphs) {
      return [{label: fileName, graphs: resp.graphs}];
    } else if (resp.graphCollections) {
      return resp.graphCollections.map((item) => {
        return {
          label: item.label === '' ? fileName : `${fileName} (${item.label})`,
          graphs: item.graphs,
        };
      });
    }
    return [];
  }
}
