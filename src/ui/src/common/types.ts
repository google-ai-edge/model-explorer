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

import {WritableSignal} from '@angular/core';

/**
 * TF versions.
 */
export enum TfVersion {
  TF1 = 'v1',
  TF2 = 'v2',
}

/** Ids for internal adapters. */
export enum InternalAdapterExtId {
  TFLITE_MLIR = 'builtin_tflite_mlir',
  TFLITE_FLATBUFFER = 'builtin_tflite_flatbuffer',
  TF_MLIR = 'builtin_tf_mlir',
  TF_DIRECT = 'builtin_tf_direct',
  GRAPHDEF = 'builtin_graphdef',
  MLIR = 'builtin_mlir',
  JSON_LOADER = 'builtin_json',
  DATA_NEXUS = 'builtin_data_nexus',
  MEDIAPIPE = 'builtin_mediapipe',
}

/** Extension types. */
export enum ExtensionType {
  ADAPTER = 'adapter',
  NODE_DATA_PROVIDER = 'node_data_provider',
}

/** Basic metadata for an extension */
export declare interface ExtensionBase {
  id: string;
  name: string;
  description: string;
  type: ExtensionType;
}

/** Metadata of an adapter extension. */
export declare interface AdapterExtension extends ExtensionBase {
  type: ExtensionType.ADAPTER;

  fileExts: string[];

  // Used internally to match remote google storage path.
  matchGoogleStorageDir?: boolean;

  // Used internally to match http/https urls.
  matchHttpUrl?: boolean;
}

/** Union type for extension. */
export type Extension = AdapterExtension;

/** An item in the model table. */
export interface ModelItem {
  path: string;
  type: ModelItemType;
  status: WritableSignal<ModelItemStatus>;
  selected: boolean;
  file?: File;
  adapterCandidates?: AdapterExtension[];
  selectedAdapter?: AdapterExtension;
  errorMessage?: string;
}

/** The type of a model item. */
export enum ModelItemType {
  // A file selected by the browser file api.
  LOCAL = 'local',

  // Remote path/url.
  REMOTE = 'remote',

  // Data nexus models.
  DATA_NEXUS = 'data_nexus',

  // External only
  //
  // Graphs json specified when starting server.
  GRAPH_JSONS_FROM_SERVER = 'graphs_json_from_server',

  // External only
  //
  // User entered file path.
  FILE_PATH = 'file_path',
}

/** The status of a model item. */
export enum ModelItemStatus {
  NOT_STARTED = 'Not started',
  PROCESSING = 'Converting',
  UPLOADING = 'Uploading',
  DONE = 'Done',
  ERROR = 'Error',
}
