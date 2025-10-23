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

/** Metadata of a node data provider extension. */
export declare interface NodeDataProviderExtension extends ExtensionBase {
  type: ExtensionType.NODE_DATA_PROVIDER;
  icon: string;
  filter?: NodeDataProviderFilter;
}

/** Filter for node data provider extensions. */
export declare interface NodeDataProviderFilter {
  modelFileExts?: string[];
  adapterIds?: string[];
}

/** Union type for extension. */
export type Extension = AdapterExtension | NodeDataProviderExtension;

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

/** The type of a config editor. */
export enum ConfigEditorType {
  TEXT_INPUT = 'text_input',
  TEXT_AREA = 'text_area',
  SLIDE_TOGGLE = 'slide_toggle',
  COLOR_PICKER = 'color_picker',
  DROP_DOWN = 'drop_down',
  BUTTON_TOGGLE = 'button_toggle',
  FILE = 'file',
}

/** The base config editor interface. */
export declare interface ConfigEditorBase {
  type: ConfigEditorType;
  id: string;
  label?: string;
  defaultValue?: ConfigValue;
  required?: boolean;
  description?: string;
  help?: string;
}

/** Text input config editor. */
export declare interface TextInputConfigEditor extends ConfigEditorBase {
  type: ConfigEditorType.TEXT_INPUT;
  number: boolean;
}

/** Text area config editor. */
export declare interface TextAreaConfigEditor extends ConfigEditorBase {
  type: ConfigEditorType.TEXT_AREA;
  height: number;
}

/** Slide toggle config editor. */
export declare interface SlideToggleConfigEditor extends ConfigEditorBase {
  type: ConfigEditorType.SLIDE_TOGGLE;
}

/** Color picker config editor. */
export declare interface ColorPickerConfigEditor extends ConfigEditorBase {
  type: ConfigEditorType.COLOR_PICKER;
}

/** Drop down config editor. */
export declare interface DropDownConfigEditor extends ConfigEditorBase {
  type: ConfigEditorType.DROP_DOWN;
  options: OptionItem[];
}

/** Button toggle config editor. */
export declare interface ButtonToggleConfigEditor extends ConfigEditorBase {
  type: ConfigEditorType.BUTTON_TOGGLE;
  options: OptionItem[];
  multiple: boolean;
}

/** File upload config editor. */
export declare interface FileConfigEditor extends ConfigEditorBase {
  type: ConfigEditorType.FILE;
  fileExts: string[];
}

/** An option item in a drop down or button toggle config editor. */
export declare interface OptionItem {
  value: string;
  label: string;
}

/** Union type for config editors. */
export type ConfigEditor =
  | TextInputConfigEditor
  | TextAreaConfigEditor
  | SlideToggleConfigEditor
  | ColorPickerConfigEditor
  | DropDownConfigEditor
  | ButtonToggleConfigEditor
  | FileConfigEditor;

/** Config value. */
export type ConfigValue = string | boolean | number | string[];

/** An NDP extension run. */
export interface NdpExtensionRun {
  runId: string;
  extensionId: string;
  runName: string;
  creationTimeTs: number;
  status: NdpExtensionRunStatus;
  finishTimeTs?: number;
  error?: string;
}

/** The status of an NDP extension run. */
export enum NdpExtensionRunStatus {
  RUNNNING = 'running',
  DONE = 'done',
  ERROR = 'error',
}
