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
  Graph,
  GraphCollection,
} from '../components/visualizer/common/input_graph';
import { NodeDataProviderGraphData } from '../custom_element';
import { ConfigValue } from './types';

/** A command sent to extension. */
export declare interface ExtensionCommand {
  cmdId: string;
  extensionId: string;
}

/** Adapter's "convert" command. */
export declare interface AdapterConvertCommand extends ExtensionCommand {
  cmdId: 'convert';
  modelPath: string;
  // tslint:disable-next-line:no-any Allow arbitrary types.
  settings: Record<string, any>;
  // Whether to delete the model file at `modelPath` after conversion is done.
  deleteAfterConversion: boolean;
}

/** Adapter's "convert" command response. */
export declare interface AdapterConvertResponse {
  graphs?: Graph[];
  graphCollections?: GraphCollection[];
  error?: string;
}

export declare interface NdpRunCommand extends ExtensionCommand {
  cmdId: 'run';
  modelPath: string;
  configValues: Record<string, ConfigValue>;
}

export declare interface NdpRunResponse {
  result?: NodeDataProviderGraphData;
  error?: string;
}
