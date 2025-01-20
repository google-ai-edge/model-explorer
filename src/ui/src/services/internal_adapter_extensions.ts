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
  AdapterExtension,
  ExtensionType,
  InternalAdapterExtId,
} from '../common/types';

const tfliteMlirAdapterExtension: AdapterExtension = {
  type: ExtensionType.ADAPTER,
  fileExts: ['tflite'],
  id: InternalAdapterExtId.TFLITE_MLIR,
  name: 'TFLite adapter (MLIR)',
  description:
    'A built-in adapter that converts a TFLite model to Model Explorer format through MLIR.',
};
const tfliteFlatbufferAdapterExtension: AdapterExtension = {
  type: ExtensionType.ADAPTER,
  fileExts: ['tflite'],
  id: InternalAdapterExtId.TFLITE_FLATBUFFER,
  name: 'TFLite adapter (Flatbuffer)',
  description:
    'A built-in adapter that converts a TFLite model to Model Explorer format by directly parsing the flatbuffer.',
};
const tfMlirAdapterExtension: AdapterExtension = {
  type: ExtensionType.ADAPTER,
  fileExts: ['pb'],
  matchGoogleStorageDir: true,
  id: InternalAdapterExtId.TF_MLIR,
  name: 'TF adapter (MLIR)',
  description:
    'A built-in adapter that converts a TF saved model to Model Explorer format through MLIR.',
};
const tfDirectAdapterExtension: AdapterExtension = {
  type: ExtensionType.ADAPTER,
  fileExts: ['pb'],
  matchGoogleStorageDir: true,
  id: InternalAdapterExtId.TF_DIRECT,
  name: 'TF adapter (direct)',
  description:
    'A built-in adapter that converts a TF saved model to Model Explorer format by directly parsing the .pb file.',
};
const graphdefAdapterExtension: AdapterExtension = {
  type: ExtensionType.ADAPTER,
  fileExts: ['pb', 'pbtxt', 'graphdef'],
  id: InternalAdapterExtId.GRAPHDEF,
  name: 'GraphDef adapter',
  description:
    'A built-in adapter that converts GraphDef file to Model Explorer format.',
};
const mlirAdapterExtension: AdapterExtension = {
  type: ExtensionType.ADAPTER,
  fileExts: ['mlir', 'mlirbc'],
  id: InternalAdapterExtId.MLIR,
  name: 'MLIR adapter',
  description:
    'A built-in adapter that converts MLIR file to Model Explorer format.',
};
const jsonAdapterExtension: AdapterExtension = {
  type: ExtensionType.ADAPTER,
  fileExts: ['json'],
  matchHttpUrl: true,
  id: InternalAdapterExtId.JSON_LOADER,
  name: 'Url/Json adapter',
  description:
    'Loads JSON graphs data file or tfjs model from the given url or uploaded file and convert them to Model Explorer format.',
};
const dataNexusAdapterExtension: AdapterExtension = {
  type: ExtensionType.ADAPTER,
  fileExts: ['data_nexus'],
  id: InternalAdapterExtId.DATA_NEXUS,
  name: 'Data Nexus adapter',
  description: 'Loads data from Data Nexus.',
};
const mediapipeAdapterExtension: AdapterExtension = {
  type: ExtensionType.ADAPTER,
  fileExts: ['pbtxt'],
  id: InternalAdapterExtId.MEDIAPIPE,
  matchGoogleStorageDir: true, // Also accept an entire directory.
  name: 'MediaPipe adapter',
  description:
    'A built-in adapter that converts a MediaPipe Pipeline to Model Explorer format.',
};

/** All internal extensions. */
export const INTERNAL_ADAPTER_EXTENSIONS: AdapterExtension[] = [
  tfliteFlatbufferAdapterExtension,
  tfliteMlirAdapterExtension,
  tfMlirAdapterExtension,
  tfDirectAdapterExtension,
  graphdefAdapterExtension,
  mlirAdapterExtension,
  jsonAdapterExtension,
  dataNexusAdapterExtension,
  mediapipeAdapterExtension,
];
