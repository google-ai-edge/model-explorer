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

import {AdapterExtension} from '../../common/types';
import {isInternalStoragePath} from '../../common/utils';
import {AdapterExtensionServiceBase} from '../../services/adapter_extension_service';

/** Gets adapter candidates for the given model path. */
export function getAdapterCandidates(
  modelPath: string,
  adapterExtensionService: AdapterExtensionServiceBase,
  isExternal: boolean,
): AdapterExtension[] {
  // Get the file extension.
  const pathComponents = modelPath.split('/').filter((s) => s !== '');
  const lastPathComponent = pathComponents[pathComponents.length - 1];
  const parts = lastPathComponent.split('.');

  // Internal + http urls.
  if (!isExternal && modelPath.startsWith('http')) {
    return adapterExtensionService.getExtensionsForHttpUrls();
  }

  // Has file extension.
  //
  // Get the matching adapter by the file extension.
  if (parts.length > 1 && !modelPath.endsWith('/')) {
    const fileExt = parts[parts.length - 1];
    return adapterExtensionService.getExtensionsByFileExt(fileExt);
  }

  // Internal + internal storage.
  if (!isExternal && isInternalStoragePath(modelPath)) {
    return adapterExtensionService.getExtensionsForGoogleStorageDir();
  }

  return [];
}
