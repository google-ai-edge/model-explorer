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

import {Injectable, effect} from '@angular/core';

import {AdapterExtension, ExtensionType} from '../common/types';
import {ExtensionService} from './extension_service';

/** The base class of AdapterExtensionService. */
export class AdapterExtensionServiceBase {
  /** File ext to supported extensions */
  readonly registry: Record<string, AdapterExtension[]> = {};

  getExtensionsByFileExt(fileExt: string): AdapterExtension[] {
    return this.registry[fileExt] || [];
  }

  getExtensionById(extId: string): AdapterExtension | undefined {
    for (const curExts of Object.values(this.registry)) {
      for (const curExt of curExts) {
        if (curExt.id === extId) {
          return curExt;
        }
      }
    }
    return undefined;
  }

  getExtensionsForGoogleStorageDir(): AdapterExtension[] {
    const exts: AdapterExtension[] = [];
    for (const curExts of Object.values(this.registry)) {
      for (const curExt of curExts) {
        if (curExt.matchGoogleStorageDir && !exts.includes(curExt)) {
          exts.push(curExt);
        }
      }
    }
    return exts;
  }

  getExtensionsForHttpUrls(): AdapterExtension[] {
    const exts: AdapterExtension[] = [];
    for (const curExts of Object.values(this.registry)) {
      for (const curExt of curExts) {
        if (curExt.matchHttpUrl) {
          exts.push(curExt);
        }
      }
    }
    return exts;
  }

  protected register(extMetadata: AdapterExtension) {
    for (const fileExt of extMetadata.fileExts) {
      // Dedup.
      if (this.registry[fileExt] == null) {
        this.registry[fileExt] = [];
      }
      const curExts = this.registry[fileExt];
      if (curExts.find((ext) => ext.id === extMetadata.id) != null) {
        console.warn('Adapter extension exists', extMetadata);
        continue;
      }
      curExts.push(extMetadata);
    }
  }
}

/**
 * Service for managing adapter extensions.
 */
@Injectable({providedIn: 'root'})
export class AdapterExtensionService extends AdapterExtensionServiceBase {
  constructor(private readonly extensionService: ExtensionService) {
    super();

    effect(() => {
      if (this.extensionService.loading()) {
        return;
      }
      for (const extension of this.extensionService.extensions) {
        if (extension.type === ExtensionType.ADAPTER) {
          this.register(extension);
        }
      }
    });
  }
}
