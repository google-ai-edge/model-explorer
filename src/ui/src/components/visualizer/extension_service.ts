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

import {Injectable} from '@angular/core';

import {ExtensionCommand} from './extension_command';

/** Metadata from an extension. */
export declare interface Extension {
  id: string;
  name: string;
  description?: string;
  type: 'node_data_provider';
  language: 'python' | 'js';
}

const DEFAULT_EXTENSION_SERVER_ADDRESS = 'http://localhost:5000';

/**
 * A service to manage extensions.
 */
@Injectable()
export class ExtensionService {
  // TODO: get the default address from an input field of the component.
  extensionServerAddress = DEFAULT_EXTENSION_SERVER_ADDRESS;
  extensions: Record<string, Extension> = {};

  async loadExtensions(): Promise<Extension[]> {
    const extensions =
      (await this.sendGetRequest<Extension[]>('api_list_extensions')) || [];
    extensions.sort((a, b) => a.name.localeCompare(b.name));

    // Build index.
    for (const extension of extensions) {
      this.extensions[extension.id] = extension;
    }

    return extensions;
  }

  async sendCommandToExtension<T>(
    cmd: ExtensionCommand,
  ): Promise<T | undefined> {
    return await this.sendGetRequest<T>('api_cmd', cmd);
  }

  updateExtensionServerAddress(address: string) {
    this.extensionServerAddress = address;
    // TODO: send the change as an event to out of the component.
  }

  private async sendGetRequest<T>(
    apiPath: string,
    cmd?: ExtensionCommand,
  ): Promise<T | undefined> {
    let path = `${this.extensionServerAddress}/${apiPath}`;
    if (cmd) {
      path = `${path}?cmd=${encodeURIComponent(JSON.stringify(cmd))}`;
    }
    try {
      const resp = await fetch(path, {credentials: 'include'});
      if (!resp.ok) {
        return undefined;
      }
      const json = await resp.json();
      return json as T;
    } catch (e) {
      console.warn(e);
      return undefined;
    }
  }

  // // Keep this in case it is needed.
  // private async sendPostRequest<T>(apiPath: string, cmd?: ExtensionCommand):
  //     Promise<T> {
  //   let path = `${this.extensionServerAddress}/${apiPath}`;
  //   const requestData: RequestInit = {
  //     method: 'POST',
  //     headers: {
  //       'Content-Type': 'application/json',
  //     },
  //   };
  //   if (cmd) {
  //     requestData.body = JSON.stringify(cmd);
  //   }
  //   const resp = await fetch(path, requestData);
  //   const json = await resp.json();
  //   return json as T;
  // }
}
