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

import {ExtensionCommand} from '../common/extension_command';
import {Extension} from '../common/types';

const EXTERNAL_GET_EXTENSIONS_API_PATH = '/api/v1/get_extensions';
const EXTERNAL_SEND_CMD_API_PATH = '/apipost/v1/send_command';

/**
 * Service for managing model explorer extensions.
 */
@Injectable({providedIn: 'root'})
export class ExtensionService {
  readonly loading = signal<boolean>(true);

  extensions: Extension[] = [];

  constructor() {
    this.loadExtensions();
  }

  async sendCommandToExtension<T>(
    cmd: ExtensionCommand,
  ): Promise<{cmdResp?: T; otherError?: string}> {
    const requestData: RequestInit = {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    };
    requestData.body = JSON.stringify(cmd);
    try {
      const resp = await fetch(EXTERNAL_SEND_CMD_API_PATH, requestData);
      if (!resp.ok) {
        return {otherError: `Failed to convert model. ${resp.status}`};
      }
      return {cmdResp: (await resp.json()) as T};
    } catch (e) {
      return {otherError: e as string};
    }
  }

  private async loadExtensions() {
    // Talk to BE to get registered extensions.
    let exts: Extension[] = [];

    exts = await this.getExtensionsForExternal();
    this.extensions = exts;
    this.loading.set(false);
  }

  private async getExtensionsForExternal(): Promise<Extension[]> {
    try {
      const resp = await fetch(EXTERNAL_GET_EXTENSIONS_API_PATH, {
        credentials: 'include',
      });
      if (!resp.ok) {
        console.error(`Failed to get extensions: ${resp.status}`);
        return [];
      }
      const json = await resp.json();
      return json as Extension[];
    } catch (e) {
      console.error('Failed to get extensions.', e);
      return [];
    }
  }
}
