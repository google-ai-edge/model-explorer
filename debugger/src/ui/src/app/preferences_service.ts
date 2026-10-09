/**
 * @license
 * Copyright 2026 The AI Edge Model Explorer Authors.
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

/** Browser preferences shared by the home page and application dialogs. */
@Injectable({providedIn: 'root'})
export class PreferencesService {
  readonly showWelcome = signal(this.read('debugger.showWelcome') !== 'false');
  read(key: string): string | null {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  }
  save(key: string, value: string) {
    try {
      localStorage.setItem(key, value);
    } catch {
      /* Keep session preferences when storage is unavailable. */
    }
  }
  setShowWelcome(value: boolean) {
    this.showWelcome.set(value);
    this.save('debugger.showWelcome', String(value));
  }
}
