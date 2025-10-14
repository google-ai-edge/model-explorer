/**
 * @license
 * Copyright 2025 The Model Explorer Authors. All Rights Reserved.
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

import {LocalStorageService} from '../components/visualizer/local_storage_service';

import {DOCUMENT} from '@angular/common';
import {Injectable, computed, effect, inject, signal} from '@angular/core';

/**
 * Theme preference enum.
 */
export enum ThemePreference {
  SYSTEM = 'system',
  LIGHT = 'light',
  DARK = 'dark',
}

const THEME_PREFERENCE_KEY = 'model_explorer_theme';

/**
 * Service for managing theme.
 */
@Injectable({providedIn: 'root'})
export class ThemeService {
  private readonly localStorageService = inject(LocalStorageService);

  readonly isDarkMode = computed(() => {
    const userTheme = this.curThemePreference();
    if (userTheme === ThemePreference.SYSTEM) {
      return this.mediaPrefersDarkMode();
    }
    return userTheme === ThemePreference.DARK;
  });

  readonly curThemePreference = signal<ThemePreference>(
    this.getThemePreference(),
  );

  private readonly document = inject(DOCUMENT);

  private readonly mediaPrefersDarkMode = signal(false);

  constructor() {
    const prefersDarkQuery = window.matchMedia('(prefers-color-scheme: dark)');

    // false means that either the media pefers light or it does not support
    // querying prefers-color-scheme. In either case, we assume light mode.
    this.mediaPrefersDarkMode.set(prefersDarkQuery.matches);
    prefersDarkQuery.onchange = (event) => {
      this.mediaPrefersDarkMode.set(event.matches);
    };

    effect(() => {
      const isDarkMode = this.isDarkMode();

      if (isDarkMode) {
        this.document.body.dataset['metheme'] = 'dark';
      } else {
        this.document.body.dataset['metheme'] = 'light';
      }
    });
  }

  setThemePreference(theme: ThemePreference) {
    this.localStorageService.setItem(THEME_PREFERENCE_KEY, theme);
    this.curThemePreference.set(theme);
  }

  private getThemePreference(): ThemePreference {
    const theme = this.localStorageService.getItem(THEME_PREFERENCE_KEY);
    return (theme as ThemePreference) ?? ThemePreference.SYSTEM;
  }
}
