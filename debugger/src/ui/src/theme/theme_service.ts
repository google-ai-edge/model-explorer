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

import {computed, DestroyRef, effect, inject, Injectable, signal} from '@angular/core';
import {DOCUMENT} from '@angular/common';
import {PreferencesService} from '../app/preferences_service';
export type ThemePreference = 'system' | 'light' | 'dark';

/** Matches Model Explorer: persisted preference, system listener, one derived theme. */
@Injectable({providedIn: 'root'})
export class ThemeService {
  private readonly preferences = inject(PreferencesService);
  private readonly document = inject(DOCUMENT);
  private readonly media = window.matchMedia('(prefers-color-scheme: dark)');
  readonly paletteReview = new URLSearchParams(window.location.search).has('debugPalette');
  readonly debugPalettes = [
    {id: 'original', label: 'Original amber'},
    {id: 'slate', label: 'Blue gray'},
    {id: 'teal', label: 'Teal'},
    {id: 'violet', label: 'Muted violet'},
    {id: 'orange', label: 'Orange accent'},
    {id: 'vscode', label: 'VS Code inspired'},
    {id: 'jetbrains', label: 'JetBrains inspired'},
    {id: 'android-studio', label: 'Android Studio inspired'},
  ] as const;
  readonly debugPalette = signal(this.initialDebugPalette());
  private initialDebugPalette() {
    const id = new URLSearchParams(window.location.search).get('debugPalette');
    return this.debugPalettes.find((p) => p.id === id)?.id ?? 'original';
  }
  setDebugPalette(id: string) {
    const choice = this.debugPalettes.find((p) => p.id === id);
    if (choice) this.debugPalette.set(choice.id);
  }
  readonly systemDark = signal(this.media.matches);
  readonly preference = signal<ThemePreference>(this.initialPreference());
  readonly dark = computed(() =>
    this.preference() === 'system' ? this.systemDark() : this.preference() === 'dark',
  );
  readonly icon = computed(() =>
    this.preference() === 'system' ? 'brightness_auto' : this.dark() ? 'dark_mode' : 'light_mode',
  );
  constructor() {
    effect(() =>
      this.document.documentElement.setAttribute('data-debug-palette', this.debugPalette()),
    );
    const update = (event: MediaQueryListEvent) => this.systemDark.set(event.matches);
    this.media.addEventListener('change', update);
    inject(DestroyRef).onDestroy(() => this.media.removeEventListener('change', update));
    effect(() =>
      this.document.documentElement.setAttribute('data-theme', this.dark() ? 'dark' : 'light'),
    );
  }
  private initialPreference(): ThemePreference {
    const value = this.preferences.read('debugger.theme');
    return value === 'light' || value === 'dark' ? value : 'system';
  }
  setPreference(value: ThemePreference) {
    this.preference.set(value);
    this.preferences.save('debugger.theme', value);
  }
  toggle() {
    this.setPreference(this.dark() ? 'light' : 'dark');
  }
}
