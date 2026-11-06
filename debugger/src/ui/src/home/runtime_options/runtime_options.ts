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

import {Component, ElementRef, inject, signal} from '@angular/core';
import {ConfigPicker} from '../../widgets/config_picker/config_picker';
import {FormsModule} from '@angular/forms';
import {MAT_DIALOG_DATA, MatDialogModule, MatDialogRef} from '@angular/material/dialog';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';
import type {RuntimeOptions as Options} from '../../server/contracts/sessions';
import {optionApplies, RuntimeOptionKey, sessionOptions} from '../runtime_options';
import type {OptionDescription, RuntimeCapability} from '../../server/contracts/runtime';
export interface RuntimeOptionsData {
  side: string;
  other: string;
  runtime: string;
  otherRuntime?: string;
  run: Options;
  otherRun: Options;
  capability?: RuntimeCapability;
}
/** A described option and the run key it sets. */
interface Field {
  key: RuntimeOptionKey;
  option: OptionDescription;
}
const SWITCH_LABELS = ['Default', 'On', 'Off'];
@Component({
  selector: 'runtime-options',
  imports: [
    ConfigPicker,
    FormsModule,
    MatDialogModule,
    MatButtonModule,
    MatIconModule,
    MatTooltipModule,
  ],
  templateUrl: './runtime_options.ng.html',
  styleUrl: './runtime_options.scss',
})
export class RuntimeOptions {
  readonly data = inject<RuntimeOptionsData>(MAT_DIALOG_DATA);
  readonly ref = inject(MatDialogRef<RuntimeOptions>);
  private readonly element: ElementRef<HTMLElement> = inject(ElementRef);
  draft: Options = {...this.data.run};
  previous: Options | null = null;
  readonly copied = signal(false);
  /** The runtime's options as the server describes them for the side's device. */
  private readonly fields: readonly Field[] = sessionOptions(this.data.capability).map(
    (option) => ({key: option.key as RuntimeOptionKey, option}),
  );
  readonly keys: readonly RuntimeOptionKey[] = this.fields.map((field) => field.key);
  /** The described options in their sections, in the server's order. */
  readonly groups = this.fields.reduce<Array<{name: string; fields: Field[]}>>((groups, field) => {
    const group = groups.find((g) => g.name === field.option.group);
    if (group) {
      group.fields.push(field);
    } else {
      groups.push({name: field.option.group, fields: [field]});
    }
    return groups;
  }, []);
  readonly switchLabels = SWITCH_LABELS;

  constructor() {
    for (const key of this.keys) {
      this.draft[key] ??= '';
    }
    // Escape and the backdrop discard the draft; only Apply validates and returns it.
    this.ref.disableClose = true;
    this.ref.backdropClick().subscribe(() => this.cancel());
    this.ref.keydownEvents().subscribe((event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        this.cancel();
      }
    });
  }

  cancel(): void {
    this.ref.close(undefined);
  }

  private described(key: RuntimeOptionKey): OptionDescription | undefined {
    return this.fields.find((field) => field.key === key)?.option;
  }

  /** Whether this runtime, and the runner on the side's device, take the option at all. */
  supported(key: RuntimeOptionKey): boolean {
    return this.described(key)?.available ?? false;
  }

  /** Supported, and what it requires chosen (CPU- or GPU-only members need that backend). */
  editable(key: RuntimeOptionKey): boolean {
    const option = this.described(key);
    return !!option && option.available && optionApplies(option, this.draft);
  }

  /** What a described option does, and why it cannot be set when it cannot. */
  help(option: OptionDescription): string {
    return [option.help, option.reason].filter(Boolean).join(' ');
  }

  choiceOptions(option: OptionDescription): string[] {
    return (option.choices ?? []).map((choice) => choice || 'Default');
  }

  choiceLabel(field: Field): string {
    return this.draft[field.key] || String(field.option.default ?? 'Default');
  }

  selectChoice(field: Field, label: string): void {
    this.draft[field.key] = label === 'Default' ? '' : label;
    this.clearUnpaired();
  }

  switchLabel(key: RuntimeOptionKey): string {
    return this.draft[key] === 'on' ? 'On' : this.draft[key] === 'off' ? 'Off' : 'Default';
  }

  setSwitch(key: RuntimeOptionKey, label: string): void {
    this.draft[key] = label === 'On' ? 'on' : label === 'Off' ? 'off' : '';
  }

  /** Members whose requirement is no longer chosen fall back to the runtime default. */
  private clearUnpaired(): void {
    for (const key of this.keys) {
      if (!this.editable(key) && this.supported(key)) {
        this.draft[key] = '';
      }
    }
  }

  canCopy(): boolean {
    return !this.data.otherRuntime || this.data.otherRuntime === this.data.runtime;
  }

  /** A described option goes back to the server's default; the backend to the first offered. */
  reset(key: RuntimeOptionKey): void {
    if (!this.supported(key)) {
      return;
    }
    const option = this.described(key);
    this.draft[key] =
      key === 'backend' && option
        ? (option.choices?.[0] ?? 'CPU')
        : option?.default != null && key === 'contextLength'
          ? String(option.default)
          : '';
    this.clearUnpaired();
  }

  copy(): void {
    if (!this.canCopy()) {
      return;
    }
    this.previous = {...this.draft};
    for (const key of this.keys) {
      if (this.supported(key)) {
        this.draft[key] = this.data.otherRun[key] ?? '';
      }
    }
    // A choice the other side made that this side's device does not offer goes back to default.
    for (const field of this.fields) {
      if (
        field.option.type === 'choice' &&
        !field.option.choices?.includes(this.draft[field.key] ?? '')
      ) {
        this.reset(field.key);
      }
    }
    this.clearUnpaired();
    this.copied.set(true);
  }

  undo(): void {
    if (this.previous) {
      this.draft = {...this.previous};
    }
    this.previous = null;
    this.copied.set(false);
  }

  close(): void {
    const form = this.element.nativeElement.querySelector('form');
    if (form && !form.reportValidity()) {
      return;
    }
    this.ref.close(
      Object.fromEntries(this.keys.map((key) => [key, String(this.draft[key] ?? '')])),
    );
  }
}
