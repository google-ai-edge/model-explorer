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
import {ConfigPicker} from '../../../shared/config_picker/config_picker';
import {FormsModule} from '@angular/forms';
import {MAT_DIALOG_DATA, MatDialogModule, MatDialogRef} from '@angular/material/dialog';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';
import type {RuntimeOptions as Options} from '../../../data/contracts/session_summary';
import {
  activationDataTypes,
  advancedRuntimeOptionFields,
  runtimeOptionKeys,
  pytorchRuntimeOptionKeys,
  RuntimeOptionKey,
} from '../runtime_options';
import type {RuntimeCapability} from '../../../data/contracts/runtime';
export interface RuntimeOptionsData {
  side: string;
  other: string;
  runtime: string;
  otherRuntime?: string;
  run: Options;
  otherRun: Options;
  capability?: RuntimeCapability;
  local?: boolean;
  native?: boolean;
  ios?: boolean;
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
  readonly pytorch = this.data.runtime === 'PyTorch';
  readonly keys: readonly RuntimeOptionKey[] = this.pytorch
    ? pytorchRuntimeOptionKeys
    : runtimeOptionKeys;
  readonly backends = [
    {label: 'Text', key: 'backend', threads: 'cpuThreads'},
    {label: 'Audio', key: 'audioBackend', threads: 'audioCpuThreads'},
    {label: 'Vision', key: 'visionBackend', threads: null},
  ] as const;
  readonly activationOptions = ['Default', ...activationDataTypes];
  readonly switchLabels = SWITCH_LABELS;
  readonly advanced = advancedRuntimeOptionFields;
  readonly extras = [
    {
      label: 'Prefill chunk size',
      key: 'prefillChunkSize',
      group: 'Batching',
      help: 'Maximum prefill chunk size (set_prefill_chunk_size); CPU backend with dynamic models only. Leave blank for the runtime default.',
    },
    {
      label: 'Context length',
      key: 'contextLength',
      group: 'Context',
      help: 'Maximum context capacity, including input and output. Leave blank for the runtime default; the model must support this capacity.',
    },
  ] as const;
  constructor() {
    for (const key of this.keys) this.draft[key] ??= '';
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
  cancel() {
    this.ref.close(undefined);
  }
  /** Whether this runtime, and the connected Runner build, take the option at all. */
  supported(key: RuntimeOptionKey) {
    if (
      this.pytorch &&
      !pytorchRuntimeOptionKeys.includes(key as (typeof pytorchRuntimeOptionKeys)[number])
    )
      return false;
    const supported = this.data.capability?.supported_options;
    if (supported) return supported.includes(key);
    if (this.data.native) return ['backend', 'contextLength'].includes(key);
    return (
      !this.data.local ||
      ['backend', 'cpuThreads', 'contextLength'].includes(key) ||
      (this.pytorch && key === 'precision')
    );
  }
  /** Supported, and paired with the selected text backend (CPU- or GPU-only members). */
  editable(key: RuntimeOptionKey) {
    if (!this.supported(key)) return false;
    const field = this.advanced.find((f) => f.key === key);
    if (field?.backend) return (this.draft.backend || 'CPU') === field.backend;
    if (key === 'prefillChunkSize' || key === 'cpuThreads') return this.draft.backend === 'CPU';
    return true;
  }
  backendOptions(key: RuntimeOptionKey) {
    if (this.data.ios) return ['CPU'];
    if (key === 'backend') {
      const backends =
        this.data.capability?.backends ??
        (this.pytorch || this.data.native ? ['CPU'] : ['CPU', 'GPU']);
      return this.data.native ? backends : ['Default', ...backends];
    }
    return ['Default', 'CPU', 'GPU'];
  }
  contextOptions() {
    return (this.data.capability?.contextLengths ?? [1024, 4096]).map(String);
  }
  precisionOptions() {
    return (this.data.capability?.precisions ?? ['', 'float32', 'float16', 'bfloat16']).map(
      (value) => value || 'Default',
    );
  }
  switchLabel(key: RuntimeOptionKey) {
    return this.draft[key] === 'on' ? 'On' : this.draft[key] === 'off' ? 'Off' : 'Default';
  }
  setSwitch(key: RuntimeOptionKey, label: string) {
    this.draft[key] = label === 'On' ? 'on' : label === 'Off' ? 'off' : '';
  }
  selectBackend(key: 'backend' | 'audioBackend' | 'visionBackend', value: string) {
    this.draft[key] = value === 'Default' ? '' : value;
    if (key === 'backend') this.clearUnpaired();
    if (key === 'audioBackend' && this.draft.audioBackend !== 'CPU')
      this.draft.audioCpuThreads = '';
  }
  /** Members paired with the other text backend fall back to the runtime default. */
  private clearUnpaired() {
    for (const key of this.keys)
      if (!this.editable(key) && this.supported(key)) this.draft[key] = '';
  }
  canCopy() {
    return !this.data.otherRuntime || this.data.otherRuntime === this.data.runtime;
  }
  hasValue(key: RuntimeOptionKey) {
    return this.draft[key] !== '' && this.draft[key] != null;
  }
  reset(key: RuntimeOptionKey) {
    if (!this.supported(key)) return;
    this.draft[key] = this.data.native
      ? key === 'backend'
        ? (this.backendOptions(key)[0] ?? 'CPU')
        : key === 'contextLength'
          ? (this.contextOptions()[0] ?? '1024')
          : ''
      : '';
    if (key === 'backend') this.clearUnpaired();
    if (key === 'audioBackend') this.draft.audioCpuThreads = '';
  }
  copy() {
    if (!this.canCopy()) return;
    this.previous = {...this.draft};
    for (const key of this.keys) {
      if (this.supported(key)) this.draft[key] = this.data.otherRun[key] ?? '';
    }
    if (!this.backendOptions('backend').includes(this.draft.backend || 'Default'))
      this.reset('backend');
    if (this.data.native && !this.contextOptions().includes(this.draft.contextLength ?? ''))
      this.reset('contextLength');
    this.clearUnpaired();
    if (this.pytorch && !this.precisionOptions().includes(this.draft.precision || 'Default'))
      this.draft.precision = '';
    this.copied.set(true);
  }
  undo() {
    if (this.previous) this.draft = {...this.previous};
    this.previous = null;
    this.copied.set(false);
  }
  close() {
    const form = this.element.nativeElement.querySelector('form')!;
    if (!form.reportValidity()) return;
    this.ref.close(
      Object.fromEntries(this.keys.map((key) => [key, String(this.draft[key] ?? '')])),
    );
  }
}
