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

import {Component, ElementRef, inject, OnDestroy, signal} from '@angular/core';
import {FormsModule} from '@angular/forms';
import {MatButtonModule} from '@angular/material/button';
import {
  MAT_DIALOG_DATA,
  MatDialogModule,
  MatDialogRef,
} from '@angular/material/dialog';
import {MatIconModule} from '@angular/material/icon';
import {WorkspaceStateService} from '../../../app/workspace_state_service';
import {ConfigPicker} from '../../../shared/config_picker/config_picker';
import {
  GENERATION_GROUPS,
  GENERATION_LABELS,
  GENERATION_RANGES,
} from '../../sessions/chat_configuration';
type Values = Record<string, string | number | null>;
@Component({
  selector: 'chat-configuration',
  imports: [
    ConfigPicker,
    FormsModule,
    MatDialogModule,
    MatButtonModule,
    MatIconModule,
  ],
  templateUrl: './chat_configuration.ng.html',
  styleUrl: './chat_configuration.scss',
})
export class ChatConfiguration implements OnDestroy {
  private readonly element = inject<ElementRef<HTMLElement>>(ElementRef);
  readonly copied = signal(new Set<string>());
  private readonly copyTimers = new Map<
    string,
    ReturnType<typeof setTimeout>
  >();
  ngOnDestroy() {
    for (const timer of this.copyTimers.values()) clearTimeout(timer);
  }
  feedback(group: string, target: string) {
    const key = group + target;
    clearTimeout(this.copyTimers.get(key));
    this.copied.update((v) => new Set([...v, key]));
    this.copyTimers.set(
      key,
      setTimeout(
        () =>
          this.copied.update((v) => new Set([...v].filter((k) => k !== key))),
        1500,
      ),
    );
  }
  readonly workspace = inject(WorkspaceStateService);
  readonly ref = inject(MatDialogRef<ChatConfiguration>);
  readonly session = this.workspace.runtime.selected();
  /** Native Runners accept fewer output tokens than the generic generation bounds. */
  readonly ranges: Record<
    string,
    {min: number; max: number; step: number | string}
  > = (() => {
    const limits = (this.session?.runs ?? [])
      .map(
        (run) =>
          this.workspace.runtime.capability(run.runtime ?? 'LiteRT-LM')
            ?.max_output_tokens,
      )
      .filter((limit): limit is number => typeof limit === 'number');
    const base = GENERATION_RANGES['maxOutputTokens'];
    return {
      ...GENERATION_RANGES,
      maxOutputTokens: {
        ...base,
        max: limits.length ? Math.min(base.max, ...limits) : base.max,
      },
    };
  })();
  readonly readOnly = false;
  readonly sides = ['ref', 'target'];
  readonly create = !!inject(MAT_DIALOG_DATA, {optional: true})?.create;
  get groups() {
    return GENERATION_GROUPS.filter((group) =>
      group.keys.some((key) =>
        this.sides.some((side) => this.supported(side, key)),
      ),
    );
  }
  readonly labels = GENERATION_LABELS;
  readonly thinkingOptions = ['Model default', 'Enabled', 'Disabled'];
  thinkingLabel(side: string) {
    return this.values[side]['thinking'] === 'on'
      ? 'Enabled'
      : this.values[side]['thinking'] === 'off'
        ? 'Disabled'
        : 'Model default';
  }
  setThinking(side: string, value: string) {
    this.values[side]['thinking'] =
      value === 'Enabled' ? 'on' : value === 'Disabled' ? 'off' : null;
  }
  readonly values: Record<string, Values> = structuredClone(
    this.session?.generation ?? {ref: {}, target: {}},
  );
  private readonly undoValues = new Map<
    string,
    {before: Values; applied: Values}
  >();
  readonly error = signal('');
  readonly saving = signal(false);
  onlyDifferences = false;
  readonly runtimeFields = [
    'runtime',
    'backend',
    'precision',
    'artifact',
    'cpuThreads',
    'audioBackend',
    'audioCpuThreads',
    'visionBackend',
    'forceF32',
    'prefillBatchSizes',
    'contextLength',
  ];
  constructor() {
    for (const side of this.sides) {
      this.values[side] ??= {};
      if (this.supported(side, 'thinking'))
        this.values[side]['thinking'] ??= null;
      else {
        delete this.values[side]['thinking'];
        delete this.values[side]['thinkingBudget'];
      }
    }
  }
  supported(side: string, key: string) {
    return (
      this.session?.runs.find((run) => run.id === side)?.runtime !==
        'PyTorch' || !['thinking', 'thinkingBudget'].includes(key)
    );
  }
  canCopy(group: {keys: string[]}, target: string) {
    const source = target === 'ref' ? 'target' : 'ref';
    return group.keys.some(
      (key) => this.supported(target, key) && this.supported(source, key),
    );
  }
  value(side: string, key: string) {
    return (
      this.values[side]?.[key] ?? (this.readOnly ? 'Not recorded' : 'Default')
    );
  }
  copy(group: {name: string; keys: string[]}, target: string) {
    if (!this.canCopy(group, target)) return;
    const source = target === 'ref' ? 'target' : 'ref';
    const keys = group.keys.filter(
      (key) => this.supported(target, key) && this.supported(source, key),
    );
    const before = Object.fromEntries(
      keys.map((k) => [k, this.values[target][k] ?? null]),
    );
    const applied = Object.fromEntries(
      keys.map((k) => [k, this.values[source][k] ?? null]),
    );
    this.undoValues.set(group.name + target, {before, applied});
    Object.assign(this.values[target], applied);
    this.feedback(group.name, target);
  }
  canUndo(group: string, target: string) {
    return this.undoValues.has(group + target);
  }
  undo(group: string, target: string) {
    const saved = this.undoValues.get(group + target);
    if (!saved) return;
    for (const key of Object.keys(saved.before)) {
      if ((this.values[target][key] ?? null) === (saved.applied[key] ?? null))
        this.values[target][key] = saved.before[key];
    }
    this.undoValues.delete(group + target);
  }
  resetField(side: string, key: string) {
    this.values[side][key] = null;
  }
  isDefault(side: string, key: string) {
    return this.values[side][key] == null || this.values[side][key] === '';
  }
  reset() {
    for (const side of this.sides) this.values[side] = {};
    this.undoValues.clear();
  }
  async save() {
    if (!this.session || this.readOnly || this.saving()) return;
    const invalid = Array.from(
      this.element.nativeElement.querySelectorAll<HTMLInputElement>(
        'input[type=number]',
      ),
    ).find((input) => !input.disabled && !input.checkValidity());
    if (invalid) {
      invalid.reportValidity();
      invalid.focus();
      return;
    }
    this.saving.set(true);
    this.ref.disableClose = true;
    this.error.set('');
    try {
      const generation = Object.fromEntries(
        this.sides.map((side) => [
          side,
          Object.fromEntries(
            Object.entries(this.values[side]).filter(
              ([key, v]) => this.supported(side, key) && v !== null && v !== '',
            ),
          ),
        ]),
      );
      if (this.create) {
        const chat = await this.workspace.sessions.manage('chat', {
          id: this.workspace.selectedId()!,
          name: 'New chat ' + this.workspace.chats().length,
          generation,
        });
        await this.workspace.chooseChat(chat.id);
      } else {
        await this.workspace.sessions.manage('chat-config', {
          id: this.session.id,
          generation,
        });
      }
      this.ref.close();
    } catch (e) {
      this.error.set(e instanceof Error ? e.message : String(e));
    } finally {
      this.saving.set(false);
      this.ref.disableClose = false;
    }
  }
}
