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

import {Component, computed, ElementRef, inject, OnDestroy, signal} from '@angular/core';
import {FormsModule} from '@angular/forms';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {isTerminal} from '../../server/contracts/jobs';
import type {ChatCapabilities, OptionDescription} from '../../server/contracts/runtime';
import {SessionsApi} from '../../server/sessions_api';
import {WorkspaceStateService} from '../../state/workspace_state_service';
import {ConfigPicker} from '../../widgets/config_picker/config_picker';
import {chatOptionGroups, type OptionGroup, switchLabel} from '../chat_configuration';

type Values = Record<string, string | number | null>;
/** Why no chat starts while a task of the Session runs: the server's word for it. */
const BUSY = 'Wait for the current task to finish.';

/**
 * The New chat form of a Session's page: each side's Chat settings, Reference and Target side by
 * side in the server's groups, a group copied from one side to the other, and Start chat. It
 * starts from the settings of the Session's last chat. Starting a chat ends the live one, which
 * stays readable; the page then shows the new chat.
 */
@Component({
  selector: 'chat-configuration',
  imports: [ConfigPicker, FormsModule, MatButtonModule, MatIconModule],
  templateUrl: './chat_configuration.ng.html',
  styleUrl: './chat_configuration.scss',
})
export class ChatConfiguration implements OnDestroy {
  readonly workspace = inject(WorkspaceStateService);
  private readonly element = inject<ElementRef<HTMLElement>>(ElementRef);
  private readonly sessionsApi = inject(SessionsApi);
  readonly copied = signal(new Set<string>());
  private readonly copyTimers = new Map<string, ReturnType<typeof setTimeout>>();
  /** The Session, as the list showed it when the form opened. */
  readonly session = this.workspace.session();
  /** The Session's last chat, whose settings the form starts from, if it has one. */
  readonly last = this.session?.chats.at(-1) ?? null;
  readonly sides = ['ref', 'target'];
  readonly switchOptions = ['Model default', 'Enabled', 'Disabled'];
  /**
   * What each side may set, as the server describes it for the side's device, runner version
   * and run (GET /api/sessions/chat-options); null until it answers.
   */
  readonly offered = signal<ChatCapabilities | null>(null);
  readonly values: Record<string, Values> = structuredClone(
    this.last?.generation ?? {ref: {}, target: {}},
  );
  private readonly undoValues = new Map<string, {before: Values; applied: Values}>();
  /** Why each side's options could not be read, if they could not. */
  readonly error = signal('');
  /** The Session's last chat as the list shows it now. */
  private readonly lastNow = computed(() =>
    this.workspace.session()?.chats.find((chat) => chat.n === this.last?.n),
  );
  /** What starting a chat does to the live one, said only while it is live. */
  readonly endsLive = computed(() => {
    const live = this.lastNow();
    return live?.state === 'live'
      ? `Starting it ends ${live.title}; ${live.title} stays readable.`
      : '';
  });
  /** Why the last chat ended, while no chat is live. */
  readonly lastEnded = computed(() => {
    const last = this.lastNow();
    return last?.state === 'ended' && last.notice ? `${last.title} ended: ${last.notice}` : '';
  });
  /** The chat start this form follows, while it runs. */
  readonly starting = computed(() => {
    const job = this.workspace.jobs.job();
    return job?.operation === 'chat' && !isTerminal(job.status) ? job : null;
  });
  /**
   * Whether a chat may start now, and if not why: the server's `new_chat` action, and no task of
   * this page running (a turn among them).
   */
  readonly action = computed(() => {
    const allowed = this.workspace.session()?.actions.new_chat ?? {allowed: false, reason: ''};
    return allowed.allowed && this.workspace.runtime.busy()
      ? {allowed: false, reason: BUSY}
      : allowed;
  });
  /** While the request to start the chat is in flight, or the start runs. */
  readonly saving = computed(() => this.workspace.runtime.submitting() || !!this.starting());

  constructor() {
    for (const side of this.sides) {
      this.values[side] ??= {};
    }
    this.keepSupported();
    if (this.session) {
      void this.loadOffered(this.session.id);
    }
  }

  ngOnDestroy(): void {
    for (const timer of this.copyTimers.values()) {
      clearTimeout(timer);
    }
  }

  feedback(group: string, target: string): void {
    const key = group + target;
    clearTimeout(this.copyTimers.get(key));
    this.copied.update((v) => new Set([...v, key]));
    this.copyTimers.set(
      key,
      setTimeout(() => this.copied.update((v) => new Set([...v].filter((k) => k !== key))), 1500),
    );
  }

  /**
   * The settings the sides' devices describe, in their sections; those neither side can set
   * are left out.
   */
  get groups(): OptionGroup[] {
    return chatOptionGroups([
      ...this.descriptions('ref'),
      ...this.descriptions('target').filter((o) => !this.describedFor('ref', o.key)),
    ]).filter((group) =>
      group.keys.some((key) => this.sides.some((side) => this.supported(side, key))),
    );
  }

  /**
   * The settings a side's device describes; the server's table for the default runtime when the
   * side's is not described.
   */
  private descriptions(side: string): readonly OptionDescription[] {
    return (
      this.capability(side)?.chat_options ?? this.workspace.runtime.capability()?.chat_options ?? []
    );
  }

  private describedFor(side: string, key: string): OptionDescription | undefined {
    return this.descriptions(side).find((option) => option.key === key);
  }

  private described(key: string): OptionDescription | undefined {
    return this.describedFor('ref', key) ?? this.describedFor('target', key);
  }

  label(key: string): string {
    return this.described(key)?.label ?? key;
  }

  help(key: string): string {
    return this.described(key)?.help ?? '';
  }

  isSwitch(key: string): boolean {
    return this.described(key)?.type === 'switch';
  }

  /** Whether the setting is text, such as the system prompt. */
  isText(key: string): boolean {
    return this.described(key)?.type === 'text';
  }

  /**
   * The input's range, as the server describes it for the side: a reply no longer than the
   * side's context among it.
   */
  range(side: string, key: string): {min: number; max: number; step: number | 'any'} {
    const option = this.describedFor(side, key);
    return {
      min: option?.minimum ?? 0,
      max: option?.maximum ?? Number.MAX_SAFE_INTEGER,
      step: option?.type === 'integer' ? 1 : 'any',
    };
  }

  /** A text setting's longest length, as the server describes it for the side. */
  textLimit(side: string, key: string): number | null {
    return this.describedFor(side, key)?.maximum ?? null;
  }

  /**
   * What the server says of a side's options, e.g. that they are its defaults until the side's
   * runner version has run on its device; '' for nothing.
   */
  notice(side: string): string {
    return this.capability(side)?.notice ?? '';
  }

  /** The default the side's sampler gives a setting, as the server describes it. */
  private defaultFor(side: string, key: string) {
    const option = this.describedFor(side, key);
    return option?.defaults?.[this.sampler(side)]?.value ?? option?.default;
  }

  placeholder(side: string, key: string): string {
    if (this.required(side, key)) {
      return 'Required';
    }
    const fallback = this.defaultFor(side, key);
    return fallback == null ? 'Default' : 'Default (' + fallback + ')';
  }

  /**
   * Whether the side's sampler uses a sampling value it has no default for, as Top-K does Top K
   * and Top-K + Top-P Top P: the server starts no chat without it.
   */
  required(side: string, key: string): boolean {
    const option = this.describedFor(side, key);
    const sampler = this.sampler(side);
    return (
      !!option?.samplers?.includes(sampler) &&
      option.fixed?.[sampler] === undefined &&
      this.defaultFor(side, key) == null
    );
  }

  /**
   * What the field says beneath it: where its default comes from, or that the side's sampler
   * needs it while it is empty; '' for nothing.
   */
  note(side: string, key: string): string {
    if (this.required(side, key)) {
      const value = this.values[side][key];
      return value == null || value === ''
        ? `${this.samplerName(side, this.sampler(side))} needs a ${this.label(key)}.`
        : '';
    }
    return this.defaultSource(side, key);
  }

  /**
   * Where the default the side's sampler gives a setting comes from, as the server says; '' when
   * the server says nothing of it.
   */
  defaultSource(side: string, key: string): string {
    const described = this.describedFor(side, key)?.defaults?.[this.sampler(side)];
    return described ? 'Default ' + described.value + ': ' + described.source : '';
  }

  /** Why neither side can set a setting, when both say the same; '' otherwise. */
  sharedUnavailable(key: string): string {
    const [ref, target] = this.sides;
    if (this.supported(ref, key) || this.supported(target, key)) {
      return '';
    }
    const reason = this.unavailableReason(ref, key);
    return reason === this.unavailableReason(target, key) ? reason : '';
  }

  /** Why a side cannot set a setting, as the server says. */
  unavailableReason(side: string, key: string): string {
    return this.describedFor(side, key)?.reason || 'Unavailable for this runtime';
  }

  /** The side's sampler: the one chosen, or the server's default. */
  private sampler(side: string): string {
    return String(
      this.values[side]['samplerType'] || this.describedFor(side, 'samplerType')?.default || '',
    );
  }

  /**
   * What a sampler that does not use a sampling value sends for it, as the server says, if it
   * does.
   */
  fixedValue(side: string, key: string) {
    return this.describedFor(side, key)?.fixed?.[this.sampler(side)];
  }

  switchLabel(side: string, key: string): string {
    return switchLabel(this.values[side][key]);
  }

  setSwitch(side: string, key: string, value: string): void {
    this.values[side][key] = value === 'Enabled' ? 'on' : value === 'Disabled' ? 'off' : null;
  }

  /** What the server calls a sampler of the side's device. */
  private samplerName(side: string, sampler: string): string {
    return this.describedFor(side, 'samplerType')?.choice_labels?.[sampler] ?? sampler;
  }

  /** The samplers the side's device describes, by the server's names of them. */
  samplerOptions(side: string): string[] {
    return [
      'Default',
      ...(this.describedFor(side, 'samplerType')?.choices ?? []).map((sampler) =>
        this.samplerName(side, sampler),
      ),
    ];
  }

  samplerLabel(side: string): string {
    const sampler = this.values[side]['samplerType'];
    return sampler ? this.samplerName(side, String(sampler)) : 'Default';
  }

  /**
   * A sampling value either sampler does not use goes back to the default: the one sent is then
   * the new sampler's, never a value fixed by the old one.
   */
  setSampler(side: string, name: string): void {
    const before = this.sampler(side);
    const value = (this.describedFor(side, 'samplerType')?.choices ?? []).find(
      (sampler) => this.samplerName(side, sampler) === name,
    );
    this.values[side]['samplerType'] = value ?? null;
    const after = this.sampler(side);
    for (const option of this.descriptions(side)) {
      if (
        option.samplers &&
        !(option.samplers.includes(before) && option.samplers.includes(after))
      ) {
        this.values[side][option.key] = null;
      }
    }
  }

  private run(side: string) {
    return this.session?.runs.find((run) => run.id === side);
  }

  /** What the side may set: what the server offers it, or else the runtime's table. */
  private capability(side: string) {
    const run = this.run(side);
    if (!run) {
      return undefined;
    }
    const runtime = run.runtime ?? 'LiteRT-LM';
    const offered = this.offered()?.[side];
    return offered?.id === runtime ? offered : this.workspace.runtime.capability(runtime);
  }

  /** Reads what each side may set; until then, and if it fails, the runtime's table decides. */
  async loadOffered(id: string): Promise<void> {
    try {
      this.offered.set(await this.sessionsApi.chatOptions(id));
      this.keepSupported();
    } catch (e) {
      this.error.set(
        "Each side's options could not be read; the server's defaults are shown: " +
          (e instanceof Error ? e.message : String(e)),
      );
    }
  }

  /** Drops the settings a side cannot set. */
  private keepSupported(): void {
    for (const side of this.sides) {
      for (const key of Object.keys(this.values[side])) {
        if (!this.supported(side, key)) {
          delete this.values[side][key];
        }
      }
    }
  }

  /** Whether the side may set the Chat option, as the server describes its device. */
  supported(side: string, key: string): boolean {
    return this.describedFor(side, key)?.available ?? false;
  }

  canCopy(group: {keys: string[]}, target: string): boolean {
    const source = target === 'ref' ? 'target' : 'ref';
    return group.keys.some((key) => this.supported(target, key) && this.supported(source, key));
  }

  value(side: string, key: string) {
    return this.values[side]?.[key] ?? 'Default';
  }

  copy(group: {name: string; keys: string[]}, target: string): void {
    if (!this.canCopy(group, target)) {
      return;
    }
    const source = target === 'ref' ? 'target' : 'ref';
    const keys = group.keys.filter(
      (key) => this.supported(target, key) && this.supported(source, key),
    );
    const before = Object.fromEntries(keys.map((k) => [k, this.values[target][k] ?? null]));
    const applied = Object.fromEntries(keys.map((k) => [k, this.values[source][k] ?? null]));
    this.undoValues.set(group.name + target, {before, applied});
    Object.assign(this.values[target], applied);
    this.feedback(group.name, target);
  }

  canUndo(group: string, target: string): boolean {
    return this.undoValues.has(group + target);
  }

  undo(group: string, target: string): void {
    const saved = this.undoValues.get(group + target);
    if (!saved) {
      return;
    }
    for (const key of Object.keys(saved.before)) {
      if ((this.values[target][key] ?? null) === (saved.applied[key] ?? null)) {
        this.values[target][key] = saved.before[key];
      }
    }
    this.undoValues.delete(group + target);
  }

  resetField(side: string, key: string): void {
    this.values[side][key] = null;
  }

  isDefault(side: string, key: string): boolean {
    return this.values[side][key] == null || this.values[side][key] === '';
  }

  /** What each side's chat runs with: the settings set, of those the side may set. */
  generation(): Record<string, Record<string, string | number>> {
    return Object.fromEntries(
      this.sides.map((side) => [
        side,
        Object.fromEntries(
          Object.entries(this.values[side]).filter(
            (entry): entry is [string, string | number] =>
              this.supported(side, entry[0]) && entry[1] !== null && entry[1] !== '',
          ),
        ),
      ]),
    );
  }

  /**
   * Starts the chat: the server ends the live one, if any. Why it did not start, the server's
   * words, shows under the form (RuntimeService.error).
   */
  async start(): Promise<void> {
    if (!this.session || this.saving() || !this.action().allowed) {
      return;
    }
    const invalid = Array.from(
      this.element.nativeElement.querySelectorAll<HTMLInputElement>('input[type=number]'),
    ).find((input) => !input.disabled && !input.checkValidity());
    if (invalid) {
      invalid.reportValidity();
      invalid.focus();
      return;
    }
    await this.workspace.runtime.startChat(this.generation());
  }
}
