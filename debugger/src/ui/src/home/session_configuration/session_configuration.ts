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

import {ChangeDetectionStrategy, Component, computed, inject, input, signal} from '@angular/core';
import {DatePipe} from '@angular/common';
import {MAT_DIALOG_DATA, MatDialogModule} from '@angular/material/dialog';
import {MatButtonModule} from '@angular/material/button';
import {MatCheckboxModule} from '@angular/material/checkbox';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';
import {runtimeOptionValue, sessionOptions, type RuntimeOptionKey} from '../runtime_options';
import {CaptureStore} from '../../server/capture_store';
import {RuntimeService} from '../../server/runtime_service';
import {SessionSummary} from '../../server/contracts/sessions';
import {SessionListService} from '../../server/session_list_service';
@Component({
  selector: 'session-configuration',
  imports: [
    DatePipe,
    MatDialogModule,
    MatButtonModule,
    MatIconModule,
    MatTooltipModule,
    MatCheckboxModule,
  ],
  templateUrl: './session_configuration.ng.html',
  styleUrl: './session_configuration.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SessionConfiguration {
  private readonly dialogSession = inject<SessionSummary | null>(MAT_DIALOG_DATA, {optional: true});
  readonly sessionInput = input<SessionSummary | null>(null, {alias: 'session'});
  readonly embedded = input(false);
  readonly sessions = inject(SessionListService);
  private readonly capture = inject(CaptureStore);
  private readonly runtime = inject(RuntimeService);
  readonly expanded = signal(false);
  readonly onlyDifferences = signal(true);
  readonly identityFields = [
    {label: 'Device', key: 'device'},
    {label: 'Runtime', key: 'runtime'},
    {label: 'Model Source', key: 'source'},
    {label: 'Model Artifact', key: 'artifact'},
  ];
  readonly visibleRuntimeFields = computed(() =>
    this.runtimeFields.filter(
      (f) =>
        !this.sameRuntime ||
        !this.onlyDifferences() ||
        this.value('ref', f.key) !== this.value('target', f.key),
    ),
  );

  /** A runtime's Session options, as the server describes them (option_catalog.py). */
  private described(runtime?: string) {
    return sessionOptions(this.runtime.capability(runtime ?? 'LiteRT-LM'));
  }

  get session(): SessionSummary {
    const original = this.sessionInput() ?? this.dialogSession;
    if (!original) {
      throw new Error('SessionConfiguration requires a session input or dialog data.');
    }
    return this.sessions.items().find((item) => item.id === original.id) ?? original;
  }

  get sameRuntime(): boolean {
    return this.value('ref', 'runtime') === this.value('target', 'runtime');
  }

  /** The Session options of the sides' runtimes, in the order the server describes them. */
  get runtimeFields(): Array<{key: string; label: string}> {
    const fields = new Map<string, string>();
    for (const run of this.session.runs) {
      for (const {key, label} of this.described(run.runtime)) {
        if (!fields.has(key)) {
          fields.set(key, label);
        }
      }
    }
    return [...fields].map(([key, label]) => ({key, label}));
  }

  /** Session settings that differ between the sides; shown, never blocked (see configuration overview). */
  get differenceNote(): string {
    if (!this.sameRuntime) {
      return '';
    }
    const keys = this.runtimeFields.filter(
      (f) => this.value('ref', f.key) !== this.value('target', f.key),
    );
    if (!keys.length) {
      return '';
    }
    return (
      (keys.length === 1
        ? 'One Session setting differs'
        : keys.length + ' Session settings differ') +
      ' between Reference and Target: ' +
      keys.map((f) => f.label).join(', ') +
      '. Numeric differences may come from these settings, not only from the model or backend.'
    );
  }

  /**
   * A side's runner version: the LiteRT-LM commit its capture recorded when the attached capture
   * is a chat of this Session, else the version it chose.
   */
  runnerVersion(side: string): string {
    const run = this.session.runs.find((r) => r.id === side);
    const chosen = run?.runnerBuild ?? '';
    // A runner of another runtime has no LiteRT-LM commit: it is shown by its build's label.
    if ((run?.runtime ?? 'LiteRT-LM') !== 'LiteRT-LM') {
      return (
        this.runtime.buildCatalogs()[run?.device ?? 'local']?.builds.find((b) => b.id === chosen)
          ?.label ||
        chosen ||
        'Not chosen'
      );
    }
    const recorded = this.recordedVersion(side);
    const development = chosen === 'dev' ? 'Development build' : '';
    if (recorded) {
      return ['LiteRT-LM ' + recorded.slice(0, 12), development].filter(Boolean).join(' · ');
    }
    if (development) {
      return development;
    }
    if (chosen) {
      return 'LiteRT-LM ' + chosen.slice(0, 12);
    }
    return run?.device && run.device !== 'local' ? 'The Runner App installed' : 'Not chosen';
  }

  /** The whole commit behind runnerVersion(), for its tooltip. */
  runnerCommit(side: string): string {
    const run = this.session.runs.find((r) => r.id === side);
    const chosen = run?.runnerBuild ?? '';
    if ((run?.runtime ?? 'LiteRT-LM') !== 'LiteRT-LM') {
      return '';
    }
    return this.recordedVersion(side) || (chosen === 'dev' ? '' : chosen);
  }

  private recordedVersion(side: string): string {
    const captured = this.capture.session();
    if (!captured || captured.capture_id !== this.session.id) {
      return '';
    }
    const version = captured.runs.find((r) => r.id === side)?.provenance?.['runtime_version'];
    return typeof version === 'string' ? version : '';
  }

  value(side: string, key: string): string {
    const run = this.session.runs.find((r) => r.id === side) as Record<string, unknown> | undefined;
    // A model from Hugging Face is shown by its repository, as the server recorded it.
    const huggingFace = run?.['huggingface'] as {repo?: string} | undefined;
    if (key === 'source' && huggingFace?.repo) {
      return 'Hugging Face · ' + huggingFace.repo;
    }
    const value = run?.[key];
    if (!this.session.has_capture && this.runtimeFields.some((field) => field.key === key)) {
      // An option the side's runtime does not describe does not apply to it.
      const option = this.described(run?.['runtime'] as string | undefined).find(
        (described) => described.key === key,
      );
      return option ? runtimeOptionValue(run ?? {}, key as RuntimeOptionKey, option) : '—';
    }
    return value == null || value === ''
      ? 'Not captured'
      : typeof value === 'object'
        ? JSON.stringify(value)
        : String(value);
  }
}
