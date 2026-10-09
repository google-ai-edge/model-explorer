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

import {DatePipe} from '@angular/common';
import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
  input,
  signal,
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatCheckboxModule} from '@angular/material/checkbox';
import {MAT_DIALOG_DATA, MatDialogModule} from '@angular/material/dialog';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';
import {SessionSummary} from '../../../data/contracts/session_summary';
import {SessionListService} from '../../../data/session_list_service';
import {
  sessionConfigurationLocked,
  sessionStatusLabel,
} from '../../../data/session_status';
import {
  pytorchRuntimeOptionKeys,
  runtimeOptionFields,
  runtimeOptionKeys,
  runtimeOptionValue,
} from '../runtime_options';
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
  private readonly dialogSession = inject<SessionSummary | null>(
    MAT_DIALOG_DATA,
    {optional: true},
  );
  readonly sessionInput = input<SessionSummary | null>(null, {
    alias: 'session',
  });
  readonly embedded = input(false);
  readonly sessions = inject(SessionListService);
  readonly statusLabel = sessionStatusLabel;
  get session() {
    const original = this.sessionInput() ?? this.dialogSession!;
    return (
      this.sessions.allItems().find((item) => item.id === original.id) ??
      original
    );
  }
  get canEdit() {
    return (
      !!this.sessions.listing()?.capabilities.create &&
      !this.session.has_capture &&
      !sessionConfigurationLocked(this.session)
    );
  }
  get canDuplicate() {
    return (
      !!this.sessions.listing()?.capabilities.duplicate &&
      !sessionConfigurationLocked(this.session)
    );
  }
  get editUnavailableReason() {
    if (this.canEdit) return '';
    if (this.session.has_capture)
      return 'Captured sessions cannot be edited. Duplicate to change configuration.';
    return sessionConfigurationLocked(this.session)
      ? 'End the session before changing its configuration.'
      : 'Session configuration changes are unavailable.';
  }
  get duplicateUnavailableReason() {
    return this.canDuplicate
      ? ''
      : sessionConfigurationLocked(this.session)
        ? 'End the session before duplicating its configuration.'
        : 'Session duplication is unavailable.';
  }
  readonly expanded = signal(false);
  readonly onlyDifferences = signal(true);
  get sameRuntime() {
    return this.value('ref', 'runtime') === this.value('target', 'runtime');
  }
  readonly identityFields = [
    {label: 'Device', key: 'device'},
    {label: 'Runtime', key: 'runtime'},
    {label: 'Model Source', key: 'source'},
    {label: 'Model Artifact', key: 'artifact'},
  ];
  get runtimeFields() {
    const onlyPyTorch = this.session.runs.every(
      (run) => run.runtime === 'PyTorch',
    );
    const fields = onlyPyTorch
      ? runtimeOptionFields.filter((field) =>
          (pytorchRuntimeOptionKeys as readonly string[]).includes(field.key),
        )
      : runtimeOptionFields;
    const precision = this.session.has_capture
      ? 'Captured precision'
      : this.session.runs.some((run) => run.runtime === 'PyTorch')
        ? 'Model dtype'
        : '';
    return [
      ...fields,
      ...(precision ? [{label: precision, key: 'precision'}] : []),
    ];
  }
  readonly visibleRuntimeFields = computed(() =>
    this.runtimeFields.filter(
      (f) =>
        !this.sameRuntime ||
        !this.onlyDifferences() ||
        this.value('ref', f.key) !== this.value('target', f.key),
    ),
  );
  value(side: string, key: string) {
    const run = this.session.runs.find((r) => r.id === side) as
      | Record<string, unknown>
      | undefined;
    const value = run?.[key];
    if (key === 'precision' && !this.session.has_capture)
      return run?.['runtime'] === 'PyTorch'
        ? value == null || value === '' || value === 'default'
          ? 'Default'
          : String(value)
        : '—';
    if (
      !this.session.has_capture &&
      (runtimeOptionKeys as readonly string[]).includes(key)
    )
      return runtimeOptionValue(
        run ?? {},
        key as (typeof runtimeOptionKeys)[number],
      );
    return value == null || value === ''
      ? 'Not captured'
      : typeof value === 'object'
        ? JSON.stringify(value)
        : String(value);
  }
}
