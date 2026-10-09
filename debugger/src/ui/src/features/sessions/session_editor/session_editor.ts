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

import {
  AfterViewInit,
  ChangeDetectorRef,
  Component,
  computed,
  ElementRef,
  inject,
  input,
  OnDestroy,
  OnInit,
  output,
  signal,
} from '@angular/core';
import {FormsModule} from '@angular/forms';
import {MatButtonModule} from '@angular/material/button';
import {MatDialog} from '@angular/material/dialog';
import {MatIconModule} from '@angular/material/icon';
import {MatMenuModule} from '@angular/material/menu';
import {MatTooltipModule} from '@angular/material/tooltip';
import type {RunnerDevice} from '../../../data/contracts/runtime';
import {
  SessionConfig,
  SessionSummary,
} from '../../../data/contracts/session_summary';
import {RuntimeService} from '../../../data/runtime_service';
import {SessionListService} from '../../../data/session_list_service';
import {
  copiedSessionName,
  sessionConfigurationLocked,
} from '../../../data/session_status';
import {ConfigPicker} from '../../../shared/config_picker/config_picker';
import {formatGiB} from '../../../shared/format/format';
import {HfFile, HfFilePicker} from '../hf_file_picker/hf_file_picker';
import {
  pytorchRuntimeOptionKeys,
  runtimeOptionFields,
  runtimeOptionKeys,
  runtimeOptionValue,
} from '../runtime_options';
import {TapPicker} from '../tap_picker/tap_picker';
export interface EditorRequest {
  session?: SessionSummary;
  mode: 'create' | 'update' | 'duplicate';
}
@Component({
  selector: 'session-editor',
  imports: [
    TapPicker,
    HfFilePicker,
    ConfigPicker,
    FormsModule,
    MatButtonModule,
    MatIconModule,
    MatTooltipModule,
    MatMenuModule,
  ],
  templateUrl: './session_editor.ng.html',
  styleUrl: './session_editor.scss',
})
export class SessionEditor implements OnInit, AfterViewInit, OnDestroy {
  readonly runtime = inject(RuntimeService);
  private readonly element: ElementRef<HTMLElement> = inject(ElementRef);
  ngAfterViewInit() {
    this.element.nativeElement
      .querySelector<HTMLElement>('#editor-title')
      ?.focus();
  }
  readonly request = input.required<EditorRequest>();
  readonly finished = output<boolean>();
  readonly created = output<string>();
  readonly sessions = inject(SessionListService);
  readonly saving = signal(false);
  readonly error = signal('');
  readonly uploading = signal<string | null>(null);
  readonly uploadProgress = signal<number | null>(null);
  readonly uploadLabel = computed(() => {
    const progress = this.uploadProgress();
    return progress === null
      ? 'Uploading model file…'
      : `Uploading model file · ${Math.round(progress * 100)}%`;
  });
  readonly models = computed(() => [
    ...new Set(
      [
        'Gemma 4 E2B',
        'Qwen3 0.6B',
        this.sessions.listing()?.configuration_template?.model,
        ...this.sessions.items().map((s) => s.model),
        ...(this.runtime.capabilities()?.models.map((m) => m.name) ?? []),
      ].filter((model): model is string => !!model),
    ),
  ]);
  readonly runtimes = computed(() => [
    ...new Set([
      ...(this.runtime.capabilities()?.runtimes?.map((r) => r.id) ?? [
        'LiteRT-LM',
      ]),
      ...this.sessions
        .items()
        .flatMap((s) => s.runs.map((r) => r.runtime ?? ''))
        .filter(Boolean),
    ]),
  ]);
  private readonly dialogs = inject(MatDialog);
  private readonly changeDetector = inject(ChangeDetectorRef);
  customName = false;
  savedId: string | undefined;
  private uploadRequest: AbortController | null = null;
  private alive = true;
  ngOnDestroy() {
    this.alive = false;
    this.uploadRequest?.abort();
  }
  cancelUpload() {
    this.uploadRequest?.abort();
  }
  private focusError() {
    requestAnimationFrame(() =>
      this.element.nativeElement
        .querySelector<HTMLElement>('[role="alert"]')
        ?.focus(),
    );
  }
  focusSelectedFile(run: SessionConfig['runs'][number]) {
    requestAnimationFrame(() =>
      this.element.nativeElement
        .querySelector<HTMLButtonElement>(
          '[aria-label="Clear ' +
            (run.id === 'ref' ? 'Reference' : 'Target') +
            ' file selection"]',
        )
        ?.focus(),
    );
  }
  async configure(run: SessionConfig['runs'][number], trigger: HTMLElement) {
    try {
      const {RuntimeOptions} = await import(
        '../runtime_options/runtime_options'
      );
      if (!this.alive) return;
      const other = this.config.runs.find((r) => r.id !== run.id)!;
      this.dialogs
        .open(RuntimeOptions, {
          data: {
            side: run.id === 'ref' ? 'Reference' : 'Target',
            other: run.id === 'ref' ? 'Target' : 'Reference',
            runtime: run.runtime,
            otherRuntime: other.runtime,
            run,
            otherRun: other,
            capability: this.deviceCapability(run),
            local: !!this.sessions.listing()?.generation_available,
            native: run.runtime === 'LiteRT-LM',
            ios: this.runnerDevice(run.device)?.platform?.startsWith('iOS'),
          },
          width: '500px',
          maxWidth: 'calc(100vw - 32px)',
          maxHeight: 'calc(100dvh - 32px)',
          ariaLabel:
            (run.id === 'ref' ? 'Reference' : 'Target') +
            ' ' +
            (run.runtime ?? 'LiteRT-LM') +
            ' configuration',
          restoreFocus: false,
        })
        .afterClosed()
        .subscribe((options) => {
          if (!this.alive) return;
          if (options) {
            Object.assign(run, options);
            if (!this.customName) this.suggest();
            this.changeDetector.markForCheck();
          }
          trigger.focus();
        });
    } catch {
      this.error.set(
        'Runtime settings could not be loaded. Reload the page and try again.',
      );
      trigger.focus();
    }
  }
  differences(run: SessionConfig['runs'][number]) {
    const other = this.config.runs.find((r) => r.id !== run.id)!;
    const fields = this.isPyTorch(run)
      ? [
          ...runtimeOptionFields.filter((f) =>
            (pytorchRuntimeOptionKeys as readonly string[]).includes(f.key),
          ),
          {key: 'precision' as const, label: 'Model dtype'},
        ]
      : runtimeOptionFields;
    return fields
      .filter(
        (f) =>
          runtimeOptionValue(run, f.key) !== runtimeOptionValue(other, f.key),
      )
      .map((f) => ({label: f.label, value: runtimeOptionValue(run, f.key)}));
  }
  isPyTorch(run: SessionConfig['runs'][number]) {
    return run.runtime === 'PyTorch';
  }
  hasPyTorch() {
    return this.config.runs.some((run) => this.isPyTorch(run));
  }
  hasLiteRT() {
    return this.config.runs.some((run) => run.runtime === 'LiteRT-LM');
  }
  deviceCapability(run: SessionConfig['runs'][number]) {
    const cap = this.runtime.capability(run.runtime),
      runner = this.runnerDevice(run.device)?.runner,
      observed = runner?.runtimes?.find((r) => r.id === run.runtime);
    return cap
      ? {
          ...cap,
          ...(observed ? {backends: observed.backends} : {}),
          ...(run.runtime === 'LiteRT-LM' &&
          runner?.capabilities?.contextLengths
            ? {contextLengths: runner.capabilities.contextLengths}
            : {}),
        }
      : cap;
  }
  runtimeNotice(run: SessionConfig['runs'][number]) {
    if (this.sessions.listing()?.generation_available === false) return '';
    const capability = this.runtime.capability(run.runtime);
    return capability && !capability.available
      ? (run.runtime ?? 'LiteRT-LM') + ' is unavailable on this server.'
      : '';
  }
  saveNotice() {
    const available = this.sessions.listing()?.generation_available;
    if (available !== true) return '';
    if (this.request().mode === 'update')
      return this.config.tap_profile
        ? 'Saves the configuration and prepares selected outputs.'
        : 'Saves the configuration.';
    return this.config.tap_profile
      ? 'Creates the Session, prepares selected outputs and starts its Model Server. Enter the Session once it is on.'
      : 'Creates the Session and starts its Model Server. Enter the Session once it is on.';
  }
  localModels(run: SessionConfig['runs'][number]) {
    return (
      this.runtime
        .capabilities()
        ?.models.filter(
          (model) =>
            (model.runtime ?? 'LiteRT-LM') === (run.runtime ?? 'LiteRT-LM'),
        ) ?? []
    );
  }
  selectRuntime(run: SessionConfig['runs'][number], runtime: string) {
    if (run.runtime === runtime) return;
    run.runtime = runtime;
    run.artifact = '';
    run.repository = '';
    run.revision = '';
    run.sourceUrl = '';
    run.source = runtime === 'PyTorch' ? 'registered' : 'upload';
    run.device = 'local';
    delete run.runnerBuild;
    for (const key of runtimeOptionKeys) {
      if (key === 'forceF32') run.forceF32 = false;
      else run[key] = '';
    }
    delete run.precision;
    const backends = this.runtime.capability(runtime)?.backends ?? ['CPU'];
    run.backend = backends[0] ?? 'CPU';
    if (runtime === 'LiteRT-LM') run.contextLength = '1024';
    this.config.tap_profile = '';
    this.config.tap_points = {};
    this.scanReady.set({});
    void this.loadSelectedBuilds();
    if (!this.customName) this.suggest();
  }
  deviceLabel(id: string | undefined) {
    if (!id || id === 'Server host') id = 'local';
    const device = this.runnerDevice(id);
    if (id === 'local' || device?.id === 'local')
      return (
        'This computer · ' +
        (device ? this.runtime.deviceStatusLabel(device) : 'Unchecked')
      );
    return device
      ? device.name +
          ' · ' +
          (device.platform ?? 'iOS') +
          ' ' +
          (device.transport ?? 'USB') +
          ' · ' +
          this.runtime.deviceStatusLabel(device) +
          ' · ' +
          id.slice(-4)
      : 'Not discovered · ' + id.slice(-4);
  }
  runnerDevice(id: string | undefined) {
    return this.runtime
      .devices()
      .find(
        (d) =>
          d.id === (!id || id === 'Server host' ? 'local' : id) ||
          d.runnerId === id,
      );
  }
  deviceCheckedAt(value: string) {
    return new Date(value).toLocaleTimeString();
  }
  deviceEnvironment(id: string | undefined) {
    const runner = this.runnerDevice(id)?.runner;
    if (!runner) return '';
    const e = runner.environment ?? {};
    return [
      e.chip ?? e.modelIdentifier ?? e.architecture,
      e.physicalMemoryBytes != null ? formatGiB(e.physicalMemoryBytes) : null,
      ...(runner.runtimes?.map((r) => r.id + ' · ' + r.backends.join('/')) ??
        (runner.capabilities
          ? [runner.capabilities.runtime, ...runner.capabilities.backends]
          : [])),
    ]
      .filter(Boolean)
      .join(' · ');
  }
  deviceOptions(run: SessionConfig['runs'][number]) {
    return this.runtime
      .devices()
      .filter(
        (d) =>
          !this.isPyTorch(run) ||
          d.id === 'local' ||
          !!d.runner?.runtimes?.some((r) => r.id === 'PyTorch'),
      )
      .map((d) => this.deviceLabel(d.id));
  }
  deviceOccupancy(device: RunnerDevice) {
    const owner = device.runner?.owner;
    return device.runnerState === 'active'
      ? 'In use' +
          (owner
            ? ' · ' +
              (owner.sessionName || owner.sessionId) +
              ' · ' +
              (owner.isCurrentServer
                ? 'Current Server'
                : owner.serverName || owner.serverId)
            : ' · Owner unknown')
      : device.runnerState === 'not_running'
        ? 'Model Server off'
        : device.runnerState === 'idle'
          ? 'Model Server idle · Current build'
          : 'Model Server state unknown';
  }
  deviceDescriptions(run: SessionConfig['runs'][number]) {
    return Object.fromEntries(
      this.runtime
        .devices()
        .map((d) => [
          this.deviceLabel(d.id),
          this.deviceOccupancy(d) + (d.reason ? ' · ' + d.reason : ''),
        ]),
    );
  }
  unavailableDevices(run: SessionConfig['runs'][number]) {
    return this.runtime
      .devices()
      .filter(
        (d) =>
          !['not_running', 'idle'].includes(d.runnerState ?? 'unknown') ||
          d.status === 'checking' ||
          d.status === 'unsupported' ||
          d.status === 'unpaired',
      )
      .map((d) => this.deviceLabel(d.id));
  }
  async checkDevice(id: string) {
    await this.runtime.inspectDevice(id);
    await this.loadSelectedBuilds();
  }
  async refreshDevices() {
    await this.runtime.refreshDevices();
    await this.loadSelectedBuilds();
  }
  async loadSelectedBuilds() {
    const ids = [
      ...new Set(
        this.config.runs
          .map((run) => this.runnerDevice(run.device))
          .filter(
            (device): device is RunnerDevice =>
              !!device && device.runnerState === 'not_running',
          )
          .map((device) => device.id),
      ),
    ];
    await Promise.all(ids.map((id) => this.runtime.loadBuilds(id)));
    for (const run of this.config.runs) {
      const device = this.runnerDevice(run.device);
      if (device?.runnerState === 'idle') {
        run.runnerBuild = device.runner?.build?.id ?? device.runner?.buildId;
        continue;
      }
      const available =
        this.buildCatalog(run)?.builds.filter((build) => build.available) ?? [];
      if (!run.runnerBuild && available.length === 1)
        run.runnerBuild = available[0].id;
    }
    this.changeDetector.markForCheck();
  }
  buildCatalog(run: SessionConfig['runs'][number]) {
    const device = this.runnerDevice(run.device);
    return device ? this.runtime.buildCatalogs()[device.id] : undefined;
  }
  buildLabel(run: SessionConfig['runs'][number]) {
    return (
      this.buildCatalog(run)?.builds.find(
        (build) => build.id === run.runnerBuild,
      )?.label ||
      run.runnerBuild ||
      'Select Runner build'
    );
  }
  buildOptions(run: SessionConfig['runs'][number]) {
    return this.buildCatalog(run)?.builds.map((build) => build.label) ?? [];
  }
  unavailableBuilds(run: SessionConfig['runs'][number]) {
    return (
      this.buildCatalog(run)
        ?.builds.filter((build) => !build.available)
        .map((build) => build.label) ?? []
    );
  }
  buildDescriptions(run: SessionConfig['runs'][number]) {
    return Object.fromEntries(
      this.buildCatalog(run)?.builds.map((build) => [
        build.label,
        build.reason ||
          [build.version, build.cl ? 'CL ' + build.cl : '']
            .filter(Boolean)
            .join(' · '),
      ]) ?? [],
    );
  }
  selectBuild(run: SessionConfig['runs'][number], label: string) {
    const build = this.buildCatalog(run)?.builds.find(
      (build) => build.label === label && build.available,
    );
    if (build) run.runnerBuild = build.id;
  }
  deviceConfigurationError() {
    if (this.sessions.listing()?.generation_available !== true) return '';
    if (this.request().mode === 'update') return '';
    for (const run of this.config.runs) {
      const device = this.runnerDevice(run.device),
        side = run.id === 'ref' ? 'Reference' : 'Target';
      if (
        !device ||
        !['not_running', 'idle'].includes(device.runnerState ?? 'unknown')
      )
        return (
          side +
          ': ' +
          (device
            ? this.deviceOccupancy(device)
            : 'device availability is unknown')
        );
      if (device.runnerState === 'idle') continue;
      const catalog = this.buildCatalog(run);
      if (!catalog?.canLaunch)
        return (
          side +
          ': ' +
          (catalog?.reason || 'No launchable Model Server builds registered')
        );
      if (
        !catalog.builds.some(
          (build) => build.id === run.runnerBuild && build.available,
        )
      )
        return side + ': select an available Model Server build';
    }
    const [ref, target] = this.config.runs,
      device = this.runnerDevice(ref?.device);
    if (
      device?.id === this.runnerDevice(target?.device)?.id &&
      device?.platform !== 'macOS'
    )
      return 'This device supports one Runner. Select different devices for Reference and Target.';
    return '';
  }
  selectDevice(run: SessionConfig['runs'][number], label: string) {
    const device = this.runtime
      .devices()
      .find((d) => this.deviceLabel(d.id) === label);
    if (!device || this.unavailableDevices(run).includes(label)) return;
    run.device = device.id;
    run.runnerBuild =
      device.runnerState === 'idle'
        ? (device.runner?.build?.id ?? device.runner?.buildId)
        : undefined;
    if (run.runtime === 'LiteRT-LM') {
      const capability = this.deviceCapability(run),
        backends = device.platform?.startsWith('iOS')
          ? ['CPU']
          : (capability?.backends ?? ['CPU']);
      if (!backends.includes(run.backend ?? ''))
        run.backend = backends[0] ?? 'CPU';
      run.cpuThreads = '';
      if (
        !(capability?.contextLengths ?? [1024, 4096]).includes(
          Number(run.contextLength),
        )
      )
        run.contextLength = '1024';
      run.audioBackend = '';
      run.audioCpuThreads = '';
      run.visionBackend = '';
      run.forceF32 = false;
      run.prefillBatchSizes = '';
    }
    void this.loadSelectedBuilds();
    if (!this.customName) this.suggest();
  }
  config: SessionConfig = {name: '', model: '', runs: []};
  ngOnInit() {
    // A saved-capture Server (--data) has no Runner registry; do not probe devices there.
    if (this.sessions.listing()?.generation_available !== false)
      void this.refreshDevices();
    this.customName = this.request().mode !== 'create';
    const source =
      this.request().session ??
      this.sessions.listing()?.configuration_template ??
      this.sessions.items()[0];
    this.config = {
      tap_points:
        this.request().mode === 'create'
          ? {}
          : structuredClone(source?.tap_points ?? {}),
      tap_profile:
        this.request().mode === 'create' ? '' : (source?.tap_profile ?? ''),
      name:
        this.request().mode === 'duplicate'
          ? copiedSessionName(source?.name ?? 'Session')
          : this.request().mode === 'update'
            ? (source?.name ?? '')
            : '',
      model: source?.model ?? '',
      runs: ['ref', 'target'].map((id) => ({
        ...(source?.runs.find((r) => r.id === id) ?? {}),
        id,
        device: source?.runs.find((r) => r.id === id)?.device ?? 'local',
        runtime: source?.runs.find((r) => r.id === id)?.runtime ?? 'LiteRT-LM',
        source: source?.runs.find((r) => r.id === id)?.source ?? 'path',
      })),
    };
    if (this.request().mode === 'create') {
      for (const run of this.config.runs) {
        run.artifact = '';
        run.repository = '';
        run.source = this.isPyTorch(run) ? 'registered' : 'upload';
        run.backend = 'CPU';
        if (run.runtime === 'LiteRT-LM') run.contextLength = '1024';
        run.forceF32 = false;
        delete run.precision;
        delete run.runnerBuild;
        run.device = 'local';
      }
      this.suggest();
    }
  }
  async upload(event: Event, run: SessionConfig['runs'][number]) {
    if (this.isPyTorch(run) || this.uploading()) return;
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    // Retain the File object for upload while allowing the same file to be picked again.
    input.value = '';
    if (!file) return;
    const limit = this.runtime.capabilities?.()?.upload_limit_bytes;
    if (limit && file.size > limit) {
      this.error.set(
        `${file.name} is ${formatGiB(file.size, 2)}; the server accepts files up to ${formatGiB(limit, 2)}.`,
      );
      this.focusError();
      return;
    }
    const request = new AbortController();
    this.uploadRequest = request;
    this.uploading.set(run.id);
    this.uploadProgress.set(null);
    this.error.set('');
    requestAnimationFrame(() =>
      this.element.nativeElement
        .querySelector<HTMLButtonElement>('[data-cancel-upload]')
        ?.focus(),
    );
    try {
      const result = await this.runtime.api.upload(file, {
        signal: request.signal,
        onProgress: (loaded, total) => {
          if (this.uploadRequest === request && total > 0)
            this.uploadProgress.set(loaded / total);
        },
      });
      if (request.signal.aborted || !this.alive) return;
      run.artifact = result.artifact;
      run.source = 'upload';
      this.focusSelectedFile(run);
      void this.runtime.refreshCapabilities();
    } catch (e) {
      if (!request.signal.aborted && this.alive) {
        this.error.set((e as Error).message);
        this.focusError();
      }
    } finally {
      if (this.uploadRequest === request) {
        this.uploadRequest = null;
        this.uploading.set(null);
        this.uploadProgress.set(null);
        if (request.signal.aborted && this.alive)
          requestAnimationFrame(() =>
            this.element.nativeElement
              .querySelector<HTMLButtonElement>(
                '[data-run="' + run.id + '"] [data-source-upload]',
              )
              ?.focus(),
          );
      }
    }
  }
  selectHf(run: SessionConfig['runs'][number], file: HfFile) {
    if (this.isPyTorch(run)) return;
    Object.assign(run, file);
    this.focusSelectedFile(run);
  }
  clearFile(run: SessionConfig['runs'][number]) {
    run.artifact = '';
    run.repository = '';
    run.revision = '';
    run.sourceUrl = '';
    run.source = this.isPyTorch(run) ? 'registered' : 'upload';
    requestAnimationFrame(() =>
      this.element.nativeElement
        .querySelector<HTMLButtonElement>(
          '[data-run="' +
            run.id +
            '"] ' +
            (this.isPyTorch(run)
              ? '[data-source-local]'
              : '[data-source-upload]'),
        )
        ?.focus(),
    );
  }
  suggest() {
    this.customName = false;
    this.config.name =
      (this.config.model || 'New session') +
      ' · ' +
      this.config.runs.map((r) => r.backend || r.runtime).join(' vs ');
  }
  canUseReference() {
    const ref = this.config.runs.find((r) => r.id === 'ref'),
      target = this.config.runs.find((r) => r.id === 'target');
    return !!ref?.artifact && ref.runtime === target?.runtime;
  }
  sameFile() {
    if (!this.canUseReference()) return;
    const ref = this.config.runs.find((r) => r.id === 'ref')!,
      target = this.config.runs.find((r) => r.id === 'target')!;
    target.source = ref.source;
    target.artifact = ref.artifact;
    target.repository = ref.repository;
    target.revision = ref.revision;
    target.sourceUrl = ref.sourceUrl;
  }
  modelName(artifact: string) {
    return (
      this.runtime.capabilities()?.models.find((m) => m.artifact === artifact)
        ?.name ?? artifact.split('/').pop()
    );
  }
  selectLocal(run: SessionConfig['runs'][number], artifact: string) {
    if (!this.localModels(run).some((model) => model.artifact === artifact))
      return;
    run.artifact = artifact;
    run.source = 'registered';
    run.repository = '';
    run.revision = '';
    run.sourceUrl = '';
    this.focusSelectedFile(run);
  }
  artifactsReady() {
    return this.config.runs.every(
      (r) =>
        r.artifact?.trim() &&
        (r.source !== 'huggingface' || r.repository?.trim()) &&
        (!this.isPyTorch(r) ||
          (r.source === 'registered' &&
            this.localModels(r).some(
              (model) => model.artifact === r.artifact,
            ))),
    );
  }
  readonly scanReady = signal<Record<string, boolean>>({});
  captureGroups() {
    const runs = this.config.runs.filter((r) => r.runtime === 'LiteRT-LM');
    return [
      ...new Set(runs.map((r) => r.artifact).filter((a): a is string => !!a)),
    ].map((artifact) => ({
      artifact,
      label:
        runs
          .filter((r) => r.artifact === artifact)
          .map((r) => (r.id === 'ref' ? 'Reference' : 'Target'))
          .join(' + ') +
        ' · ' +
        this.modelName(artifact),
    }));
  }
  selectTaps(artifact: string, points: string[]) {
    this.config.tap_points = {...this.config.tap_points, [artifact]: points};
    this.config.tap_profile = 'custom-outputs-v1';
  }
  captureReady() {
    return (
      this.sessions.listing()?.generation_available !== true ||
      !this.config.tap_profile ||
      this.captureGroups().every(
        (g) =>
          this.scanReady()[g.artifact] &&
          (this.config.tap_points?.[g.artifact]?.length ?? 0) > 0,
      )
    );
  }
  setScanReady(artifact: string, ready: boolean) {
    this.scanReady.update((value) => ({...value, [artifact]: ready}));
  }
  async save() {
    if (
      this.saving() ||
      !this.captureReady() ||
      !this.artifactsReady() ||
      this.deviceConfigurationError()
    )
      return;
    const capability =
      this.request().mode === 'duplicate' ? 'duplicate' : 'create';
    if (!this.sessions.listing()?.capabilities[capability]) {
      this.error.set(
        'Session configuration changes are unavailable. Return to the session list.',
      );
      return;
    }
    const original = this.request().session;
    const current =
      this.sessions
        .items()
        .find((item) => item.id === (this.savedId ?? original?.id)) ?? original;
    if (
      current &&
      (sessionConfigurationLocked(current) ||
        ((this.savedId || this.request().mode === 'update') &&
          current.has_capture))
    ) {
      this.error.set(
        'This session can no longer be edited. Return to the list and check its current state.',
      );
      return;
    }
    this.saving.set(true);
    this.error.set('');
    let preparing = false;
    try {
      const id =
        this.savedId ??
        (this.request().mode === 'update'
          ? this.request().session?.id
          : undefined);
      const session = await this.sessions.manage(id ? 'update' : 'create', {
        ...this.config,
        tap_points: Object.fromEntries(
          this.captureGroups().map((g) => [
            g.artifact,
            this.config.tap_points?.[g.artifact] ?? [],
          ]),
        ),
        id,
      });
      this.savedId = session.id;
      if (
        this.config.tap_profile &&
        this.sessions.listing()?.generation_available === true
      ) {
        preparing = true;
        await this.runtime.lifecycle(session.id, 'prepare');
        await this.sessions.load();
      }
      // A new Session starts its Model Server right away; with selected outputs, as soon as the
      // capture model is prepared. The list shows it starting and opens the way in once it is on.
      const starting =
        this.request().mode !== 'update' &&
        this.sessions.listing()?.generation_available === true;
      if (starting) {
        if (preparing) this.runtime.startWhenPrepared(session.id);
        else {
          await this.sessions.load();
          const summary = this.sessions
            .allItems()
            .find((item) => item.id === session.id);
          if (summary) void this.runtime.toggleModelServer(summary);
        }
      }
      if (this.alive) {
        this.finished.emit(true);
        if (starting) this.created.emit(session.id);
      }
    } catch (e) {
      this.error.set(
        (preparing ? 'Configuration saved. Capture preparation failed: ' : '') +
          (e as Error).message,
      );
      await this.sessions.load();
      if (this.alive) this.focusError();
    } finally {
      this.saving.set(false);
    }
  }
}
