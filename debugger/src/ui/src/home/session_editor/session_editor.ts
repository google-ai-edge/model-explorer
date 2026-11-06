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
import {ArtifactsApi} from '../../server/artifacts_api';
import type {ModelComparison} from '../../server/contracts/artifacts';
import type {RunnerApp, RunnerBuild, RunnerDevice} from '../../server/contracts/devices';
import {isTerminal} from '../../server/contracts/jobs';
import type {RuntimeCapability} from '../../server/contracts/runtime';
import {SessionConfig} from '../../server/contracts/sessions';
import {JobsApi} from '../../server/jobs_api';
import {RuntimeService} from '../../server/runtime_service';
import {SessionListService} from '../../server/session_list_service';
import {ConfigPicker} from '../../widgets/config_picker/config_picker';
import {formatGiB} from '../../widgets/format/format';
import {AnchorSummary} from '../anchor_summary/anchor_summary';
import {CounterpartSummary} from '../counterpart_summary/counterpart_summary';
import {HfFilePicker, type HfModel} from '../hf_file_picker/hf_file_picker';
import {LayerOpsPicker} from '../layer_ops_picker/layer_ops_picker';
import {runtimeOptionValue, type RuntimeOptionKey, sessionOptions} from '../runtime_options';
import {TapPicker} from '../tap_picker/tap_picker';

type Run = SessionConfig['runs'][number];
/** The profile a side's outputs chosen are observed by (the server's tap_profiles). */
const CHOSEN_OUTPUTS = 'custom-outputs-v1';

function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/**
 * Orchestrates chunked model file uploads and polled Hugging Face download jobs so network
 * transfer and job cancellation lifecycles stay isolated from the SessionEditor form state.
 */
class ArtifactTransferController {
  readonly uploading = signal<string | null>(null);
  readonly uploadProgress = signal<number | null>(null);
  /**
   * Whether the side `uploading()` names fetches its model from Hugging Face, not an upload;
   * the server's progress line of it.
   */
  readonly downloading = signal(false);
  readonly downloadText = signal('');
  readonly uploadLabel = computed(() => {
    if (this.downloading()) {
      return this.downloadText() || 'Resolving the Hugging Face model…';
    }
    const progress = this.uploadProgress();
    return progress === null
      ? 'Uploading model file…'
      : `Uploading model file · ${Math.round(progress * 100)}%`;
  });
  private uploadRequest: AbortController | null = null;
  /** The download job of the side `uploading()` names, once the server started it. */
  private downloadJob: string | null = null;
  private alive = true;

  constructor(
    private readonly artifactsApi: ArtifactsApi,
    private readonly jobsApi: JobsApi,
  ) {}

  markDestroyed(): void {
    this.alive = false;
  }

  cancelUpload(): void {
    this.uploadRequest?.abort();
  }

  async cancelDownload(onError?: (message: string) => void): Promise<void> {
    const id = this.downloadJob;
    if (!id) {
      return;
    }
    try {
      await this.jobsApi.cancel(id);
    } catch (e) {
      if (this.alive && onError) {
        onError(errorMessage(e));
      }
    }
  }

  async uploadFile(
    runId: string,
    file: File,
    callbacks: {
      onStarted: () => void;
      onCompleted: (artifact: string) => void;
      onAborted: () => void;
      onError: (message: string) => void;
    },
  ): Promise<void> {
    const request = new AbortController();
    this.uploadRequest = request;
    this.uploading.set(runId);
    this.uploadProgress.set(null);
    callbacks.onStarted();
    try {
      const result = await this.artifactsApi.upload(file, {
        signal: request.signal,
        onProgress: (loaded, total) => {
          if (this.uploadRequest === request && total > 0) {
            this.uploadProgress.set(loaded / total);
          }
        },
      });
      if (request.signal.aborted || !this.alive) {
        return;
      }
      callbacks.onCompleted(result.artifact);
    } catch (e) {
      if (!request.signal.aborted && this.alive) {
        callbacks.onError(errorMessage(e));
      }
    } finally {
      if (this.uploadRequest === request) {
        this.uploadRequest = null;
        this.uploading.set(null);
        this.uploadProgress.set(null);
        if (request.signal.aborted && this.alive) {
          callbacks.onAborted();
        }
      }
    }
  }

  async fetchHuggingFace(
    runId: string,
    runtime: string,
    model: HfModel,
    callbacks: {
      onCompleted: (artifact: string) => Promise<void>;
      onError: (message: string) => void;
    },
  ): Promise<void> {
    this.uploading.set(runId);
    this.downloading.set(true);
    this.downloadText.set('');
    try {
      let job = await this.artifactsApi.fetchHuggingFace({runtime, ...model});
      this.downloadJob = job.id;
      while (!isTerminal(job.status)) {
        this.downloadText.set(job.progress ?? '');
        await new Promise((resolve) => setTimeout(resolve, 500));
        if (!this.alive) {
          return;
        }
        job = await this.jobsApi.job(job.id);
      }
      if (!this.alive || job.status === 'cancelled') {
        return;
      }
      if (job.status !== 'completed' || !job.model) {
        throw new Error(job.error || 'Hugging Face model download failed');
      }
      await callbacks.onCompleted(job.model.artifact);
    } catch (e) {
      if (this.alive) {
        callbacks.onError(errorMessage(e));
      }
    } finally {
      this.downloadJob = null;
      this.uploading.set(null);
      this.downloading.set(false);
      this.downloadText.set('');
    }
  }
}

@Component({
  selector: 'session-editor',
  imports: [
    AnchorSummary,
    CounterpartSummary,
    LayerOpsPicker,
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
  readonly sessions = inject(SessionListService);
  private readonly artifactsApi = inject(ArtifactsApi);
  private readonly jobsApi = inject(JobsApi);
  private readonly element: ElementRef<HTMLElement> = inject(ElementRef);
  private readonly dialogs = inject(MatDialog);
  private readonly changeDetector = inject(ChangeDetectorRef);
  private readonly transfers = new ArtifactTransferController(this.artifactsApi, this.jobsApi);
  readonly finished = output<void>();
  readonly created = output<string>();
  readonly saving = signal(false);
  readonly error = signal('');
  readonly uploading = this.transfers.uploading;
  readonly uploadProgress = this.transfers.uploadProgress;
  readonly downloading = this.transfers.downloading;
  readonly downloadText = this.transfers.downloadText;
  readonly uploadLabel = this.transfers.uploadLabel;
  /** What of the two models chosen compares, as the server says; null until both are chosen. */
  readonly comparison = signal<ModelComparison | null>(null);
  /** Whether the devices are being listed anew: Refresh spins meanwhile. */
  readonly refreshing = signal(false);
  readonly scanReady = signal<Record<string, boolean>>({});
  /** Whether the server refuses none of the operators ticked inside the layers. */
  readonly opsReady = signal(true);
  readonly runtimes = computed(() => [
    ...new Set([
      ...(this.runtime.capabilities()?.runtimes?.map((r) => r.id) ?? ['LiteRT-LM']),
      ...this.sessions
        .items()
        .flatMap((s) => s.runs.map((r) => r.runtime ?? ''))
        .filter(Boolean),
    ]),
  ]);
  /**
   * The runner version picker is hidden: each side runs its runtime's default build, as the
   * server marks it (`default`), and the Session records it as before. Its code stays for when
   * versions are chosen again.
   */
  readonly showRunnerVersion: boolean = false;
  /**
   * The per-tensor picker of each side ("Advanced · more tensor outputs") is hidden: the
   * operators inside the layers are chosen once for both sides instead (layer-ops-picker). A
   * Session saved with outputs chosen keeps them. Its code stays.
   */
  readonly showTapPicker: boolean = false;
  /**
   * Per side, whether its advanced capture (outputs chosen) is on: the editor's own state, sent
   * as the Session's points.
   */
  advanced: Record<string, boolean> = {};
  customName = false;
  config: SessionConfig = {name: '', runs: []};
  private comparisonRequest = 0;
  private alive = true;

  ngOnInit(): void {
    // The devices, and with them the capabilities, as the server has them now: one may have
    // been connected since the page loaded.
    void this.refreshDevices();
    const source = this.sessions.listing()?.configuration_template ?? this.sessions.items()[0];
    this.config = {
      tap_points: {},
      tap_profile: '',
      tap_ops: [],
      name: '',
      runs: ['ref', 'target'].map((id) => ({
        ...(source?.runs.find((r) => r.id === id) ?? {}),
        id,
        artifact: '',
        repository: '',
        backend: 'CPU',
        device: 'local',
        runtime: source?.runs.find((r) => r.id === id)?.runtime ?? 'LiteRT-LM',
      })),
    };
    for (const run of this.config.runs) {
      run.source = this.isPyTorch(run) ? 'huggingface' : 'upload';
      run.contextLength = this.defaultContextLength(run);
      delete run.runnerBuild;
    }
    this.suggest();
    // Every side captures its anchors; a PyTorch side says so by its run's own option.
    for (const run of this.config.runs) {
      if (this.isPyTorch(run)) {
        run.observe = 'anchors';
      }
    }
    void this.loadComparison();
  }

  ngAfterViewInit(): void {
    this.element.nativeElement.querySelector<HTMLElement>('#editor-title')?.focus();
  }

  ngOnDestroy(): void {
    this.alive = false;
    this.transfers.markDestroyed();
    this.cancelUpload();
    void this.cancelDownload();
  }

  cancelUpload(): void {
    this.transfers.cancelUpload();
  }

  /** Cancels the Hugging Face download: the server stops it at once and deletes what it wrote. */
  async cancelDownload(): Promise<void> {
    await this.transfers.cancelDownload((message) => {
      if (this.alive) {
        this.error.set(message);
      }
    });
  }

  private focusError(): void {
    requestAnimationFrame(() =>
      this.element.nativeElement.querySelector<HTMLElement>('[role="alert"]')?.focus(),
    );
  }

  focusSelectedFile(run: SessionConfig['runs'][number]): void {
    requestAnimationFrame(() =>
      this.element.nativeElement
        .querySelector<HTMLButtonElement>(
          '[aria-label="Clear ' + (run.id === 'ref' ? 'Reference' : 'Target') + ' file selection"]',
        )
        ?.focus(),
    );
  }

  async configure(run: SessionConfig['runs'][number], trigger: HTMLElement): Promise<void> {
    try {
      const {RuntimeOptions} = await import('../runtime_options/runtime_options');
      if (!this.alive) {
        return;
      }
      const other = this.other(run);
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
          if (!this.alive) {
            return;
          }
          if (options) {
            Object.assign(run, options);
            if (!this.customName) {
              this.suggest();
            }
            this.changeDetector.markForCheck();
          }
          trigger.focus();
        });
    } catch {
      this.error.set('Runtime settings could not be loaded. Reload the page and try again.');
      trigger.focus();
    }
  }

  differences(run: SessionConfig['runs'][number]): Array<{label: string; value: string}> {
    const other = this.other(run);
    return sessionOptions(this.deviceCapability(run))
      .filter((option) => {
        const key = option.key as RuntimeOptionKey;
        return runtimeOptionValue(run, key, option) !== runtimeOptionValue(other, key, option);
      })
      .map((option) => ({
        label: option.label,
        value: runtimeOptionValue(run, option.key as RuntimeOptionKey, option),
      }));
  }

  /** Every Session option a run may hold: each runtime's, as the server describes them. */
  private optionKeys(): RuntimeOptionKey[] {
    return [
      ...new Set(
        (this.runtime.capabilities()?.runtimes ?? []).flatMap((capability) =>
          sessionOptions(capability).map((o) => o.key as RuntimeOptionKey),
        ),
      ),
    ];
  }

  /** The context length a new run starts with: the server's default. */
  private defaultContextLength(run: SessionConfig['runs'][number]): string {
    const option = sessionOptions(this.deviceCapability(run)).find(
      (o) => o.key === 'contextLength',
    );
    return String(option?.default ?? '');
  }

  isPyTorch(run: SessionConfig['runs'][number]): boolean {
    return run.runtime === 'PyTorch';
  }

  /**
   * Whether a run may capture intermediate tensors it chooses, as the server says of its
   * runtime; a PyTorch run captures its anchors by its own options instead (pytorchCapture()).
   */
  capturable(run: Run): boolean {
    const tap = this.deviceCapability(run)?.tap;
    return tap?.available !== false && tap?.kind !== 'anchors';
  }

  /** Why a side cannot capture tensors, as the server says; '' if it can. */
  tapReason(run: Run): string {
    const tap = this.deviceCapability(run)?.tap;
    return tap && !tap.available ? tap.reason : '';
  }

  /**
   * How a PyTorch side captures its anchors, by its own options, as the server says; null for a
   * side that does not.
   */
  pytorchCapture(run: Run) {
    const tap = this.deviceCapability(run)?.tap;
    if (tap?.kind !== 'anchors' || !tap.available) {
      return null;
    }
    const option = (key: string) => tap.options?.find((o) => o.key === key);
    return {text: tap.text ?? '', details: tap.details ?? '', budget: option('observeMemoryGiB')};
  }

  /** The other side of a run. */
  other(run: Run): Run {
    const found = this.config.runs.find((r) => r.id !== run.id);
    if (!found) {
      throw new Error(`Missing counterpart run for ${run.id}`);
    }
    return found;
  }

  /**
   * The LiteRT-LM side beside a PyTorch run that observes, whose outputs chosen the PyTorch side
   * captures the modules of; null if there is none, or it chose none.
   */
  pairedLitert(run: Run): {artifact: string; points: string[]} | null {
    const other = this.other(run);
    if (run.observe !== 'anchors' || !run.artifact || !other.artifact) {
      return null;
    }
    if (!this.capturable(other) || !this.sideAdvanced(other)) {
      return null;
    }
    const points = this.config.tap_points?.[other.artifact] ?? [];
    return points.length ? {artifact: other.artifact, points} : null;
  }

  /**
   * The PyTorch model of a run that observes, if any: the outputs chosen of the LiteRT-LM model
   * pair with its modules, as the server says (tap_picker).
   */
  pairedPyTorch(): string {
    return (
      this.config.runs.find((run) => this.pytorchCapture(run) && run.observe === 'anchors')
        ?.artifact ?? ''
    );
  }

  setBudget(run: Run, value: number | string | null): void {
    run.observeMemoryGiB = value == null ? '' : String(value);
  }

  /**
   * The LiteRT-LM sides that load one file with `run`, itself among them: they share one copy
   * of it, so one capture (the server's capture_shared).
   */
  private sharing(run: Run): Run[] {
    return this.config.runs.filter(
      (r) =>
        r === run ||
        (!!run.artifact &&
          r.artifact === run.artifact &&
          !this.isPyTorch(r) &&
          !this.isPyTorch(run)),
    );
  }

  /**
   * Whether a side captures its anchors: every LiteRT-LM side does, always (the server sets it);
   * a PyTorch side does by its run's `observe`, which is always 'anchors'.
   */
  sideAnchors(run: Run): boolean {
    return !this.isPyTorch(run);
  }

  /** Whether a LiteRT-LM side captures outputs chosen beside or instead of its anchors. */
  sideAdvanced(run: Run): boolean {
    return !!this.advanced[run.id];
  }

  setAdvanced(run: Run, on: boolean): void {
    for (const r of this.sharing(run)) {
      this.advanced[r.id] = on;
    }
  }

  /**
   * A run that now loads the file the other LiteRT-LM side loads takes that side's outputs
   * chosen: the two share one copy of it.
   */
  private shareCapture(run: Run): void {
    const other = this.other(run);
    if (!this.sharing(run).includes(other)) {
      return;
    }
    this.advanced[run.id] = this.sideAdvanced(other);
  }

  /**
   * What a run may set: the offer of the runner version it chose, on a device that runs the one
   * chosen; else its device's offer as the server merged it; or else the runtime's.
   */
  deviceCapability(run: SessionConfig['runs'][number]): RuntimeCapability | undefined {
    const device = this.runnerDevice(run.device);
    const offered =
      (device?.runnerState === 'not_running' &&
        this.builds(run).find((build) => build.id === run.runnerBuild)?.capability) ||
      device?.capability;
    return offered && offered.id === (run.runtime ?? 'LiteRT-LM')
      ? offered
      : this.runtime.capability(run.runtime);
  }

  /**
   * A side's runtime and where it runs, as the device notice says it: the side's runtime's
   * backends, the GPU of a PyTorch side being this computer's MPS.
   */
  backendsText(capability: RuntimeCapability): string {
    const backends =
      capability.id === 'PyTorch'
        ? capability.backends.map((backend) => (backend === 'GPU' ? 'MPS' : backend))
        : capability.backends;
    return capability.id + ' · ' + backends.join('/');
  }

  runtimeNotice(run: SessionConfig['runs'][number]): string {
    if (this.runtime.capabilities()?.available === false) {
      return '';
    }
    const capability = this.runtime.capability(run.runtime);
    return capability && !capability.available
      ? (run.runtime ?? 'LiteRT-LM') + ' is unavailable on this server.'
      : '';
  }

  /**
   * Whether the Session observes tensors of its LiteRT-LM models, which a copy of each prepared
   * first observes: their anchors, outputs chosen, or both.
   */
  capturesTensors(): boolean {
    return Object.keys(this.chosenPoints()).length > 0 || this.capturesAnchors();
  }

  capturesAnchors(): boolean {
    return this.config.runs.some((run) => this.capturable(run) && this.sideAnchors(run));
  }

  /** The outputs chosen, of each model of a side whose advanced capture is on. */
  chosenPoints(): Record<string, string[]> {
    const entries: Array<[string, string[]]> = [];
    for (const run of this.config.runs) {
      const artifact = run.artifact;
      if (artifact && this.capturable(run) && this.sideAdvanced(run)) {
        entries.push([artifact, this.config.tap_points?.[artifact] ?? []]);
      }
    }
    return Object.fromEntries(entries);
  }

  saveNotice(): string {
    if (this.runtime.capabilities()?.available !== true) {
      return '';
    }
    return this.capturesTensors()
      ? 'Creates the Session, prepares the tensor capture and starts its Model Server. Enter the Session once it is on, and start a chat there.'
      : 'Creates the Session and starts its Model Server. Enter the Session once it is on, and start a chat there.';
  }

  /**
   * The models of a side's runtime already on the server, which a Session may load: uploaded
   * or downloaded before, one copy shared by every Session.
   */
  serverModels(run: SessionConfig['runs'][number]) {
    return (
      this.runtime
        .capabilities()
        ?.models.filter(
          (model) =>
            model.on_server !== false &&
            (model.runtime ?? 'LiteRT-LM') === (run.runtime ?? 'LiteRT-LM'),
        ) ?? []
    );
  }

  /**
   * Whether a model's file is on the server, as the catalog says: not for one an imported
   * Session came with, or an older server kept outside its workspace.
   */
  onServer(artifact: string | undefined): boolean {
    return (
      this.runtime.capabilities()?.models.find((m) => m.artifact === artifact)?.on_server !== false
    );
  }

  /** A size as the server's lines say it, in GB or MB. */
  sizeText(size: number | undefined): string {
    if (size == null) {
      return '';
    }
    return size >= 1e9 ? `${(size / 1e9).toFixed(1)} GB` : `${(size / 1e6).toFixed(1)} MB`;
  }

  selectRuntime(run: SessionConfig['runs'][number], runtime: string): void {
    if (run.runtime === runtime) {
      return;
    }
    run.runtime = runtime;
    run.artifact = '';
    run.repository = '';
    run.revision = '';
    run.sourceUrl = '';
    run.source = runtime === 'PyTorch' ? 'huggingface' : 'upload';
    delete run.huggingface;
    run.device = 'local';
    delete run.runnerBuild;
    for (const key of this.optionKeys()) {
      run[key] = '';
    }
    const backends = this.runtime.capability(runtime)?.backends ?? ['CPU'];
    run.backend = backends[0] ?? 'CPU';
    run.contextLength = this.defaultContextLength(run);
    // Every side captures its anchors: a PyTorch side by its run's `observe`, a LiteRT-LM side
    // always (the server sets it); it has no outputs chosen.
    run.observe = runtime === 'PyTorch' ? 'anchors' : '';
    run.observeMemoryGiB = '';
    this.advanced[run.id] = false;
    this.comparison.set(null);
    void this.loadSelectedBuilds();
    if (!this.customName) {
      this.suggest();
    }
  }

  deviceLabel(id: string | undefined): string {
    const resolvedId = !id || id === 'Server host' ? 'local' : id;
    const device = this.runnerDevice(resolvedId);
    if (resolvedId === 'local' || device?.id === 'local') {
      return 'This computer · ' + (device ? this.runtime.deviceStatusText(device) : 'Unchecked');
    }
    if (!device) {
      return 'Not discovered · ' + resolvedId.slice(-4);
    }
    const platform = device.platform ?? 'iOS';
    const transport = device.transport ?? 'USB';
    // A transport the platform names already ('iOS Simulator', 'simulator') is said once.
    const kind = platform.toLowerCase().includes(transport.toLowerCase())
      ? platform
      : platform + ' ' + transport;
    return [device.name, kind, this.runtime.deviceStatusText(device), resolvedId.slice(-4)].join(
      ' · ',
    );
  }

  runnerDevice(id: string | undefined): RunnerDevice | undefined {
    return this.runtime
      .devices()
      .find((d) => d.id === (!id || id === 'Server host' ? 'local' : id) || d.runnerId === id);
  }

  deviceCheckedAt(value: string): string {
    return new Date(value).toLocaleTimeString();
  }

  deviceOptions(run: SessionConfig['runs'][number]): string[] {
    return this.runtime
      .devices()
      .filter((d) => !this.isPyTorch(run) || d.id === 'local' || d.capability?.id === 'PyTorch')
      .map((d) => this.deviceLabel(d.id));
  }

  /**
   * Who holds each device, or why it cannot be chosen, as the server says; a phone's App. Its
   * status is in its label already.
   */
  deviceDescriptions(): Record<string, string> {
    return Object.fromEntries(
      this.runtime
        .devices()
        .map((d) => [
          this.deviceLabel(d.id),
          [d.holder || d.choosable.reason, d.app?.installed ? this.appLabel(d.app) : '', d.reason]
            .filter(Boolean)
            .join(' · '),
        ]),
    );
  }

  /** A phone's Runner App in words: the version it runs, which the phone has, one per phone. */
  appLabel(app: RunnerApp): string {
    return app.installed ? 'Runner App: ' + app.label : app.label;
  }

  /** The devices the server refuses for a Session now, and those being checked. */
  unavailableDevices(): string[] {
    return this.runtime
      .devices()
      .filter((d) => !d.choosable.allowed || this.runtime.checking()[d.id])
      .map((d) => this.deviceLabel(d.id));
  }

  async checkDevice(id: string): Promise<void> {
    await this.runtime.inspectDevice(id);
    await this.loadSelectedBuilds();
  }

  async refreshDevices(): Promise<void> {
    this.refreshing.set(true);
    try {
      await this.runtime.refreshDevices();
      await this.loadSelectedBuilds();
    } finally {
      this.refreshing.set(false);
    }
  }

  // The runner version picker: a device whose runnerState is 'not_running' (this computer) offers
  // the builds the server lists (GET /api/devices/builds), and each start of the side runs the
  // one chosen. A phone runs the App installed on it and sends no runnerState. While the picker
  // is hidden (showRunnerVersion), a side that chose none, or one no longer available, runs its
  // runtime's default, as the server marks it.
  async loadSelectedBuilds(): Promise<void> {
    const ids = [
      ...new Set(
        this.config.runs
          .map((run) => this.runnerDevice(run.device))
          .filter(
            (device): device is RunnerDevice => !!device && device.runnerState === 'not_running',
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
      const available = this.builds(run).filter((build) => build.available);
      const chosen = available.some((build) => build.id === run.runnerBuild);
      const fallback =
        available.find((build) => build.default) ??
        (available.length === 1 ? available[0] : undefined);
      if (fallback && (!run.runnerBuild || (!this.showRunnerVersion && !chosen))) {
        run.runnerBuild = fallback.id;
        this.fitOptions(run);
      }
    }
    this.changeDetector.markForCheck();
  }

  buildCatalog(run: SessionConfig['runs'][number]) {
    const device = this.runnerDevice(run.device);
    return device ? this.runtime.buildCatalogs()[device.id] : undefined;
  }

  /** The builds of the run's device that run the run's runtime. */
  builds(run: SessionConfig['runs'][number]): RunnerBuild[] {
    return (
      this.buildCatalog(run)?.builds.filter(
        (build) => (build.runtime ?? 'LiteRT-LM') === (run.runtime ?? 'LiteRT-LM'),
      ) ?? []
    );
  }

  buildLabel(run: SessionConfig['runs'][number]): string {
    return (
      this.builds(run).find((build) => build.id === run.runnerBuild)?.label ||
      run.runnerBuild ||
      'Select runner version'
    );
  }

  buildOptions(run: SessionConfig['runs'][number]): string[] {
    return this.builds(run).map((build) => build.label);
  }

  unavailableBuilds(run: SessionConfig['runs'][number]): string[] {
    return this.builds(run)
      .filter((build) => !build.available)
      .map((build) => build.label);
  }

  buildDescriptions(run: SessionConfig['runs'][number]): Record<string, string> {
    return Object.fromEntries(
      this.builds(run).map((build) => [
        build.label,
        build.reason ||
          [build.version, build.cl ? 'CL ' + build.cl : ''].filter(Boolean).join(' · '),
      ]),
    );
  }

  selectBuild(run: SessionConfig['runs'][number], label: string): void {
    const build = this.builds(run).find((b) => b.label === label && b.available);
    if (!build) {
      return;
    }
    run.runnerBuild = build.id;
    this.fitOptions(run);
  }

  /**
   * Why a new Session cannot be created yet: a run on a device that runs the version chosen has
   * no available one chosen. Whether its devices can run it now is the start's to say, as
   * Create's refusal shows.
   */
  deviceConfigurationError(): string {
    for (const run of this.config.runs) {
      if (this.runnerDevice(run.device)?.runnerState !== 'not_running') {
        continue;
      }
      const side = run.id === 'ref' ? 'Reference' : 'Target';
      const catalog = this.buildCatalog(run);
      if (!catalog?.canLaunch) {
        return side + ': ' + (catalog?.reason || 'No runner version can be launched');
      }
      if (!this.builds(run).some((build) => build.id === run.runnerBuild && build.available)) {
        return side + ': select an available runner version';
      }
    }
    return '';
  }

  selectDevice(run: SessionConfig['runs'][number], label: string): void {
    const device = this.runtime.devices().find((d) => this.deviceLabel(d.id) === label);
    if (!device || this.unavailableDevices().includes(label)) {
      return;
    }
    run.device = device.id;
    run.runnerBuild =
      device.runnerState === 'idle'
        ? (device.runner?.build?.id ?? device.runner?.buildId)
        : undefined;
    this.fitOptions(run);
    void this.loadSelectedBuilds();
    if (!this.customName) {
      this.suggest();
    }
  }

  /** A device or runner version change keeps only the Session options its runner applies. */
  private fitOptions(run: SessionConfig['runs'][number]): void {
    const capability = this.deviceCapability(run);
    const backends = capability?.backends ?? ['CPU'];
    if (!backends.includes(run.backend ?? '')) {
      run.backend = backends[0] ?? 'CPU';
    }
    const described = sessionOptions(capability);
    for (const option of described) {
      const key = option.key as RuntimeOptionKey;
      if (!option.available && key !== 'backend' && key !== 'contextLength') {
        run[key] = '';
      }
    }
    const context = described.find((o) => o.key === 'contextLength');
    if (context?.minimum != null && Number(run.contextLength) < context.minimum) {
      run.contextLength = this.defaultContextLength(run);
    }
  }

  /** Asks the server what of the two models chosen compares; the name suggested follows it. */
  async loadComparison(): Promise<void> {
    const request = ++this.comparisonRequest;
    const [ref, target] = ['ref', 'target'].map(
      (id) => this.config.runs.find((run) => run.id === id)?.artifact ?? '',
    );
    if (!ref || !target) {
      this.comparison.set(null);
      if (!this.customName) {
        this.suggest();
      }
      return;
    }
    try {
      const found = await this.artifactsApi.compareModels(ref, target);
      if (request !== this.comparisonRequest || !this.alive) {
        return;
      }
      this.comparison.set(found);
    } catch {
      if (request === this.comparisonRequest && this.alive) {
        this.comparison.set(null);
      }
    }
    if (!this.customName) {
      this.suggest();
    }
    this.changeDetector.markForCheck();
  }

  async upload(event: Event, run: SessionConfig['runs'][number]): Promise<void> {
    if (this.isPyTorch(run) || this.uploading()) {
      return;
    }
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    // Retain the File object for upload while allowing the same file to be picked again.
    input.value = '';
    if (!file) {
      return;
    }
    const limit = this.runtime.capabilities?.()?.upload_limit_bytes;
    if (limit && file.size > limit) {
      this.error.set(
        `${file.name} is ${formatGiB(file.size, 2)}; the server accepts files up to ${formatGiB(limit, 2)}.`,
      );
      this.focusError();
      return;
    }
    this.error.set('');
    await this.transfers.uploadFile(run.id, file, {
      onStarted: () => {
        requestAnimationFrame(() =>
          this.element.nativeElement
            .querySelector<HTMLButtonElement>('[data-cancel-upload]')
            ?.focus(),
        );
      },
      onCompleted: (artifact) => {
        run.artifact = artifact;
        run.source = 'upload';
        delete run.huggingface;
        this.chosen(run);
        this.focusSelectedFile(run);
        void this.runtime.refreshCapabilities();
      },
      onAborted: () => {
        requestAnimationFrame(() =>
          this.element.nativeElement
            .querySelector<HTMLButtonElement>('[data-run="' + run.id + '"] [data-source-upload]')
            ?.focus(),
        );
      },
      onError: (message) => {
        this.error.set(message);
        this.focusError();
      },
    });
  }

  /**
   * Where a run's model came from on Hugging Face: as the run holds it, else as the server lists
   * the model; null for a model from elsewhere.
   */
  hfSource(run: Run) {
    return (
      run.huggingface ??
      this.runtime.capabilities()?.models.find((m) => m.artifact === run.artifact)?.source ??
      null
    );
  }

  /**
   * Names a side's model on Hugging Face: the server resolves and fetches it in a job, followed
   * here until it ends, and the side then loads the model it downloaded to the server.
   */
  async selectHf(run: Run, model: HfModel): Promise<void> {
    if (this.uploading()) {
      return;
    }
    this.error.set('');
    await this.transfers.fetchHuggingFace(run.id, run.runtime ?? 'LiteRT-LM', model, {
      onCompleted: async (artifact) => {
        await this.runtime.refreshCapabilities();
        if (!this.alive) {
          return;
        }
        run.artifact = artifact;
        run.source = 'huggingface';
        run.repository = '';
        run.revision = '';
        run.sourceUrl = '';
        delete run.huggingface;
        this.chosen(run);
        this.focusSelectedFile(run);
      },
      onError: (message) => {
        this.error.set(message);
        this.focusError();
      },
    });
  }

  /**
   * A side chose its model: it shares the capture of the other side of that file, and the
   * server says anew what of the two compares.
   */
  private chosen(run: Run): void {
    this.shareCapture(run);
    void this.loadComparison();
  }

  clearFile(run: SessionConfig['runs'][number]): void {
    run.artifact = '';
    run.repository = '';
    run.revision = '';
    run.sourceUrl = '';
    delete run.huggingface;
    run.source = this.isPyTorch(run) ? 'huggingface' : 'upload';
    void this.loadComparison();
    requestAnimationFrame(() =>
      this.element.nativeElement
        .querySelector<HTMLButtonElement>(
          '[data-run="' +
            run.id +
            '"] ' +
            (this.isPyTorch(run) ? '[data-source-hf]' : '[data-source-upload]'),
        )
        ?.focus(),
    );
  }

  /**
   * A name of the two models, as the server calls them, and each side's backend; each side's
   * device too, as the device picker names it, unless both run on this computer.
   */
  suggest(): void {
    this.customName = false;
    const onThisComputer = this.config.runs.every(
      (r) => !r.device || r.device === 'Server host' || this.runnerDevice(r.device)?.id === 'local',
    );
    const side = (r: SessionConfig['runs'][number]) => {
      const backend = r.backend || r.runtime;
      if (onThisComputer) {
        return backend;
      }
      return (
        (this.runnerDevice(r.device)?.name || 'Device ' + (r.device ?? '').slice(-4)) +
        ' ' +
        backend
      );
    };
    this.config.name =
      (this.comparison()?.label || this.modelsLabel() || 'New session') +
      ' · ' +
      this.config.runs.map(side).join(' vs ');
  }

  /**
   * The two models chosen by their names, as the server labels them (one name when both sides
   * load one file); '' until both are chosen. The server's comparison may take a while: reading a
   * large model's facts, say.
   */
  private modelsLabel(): string {
    const [ref, target] = ['ref', 'target'].map(
      (id) => this.config.runs.find((run) => run.id === id)?.artifact ?? '',
    );
    if (!ref || !target) {
      return '';
    }
    return ref === target
      ? (this.modelName(ref) ?? '')
      : `${this.modelName(ref)} vs ${this.modelName(target)}`;
  }

  canUseReference(): boolean {
    const ref = this.config.runs.find((r) => r.id === 'ref');
    const target = this.config.runs.find((r) => r.id === 'target');
    return !!ref?.artifact && ref.runtime === target?.runtime;
  }

  sameFile(): void {
    if (!this.canUseReference()) {
      return;
    }
    const ref = this.config.runs.find((r) => r.id === 'ref');
    const target = this.config.runs.find((r) => r.id === 'target');
    if (!ref || !target) {
      return;
    }
    target.source = ref.source;
    target.artifact = ref.artifact;
    target.repository = ref.repository;
    target.revision = ref.revision;
    target.sourceUrl = ref.sourceUrl;
    target.huggingface = ref.huggingface;
    this.chosen(target);
  }

  modelName(artifact: string): string | undefined {
    return (
      this.runtime.capabilities()?.models.find((m) => m.artifact === artifact)?.name ??
      artifact.split('/').pop()
    );
  }

  /** Chooses a model already on the server for a side: it came uploaded or from Hugging Face. */
  selectServerModel(run: SessionConfig['runs'][number], artifact: string): void {
    const model = this.serverModels(run).find((m) => m.artifact === artifact);
    if (!model) {
      return;
    }
    run.artifact = artifact;
    // A PyTorch model only comes from Hugging Face.
    const fromHf = !!model.source || this.isPyTorch(run);
    run.source = fromHf ? 'huggingface' : 'upload';
    delete run.huggingface;
    run.repository = '';
    run.revision = '';
    run.sourceUrl = '';
    this.chosen(run);
    this.focusSelectedFile(run);
  }

  artifactsReady(): boolean {
    return this.config.runs.every(
      (r) =>
        !!r.artifact?.trim() &&
        (!this.isPyTorch(r) ||
          (r.source === 'huggingface' &&
            this.serverModels(r).some((model) => model.artifact === r.artifact))),
    );
  }

  /**
   * The two sides' models, once both are chosen: the operators inside the layers are listed for
   * them.
   */
  bothModels(): {ref: string; target: string} | null {
    const [ref, target] = ['ref', 'target'].map(
      (id) => this.config.runs.find((run) => run.id === id)?.artifact ?? '',
    );
    return ref && target ? {ref, target} : null;
  }

  /** A side's outputs chosen, which both sides of its file observe. */
  selectTaps(run: Run, points: string[]): void {
    if (!run.artifact) {
      return;
    }
    this.config.tap_points = {...this.config.tap_points, [run.artifact]: points};
    this.setAdvanced(run, true);
  }

  /**
   * Whether the tensor capture can be saved: the server refuses none of the operators ticked
   * inside the layers, and, while the per-tensor picker is shown, of each side whose advanced
   * capture is on, its model's scan is read and one is chosen, unless the side captures its
   * anchors beside them.
   */
  captureReady(): boolean {
    return (
      this.opsReady() &&
      (this.runtime.capabilities()?.available !== true ||
        this.config.runs.every(
          (run) =>
            !run.artifact ||
            !this.capturable(run) ||
            !this.sideAdvanced(run) ||
            !this.showTapPicker ||
            (this.scanReady()[run.artifact] &&
              (this.sideAnchors(run) || (this.config.tap_points?.[run.artifact]?.length ?? 0) > 0)),
        ))
    );
  }

  setScanReady(artifact: string, ready: boolean): void {
    this.scanReady.update((value) => ({...value, [artifact]: ready}));
  }

  async save(): Promise<void> {
    if (
      this.saving() ||
      !this.captureReady() ||
      !this.artifactsReady() ||
      this.deviceConfigurationError()
    ) {
      return;
    }
    this.saving.set(true);
    this.error.set('');
    try {
      // The outputs chosen of the sides that chose some. Every side captures its anchors, which
      // the server sets: the editor sends no switch for them.
      const points = this.chosenPoints();
      // Create starts the Session's Model Server too, which prepares its tensor capture first, in
      // the same job; a Session that cannot start now is refused, and nothing is saved: the editor
      // stays open, with what was chosen, and says why. The list shows it starting, and opens the
      // way in once it is on.
      const session = await this.sessions.manage('create', {
        ...this.config,
        tap_profile: Object.keys(points).length ? CHOSEN_OUTPUTS : '',
        tap_points: points,
        tap_ops: this.config.tap_ops ?? [],
      });
      if (this.alive) {
        this.finished.emit();
        this.created.emit(session.id);
      }
    } catch (e) {
      this.error.set(errorMessage(e));
      await this.sessions.load();
      if (this.alive) {
        this.focusError();
      }
    } finally {
      this.saving.set(false);
    }
  }
}
