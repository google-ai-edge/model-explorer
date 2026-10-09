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
  computed,
  effect,
  inject,
  Injectable,
  signal,
  untracked,
} from '@angular/core';
import type {
  RunnerBuildCatalog,
  RunnerDevice,
  RuntimeCapabilities,
  RuntimeCapability,
  RuntimeJob,
  RuntimeJobEvent,
} from './contracts/runtime';
import {isTerminal} from './contracts/runtime';
import type {SessionSummary} from './contracts/session_summary';
import {ReportApiService} from './report_api_service';
import {ReportStateService} from './report_state_service';
import {SessionListService} from './session_list_service';
import {
  modelServerState,
  sessionExecutionPhase,
  sessionIsActive,
} from './session_status';
export type LifecycleOperation =
  | 'prepare'
  | 'initialize'
  | 'generate'
  | 'close';
@Injectable({providedIn: 'root'})
export class RuntimeService {
  readonly api = inject(ReportApiService);
  readonly sessions = inject(SessionListService);
  readonly report = inject(ReportStateService);
  readonly capabilities = signal<RuntimeCapabilities | null>(null);
  readonly job = signal<RuntimeJob | null>(null);
  readonly devices = signal<RunnerDevice[]>([]);
  readonly deviceError = signal('');
  readonly buildCatalogs = signal<Record<string, RunnerBuildCatalog>>({});
  readonly buildLoading = signal<Record<string, boolean>>({});
  async loadBuilds(id: string) {
    if (this.buildLoading()[id]) return;
    this.buildLoading.update((value) => ({...value, [id]: true}));
    try {
      const result = await this.api.deviceBuilds(id);
      this.buildCatalogs.update((value) => ({...value, [id]: result}));
    } catch (error) {
      this.buildCatalogs.update((value) => ({
        ...value,
        [id]: {
          deviceId: id,
          builds: [],
          canLaunch: false,
          reason: error instanceof Error ? error.message : String(error),
        },
      }));
    } finally {
      this.buildLoading.update((value) => ({...value, [id]: false}));
    }
  }
  private deviceEpoch = 0;
  private inspectionSequence = 0;
  private readonly deviceRequests = new Map<string, number>();
  deviceStatusLabel(device: RunnerDevice) {
    return (
      {
        unknown: 'Unchecked',
        checking: 'Checking…',
        ready: 'Online',
        connected: 'Connected',
        busy: 'Busy',
        unreachable: 'Unreachable',
        unsupported: 'Incompatible',
        unpaired: 'Pairing required',
      } as const
    )[device.status ?? 'unknown'];
  }
  async refreshDevices() {
    const epoch = ++this.deviceEpoch;
    try {
      const data = await this.api.devices();
      if (epoch !== this.deviceEpoch) return;
      this.devices.set(data.devices);
      this.deviceError.set(data.notice ?? '');
      await Promise.allSettled(
        this.devices().map((device) => this.inspectDevice(device.id)),
      );
    } catch (error) {
      if (epoch !== this.deviceEpoch) return;
      this.devices.set([]);
      this.deviceError.set(
        error instanceof Error ? error.message : String(error),
      );
    }
  }
  async inspectDevice(id: string) {
    id =
      this.devices().find((d) => d.runnerId === id)?.id ??
      (id === 'Server host' ? 'local' : id);
    if (
      !this.devices().some((d) => d.id === id) ||
      this.devices().find((d) => d.id === id)?.status === 'checking'
    )
      return;
    const epoch = this.deviceEpoch,
      ticket = ++this.inspectionSequence;
    this.deviceRequests.set(id, ticket);
    this.devices.update((devices) =>
      devices.map((d) =>
        d.id === id
          ? {
              ...d,
              status: 'checking',
              runnerState: 'unknown',
              reason: 'Checking without interrupting Sessions.',
              runner: undefined,
              checkedAt: undefined,
            }
          : d,
      ),
    );
    try {
      const result = await this.api.inspectDevice(id);
      if (epoch !== this.deviceEpoch || this.deviceRequests.get(id) !== ticket)
        return;
      this.devices.update((devices) =>
        devices.map((d) => (d.id === id ? {...d, ...result} : d)),
      );
    } catch {
      if (epoch !== this.deviceEpoch || this.deviceRequests.get(id) !== ticket)
        return;
      this.devices.update((devices) =>
        devices.map((d) =>
          d.id === id
            ? {
                ...d,
                status: 'unknown',
                reason: 'Check failed; Runner availability is unknown.',
              }
            : d,
        ),
      );
    } finally {
      if (this.deviceRequests.get(id) === ticket)
        this.deviceRequests.delete(id);
    }
  }
  readonly error = signal('');
  readonly submitting = signal(false);
  private readonly attachedId = signal<string | null>(null);
  readonly captureRecordId = this.attachedId.asReadonly();
  readonly selected = computed(() =>
    this.sessions.allItems().find((s) => s.id === this.captureRecordId()),
  );
  readonly ownerSession = computed(() => {
    const selected = this.selected();
    return selected?.parent_session_id
      ? this.sessions
          .allItems()
          .find((s) => s.id === selected.parent_session_id)
      : selected;
  });
  readonly execution = computed(() => this.ownerSession()?.execution);
  readonly active = computed(() => this.execution()?.phase === 'active');
  readonly generating = computed(
    () =>
      this.job()?.operation === 'generate' && !isTerminal(this.job()!.status),
  );
  readonly busy = computed(
    () =>
      this.submitting() ||
      ['starting', 'ending'].includes(this.execution()?.phase ?? '') ||
      (!!this.job() && !isTerminal(this.job()!.status)),
  );
  private source: EventSource | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private epoch = 0;
  private reconnectAttempts = 0;
  private readonly submittedPrompts = new Map<string, string>();
  constructor() {
    void this.refreshCapabilities();
    // A Session whose Model Server was asked for while its capture model is still being prepared
    // starts as soon as the list shows the preparation finished.
    effect(() => {
      const items = this.sessions.allItems();
      untracked(() => this.startPrepared(items));
    });
  }
  async refreshCapabilities() {
    try {
      this.capabilities.set(await this.api.capabilities());
    } catch (error) {
      this.error.set(String(error));
    }
  }
  capability(id = 'LiteRT-LM'): RuntimeCapability | undefined {
    const capabilities = this.capabilities();
    return (
      capabilities?.runtimes?.find((runtime) => runtime.id === id) ??
      (id === 'LiteRT-LM' && capabilities && !capabilities.runtimes
        ? {
            id,
            available: capabilities.available,
            reason: capabilities.reason,
            backends: capabilities.backends ?? ['CPU', 'GPU'],
            supported_options: capabilities.supported_options,
          }
        : undefined)
    );
  }
  async attach(id: string | null) {
    this.attachedId.set(
      this.sessions.allItems().some((record) => record.id === id) ? id : null,
    );
    const epoch = ++this.epoch;
    this.reconnectAttempts = 0;
    this.stopWatching();
    this.job.set(null);
    this.error.set('');
    this.submitting.set(false);
    const jobId = this.sessions.allItems().find((s) => s.id === id)?.job_id;
    if (jobId) {
      try {
        const job = await this.api.job(jobId);
        if (epoch !== this.epoch) return;
        this.job.set({
          ...job,
          prompt: this.submittedPrompts.get(job.id) ?? job.prompt,
        });
        if (!isTerminal(job.status)) this.watch(job, epoch);
        else await this.sessions.load();
      } catch (error) {
        if (epoch === this.epoch) this.error.set(String(error));
      }
    }
  }
  /**
   * The one place Session lifecycle requests are posted. Every operation except close
   * carries a fresh request id so a retried POST cannot start a second job.
   */
  lifecycle(
    id: string,
    operation: LifecycleOperation,
    options: {prompt?: string; maxTokens?: number} = {},
  ) {
    const path = `sessions/${encodeURIComponent(id)}/${operation === 'generate' ? 'turns' : operation}`;
    const body =
      operation === 'close'
        ? {}
        : {
            request_id: crypto.randomUUID(),
            ...(operation === 'generate'
              ? {
                  prompt: options.prompt ?? '',
                  max_output_tokens: options.maxTokens ?? 32,
                }
              : {}),
          };
    return this.api.post<RuntimeJob>(path, body);
  }
  /** Per-Session lifecycle requests from the home page; the row shows its own pending/error state. */
  readonly lifecyclePending = signal<Record<string, boolean>>({});
  readonly lifecycleErrors = signal<Record<string, string>>({});
  private async requestLifecycle(
    session: SessionSummary,
    operation: LifecycleOperation,
  ) {
    this.lifecyclePending.update((value) => ({...value, [session.id]: true}));
    this.lifecycleErrors.update((value) => ({...value, [session.id]: ''}));
    try {
      await this.lifecycle(session.id, operation);
      await this.sessions.load(true);
    } catch (error) {
      this.lifecycleErrors.update((value) => ({
        ...value,
        [session.id]: error instanceof Error ? error.message : String(error),
      }));
      if (operation !== 'prepare') await this.sessions.load(true);
    } finally {
      this.lifecyclePending.update((value) => ({
        ...value,
        [session.id]: false,
      }));
    }
  }
  async prepare(session: SessionSummary) {
    if (this.lifecyclePending()[session.id] || sessionIsActive(session)) return;
    await this.requestLifecycle(session, 'prepare');
  }
  /** Sessions to start once prepared: whether the list has shown them preparing, and since when. */
  private readonly wantOn = new Map<
    string,
    {seenPreparing: boolean; since: number}
  >();
  private startPrepared(items: readonly SessionSummary[]) {
    for (const [id, wait] of this.wantOn) {
      const session = items.find((item) => item.id === id);
      if (!session) this.wantOn.delete(id);
      else if (this.lifecyclePending()[id]) continue;
      else if (session.status === 'preparing') wait.seenPreparing = true;
      else if (!session.tap_profile || session.tap_prepared) {
        this.wantOn.delete(id);
        if (!sessionIsActive(session))
          void this.requestLifecycle(session, 'initialize');
      }
      // Preparation ended without a prepared model, or the list never showed it: give up. The
      // Server's failed status then reads as a failure and the switch retries from the start.
      else if (wait.seenPreparing || Date.now() - wait.since > 30_000)
        this.wantOn.delete(id);
    }
  }
  /** Start this Session's Model Server after its capture preparation, which is already requested. */
  startWhenPrepared(id: string) {
    this.wantOn.set(id, {seenPreparing: false, since: Date.now()});
    this.startPrepared(this.sessions.allItems());
  }
  /** The Model Server switch: turn it on, preparing the capture model first when that is still
   *  due, or turn off one that is starting, on or holding a failed slot. */
  async toggleModelServer(session: SessionSummary) {
    const phase = sessionExecutionPhase(session);
    if (this.lifecyclePending()[session.id] || phase === 'ending') return;
    if (['starting', 'active', 'unavailable'].includes(phase)) {
      this.wantOn.delete(session.id);
      await this.requestLifecycle(session, 'close');
    } else if (
      this.wantOn.delete(session.id) ||
      session.status === 'preparing'
    ) {
      // Cancelled while preparing: the preparation finishes, the Model Server stays off.
      return;
    } else if (session.tap_profile && !session.tap_prepared) {
      await this.requestLifecycle(session, 'prepare');
      if (!this.lifecycleErrors()[session.id])
        this.startWhenPrepared(session.id);
    } else await this.requestLifecycle(session, 'initialize');
  }
  /** Whether a start is queued behind this Session's capture preparation. */
  startQueued(id: string) {
    return this.wantOn.has(id);
  }
  modelServerState(session: SessionSummary) {
    return modelServerState(session, this.lifecycleErrors()[session.id]);
  }
  /** Start an idle Session, or end one that is starting, active or unavailable. */
  async toggleLifecycle(session: SessionSummary) {
    const phase = sessionExecutionPhase(session);
    if (this.lifecyclePending()[session.id] || phase === 'ending') return;
    const ending = ['starting', 'active', 'unavailable'].includes(phase);
    await this.requestLifecycle(session, ending ? 'close' : 'initialize');
  }
  async start(
    operation: 'prepare' | 'initialize' | 'generate',
    prompt = '',
    maxTokens = 32,
  ) {
    const id = this.captureRecordId();
    if (!id || this.busy()) return false;
    const epoch = this.epoch;
    this.error.set('');
    this.submitting.set(true);
    try {
      const job = await this.lifecycle(id, operation, {prompt, maxTokens});
      if (operation === 'generate') this.submittedPrompts.set(job.id, prompt);
      if (epoch !== this.epoch) {
        await this.sessions.load();
        return false;
      }
      this.job.set({...job, ...(operation === 'generate' ? {prompt} : {})});
      await this.sessions.load();
      if (epoch !== this.epoch) return false;
      if (isTerminal(job.status)) await this.finish(job.id, epoch);
      else this.watch(job, epoch);
      return job.status !== 'failed' && job.status !== 'cancelled';
    } catch (error) {
      if (epoch === this.epoch)
        this.error.set(error instanceof Error ? error.message : String(error));
      return false;
    } finally {
      if (epoch === this.epoch) this.submitting.set(false);
    }
  }
  async stop() {
    const job = this.job();
    if (!job || !this.generating()) return;
    try {
      await this.api.post(`jobs/${job.id}/cancel`, {});
    } catch (error) {
      this.error.set(String(error));
    }
  }
  private clearReconnectTimer() {
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }
  private stopWatching() {
    this.clearReconnectTimer();
    this.source?.close();
    this.source = null;
  }
  private watch(job: RuntimeJob, epoch: number) {
    this.stopWatching();
    let sequence = job.sequence;
    const source = new EventSource(
      `/api/jobs/${job.id}/events?after=${sequence}`,
    );
    this.source = source;
    const current = () => epoch === this.epoch && source === this.source;
    const connected = () => {
      this.clearReconnectTimer();
      this.reconnectAttempts = 0;
      this.error.set('');
    };
    source.onmessage = (event) => {
      if (!current()) return;
      let data: RuntimeJobEvent;
      try {
        data = JSON.parse(event.data) as RuntimeJobEvent;
      } catch {
        this.error.set('Invalid task update. Reconnecting…');
        return;
      }
      connected();
      if (data.sequence <= sequence) return;
      sequence = data.sequence;
      this.job.update((value) =>
        value
          ? {
              ...value,
              sequence,
              ...(data.type === 'status'
                ? {status: data.status, error: data.error ?? ''}
                : {}),
              ...(data.type === 'progress' ? {progress: data.message} : {}),
              ...(data.type === 'delta'
                ? {
                    output: {
                      ...value.output,
                      [data.runId ?? '']:
                        (value.output[data.runId ?? ''] ?? '') +
                        (data.text ?? ''),
                    },
                  }
                : {}),
            }
          : value,
      );
      if (data.type === 'status' && data.status && isTerminal(data.status)) {
        this.stopWatching();
        void this.finish(job.id, epoch);
      }
    };
    source.onopen = () => {
      if (current()) connected();
    };
    source.onerror = () => {
      if (!current()) return;
      // A stream the browser closed for good (HTTP 4xx) never reconnects by itself.
      if (source.readyState === EventSource.CLOSED) {
        void this.recover(job, epoch, source);
        return;
      }
      const interrupted = () =>
        this.error.set(
          'Connection interrupted. Reconnecting to the local task…',
        );
      // The server rotates healthy streams every 25 seconds; allow EventSource to reconnect.
      if (this.reconnectTimer === null)
        this.reconnectTimer = setTimeout(() => {
          if (!current()) return;
          this.reconnectTimer = null;
          if (source.readyState !== EventSource.OPEN) interrupted();
        }, 5000);
      void this.api
        .job(job.id)
        .then((jobState) => {
          if (!current()) return;
          if (isTerminal(jobState.status)) {
            this.stopWatching();
            void this.finish(job.id, epoch);
          }
        })
        .catch(() => {
          if (current() && source.readyState !== EventSource.OPEN)
            interrupted();
        });
    };
  }
  private watching(epoch: number, source: EventSource) {
    return epoch === this.epoch && source === this.source;
  }
  /** Re-create a closed stream with backoff; after five failures poll the job until it ends. */
  private async recover(job: RuntimeJob, epoch: number, source: EventSource) {
    this.clearReconnectTimer();
    let state: RuntimeJob;
    try {
      state = await this.api.job(job.id);
    } catch (error) {
      if (!this.watching(epoch, source)) return;
      const status = (error as {status?: unknown}).status;
      if (status === 400 || status === 404) {
        this.stopWatching();
        this.job.set(null);
        this.error.set(
          'The task no longer exists on the server. Reload the Session.',
        );
        return;
      }
      this.error.set('Connection interrupted. Retrying the local task…');
      this.reconnectTimer = setTimeout(() => {
        this.reconnectTimer = null;
        if (this.watching(epoch, source)) void this.recover(job, epoch, source);
      }, 2000);
      return;
    }
    if (!this.watching(epoch, source)) return;
    this.job.update((value) =>
      value && value.id === state.id
        ? {
            ...value,
            sequence: Math.max(value.sequence, state.sequence),
            status: state.status,
            error: state.error,
            output: state.output,
            progress: state.progress ?? value.progress,
          }
        : value,
    );
    if (isTerminal(state.status)) {
      this.stopWatching();
      void this.finish(job.id, epoch);
      return;
    }
    const attempt = ++this.reconnectAttempts;
    if (attempt <= 5) {
      this.error.set('Connection interrupted. Reconnecting to the local task…');
      this.reconnectTimer = setTimeout(
        () => {
          this.reconnectTimer = null;
          if (this.watching(epoch, source))
            this.watch(
              {...job, sequence: this.job()?.sequence ?? job.sequence},
              epoch,
            );
        },
        Math.min(16000, 1000 * 2 ** (attempt - 1)),
      );
    } else {
      this.error.set(
        'Live updates are unavailable; checking the task every few seconds.',
      );
      this.reconnectTimer = setTimeout(() => {
        this.reconnectTimer = null;
        if (this.watching(epoch, source)) void this.recover(job, epoch, source);
      }, 2000);
    }
  }
  private async finish(id: string, epoch: number) {
    try {
      const current = await this.api.job(id);
      if (epoch !== this.epoch) return;
      this.job.set({
        ...current,
        ...(this.job()?.id === current.id ? {prompt: this.job()?.prompt} : {}),
      });
      this.submittedPrompts.delete(id);
      await this.sessions.load();
      if (epoch !== this.epoch) return;
      if (
        current.status === 'completed' &&
        current.operation === 'generate' &&
        this.report.captureId() === current.session_id
      )
        await this.report.load();
      if (epoch !== this.epoch) return;
      if (current.status === 'completed' && current.operation === 'prepare')
        await this.refreshCapabilities();
      if (epoch !== this.epoch) return;
      this.error.set(current.error ?? '');
    } catch (error) {
      if (epoch === this.epoch) this.error.set(String(error));
    }
  }
}
