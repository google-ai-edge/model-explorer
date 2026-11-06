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

import {computed, inject, Injectable, signal} from '@angular/core';
import {captureId} from './capture_id';
import {CaptureStore} from './capture_store';
import {DevicesApi} from './devices_api';
import {JobsApi} from './jobs_api';
import {JobsService, type JobOwner} from './jobs_service';
import {RuntimeApi} from './runtime_api';
import {SessionListService} from './session_list_service';
import {SessionsApi} from './sessions_api';
import {isTerminal} from './contracts/jobs';
import type {ChatGeneration, SessionSummary} from './contracts/sessions';
import type {RunnerDevice, RunnerBuildCatalog} from './contracts/devices';
import type {RuntimeJob} from './contracts/jobs';
import type {RuntimeCapability, RuntimeCapabilities} from './contracts/runtime';

/** The runtimes and devices on offer, the attached capture record's Session lifecycle and its
 *  Model Server switch. The job a lifecycle request starts is followed by JobsService. */
@Injectable({providedIn: 'root'})
export class RuntimeService {
  private readonly runtimeApi = inject(RuntimeApi);
  private readonly devicesApi = inject(DevicesApi);
  private readonly sessionsApi = inject(SessionsApi);
  private readonly jobsApi = inject(JobsApi);
  private readonly jobs = inject(JobsService);
  readonly sessions = inject(SessionListService);
  readonly report = inject(CaptureStore);
  readonly capabilities = signal<RuntimeCapabilities | null>(null);
  readonly devices = signal<RunnerDevice[]>([]);
  readonly deviceError = signal('');
  readonly buildCatalogs = signal<Record<string, RunnerBuildCatalog>>({});
  readonly buildLoading = signal<Record<string, boolean>>({});
  async loadBuilds(id: string): Promise<void> {
    if (this.buildLoading()[id]) {
      return;
    }
    this.buildLoading.update((value) => ({...value, [id]: true}));
    try {
      const result = await this.devicesApi.deviceBuilds(id);
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
  /** Devices whose check is in flight. */
  readonly checking = signal<Record<string, boolean>>({});
  private deviceEpoch = 0;
  private capabilityRequest = 0;
  private inspectionSequence = 0;
  private readonly deviceRequests = new Map<string, number>();
  /** The device in words, as the server says it, or that its check is in flight. */
  deviceStatusText(device: RunnerDevice): string {
    return this.checking()[device.id] ? 'Checking…' : device.statusText;
  }
  /** Lists the devices anew; the listing already carries what each one is and offers. The
   *  server keeps that listing, so the capabilities, which say whether any device runs a runner,
   *  are read again after it. */
  async refreshDevices(): Promise<void> {
    const epoch = ++this.deviceEpoch;
    try {
      const data = await this.devicesApi.devices();
      if (epoch !== this.deviceEpoch) {
        return;
      }
      this.devices.set(data.devices);
      this.deviceError.set(data.notice ?? '');
    } catch (error) {
      if (epoch !== this.deviceEpoch) {
        return;
      }
      this.devices.set([]);
      this.deviceError.set(error instanceof Error ? error.message : String(error));
    }
    await this.refreshCapabilities();
  }
  async inspectDevice(id: string): Promise<void> {
    id = this.devices().find((d) => d.runnerId === id)?.id ?? (id === 'Server host' ? 'local' : id);
    if (!this.devices().some((d) => d.id === id) || this.checking()[id]) {
      return;
    }
    const epoch = this.deviceEpoch;
    const ticket = ++this.inspectionSequence;
    this.deviceRequests.set(id, ticket);
    this.checking.update((value) => ({...value, [id]: true}));
    try {
      const result = await this.devicesApi.inspectDevice(id);
      if (epoch !== this.deviceEpoch || this.deviceRequests.get(id) !== ticket) {
        return;
      }
      this.devices.update((devices) =>
        devices.map((d) => (d.id === id ? {...d, ...result, reason: ''} : d)),
      );
    } catch (error) {
      if (epoch !== this.deviceEpoch || this.deviceRequests.get(id) !== ticket) {
        return;
      }
      const detail = error instanceof Error ? error.message : String(error);
      this.devices.update((devices) =>
        devices.map((d) => (d.id === id ? {...d, reason: `Check failed: ${detail}`} : d)),
      );
    } finally {
      if (this.deviceRequests.get(id) === ticket) {
        this.deviceRequests.delete(id);
        this.checking.update((value) => ({...value, [id]: false}));
      }
    }
  }
  readonly error = signal('');
  readonly submitting = signal(false);
  private readonly attachedId = signal<string | null>(null);
  readonly captureRecordId = this.attachedId.asReadonly();
  readonly selected = computed(() =>
    this.sessions.items().find((s) => s.id === this.captureRecordId()),
  );
  readonly execution = computed(() => this.selected()?.execution);
  /** A request this page sent, or the job it follows, is still running. What the server allows
   *  is each record's `actions`; this covers the moment before the list shows the change. */
  readonly busy = computed(() => {
    const current = this.jobs.job();
    return this.submitting() || (current !== null && !isTerminal(current.status));
  });
  private epoch = 0;
  constructor() {
    void this.refreshCapabilities();
  }
  /** The job this attachment follows or is fetching, whoever started it. */
  private followed: string | null = null;
  /** Follows the attached Session's job `jobId`, if it is new to this page: a job another tab
   *  started (a turn, say) is followed as a reload follows it, its prompt, its live reply and Stop.
   *  The polled listing names the last job (`job_id`). */
  async follow(jobId: string | undefined): Promise<void> {
    // This page's own request names its job itself (start()).
    if (!jobId || this.submitting() || jobId === this.followed) {
      return;
    }
    this.followed = jobId;
    const epoch = this.epoch;
    try {
      const job = await this.jobsApi.job(jobId);
      if (epoch !== this.epoch || this.jobs.job()?.id === jobId) {
        return;
      }
      this.jobs.job.set(job);
      if (!isTerminal(job.status)) {
        this.watch(job, epoch);
      } else {
        await this.finish(job.id, epoch);
      }
    } catch (error) {
      if (epoch === this.epoch) {
        this.error.set(String(error));
      }
    }
  }
  /** What the runtimes offer now, and whether any device runs a runner. A newer request wins. */
  async refreshCapabilities(): Promise<void> {
    const request = ++this.capabilityRequest;
    try {
      const capabilities = await this.runtimeApi.capabilities();
      if (request === this.capabilityRequest) {
        this.capabilities.set(capabilities);
      }
    } catch (error) {
      if (request === this.capabilityRequest) {
        this.error.set(String(error));
      }
    }
  }
  /** A runtime's capability, its options as the server's table has them; a device's own is its
   *  RunnerDevice.capability. */
  capability(id = 'LiteRT-LM'): RuntimeCapability | undefined {
    return this.capabilities()?.runtimes?.find((runtime) => runtime.id === id);
  }
  async attach(id: string | null): Promise<void> {
    this.attachedId.set(this.sessions.items().some((record) => record.id === id) ? id : null);
    const epoch = ++this.epoch;
    this.jobs.reset();
    this.error.set('');
    this.submitting.set(false);
    const jobId = this.sessions.items().find((s) => s.id === id)?.job_id;
    this.followed = jobId ?? null;
    if (jobId) {
      try {
        const job = await this.jobsApi.job(jobId);
        if (epoch !== this.epoch) {
          return;
        }
        // A turn's job carries its prompt: a stopped turn shows what it asked after a reload too.
        this.jobs.job.set(job);
        if (!isTerminal(job.status)) {
          this.watch(job, epoch);
        } else {
          await this.sessions.load();
        }
      } catch (error) {
        if (epoch === this.epoch) {
          this.error.set(String(error));
        }
      }
    }
  }
  /** Per-Session lifecycle requests from the home page; the row shows its own pending/error state. */
  readonly lifecyclePending = signal<Record<string, boolean>>({});
  readonly lifecycleErrors = signal<Record<string, string>>({});
  /** The Model Server switch: turns it off while it is on, cancels its start while it starts, as
   *  the server allows (`model_server.button`). A Session starts as it is created: nothing turns
   *  one on. */
  async closeModelServer(session: SessionSummary): Promise<void> {
    const button = session.model_server.button;
    if (this.lifecyclePending()[session.id] || !button?.allowed) {
      return;
    }
    this.lifecyclePending.update((value) => ({...value, [session.id]: true}));
    this.lifecycleErrors.update((value) => ({...value, [session.id]: ''}));
    try {
      await this.sessionsApi.close(session.id);
      await this.sessions.load(true);
    } catch (error) {
      this.lifecycleErrors.update((value) => ({
        ...value,
        [session.id]: error instanceof Error ? error.message : String(error),
      }));
      await this.sessions.load(true);
    } finally {
      this.lifecyclePending.update((value) => ({...value, [session.id]: false}));
    }
  }
  /** Sends a turn to chat `chat` of the attached Session, the live one as the page shows it. */
  start(operation: 'generate', chat: number, prompt = ''): Promise<boolean> {
    return this.run((id) => this.sessionsApi.turn(id, chat, prompt));
  }
  /** Starts a new chat of the attached Session, each side with its settings; its live chat, if
   *  any, ends. The job is followed as a turn's is. */
  startChat(generation: ChatGeneration): Promise<boolean> {
    return this.run((id) => this.sessionsApi.startChat(id, generation));
  }
  /** Posts a request of the attached Session that answers a job, and follows the job; whether it
   *  did not fail. */
  private async run(request: (id: string) => Promise<RuntimeJob>): Promise<boolean> {
    const id = this.captureRecordId();
    if (!id || this.busy()) {
      return false;
    }
    const epoch = this.epoch;
    this.error.set('');
    this.submitting.set(true);
    try {
      const job = await request(id);
      this.followed = job.id;
      if (epoch !== this.epoch) {
        await this.sessions.load();
        return false;
      }
      this.jobs.job.set(job);
      await this.sessions.load();
      if (epoch !== this.epoch) {
        return false;
      }
      if (isTerminal(job.status)) {
        await this.finish(job.id, epoch);
      } else {
        this.watch(job, epoch);
      }
      return job.status !== 'failed' && job.status !== 'cancelled';
    } catch (error) {
      if (epoch === this.epoch) {
        this.error.set(error instanceof Error ? error.message : String(error));
      }
      return false;
    } finally {
      if (epoch === this.epoch) {
        this.submitting.set(false);
      }
    }
  }
  async stop(): Promise<void> {
    const job = this.jobs.job();
    if (!job || !this.jobs.generating()) {
      return;
    }
    try {
      await this.jobsApi.cancel(job.id);
    } catch (error) {
      this.error.set(String(error));
    }
  }
  /** Follows a job for the attachment of `epoch`. */
  private watch(job: RuntimeJob, epoch: number): void {
    this.jobs.watch(job, this.owner(epoch));
  }
  private owner(epoch: number): JobOwner {
    return {
      current: () => epoch === this.epoch,
      error: this.error,
      finish: (id) => void this.finish(id, epoch),
    };
  }
  private async finish(id: string, epoch: number): Promise<void> {
    try {
      const current = await this.jobsApi.job(id);
      if (epoch !== this.epoch) {
        return;
      }
      this.jobs.job.set(current);
      await this.sessions.load();
      if (epoch !== this.epoch) {
        return;
      }
      // A turn of the chat shown that ended, completed or not, is read anew once the chat has a
      // capture: a rejected turn is kept, and the first one kept is what the page waits for.
      if (
        current.operation === 'generate' &&
        current.chat != null &&
        this.report.captureId() === captureId(current.session_id, current.chat) &&
        this.sessions
          .items()
          .find((record) => record.id === current.session_id)
          ?.chats.find((chat) => chat.n === current.chat)?.has_capture
      ) {
        await this.report.load();
      }
      if (epoch !== this.epoch) {
        return;
      }
      // A turn that did not fit says so under the composer, which keeps its text.
      this.error.set(current.does_not_fit ? '' : (current.error ?? ''));
    } catch (error) {
      if (epoch === this.epoch) {
        this.error.set(String(error));
      }
    }
  }
}
