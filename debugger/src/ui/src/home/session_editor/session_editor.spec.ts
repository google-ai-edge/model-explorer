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

import {ChangeDetectorRef, ElementRef, Provider} from '@angular/core';
import {TestBed} from '@angular/core/testing';
import {MatDialog} from '@angular/material/dialog';
import {ArtifactsApi} from '../../server/artifacts_api';
import type {SessionConfig, SessionSummary} from '../../server/contracts/sessions';
import {RuntimeService} from '../../server/runtime_service';
import {SessionListService} from '../../server/session_list_service';
import {JobsApi} from '../../server/jobs_api';
import {
  allowed,
  chat,
  deferred,
  listing,
  pytorchCapability,
  refused,
  session,
} from '../../server/testing/session_fixtures';
import {SessionEditor} from './session_editor';

type Run = SessionConfig['runs'][number];
type Payload = Partial<SessionConfig> & {id?: string};

/** The services the editor reads, as fakes the spec drives. */
function fakes() {
  const state = {requests: [] as {operation: string; payload: Payload}[]};
  const listingState = {value: listing()};
  const failure = {on: false};
  const sessionMock = {
    listing: () => listingState.value,
    items: (): SessionSummary[] => listingState.value.sessions,
    manage: async (operation: string, payload: Payload): Promise<SessionSummary> => {
      state.requests.push({operation, payload});
      if (failure.on) throw new Error('Target: Pixel 9 is in use by Session X.');
      return session(payload.id ?? 'saved', {...payload, id: payload.id ?? 'saved'});
    },
    load: async () => {},
  };
  const runtimeMock: Record<string, unknown> = {
    devices: () => [],
    checking: () => ({}),
    capabilities: () => null,
    capability: () => undefined,
    refreshCapabilities: async () => {},
  };
  const artifactsMock: Record<string, unknown> = {};
  const jobsMock: Record<string, unknown> = {};
  const element = {nativeElement: {querySelector: () => null}};
  return {
    state,
    failure,
    sessionMock,
    runtimeMock,
    artifactsMock,
    jobsMock,
    element,
  };
}

/** The providers of an editor over `fakes`, with `overrides` laid over them. */
function providers(f: ReturnType<typeof fakes>, overrides: Provider[] = []): Provider[] {
  return [
    {provide: RuntimeService, useValue: f.runtimeMock},
    {provide: ArtifactsApi, useValue: f.artifactsMock},
    {provide: SessionListService, useValue: f.sessionMock},
    {provide: ElementRef, useValue: f.element},
    {provide: ChangeDetectorRef, useValue: {markForCheck() {}}},
    {provide: MatDialog, useValue: {}},
    {provide: JobsApi, useValue: f.jobsMock},
    ...overrides,
  ];
}

/** An editor creating the offline Session the cases start from. */
function offlineEditor() {
  const f = fakes();
  TestBed.configureTestingModule({providers: providers(f)});
  const editor = TestBed.runInInjectionContext(() => new SessionEditor());
  editor.config = {
    name: 'Offline',
    runs: session().runs,
    tap_profile: 'custom-outputs-v1',
    tap_points: {'model.litertlm': ['out']},
  };
  editor.advanced = {ref: true, target: true};
  return {...f, editor};
}

describe('SessionEditor', () => {
  beforeEach(() =>
    vi.stubGlobal('requestAnimationFrame', (fn: () => void) => {
      fn();
      return 1;
    }),
  );
  afterEach(() => vi.unstubAllGlobals());

  it('a create the server refuses keeps everything chosen and says why', async () => {
    // Create starts the Session too: one that cannot start now is refused, nothing is saved, and
    // the editor stays open with everything filled in.
    const {editor, state, failure} = offlineEditor();
    expect(editor.deviceConfigurationError()).toBe('');
    expect(editor.captureReady()).toBe(true);
    const finished: string[] = [];
    editor.finished.subscribe(() => finished.push('finished'));
    failure.on = true;
    await editor.save();
    expect(editor.config.name).toBe('Offline');
    expect(editor.config.tap_points).toStrictEqual({'model.litertlm': ['out']});
    expect(editor.error()).toBe('Target: Pixel 9 is in use by Session X.');
    expect(editor.saving()).toBe(false);
    expect(finished).toStrictEqual([]);
    failure.on = false;
    await editor.save();
    expect(state.requests.map((r) => r.operation)).toStrictEqual(['create', 'create']);
    expect(editor.error()).toBe('');
    expect(finished).toStrictEqual(['finished']);
  });

  it('creating a Session sends no chat settings', async () => {
    // A Session is created with no chat, and each chat's settings are its own, set as it starts
    // on the Session's page.
    const f = fakes();
    TestBed.configureTestingModule({
      providers: providers(f, [
        {provide: RuntimeService, useValue: {...f.runtimeMock, refreshDevices: async () => {}}},
      ]),
    });
    const saving = TestBed.runInInjectionContext(() => new SessionEditor());
    saving.ngOnInit();
    expect('chats' in saving.config).toBe(false);
    saving.deviceConfigurationError = () => '';
    saving.artifactsReady = () => true;
    saving.captureReady = () => true;
    await saving.save();
    expect(f.state.requests.map((r) => r.operation)).toStrictEqual(['create']);
    expect('generation' in f.state.requests[0].payload).toBe(false);
    expect('chats' in f.state.requests[0].payload).toBe(false);
    expect('id' in f.state.requests[0].payload).toBe(false);
  });

  it('the operators ticked inside the layers are saved once for both sides', async () => {
    const f = fakes();
    TestBed.configureTestingModule({
      providers: providers(f, [
        {provide: RuntimeService, useValue: {...f.runtimeMock, refreshDevices: async () => {}}},
      ]),
    });
    const editor = TestBed.runInInjectionContext(() => new SessionEditor());
    editor.ngOnInit();
    expect(editor.config.tap_ops, 'a new Session ticks none').toEqual([]);
    editor.config.tap_ops = ['attn/q_proj', 'ffn/gate_proj'];
    editor.deviceConfigurationError = () => '';
    editor.artifactsReady = () => true;
    // The server refuses those ticked: the editor does not save.
    editor.opsReady.set(false);
    expect(editor.captureReady()).toBe(false);
    editor.opsReady.set(true);
    await editor.save();
    expect(f.state.requests.map((r) => r.payload.tap_ops)).toEqual([
      ['attn/q_proj', 'ffn/gate_proj'],
    ]);
  });

  it('device picker refuses with the server words; save waits only for a runner version', () => {
    // The device picker refuses what the server refuses, with its words; the run's options are the
    // device's offer as the server merged it.
    const {editor, runtimeMock} = offlineEditor();
    // As the save before left it: the server runs a runtime.
    runtimeMock['capabilities'] = () => ({available: true, models: []});
    const offer = {id: 'LiteRT-LM', available: true, reason: '', backends: ['CPU', 'GPU']};
    const device = (id: string, extra: object = {}) => ({
      id,
      name: id === 'local' ? 'This Mac' : 'Pixel',
      platform: id === 'local' ? 'macOS' : 'Android',
      statusText: 'Connected',
      choosable: allowed,
      holder: '',
      capability: offer,
      ...extra,
    });
    const busy = device('android:A', {
      statusText: 'Busy',
      choosable: refused('Pixel is in use by another Session.'),
      holder: 'In use by CPU vs CPU · Reference',
    });
    runtimeMock['devices'] = () => [device('local'), busy];
    runtimeMock['deviceStatusText'] = (d: {statusText: string}) => d.statusText;
    editor.config.runs = [
      {id: 'ref', runtime: 'LiteRT-LM', device: 'local'},
      {id: 'target', runtime: 'LiteRT-LM', device: 'android:A'},
    ];
    expect(
      editor.deviceConfigurationError(),
      'a held device does not stop a save; turning it on shows the reason',
    ).toBe('');
    expect(editor.unavailableDevices()).toStrictEqual([editor.deviceLabel('android:A')]);
    expect(editor.deviceDescriptions()[editor.deviceLabel('android:A')]).toMatch(
      /In use by CPU vs CPU · Reference/,
    );
    expect(editor.deviceCapability(editor.config.runs[0])).toBe(offer);
    editor.config.runs[1].device = 'local';
    expect(editor.deviceConfigurationError(), 'the server, not the editor, pairs devices').toBe('');
    editor.config.runs[1].device = 'ios:gone';
    expect(editor.deviceConfigurationError(), 'nor does a device not listed now').toBe('');
    // The runner version picker: this computer ('not_running') offers its builds, a phone none;
    // saving waits for an available one, and a Session's version no longer kept shows bare.
    runtimeMock['devices'] = () => [
      device('local', {runnerState: 'not_running'}),
      device('android:A'),
    ];
    const commit = '61dcfc77225019e067580dceb3ea268483d1938c';
    // What the kept build's last runner here reported: it takes no CPU threads.
    const buildOffer = {
      ...offer,
      session_options: [
        {key: 'cpuThreads', label: 'Text CPU threads', available: false, reason: 'Not taken.'},
      ],
    };
    runtimeMock['buildCatalogs'] = () => ({
      local: {
        deviceId: 'local',
        canLaunch: true,
        builds: [
          {
            id: commit,
            label: 'LiteRT-LM 61dcfc772250 · 2026-10-15',
            version: commit,
            cl: '999909015',
            available: true,
            capability: buildOffer,
          },
          {
            id: 'a484',
            label: 'LiteRT-LM a484d21d5113 · 2026-10-12',
            available: false,
            reason: 'runner is missing. Keep it again.',
          },
          {
            id: 'dev',
            label: 'Development build · commit known at start',
            version: '',
            cl: '',
            available: true,
          },
        ],
      },
    });
    editor.config.runs[0].device = 'local';
    editor.config.runs[1].device = 'android:A';
    const ref: Run = editor.config.runs[0];
    expect(editor.buildLabel(ref)).toBe('Select runner version');
    expect(editor.deviceConfigurationError(), 'a phone needs no version; this computer does').toBe(
      'Reference: select an available runner version',
    );
    expect(editor.unavailableBuilds(ref)).toStrictEqual(['LiteRT-LM a484d21d5113 · 2026-10-12']);
    expect(editor.buildDescriptions(ref)['LiteRT-LM 61dcfc772250 · 2026-10-15']).toBe(
      commit + ' · CL 999909015',
    );
    expect(editor.buildDescriptions(ref)['LiteRT-LM a484d21d5113 · 2026-10-12']).toBe(
      'runner is missing. Keep it again.',
    );
    editor.selectBuild(ref, 'LiteRT-LM a484d21d5113 · 2026-10-12');
    expect(ref.runnerBuild, 'an unavailable version cannot be chosen').toBe(undefined);
    editor.selectBuild(ref, 'Development build · commit known at start');
    expect(ref.runnerBuild).toBe('dev');
    expect(editor.deviceCapability(ref), 'a build that never ran: the device offer').toBe(offer);
    ref.runtime = 'LiteRT-LM';
    ref.cpuThreads = '4';
    editor.selectBuild(ref, 'LiteRT-LM 61dcfc772250 · 2026-10-15');
    expect(ref.runnerBuild).toBe(commit);
    expect(editor.deviceCapability(ref), "the chosen build's own offer").toBe(buildOffer);
    expect(ref.cpuThreads, 'an option the chosen build does not take is reset').toBe('');
    expect(editor.deviceConfigurationError()).toBe('');
    ref.runnerBuild = 'b'.repeat(40);
    expect(editor.buildLabel(ref), 'a version no longer kept shows bare').toBe('b'.repeat(40));
    expect(editor.deviceConfigurationError()).toBe('Reference: select an available runner version');
  });

  it("a phone's picker entry shows the version of its App, or that it has none", () => {
    // A phone runs the App installed on it, one version per phone: the server reads its version
    // without launching it and says when it is not the one the server is written for.
    const {editor, runtimeMock} = offlineEditor();
    const commit = '0123456789abcdef0123456789abcdef01234567';
    const phone = (id: string, name: string, extra: object) => ({
      id,
      name,
      platform: 'Android',
      statusText: 'Connected',
      choosable: allowed,
      holder: '',
      ...extra,
    });
    runtimeMock['devices'] = () => [
      phone('android:P', 'Pixel 10', {
        app: {
          installed: true,
          version: commit,
          commitDate: '2026-10-12',
          label: 'LiteRT-LM 0123456789ab · 2026-10-12',
          notice: 'Built against LiteRT-LM 0123456789ab; this server is written for 61dcfc772250.',
        },
      }),
      phone('android:N', 'sdk_gphone64', {
        statusText: 'Connected · no Runner App',
        choosable: refused('No Runner App is installed on sdk_gphone64.'),
        app: {installed: false, version: '', label: 'No Runner App installed'},
      }),
      phone('ios-sim:ABCD', 'iPhone 17', {
        platform: 'iOS Simulator',
        transport: 'simulator',
        statusText: 'Connected · 2 runners running',
      }),
    ];
    runtimeMock['deviceStatusText'] = (d: {statusText: string}) => d.statusText;
    // A transport its platform names is said once; its status once, in the label.
    expect(editor.deviceLabel('ios-sim:ABCD')).toBe(
      'iPhone 17 · iOS Simulator · Connected · 2 runners running · ABCD',
    );
    expect(editor.deviceDescriptions()[editor.deviceLabel('ios-sim:ABCD')]).toBe('');
    const descriptions = editor.deviceDescriptions();
    // Its status is in its label: the description does not say it again.
    expect(editor.deviceLabel('android:P')).toContain('Connected');
    expect(descriptions[editor.deviceLabel('android:P')]).toBe(
      'Runner App: LiteRT-LM 0123456789ab · 2026-10-12',
    );
    expect(descriptions[editor.deviceLabel('android:N')]).toBe(
      'No Runner App is installed on sdk_gphone64.',
    );
    expect(editor.unavailableDevices()).toStrictEqual([editor.deviceLabel('android:N')]);
    const pixel = editor.runnerDevice('android:P')!;
    expect(editor.appLabel(pixel.app!)).toBe('Runner App: LiteRT-LM 0123456789ab · 2026-10-12');
    expect(editor.appLabel(editor.runnerDevice('android:N')!.app!)).toBe('No Runner App installed');
  });

  it("a PyTorch run offers the server's PyTorch models, build and options; a LiteRT-LM run never sees them", async () => {
    const {editor, runtimeMock} = offlineEditor();
    const pytorch = pytorchCapability();
    const litert = {
      id: 'LiteRT-LM',
      available: true,
      reason: '',
      backends: ['CPU', 'GPU'],
      session_options: [
        {
          key: 'contextLength',
          label: 'Context length',
          help: '',
          group: 'Context',
          type: 'integer' as const,
          default: 2048,
          available: true,
        },
      ],
    };
    const runtimes = [litert, pytorch];
    runtimeMock['capabilities'] = () => ({
      available: true,
      models: [
        {artifact: 'gemma.litertlm', name: 'Gemma', runtime: 'LiteRT-LM'},
        {artifact: '/models/qwen', name: 'Qwen', runtime: 'PyTorch'},
      ],
      runtimes,
      tap_profiles: [{id: 'custom-outputs-v1', name: 'Custom', description: ''}],
    });
    runtimeMock['capability'] = (id = 'LiteRT-LM') => runtimes.find((r) => r.id === id);
    runtimeMock['devices'] = () => [
      {
        id: 'local',
        name: 'This Mac',
        statusText: 'Connected',
        choosable: allowed,
        holder: '',
        runnerState: 'not_running',
      },
    ];
    runtimeMock['loadBuilds'] = async () => {};
    runtimeMock['buildCatalogs'] = () => ({
      local: {
        deviceId: 'local',
        canLaunch: true,
        builds: [
          {id: 'dev', label: 'Development build', runtime: 'LiteRT-LM', available: true},
          {
            id: 'pytorch',
            label: 'PyTorch runner',
            runtime: 'PyTorch',
            available: true,
            capability: pytorch,
          },
        ],
      },
    });
    editor.config.runs = [
      {id: 'ref', runtime: 'LiteRT-LM', device: 'local', artifact: 'gemma.litertlm'},
      {id: 'target', runtime: 'LiteRT-LM', device: 'local', artifact: 'gemma.litertlm'},
    ];
    const [ref, target] = editor.config.runs;
    await editor.loadSelectedBuilds();
    expect(editor.buildOptions(target), 'a LiteRT-LM run never sees the PyTorch build').toEqual([
      'Development build',
    ]);
    expect(target.runnerBuild).toBe('dev');
    editor.selectRuntime(ref, 'PyTorch');
    await editor.loadSelectedBuilds();
    expect(editor.serverModels(ref).map((m) => m.name)).toEqual(['Qwen']);
    expect(ref.source).toBe('huggingface');
    expect(editor.buildOptions(ref)).toEqual(['PyTorch runner']);
    expect(ref.runnerBuild, 'the one PyTorch build is chosen').toBe('pytorch');
    expect(editor.deviceConfigurationError()).toBe('');
    expect(editor.deviceCapability(ref)).toBe(pytorch);
    // The device notice follows each side's runtime.
    expect(editor.backendsText(editor.deviceCapability(ref)!)).toBe('PyTorch · CPU/MPS');
    expect(editor.backendsText(editor.deviceCapability(target)!)).toBe('LiteRT-LM · CPU/GPU');
    expect(ref.backend).toBe('CPU');
    expect(ref.contextLength, "the server's default").toBe('4096');
    ref.dtype = 'bfloat16';
    expect(editor.differences(ref)).toContainEqual({label: 'Model dtype', value: 'bfloat16'});
    expect(
      editor.differences(ref).map((d) => d.label),
      'only the PyTorch options the server describes',
    ).toEqual(['Model dtype']);
    editor.selectServerModel(ref, 'gemma.litertlm');
    expect(ref.artifact, 'a LiteRT-LM model is not a PyTorch run').toBe('');
    editor.selectServerModel(ref, '/models/qwen');
    expect(editor.artifactsReady()).toBe(true);
    expect(editor.tapReason(ref), 'PyTorch captures by its own options').toBe('');
    expect(editor.capturable(ref)).toBe(false);
    expect(editor.capturable(target)).toBe(true);
    expect(editor.pytorchCapture(target), 'a LiteRT-LM side has its anchors instead').toBe(null);
    expect(editor.pytorchCapture(ref)?.text).toBe("PyTorch captures every layer's four anchors.");
    expect(ref.observe, 'a PyTorch side always captures its anchors').toBe('anchors');
    expect(editor.pairedPyTorch(), 'the outputs chosen pair with its modules').toBe('/models/qwen');
    editor.setBudget(ref, 6);
    expect([ref.observe, ref.observeMemoryGiB]).toEqual(['anchors', '6']);
    editor.selectRuntime(ref, 'LiteRT-LM');
    expect([ref.observe, ref.observeMemoryGiB], 'a LiteRT-LM run has no such option').toEqual([
      '',
      '',
    ]);
  });

  it('Refresh devices spins until the devices are listed anew', async () => {
    const {editor, runtimeMock} = offlineEditor();
    const listed = deferred<void>();
    runtimeMock['refreshDevices'] = () => listed.promise;
    const refreshing = editor.refreshDevices();
    expect(editor.refreshing()).toBe(true);
    listed.resolve();
    await refreshing;
    expect(editor.refreshing()).toBe(false);
    // A listing that fails stops the spin too.
    runtimeMock['refreshDevices'] = async () => {
      throw new Error('offline');
    };
    await expect(editor.refreshDevices()).rejects.toThrow('offline');
    expect(editor.refreshing()).toBe(false);
  });

  it("a PyTorch side always observes, and captures modules for the LiteRT-LM side's outputs chosen", () => {
    const {editor, runtimeMock} = offlineEditor();
    const pytorch = pytorchCapability();
    runtimeMock['capabilities'] = () => ({
      available: true,
      models: [{artifact: '/models/gemma', name: 'gemma', runtime: 'PyTorch'}],
    });
    runtimeMock['capability'] = (id = 'LiteRT-LM') => (id === 'PyTorch' ? pytorch : undefined);
    editor.config.tap_points = {};
    const [ref, target] = editor.config.runs;
    editor.selectRuntime(ref, 'PyTorch');
    expect(ref.observe, 'no switch: always').toBe('anchors');
    expect(editor.sideAdvanced(ref)).toBe(false);
    editor.selectServerModel(ref, '/models/gemma');
    expect(editor.pairedLitert(ref), 'the other side chose nothing').toBe(null);
    editor.selectTaps(target, ['out']);
    expect(editor.pairedLitert(ref)).toEqual({artifact: 'model.litertlm', points: ['out']});
    expect(editor.pairedPyTorch()).toBe('/models/gemma');
  });

  it('creating a Session starts its Model Server, which prepares selected outputs on the server', async () => {
    // Create is the one request: the server starts the Session's Model Server as it creates it,
    // and prepares selected outputs first, in the same job.
    const f = fakes();
    const starting = {...f.runtimeMock, capabilities: () => ({available: true, models: []})};
    TestBed.configureTestingModule({
      providers: providers(f, [{provide: RuntimeService, useValue: starting}]),
    });
    const create = (tap_profile: string) => {
      const item = TestBed.runInInjectionContext(() => new SessionEditor());
      item.config = {
        name: 'Live',
        runs: session().runs,
        tap_profile,
        tap_points: tap_profile ? {'model.litertlm': ['out']} : {},
      };
      item.advanced = tap_profile ? {ref: true, target: true} : {};
      item.deviceConfigurationError = () => '';
      if (tap_profile) item.setScanReady('model.litertlm', true);
      const created: string[] = [];
      item.created.subscribe((id) => created.push(id));
      return {item, created};
    };
    const plain = create('');
    await plain.item.save();
    expect(plain.created).toStrictEqual(['saved']);
    const tapped = create('custom-outputs-v1');
    await tapped.item.save();
    expect(tapped.created).toStrictEqual(['saved']);
    expect(tapped.item.saveNotice()).toMatch(/starts its Model Server/);
    expect(f.state.requests.map((r) => r.operation)).toStrictEqual(['create', 'create']);
  });

  it('every side captures its anchors, with no switch; outputs chosen go beside them', async () => {
    const f = fakes();
    const base = pytorchCapability();
    f.runtimeMock['capabilities'] = () => ({
      available: true,
      models: [{artifact: '/models/gemma', name: 'gemma', runtime: 'PyTorch'}],
      tap_anchors: {name: 'Capture the anchors', description: 'Every layer.'},
    });
    f.runtimeMock['capability'] = (id = 'LiteRT-LM') => (id === 'PyTorch' ? base : undefined);
    f.runtimeMock['refreshDevices'] = async () => {};
    TestBed.configureTestingModule({providers: providers(f)});
    const editor = TestBed.runInInjectionContext(() => new SessionEditor());
    editor.ngOnInit();
    editor.deviceConfigurationError = () => '';
    const [ref, target] = editor.config.runs;
    expect(editor.sideAnchors(ref) && editor.sideAnchors(target)).toBe(true);
    expect(editor.capturesAnchors()).toBe(true);
    expect(editor.saveNotice()).toMatch(/prepares the tensor capture/);
    // One file: one copy observes both sides, so its outputs are set for both.
    ref.artifact = 'a.litertlm';
    target.artifact = 'a.litertlm';
    editor.selectTaps(target, ['out']);
    expect(editor.advanced).toEqual({ref: true, target: true});
    editor.setScanReady('a.litertlm', true);
    editor.selectTaps(target, []);
    expect(editor.captureReady(), 'beside the anchors, a side may choose none').toBe(true);
    // The create payload: no anchors switch (the server sets it), and a PyTorch side observes.
    editor.selectRuntime(ref, 'PyTorch');
    editor.selectServerModel(ref, '/models/gemma');
    editor.config.name = 'Mixed';
    await editor.save();
    const payload = f.state.requests.at(-1)?.payload;
    expect(f.state.requests.at(-1)?.operation).toBe('create');
    expect('tap_anchors' in (payload ?? {}), 'the server sets the anchors').toBe(false);
    expect(payload?.runs?.find((run) => run.id === 'ref')?.observe).toBe('anchors');
    expect(payload?.runs?.find((run) => run.id === 'target')?.observe ?? '').toBe('');
    expect('model' in (payload ?? {}), 'the server names the models').toBe(false);
  });

  it('names the devices in the suggested name only when a side runs off this computer', () => {
    const {editor, runtimeMock} = offlineEditor();
    runtimeMock['devices'] = () => [
      {id: 'local', name: 'This Mac', statusText: 'Connected', choosable: allowed, holder: ''},
      {id: 'android:A', name: 'Pixel 9', statusText: 'Connected', choosable: allowed, holder: ''},
    ];
    const [ref, target] = editor.config.runs;
    ref.backend = 'CPU';
    target.backend = 'GPU';
    ref.device = 'local';
    target.device = 'Server host';
    editor.suggest();
    // Both models chosen, before the server's comparison answers: their names.
    expect(editor.config.name).toBe('model.litertlm · CPU vs GPU');
    target.device = 'android:A';
    editor.suggest();
    expect(editor.config.name).toBe('model.litertlm · This Mac CPU vs Pixel 9 GPU');
    target.artifact = '';
    editor.suggest();
    expect(editor.config.name).toBe('New session · This Mac CPU vs Pixel 9 GPU');
  });

  it("shows the server's hint of the two models and suggests a name by them; each side runs the default build", async () => {
    const {editor, runtimeMock, artifactsMock} = offlineEditor();
    const asked: string[][] = [];
    artifactsMock['compareModels'] = async (ref: string, target: string) => {
      asked.push([ref, target]);
      return {
        code: 'same_file',
        text: 'Same model file: every analysis compares.',
        label: 'gemma.litertlm',
        capture_shared: 'Both sides load this file.',
      };
    };
    await editor.loadComparison();
    expect(asked).toEqual([['model.litertlm', 'model.litertlm']]);
    expect(editor.comparison()?.text).toBe('Same model file: every analysis compares.');
    editor.suggest();
    expect(editor.config.name).toMatch(/^gemma\.litertlm · /);
    const [ref] = editor.config.runs;
    editor.clearFile(ref);
    await Promise.resolve();
    expect(editor.comparison(), 'no hint until both models are chosen').toBe(null);
    // The runner version picker is hidden: a side runs its runtime's default build.
    expect(editor.showRunnerVersion).toBe(false);
    runtimeMock['devices'] = () => [
      {
        id: 'local',
        name: 'This Mac',
        statusText: 'Connected',
        choosable: allowed,
        holder: '',
        runnerState: 'not_running',
      },
    ];
    runtimeMock['loadBuilds'] = async () => {};
    runtimeMock['buildCatalogs'] = () => ({
      local: {
        deviceId: 'local',
        canLaunch: true,
        builds: [
          {id: 'c'.repeat(40), label: 'LiteRT-LM new', available: true, default: true},
          {id: 'b'.repeat(40), label: 'LiteRT-LM old', available: true},
          {id: 'dev', label: 'Development build', available: true},
        ],
      },
    });
    editor.config.runs = [
      {id: 'ref', runtime: 'LiteRT-LM', device: 'local'},
      {id: 'target', runtime: 'LiteRT-LM', device: 'local', runnerBuild: 'a'.repeat(40)},
    ];
    await editor.loadSelectedBuilds();
    expect(
      editor.config.runs.map((run) => run.runnerBuild),
      'the default, for a side that chose none or one no longer kept',
    ).toEqual(['c'.repeat(40), 'c'.repeat(40)]);
  });

  it('same-file input reset, upload cancellation, ignored late result, successful same-file retry, progress', async () => {
    const {editor, runtimeMock, artifactsMock} = offlineEditor();
    runtimeMock['capabilities'] = () => ({available: true, models: []});
    type Upload = (file: File, options: {signal: AbortSignal; onProgress: Progress}) => unknown;
    type Progress = (loaded: number, total: number) => void;
    let observedSignal: AbortSignal | undefined;
    const pending = deferred<{artifact: string}>();
    artifactsMock['upload'] = ((_file, options) => {
      observedSignal = options.signal;
      return pending.promise;
    }) as Upload;
    const run: Run = {id: 'ref', runtime: 'LiteRT-LM', artifact: ''};
    const file = {name: 'same.litertlm'};
    const input = {files: [file], value: 'C:\\fakepath\\same.litertlm'};
    const event = () => ({target: input}) as unknown as Event;
    const upload = editor.upload(event(), run);
    expect(input.value).toBe('');
    expect(editor.uploading()).toBe('ref');
    editor.cancelUpload();
    expect(observedSignal!.aborted).toBe(true);
    // A transport that resolves even after abort cannot apply its stale result.
    pending.resolve({artifact: 'late.litertlm'});
    await upload;
    expect(run.artifact).toBe('');
    expect(editor.uploading()).toBe(null);
    const progress: (number | null)[] = [];
    artifactsMock['upload'] = (async (_file, options) => {
      options.onProgress(5, 10);
      progress.push(editor.uploadProgress());
      return {artifact: 'same.litertlm'};
    }) as Upload;
    input.value = 'C:\\fakepath\\same.litertlm';
    await editor.upload(event(), run);
    expect(run.artifact).toBe('same.litertlm');
    expect(input.value).toBe('');
    expect(progress).toStrictEqual([0.5]);
    expect(editor.uploadProgress()).toBe(null);
  });

  it("names a side's model on Hugging Face: the server's job fetches it, its progress shown", async () => {
    vi.useFakeTimers();
    const {editor, runtimeMock, artifactsMock, jobsMock} = offlineEditor();
    const source = {
      repo: 'google/gemma-4-E2B-it',
      file: '',
      revision: '3e22461f65e89153144f8adb70e3b8c2cc9845a7',
      commit: '3e22461f65e89153144f8adb70e3b8c2cc9845a7',
    };
    const models: {artifact: string; name: string; runtime: string; source?: typeof source}[] = [];
    runtimeMock['capabilities'] = () => ({available: true, models});
    runtimeMock['refreshCapabilities'] = async () => {
      models.push({artifact: 'model-522738ce', name: source.repo, runtime: 'PyTorch', source});
    };
    const asked: unknown[] = [];
    artifactsMock['fetchHuggingFace'] = async (request: unknown) => {
      asked.push(request);
      return {id: 'download', status: 'downloading', progress: 'Resolving google/gemma-4-E2B-it'};
    };
    jobsMock['job'] = async () => ({
      id: 'download',
      status: 'completed',
      error: '',
      model: {artifact: 'model-522738ce', name: source.repo, runtime: 'PyTorch', source},
    });
    const run: Run = {id: 'ref', runtime: 'PyTorch', artifact: '', source: 'huggingface'};
    editor.config.runs = [run, editor.config.runs[1]];
    const fetching = editor.selectHf(run, {
      link: 'google/gemma-4-E2B-it',
      file: '',
      revision: source.revision,
    });
    await vi.advanceTimersByTimeAsync(0);
    expect(editor.uploading()).toBe('ref');
    expect(editor.uploadLabel()).toBe('Resolving google/gemma-4-E2B-it');
    await vi.advanceTimersByTimeAsync(500);
    await fetching;
    vi.useRealTimers();
    expect(asked).toEqual([
      {runtime: 'PyTorch', link: 'google/gemma-4-E2B-it', file: '', revision: source.revision},
    ]);
    expect([run.artifact, run.source]).toEqual(['model-522738ce', 'huggingface']);
    expect(editor.hfSource(run)?.repo).toBe('google/gemma-4-E2B-it');
    expect(editor.uploading()).toBe(null);
    expect(editor.artifactsReady(), 'a PyTorch model from Hugging Face is a registered one').toBe(
      true,
    );
    // A job that fails says why, in the server's words; the side keeps no model.
    artifactsMock['fetchHuggingFace'] = async () => ({
      id: 'gated',
      status: 'failed',
      error: 'google/gemma-4-E2B-it is gated: accept its terms.',
    });
    const other: Run = {id: 'target', runtime: 'LiteRT-LM', artifact: ''};
    await editor.selectHf(other, {link: 'google/gemma-4-E2B-it', file: 'x.litertlm', revision: ''});
    expect(other.artifact).toBe('');
    expect(editor.error()).toBe('google/gemma-4-E2B-it is gated: accept its terms.');
  });

  it('Cancel download asks the server to stop it; the side keeps no model and no error', async () => {
    vi.useFakeTimers();
    const {editor, artifactsMock, jobsMock} = offlineEditor();
    artifactsMock['fetchHuggingFace'] = async () => ({
      id: 'download',
      status: 'running',
      progress: 'Downloading m.litertlm · 1.0 MB of 4.0 MB · 25%',
    });
    let status = 'running';
    const cancelled: string[] = [];
    jobsMock['cancel'] = async (id: string) => {
      cancelled.push(id);
      status = 'cancelled';
    };
    jobsMock['job'] = async () => ({id: 'download', status, error: 'The download was cancelled.'});
    const run: Run = {id: 'ref', runtime: 'LiteRT-LM', artifact: ''};
    const fetching = editor.selectHf(run, {link: 'org/repo', file: 'm.litertlm', revision: ''});
    await vi.advanceTimersByTimeAsync(0);
    expect(editor.downloading()).toBe(true);
    await editor.cancelDownload();
    await vi.advanceTimersByTimeAsync(500);
    await fetching;
    vi.useRealTimers();
    expect(cancelled).toEqual(['download']);
    expect([run.artifact, editor.uploading(), editor.error()]).toEqual(['', null, '']);
  });

  it('destroying the editor cancels an in-flight Hugging Face download job', async () => {
    vi.useFakeTimers();
    const {editor, artifactsMock, jobsMock} = offlineEditor();
    artifactsMock['fetchHuggingFace'] = async () => ({
      id: 'download-teardown',
      status: 'running',
      progress: 'Downloading…',
    });
    const cancelled: string[] = [];
    jobsMock['cancel'] = async (id: string) => {
      cancelled.push(id);
    };
    jobsMock['job'] = async () => ({id: 'download-teardown', status: 'cancelled', error: ''});
    const run: Run = {id: 'ref', runtime: 'LiteRT-LM', artifact: ''};
    const fetching = editor.selectHf(run, {link: 'org/repo', file: 'm.litertlm', revision: ''});
    await vi.advanceTimersByTimeAsync(0);
    editor.ngOnDestroy();
    await vi.advanceTimersByTimeAsync(500);
    await fetching;
    vi.useRealTimers();
    expect(cancelled).toEqual(['download-teardown']);
  });

  it('upload size limit from runtime capabilities is enforced client-side', async () => {
    const {editor, runtimeMock, artifactsMock} = offlineEditor();
    let uploads = 0;
    artifactsMock['upload'] = async () => {
      uploads++;
      return {artifact: 'never.litertlm'};
    };
    runtimeMock['capabilities'] = () => ({upload_limit_bytes: 10});
    const input = {files: [{name: 'big.litertlm', size: 11}], value: 'C:\\fakepath\\big.litertlm'};
    await editor.upload({target: input} as unknown as Event, {
      id: 'ref',
      runtime: 'LiteRT-LM',
      artifact: '',
    });
    expect(uploads, 'oversize files are rejected before any transfer starts').toBe(0);
    expect(editor.error()).toMatch(/accepts files up to/);
    expect(editor.uploading()).toBe(null);
  });
});
