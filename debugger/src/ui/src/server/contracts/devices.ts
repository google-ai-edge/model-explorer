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

/** GET /api/devices, POST /api/devices/inspect and GET /api/devices/builds
 *  (device_catalog.py). */

import type {RuntimeCapability} from './runtime';
import type {Permission} from './sessions';

export interface DeviceCandidates {
  devices: RunnerDevice[];
  notice?: string;
}
export interface RunnerDevice {
  id: string;
  name: string;
  runnerId?: string;
  platform?: string;
  transport?: string;
  /** The device's state in words: 'Connected', 'Connected · 2 runners running' or 'Busy'. */
  statusText: string;
  /** Whether a Session may run on it now; a refusal is the one a start on it would get. */
  choosable: Permission;
  /** Who holds it, e.g. 'In use by CPU vs CPU · Reference'; '' when free. */
  holder: string;
  /** What a run on it may set: the runtime's options with the device's own. On a device that
   *  runs the version a Session chose, the server's table: each build's is its own. */
  capability?: RuntimeCapability;
  checkedAt?: string;
  reason?: string;
  /** 'not_running' on a device whose each start runs the runner version a Session chose (this
   *  computer): the editor offers its builds. A phone, which runs the App installed, sends none. */
  runnerState?: 'not_running' | 'idle' | 'active' | 'unknown';
  /** A phone's Runner App, as its version says, read without launching it; this computer, which
   *  runs the version a Session chose, sends none. */
  app?: RunnerApp;
  runner?: {
    /** The build running there; the server sends none yet. */
    buildId?: string;
    build?: RunnerBuild;
  };
}
/** The Runner App installed on a phone: one version per phone. */
export interface RunnerApp {
  installed: boolean;
  /** The LiteRT-LM commit it was built against: whole, or its first 12 hex digits on an iPhone,
   *  whose App's version holds no more; '' when its version names none. */
  version: string;
  /** That commit's date, YYYY-MM-DD, when its version says. */
  commitDate?: string;
  /** E.g. 'LiteRT-LM 61dcfc772250 · 2026-10-15', or 'No Runner App installed'. */
  label: string;
  /** Why it may not be the App wanted: none installed, no commit named, or another commit than
   *  the server is written for. */
  notice?: string;
}
/** A runner version a side can run: a kept build, the development build, or the PyTorch runner. */
export interface RunnerBuild {
  /** The LiteRT-LM commit of a kept build; 'dev' for the development build; 'pytorch' for the
   *  PyTorch runner. */
  id: string;
  /** The runtime it runs: 'LiteRT-LM' when absent. */
  runtime?: string;
  /** Unique among a device's builds, e.g. 'LiteRT-LM 61dcfc772250 · 2026-10-15'. */
  label: string;
  /** The whole LiteRT-LM commit; '' when known only once it runs. */
  version?: string;
  cl?: string;
  platform?: string;
  /** Whether it can be launched, as far as its files tell; `reason` says why not. */
  available?: boolean;
  /** Whether a side of its runtime runs it unless it chose another: the newest kept LiteRT-LM
   *  build available, else the development build; the PyTorch runner for PyTorch. */
  default?: boolean;
  reason?: string;
  /** What a run of it may set: the runtime's options with what its last runner here reported. */
  capability?: RuntimeCapability;
}
/** GET /api/devices/builds?id=: the runner versions of a device, read without launching any. */
export interface RunnerBuildCatalog {
  deviceId: string;
  builds: RunnerBuild[];
  canLaunch: boolean;
  reason?: string;
  /** A phone's Runner App: a phone offers no builds, it runs the App installed. */
  app?: RunnerApp;
}
