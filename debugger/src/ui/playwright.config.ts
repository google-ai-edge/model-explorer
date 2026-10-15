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

import {mkdtempSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {defineConfig, devices} from '@playwright/test';

// End-to-end tests against the production build served by the Python Server.
// Run `npm run build` first. The server keeps its records in E2E_WORKSPACE, a
// new empty directory by default; specs that open a captured session skip
// unless E2E_WORKSPACE names a workspace holding one. The server interpreter
// defaults to the repository venv and can be overridden with SERVER_PYTHON.
const port = Number(process.env['E2E_PORT'] ?? 8897);
const workspace =
  process.env['E2E_WORKSPACE'] ?? mkdtempSync(join(tmpdir(), 'model-debugger-e2e-'));
if (!/^[a-zA-Z0-9_./-]+$/.test(workspace)) throw new Error('Invalid E2E_WORKSPACE path');
const python = process.env['SERVER_PYTHON'] ?? '../server/.venv/bin/python';

export default defineConfig({
  testDir: './e2e',
  timeout: 60_000,
  expect: {timeout: 15_000},
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [['list']],
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    trace: 'retain-on-failure',
    ...(process.env['PLAYWRIGHT_CHANNEL'] ? {channel: process.env['PLAYWRIGHT_CHANNEL']} : {}),
  },
  projects: [{name: 'chromium', use: {...devices['Desktop Chrome']}}],
  webServer: {
    command: `${python} -m model_debugger_server --workspace ${workspace} --port ${port} --ui dist/model_debugger/browser`,
    url: `http://127.0.0.1:${port}/api/health`,
    reuseExistingServer: false,
    timeout: 60_000,
    env: {PYTHONDONTWRITEBYTECODE: '1'},
  },
});
