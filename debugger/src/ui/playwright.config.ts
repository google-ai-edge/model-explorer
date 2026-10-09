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

import {defineConfig, devices} from '@playwright/test';

// End-to-end smoke against the production build served by the Python Server in
// saved-capture mode (examples/gemma4-e2b). Run `npm run build` first; the server
// interpreter defaults to the repository venv and can be overridden with SERVER_PYTHON.
const port = 8897;
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
    ...(process.env['PLAYWRIGHT_CHANNEL']
      ? {channel: process.env['PLAYWRIGHT_CHANNEL']}
      : {}),
  },
  projects: [{name: 'chromium', use: {...devices['Desktop Chrome']}}],
  webServer: {
    command: `${python} -m model_explorer_debugger.server --data ../../examples/gemma4-e2b --port ${port} --ui dist/model_explorer_debugger/browser`,
    url: `http://127.0.0.1:${port}/api/health`,
    reuseExistingServer: false,
    timeout: 60_000,
    env: {PYTHONDONTWRITEBYTECODE: '1'},
  },
});
