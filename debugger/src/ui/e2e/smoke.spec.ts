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

import {expect, test, type Page} from '@playwright/test';

// The bundled Gemma example has architecture/reference metadata but no KV evidence,
// so the smoke asserts the real navigation path and the documented empty states.

function watchConsole(page: Page) {
  const problems: string[] = [];
  page.on('console', (message) => {
    if (message.type() === 'error')
      problems.push(`console.error: ${message.text()}`);
  });
  page.on('pageerror', (error) => problems.push(`pageerror: ${error.message}`));
  return problems;
}

test('server API answers with the contract the UI depends on', async ({
  request,
}) => {
  const health = await request.get('/api/health');
  expect(health.status()).toBe(200);
  const sessions = await request.get('/api/sessions');
  expect(sessions.status()).toBe(200);
  const listing = await sessions.json();
  expect(listing.sessions.length).toBeGreaterThan(0);
  expect(listing.sessions[0].has_capture).toBe(true);
  const capabilities = await (
    await request.get('/api/runtime/capabilities')
  ).json();
  expect(capabilities.available).toBe(false);
  expect(typeof capabilities.upload_limit_bytes).toBe('number');
  // Errors always carry a JSON body with `error`; saved-capture mode has no job routes (404).
  const missing = await request.get('/api/jobs/does-not-exist');
  expect([400, 404]).toContain(missing.status());
  expect(typeof (await missing.json()).error).toBe('string');
});

test('bundled fonts and third-party notices are served with the build', async ({
  request,
}) => {
  const font = await request.get('/fonts/material_icon.woff2');
  expect(font.status()).toBe(200);
  expect((await font.body()).length).toBeGreaterThan(10000);
  const notices = await (await request.get('/3rdpartylicenses.txt')).text();
  for (const expected of ['Plotly.js', 'Material Icons', 'PROVENANCE'])
    expect(notices).toContain(expected);
});

test('home → captured session → Debug views render without console errors', async ({
  page,
}) => {
  const problems = watchConsole(page);
  await page.goto('/');
  await expect(page).toHaveTitle(/Model Debugger/);
  await expect(
    page.getByRole('heading', {name: 'Debug Sessions'}),
  ).toBeVisible();

  const open = page.getByRole('button', {name: /^Open /}).first();
  await expect(open).toBeVisible();
  await open.click();
  await expect(page.locator('[aria-label="Message composer"]')).toBeVisible();

  // The mode switch is rendered by the toolbar and again by the compact panel header.
  const modes = page
    .locator('app-toolbar')
    .getByRole('group', {name: 'Workspace mode'});
  await modes.getByRole('button', {name: 'Debug', exact: true}).click();

  const views = page.getByRole('button', {name: 'View', exact: true});
  await views.click();
  await page
    .locator('.cdk-overlay-container')
    .getByText('Graph Diff', {exact: true})
    .click();
  await expect(page.getByRole('main', {name: 'Graph analysis'})).toBeVisible();
  await expect(
    page.getByRole('toolbar', {name: 'Graph analysis controls'}),
  ).toBeVisible();
  await expect(page.locator('graph-architecture-view')).toBeAttached();

  await views.click();
  await page
    .locator('.cdk-overlay-container')
    .getByText('KV Diff', {exact: true})
    .click();
  await expect(
    page.getByRole('heading', {
      name: /KV cache not captured|KV observation unavailable/,
    }),
  ).toBeVisible();

  await views.click();
  await page
    .locator('.cdk-overlay-container')
    .getByText('Graph Diff', {exact: true})
    .click();
  await expect(page.getByRole('main', {name: 'Graph analysis'})).toBeVisible();

  await modes.getByRole('button', {name: 'Chat', exact: true}).click();
  await expect(page.locator('[aria-label="Message composer"]')).toBeVisible();

  await page.getByLabel('Model Debugger home').click();
  await expect(
    page.getByRole('heading', {name: 'Debug Sessions'}),
  ).toBeVisible();
  expect(problems).toEqual([]);
});

test('anchored dialogs open on their trigger, capture focus, and close on Escape or backdrop', async ({
  page,
}) => {
  const problems = watchConsole(page);
  await page.goto('/');
  await page
    .getByRole('button', {name: /^Open /})
    .first()
    .click();
  await page
    .locator('app-toolbar')
    .getByRole('group', {name: 'Workspace mode'})
    .getByRole('button', {name: 'Debug', exact: true})
    .click();

  await page.getByRole('button', {name: 'Find condition'}).click();
  const find = page.getByRole('dialog', {name: 'Find condition'});
  await expect(find).toBeVisible();
  await expect(find.getByLabel('Find formula')).toBeFocused();
  await page.keyboard.press('Escape');
  await expect(find).toBeHidden();
  // The CDK keeps a dismissed backdrop in the DOM until its fade-out ends.
  await expect(page.locator('.cdk-overlay-backdrop')).toHaveCount(0);

  await page.getByRole('button', {name: 'Bookmarks'}).click();
  const bookmarks = page.getByRole('dialog', {name: 'Chat bookmarks'});
  await expect(bookmarks).toBeVisible();
  await page
    .locator('.cdk-overlay-backdrop')
    .last()
    .click({position: {x: 5, y: 5}});
  await expect(bookmarks).toBeHidden();
  // The trigger is enabled again and reopens the same dialog.
  await page.getByRole('button', {name: 'Bookmarks'}).click();
  await expect(bookmarks).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(bookmarks).toBeHidden();
  expect(problems).toEqual([]);
});

test('the theme menu flips data-theme, color-scheme and every light-dark token', async ({
  page,
}) => {
  const problems = watchConsole(page);
  await page.goto('/');
  const sample = () =>
    page.evaluate(() => ({
      theme: document.documentElement.dataset['theme'],
      scheme: getComputedStyle(document.documentElement).colorScheme,
      body: getComputedStyle(document.body).backgroundColor,
      primary: getComputedStyle(document.body)
        .getPropertyValue('--me-primary-color')
        .trim(),
    }));
  await page.getByRole('button', {name: 'Theme'}).click();
  await page.getByRole('menuitemradio', {name: /^Dark/}).click();
  const dark = await sample();
  expect(dark.theme).toBe('dark');
  expect(dark.scheme).toBe('dark');
  await page.getByRole('button', {name: 'Theme'}).click();
  await page.getByRole('menuitemradio', {name: /^Light/}).click();
  const light = await sample();
  expect(light.theme).toBe('light');
  expect(light.scheme).toBe('light');
  expect(light.body).not.toBe(dark.body);
  expect(light.primary).toContain('light-dark(');
  expect(problems).toEqual([]);
});

test('the execution-graph frame loads its pinned renderer under its Content-Security-Policy', async ({
  page,
}) => {
  const problems = watchConsole(page);
  await page.goto('/graph-execution/index.html');
  await page.waitForFunction(
    () => !!customElements.get('model-explorer-visualizer'),
  );
  const csp = await page.getAttribute(
    'meta[http-equiv="Content-Security-Policy"]',
    'content',
  );
  expect(csp).toContain("script-src 'self'");
  expect(csp).toContain("default-src 'none'");
  await expect(page.locator('#empty')).toBeVisible();
  expect(await page.locator('#searchButton').count()).toBe(0);
  expect(problems).toEqual([]);
});
