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

// The API and static checks run against any workspace. The tests that open a
// captured session need E2E_WORKSPACE to name a workspace holding one; the
// default empty workspace skips them.
const captured = !!process.env['E2E_WORKSPACE'];

function watchConsole(page: Page) {
  const problems: string[] = [];
  page.on('console', (message) => {
    if (message.type() === 'error') problems.push(`console.error: ${message.text()}`);
  });
  page.on('pageerror', (error) => problems.push(`pageerror: ${error.message}`));
  return problems;
}

test('server API answers with the contract the UI depends on', async ({request}) => {
  const health = await request.get('/api/health');
  expect(health.status()).toBe(200);
  const sessions = await request.get('/api/sessions');
  expect(sessions.status()).toBe(200);
  const listing = await sessions.json();
  expect(Array.isArray(listing.sessions)).toBe(true);
  expect(listing.capabilities.create).toBe(true);
  const capabilities = await (await request.get('/api/runtime/capabilities')).json();
  expect(typeof capabilities.available).toBe('boolean');
  expect(typeof capabilities.upload_limit_bytes).toBe('number');
  // Errors always carry a JSON body with `error`; an unknown job is 404.
  const missing = await request.get('/api/jobs/does-not-exist');
  expect([400, 404]).toContain(missing.status());
  expect(typeof (await missing.json()).error).toBe('string');
});

test('bundled fonts and third-party notices are served with the build', async ({request}) => {
  const font = await request.get('/fonts/material_icon.woff2');
  expect(font.status()).toBe(200);
  expect((await font.body()).length).toBeGreaterThan(10000);
  const notices = await (await request.get('/3rdpartylicenses.txt')).text();
  for (const expected of ['Plotly.js', 'Material Icons', 'PROVENANCE'])
    expect(notices).toContain(expected);
});

test('home → captured session → Debug views render without console errors', async ({page}) => {
  test.skip(!captured, 'needs E2E_WORKSPACE with a captured session');
  const problems = watchConsole(page);
  await page.goto('/');
  await expect(page).toHaveTitle(/Model Debugger/);
  await expect(page.getByRole('heading', {name: 'Debug Sessions'})).toBeVisible();

  const open = page.getByRole('button', {name: /^Open /}).first();
  await expect(open).toBeVisible();
  await open.click();
  await expect(page.locator('[aria-label="Message composer"]')).toBeVisible();

  // The mode switch is rendered by the toolbar and again by the compact panel header.
  const modes = page.locator('app-toolbar').getByRole('group', {name: 'Workspace mode'});
  await modes.getByRole('button', {name: 'Debug', exact: true}).click();

  const views = page.getByRole('button', {name: 'View', exact: true});
  await views.click();
  await page.locator('.cdk-overlay-container').getByText('Graph Diff', {exact: true}).click();
  await expect(page.getByRole('main', {name: 'Graph analysis'})).toBeVisible();
  await expect(page.getByRole('toolbar', {name: 'Graph analysis controls'})).toBeVisible();
  await expect(page.locator('graph-architecture-view')).toBeAttached();

  await views.click();
  await page.locator('.cdk-overlay-container').getByText('KV Diff', {exact: true}).click();
  await expect(
    page.getByRole('heading', {name: /KV cache not captured|KV observation unavailable/}),
  ).toBeVisible();

  await views.click();
  await page.locator('.cdk-overlay-container').getByText('Graph Diff', {exact: true}).click();
  await expect(page.getByRole('main', {name: 'Graph analysis'})).toBeVisible();

  await modes.getByRole('button', {name: 'Chat', exact: true}).click();
  await expect(page.locator('[aria-label="Message composer"]')).toBeVisible();

  await page.getByLabel('Model Debugger home').click();
  await expect(page.getByRole('heading', {name: 'Debug Sessions'})).toBeVisible();
  expect(problems).toEqual([]);
});

test('anchored dialogs open on their trigger, capture focus, and close on Escape or backdrop', async ({
  page,
}) => {
  test.skip(!captured, 'needs E2E_WORKSPACE with a captured session');
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

test('the theme menu flips data-theme, color-scheme and every light-dark token', async ({page}) => {
  const problems = watchConsole(page);
  await page.goto('/');
  const sample = () =>
    page.evaluate(() => ({
      theme: document.documentElement.dataset['theme'],
      scheme: getComputedStyle(document.documentElement).colorScheme,
      body: getComputedStyle(document.body).backgroundColor,
      primary: getComputedStyle(document.body).getPropertyValue('--me-primary-color').trim(),
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
  await page.goto('/graph_execution/index.html');
  await page.waitForFunction(() => !!customElements.get('model-explorer-visualizer'));
  const csp = await page.getAttribute('meta[http-equiv="Content-Security-Policy"]', 'content');
  expect(csp).toContain("script-src 'self'");
  expect(csp).toContain("default-src 'none'");
  await expect(page.locator('#empty')).toBeVisible();
  expect(await page.locator('#searchButton').count()).toBe(0);
  expect(problems).toEqual([]);
});

test('stage analysis panels reuse the real workspace, collapse independently and preserve empty states', async ({
  page,
}) => {
  test.skip(!captured, 'needs E2E_WORKSPACE with a captured session');
  const problems = watchConsole(page);
  await page.goto('/');
  await page
    .getByRole('button', {name: /^Open /})
    .first()
    .click();
  await page.locator('app-toolbar').getByRole('button', {name: 'Debug', exact: true}).click();
  const reference = page.locator('.debug-turn > article[data-runtime="ref"]').first();
  const target = page.locator('.debug-turn > article[data-runtime="target"]').first();
  await expect(reference).toHaveCSS('display', 'grid');
  await expect(target).toHaveCSS('display', 'grid');
  const beforeNode = await target.locator('.decodeNode').boundingBox();
  const beforeReference = await reference.locator('.decodeNode').boundingBox();
  expect(beforeReference!.y).toBeCloseTo(beforeNode!.y, 0);
  const decode = page.locator('.stage-analysis-row[data-stage="decode"]').first();
  await decode.getByRole('button', {name: 'Add analysis view', exact: true}).click();
  const picker = page.getByRole('dialog');
  await expect(picker.getByRole('button', {name: 'Analysis type', exact: true})).toContainText(
    'Select analysis type',
  );
  await expect(picker.getByRole('button', {name: 'Add', exact: true})).toBeDisabled();
  await expect(picker.locator('svg')).toHaveCount(0);
  await picker.getByRole('button', {name: 'Analysis type', exact: true}).click();
  await expect(page.getByRole('listbox', {name: 'analysis types'}).getByRole('option')).toHaveCount(
    5,
  );
  await expect(page.getByRole('option', {name: /Final Logits/})).toBeDisabled();
  await expect(page.getByRole('option', {name: /Sampling Distribution/})).toBeDisabled();
  await expect(page.getByRole('option', {name: /KV cache/})).toBeDisabled();
  await expect(page.getByRole('option', {name: /KV cache/})).toContainText('No capture');
  await expect(decode.locator('.stage-analysis')).toHaveCount(0);
  await page.keyboard.press('Escape');
  await picker.getByRole('button', {name: 'Cancel', exact: true}).click();
  await page
    .locator('.stage-analysis-row[data-stage="prefill"]')
    .first()
    .getByRole('button', {name: 'Add analysis view', exact: true})
    .click();
  await picker.getByRole('button', {name: 'Analysis type', exact: true}).click();
  await expect(page.getByRole('option', {name: /Final Logits/})).toContainText('Not supported');
  await expect(page.getByRole('option', {name: /Final Logits/})).toBeDisabled();
  await page.keyboard.press('Escape');
  await picker.getByRole('button', {name: 'Cancel', exact: true}).click();
  // Explicit fixture for panel lifecycle; the saved example above has no paired tensors.
  await page.route('**/comparisons?*', (route) =>
    route.fulfill({
      json: {
        batch: 1,
        layers: [],
        rows: [
          {
            layer: 0,
            anchor: 'output',
            shape: [1, 8],
            reference: 'fixture-ref',
            target: 'fixture-target',
            status: 'ok',
            metrics: {CosSim: {value: 0.98, status: 'ok'}},
          },
        ],
      },
    }),
  );
  // Layer-only rows must never unlock a positional analysis or a Summary fallback.
  await decode.getByRole('button', {name: 'Add analysis view', exact: true}).click();
  await picker.getByRole('button', {name: 'Analysis type', exact: true}).click();
  await expect(page.getByRole('option', {name: /Activations/})).toBeDisabled();
  await page.keyboard.press('Escape');
  await picker.getByRole('button', {name: 'Cancel', exact: true}).click();
  // Explicit positional KV fixture exercises the shared controls and panel lifecycle.
  await page.route('**/kv-analysis?*', (route) =>
    route.fulfill({
      json: {
        contexts: [
          {
            id: 'fixture',
            turn: 1,
            phase: 'decode',
            moment: 'terminal',
            source: 'snapshot',
            runtime: null,
            step: 0,
            forward_id: 0,
            label: 'Fixture',
            layers: ['key', 'value'].map((kind) => ({
              kind,
              layer: 0,
              status: 'ok',
              head_count: 2,
              position_start: 0,
              position_count: 2000,
              cells: [],
            })),
          },
        ],
      },
    }),
  );
  await page.route('**/kv-selection?*', (route) =>
    route.fulfill({
      json: {
        context_id: 'fixture',
        start: Number(new URL(route.request().url()).searchParams.get('start')),
        end: Number(new URL(route.request().url()).searchParams.get('end')),
        rows: [],
      },
    }),
  );
  await page.route('**/kv-range?*', (route) => {
    const query = new URL(route.request().url()).searchParams;
    const kind = query.get('kind'),
      start = Number(query.get('start')),
      end = Number(query.get('end'));
    return route.fulfill({
      json: {
        context_id: 'fixture',
        start,
        end,
        bins: [{start, end}],
        rows: [
          {
            layer: 0,
            kind,
            status: 'ok',
            bins: [
              {
                start,
                end,
                metrics: {
                  cosine_similarity: {min: 0.98, max: 0.99, mean: 0.985, valid_count: 2},
                  relative_l2: {min: 0.01, max: 0.02, mean: 0.015, valid_count: 2},
                },
              },
            ],
          },
        ],
      },
    });
  });
  for (let i = 0; i < 2; i++) {
    await decode.getByRole('button', {name: 'Add analysis view', exact: true}).click();
    await picker.getByRole('button', {name: 'Analysis type', exact: true}).click();
    await page.getByRole('option', {name: /KV cache/}).click();
    await picker.getByRole('button', {name: 'Add', exact: true}).click();
  }
  await expect(page.locator('.token-inspector')).toHaveCount(1);
  await expect(decode.locator('.stage-analysis')).toHaveCount(2);
  const controls = decode.locator('analysis-heatmap-controls').first();
  await expect(controls.getByRole('button', {name: 'Analysis metric', exact: true})).toContainText(
    'CosSim',
  );
  await expect(controls.getByRole('button', {name: 'Heads', exact: true})).toContainText('Min');
  await expect(controls.getByRole('button', {name: 'Position bins', exact: true})).toContainText(
    'Min',
  );
  await expect(controls.getByRole('button', {name: 'Position bins', exact: true})).toBeEnabled();
  await controls.getByRole('button', {name: 'Heads', exact: true}).click();
  await page.getByRole('option', {name: 'Head 1', exact: true}).click();
  await expect(controls.getByRole('button', {name: 'Heads', exact: true})).toContainText('Head 1');
  await controls.getByRole('button', {name: 'Analysis metric', exact: true}).click();
  await page.getByRole('option', {name: 'Relative L2', exact: true}).click();
  await expect(controls.getByRole('button', {name: 'Heads', exact: true})).toContainText('Head 1');
  await expect(controls.getByRole('button', {name: 'Position bins', exact: true})).toContainText(
    'Max',
  );
  await controls.getByRole('button', {name: 'Position bins', exact: true}).click();
  await page.getByRole('option', {name: 'Mean', exact: true}).click();
  await expect(controls.getByRole('button', {name: 'Position bins', exact: true})).toContainText(
    'Mean',
  );
  await controls.getByRole('button', {name: 'Analysis metric', exact: true}).click();
  await page.getByRole('option', {name: 'CosSim', exact: true}).click();
  await expect(controls.getByRole('button', {name: 'Position bins', exact: true})).toContainText(
    'Mean',
  );

  const firstPanel = decode.locator('kv-panel').first();
  const plot = firstPanel.locator('kv-range-chart .plot');
  await expect(plot.locator('.rangeslider-container')).toBeVisible();
  const colorLegend = firstPanel.locator('[aria-label="Metric color scale"]');
  await expect(colorLegend).not.toContainText('snapshot range');
  await expect(colorLegend).not.toContainText('Positions:');
  await expect(colorLegend.locator('.color-endpoint')).toHaveCount(2);
  await controls.getByRole('button', {name: 'Analysis metric', exact: true}).click();
  await page.getByRole('option', {name: 'Relative L2', exact: true}).click();
  for (const view of ['Line chart', 'Heatmap', 'Line chart']) {
    await firstPanel.getByRole('button', {name: view, exact: true}).click();
    if (view === 'Line chart')
      await expect(
        firstPanel.getByRole('button', {name: 'Analysis metric', exact: true}),
      ).toContainText('Relative L2');
    await expect(plot.locator('.rangeslider-container')).toBeVisible();
    if (view === 'Line chart')
      await expect(plot.locator('.rangeslider-container .heatmaplayer image')).toHaveCount(0);
    else
      await expect(
        plot.locator('.rangeslider-container .heatmaplayer image').first(),
      ).toBeVisible();
    const before = await plot.evaluate((node: any) => [...node._fullLayout.xaxis.range]);
    const box = await plot.boundingBox();
    await page.mouse.move(box!.x + 150, box!.y + 70);
    await page.mouse.down();
    await page.mouse.move(box!.x + 300, box!.y + 150, {steps: 8});
    await page.mouse.up();
    await page.mouse.dblclick(box!.x + 200, box!.y + 100);
    expect(await plot.evaluate((node: any) => node._fullLayout.xaxis.range)).toEqual(before);
  }
  // Range changes belong to the native bottom slider, even with fixed plot axes.
  await plot.locator('.rangeslider-grabber-max').scrollIntoViewIfNeeded();
  const slider = await plot.locator('.rangeslider-grabber-max').boundingBox();
  await page.mouse.move(slider!.x + slider!.width / 2, slider!.y + slider!.height / 2);
  await page.mouse.down();
  await page.mouse.move(slider!.x - 100, slider!.y + slider!.height / 2, {steps: 8});
  await page.mouse.up();
  await expect
    .poll(() => plot.evaluate((node: any) => node._fullLayout.xaxis.range[1]))
    .toBeLessThan(1999.5);
  await firstPanel.getByRole('button', {name: 'Heatmap', exact: true}).click();

  // Native hover marks the corresponding cell in both K/V planes, with a concise tooltip.
  await plot.scrollIntoViewIfNeeded();
  const cellPoint = await plot.evaluate((node: any) => {
    const bounds = node.getBoundingClientRect(),
      layout = node._fullLayout;
    return {
      x: bounds.left + layout.xaxis._offset + layout.xaxis._length / 2,
      y: bounds.top + layout.yaxis._offset + layout.yaxis.d2p(0),
    };
  });
  await page.mouse.move(cellPoint.x, cellPoint.y);
  await expect(firstPanel.locator('.cell-hover')).toHaveCount(2);
  await expect(plot.locator('..').locator('.chart-tooltip')).toContainText('Layer 0');
  await expect(plot.locator('..').locator('.chart-tooltip')).not.toContainText('Heads');
  await expect(plot.locator('..').locator('.chart-tooltip')).not.toContainText('relative_l2');
  const columnAxis = firstPanel.getByRole('button', {
    name: 'Select a Position column',
    exact: true,
  });
  await columnAxis.hover();
  await expect(firstPanel.locator('.column-preview')).toHaveCount(2);
  await expect(firstPanel.locator('.column-preview').first()).toBeVisible();
  await expect(firstPanel.getByRole('tooltip')).toContainText('All layers · Position');
  await columnAxis.click();
  await expect(page.getByText('Column Info', {exact: true})).toBeVisible();
  await expect(page.getByRole('table', {name: 'Selected KV column', exact: true})).toContainText(
    'Global position',
  );
  await expect(firstPanel.locator('.axis-selection')).toBeVisible();
  await expect(firstPanel.locator('.axis-bin').first()).toBeVisible();
  await expect(firstPanel.locator('.cell-selected')).toHaveCount(2);
  await page.mouse.move(5, 5);
  await expect(firstPanel.locator('.cell-hover')).toHaveCount(0);
  await expect(firstPanel.locator('.axis-selection')).toBeVisible();
  await expect(plot.locator('.heatmaplayer image').first()).toBeVisible();
  await columnAxis.press('ArrowRight');
  await expect(firstPanel.locator('kv-range-chart [role="status"]')).toContainText(
    'all layers selected',
  );

  const kvEntry = decode.locator('.analysis-entry').first();
  await kvEntry.click();
  await expect(kvEntry).toHaveAttribute('aria-expanded', 'false');
  await expect(decode.locator('kv-panel').last()).toBeVisible();
  await kvEntry.click();
  await expect(decode.locator('kv-panel').first()).toBeVisible();
  const band = await decode.locator('.analysis-content').first().boundingBox();
  const lanes = await page.locator('.column-headings').boundingBox();
  expect(band!.x).toBeCloseTo(lanes!.x, 0);
  expect(band!.width).toBeCloseTo(lanes!.width, 0);
  await expect(kvEntry).toHaveCSS('width', '160px');
  const afterNode = await target.locator('.decodeNode').boundingBox();
  expect(afterNode!.x).toBeCloseTo(beforeNode!.x, 0);
  expect(afterNode!.width).toBeCloseTo(beforeNode!.width, 0);
  expect((await target.boundingBox())!.width).toBeGreaterThan(0);

  await expect(page.getByRole('combobox', {name: 'Find scope', exact: true})).toHaveCount(0);
  await expect(page.locator('.debug-primary token-query-control')).toBeVisible();
  await expect(page.getByRole('button', {name: 'Find condition', exact: true})).toHaveCount(1);
  await expect(decode.getByRole('button', {name: 'Find by metric', exact: true})).toHaveCount(0);
  await expect(
    decode.getByRole('button', {name: 'Choose KV observation', exact: true}),
  ).toHaveCount(0);
  await page.getByRole('button', {name: 'Find condition', exact: true}).click();
  await expect(page.getByRole('dialog', {name: 'Find condition', exact: true})).toBeVisible();
  await page.getByRole('button', {name: 'Close find condition', exact: true}).click();
  await decode
    .getByRole('button', {name: 'Remove KV Cache Analysis 0', exact: true})
    .first()
    .click();
  await expect(decode.locator('.stage-analysis')).toHaveCount(1);
  await expect(page.getByRole('button', {name: 'Find condition', exact: true})).toBeVisible();
  expect(problems).toEqual([]);
});
