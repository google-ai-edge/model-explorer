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

import {expect, test, type Locator, type Page} from '@playwright/test';

test.skip(
  !process.env['E2E_WORKSPACE']?.includes('analysis-review') ||
    process.env['E2E_WORKSPACE']?.includes('analysis-review-dense'),
  'Requires the small analysis-review capture',
);

async function openReview(page: Page) {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.setViewportSize({width: 1600, height: 1100});
  await page.emulateMedia({reducedMotion: 'reduce'});
  await page.goto('/');
  await page
    .getByRole('button', {name: 'Open Analysis review · Gemma 4 · REAL + TEST DATA', exact: true})
    .click();
  await page.locator('app-toolbar').getByRole('button', {name: 'Debug', exact: true}).click();
  const stage = page.locator('.stage-analysis-row[data-stage="prefill"]').nth(1);
  const tokens = page.getByRole('group', {name: 'Turn 2 / Prefill · Target', exact: true});
  await tokens.press('Enter');
  const details = page.getByRole('complementary', {name: 'Comparison details'});
  return {stage, tokens, details, errors};
}

async function addKv(page: Page, stage: Locator) {
  await stage.getByRole('button', {name: 'Add analysis view', exact: true}).click();
  await page.getByRole('button', {name: 'Analysis type', exact: true}).click();
  await page.getByRole('option', {name: 'KV cache Available', exact: true}).click();
  await page.getByRole('button', {name: 'Add', exact: true}).click();
  await expect(page.getByRole('dialog', {name: 'Add analysis view', exact: true})).toHaveCount(0);
  const panel = stage.locator('kv-panel').last();
  await ready(panel);
  await expect(panel.locator('.heatmaplayer:not(.rangeplot) image')).toHaveCount(2);
  return panel;
}

async function ready(panel: Locator) {
  await expect(panel.locator('.range-canvas')).toHaveAttribute('aria-busy', 'false');
  await expect(panel.locator('.js-plotly-plot')).toBeVisible();
}

async function clickPlot(panel: Locator, target: 'cell' | 'between-tensors' | 'empty-trends') {
  await ready(panel);
  const plot = panel.locator('.js-plotly-plot');
  const point = await plot.evaluate((node: any, target) => {
    const layout = node._fullLayout;
    const x = layout.xaxis;
    const y = layout.yaxis;
    if (target === 'cell') {
      // The first visible global Position and first actual Layer, independent of capture size.
      const position = Math.ceil(x.range[0] + 0.5);
      return {x: x._offset + x.d2p(position), y: y._offset + y.d2p(0)};
    }
    if (target === 'between-tensors') {
      const bottom = layout.yaxis2;
      if (!bottom) throw new Error('Expected separate Key and Value plot areas');
      return {
        x: x._offset + x._length / 2,
        y: (y._offset + y._length + bottom._offset) / 2,
      };
    }
    return {x: x._offset + x._length / 2, y: y._offset + y._length / 2};
  }, target);
  await plot.click({position: point});
}

async function clickPadding(panel: Locator) {
  // The content wrapper owns visible padding in normal and expanded layouts.
  await panel.locator('..').click({position: {x: 4, y: 4}});
}

async function expectPoint(details: Locator) {
  await expect(details.locator('.statistic-group-header .selection').first()).toHaveCount(1);
}

async function expectOverview(details: Locator, label: string) {
  const overview = details.locator('stage-state-info');
  await expect(overview).toBeVisible();
  await expect(overview.locator('.statistic-group-header .selection').first()).toHaveCount(0);
  await expect(overview.getByRole('region', {name: 'Analysis Info', exact: true})).toContainText(
    label,
  );
  await expect(details.getByRole('region', {name: 'Stage Info', exact: true})).toHaveCount(0);
  await expect(details.getByRole('region', {name: 'Conversation Info', exact: true})).toHaveCount(
    0,
  );
}

async function settings(panel: Locator) {
  return {
    metric: await panel.getByRole('button', {name: 'Analysis metric', exact: true}).innerText(),
    heads: await panel.getByRole('button', {name: 'Heads aggregation', exact: true}).innerText(),
    start: await panel.getByRole('spinbutton', {name: 'Token Range start', exact: true}).inputValue(),
    end: await panel.getByRole('spinbutton', {name: 'Token Range end', exact: true}).inputValue(),
    view: await panel.getByRole('button', {name: 'Visualization', exact: true}).innerText(),
  };
}

test('KV background opens the visible range overview while controls and slider preserve point inspection', async ({
  page,
}) => {
  test.setTimeout(120000);
  const {stage, details, errors} = await openReview(page);
  const panel = await addKv(page, stage);
  await clickPlot(panel, 'cell');
  await expectPoint(details);
  await expect(panel.locator('.cell-selected')).toHaveCount(2);

  // Metric and Head changes must not be mistaken for panel-background clicks.
  await panel.getByRole('button', {name: 'Analysis metric', exact: true}).click();
  await page.getByRole('option', {name: 'Relative L2', exact: true}).click();
  await expectPoint(details);
  await panel.getByRole('button', {name: 'Heads aggregation', exact: true}).click();
  await page.getByRole('button', {name: 'Aggregation function', exact: true}).click();
  await page.getByRole('option', {name: 'Mean', exact: true}).click();
  await page.getByRole('button', {name: 'Apply', exact: true}).click();
  await ready(panel);
  await expectPoint(details);

  const beforeDrag = Number(
    await panel.getByRole('spinbutton', {name: 'Token Range end', exact: true}).inputValue(),
  );
  const handle = panel.locator('.rangeslider-grabber-max');
  await handle.scrollIntoViewIfNeeded();
  const box = (await handle.boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x - 100, box.y + box.height / 2, {steps: 8});
  await page.mouse.up();
  await expect
    .poll(async () => Number(await panel.getByRole('spinbutton', {name: 'Token Range end'}).inputValue()))
    .toBeLessThan(beforeDrag);
  await ready(panel);
  await expectPoint(details);

  const beforeBackground = await settings(panel);
  await clickPadding(panel);
  await expectOverview(details, 'KV cache');
  await expect(panel.locator('.cell-selected, .axis-selection')).toHaveCount(0);
  expect(await settings(panel)).toEqual(beforeBackground);

  await clickPlot(panel, 'cell');
  await expectPoint(details);
  await clickPlot(panel, 'between-tensors');
  await expectOverview(details, 'KV cache');
  await expect(panel.locator('.cell-selected, .axis-selection')).toHaveCount(0);
  expect(await settings(panel)).toEqual(beforeBackground);

  await panel.getByRole('button', {name: 'Visualization', exact: true}).click();
  await page.getByRole('option', {name: 'Tensor Comparison · Line chart', exact: true}).click();
  await panel.getByRole('button', {name: 'Layers', exact: true}).click();
  await page.getByRole('checkbox', {name: 'Layer 0', exact: true}).check();
  await page.getByRole('checkbox', {name: 'Layer 1', exact: true}).check();
  await page.getByRole('button', {name: 'Done', exact: true}).click();
  await panel.getByRole('button', {name: 'Select a Position column', exact: true}).click();
  await expectPoint(details);
  await expect(
    details.getByRole('button', {name: 'Collapse Statistic', exact: true}),
  ).toBeVisible();
  for (const name of ['Key cache', 'Value cache']) {
    const table = details.getByRole('table', {name: name + ' layers', exact: true});
    await expect(table.locator('.sort-label')).toHaveText(['Layer', 'Mean', 'Min', 'Max']);
    await expect.poll(() => table.locator('.statistic-row').count()).toBeGreaterThan(1);
    const values = await table
      .locator('.statistic-row')
      .first()
      .locator('.stat-value .value')
      .allTextContents();
    expect(new Set(values).size).toBe(1);
  }
  await expect(details.getByRole('region', {name: 'Tensor data', exact: true})).toHaveCount(0);
  await panel.getByRole('button', {name: 'Layers', exact: true}).click();
  await page.getByRole('button', {name: 'Clear', exact: true}).click();
  await page.getByRole('button', {name: 'Done', exact: true}).click();
  await expect(panel.locator('.scatterlayer:not(.rangeplot) .point')).toHaveCount(0);
  await expectOverview(details, 'KV cache');
  const emptyPlot = panel.locator('.js-plotly-plot');
  await emptyPlot.scrollIntoViewIfNeeded();
  const drag = await emptyPlot.evaluate((node: any) => {
    const rect = node.getBoundingClientRect();
    const layout = node._fullLayout;
    return {
      x: rect.x + layout.xaxis._offset + layout.xaxis._length / 2,
      y: rect.y + layout.yaxis._offset + layout.yaxis._length / 2,
      outsideX: Math.max(1, rect.x - 8),
    };
  });
  // Plotly places a document-wide cover over an active gesture. Returning to
  // the initial coordinate must still count as a drag, not a background click.
  await page.mouse.move(drag.x, drag.y);
  await page.mouse.down();
  await page.mouse.move(drag.outsideX, drag.y, {steps: 8});
  await page.mouse.move(drag.x, drag.y, {steps: 8});
  await page.mouse.up();
  await page.evaluate(
    () =>
      new Promise<void>((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
      ),
  );
  await expectOverview(details, 'KV cache');
  const beforeEmptyPlot = await settings(panel);
  const layers = await panel.getByRole('button', {name: 'Layers', exact: true}).innerText();
  await clickPlot(panel, 'empty-trends');
  await expectOverview(details, 'KV cache');
  await expect(panel.locator('.cell-selected, .axis-selection')).toHaveCount(0);
  expect(await settings(panel)).toEqual(beforeEmptyPlot);
  expect(await panel.getByRole('button', {name: 'Layers', exact: true}).innerText()).toBe(layers);
  expect(errors).toEqual([]);
});

test('KV background selects the exact duplicate instance and clears linked token selection', async ({
  page,
}) => {
  test.setTimeout(120000);
  const {stage, tokens, details, errors} = await openReview(page);
  await addKv(page, stage);
  await addKv(page, stage);
  const first = stage.locator('kv-panel').nth(0);
  const second = stage.locator('kv-panel').nth(1);
  const entries = stage.locator('.analysis-entry span');
  const firstLabel = (await entries.nth(0).innerText()).trim();
  const secondLabel = (await entries.nth(1).innerText()).trim();
  expect(firstLabel).not.toEqual(secondLabel);
  const ids = await stage
    .locator('.analysis-content')
    .evaluateAll((nodes) => nodes.map((n) => n.id));

  await clickPlot(second, 'cell');
  await expectPoint(details);
  await clickPadding(first);
  await expectOverview(details, firstLabel);
  await expect(stage.locator('.cell-selected, .axis-selection')).toHaveCount(0);
  await clickPadding(second);
  await expectOverview(details, secondLabel);
  await details.getByRole('button', {name: 'Locate Value cache Heatmap', exact: true}).click();
  await expect(second.locator('.cache-location')).toHaveAttribute('data-kind', 'value');
  await expect(first.locator('.cache-location')).toHaveCount(0);
  await expectOverview(details, secondLabel);

  const token = tokens.locator('.input-token[data-side="target"]').first();
  await token.click();
  await expect(details.getByRole('region', {name: 'Token Info', exact: true})).toBeVisible();
  await expect(token).toHaveAttribute('aria-pressed', 'true');
  await expect(first.locator('.cell-selected')).toHaveCount(2);
  await expect(second.locator('.cell-selected')).toHaveCount(2);
  await clickPadding(first);
  await expectOverview(details, firstLabel);
  await expect(token).toHaveAttribute('aria-pressed', 'false');
  await expect(
    page.locator('.selected-token, .selection-origin, .linked-token-selected'),
  ).toHaveCount(0);
  await expect(stage.locator('.cell-selected, .axis-selection')).toHaveCount(0);

  await second.getByRole('button', {name: 'Expand analysis', exact: true}).click();
  await expect(page.locator('.expanded-analysis')).toBeVisible();
  await second.getByRole('button', {name: 'Select a Position column', exact: true}).click();
  await expectPoint(details);
  await clickPadding(second);
  await expectOverview(details, secondLabel);
  await expect(page.locator('.expanded-analysis')).toBeVisible();
  await expect(stage.locator('.cell-selected, .axis-selection')).toHaveCount(0);
  await details.getByRole('button', {name: 'Locate Key cache Heatmap', exact: true}).click();
  await expect(second.locator('.cache-location')).toHaveAttribute('data-kind', 'key');
  await expect(first.locator('.cache-location')).toHaveCount(0);
  await expectOverview(details, secondLabel);
  expect(
    await stage.locator('.analysis-content').evaluateAll((nodes) => nodes.map((n) => n.id)),
  ).toEqual(ids);

  await page.getByRole('button', {name: 'Exit expanded analysis', exact: true}).click();
  await stage.getByRole('button', {name: secondLabel, exact: true}).click();
  await expect(details.getByRole('region', {name: 'Stage Info', exact: true})).toBeVisible();
  await expect(details.locator('stage-state-info')).toHaveCount(0);
  await expect(stage.getByRole('button', {name: firstLabel, exact: true})).toHaveAttribute(
    'aria-expanded',
    'true',
  );
  await expect(stage.getByRole('button', {name: secondLabel, exact: true})).toHaveAttribute(
    'aria-expanded',
    'false',
  );
  expect(errors).toEqual([]);
});

test('KV State Info follows the panel range and reset without selecting a Position', async ({
  page,
}) => {
  const {stage, details, errors} = await openReview(page);
  const panel = await addKv(page, stage);
  await clickPadding(panel);
  const info = details.getByRole('region', {name: 'Analysis Info', exact: true});
  const positions = details
    .getByRole('table', {name: 'Token Range information'})
    .first()
    .getByRole('row')
    .getByRole('cell');
  const original = await settings(panel);
  await expect(positions).toHaveText(`${original.start}–${original.end}`);
  const start = Number(original.start) + 2;
  const end = Number(original.end) - 2;
  await panel.getByRole('spinbutton', {name: 'Token Range start', exact: true}).fill(String(start));
  await panel.getByRole('spinbutton', {name: 'Token Range start', exact: true}).press('Tab');
  await panel.getByRole('spinbutton', {name: 'Token Range end', exact: true}).fill(String(end));
  await panel.getByRole('spinbutton', {name: 'Token Range end', exact: true}).press('Tab');
  await expect(positions).toHaveText(`${start}–${end}`);
  for (const name of ['Key cache', 'Value cache']) {
    const rows = details.getByRole('region', {name, exact: true}).locator('.kv-layer > tr');
    await expect(rows.first().getByRole('cell')).toHaveCount(4);
  }
  await expect(details.getByRole('region', {name: 'Metric Info', exact: true})).toContainText(
    'Heads',
  );
  await expect(details).not.toContainText('Full-stage statistics');
  await expect(panel.locator('.cell-selected, .axis-selection')).toHaveCount(0);
  await panel.getByRole('button', {name: 'Reset view', exact: true}).click();
  await expect(positions).toHaveText(`${original.start}–${original.end}`);
  await expect(
    details.getByRole('region', {name: 'Key cache', exact: true}).locator('.kv-layer > tr').first(),
  ).toContainText('0');
  expect(errors).toEqual([]);
});

test('KV table follows the panel metric and locates extrema directly', async ({page}) => {
  const {stage, details, errors} = await openReview(page);
  const summaryResponse = page.waitForResponse((response) => {
    const url = new URL(response.url());
    return (
      url.pathname.endsWith('/kv-range') &&
      url.searchParams.get('bins') === '1' &&
      url.searchParams.get('kind') === 'key' &&
      response.ok()
    );
  });
  const panel = await addKv(page, stage);
  const data = await (await summaryResponse).json();
  const source = data.rows.find((row: {layer: number}) => row.layer === 0);
  const key = details.getByRole('region', {name: 'Key cache', exact: true});
  const row = key.locator('.kv-layer').first();
  await expect(key.locator('.sort-label')).toHaveText(['Layer', 'Mean', 'Min', 'Max']);
  await expect(details.getByText('All metrics', {exact: true})).toHaveCount(0);
  await expect(details.getByRole('button', {name: 'Inspect heads', exact: true})).toHaveCount(0);
  for (const [id, label] of [
    ['cosine_similarity', 'Cosine similarity'],
    ['relative_l2', 'Relative L2'],
  ]) {
    if (id !== 'cosine_similarity') {
      await panel.getByRole('button', {name: 'Analysis metric', exact: true}).click();
      await page.getByRole('option', {name: label, exact: true}).click();
    }
    const stats = source.bins[0].metrics[id];
    for (const [index, stat] of ['mean', 'min', 'max'].entries()) {
      await expect
        .poll(
          async () =>
            Number(
              (await row.locator('.stat-value .value').nth(index).innerText()).replace('%', ''),
            ) / (id === 'relative_l2' ? 100 : 1),
        )
        .toBeCloseTo(stats[stat], 3);
    }
    for (const stat of ['min', 'max']) {
      const locate = row.getByRole('button', {
        name: `Locate ${stat} position in Layer 0`,
        exact: true,
      });
      await expect(locate).toHaveAttribute(
        'title',
        `Locate ${stat === 'min' ? 'Min' : 'Max'} · Layer 0 · Position ${stats[stat + '_position']}`,
      );
      await locate.click();
      await expect(panel.locator('.cache-location')).toHaveAttribute(
        'data-position',
        String(stats[stat + '_position']),
      );
      await expect(panel.locator('.cache-location')).toHaveAttribute('data-kind', 'key');
      await expect(panel.locator('.cache-location')).toHaveAttribute('data-layer', '0');
      await expectOverview(details, 'KV cache');
      await expect(panel.getByRole('button', {name: 'Analysis metric', exact: true})).toContainText(
        label,
      );
    }
  }
  expect(errors).toEqual([]);
});

test('Unavailable KV statistics do not trigger automatic popups', async ({page}) => {
  const {stage, details} = await openReview(page);
  await page.route('**/api/kv-range?**', async (route) => {
    const response = await route.fetch();
    const data = await response.json();
    const url = new URL(route.request().url());
    if (url.searchParams.get('bins') === '1' && url.searchParams.get('kind') === 'key') {
      data.rows[0].reason = 'Test capture has no comparable positions';
      for (const metric of Object.values(data.rows[0].bins[0].metrics) as any[]) {
        Object.assign(metric, {
          mean: null,
          min: null,
          max: null,
          min_position: null,
          max_position: null,
          valid_count: 0,
        });
      }
    }
    await route.fulfill({response, json: data});
  });
  await addKv(page, stage);
  const row = details
    .getByRole('region', {name: 'Key cache', exact: true})
    .locator('.kv-layer')
    .first();
  await expect(row.locator('.stat-value .value')).toHaveText(['—', '—', '—']);
  await expect(row.getByRole('button')).toHaveCount(0);
  await expect(details).not.toContainText('Test capture has no comparable positions');
  await expect(page.locator('mat-snack-bar-container')).toHaveCount(0);
});

test('KV loading stays within its cache group while the other group remains usable', async ({
  page,
}) => {
  const {stage, details, errors} = await openReview(page);
  const panel = await addKv(page, stage);
  let releaseKey!: () => void;
  const keyReady = new Promise<void>((resolve) => {
    releaseKey = resolve;
  });
  await page.route('**/api/kv-range?**', async (route) => {
    const url = new URL(route.request().url());
    if (url.searchParams.get('bins') === '1' && url.searchParams.get('kind') === 'key')
      await keyReady;
    await route.continue();
  });
  try {
    await panel.getByRole('button', {name: 'Heads aggregation', exact: true}).click();
    await page.getByRole('button', {name: 'Aggregation function', exact: true}).click();
    await page.getByRole('option', {name: 'Mean', exact: true}).click();
    await page.getByRole('button', {name: 'Apply', exact: true}).click();
    const info = details.getByRole('region', {name: 'Analysis Info', exact: true});
    const statistic = details.getByRole('region', {name: 'Statistic', exact: true});
    const key = statistic.getByRole('region', {name: 'Key cache', exact: true});
    const value = statistic.getByRole('region', {name: 'Value cache', exact: true});
    await expect(info.getByRole('status')).toHaveCount(0);
    await expect(key).toHaveAttribute('aria-busy', 'true');
    await expect(key.getByRole('status')).toHaveText('Loading statistics…');
    await expect(value).toHaveAttribute('aria-busy', 'false');
    await expect(value.locator('.kv-layer')).toHaveCount(15);
    await page.screenshot({path: '/tmp/kv-group-loading.png'});
    await value.getByRole('button', {name: 'Locate min position in Layer 0', exact: true}).click();
    await expect(key.getByRole('status')).toBeVisible();
    releaseKey();
    await expect(key).toHaveAttribute('aria-busy', 'false');
    await expect(key.getByRole('status')).toHaveCount(0);
    await expect(key.locator('.kv-layer')).toHaveCount(15);
    expect(errors).toEqual([]);
  } finally {
    releaseKey();
  }
});
