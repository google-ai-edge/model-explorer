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
  !process.env['E2E_WORKSPACE']?.includes('analysis-review'),
  'Requires the generated real + test review capture',
);
async function add(page: Page, stage: Locator, name: string) {
  await stage.getByRole('button', {name: 'Add analysis view', exact: true}).click();
  await page.getByRole('button', {name: 'Analysis type', exact: true}).click();
  await page.getByRole('option', {name: `${name} Available`, exact: true}).click();
  await expect(page.getByRole('region', {name: 'Analysis preview'})).not.toContainText(
    'Not available',
  );
  await page.getByRole('button', {name: 'Add', exact: true}).click();
  await expect(page.getByRole('dialog', {name: 'Add analysis view', exact: true})).toHaveCount(0);
}
test('one capture exercises all five analyses, both views, selections and empty data', async ({
  page,
}) => {
  test.skip(
    process.env['E2E_WORKSPACE']?.includes('analysis-review-dense'),
    'Use the dense scenario below',
  );
  test.setTimeout(180_000);
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
  });
  await page.setViewportSize({width: 1440, height: 1000});
  await page.goto('/');
  await page
    .getByRole('button', {name: 'Open Analysis review · Gemma 4 · REAL + TEST DATA', exact: true})
    .click();
  await page.locator('app-toolbar').getByRole('button', {name: 'Debug', exact: true}).click();
  const stages = page.locator('.stage-analysis-row[data-stage="prefill"]');
  const stage = stages.nth(1);
  for (const name of ['Activations', 'Intermediate tensors', 'KV cache'])
    await add(page, stage, name);
  const panels = stage.locator('stage-graph-metrics');
  await expect(panels).toHaveCount(2);
  for (const panel of await panels.all()) {
    await expect(panel.getByRole('note')).toHaveCount(0);
    await expect(panel).not.toContainText('generated paired vectors');
    const plot = panel.locator('.js-plotly-plot');
    await expect(plot.locator('.heatmaplayer:not(.rangeplot) image')).toHaveCount(1);
    expect(await plot.evaluate((node: any) => node._fullLayout.xaxis.range[0])).toBe(16.5);
    const slider = plot.locator('.rangeslider-grabber-max');
    await slider.scrollIntoViewIfNeeded();
    const handle = (await slider.boundingBox())!;
    await page.mouse.move(handle.x + handle.width / 2, handle.y + handle.height / 2);
    await page.mouse.down();
    await page.mouse.move(handle.x - 120, handle.y + handle.height / 2, {steps: 8});
    await page.mouse.up();
    await expect
      .poll(() => plot.evaluate((node: any) => node._fullLayout.xaxis.range[1]))
      .toBeLessThan(31.5);
    await panel.getByRole('button', {name: 'Select a Position column', exact: true}).click();
    await expect(panel.locator('.cell-selected')).toHaveCount(1);
    await page.getByRole('button', {name: 'Expand Tensor data', exact: true}).click();
    await expect(page.getByRole('complementary', {name: 'Comparison details'})).toContainText(
      'Test data',
    );
    await page.getByRole('button', {name: 'Collapse Tensor data', exact: true}).click();
    await panel.getByRole('button', {name: 'Analysis metric', exact: true}).click();
    expect(await page.getByRole('listbox').evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(
      true,
    );
    await page.getByRole('option', {name: 'Relative L2 (%)', exact: true}).click();
    await panel.getByRole('button', {name: 'Line chart', exact: true}).click();
    await expect(plot.locator('.heatmaplayer:not(.rangeplot) image')).toHaveCount(0);
    await expect(plot.locator('.scatterlayer .trace').first()).toBeVisible();
    await panel.getByRole('button', {name: 'Layers', exact: true}).click();
    await page.getByRole('searchbox', {name: 'Search layers'}).fill('L7');
    await page.getByRole('checkbox', {name: 'Layer 7', exact: true}).check();
    await expect(page.getByRole('checkbox', {name: 'Layer 7', exact: true})).toBeChecked();
    await page.getByRole('button', {name: 'Done', exact: true}).click();
    await expect
      .poll(() =>
        plot.evaluate((node: any) => node.data.some((s: any) => s.name?.startsWith('L7 ·'))),
      )
      .toBe(true);
    await expect(panel.getByRole('button', {name: 'Select a Position column'})).toBeVisible();
    await panel.getByRole('button', {name: 'Heatmap', exact: true}).click();
    await expect(plot.locator('.heatmaplayer:not(.rangeplot) image')).toHaveCount(1);
  }
  const intermediate = panels.nth(1);
  await intermediate.getByRole('button', {name: 'Sync Points', exact: true}).click();
  await page.getByRole('option', {name: 'attention_output', exact: true}).click();
  await expect(intermediate.locator('.cell-unavailable')).toHaveCount(1);
  const tensorPlot = intermediate.locator('.js-plotly-plot');
  await tensorPlot.scrollIntoViewIfNeeded();
  const missingPoint = await tensorPlot.evaluate((node: any) => {
    const box = node.getBoundingClientRect(),
      l = node._fullLayout;
    return {
      x: box.x + l.xaxis._offset + l.xaxis.d2p(19),
      y: box.y + l.yaxis._offset + l.yaxis.d2p(3),
    };
  });
  await page.mouse.move(missingPoint.x, missingPoint.y);
  await expect(intermediate.locator('.cell-hover')).toHaveCount(1);
  await page.mouse.click(missingPoint.x, missingPoint.y);
  await page.getByRole('button', {name: 'Expand Tensor data', exact: true}).click();
  await expect(page.getByRole('complementary', {name: 'Comparison details'})).toContainText(
    'not_captured',
  );
  await tensorPlot.press('Home');
  for (let i = 0; i < 3; i++) await tensorPlot.press('ArrowUp');
  for (const key of ['ArrowDown', 'ArrowDown', 'ArrowDown', 'ArrowRight', 'ArrowRight'])
    await tensorPlot.press(key);
  await expect(page.getByRole('complementary', {name: 'Comparison details'})).toContainText(
    'not_captured',
  );
  await page.getByRole('button', {name: 'Theme', exact: true}).click();
  await page.getByRole('menuitemradio', {name: /^Dark/}).click();
  await expect
    .poll(() => page.evaluate(() => document.documentElement.dataset['theme']))
    .toBe('dark');
  await page.getByRole('button', {name: 'Theme', exact: true}).click();
  await page.getByRole('menuitemradio', {name: /^Light/}).click();
  const kv = stage.locator('kv-panel');
  await expect(kv.getByRole('spinbutton', {name: 'Token Range start'})).toHaveValue('17');
  await expect(kv.locator('.heatmaplayer:not(.rangeplot) image')).toHaveCount(2);
  await kv.getByRole('button', {name: 'Select a Position column', exact: true}).click();
  await expect(kv.locator('.cell-selected')).toHaveCount(2);
  const decode = page.locator('.stage-analysis-row[data-stage="decode"]').first();
  await add(page, decode, 'Final Logits');
  await expect(decode.locator('token-trends .js-plotly-plot')).toBeVisible();
  const logits = decode.locator('final-logits-trends');
  await expect(logits.getByRole('button', {name: 'Analysis metric', exact: true})).toContainText(
    'Root mean square error',
  );
  await expect(logits.getByRole('button', {name: 'Logits transform', exact: true})).toHaveCount(0);
  await expect(logits.getByRole('button', {name: 'Reset view', exact: true})).toBeDisabled();
  await expect(logits.locator('.position-column-axis')).toBeVisible();
  const initialEnd = await logits.getByRole('spinbutton', {name: 'Token Range end'}).inputValue();
  const initialStart = await logits
    .getByRole('spinbutton', {name: 'Token Range start'})
    .inputValue();
  expect(Number(initialStart)).toBeGreaterThan(0);
  await logits.getByRole('spinbutton', {name: 'Token Range end'}).fill(initialStart);
  await logits.getByRole('spinbutton', {name: 'Token Range end'}).press('Tab');
  await logits.getByRole('button', {name: 'Reset view', exact: true}).click();
  await expect(logits.getByRole('spinbutton', {name: 'Token Range end'})).toHaveValue(initialEnd);
  await logits.getByRole('button', {name: 'Expand analysis', exact: true}).click();
  await expect(page.locator('.expanded-analysis')).toBeVisible();
  await expect(
    page.getByRole('tab', {name: 'Final Logits Analysis 0', exact: true}),
  ).toHaveAttribute('aria-selected', 'true');
  await page
    .locator('.expanded-analysis-header')
    .getByRole('button', {name: 'Exit expanded analysis'})
    .click();
  await add(page, decode, 'Sampling Distribution');
  const distribution = decode.locator('final-logits-trends').nth(1);
  await expect(distribution.getByRole('note')).toHaveCount(0);
  await expect(
    distribution.getByRole('button', {name: 'Visualization', exact: true}),
  ).toContainText('Distribution Comparison');
  await distribution.getByRole('button', {name: 'Analysis metric', exact: true}).click();
  await expect(page.getByRole('option')).toHaveText([
    /^Selected probability/,
    /^Entropy/,
    /^Top-2 margin/,
  ]);
  await page.getByRole('option', {name: 'Top-2 margin', exact: true}).click();
  await logits.first().getByRole('button', {name: 'Analysis metric', exact: true}).click();
  await expect(page.getByRole('option')).toHaveText([
    /^Root mean square error/,
    /^Max absolute error/,
    /^Kullback–Leibler divergence/,
    /^Jensen–Shannon divergence/,
  ]);
  await page.getByRole('option', {name: 'Jensen–Shannon divergence', exact: true}).click();
  await distribution.getByRole('button', {name: 'Expand analysis', exact: true}).click();
  await expect(
    page.getByRole('tab', {name: 'Sampling Distribution Analysis 0', exact: true}),
  ).toHaveAttribute('aria-selected', 'true');
  await page.getByRole('tab', {name: 'Final Logits Analysis 0', exact: true}).click();
  await expect(logits.first().getByRole('button', {name: 'Analysis metric'})).toContainText(
    'Jensen',
  );
  await page.getByRole('button', {name: 'Exit expanded analysis'}).click();
  await kv.getByRole('button', {name: 'Expand analysis', exact: true}).click();
  await expect(page.getByRole('tab')).toHaveCount(3);
  await page.getByRole('tab', {name: 'Activation Analysis 0', exact: true}).click();
  await expect(page.locator('.expanded-analysis stage-graph-metrics').first()).toBeVisible();
  await page.getByRole('tab', {name: 'Activation Analysis 0', exact: true}).press('ArrowRight');
  await expect(
    page.getByRole('tab', {name: 'Intermediate Tensor Analysis 0', exact: true}),
  ).toHaveAttribute('aria-selected', 'true');
  await page
    .getByRole('tab', {name: 'Intermediate Tensor Analysis 0', exact: true})
    .press('Escape');
  await expect(page.locator('.expanded-analysis')).toHaveCount(0);

  await expect(page.getByRole('button', {name: 'Find condition', exact: true})).toHaveCount(1);
  await expect(
    page.getByRole('button', {name: 'Remove Activation Analysis 0', exact: true}),
  ).toHaveCount(1);
  await page.getByRole('button', {name: 'Remove Activation Analysis 0', exact: true}).click();
  await expect(panels).toHaveCount(1);
  // A one-sided terminal capture still has real axes, but no comparable values.
  const oneSided = page.locator('.stage-analysis-row[data-stage="decode"]').nth(1);
  await add(page, oneSided, 'KV cache');
  await expect(oneSided.locator('kv-panel .cell-unavailable').first()).toBeVisible();
  // Return to a token; Details must stop presenting the last analysis selection.
  await page.getByRole('button', {name: 'target aligned position 0: hello', exact: true}).click();
  await expect(page.getByRole('complementary', {name: 'Comparison details'})).not.toContainText(
    'Test data',
  );
  await page.setViewportSize({width: 1280, height: 800});
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(errors).toEqual([]);
});

test('dense Position bins and manual range stay synchronized with the native slider', async ({
  page,
}) => {
  test.skip(
    !process.env['E2E_WORKSPACE']?.includes('analysis-review-dense'),
    'Requires dense fixture',
  );
  test.setTimeout(180_000);
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.setViewportSize({width: 1440, height: 1000});
  await page.goto('/');
  await page
    .getByRole('button', {
      name: 'Open Analysis review · 1024 positions · REAL + TEST DATA',
      exact: true,
    })
    .click();
  await page.locator('app-toolbar').getByRole('button', {name: 'Debug', exact: true}).click();
  const stage = page.locator('.stage-analysis-row[data-stage="prefill"]').nth(1);
  await add(page, stage, 'Intermediate tensors');
  const panel = stage.locator('stage-graph-metrics'),
    plot = panel.locator('.js-plotly-plot');
  await expect(plot.locator('.heatmaplayer:not(.rangeplot) image')).toHaveCount(1);
  await expect(panel.getByRole('spinbutton', {name: 'Token Range start'})).toHaveValue('17');
  await expect(panel.getByRole('spinbutton', {name: 'Token Range end'})).toHaveValue('1040');
  await expect(panel.getByRole('button', {name: 'Position bins', exact: true})).toBeEnabled();
  await expect(panel.getByRole('button', {name: 'Heatmap', exact: true})).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  await panel.getByRole('button', {name: 'Sync Points', exact: true}).click();
  expect(await page.getByRole('listbox').evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(
    true,
  );
  await page.getByRole('option', {name: 'Min', exact: true}).click();
  const stripe = await panel.locator('.position-column-axis').boundingBox();
  const ticks = await plot.locator('.xtick text').evaluateAll((nodes) =>
    nodes.map((node) => ({
      top: node.getBoundingClientRect().top,
      bottom: node.getBoundingClientRect().bottom,
    })),
  );
  expect(ticks.length).toBeGreaterThan(0);
  for (const tick of ticks) expect(tick.top).toBeGreaterThanOrEqual(stripe!.y + stripe!.height);
  const bins = await plot.evaluate((node: any) => node.data[0].customdata[0]);
  const tickCenters = await plot.evaluate((node: any) => node.layout.xaxis.tickvals);
  expect(tickCenters.length).toBeGreaterThan(0);
  for (const center of tickCenters)
    expect(bins.some((bin: any) => center === (bin.start + bin.end - 1) / 2)).toBe(true);

  expect(bins.length).toBeLessThan(1024);
  expect(bins[0].end - bins[0].start).toBeGreaterThan(1);
  await plot.scrollIntoViewIfNeeded();
  const hit = await plot.evaluate((node: any) => {
    const r = node.getBoundingClientRect(),
      l = node._fullLayout;
    return {x: r.x + l.xaxis._offset + 100, y: r.y + l.yaxis._offset + l.yaxis.d2p(3)};
  });
  await page.mouse.move(hit.x, hit.y);
  await expect(plot.locator('..').locator('.chart-tooltip')).toContainText(/Position \d+–\d+/);
  await panel.getByRole('spinbutton', {name: 'Token Range end'}).fill('48');
  await panel.getByRole('spinbutton', {name: 'Token Range end'}).press('Tab');
  await panel.getByRole('spinbutton', {name: 'Token Range start'}).fill('20');
  await panel.getByRole('spinbutton', {name: 'Token Range start'}).press('Tab');
  await expect
    .poll(() => plot.evaluate((node: any) => node._fullLayout.xaxis.range))
    .toEqual([19.5, 48.5]);
  await expect(panel.getByRole('button', {name: 'Position bins', exact: true})).toBeDisabled();
  await panel.getByRole('spinbutton', {name: 'Token Range start'}).fill('17');
  await panel.getByRole('spinbutton', {name: 'Token Range start'}).press('Tab');
  await panel.getByRole('spinbutton', {name: 'Token Range end'}).fill('1040');
  await panel.getByRole('spinbutton', {name: 'Token Range end'}).press('Tab');
  const handle = plot.locator('.rangeslider-grabber-max');
  await handle.scrollIntoViewIfNeeded();
  const box = (await handle.boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x - 140, box.y + box.height / 2, {steps: 8});
  await page.mouse.up();
  await expect
    .poll(async () =>
      Number(await panel.getByRole('spinbutton', {name: 'Token Range end'}).inputValue()),
    )
    .toBeLessThan(1040);
  if (process.env['E2E_WORKSPACE']?.endsWith('-dense-kv')) {
    const singleHeight = await plot.evaluate((node: any) => node._fullLayout.yaxis._length);
    await add(page, stage, 'KV cache');
    const kv = stage.locator('kv-panel'),
      kvPlot = kv.locator('.js-plotly-plot');
    await expect(kvPlot.locator('.heatmaplayer:not(.rangeplot) image')).toHaveCount(2);
    await expect(kv.getByRole('spinbutton', {name: 'Token Range start'})).toHaveValue('17');
    await expect(kv.getByRole('spinbutton', {name: 'Token Range end'})).toHaveValue('1040');
    const minimumColumn = await kv
      .locator('.axis-bin')
      .evaluateAll((nodes) => Math.min(...nodes.map((node) => node.getBoundingClientRect().width)));
    expect(minimumColumn).toBeGreaterThanOrEqual(8);
    const sizes = await kvPlot.evaluate((node: any) => [
      node._fullLayout.yaxis._length,
      node._fullLayout.yaxis2._length,
    ]);
    for (const height of sizes) expect(Math.abs(height - singleHeight)).toBeLessThan(3);
    expect(
      await kv.locator('.position-column-axis').evaluate((el) => el.getBoundingClientRect().height),
    ).toBe(8);
    await expect(kvPlot.locator('.annotation')).toContainText(['Key cache', 'Value cache']);
    const columnStrip = kv.locator('.position-column-axis');
    await columnStrip.scrollIntoViewIfNeeded();
    const stripBox = (await columnStrip.boundingBox())!;
    await page.mouse.move(stripBox.x + 100, stripBox.y + stripBox.height / 2);
    await expect(kv.locator('.column-preview')).toHaveCount(2);
    const hoverBoxes = await kv.locator('.column-preview').evaluateAll((nodes) =>
      nodes.map((node) => ({
        top: node.getBoundingClientRect().top,
        bottom: node.getBoundingClientRect().bottom,
      })),
    );
    expect(hoverBoxes[1].top - hoverBoxes[0].bottom).toBeGreaterThan(40);
    for (const box of hoverBoxes)
      expect(Math.abs(box.bottom - box.top - singleHeight)).toBeLessThan(3);
    await page.mouse.move(stripBox.x - 20, stripBox.y);
    await expect(kv.locator('.column-preview')).toHaveCount(0);
    await kv.getByRole('button', {name: 'Heads', exact: true}).click();
    expect(await page.getByRole('listbox').evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(
      true,
    );
    await page.getByRole('option', {name: 'Min', exact: true}).click();
    const inputs = page.locator('.input-token[data-turn="2"]');
    const refToken = inputs.filter({hasText: '<|turn>'}).first();
    await refToken.click();
    await expect(refToken).toHaveClass(/selection-origin/);
    await expect(
      page.locator('.input-token[data-turn="2"][data-side="target"].linked-token-selected'),
    ).toHaveCount(1);
    await columnStrip.scrollIntoViewIfNeeded();
    const firstStrip = (await columnStrip.boundingBox())!;
    await page.mouse.move(firstStrip.x + 4, firstStrip.y + 4);
    await expect
      .poll(() => page.locator('.input-token[data-turn="2"].linked-token-hover').count())
      .toBeGreaterThan(0);
    await expect(refToken).toHaveClass(/selection-origin/);
    await page.mouse.click(firstStrip.x + 4, firstStrip.y + 4);
    await expect(page.getByRole('complementary', {name: 'Comparison details'})).toContainText(
      'Analysis Info',
    );
    await expect
      .poll(() => page.locator('.input-token[data-turn="2"].linked-token-selected').count())
      .toBeGreaterThan(2);
    await expect(page.locator('.input-token.selection-origin')).toHaveCount(0);
    await page.mouse.move(firstStrip.x - 20, firstStrip.y);
    await expect(page.locator('.input-token.linked-token-hover')).toHaveCount(0);
    await expect
      .poll(() => page.locator('.input-token[data-turn="2"].linked-token-selected').count())
      .toBeGreaterThan(2);
    // Line chart shares bins, range controls and selections with the two Heatmap planes.
    const heatValues = await kvPlot.evaluate((node: any) =>
      node.data.filter((t: any) => t.type === 'heatmap').map((t: any) => t.z),
    );
    await kv.getByRole('button', {name: 'Visualization', exact: true}).click();
    await page.getByRole('option', {name: 'Tensor Comparison · Line chart', exact: true}).click();
    await expect(kvPlot.locator('.annotation')).toContainText(['Key cache', 'Value cache']);
    await expect
      .poll(() => kvPlot.evaluate((node: any) => node._fullLayout.yaxis2?.matches))
      .toBe('y');
    await expect(kvPlot.locator('.rangeslider-container')).toHaveCount(1);
    await expect(kvPlot.locator('.rangeslider-container .heatmaplayer image')).toHaveCount(0);
    await expect(kv.locator('.cell-selected')).toHaveCount(2);
    const trendValues = await kvPlot.evaluate((node: any) =>
      node.data
        .filter((t: any) => t.customdata)
        .map((t: any) => ({
          layer: t.customdata.find(Boolean)?.layer,
          plane: t.yaxis === 'y2' ? 1 : 0,
          values: t.y,
        })),
    );
    for (const t of trendValues) {
      expect(t.values.length).toBe(heatValues[t.plane][t.layer].length);
      t.values.forEach((value: number | null, i: number) => {
        const normalized = heatValues[t.plane][t.layer][i];
        if (value === null) expect(normalized).toBeNull();
        else expect(value).toBeCloseTo(1 - 2 * normalized, 10);
      });
    }
    await columnStrip.scrollIntoViewIfNeeded();
    const trendStrip = (await columnStrip.boundingBox())!;
    await page.mouse.move(trendStrip.x + 100, trendStrip.y + 4);
    await expect(kv.locator('.column-preview')).toHaveCount(2);
    await page.mouse.click(trendStrip.x + 100, trendStrip.y + 4);
    await expect(kv.locator('.cell-selected')).toHaveCount(2);
    await kv.getByRole('button', {name: 'Layers', exact: true}).click();
    const legend = page.getByRole('dialog', {name: 'Layer legend'});
    await legend.getByRole('searchbox', {name: 'Search layers'}).fill('L7');
    await expect(legend.getByRole('checkbox')).toHaveCount(1);
    await legend.getByRole('checkbox', {name: 'Layer 7', exact: true}).check();
    await legend.getByRole('searchbox', {name: 'Search layers'}).fill('L12');
    await legend.getByRole('checkbox', {name: 'Layer 12', exact: true}).check();
    await legend.getByRole('button', {name: 'Done', exact: true}).click();
    await expect
      .poll(() =>
        kvPlot.evaluate(
          (node: any) => node.data.filter((t: any) => t.name?.startsWith('L7 ·')).length,
        ),
      )
      .toBe(2);
    const colors = await kvPlot.evaluate((node: any) =>
      node.data.filter((t: any) => t.name?.startsWith('L7 ·')).map((t: any) => t.line.color),
    );
    expect(colors[0]).toBe(colors[1]);
    await kv.getByRole('button', {name: 'Layers', exact: true}).click();
    await legend.getByRole('searchbox', {name: 'Search layers'}).fill('L7');
    await legend.getByText('Layer 7', {exact: true}).hover();
    await expect
      .poll(() =>
        kvPlot.evaluate((node: any) =>
          node.data
            .filter((t: any) => t.name && !t.name.startsWith('L7 ·'))
            .every((t: any) => t.opacity === 0.18),
        ),
      )
      .toBe(true);
    await legend.getByRole('button', {name: 'Done', exact: true}).click();
    await expect
      .poll(() => kvPlot.evaluate((node: any) => node.data.every((t: any) => t.opacity === 1)))
      .toBe(true);
  }
  expect(errors).toEqual([]);
});
