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

import {expect, test} from '@playwright/test';

test('KV Context columns recompute paired positions across charts, statistics and tensor data', async ({
  page,
}) => {
  test.skip(!process.env['E2E_WORKSPACE']?.includes('analysis-review-dense'));
  test.setTimeout(120000);
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.setViewportSize({width: 1400, height: 1000});
  await page.goto('/');
  await page.getByRole('button', {name: /^Open Analysis review/}).click();
  await page.locator('app-toolbar').getByRole('button', {name: 'Debug', exact: true}).click();
  await page.getByRole('group', {name: 'Turn 2 / Decode · Target', exact: true}).press('Enter');
  const details = page.getByRole('complementary', {name: 'Comparison details'});
  await details.getByRole('button', {name: 'Add analysis view', exact: true}).click();
  await page.getByRole('button', {name: 'Analysis type', exact: true}).click();
  await page.getByRole('option', {name: 'KV cache Available', exact: true}).click();
  const firstRequest = page.waitForResponse(
    (response) =>
      response.url().includes('/kv-range?') &&
      response.request().method() === 'POST' &&
      new URL(response.url()).searchParams.get('kind') === 'key',
  );
  await page.getByRole('button', {name: 'Add', exact: true}).click();
  const initial = await firstRequest;
  expect(initial.ok()).toBe(true);
  const alignment = initial.request().postDataJSON().alignment;
  expect(
    alignment.pairs.some(
      (pair: (number | null)[]) => pair[0] !== null && pair[1] !== null && pair[0] !== pair[1],
    ),
  ).toBe(true);
  const gapIndex = alignment.pairs.findIndex(
    (pair: (number | null)[]) => pair[0] === null && pair[1] !== null,
  );
  const pairIndex = alignment.pairs.findIndex(
    (pair: (number | null)[], index: number) =>
      index > gapIndex && pair[0] !== null && pair[1] !== null && pair[0] !== pair[1],
  );
  const position = alignment.start + pairIndex;
  const panel = page.locator('kv-panel');
  await panel.getByRole('button', {name: 'Heads', exact: true}).click();
  await page.getByRole('option', {name: 'Head 0', exact: true}).click();
  await panel
    .getByRole('spinbutton', {name: 'Token Range start', exact: true})
    .fill(String(alignment.start + gapIndex));
  await panel.getByRole('spinbutton', {name: 'Token Range start', exact: true}).press('Tab');
  const rangeReply = page.waitForResponse((response) => {
    const url = new URL(response.url());
    return (
      url.pathname.endsWith('/kv-range') &&
      response.request().method() === 'POST' &&
      url.searchParams.get('kind') === 'key' &&
      url.searchParams.get('head') === '0' &&
      url.searchParams.get('end') === String(position + 1) &&
      Number(url.searchParams.get('bins')) > 1 &&
      response.ok()
    );
  });
  await panel.getByRole('spinbutton', {name: 'Token Range end', exact: true}).fill(String(position));
  await panel.getByRole('spinbutton', {name: 'Token Range end', exact: true}).press('Tab');
  const response = await rangeReply;
  const data = await response.json();
  const params = new URL(response.url()).searchParams;
  const raw = async (p: number) => {
    const query = new URLSearchParams({
      session_id: params.get('session_id')!,
      turn: '2',
      context_id: params.get('context')!,
      layer: '0',
      kind: 'key',
      position: String(p),
      head: '0',
    });
    const result = await page.request.get('/api/kv-head?' + query);
    expect(result.ok()).toBe(true);
    return result.json();
  };
  const ref = await raw(alignment.pairs[pairIndex][0]);
  const target = await raw(alignment.pairs[pairIndex][1]);
  const x = ref.channels.map((row: any) => row.ref),
    y = target.channels.map((row: any) => row.target);
  const dot = x.reduce((sum: number, value: number, i: number) => sum + value * y[i], 0);
  const expected = dot / Math.hypot(...x) / Math.hypot(...y);
  const row = data.rows.find((row: any) => row.layer === 0);
  expect(row.bins[0].metrics.cosine_similarity.valid_count).toBe(0);
  expect(row.bins.at(-1).metrics.cosine_similarity.mean).toBeCloseTo(expected, 10);
  await expect(panel.locator('.range-canvas')).toHaveAttribute('aria-busy', 'false');
  const plot = panel.locator('.js-plotly-plot');
  const clickColumn = async (p: number) => {
    const point = await plot.evaluate(
      (node: any, p) => ({
        x: node._fullLayout.xaxis._offset + node._fullLayout.xaxis.d2p(p),
        y: node._fullLayout.yaxis._offset + node._fullLayout.yaxis.d2p(0),
      }),
      p,
    );
    await plot.click({position: point});
  };
  await clickColumn(position);
  const summary = details.getByRole('table', {name: 'Key cache layers', exact: true});
  await expect(summary.locator('.stat-value .value').first()).toHaveText(expected.toFixed(6));
  const headReply = page.waitForResponse(
    (response) =>
      response.url().includes('/kv-head?') &&
      response.request().method() === 'POST' &&
      response.url().includes('kind=key') &&
      response.url().includes('head=0'),
  );
  await panel.getByRole('button', {name: 'Visualization', exact: true}).click();
  await page.getByRole('option', {name: 'Tensor Inspection · Line chart', exact: true}).click();
  const tensor = details.getByRole('region', {name: 'Tensor Info', exact: true});
  const head = await (await headReply).json();
  expect(head.channels.map((row: any) => row.ref)).toEqual(x);
  expect(head.channels.map((row: any) => row.target)).toEqual(y);
  await tensor.getByRole('button', {name: 'Locate Key cache Head 0', exact: true}).click();
  await expect(panel.locator('kv-channel-chart')).toHaveCount(1);
  await expect(panel.locator('kv-channel-chart .rangeslider-container')).toHaveCount(1);
  await panel.getByRole('button', {name: 'Visualization', exact: true}).click();
  await page.getByRole('option', {name: 'Tensor Comparison · Line chart', exact: true}).click();
  await expect
    .poll(() =>
      plot.evaluate((node: any) =>
        node._fullData.some((trace: any) => trace.type === 'scatter' && trace.name?.includes('0')),
      ),
    )
    .toBe(true);
  const trend = await plot.evaluate(
    (node: any, position) =>
      node._fullData
        .filter((trace: any) => trace.type === 'scatter')
        .flatMap(
          (trace: any) =>
            trace.customdata?.map((cell: any, i: number) =>
              cell?.start === position && cell?.layer === 0 && cell?.kind === 'key'
                ? trace.y[i]
                : null,
            ) ?? [],
        )
        .filter((value: any) => value !== null),
    position,
  );
  expect(trend).toContainEqual(expect.closeTo(expected, 8));
  await panel.getByRole('button', {name: 'Visualization', exact: true}).click();
  await page.getByRole('option', {name: 'Tensor Comparison · Heatmap', exact: true}).click();
  await expect(panel.locator('.range-canvas')).toHaveAttribute('aria-busy', 'false');
  await clickColumn(alignment.start + gapIndex);
  const gapReply = page.waitForResponse(
    (response) =>
      response.url().includes('/kv-head?') &&
      response.request().method() === 'POST' &&
      response.url().includes('kind=key') &&
      response.url().includes('head=0'),
  );
  await panel.getByRole('button', {name: 'Visualization', exact: true}).click();
  await page.getByRole('option', {name: 'Tensor Inspection · Line chart', exact: true}).click();
  const gap = await (await gapReply).json();
  expect(gap.channels.length).toBeGreaterThan(0);
  expect(gap.channels.every((row: any) => row.ref === null && row.delta === null)).toBe(true);
  await expect(tensor.getByRole('table', {name: 'Key cache metadata'})).toContainText(
    'Not aligned',
  );
  await tensor.getByRole('button', {name: 'Locate Key cache Head 0', exact: true}).click();
  await expect(panel.locator('kv-channel-chart')).toHaveCount(1);
  await expect(panel.locator('kv-channel-chart .rangeslider-container')).toHaveCount(1);
  await panel.getByRole('button', {name: 'Visualization', exact: true}).click();
  await page.getByRole('option', {name: 'Tensor Comparison · Heatmap', exact: true}).click();
  await page.getByRole('button', {name: 'Comparison alignment', exact: true}).click();
  const stepsReply = page.waitForResponse(
    (response) => response.url().includes('/kv-range?') && response.request().method() === 'GET',
  );
  await page.getByRole('option', {name: 'Steps', exact: true}).click();
  expect((await stepsReply).ok()).toBe(true);
  await expect(panel.getByRole('spinbutton', {name: 'Token Range start', exact: true})).toHaveValue('32');
  await expect(panel.getByRole('spinbutton', {name: 'Token Range end', exact: true})).toHaveValue('63');
  expect(errors).toEqual([]);
});
