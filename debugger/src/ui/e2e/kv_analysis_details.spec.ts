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

test.skip(
  !process.env['E2E_WORKSPACE']?.includes('analysis-review-dense'),
  'Requires the dense KV review capture',
);

test('Analysis Info follows metric, heads, bin reduction, visible layers and local selection', async ({
  page,
}) => {
  test.setTimeout(120000);
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.setViewportSize({width: 1280, height: 900});
  await page.goto('/');
  await page
    .getByRole('button', {
      name: 'Open Analysis review · 1024 positions · REAL + TEST DATA',
      exact: true,
    })
    .click();
  await page.locator('app-toolbar').getByRole('button', {name: 'Debug', exact: true}).click();
  await page.getByRole('button', {name: 'Comparison alignment', exact: true}).click();
  await page.getByRole('option', {name: 'Steps', exact: true}).click();
  await page.getByRole('group', {name: 'Turn 2 / Prefill · Target', exact: true}).press('Enter');
  const details = page.getByRole('complementary', {name: 'Comparison details'});
  await details.getByRole('button', {name: 'Add analysis view', exact: true}).click();
  await page.getByRole('button', {name: 'Analysis type', exact: true}).click();
  await page.getByRole('option', {name: 'KV cache Available', exact: true}).click();
  await page.getByRole('button', {name: 'Add', exact: true}).click();
  const panel = page.locator('kv-panel');
  const info = details.getByRole('region', {name: 'Analysis Info', exact: true});
  const settings = details.getByRole('region', {name: 'Metric Info', exact: true});
  const selection = details.locator('.statistic-group-header .selection').first();
  const key = details.getByRole('region', {name: 'Key cache', exact: true});
  const value = details.getByRole('region', {name: 'Value cache', exact: true});
  await expect(key.locator('.kv-layer')).toHaveCount(15);
  await expect(value.locator('.kv-layer')).toHaveCount(15);
  await expect(key.getByRole('button', {name: /^Expand Key cache Layer \d+ Mean$/})).toHaveCount(0);
  await expect(
    value.getByRole('button', {name: /^Expand Value cache Layer \d+ Mean$/}),
  ).toHaveCount(0);
  const layerOrder = Array.from({length: 15}, (_, i) => String(i));
  await expect(
    details.getByRole('region', {name: 'Statistic', exact: true}).locator(':scope > header'),
  ).toContainText('Statistic (2)');
  await expect(key.locator('.item-index')).toHaveText('0');
  await expect(value.locator('.item-index')).toHaveText('1');
  const sort = (group: typeof key, column: string) =>
    group.getByRole('button', {name: new RegExp(`^Sort .* by ${column} `)});
  const order = (group: typeof key) => group.locator('.statistic-row > td:first-child');
  await expect(key.getByRole('navigation')).toHaveCount(0);
  await expect(value.getByRole('navigation')).toHaveCount(0);
  await expect(order(value)).toHaveText(layerOrder);
  const alignment = await key.evaluate((group) => {
    const headers = [...group.querySelectorAll('.column-sort')];
    const cells = [...group.querySelectorAll('.statistic-row:first-child .stat-cell')].slice(0, 4);
    return headers.map((header, i) => {
      const label = header.querySelector('.sort-label')!.getBoundingClientRect();
      const icon = header.querySelector('.sort-icon')!.getBoundingClientRect();
      const value = cells[i].firstElementChild!.getBoundingClientRect();
      const action = cells[i].querySelector('.stat-locator')!.getBoundingClientRect();
      return [Math.abs(label.right - value.right), Math.abs(icon.left - action.left)];
    });
  });
  expect(alignment.flat().every((delta) => delta < 0.1)).toBe(true);
  await expect(key.locator('.column-sort mat-icon')).toHaveText([
    'arrow_upward',
    'swap_vert',
    'swap_vert',
    'swap_vert',
  ]);
  await expect(key.locator('.column-sort:disabled')).toHaveCount(0);
  await expect(
    key.getByRole('button', {name: 'Locate min position in Layer 0', exact: true}),
  ).toHaveAttribute('title', /^Locate Min · Layer 0 · Position \d+$/);
  // The header text itself sorts, using ME's normal weight and neutral direction arrow.
  await sort(key, 'Min').getByText('Min', {exact: true}).click();
  await expect(sort(key, 'Min')).toHaveCSS('font-weight', '400');
  await expect(key.locator('.column-sort mat-icon')).toHaveText([
    'swap_vert',
    'swap_vert',
    'arrow_upward',
    'swap_vert',
  ]);
  await expect(key.getByRole('columnheader').nth(2)).toHaveAttribute('aria-sort', 'ascending');
  const keyMin = (
    await key.locator('.statistic-row > td:nth-child(3) .value').allTextContents()
  ).map(Number);
  expect(keyMin).toHaveLength(15);
  expect(keyMin).toEqual([...keyMin].sort((a, b) => a - b));
  await sort(key, 'Min').click();
  await expect(key.getByRole('columnheader').nth(2)).toHaveAttribute('aria-sort', 'descending');
  await expect(sort(key, 'Min').locator('mat-icon')).toHaveText('arrow_downward');
  const descendingMin = (
    await key.locator('.statistic-row > td:nth-child(3) .value').allTextContents()
  ).map(Number);
  expect(descendingMin).toEqual([...keyMin].reverse());
  const sortedKeyLayers = await order(key).allTextContents();
  await sort(value, 'Mean').press('Enter');
  await expect(order(key)).toHaveText(sortedKeyLayers);
  await expect(value.getByRole('columnheader').nth(1)).toHaveAttribute('aria-sort', 'ascending');
  await sort(value, 'Max').click();
  await expect(value.getByRole('columnheader').nth(3)).toHaveAttribute('aria-sort', 'ascending');
  await sort(key, 'Layer').click();
  await expect(order(key)).toHaveText(layerOrder);
  await expect(value.getByRole('columnheader').nth(3)).toHaveAttribute('aria-sort', 'ascending');
  await sort(value, 'Layer').click();
  // Compare every statistic background against the rendered heatmap palette,
  // using raw CosSim values rather than the rounded display labels.
  await expect
    .poll(() =>
      page.evaluate(() => {
        const plot = document.querySelector('kv-panel .js-plotly-plot') as any;
        const trace = plot?.data?.find((t: any) => t.type === 'heatmap');
        const cells = [...document.querySelectorAll('.statistic-group .stat-value')];
        if (!trace || cells.length !== 90) return false;
        const context = document.createElement('canvas').getContext('2d')!;
        const rgb = (color: string) => {
          context.clearRect(0, 0, 1, 1);
          context.fillStyle = color;
          context.fillRect(0, 0, 1, 1);
          return [...context.getImageData(0, 0, 1, 1).data];
        };
        const low = rgb(trace.colorscale[0][1]),
          high = rgb(trace.colorscale[1][1]);
        return cells.every((cell) => {
          const raw = Number(cell.closest('td')!.title.split(': ').at(-1));
          const fraction = Math.min(1, Math.max(0, (1 - raw) / 2));
          const actual = rgb(getComputedStyle(cell).backgroundColor);
          return (
            actual[3] === 255 &&
            low
              .slice(0, 3)
              .every(
                (channel, i) =>
                  Math.abs(actual[i] - (channel + (high[i] - channel) * fraction)) <= 2,
              )
          );
        });
      }),
    )
    .toBe(true);
  const beforeLocate = await info.innerText();
  for (const view of ['Heatmap', 'Line chart']) {
    await panel.getByRole('button', {name: 'Visualization', exact: true}).click();
    await page.getByRole('option', {name: view, exact: true}).click();
    await expect(panel.locator('.range-canvas')).toHaveAttribute('aria-busy', 'false');
    for (const kind of ['key', 'value']) {
      const name = kind === 'key' ? 'Key cache' : 'Value cache';
      const link = details.getByRole('button', {name: `Locate ${name} ${view}`, exact: true});
      if (kind === 'key') await link.getByText(name, {exact: true}).click();
      else await link.locator('mat-icon').click();
      await expect(link).toHaveCSS('font-weight', '500');
      const colors = await link.evaluate((button) => ({
        text: getComputedStyle(button).color,
        icon: getComputedStyle(button.querySelector('mat-icon')!).color,
      }));
      expect(colors.text).not.toBe(colors.icon);
      const located = panel.locator('.cache-location');
      await expect(located).toHaveAttribute('data-kind', kind);
      const box = await panel.locator('.js-plotly-plot').evaluate((node: any, kind) => {
        const layout = node._fullLayout;
        const axis = kind === 'key' ? layout.yaxis : layout.yaxis2;
        return {top: axis._offset - 22, height: axis._length + 26};
      }, kind);
      const actualBox = await located.evaluate((el) => ({
        top: parseFloat(getComputedStyle(el).top),
        height: parseFloat(getComputedStyle(el).height),
      }));
      expect(actualBox.top).toBeCloseTo(box.top, 2);
      expect(actualBox.height).toBeCloseTo(box.height, 2);
      await expect(info).toHaveText(beforeLocate, {useInnerText: true});
      await expect(details.locator('.statistic-group-header .selection')).toHaveCount(0);
      await expect(panel.locator('.cell-selected')).toHaveCount(0);
      const group = kind === 'key' ? key : value;
      const layer = (
        await group.locator('.statistic-row > td:first-child').first().innerText()
      ).trim();
      const rowCount = await group.locator('.statistic-row').count();
      for (const stat of ['min', 'max']) {
        const locator = group.getByRole('button', {
          name: `Locate ${stat} position in Layer ${layer}`,
          exact: true,
        });
        const position = (await locator.getAttribute('title'))!.match(/Position (\d+)/)![1];
        await locator.click();
        await expect(located).toHaveAttribute('data-position', position, {timeout: 2000});
        await expect(located).toHaveAttribute('data-kind', kind);
        await expect(located).toHaveAttribute('data-layer', layer);
        await expect(panel.locator('.location-label')).toHaveText(
          `${name} · Layer ${layer} · Position ${position}`,
        );
        await expect(info).toHaveText(beforeLocate, {useInnerText: true});
        await expect(group.locator('.statistic-row')).toHaveCount(rowCount);
        await expect(panel.locator('.cell-selected')).toHaveCount(0);
      }
    }
  }
  await panel.getByRole('button', {name: 'Visualization', exact: true}).click();
  await page.getByRole('option', {name: 'Tensor Comparison · Heatmap', exact: true}).click();
  await expect(key.locator('.kv-layer')).toHaveCount(15);
  // At narrow inspector widths a clipped value expands in place, without opening a dialog.
  const resize = page.getByRole('separator', {name: 'Resize details panel', exact: true});
  const originalWidth = Number(await resize.getAttribute('aria-valuenow'));
  await resize.press('Home');
  const smallWidth = Number(await resize.getAttribute('aria-valuenow'));
  const expander = key.getByRole('button', {name: /^Expand Key cache Layer .* Min$/}).first();
  await expect(expander).toBeVisible();
  const expandedName = (await expander.getAttribute('aria-label'))!.replace('Expand ', 'Collapse ');
  const valueItem = key.locator('.kv-layer').first().locator('explorer-info-value').nth(1);
  const collapsedRowHeight = await valueItem
    .locator('xpath=ancestor::tr')
    .evaluate((row) => row.getBoundingClientRect().height);
  await expander.click();
  const expanded = key.locator('explorer-info-value.expanded');
  await expect(expanded.locator('.expanded-value')).toHaveText(
    (await expanded.locator('.value').textContent()) ?? '',
  );
  await expect(key.getByRole('button', {name: expandedName, exact: true})).toHaveAttribute(
    'aria-expanded',
    'true',
  );
  await expect(page.getByRole('dialog')).toHaveCount(0);
  await expect(expanded.locator('..').getByRole('button', {name: /^Locate/})).toBeVisible();
  expect(
    await valueItem
      .locator('xpath=ancestor::tr')
      .evaluate((row) => row.getBoundingClientRect().height),
  ).toBeGreaterThan(collapsedRowHeight);
  await expect(expanded.locator('.expanded-value')).toHaveCSS('white-space', 'pre-wrap');
  await expanded.locator('.expanded-value').click();
  await expect(valueItem).toHaveClass(/expanded/);

  await key.getByRole('button', {name: expandedName, exact: true}).click();
  await valueItem.locator('.value').click();
  await expect(valueItem).toHaveClass(/expanded/);
  await key.getByRole('button', {name: expandedName, exact: true}).press('Enter');
  await expect(valueItem).not.toHaveClass(/expanded/);
  for (let size = smallWidth; size < originalWidth; size += 5)
    await resize.press('Shift+ArrowLeft');
  await expect(resize).toHaveAttribute('aria-valuenow', String(originalWidth));
  await expect(key.getByRole('button', {name: /^Expand Key cache Layer \d+ Mean$/})).toHaveCount(0);
  // Fractional widths must not silently clip text when clientWidth rounds up.
  for (let i = 0; i < 3; i++) await resize.press('Shift+ArrowRight');
  await expect(key.getByRole('button', {name: /^Expand Key cache Layer/}).first()).toBeVisible();
  const clipped = await key.locator('explorer-info-value').evaluateAll((items) =>
    items
      .map((item) => {
        const range = document.createRange();
        range.selectNodeContents(item.querySelector('.value')!);
        return {
          clipped: range.getBoundingClientRect().width > item.getBoundingClientRect().width,
          expandable: !!item.querySelector('button'),
        };
      })
      .filter((item) => item.clipped),
  );
  expect(clipped.length).toBeGreaterThan(0);
  expect(clipped.every((item) => item.expandable)).toBe(true);
  const previewToggle = key.getByRole('button', {name: /^Expand Key cache Layer/}).first();
  const previewName = (await previewToggle.getAttribute('aria-label'))!.replace(
    'Expand ',
    'Collapse ',
  );
  await previewToggle.click();
  await key.screenshot({path: '/tmp/metric-value-wrap.png'});
  await key.getByRole('button', {name: previewName, exact: true}).click();

  for (let i = 0; i < 3; i++) await resize.press('Shift+ArrowLeft');

  await page.setViewportSize({width: 1280, height: 1200});
  await details
    .getByRole('region', {name: 'Statistic', exact: true})
    .screenshot({path: '/tmp/statistic-inspector-full.png'});
  await page.setViewportSize({width: 1280, height: 900});
  // Match the upstream info_panel items-container, including the real DOM gap.
  for (const label of ['Analysis Info', 'Metric Info', 'Statistic']) {
    const section = details.getByRole('region', {name: label, exact: true});
    const spacing = await section.evaluate((section) => {
      const header = section.querySelector(':scope > header')!;
      const content = section.querySelector(':scope > .items-container')!;
      const style = getComputedStyle(content);
      return {
        top: content.getBoundingClientRect().top - header.getBoundingClientRect().bottom,
        gap: style.gap,
        bottom: style.marginBottom,
      };
    });
    expect(spacing).toEqual({top: 10, gap: '4px', bottom: '6px'});
  }
  await details.getByRole('button', {name: 'Collapse Statistic', exact: true}).click();
  await expect(key).not.toBeVisible();
  await details.getByRole('button', {name: 'Expand Statistic', exact: true}).click();
  await expect(key).toBeVisible();

  await expect(settings.getByRole('listitem')).toHaveCount(3);
  await expect(settings.locator('.item-index')).toHaveText(['0']);
  await expect(settings.getByRole('heading', {name: 'Aggregation (1)'})).toBeVisible();
  const aggregation = settings.getByRole('heading', {name: 'Aggregation (1)'});
  const metricKey = settings
    .getByRole('table', {name: 'Metric Name information', exact: true})
    .locator('th');
  for (const property of ['font-size', 'font-weight', 'color']) {
    await expect(aggregation).toHaveCSS(
      property,
      await metricKey.evaluate(
        (element, property) => getComputedStyle(element).getPropertyValue(property),
        property,
      ),
    );
  }
  const headsControl = panel.getByRole('button', {name: 'Heads aggregation', exact: true});
  const selectionBeforeLocate = await settings
    .getByRole('table', {name: 'Heads information'})
    .innerText();
  await settings.getByRole('button', {name: 'Locate Heads control'}).click();
  await expect.poll(() => headsControl.evaluate((button) => button.getAnimations().length)).toBe(1);
  await expect(headsControl).toHaveAttribute('aria-expanded', 'false');
  await expect(settings.getByRole('table', {name: 'Heads information'})).toHaveText(
    selectionBeforeLocate,
    {useInnerText: true},
  );
  await expect.poll(() => headsControl.evaluate((button) => button.getAnimations().length)).toBe(0);
  await expect(info.getByRole('rowheader', {name: 'Token Range', exact: true})).toHaveCount(0);
  await expect(settings.locator('.bullet')).toHaveCount(2);
  await expect(key.locator('.column-sort')).toHaveCount(4);
  await expect(info.getByRole('rowheader', {name: 'Scope', exact: true})).toHaveCount(0);
  await expect(info.getByRole('rowheader', {name: 'Position bins', exact: true})).toHaveCount(0);
  for (const width of [1280, 960]) {
    await page.setViewportSize({width, height: 900});
    const expandOffset = async () =>
      panel.getByRole('button', {name: 'Expand analysis', exact: true}).evaluate((button) => {
        const toolbar = button.closest('.toolbar')!.getBoundingClientRect();
        const rect = button.getBoundingClientRect();
        return {top: rect.top - toolbar.top, right: toolbar.right - rect.right};
      });
    const before = await expandOffset();
    const fixedWidths = async () => ({
      metric: (await panel
        .getByRole('button', {name: 'Analysis metric', exact: true})
        .boundingBox())!.width,
      legend: (await panel.locator('.heatmap-color-legend').boundingBox())!.width,
      label: (await panel.locator('.color-metric').boundingBox())!.width,
      ramp: (await panel.locator('.color-ramp').boundingBox())!.width,
    });
    const widths = await fixedWidths();
    for (const metric of ['Relative L2', 'Max abs error', 'Cosine similarity']) {
      await panel.getByRole('button', {name: 'Analysis metric', exact: true}).click();
      await expect(page.getByRole('option', {name: 'Cosine distance', exact: true})).toHaveCount(0);
      await page.getByRole('option', {name: metric, exact: true}).click();
      await expect.poll(expandOffset).toEqual(before);
      await expect.poll(fixedWidths).toEqual(widths);
      await expect(
        settings.getByRole('listitem', {name: 'Metric Name', exact: true}),
      ).toContainText(metric);
    }
  }
  await page.setViewportSize({width: 1280, height: 900});
  await panel.getByRole('button', {name: 'Heads aggregation', exact: true}).click();
  await page.getByRole('checkbox', {name: 'All heads', exact: true}).uncheck();
  await expect(page.getByRole('button', {name: 'Apply', exact: true})).toBeDisabled();
  await page.getByRole('checkbox', {name: 'Head 0', exact: true}).check();
  await page.getByRole('button', {name: 'Apply', exact: true}).click();
  await expect(
    settings.getByRole('table', {name: 'Heads information', exact: true}).getByRole('cell'),
  ).toHaveText(['[0]', 'Minimum']);
  await expect(settings.getByRole('listitem')).toHaveCount(3);
  await panel.getByRole('button', {name: 'Position bins', exact: true}).click();
  await page.getByRole('option', {name: 'Mean', exact: true}).click();
  await expect(key.locator('.column-sort')).toHaveCount(4);
  await panel.getByRole('button', {name: 'Heads aggregation', exact: true}).click();
  await page.getByRole('checkbox', {name: 'Head 1', exact: true}).check();
  await page.getByRole('button', {name: 'Aggregation function', exact: true}).click();
  const meanResponse = page.waitForResponse((response) => {
    const url = new URL(response.url());
    return (
      url.pathname.endsWith('/kv-range') &&
      url.searchParams.get('head') === 'mean' &&
      url.searchParams.get('kind') === 'key' &&
      url.searchParams.get('bins') === '1' &&
      response.ok()
    );
  });
  await page.getByRole('option', {name: 'Mean', exact: true}).click();
  await page.getByRole('button', {name: 'Apply', exact: true}).click();
  await expect(
    settings.getByRole('table', {name: 'Heads information'}).getByRole('cell'),
  ).toHaveText(['[0,1]', 'Mean']);
  const meanData = await (await meanResponse).json();
  expect(meanData.heads).toEqual([0, 1]);
  await expect(settings.getByRole('row').filter({hasText: 'Function'})).toContainText('Mean');
  await expect(panel.locator('.range-canvas')).toHaveAttribute('aria-busy', 'false');
  const mean = key.locator('.kv-layer').first().locator('.stat-value .value').first();
  await expect
    .poll(async () => Number(await mean.innerText()))
    .toBeCloseTo(meanData.rows[0].bins[0].metrics.cosine_similarity.mean, 3);
  // Match one rendered heatmap bin, not an average of chart samples.
  const plot = panel.locator('.js-plotly-plot');
  const hit = await plot.evaluate((node: any) => {
    const trace = node._fullData.find(
      (trace: any) => trace.type === 'heatmap' && trace.name === 'Key cache',
    );
    const index = Math.floor(trace.customdata[0].length / 2);
    const bin = trace.customdata[0][index];
    const layout = node._fullLayout;
    return {
      bin,
      cosine: 1 - trace.z[0][index] * 2,
      x: layout.xaxis._offset + layout.xaxis.d2p((bin.start + bin.end - 1) / 2),
      y: layout.yaxis._offset + layout.yaxis.d2p(0),
    };
  });
  expect(hit.bin.end - hit.bin.start).toBeGreaterThan(1);
  await plot.click({position: {x: hit.x, y: hit.y}});
  await expect(selection).toHaveCount(1);
  await expect(
    key.getByRole('table', {name: 'Token Range information'}).getByRole('row'),
  ).toContainText(`${hit.bin.start}–${hit.bin.end - 1}`);
  await expect(key.locator('.kv-layer')).toHaveCount(1);
  await expect(key.locator('.column-sort:disabled')).toHaveCount(4);
  await expect(value).toHaveCount(0);
  await expect
    .poll(async () => Number(await key.locator('.kv-layer .stat-value .value').first().innerText()))
    .toBeCloseTo(hit.cosine, 3);
  await panel.getByRole('button', {name: 'Analysis metric', exact: true}).click();
  await page.getByRole('option', {name: 'Relative L2', exact: true}).click();
  await expect(settings).toContainText('Relative L2');
  await expect(selection).toHaveCount(1);
  await panel.getByRole('button', {name: 'Visualization', exact: true}).click();
  await page.getByRole('option', {name: 'Tensor Comparison · Line chart', exact: true}).click();
  // Heatmap selection adds its layer to the trend visibility set.
  await expect(selection).toHaveCount(1);
  await panel.getByRole('button', {name: 'Layers', exact: true}).click();
  await page.getByRole('button', {name: 'Clear', exact: true}).click();
  await page.getByRole('button', {name: 'Done', exact: true}).click();
  await expect(selection).toHaveCount(0);
  await expect(key.locator('.kv-layer')).toHaveCount(0);
  await expect(value.locator('.kv-layer')).toHaveCount(0);
  await panel.getByRole('button', {name: 'Visualization', exact: true}).click();
  await page.getByRole('option', {name: 'Tensor Comparison · Heatmap', exact: true}).click();
  await expect(key.locator('.kv-layer')).toHaveCount(15);
  // Select first visible cell, then move Range so that cell is no longer visible.
  const first = await plot.evaluate((node: any) => ({
    x: node._fullLayout.xaxis._offset + 4,
    y: node._fullLayout.yaxis._offset + node._fullLayout.yaxis.d2p(0),
  }));
  await plot.click({position: first});
  await expect(selection).toHaveCount(1);
  await panel.getByRole('spinbutton', {name: 'Token Range start', exact: true}).fill('500');
  await panel.getByRole('spinbutton', {name: 'Token Range start', exact: true}).press('Tab');
  await expect(selection).toHaveCount(0);
  await expect(
    key.getByRole('table', {name: 'Token Range information'}).getByRole('row'),
  ).toContainText('500–1040');
  await expect(panel.locator('.cell-selected, .axis-selection')).toHaveCount(0);
  await expect(panel.locator('.range-canvas')).toHaveAttribute('aria-busy', 'false');
  const valueHit = await plot.evaluate((node: any) => {
    const trace = node._fullData.find(
      (trace: any) => trace.type === 'heatmap' && trace.name === 'Value cache',
    );
    const bin = trace.customdata[0].find((bin: any) => bin.start >= 500);
    return {
      x: node._fullLayout.xaxis._offset + node._fullLayout.xaxis.d2p((bin.start + bin.end - 1) / 2),
      y: node._fullLayout.yaxis2._offset + node._fullLayout.yaxis2.d2p(0),
    };
  });
  await plot.click({position: valueHit});
  await expect(selection).toHaveCount(1);
  await expect(details.locator('.statistic-group-header')).toContainText('Value cache');
  await expect(key).toHaveCount(0);
  await expect(value.locator('.kv-layer')).toHaveCount(1);
  await expect(value.locator('.layer-toggle')).toHaveCount(0);
  await expect(details.getByRole('region', {name: 'Tensor Info', exact: true})).toHaveCount(0);

  await page.setViewportSize({width: 960, height: 800});
  await expect(value.getByRole('table', {name: 'Value cache layers', exact: true})).toBeVisible();
  await panel.locator('..').click({position: {x: 4, y: 4}});
  await expect(selection).toHaveCount(0);
  await expect(key.locator('.kv-layer')).toHaveCount(15);
  await expect(value.locator('.kv-layer')).toHaveCount(15);
  const settingsSize = await settings.evaluate((node) => ({
    width: node.clientWidth,
    scroll: node.scrollWidth,
  }));
  expect(settingsSize.scroll).toBe(settingsSize.width);
  await page.screenshot({path: '/tmp/analysis-settings-960.png'});
  // Exact cells remain chart selections. Tensor view is an explicit, reversible choice.
  await panel.getByRole('spinbutton', {name: 'Token Range end', exact: true}).fill('501');
  await panel.getByRole('spinbutton', {name: 'Token Range end', exact: true}).press('Tab');
  await expect(panel.locator('.range-canvas')).toHaveAttribute('aria-busy', 'false');
  const exactPoint = await plot.evaluate((node: any) => ({
    x: node._fullLayout.xaxis._offset + node._fullLayout.xaxis.d2p(500),
    y: node._fullLayout.yaxis._offset + node._fullLayout.yaxis.d2p(0),
  }));
  await plot.click({position: exactPoint});
  const pointSummary = details.getByRole('table', {name: 'Key cache layers', exact: true});
  await expect(pointSummary.locator('.statistic-row')).toHaveCount(1);
  await expect(details.getByRole('region', {name: 'Tensor Info', exact: true})).toHaveCount(0);
  await panel.getByRole('button', {name: 'Visualization', exact: true}).click();
  await page.getByRole('option', {name: 'Tensor Inspection · Line chart', exact: true}).click();
  const tensor = details.getByRole('region', {name: 'Tensor Info', exact: true});
  await expect(tensor).toBeVisible();
  await expect(settings).toHaveCount(0);
  await expect(panel.getByRole('spinbutton', {name: 'Token Position'})).toHaveValue('500');
  for (const cache of ['Key cache', 'Value cache']) {
    await expect(tensor.getByRole('table', {name: cache + ' metadata', exact: true})).toBeVisible();
    await expect(
      tensor
        .getByRole('table', {name: cache + ' head comparison metrics', exact: true})
        .locator('tbody'),
    ).toHaveCount(4);
  }
  await tensor.getByRole('button', {name: 'Locate Key cache Head 0', exact: true}).click();
  await expect(panel.locator('kv-channel-chart')).toHaveCount(1);
  await expect(panel.locator('kv-channel-chart .rangeslider-container')).toHaveCount(1);
  await panel.getByRole('button', {name: 'Visualization', exact: true}).click();
  await page.getByRole('option', {name: 'Tensor Inspection · Scatter plot', exact: true}).click();
  await expect(panel.locator('svg.scatter-chart')).toHaveCount(2);
  await panel.getByRole('button', {name: 'Next tensor position', exact: true}).click();
  await expect(panel.getByRole('spinbutton', {name: 'Token Position'})).toHaveValue('501');
  await expect(tensor.getByRole('table', {name: 'Position information'}).first()).toContainText(
    '501',
  );
  await panel.getByRole('button', {name: 'Visualization', exact: true}).click();
  await page.getByRole('option', {name: 'Tensor Comparison · Heatmap', exact: true}).click();
  await expect(settings).toBeVisible();
  await expect(pointSummary.locator('.statistic-row')).toHaveCount(1);
  await expect(panel.getByRole('spinbutton', {name: 'Token Range end', exact: true})).toHaveValue('501');
  expect(errors).toEqual([]);
});
