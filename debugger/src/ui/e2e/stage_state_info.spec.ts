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
test('all five entire-stage inspectors retain overview on locate and drill into positions', async ({
  page,
}) => {
  test.skip(!process.env['E2E_WORKSPACE']?.includes('analysis-review'));
  test.setTimeout(120000);
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('/');
  await page.getByRole('button', {name: /^Open Analysis review/}).click();
  await page.locator('app-toolbar').getByRole('button', {name: 'Debug', exact: true}).click();
  const details = page.getByRole('complementary', {name: 'Comparison details'});
  for (const name of [
    'Final Logits',
    'Sampling Distribution',
    'Activations',
    'Intermediate tensors',
    'KV cache',
  ]) {
    await page
      .getByRole('group', {
        name: ['Final Logits', 'Sampling Distribution'].includes(name)
          ? 'Turn 2 / Decode · Target'
          : 'Turn 2 / Prefill · Target',
        exact: true,
      })
      .press('Enter');
    await details
      .getByRole('region', {name: 'Analyses', exact: true})
      .getByRole('button', {name: 'Add analysis view', exact: true})
      .click();
    await page.getByRole('button', {name: 'Analysis type', exact: true}).click();
    await page.getByRole('option', {name: name + ' Available', exact: true}).click();
    await page.getByRole('button', {name: 'Add', exact: true}).click();
    await expect(
      details.getByRole('region', {
        name: 'Analysis Info',
        exact: true,
      }),
    ).toContainText(name);
    await expect(details.getByText('Loading statistics…')).toHaveCount(0);
    await expect(details).not.toContainText('Statistics unavailable');
    const info = details.getByRole('region', {name: 'Analysis Info', exact: true});
    if (name === 'Sampling Distribution') {
      await expect(info.getByRole('listitem', {name: 'Name', exact: true})).toContainText(
        'Sampling Distribution Analysis 0',
      );
      await expect(info.getByRole('listitem', {name: 'Mode', exact: true})).toContainText(
        'Distribution Comparison',
      );
      await expect(info.getByRole('listitem', {name: 'Visualization', exact: true})).toHaveCount(0);
      await expect(info.getByRole('listitem', {name: 'Metric', exact: true})).toContainText(
        'Entropy',
      );
      await expect(details.getByRole('region', {name: 'Metric Info', exact: true})).toHaveCount(0);
      const commonSections = details.locator(
        'stage-state-info > explorer-info-section > section > .header > strong',
      );
      const commonHeadings = [
        'Analysis Info',
        'Statistics (1)',
        'Distribution Metadata (2)',
        'Metric Metadata (1)',
      ];
      await expect(commonSections).toHaveText(commonHeadings);
      const content = page.locator('.stage-analysis.inspected final-logits-trends');
      await expect(content.getByRole('note')).toHaveCount(0);
      await expect(
        content.locator('analysis-runtime-legend .tensor-runtime-legend > span'),
      ).toHaveText(['Ref', 'Target']);
      const statistics = details.getByRole('region', {name: 'Statistics', exact: true});
      const group = statistics.getByRole('listitem', {
        name: 'Sampling distribution statistics summary',
        exact: true,
      });
      await expect(group).toHaveAttribute('aria-busy', 'false');
      await expect(group.locator('.sort-label')).toHaveText(['Grouping', 'Min', 'Max', 'Mean']);
      await expect(group.locator('.statistic-row td:first-child')).toHaveText(['Ref', 'Target']);
      const metadata = details.getByRole('region', {name: 'Distribution Metadata', exact: true});
      await expect(metadata.locator('.item-name')).toHaveText(['Reference', 'Target']);
      const sources = metadata.locator('.analysis-item');
      await expect(sources).toHaveCount(2);
      for (const source of await sources.all()) {
        const properties = source.locator('explorer-info-list .flat-item');
        await expect(properties).toHaveCount(2);
        for (const [index, label] of ['Vocabulary size', 'Sampling parameters'].entries()) {
          await expect(properties.nth(index)).toHaveAttribute('aria-label', label);
        }
        await expect(properties.locator('explorer-help-term')).toHaveCount(0);
      }
      for (const label of ['Dimensions', 'Shape', 'Dtype', 'Status', 'Basis', 'Source forwards']) {
        await expect(metadata.getByRole('listitem', {name: label, exact: true})).toHaveCount(0);
      }
      const parameters = sources.first().getByRole('listitem', {
        name: 'Sampling parameters',
        exact: true,
      });
      await expect(parameters).toContainText('Temperature=0 · Top-k=1 · Top-p=1 · Seed=0');
      const metricMetadata = details.getByRole('region', {name: 'Metric Metadata', exact: true});
      for (const label of ['Shape', 'Dtype', 'Display precision']) {
        const property = metricMetadata.getByRole('listitem', {name: label, exact: true});
        await expect(property).toBeVisible();
        await expect(property.locator('explorer-help-term')).toHaveCount(0);
      }
      await expect(
        metricMetadata
          .getByRole('listitem', {name: 'Dimensions', exact: true})
          .locator('explorer-help-term'),
      ).toHaveCount(2);
      const start = content.getByRole('spinbutton', {name: 'Token Range start', exact: true});
      const end = content.getByRole('spinbutton', {name: 'Token Range end', exact: true});
      const originalStart = await start.inputValue();
      const originalEnd = await end.inputValue();
      const positions = info.getByRole('listitem', {name: 'Token Range', exact: true});
      await expect(positions).toContainText(`${originalStart}–${originalEnd}`);
      await expect(details.locator('token-distribution')).toHaveCount(0);
      await expect(details.getByRole('region', {name: 'Candidates', exact: true})).toHaveCount(0);
      await info.getByRole('button', {name: 'Locate analysis view', exact: true}).click();
      await expect(group.locator('.statistic-row')).toHaveCount(2);
      await group
        .getByRole('button', {name: 'Locate Sampling distribution Line chart', exact: true})
        .click();
      await expect(content.locator('.cache-location')).toBeVisible();
      for (const runtime of ['Ref', 'Target']) {
        await group
          .getByRole('button', {name: `Locate min position in ${runtime}`, exact: true})
          .click();
        await expect(content.locator('.location-label')).toBeVisible();
        await expect(positions).toContainText(`${originalStart}–${originalEnd}`);
        await expect(group.locator('.statistic-row')).toHaveCount(2);
      }
      await content.getByRole('button', {name: 'Analysis metric', exact: true}).click();
      await expect(page.getByRole('option')).toHaveText([
        /^Selected probability/,
        /^Entropy/,
        /^Top-2 margin/,
      ]);
      await page.getByRole('option', {name: 'Selected probability', exact: true}).click();
      await expect(info.getByRole('listitem', {name: 'Metric', exact: true})).toContainText(
        'Selected probability',
      );
      for (const value of await group.locator('.stat-value .value').allTextContents()) {
        expect(value.trim()).toMatch(/%$/);
        expect(parseFloat(value)).toBeGreaterThanOrEqual(0);
        expect(parseFloat(value)).toBeLessThanOrEqual(100);
      }
      await end.fill(originalStart);
      await end.press('Tab');
      await expect(positions).toContainText(`${originalStart}–${originalStart}`);
      await expect(commonSections).toHaveText(commonHeadings);
      await expect(group.locator('.statistic-row')).toHaveCount(2);
      await expect(details.locator('token-distribution')).toHaveCount(0);
      await expect(details.getByRole('region', {name: 'Candidates', exact: true})).toHaveCount(0);
      await expect(details.locator('.summary-table')).toHaveCount(0);
      for (const source of await sources.all()) {
        await expect(source.locator('explorer-info-list .flat-item')).toHaveCount(2);
        await expect(source.locator('explorer-help-term')).toHaveCount(0);
      }
      for (const label of ['Shape', 'Dtype', 'Display precision']) {
        const property = metricMetadata.getByRole('listitem', {name: label, exact: true});
        await expect(property).toBeVisible();
        await expect(property.locator('explorer-help-term')).toHaveCount(0);
      }
      await expect(content.getByRole('note')).toHaveCount(0);
      await end.fill(originalEnd);
      await end.press('Tab');
      await expect(positions).toContainText(`${originalStart}–${originalEnd}`);
      await expect(group.locator('.statistic-row')).toHaveCount(2);
      await expect(commonSections).toHaveText(commonHeadings);
      await expect(details.getByRole('region', {name: 'Candidates', exact: true})).toHaveCount(0);
      continue;
    }
    if (name === 'Final Logits') {
      await expect(info.getByRole('listitem', {name: 'Name', exact: true})).toContainText(
        'Final Logits Analysis 0',
      );
      await expect(info.getByRole('listitem', {name: 'Metric', exact: true})).toContainText(
        'Root mean square error',
      );
      await expect(info.getByRole('listitem', {name: 'Transform', exact: true})).toHaveCount(0);
      await expect(details.getByRole('region', {name: 'Metric Info', exact: true})).toHaveCount(0);
      await expect(
        details.locator('stage-state-info > explorer-info-section > section > .header > strong'),
      ).toHaveText([
        'Analysis Info',
        'Statistics (1)',
        'Tensor Metadata (2)',
        'Metric Metadata (1)',
      ]);
      await info.getByRole('button', {name: 'Locate analysis view', exact: true}).click();
      const content = page.locator('.stage-analysis.inspected final-logits-trends');
      const metric = content.getByRole('button', {name: 'Analysis metric', exact: true});
      const transform = content.getByRole('button', {name: 'Logits transform', exact: true});
      await expect(metric).toContainText('Root mean square error');
      await expect(transform).toHaveCount(0);
      await expect(content.locator('.series-legend')).toHaveText('Final logits');
      const metadata = details.getByRole('region', {name: 'Tensor Metadata', exact: true});
      const sources = metadata.locator('.analysis-item');
      await expect(sources).toHaveCount(2);
      for (const source of await sources.all()) {
        for (const label of ['Shape', 'Dtype']) {
          const property = source.getByRole('listitem', {name: label, exact: true});
          await expect(property).toBeVisible();
          await expect(property.locator('explorer-help-term')).toHaveCount(0);
        }
      }
      const metricMetadata = details.getByRole('region', {name: 'Metric Metadata', exact: true});
      for (const label of ['Shape', 'Dtype', 'Display precision']) {
        const property = metricMetadata.getByRole('listitem', {name: label, exact: true});
        await expect(property).toBeVisible();
        await expect(property.locator('explorer-help-term')).toHaveCount(0);
      }
      const statistics = details.getByRole('region', {name: 'Statistics', exact: true});
      const group = statistics.getByRole('listitem', {
        name: 'Final logits statistics summary',
        exact: true,
      });
      await expect(group).toHaveAttribute('aria-busy', 'false');
      await expect(group.locator('.item-name')).toHaveText('Final logits');
      await expect(group.locator('.statistic-row')).toHaveCount(1);
      await expect(group.locator('.statistic-row td').first()).toHaveText('Final logits');
      await expect(group.locator('.sort-label')).toHaveText(['Grouping', 'Min', 'Max', 'Mean']);
      await group
        .getByRole('button', {name: 'Locate Final logits Line chart', exact: true})
        .click();
      await expect(content.locator('.cache-location')).toBeVisible();
      const start = content.getByRole('spinbutton', {name: 'Token Range start', exact: true});
      const end = content.getByRole('spinbutton', {name: 'Token Range end', exact: true});
      const originalStart = await start.inputValue();
      const originalEnd = await end.inputValue();
      for (const stat of ['min', 'max']) {
        await group
          .getByRole('button', {name: `Locate ${stat} position in Final`, exact: true})
          .click();
        await expect(content.locator('.location-label')).toContainText('Final');
        await expect(info.getByRole('listitem', {name: 'Token Range', exact: true})).toContainText(
          `${originalStart}–${originalEnd}`,
        );
        await expect(group.locator('.statistic-row')).toHaveCount(1);
      }
      await metric.click();
      await page.getByRole('option', {name: 'Kullback–Leibler divergence', exact: true}).click();
      await expect(info.getByRole('listitem', {name: 'Metric', exact: true})).toContainText(
        'Kullback',
      );
      await expect(transform).toHaveCount(0);
      await expect(content.getByRole('note', {name: 'Softmax before divergence'})).toContainText(
        'KL divergence is computed after applying softmax (T = 1) to the full Ref and Target logits.',
      );
      await metric.click();
      await page.getByRole('option', {name: 'Root mean square error', exact: true}).click();
      await expect(transform).toHaveCount(0);
      await expect(content.getByRole('note', {name: 'Softmax before divergence'})).toHaveCount(0);
      continue;
    }
    await expect(
      info.locator(':scope > .section-content > table').getByRole('rowheader'),
    ).toHaveText(['Analysis', 'Turn', 'Stage']);
    const settings = details.getByRole('region', {name: 'Metric Info', exact: true});
    await expect(settings.getByRole('listitem')).toHaveCount(3);
    await expect(settings.getByRole('listitem', {name: 'Granularity', exact: true})).toContainText(
      'Token position',
    );
    await expect(settings.getByRole('listitem', {name: 'Group by', exact: true})).toHaveCount(0);
    await expect(settings.getByRole('button', {name: /^Locate /})).toHaveCount(
      name === 'KV cache' || name === 'Intermediate tensors' ? 1 : 0,
    );
    await expect(settings.getByRole('listitem', {name: 'Metric Name', exact: true})).toBeVisible();
    if (name === 'KV cache' || name === 'Intermediate tensors') {
      const aggregationValues = settings
        .getByRole('table', {
          name: `${name === 'KV cache' ? 'Heads' : 'Sync points'} information`,
          exact: true,
        })
        .locator('explorer-info-value .value');
      await expect(aggregationValues).toHaveCount(2);
      await expect(aggregationValues.nth(0)).toHaveText(/^\[.+\]$/);
      expect(Array.isArray(JSON.parse(await aggregationValues.nth(0).innerText()))).toBe(true);
      await expect(aggregationValues.nth(1)).toHaveText('Minimum');
    }
    await info.getByRole('button', {name: 'Collapse Analysis Info', exact: true}).click();
    await expect(settings.getByRole('list').first()).toBeVisible();
    await info.getByRole('button', {name: 'Expand Analysis Info', exact: true}).click();
    await settings.getByRole('button', {name: 'Collapse Metric Info', exact: true}).click();
    await expect(settings.getByRole('list').first()).not.toBeVisible();
    await expect(info.getByRole('table')).toBeVisible();
    await settings.getByRole('button', {name: 'Expand Metric Info', exact: true}).click();
    await expect(settings.getByRole('list').first()).toBeVisible();
    await expect(
      details.locator('stage-state-info > explorer-info-section > section > .header > strong'),
    ).toHaveText([
      'Analysis Info',
      'Metric Info',
      name === 'KV cache' ? 'Statistic (2)' : 'Statistic (1)',
    ]);
    await expect(details).not.toContainText('Full-stage statistics');
    await details.getByRole('button', {name: 'Locate analysis instance', exact: true}).click();
    await expect(
      details.getByRole('region', {
        name: 'Analysis Info',
        exact: true,
      }),
    ).toContainText(name);
    const content = page
      .locator('.stage-analysis.inspected .analysis-content')
      .filter({has: page.getByRole('button', {name: 'Analysis metric', exact: true})})
      .last();
    if (!['KV cache', 'Intermediate tensors'].includes(name)) {
      await expect(content.getByRole('button', {name: 'Aggregation', exact: true})).toBeDisabled();
      await expect(
        settings.getByRole('table', {name: 'Aggregation (0) information'}).getByRole('cell'),
      ).toHaveText('-');
    }
    const startControl = content.getByRole('spinbutton', {name: 'Token Range start', exact: true});
    const endControl = content.getByRole('spinbutton', {name: 'Token Range end', exact: true});
    const metricControl = content.getByRole('button', {name: 'Analysis metric', exact: true});
    const aggregationControl = content.locator('analysis-aggregation > button');
    expect((await aggregationControl.boundingBox())!.width).toBe(
      (await metricControl.boundingBox())!.width,
    );
    const toolbar =
      name === 'KV cache'
        ? content.getByRole('group', {name: 'KV analysis controls', exact: true})
        : content.getByRole('toolbar');
    const expand = content.getByRole('button', {name: 'Expand analysis', exact: true});
    const offsets = await toolbar.evaluate((el) => {
      const box = el.getBoundingClientRect(),
        action = el.querySelector('.panel-expand')!.getBoundingClientRect();
      return {top: action.top - box.top, right: box.right - action.right};
    });
    expect(offsets.top).toBeCloseTo(8, 0);
    expect(offsets.right).toBeCloseTo(8, 0);
    const originalStart = Number(await startControl.inputValue());
    const originalEnd = Number(await endControl.inputValue());
    const bins = content.getByRole('button', {name: 'Position bins', exact: true});
    await endControl.fill(String(originalStart + 1));
    await endControl.press('Tab');
    await expect(bins).toBeDisabled();
    await expect(bins).toContainText('-');
    if (!['KV cache', 'Intermediate tensors'].includes(name)) {
      const styles = (button: HTMLElement) => {
        const css = getComputedStyle(button);
        const icon = getComputedStyle(button.querySelector('mat-icon')!);
        return [
          css.opacity,
          css.color,
          css.backgroundColor,
          css.height,
          css.fontSize,
          css.border,
          icon.fontSize,
        ];
      };
      expect(
        await content.getByRole('button', {name: 'Aggregation', exact: true}).evaluate(styles),
      ).toEqual(await bins.evaluate(styles));
    }
    await endControl.fill(String(originalEnd));
    await endControl.press('Tab');
    const rangeRow = details
      .getByRole('table', {name: 'Token Range information'})
      .first()
      .getByRole('row');
    if (originalEnd - originalStart > 2) {
      await startControl.fill(String(originalStart + 1));
      await startControl.press('Tab');
      await expect(rangeRow).toContainText(`${originalStart + 1}–${originalEnd}`);
      await content.getByRole('button', {name: 'Reset view', exact: true}).click();
      await expect(rangeRow).toContainText(`${originalStart}–${originalEnd}`);
    }
    if (name === 'KV cache') {
      await expect(details.getByRole('region', {name: 'Key cache', exact: true})).toBeVisible();
      await expect(details.getByRole('region', {name: 'Value cache', exact: true})).toBeVisible();
      const key = details.getByRole('region', {name: 'Key cache', exact: true});
      await expect(key.locator('.sort-label')).toHaveText(['Layer', 'Mean', 'Min', 'Max']);
      await key.getByRole('button', {name: 'Locate min position in Layer 0', exact: true}).click();
      await expect(details.getByRole('region', {name: 'Analysis Info', exact: true})).toBeVisible();
    } else if (name === 'Activations' || name === 'Intermediate tensors') {
      const layers = details.getByRole('region', {name, exact: true});
      await expect(layers.locator('.sort-label')).toHaveText(['Layer', 'Mean', 'Min', 'Max']);
      await expect(layers.locator('.item-index')).toHaveText('0');
      await expect(layers.locator('details')).toHaveCount(0);
      const rowCount = await layers.locator('.statistic-row').count();
      expect(rowCount).toBeGreaterThan(0);
      await layers
        .getByRole('button', {name: `Sort ${name} by Mean ascending`, exact: true})
        .click();
      await expect(layers.getByRole('columnheader').nth(1)).toHaveAttribute(
        'aria-sort',
        'ascending',
      );
      await layers
        .getByRole('button', {name: /^Locate min position/})
        .first()
        .click();
      await expect(content.locator('.cache-location')).toBeVisible();
      await expect(layers.locator('.statistic-row')).toHaveCount(rowCount);
      await expect(info).toContainText(name);
      await layers.getByRole('button', {name: `Locate ${name} Heatmap`, exact: true}).click();
      await expect(content.locator('.location-label')).toContainText(name);
      if (name === 'Intermediate tensors') {
        const selection = settings
          .getByRole('table', {name: 'Sync points information'})
          .locator('explorer-info-value')
          .first();
        await expect(selection).not.toHaveClass(/expanded/);
        await expect(selection.locator('.value')).toHaveCSS('white-space', 'nowrap');
        await selection
          .getByRole('button', {name: 'Expand Sync points Selection', exact: true})
          .click();
        await expect(selection).toHaveClass(/expanded/);
        await expect(selection.locator('.expanded-value')).toHaveCSS('white-space', 'pre-wrap');
        await selection
          .getByRole('button', {name: 'Collapse Sync points Selection', exact: true})
          .click();
        await expect(selection).not.toHaveClass(/expanded/);
        await settings.getByRole('button', {name: 'Locate Sync points control'}).click();
        const control = content.getByRole('button', {name: 'Sync points aggregation', exact: true});
        await expect
          .poll(() => control.evaluate((button) => button.getAnimations().length))
          .toBe(1);
        await control.click();
        const popup = page.getByRole('dialog', {name: 'Sync points aggregation', exact: true});
        const choices = popup.getByRole('checkbox');
        const first = await choices.nth(1).getAttribute('aria-label');
        await popup.getByRole('checkbox', {name: 'All sync points', exact: true}).uncheck();
        await expect(popup.getByRole('button', {name: 'Apply', exact: true})).toBeDisabled();
        await choices.nth(1).check();
        await popup.getByRole('button', {name: 'Aggregation function', exact: true}).click();
        await page.getByRole('option', {name: 'Mean', exact: true}).click();
        await popup.getByRole('button', {name: 'Apply', exact: true}).click();
        await expect(
          settings
            .getByRole('table', {name: 'Sync points information'})
            .locator('explorer-info-value .value'),
        ).toHaveText([JSON.stringify([first]), 'Mean']);
        await expect(layers.locator('.statistic-row')).toHaveCount(rowCount);
      }
      const plot = content.locator('.js-plotly-plot');
      const point = await plot.evaluate((node: any) => {
        const {xaxis: x, yaxis: y} = node._fullLayout;
        return {x: x._offset + x.d2p(Math.ceil(x.range[0] + 0.5)), y: y._offset + y.d2p(0)};
      });
      await plot.click({position: point});
      await expect(layers.locator('.statistic-row')).toHaveCount(1);
      await expect(details.getByRole('region', {name: 'Tensor data', exact: true})).toBeVisible();
      await content.click({position: {x: 4, y: 4}});
      await expect(layers.locator('.statistic-row')).toHaveCount(rowCount);
    }
  }
  expect(errors).toEqual([]);
});
