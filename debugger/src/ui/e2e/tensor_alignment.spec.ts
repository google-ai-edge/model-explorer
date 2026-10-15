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

test('tensor panels recompute Context pairs and restore Steps without stale inspection', async ({page}) => {
  test.skip(!process.env['E2E_WORKSPACE']?.includes('analysis-review'));
  test.setTimeout(120000);
  await page.goto('/');
  await page.getByRole('button', {name: /^Open Analysis review/}).click();
  await page.locator('app-toolbar').getByRole('button', {name: 'Debug', exact: true}).click();
  const details = page.getByRole('complementary', {name: 'Comparison details'});
  for (const name of ['Intermediate tensors', 'Activations']) {
    await page.getByRole('group', {name: 'Turn 2 / Decode · Target', exact: true}).press('Enter');
    await details.getByRole('button', {name: 'Add analysis view', exact: true}).click();
    await page.getByRole('button', {name: 'Analysis type', exact: true}).click();
    await page.getByRole('option', {name: name + ' Available', exact: true}).click();
    const reply = page.waitForResponse(r => r.url().includes('/position-analysis?') && r.request().method() === 'POST');
    await page.getByRole('button', {name: 'Add', exact: true}).click();
    const response = await reply;
    expect(response.ok()).toBe(true);
    const alignment = response.request().postDataJSON().alignment;
    const data = await response.json();
    const gap = alignment.pairs.findIndex((p: (number|null)[]) => p.includes(null));
    expect(gap).toBeGreaterThanOrEqual(0);
    const gapRows = data.rows.filter((r: any) => r.position === alignment.start + gap);
    expect(gapRows.length).toBeGreaterThan(0);
    expect(gapRows.every((r: any) => r.status === 'not_captured' && !Object.keys(r.metrics).length)).toBe(true);
    const content = page.locator('.stage-analysis.inspected .analysis-content');
    await expect(content.getByRole('spinbutton', {name: 'Token Range end', exact: true}))
      .toHaveValue(String(alignment.start + alignment.pairs.length - 1));
    const stepsReply = page.waitForResponse(r => r.url().includes('/position-analysis?') && r.request().method() === 'GET');
    await page.getByRole('button', {name: 'Comparison alignment', exact: true}).click();
    await page.getByRole('option', {name: 'Steps', exact: true}).click();
    expect((await stepsReply).ok()).toBe(true);
    await expect(details.getByRole('region', {name: 'Analysis Info', exact: true})).toContainText(name);
    await page.getByRole('button', {name: 'Comparison alignment', exact: true}).click();
    await page.getByRole('option', {name: 'Context', exact: true}).click();
    await expect(content.getByRole('spinbutton', {name: 'Token Range end', exact: true}))
      .toHaveValue(String(alignment.start + alignment.pairs.length - 1));
  }
});
