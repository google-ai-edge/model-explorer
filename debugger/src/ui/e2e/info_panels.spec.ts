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

test('conversation locator and stage analyses preserve inspection until Add succeeds', async ({
  page,
}) => {
  test.skip(!process.env['E2E_WORKSPACE']?.includes('analysis-review'));
  await page.goto('/');
  await page.getByRole('button', {name: /^Open Analysis review/}).click();
  await page.locator('app-toolbar').getByRole('button', {name: 'Debug', exact: true}).click();
  const details = page.getByRole('complementary', {name: 'Comparison details'});
  const conversation = details.getByRole('region', {name: 'Conversation Info', exact: true});
  await expect(conversation).toContainText('Turns');
  const stageList = details.getByRole('region', {name: 'Stages', exact: true});
  await expect(stageList).toContainText('Stages (4)');
  await expect(conversation).not.toContainText('Generated tokens');
  await details
    .getByRole('region', {name: 'Stages', exact: true})
    .getByRole('button', {name: 'Locate Turn 2 · Prefill', exact: true})
    .click();
  await expect(conversation).toBeVisible();
  await expect(stageList).not.toContainText('<|turn>');
  await expect(stageList).not.toContainText('History matched');
  await expect(stageList.getByText('History Match', {exact: true})).toHaveCount(4);
  await expect(stageList.getByText('Cumulative Match Rate', {exact: true})).toHaveCount(4);
  await expect(
    stageList
      .getByRole('listitem', {name: 'Turn 2 · Decode', exact: true})
      .getByRole('row')
      .filter({hasText: 'First Divergence'}),
  ).toContainText('Yes');
  await expect(page.locator('.debug-node .stage-mismatch-badge')).toHaveCount(2);
  await expect(page.locator('.reading')).not.toContainText('History matched');
  await page.getByRole('group', {name: 'Turn 1 / Prefill · Target', exact: true}).press('Enter');
  const stage = details.getByRole('region', {name: 'Stage Info', exact: true});
  await expect(stage).toContainText('Selection');
  await expect(stage).not.toContainText('Entire stage');
  await expect(stage).toContainText('Match Rate');
  await expect(stage).not.toContainText('All Tokens Match');
  await expect(stage).toContainText('Token Range');
  await expect(stage).not.toContainText('Input tokens match');
  await expect(stage).not.toContainText('Peak KL');
  const analyses = details.getByRole('region', {name: 'Analyses', exact: true});
  await expect(analyses.getByRole('button', {name: 'Collapse Analyses'})).toHaveCount(0);
  await analyses.getByRole('button', {name: 'Add analysis view', exact: true}).click();
  await page.getByRole('button', {name: 'Cancel', exact: true}).click();
  await expect(stage).toBeVisible();
  await analyses.getByRole('button', {name: 'Add analysis view', exact: true}).click();
  await page.getByRole('button', {name: 'Analysis type', exact: true}).click();
  await page.getByRole('option', {name: 'Activations Available', exact: true}).click();
  await page.getByRole('button', {name: 'Add', exact: true}).click();
  await expect(details.getByRole('region', {name: 'Analysis Info', exact: true})).toBeVisible();
  const entry = page.getByRole('button', {name: 'Activations', exact: true});
  await expect(entry).toHaveAttribute('aria-expanded', 'true');
  await entry.click();
  await expect(stage).toBeVisible();
  await analyses.getByRole('button', {name: 'Locate Activations analysis', exact: true}).click();
  await expect(entry).toHaveAttribute('aria-expanded', 'true');
  await expect(stage).toBeVisible();
  await page.getByRole('button', {name: 'Remove Activations analysis', exact: true}).click();
  await expect(stage).toBeVisible();
  await expect(analyses.getByRole('button', {name: 'Collapse Analyses'})).toHaveCount(0);
  await page.getByRole('group', {name: 'Turn 2 / Decode · Target', exact: true}).press('Enter');
  await expect(stage).not.toContainText('First divergence');
  await expect(stage.getByRole('row', {name: 'Match Rate 57.14%', exact: true})).toBeVisible();
  await expect(stage).not.toContainText('Top-2 margin');
  await stage.getByRole('button', {name: 'Locate Turn 2 · Decode', exact: true}).click();
  await expect(stage).toBeVisible();
});

test('metric request failure does not leak into basic conversation, stage or input-token info', async ({
  page,
}) => {
  test.skip(!process.env['E2E_WORKSPACE']?.includes('analysis-review'));
  await page.route('**/api/token-analysis*', (route) =>
    route.fulfill({
      status: 500,
      contentType: 'application/json',
      body: JSON.stringify({error: 'Metric request failed'}),
    }),
  );
  await page.goto('/');
  await page.getByRole('button', {name: /^Open Analysis review/}).click();
  const failed = page.waitForResponse(
    (r) => r.url().includes('/api/token-analysis') && r.status() === 500,
  );
  await page.locator('app-toolbar').getByRole('button', {name: 'Debug', exact: true}).click();
  await failed;
  const details = page.getByRole('complementary', {name: 'Comparison details'});
  await expect(details).not.toContainText('Token metrics could not be loaded.');
  await page.getByRole('group', {name: 'Turn 1 / Prefill · Target', exact: true}).press('Enter');
  await expect(details.getByRole('region', {name: 'Stage Info', exact: true})).toBeVisible();
  await expect(details).not.toContainText('Token metrics could not be loaded.');
  await page.getByRole('button', {name: 'target input position 0: <bos>', exact: true}).click();
  await expect(details.getByRole('region', {name: 'Token Info', exact: true})).toBeVisible();
  await expect(details).not.toContainText('Token metrics could not be loaded.');
});

test('token analyses locate exact instances and preserve selection on close and delete', async ({
  page,
}) => {
  test.skip(!process.env['E2E_WORKSPACE']?.includes('analysis-review'));
  await page.goto('/');
  await page.getByRole('button', {name: /^Open Analysis review/}).click();
  await page.locator('app-toolbar').getByRole('button', {name: 'Debug', exact: true}).click();
  const token = page.getByRole('button', {name: 'target input position 0: <bos>', exact: true});
  const details = page.getByRole('complementary', {name: 'Comparison details'});
  const info = details.getByRole('region', {name: 'Token Info', exact: true});
  const analyses = details.getByRole('region', {name: 'Analyses', exact: true});
  await token.click();
  await expect(info).toContainText('Token ID');
  await expect(details.getByRole('region', {name: 'Actions', exact: true})).toHaveCount(0);
  const before = await info.innerText();
  await analyses.getByRole('button', {name: 'Add analysis view', exact: true}).click();
  await page.getByRole('button', {name: 'Cancel', exact: true}).click();
  await expect(info).toHaveText(before, {useInnerText: true});
  for (let i = 0; i < 2; i++) {
    await analyses.getByRole('button', {name: 'Add analysis view', exact: true}).click();
    await page.getByRole('button', {name: 'Analysis type', exact: true}).click();
    await page.getByRole('option', {name: 'Activations Available', exact: true}).click();
    await page.getByRole('button', {name: 'Add', exact: true}).click();
    await expect(info).toHaveCount(0);
    await token.click();
  }
  const first = page.getByRole('button', {name: 'Activations · 1', exact: true});
  const second = page.getByRole('button', {name: 'Activations · 2', exact: true});
  await first.click();
  await second.click();
  await expect(info).toHaveText(before, {useInnerText: true});
  await analyses
    .getByRole('button', {name: 'Locate Activations · 2 analysis', exact: true})
    .click();
  await expect(second).toHaveAttribute('aria-expanded', 'true');
  await expect(first).toHaveAttribute('aria-expanded', 'false');
  await expect(info).toHaveText(before, {useInnerText: true});
  await page.getByRole('button', {name: 'Remove Activations · 1 analysis', exact: true}).click();
  await expect(second).toHaveText(/Activations · 2/);
  await expect(info).toHaveText(before, {useInnerText: true});
  await page.getByRole('button', {name: 'Remove Activations · 2 analysis', exact: true}).click();
  await expect(info).toHaveText(before, {useInnerText: true});
  // Closing the current, newly added analysis returns to the original single-token selection.
  await analyses.getByRole('button', {name: 'Add analysis view', exact: true}).click();
  await page.getByRole('button', {name: 'Analysis type', exact: true}).click();
  await page.getByRole('option', {name: 'Activations Available', exact: true}).click();
  await page.getByRole('button', {name: 'Add', exact: true}).click();
  await page.getByRole('button', {name: 'Activations', exact: true}).click();
  await expect(info).toHaveText(before, {useInnerText: true});
});

test('background restores Conversation Info and stage locator aligns at the top', async ({
  page,
}) => {
  test.skip(!process.env['E2E_WORKSPACE']?.includes('analysis-review'));
  await page.goto('/');
  await page.getByRole('button', {name: /^Open Analysis review/}).click();
  await page.locator('app-toolbar').getByRole('button', {name: 'Debug', exact: true}).click();
  const details = page.getByRole('complementary', {name: 'Comparison details'});
  await page.getByRole('group', {name: 'Turn 1 / Prefill · Target', exact: true}).press('Enter');
  await expect(details.getByRole('region', {name: 'Stage Info', exact: true})).toBeVisible();
  await page.locator('.reading').click({position: {x: 2, y: 100}});
  const conversation = details.getByRole('region', {name: 'Conversation Info', exact: true});
  await expect(conversation).toBeVisible();
  await details
    .getByRole('region', {name: 'Stages', exact: true})
    .getByRole('button', {name: 'Locate Turn 2 · Prefill', exact: true})
    .click();
  await expect(conversation).toBeVisible();
  await expect
    .poll(() =>
      page.locator('.reading').evaluate((el) => {
        const turn = el.querySelector('[data-node-turn="2"][data-node-stage="prefill"]')!;
        const heading = el.querySelector('.column-headings')!;
        const gap =
          turn.getBoundingClientRect().top -
          el.getBoundingClientRect().top -
          heading.getBoundingClientRect().height -
          8;
        const remaining = el.scrollHeight - el.clientHeight - el.scrollTop;
        return Math.min(Math.abs(gap), Math.abs(remaining));
      }),
    )
    .toBeLessThan(2);
});

test('empty space beside Add analysis restores Conversation Info', async ({page}) => {
  test.skip(!process.env['E2E_WORKSPACE']?.includes('analysis-review'));
  await page.goto('/');
  await page.getByRole('button', {name: /^Open Analysis review/}).click();
  await page.locator('app-toolbar').getByRole('button', {name: 'Debug', exact: true}).click();
  const details = page.getByRole('complementary', {name: 'Comparison details'});
  await page.getByRole('group', {name: 'Turn 1 / Prefill · Target', exact: true}).press('Enter');
  await expect(details.getByRole('region', {name: 'Stage Info', exact: true})).toBeVisible();
  const row = page.locator('.stage-analysis-row').first();
  const box = await row.boundingBox();
  await row.click({position: {x: box!.width / 4, y: 8}});
  await expect(details.getByRole('region', {name: 'Conversation Info', exact: true})).toBeVisible();
});
