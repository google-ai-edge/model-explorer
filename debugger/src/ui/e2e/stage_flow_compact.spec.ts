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
test('history is inline and compact stages retain clickable outlined head and tail tokens', async ({
  page,
}) => {
  test.skip(!process.env['E2E_WORKSPACE']?.includes('analysis-review'));
  await page.goto('/');
  await page.getByRole('button', {name: /^Open Analysis review/}).click();
  await page.locator('app-toolbar').getByRole('button', {name: 'Debug', exact: true}).click();
  await page.getByRole('button', {name: 'Display', exact: true}).click();
  await expect(page.getByLabel('Token outlines', {exact: true})).toBeChecked();
  await page.keyboard.press('Escape');
  for (const phase of ['Prefill', 'Decode']) {
    const stage = page.getByRole('group', {name: `Turn 2 / ${phase} · Target`, exact: true});
    if (phase === 'Prefill') {
      const title = await stage.locator('.nodeTitleLabel').boundingBox();
      const history = await stage.getByRole('status').boundingBox();
      expect(Math.abs(title!.y - history!.y)).toBeLessThan(10);
      expect(history!.x).toBeGreaterThan(title!.x);
    }
    await stage.getByRole('button', {name: 'Collapse node', exact: true}).click();
    const preview = stage.getByLabel(`Compact ${phase} preview · target · Turn 2`, {exact: true});
    const tokens = preview.locator('.nodePreviewToken');
    await expect.poll(() => tokens.count()).toBeGreaterThanOrEqual(2);
    expect(await tokens.count()).toBeLessThanOrEqual(12);
    await expect(preview.locator('.nodePreviewEllipsis')).toHaveCount(1);
    await expect(tokens.last()).toBeVisible();
    expect(await tokens.last().evaluate((el) => getComputedStyle(el).borderTopStyle)).toBe('solid');
    await tokens.last().click();
    await expect(stage.getByRole('button', {name: 'Expand node', exact: true})).toBeVisible();
    await expect(
      page
        .getByRole('complementary', {name: 'Comparison details'})
        .getByRole('region', {name: 'Token Info', exact: true}),
    ).toBeVisible();
    await expect(tokens.last()).toHaveAttribute('aria-pressed', 'true');
  }
});

test('compact previews share aligned slots and full-token hover and selection feedback', async ({
  page,
}) => {
  await page.setViewportSize({width: 2560, height: 1000});
  test.skip(!process.env['E2E_WORKSPACE']?.includes('analysis-review'));
  await page.goto('/');
  await page.getByRole('button', {name: /^Open Analysis review/}).click();
  await page.locator('app-toolbar').getByRole('button', {name: 'Debug', exact: true}).click();
  const target = page.getByRole('group', {name: 'Turn 2 / Decode · Target', exact: true});
  const ref = page.getByRole('group', {name: 'Turn 2 / Decode · Reference', exact: true});
  await target.getByRole('button', {name: 'Collapse node', exact: true}).click();
  await expect(ref.locator('.nodeContextCount')).toHaveText('32 tokens · 3 gaps');
  await expect(target.locator('.nodeContextCount')).toHaveText('24 tokens · 11 gaps');
  const targets = target.locator('.nodePreviewToken');
  const refs = ref.locator('.nodePreviewToken');
  const indices = await targets.evaluateAll((nodes) =>
    nodes.map((n) => n.getAttribute('data-row')),
  );
  expect(await refs.evaluateAll((nodes) => nodes.map((n) => n.getAttribute('data-row')))).toEqual(
    indices,
  );
  const targetBox = (await target.boundingBox())!;
  const refBox = (await ref.boundingBox())!;
  for (let i = 0; i < indices.length; i++) {
    const a = (await targets.nth(i).boundingBox())!,
      b = (await refs.nth(i).boundingBox())!;
    expect(Math.abs(a.x - targetBox.x - (b.x - refBox.x))).toBeLessThan(2);
    expect(Math.abs(a.width - b.width)).toBeLessThan(2);
  }
  await expect(target.locator('.nodePreviewToken.missing')).not.toHaveCount(0);
  await targets.first().hover();
  await expect(targets.first()).toHaveClass(/hover-origin/);
  await expect(refs.first()).toHaveClass(/pair-hover/);
  expect(await refs.first().evaluate((el) => getComputedStyle(el).outlineStyle)).toBe('dashed');
  await targets.first().click();
  await expect(targets.first()).toHaveClass(/selection-origin/);
  await expect(refs.first()).toHaveClass(/linked-token-selected/);
  expect(await targets.first().evaluate((el) => getComputedStyle(el).outlineStyle)).toBe('solid');
  expect(await refs.first().evaluate((el) => getComputedStyle(el).outlineStyle)).toBe('dashed');
});

test('compact previews fit complete paired tokens after resizing', async ({page}) => {
  test.skip(!process.env['E2E_WORKSPACE']?.includes('analysis-review'));
  await page.goto('/');
  await page.getByRole('button', {name: /^Open Analysis review/}).click();
  await page.locator('app-toolbar').getByRole('button', {name: 'Debug', exact: true}).click();
  const target = page.getByRole('group', {name: 'Turn 2 / Decode · Target', exact: true});
  const ref = page.getByRole('group', {name: 'Turn 2 / Decode · Reference', exact: true});
  await target.getByRole('button', {name: 'Collapse node', exact: true}).click();
  let wideCount = 0;
  for (const width of [2560, 1440]) {
    await page.setViewportSize({width, height: 1000});
    await expect
      .poll(() =>
        target.locator('.nodePreviewTokens').evaluate((el) => {
          const parent = el.parentElement!;
          return parent.scrollWidth - parent.clientWidth;
        }),
      )
      .toBeLessThanOrEqual(1);
    const count = await target.locator('.nodePreviewToken').count();
    if (width === 2560) wideCount = count;
    else expect(count).toBeLessThan(wideCount);
    const indices = (side: typeof target) =>
      side
        .locator('.nodePreviewToken')
        .evaluateAll((nodes) => nodes.map((n) => n.getAttribute('data-row')));
    expect(await indices(target)).toEqual(await indices(ref));
    for (const side of [ref, target]) {
      expect(
        await side
          .locator('.nodePreviewToken')
          .evaluateAll((nodes) => nodes.every((n) => n.scrollWidth <= n.clientWidth + 1)),
      ).toBe(true);
    }
  }
});

test('compact tokens follow global Display outlines and coloring', async ({page}) => {
  test.skip(!process.env['E2E_WORKSPACE']?.includes('analysis-review'));
  await page.setViewportSize({width: 2560, height: 1000});
  await page.goto('/');
  await page.getByRole('button', {name: /^Open Analysis review/}).click();
  await page.locator('app-toolbar').getByRole('button', {name: 'Debug', exact: true}).click();
  const stage = page.getByRole('group', {name: 'Turn 2 / Decode · Target', exact: true});
  await stage.getByRole('button', {name: 'Collapse node', exact: true}).click();
  const token = stage.locator('.nodePreviewToken').first();
  const gap = stage.locator('.nodePreviewToken.missing').first();
  const gapBackground = () => gap.evaluate((el) => getComputedStyle(el).backgroundColor);
  await expect.poll(gapBackground).not.toBe('rgba(0, 0, 0, 0)');
  const matchBackground = await gapBackground();

  await page.getByRole('button', {name: 'Display', exact: true}).click();
  await page.getByLabel('Token outlines', {exact: true}).uncheck();
  await page.getByRole('button', {name: 'Color by', exact: true}).click();
  await page.getByRole('option', {name: '-', exact: true}).click();
  await page.getByRole('button', {name: 'Apply', exact: true}).click();
  await page.getByRole('button', {name: 'Close display settings', exact: true}).click();
  await expect(token).toHaveCSS('border-top-style', 'none');
  await expect.poll(gapBackground).toBe('rgba(0, 0, 0, 0)');
  await expect(stage.locator('.nodePreviewToken.mismatch')).toHaveCount(0);
  await expect(stage.locator('.nodePreviewToken.numeric-token')).toHaveCount(0);
  await page.getByRole('button', {name: 'Display', exact: true}).click();
  await page.getByLabel('Token outlines', {exact: true}).focus();
  await page.keyboard.press('Space');
  await page.getByRole('button', {name: 'Color by', exact: true}).click();
  await page.getByRole('option', {name: 'Activation · Cosine similarity', exact: true}).click();
  await page.getByRole('button', {name: 'Apply', exact: true}).click();
  await page.getByRole('button', {name: 'Close display settings', exact: true}).click();
  await expect(token).toHaveCSS('border-top-style', 'solid');
  await expect.poll(gapBackground).not.toBe('rgba(0, 0, 0, 0)');
  expect(await gapBackground()).not.toBe(matchBackground);
  await expect(gap).toHaveCSS('background-image', /repeating-linear-gradient/);
  await expect(
    stage.locator('.nodePreviewToken.numeric-token, .nodePreviewToken.numeric-missing'),
  ).not.toHaveCount(0);
  await expect(stage.locator('.nodePreviewToken.mismatch')).toHaveCount(0);
});
