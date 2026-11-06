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

import {TestBed} from '@angular/core/testing';
import {
  METRIC_CATALOG,
  METRIC_CATALOG_RETRIES,
  METRIC_CATALOG_SLOW_RETRY,
  MetricCatalogStore,
  MetricLookup,
} from './metric_catalog';
import {MetricsApi} from './metrics_api';
import {REASON_CODES} from './reasons';
import {METRIC_CATALOG_FIXTURE} from './testing/metric_catalog_fixture';

/** A store whose requests answer, in turn, as `answers` says: an Error fails one. */
function store(answers: Array<Error | 'catalog'>) {
  const asked: number[] = [];
  TestBed.configureTestingModule({
    providers: [
      // No preset: the store asks the server.
      {provide: METRIC_CATALOG, useValue: null},
      {
        provide: MetricsApi,
        useValue: {
          catalog: () => {
            asked.push(Date.now());
            const answer = answers.shift();
            return answer === 'catalog'
              ? Promise.resolve(METRIC_CATALOG_FIXTURE)
              : Promise.reject(answer);
          },
        },
      },
    ],
  });
  return {catalog: TestBed.inject(MetricCatalogStore), asked};
}

describe('reason codes', () => {
  it('decides only on codes the server lists', () => {
    const listed = new Set(METRIC_CATALOG_FIXTURE.reasons.map((reason) => reason.code));
    expect(Object.values(REASON_CODES).filter((code) => !listed.has(code))).toEqual([]);
  });

  it('gives the words of a code a response gives alone', () => {
    const lookup = new MetricLookup(METRIC_CATALOG_FIXTURE);
    expect(lookup.reason('undefined_zero_norm')).toBe(
      'Undefined because a required vector norm is zero.',
    );
    expect(lookup.reason('no_such_code')).toBe('');
  });
});

describe('metric catalog store', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('asks again after a failed request, waiting longer each time', async () => {
    const {catalog, asked} = store([new Error('offline'), new Error('offline'), 'catalog']);
    await vi.advanceTimersByTimeAsync(0);
    expect(asked.length).toBe(1);
    // While it asks again, a label is the bare id.
    expect(catalog.lookup().label('kl')).toBe('kl');
    await vi.advanceTimersByTimeAsync(METRIC_CATALOG_RETRIES[0]);
    expect(asked.length).toBe(2);
    await vi.advanceTimersByTimeAsync(METRIC_CATALOG_RETRIES[1]);
    expect(asked.length).toBe(3);
    expect(asked[2] - asked[1]).toBeGreaterThan(asked[1] - asked[0]);
    expect(catalog.lookup().label('kl')).toBe('Kullback–Leibler divergence');
    expect(catalog.error()).toBe('');
  });

  it('says why where a label falls back, once every request failed', async () => {
    // The first request and every one after it fail.
    const {catalog, asked} = store([0, ...METRIC_CATALOG_RETRIES].map(() => new Error('offline')));
    await vi.advanceTimersByTimeAsync(METRIC_CATALOG_RETRIES.reduce((a, b) => a + b, 0) + 1);
    expect(asked.length).toBe(METRIC_CATALOG_RETRIES.length + 1);
    expect(catalog.error()).toBe('metric descriptions could not be read: offline');
    expect(catalog.lookup().label('kl')).toBe(
      'kl (metric descriptions could not be read: offline)',
    );
  });

  it('keeps asking slowly once the first requests failed, and clears the error', async () => {
    // The first requests fail, then one slow one more; the next brings the catalog.
    const failures = [0, ...METRIC_CATALOG_RETRIES, 0].map(() => new Error('offline'));
    const {catalog, asked} = store([...failures, 'catalog']);
    await vi.advanceTimersByTimeAsync(METRIC_CATALOG_RETRIES.reduce((a, b) => a + b, 0) + 1);
    expect(asked.length).toBe(METRIC_CATALOG_RETRIES.length + 1);
    expect(catalog.error()).toBe('metric descriptions could not be read: offline');
    await vi.advanceTimersByTimeAsync(METRIC_CATALOG_SLOW_RETRY - 2);
    expect(asked.length).toBe(METRIC_CATALOG_RETRIES.length + 1);
    await vi.advanceTimersByTimeAsync(1);
    expect(asked.length).toBe(METRIC_CATALOG_RETRIES.length + 2);
    expect(catalog.error()).toBe('metric descriptions could not be read: offline');
    await vi.advanceTimersByTimeAsync(METRIC_CATALOG_SLOW_RETRY);
    expect(asked.length).toBe(METRIC_CATALOG_RETRIES.length + 3);
    expect(catalog.error()).toBe('');
    expect(catalog.lookup().label('kl')).toBe('Kullback–Leibler divergence');
  });

  it('cancels pending retry timers when destroyed', async () => {
    const {asked} = store([new Error('offline'), 'catalog']);
    await vi.advanceTimersByTimeAsync(0);
    expect(asked.length).toBe(1);
    TestBed.resetTestingModule();
    await vi.advanceTimersByTimeAsync(METRIC_CATALOG_SLOW_RETRY);
    expect(asked.length).toBe(1);
  });
});
