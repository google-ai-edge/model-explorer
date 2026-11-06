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

import {MetricLookup} from '../server/metric_catalog';
import {METRIC_CATALOG_FIXTURE} from '../server/testing/metric_catalog_fixture';
import {
  compileQuery as compileWith,
  compileQueryOrNothing,
  compilesWithSomeCatalog,
  queryFields,
} from './token_query';

describe('token queries', () => {
  // The server's catalog, looked up by MetricLookup.
  const catalog = new MetricLookup(METRIC_CATALOG_FIXTURE);

  const compileQuery = (source: string) => compileWith(source, catalog);

  it('unknown propagation, types, percent-as-percent and invalid syntax', () => {
    expect(compileQuery('NOT token_match')({token_match: null})).toBe(null);
    expect(compileQuery('NOT token_match')({token_match: false})).toBe(true);
    expect(compileQuery('kl > 0.005 AND relative_l2 > 0.4%')({kl: 0.01, relative_l2: 0.5})).toBe(
      true,
    );
    expect(compileQuery('NOT (kl > 0.005)')({kl: null})).toBe(null);
    expect(() => compileQuery('window.alert(1)')).toThrow();
    expect(() => compileQuery('token_match > true')).toThrow();
  });

  it('% goes with the metric the catalog says is a percentage, and none without it', () => {
    expect(compileQuery('relative_l2 <= 0.4%')({relative_l2: 0.4})).toBe(true);
    expect(() => compileQuery('kl > 1%')).toThrow(/% is only valid with relative_l2/);
    expect(compileWith('kl > 0.005', null)({kl: 0.01})).toBe(true);
    expect(() => compileWith('relative_l2 > 0.4%', null)).toThrow(/no metric is known/);
  });

  it('a formula that needs the catalog finds nothing until it has loaded', () => {
    const before = compileQueryOrNothing('relative_l2 > 0.4%', null);
    expect(before({relative_l2: 1, token_match: false})).toBe(null);
    expect(before.requiresMetrics).toBe(false);
    const after = compileQueryOrNothing('relative_l2 > 0.4%', catalog);
    expect(after({relative_l2: 1})).toBe(true);
    expect(after.requiresMetrics).toBe(true);
    expect(compilesWithSomeCatalog('relative_l2 > 0.4%')).toBe(true);
    expect(compilesWithSomeCatalog('kl >')).toBe(false);
    expect(compilesWithSomeCatalog(' ')).toBe(false);
  });

  it('requiresMetrics and the blank-formula message', () => {
    expect(compileQuery('NOT token_match').requiresMetrics).toBe(false);
    expect(compileQuery('token_match OR js > 1').requiresMetrics).toBe(true);
    expect(() => compileQuery('  ')).toThrow(/reset to NOT token_match/);
    expect(Object.keys(queryFields)[0]).toStrictEqual('token_match');
  });
});
