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

import {compileMetricQuery, type MetricQueryLanguage} from './metric_query';

describe('compileMetricQuery', () => {
  const ratio: MetricQueryLanguage = {
    fields: {
      a: {type: 'number'},
      b: {type: 'number'},
      b_alias: {type: 'number', key: 'b'},
      ok: {type: 'boolean'},
    },
    percentField: 'a',
    percentScale: 0.01,
  };

  const percent: MetricQueryLanguage = {...ratio, percentScale: 1};

  const run = (formula: string, row: object, language = ratio) =>
    compileMetricQuery(formula, language)(row);

  it('precedence, grouping, aliases and case-insensitive keywords', () => {
    const row = {a: 0.12, b: 0.5, ok: true};
    expect(run('a > .1 OR b < 0 AND a < 0', row)).toBe(true);
    expect(run('(a > .1 OR b < 0) AND a < 0', row)).toBe(false);
    expect(run('nOt b < 0 aNd a >= 0', row)).toBe(true);
    expect(run('b_alias = b', row)).toBe(true);
    expect(run('ok AND NOT ok = false', row)).toBe(true);
  });

  it('percent literals scale per language and are tied to the percent field', () => {
    expect(run('a > 10%', {a: 0.12})).toBe(true);
    expect(run('a > 12%', {a: 0.12})).toBe(false);
    expect(run('a > 10%', {a: 0.12}, percent)).toBe(false);
    expect(run('a > 0.1%', {a: 0.12}, percent)).toBe(true);
    expect(run('10% < a', {a: 0.5})).toBe(true);
    for (const formula of ['b > 10%', '10% > .1'])
      expect(() => compileMetricQuery(formula, ratio)).toThrow(/% is only valid with a/);
  });

  it('unknown metrics propagate as unknown through NOT, AND and OR', () => {
    for (const unknown of [null, undefined, NaN, Infinity, -Infinity]) {
      const row = {a: unknown, b: 0.5, ok: null};
      for (const formula of [
        'a > 0',
        'a != 0',
        'NOT a > 0',
        'NOT NOT a > 0',
        'a > 0 AND true',
        'a > 0 OR false',
        'ok',
        'NOT ok',
      ])
        expect(run(formula, row), `${formula}, unknown=${unknown}`).toBe(null);
      expect(run('a > 0 OR b > 0', row)).toBe(true);
      expect(run('NOT (a > 0 AND b < 0)', row)).toBe(true);
      expect(run('NOT (a > 0 OR b > 0)', row)).toBe(false);
    }
  });

  it('requiresMetrics flags numeric fields only; a blank formula matches everything', () => {
    expect(compileMetricQuery('NOT ok', ratio).requiresMetrics).toBe(false);
    expect(compileMetricQuery('ok AND a > 1', ratio).requiresMetrics).toBe(true);
    const blank = compileMetricQuery(' \n\t', ratio);
    expect(blank({})).toBe(true);
    expect(blank.requiresMetrics).toBe(false);
  });

  it('helpful errors, type checks, injection rejection and size bounds', () => {
    const cases: [string, RegExp][] = [
      ['a >', /Expected a metric or value at column 4/],
      ['(a > .1', /Expected closing parenthesis/],
      ['a', /Add a comparison after a/],
      ['unknown > .1', /Unknown metric: unknown/],
      ['a > 1e999', /Number must be finite/],
      ['a == true', /same type/],
      ['true > false', /Use =, == or !=/],
      ['ok > true', /Use =, == or !=/],
      ['a > 0 > 1', /Unexpected term/],
      ['a > .1 && true', /Unexpected character/],
      ['a > .1; globalThis.attacked = true', /Unexpected character/],
      ['constructor = 0', /Unknown metric/],
      ['__proto__ = 0', /Unknown metric/],
      ['hasOwnProperty = 0', /Unknown metric/],
      ['globalThis.alert(1)', /Unexpected character/],
      ['b[0] > .1', /Unexpected character/],
      ['b > "0"', /Unexpected character/],
    ];
    for (const [formula, message] of cases)
      expect(() => compileMetricQuery(formula, ratio), formula).toThrow(message);
    expect(() => compileMetricQuery(' '.repeat(1001), ratio)).toThrow(/1,000 characters/);
    expect(() => compileMetricQuery(Array(51).fill('1=1').join(' OR '), ratio)).toThrow(
      /200 terms/,
    );
    expect(() => compileMetricQuery('('.repeat(33) + 'true' + ')'.repeat(33), ratio)).toThrow(
      /32 levels/,
    );
    expect(() => compileMetricQuery('NOT '.repeat(33) + 'true', ratio)).toThrow(/32 levels/);
    expect(run('('.repeat(32) + 'true' + ')'.repeat(32), {})).toBe(true);
    expect((globalThis as {attacked?: unknown}).attacked).toBe(undefined);
  });
});
