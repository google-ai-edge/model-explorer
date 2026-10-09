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

/**
 * The find-formula language shared by token analysis and the KV explorer.
 *
 * Grammar: NOT binds before AND, which binds before OR; parentheses group. A
 * comparison joins two scalars of the same type. Missing or non-finite metrics make a
 * comparison unknown; unknown stays unknown under NOT, and AND/OR follow three-valued
 * logic. Nothing is evaluated as JavaScript: the source is tokenised and compiled
 * into closures.
 */
export type Truth = boolean | null;
export interface MetricQueryField {
  type: 'number' | 'boolean';
  /** Row property to read; an alias names another field's property. */
  key?: string;
}
export interface MetricQueryLanguage {
  fields: Record<string, MetricQueryField>;
  /** The one numeric field a `%` literal may be compared with. */
  percentField: string;
  /** What a `%` literal is multiplied by: 1 when the field is already a percentage, 0.01 for a ratio. */
  percentScale: number;
}
export interface MetricQuery<Row> {
  (row: Row): Truth;
  /** True when a numeric field is referenced, so per-row metrics must be loaded first. */
  requiresMetrics: boolean;
}
export const METRIC_QUERY_LIMITS = {
  characters: 1000,
  terms: 200,
  depth: 32,
} as const;

type Comparison = '>' | '<' | '>=' | '<=' | '=' | '==' | '!=';
interface Token {
  text: string;
  column: number;
}
interface Scalar<Row> {
  type: 'number' | 'boolean';
  field?: string;
  percent?: boolean;
  read: (row: Row) => number | boolean | null;
}
const COMPARISONS = new Set<string>(['>', '<', '>=', '<=', '=', '==', '!=']);
const TOKEN =
  />=|<=|!=|==|>|<|=|\(|\)|-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?%?|[A-Za-z_][A-Za-z_0-9]*/iy;

/** Splits a formula into terms; a blank formula has no terms. */
export function tokenizeMetricQuery(source: string): Token[] {
  if (typeof source !== 'string') throw new Error('Enter a formula as text');
  if (source.length > METRIC_QUERY_LIMITS.characters)
    throw new Error('Formula is limited to 1,000 characters');
  const tokens: Token[] = [];
  let offset = 0;
  while (offset < source.length) {
    if (/\s/.test(source[offset])) {
      offset++;
      continue;
    }
    TOKEN.lastIndex = offset;
    const match = TOKEN.exec(source);
    if (!match) throw new Error(`Unexpected character at column ${offset + 1}`);
    tokens.push({text: match[0], column: offset + 1});
    offset = TOKEN.lastIndex;
    if (tokens.length > METRIC_QUERY_LIMITS.terms)
      throw new Error('Formula is limited to 200 terms');
  }
  return tokens;
}

export function compileMetricQuery<Row extends object>(
  source: string,
  language: MetricQueryLanguage,
): MetricQuery<Row> {
  const tokens = tokenizeMetricQuery(source);
  const requiresMetrics = tokens.some(
    (token) => language.fields[token.text]?.type === 'number',
  );
  if (tokens.length === 0)
    return Object.assign(() => true as Truth, {requiresMetrics});

  let cursor = 0;
  const peek = () => tokens[cursor]?.text;
  const take = () => tokens[cursor++];
  const keyword = (word: string) => peek()?.toUpperCase() === word;
  const fail = (message: string, token = tokens[cursor]): never => {
    throw new Error(
      `${message} at column ${token?.column ?? source.length + 1}`,
    );
  };
  const checkDepth = (depth: number) => {
    if (depth > METRIC_QUERY_LIMITS.depth)
      fail('Formula nesting is limited to 32 levels');
  };

  function scalar(): Scalar<Row> {
    const token = tokens[cursor];
    if (!token) return fail('Expected a metric or value');
    take();
    const field = Object.hasOwn(language.fields, token.text)
      ? language.fields[token.text]
      : null;
    if (field) {
      const key = (field.key ?? token.text) as keyof Row;
      const name = field.key ?? token.text;
      return field.type === 'number'
        ? {
            type: 'number',
            field: name,
            read: (row) =>
              Number.isFinite(row?.[key]) ? (row[key] as number) : null,
          }
        : {
            type: 'boolean',
            field: name,
            read: (row) =>
              typeof row?.[key] === 'boolean' ? (row[key] as boolean) : null,
          };
    }
    if (/^-?(?:\d|\.)/.test(token.text)) {
      const percent = token.text.endsWith('%');
      const literal = Number(percent ? token.text.slice(0, -1) : token.text);
      if (!Number.isFinite(literal))
        return fail('Number must be finite', token);
      const value = percent ? literal * language.percentScale : literal;
      return {type: 'number', percent, read: () => value};
    }
    if (/^(true|false)$/i.test(token.text))
      return {type: 'boolean', read: () => token.text.toLowerCase() === 'true'};
    if (/^[A-Za-z_]/.test(token.text))
      return fail(`Unknown metric: ${token.text}`, token);
    return fail('Expected a metric or value', token);
  }

  function atom(depth: number): (row: Row) => Truth {
    checkDepth(depth);
    if (peek() === '(') {
      take();
      const child = or(depth + 1);
      if (peek() !== ')') return fail('Expected closing parenthesis');
      take();
      return child;
    }
    const left = scalar();
    if (!COMPARISONS.has(peek() ?? '')) {
      if (left.type !== 'boolean')
        return fail(`Add a comparison after ${left.field ?? 'value'}`);
      return (row) => left.read(row) as Truth;
    }
    const operatorToken = take();
    const operator = operatorToken.text as Comparison;
    const right = scalar();
    if (left.type !== right.type)
      return fail('Compare values of the same type', operatorToken);
    if (left.type === 'boolean' && !['=', '==', '!='].includes(operator))
      return fail('Use =, == or != for booleans', operatorToken);
    if (
      (left.percent && right.field !== language.percentField) ||
      (right.percent && left.field !== language.percentField)
    )
      return fail(
        `% is only valid with ${language.percentField}`,
        operatorToken,
      );
    return (row) => {
      const a = left.read(row),
        b = right.read(row);
      if (a === null || b === null) return null;
      switch (operator) {
        case '>':
          return a > b;
        case '<':
          return a < b;
        case '>=':
          return a >= b;
        case '<=':
          return a <= b;
        case '=':
        case '==':
          return a === b;
        case '!=':
          return a !== b;
      }
    };
  }

  function not(depth: number): (row: Row) => Truth {
    checkDepth(depth);
    if (!keyword('NOT')) return atom(depth);
    take();
    const child = not(depth + 1);
    return (row) => {
      const value = child(row);
      return value === null ? null : !value;
    };
  }

  function and(depth: number): (row: Row) => Truth {
    let node = not(depth);
    while (keyword('AND')) {
      take();
      const left = node,
        right = not(depth);
      node = (row) => {
        const a = left(row);
        if (a === false) return false;
        const b = right(row);
        return b === false ? false : a === null || b === null ? null : true;
      };
    }
    return node;
  }

  function or(depth: number): (row: Row) => Truth {
    let node = and(depth);
    while (keyword('OR')) {
      take();
      const left = node,
        right = and(depth);
      node = (row) => {
        const a = left(row);
        if (a === true) return true;
        const b = right(row);
        return b === true ? true : a === null || b === null ? null : false;
      };
    }
    return node;
  }

  const root = or(0);
  if (cursor !== tokens.length) fail(`Unexpected term: ${peek()}`);
  return Object.assign(root, {requiresMetrics});
}
