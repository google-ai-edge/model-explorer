# Copyright 2026 The AI Edge Model Explorer Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""The existing KV Find language, evaluated over bounded NumPy metric chunks."""

import math
import re

import numpy as np

FIELDS = {
    'relative_l2': 'relative_l2',
    'max_abs': 'max_abs',
    'max_abs_delta': 'max_abs',
    'cosine_distance': 'cosine_distance',
}
COMPARISONS = {'>', '<', '>=', '<=', '=', '==', '!='}
TOKEN = re.compile(
    r'>=|<=|!=|==|>|<|=|\(|\)|'
    r'-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?%?|'
    r'[A-Za-z_][A-Za-z_0-9]*',
    re.I,
)


def compile_kv_formula(expression):
  """Returns a safe predicate.

  A missing value stays unknown under NOT; only true matches.
  """
  if not isinstance(expression, str):
    raise ValueError('Enter a formula as text')
  if len(expression) > 1000:
    raise ValueError('Formula is limited to 1,000 characters')
  tokens, offset, cursor = [], 0, 0
  while offset < len(expression):
    if expression[offset].isspace():
      offset += 1
      continue
    match = TOKEN.match(expression, offset)
    if not match:
      raise ValueError(f'Unexpected character at column {offset + 1}')
    tokens.append((match[0], offset + 1))
    offset = match.end()
    if len(tokens) > 200:
      raise ValueError('Formula is limited to 200 terms')
  if not tokens:
    return lambda metrics: np.ones_like(metrics['relative_l2'], dtype=bool)

  def peek():
    return tokens[cursor][0] if cursor < len(tokens) else ''

  def take():
    nonlocal cursor
    value = tokens[cursor]
    cursor += 1
    return value

  def fail(message, column=None):
    column = column or (
        tokens[cursor][1] if cursor < len(tokens) else len(expression) + 1
    )
    raise ValueError(f'{message} at column {column}')

  def scalar():
    if not peek():
      fail('Expected a metric or value')
    value, column = take()
    if value in FIELDS:
      field = FIELDS[value]
      return 'number', field, False, lambda metrics: metrics[field]
    if re.match(r'^-?(?:\d|\.)', value):
      percent = value.endswith('%')
      number = float(value[:-1] if percent else value)
      if not math.isfinite(number):
        fail('Number must be finite', column)
      return (
          'number',
          None,
          percent,
          lambda metrics: number / 100 if percent else number,
      )
    if value.lower() in ('true', 'false'):
      return 'boolean', None, False, lambda metrics: value.lower() == 'true'
    fail(
        'Unknown metric: ' + value
        if re.match('[A-Za-z_]', value)
        else 'Expected a metric or value',
        column,
    )

  def atom(depth):
    if depth > 32:
      fail('Formula nesting is limited to 32 levels')
    if peek() == '(':
      take()
      child = parse_or(depth + 1)
      if peek() != ')':
        fail('Expected closing parenthesis')
      take()
      return child
    left = scalar()
    if peek() not in COMPARISONS:
      if left[0] != 'boolean':
        fail('Add a comparison after ' + (left[1] or 'value'))
      return lambda metrics: np.asarray(left[3](metrics), dtype=np.int8)
    operator, column = take()
    right = scalar()
    if left[0] != right[0]:
      fail('Compare values of the same type', column)
    if left[0] == 'boolean' and operator not in ('=', '==', '!='):
      fail('Use =, == or != for booleans', column)
    if (left[2] and right[1] != 'relative_l2') or (
        right[2] and left[1] != 'relative_l2'
    ):
      fail('% is only valid with relative_l2', column)
    operation = {
        '>': np.greater,
        '<': np.less,
        '>=': np.greater_equal,
        '<=': np.less_equal,
        '=': np.equal,
        '==': np.equal,
        '!=': np.not_equal,
    }[operator]

    def compare(metrics):
      a, b = left[3](metrics), right[3](metrics)
      return np.where(
          np.isfinite(a) & np.isfinite(b), operation(a, b).astype(np.int8), -1
      )

    return compare

  def parse_not(depth):
    if depth > 32:
      fail('Formula nesting is limited to 32 levels')
    if peek().upper() != 'NOT':
      return atom(depth)
    take()
    child = parse_not(depth + 1)

    def negate(metrics):
      value = child(metrics)
      return np.where(value < 0, -1, 1 - value)

    return negate

  def combine(left, right, conjunction):
    def result(metrics):
      a, b = left(metrics), right(metrics)
      if conjunction:
        return np.where(
            (a == 0) | (b == 0), 0, np.where((a < 0) | (b < 0), -1, 1)
        )
      return np.where(
          (a == 1) | (b == 1), 1, np.where((a < 0) | (b < 0), -1, 0)
      )

    return result

  def parse_and(depth):
    node = parse_not(depth)
    while peek().upper() == 'AND':
      take()
      node = combine(node, parse_not(depth), True)
    return node

  def parse_or(depth):
    node = parse_and(depth)
    while peek().upper() == 'OR':
      take()
      node = combine(node, parse_and(depth), False)
    return node

  root = parse_or(0)
  if cursor != len(tokens):
    fail('Unexpected term: ' + peek())
  return lambda metrics: np.broadcast_to(
      root(metrics) == 1, np.shape(metrics['relative_l2'])
  )
