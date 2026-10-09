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

"""Dependency-free JSON Schema validation for the shared wire documents.

The schemas under ``schemas/`` are the contract; this module implements the
subset of JSON Schema 2020-12 they use: type, const, enum, required, properties,
additionalProperties, items, minItems/maxItems, minLength/maxLength,
minimum/maximum, pattern, anyOf and $ref (to ``#/$defs/...`` or another schema
by id).
"""

from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any

SCHEMA_DIR = Path(__file__).resolve().parent / 'schemas'
SCHEMAS = ('capture-job', 'tap-manifest', 'capture-index-v2', 'runner-wire')
_TYPES = {
    'object': lambda v: isinstance(v, dict),
    'array': lambda v: isinstance(v, list),
    'string': lambda v: isinstance(v, str),
    'integer': lambda v: isinstance(v, int) and not isinstance(v, bool),
    'number': lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    'boolean': lambda v: isinstance(v, bool),
    'null': lambda v: v is None,
}


class SchemaError(ValueError):
  """The document violates its schema; ``path`` names the offending member."""

  def __init__(self, message, path):
    self.path = path or '$'
    super().__init__(f'{self.path}: {message}')


@lru_cache(maxsize=None)
def load_schema(name: str) -> dict[str, Any]:
  if name not in SCHEMAS:
    raise KeyError(f'Unknown schema: {name}')
  return json.loads((SCHEMA_DIR / f'{name}.schema.json').read_text())


def validate(name: str, instance: Any) -> Any:
  """Raises SchemaError for the first violation of schema ``name``.

  Otherwise, returns the instance.
  """
  schema = load_schema(name)
  _check(instance, schema, schema, '$')
  return instance


def validate_wire(sender: str, message: Any) -> Any:
  """Checks one Server ↔ Runner message.

  ``sender`` is ``'server'`` or ``'runner'``.
  """
  schema = load_schema('runner-wire')
  kind = message.get('type') if isinstance(message, dict) else None
  definition = schema['$defs'].get(f'{sender}.{kind}')
  if definition is None:
    raise SchemaError(f'unknown {sender} message {kind!r}', '$.type')
  _check(message, definition, schema, '$')
  return message


def _resolve(ref, root):
  if ref.startswith('#/'):
    node = root
    for part in ref[2:].split('/'):
      node = node[part]
    return node, root
  target = load_schema(ref)
  return target, target


def _check(value, schema, root, path):
  if '$ref' in schema:
    target, target_root = _resolve(schema['$ref'], root)
    _check(value, target, target_root, path)
  if 'const' in schema and value != schema['const']:
    raise SchemaError(f'expected {schema["const"]!r}, got {value!r}', path)
  if 'enum' in schema and value not in schema['enum']:
    raise SchemaError(
        f'expected one of {schema["enum"]!r}, got {value!r}', path
    )
  if 'type' in schema:
    types = (
        schema['type'] if isinstance(schema['type'], list) else [schema['type']]
    )
    if not any(_TYPES[kind](value) for kind in types):
      raise SchemaError(
          f'expected {" or ".join(types)}, got {type(value).__name__}', path
      )
  if 'anyOf' in schema:
    errors = []
    for option in schema['anyOf']:
      try:
        _check(value, option, root, path)
        break
      except SchemaError as error:
        errors.append(str(error))
    else:
      raise SchemaError(
          'matches none of the alternatives: ' + '; '.join(errors), path
      )
  if isinstance(value, dict):
    for key in schema.get('required', ()):
      if key not in value:
        raise SchemaError(f'missing required member {key!r}', path)
    properties = schema.get('properties', {})
    for key, child in value.items():
      if key in properties:
        _check(child, properties[key], root, f'{path}.{key}')
      elif schema.get('additionalProperties') is False:
        raise SchemaError(f'unexpected member {key!r}', path)
      elif isinstance(schema.get('additionalProperties'), dict):
        _check(child, schema['additionalProperties'], root, f'{path}.{key}')
  if isinstance(value, list):
    if 'minItems' in schema and len(value) < schema['minItems']:
      raise SchemaError(
          f'expected at least {schema["minItems"]} items, got {len(value)}',
          path,
      )
    if 'maxItems' in schema and len(value) > schema['maxItems']:
      raise SchemaError(
          f'expected at most {schema["maxItems"]} items, got {len(value)}', path
      )
    if 'items' in schema:
      for index, item in enumerate(value):
        _check(item, schema['items'], root, f'{path}[{index}]')
  if isinstance(value, str):
    if 'minLength' in schema and len(value) < schema['minLength']:
      raise SchemaError(
          f'expected at least {schema["minLength"]} characters', path
      )
    if 'maxLength' in schema and len(value) > schema['maxLength']:
      raise SchemaError(
          f'expected at most {schema["maxLength"]} characters', path
      )
    if 'pattern' in schema and not re.search(schema['pattern'], value):
      raise SchemaError(f'does not match {schema["pattern"]!r}', path)
  if isinstance(value, (int, float)) and not isinstance(value, bool):
    if 'minimum' in schema and value < schema['minimum']:
      raise SchemaError(
          f'expected at least {schema["minimum"]}, got {value}', path
      )
    if 'maximum' in schema and value > schema['maximum']:
      raise SchemaError(
          f'expected at most {schema["maximum"]}, got {value}', path
      )
