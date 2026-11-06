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

"""Checks runtime options against the runtime's JSON Schema.

The rules of the LiteRT runner's ValidateOptions (options_schema.cc), as the
server's fake runner keeps them, so both runners refuse alike: the schema
must be JSON and use only the keywords checked here; then every number must
be finite; then the options must match the schema, the first mismatch in key
order answered.

Values are JSON as nlohmann::json holds them: a Struct's numbers are all
floats, a schema's whole numbers ints; objects are read in key order.
"""

import dataclasses
import decimal
import json
import math
from typing import Any

import grpc

# pyformat: disable
from google.protobuf import struct_pb2
# pyformat: enable

_INVALID = grpc.StatusCode.INVALID_ARGUMENT
_INTERNAL = grpc.StatusCode.INTERNAL


@dataclasses.dataclass(frozen=True)
class Refusal:
  """A status an RPC fails with.

  Attributes:
      code: The gRPC status code.
      message: What the status says.
  """

  code: grpc.StatusCode
  message: str


# The types the keyword "type" may name.
_JSON_TYPES = (
    'null',
    'boolean',
    'object',
    'array',
    'number',
    'string',
    'integer',
)


def validate_options(
    schema: str, options: struct_pb2.Struct, field: str
) -> Refusal | None:
  """Why runtime options are refused, or None if they are not.

  Args:
      schema: The runtime's JSON Schema for the options, as GetInfo has it.
      options: The options a request carries.
      field: The request's field, which names the options in messages, as
        'LoadRequest.runtime_options'.

  Returns:
      INTERNAL for a schema it cannot check, INVALID_ARGUMENT for options
      the schema does not allow, or None.
  """
  try:
    parsed = json.loads(schema, parse_constant=_no_constant)
  except ValueError:
    return Refusal(_INTERNAL, f"The runtime's schema for {field} is not JSON.")
  value = _struct_json(options)
  return (
      _check_keywords(parsed, '#')
      or _check_finite(value, field)
      or _check(parsed, value, field)
  )


def _no_constant(name: str) -> None:
  """Refuses NaN and Infinity, which JSON has not."""
  raise ValueError(f'{name} is not JSON.')


def _value_json(value: struct_pb2.Value) -> object:
  """A Value as JSON: None, bool, float, str, list or dict."""
  kind = value.WhichOneof('kind')
  if kind == 'bool_value':
    return value.bool_value
  if kind == 'number_value':
    return value.number_value
  if kind == 'string_value':
    return value.string_value
  if kind == 'struct_value':
    return _struct_json(value.struct_value)
  if kind == 'list_value':
    return [_value_json(item) for item in value.list_value.values]
  return None  # A null, or a value of no kind, which JSON writes as null.


def _struct_json(struct: struct_pb2.Struct) -> dict[str, object]:
  """A Struct as a JSON object."""
  return {name: _value_json(value) for name, value in struct.fields.items()}


def _is_number(value: object) -> bool:
  """Whether a JSON value is a number; a bool is not."""
  return isinstance(value, (int, float)) and not isinstance(value, bool)


def _show(value: object) -> str:
  """A JSON value as the LiteRT runner writes it in a message: compact."""
  if value is None:
    return 'null'
  if isinstance(value, bool):
    return 'true' if value else 'false'
  if isinstance(value, int):
    return str(value)
  if isinstance(value, float):
    return _show_float(value)
  if isinstance(value, str):
    return json.dumps(value, ensure_ascii=False)
  if isinstance(value, list):
    return '[' + ','.join(_show(item) for item in value) + ']'
  return (
      '{'
      + ','.join(f'{_show(k)}:{_show(v)}' for k, v in sorted(value.items()))
      + '}'
  )


def _show_float(number: float) -> str:
  """A finite float as nlohmann::json writes it.

  The shortest digits that read back as the number, fixed for a decimal
  point from 10^-4 to 10^15, else as an exponent; a whole number keeps
  '.0'. 2.0 is '2.0', 1e15 '1e+15', 1e-5 '1e-05'.

  Args:
    number: The finite float to format.

  Returns:
    The formatted string representation.
  """
  if number == 0:
    return '-0.0' if math.copysign(1, number) < 0 else '0.0'
  sign = '-' if number < 0 else ''
  exact = decimal.Decimal(repr(abs(number))).normalize().as_tuple()
  digits = ''.join(str(digit) for digit in exact.digits)
  # The number is 0.digits * 10^point.
  k, point = len(digits), len(digits) + exact.exponent
  if k <= point <= 15:
    text = digits + '0' * (point - k) + '.0'
  elif 0 < point <= 15:
    text = f'{digits[:point]}.{digits[point:]}'
  elif -4 < point <= 0:
    text = '0.' + '0' * -point + digits
  else:
    mantissa = digits if k == 1 else f'{digits[0]}.{digits[1:]}'
    exponent = point - 1
    text = f'{mantissa}e{"-" if exponent < 0 else "+"}{abs(exponent):02d}'
  return sign + text


def _not_understood(where: str, what: str) -> Refusal:
  """The refusal of a schema the runner does not check."""
  return Refusal(
      _INTERNAL,
      f"The runtime's options schema {what} at {where}, which the runner "
      'does not check.',
  )


def _is_type_name(name: object) -> bool:
  """Whether a value names a JSON Schema type."""
  return isinstance(name, str) and name in _JSON_TYPES


def _check_keywords(schema: object, where: str) -> Refusal | None:
  """Refuses a schema that uses a keyword, or a form of one, it cannot check.

  The schema is found at `where`; a schema inside it is checked too.

  Args:
    schema: The schema object to check.
    where: The path where the schema is found.

  Returns:
    A Refusal if a keyword is not understood, or None if valid.
  """
  if not isinstance(schema, dict):
    return _not_understood(where, 'is not an object')
  for keyword, value in sorted(schema.items()):
    at = f'{where}/{keyword}'
    if keyword == 'type':
      understood = _is_type_name(value) or (
          isinstance(value, list) and all(map(_is_type_name, value))
      )
      if not understood:
        return _not_understood(at, 'names no type it knows')
    elif keyword == 'properties':
      if not isinstance(value, dict):
        return _not_understood(at, 'is not an object')
      for name, property_schema in sorted(value.items()):
        refusal = _check_keywords(property_schema, f'{at}/{name}')
        if refusal:
          return refusal
    elif keyword == 'required':
      if not isinstance(value, list) or not all(
          isinstance(name, str) for name in value
      ):
        return _not_understood(at, 'is not a list of names')
    elif keyword == 'additionalProperties':
      if isinstance(value, bool):
        continue
      refusal = _check_keywords(value, at)
      if refusal:
        return refusal
    elif keyword == 'enum':
      if not isinstance(value, list):
        return _not_understood(at, 'is not a list')
    elif keyword in ('minimum', 'maximum'):
      if not _is_number(value):
        return _not_understood(at, 'is not a number')
    else:
      return _not_understood(at, 'uses a keyword')
  return None


def _check_finite(value: object, name: str) -> Refusal | None:
  """Refuses a value that is or holds a number that is not finite.

  JSON has no NaN or infinity, though a Struct, read from binary, may.

  Args:
    value: The JSON value to check.
    name: The name or path of the value being checked.

  Returns:
    A Refusal if an infinite or NaN number is found, or None if finite.
  """
  if isinstance(value, float) and not math.isfinite(value):
    return Refusal(_INVALID, f'{name} is not a finite number.')
  if isinstance(value, dict):
    for key, member in sorted(value.items()):
      refusal = _check_finite(member, f'{name}.{key}')
      if refusal:
        return refusal
  elif isinstance(value, list):
    for i, item in enumerate(value):
      refusal = _check_finite(item, f'{name}[{i}]')
      if refusal:
        return refusal
  return None


def _has_type(value: object, type_name: str) -> bool:
  """Whether a value is of a JSON Schema type; 2.0 is an integer."""
  if type_name == 'null':
    return value is None
  if type_name == 'boolean':
    return isinstance(value, bool)
  if type_name == 'object':
    return isinstance(value, dict)
  if type_name == 'array':
    return isinstance(value, list)
  if type_name == 'number':
    return _is_number(value)
  if type_name == 'string':
    return isinstance(value, str)
  return _is_number(value) and float(value).is_integer()  # 'integer'


def _json_equal(a: object, b: object) -> bool:
  """Whether two JSON values are equal as nlohmann::json compares them."""
  if isinstance(a, bool) or isinstance(b, bool):
    return type(a) is type(b) and a == b
  if _is_number(a) and _is_number(b):
    return a == b
  if isinstance(a, list) and isinstance(b, list):
    return len(a) == len(b) and all(map(_json_equal, a, b))
  if isinstance(a, dict) and isinstance(b, dict):
    return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
  return type(a) is type(b) and a == b


def _check(schema: dict[str, Any], value: object, name: str) -> Refusal | None:
  """Refuses a value, named `name`, the schema does not allow.

  The schema's keywords are understood, and every number in `value` is
  finite.

  Args:
    schema: The JSON schema to validate against.
    value: The value to validate.
    name: The name of the option or field being checked.

  Returns:
    A Refusal if validation fails, or None if the value matches the schema.
  """
  if 'type' in schema:
    types = schema['type']
    names = [types] if isinstance(types, str) else types
    if not any(_has_type(value, each) for each in names):
      return Refusal(
          _INVALID,
          f'{name} is {_show(value)}; it must be of type {_show(types)}.',
      )
  if 'enum' in schema:
    if not any(_json_equal(each, value) for each in schema['enum']):
      return Refusal(
          _INVALID,
          f'{name} is {_show(value)}; it must be one of '
          f'{_show(schema["enum"])}.',
      )
  if _is_number(value):
    if 'minimum' in schema and value < schema['minimum']:
      return Refusal(
          _INVALID,
          f'{name} is {_show(value)}; it must be at least '
          f'{_show(schema["minimum"])}.',
      )
    if 'maximum' in schema and value > schema['maximum']:
      return Refusal(
          _INVALID,
          f'{name} is {_show(value)}; it must be at most '
          f'{_show(schema["maximum"])}.',
      )
  if not isinstance(value, dict):
    return None
  for member in schema.get('required', []):
    if member not in value:
      return Refusal(_INVALID, f'{name} lacks {member}, which is required.')
  properties = schema.get('properties')
  additional = schema.get('additionalProperties')
  for key, member in sorted(value.items()):
    if properties is not None and key in properties:
      refusal = _check(properties[key], member, f'{name}.{key}')
      if refusal:
        return refusal
      continue
    if additional is None or (isinstance(additional, bool) and additional):
      continue
    if isinstance(additional, bool) and not additional:
      return Refusal(
          _INVALID,
          f'{name} has {key}, which is not an option of this runtime.',
      )
    refusal = _check(additional, member, f'{name}.{key}')
    if refusal:
      return refusal
  return None
