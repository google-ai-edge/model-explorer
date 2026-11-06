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

"""The architectures the server knows, and their skeletons.

An architecture is a model family's definition (a module: gemma4) and its
file, `data/<id>/architecture.json`: its id and name; the Hugging Face text
model_type it defines (`model_type`); its definition's module and where it
was ported from (`generator`: `{definition, derived_from}`); the modules of
the model its skeleton does not hold, each with why (`outside`: `{id, name,
text}`); and its skeletons, one per checkpoint, each `{id, file, sha256,
pinned}`: the skeleton file and its SHA-256, and the checkpoint's
config.json pinned beside it (`pinned`: the file, its Hugging Face model,
revision and source, its git blob id and SHA-256, and where it was copied
from). Where a skeleton comes from, as the server says it, is
Architecture.provenance(): this package's generator and the pinned file,
by their paths in the repository.

A skeleton is what its definition makes of a config.json (generate()): the
generate binary writes each pinned checkpoint's, and checks that they are
the files. Any config.json of a known model_type makes one, so a runtime
that has its model's config, as a PyTorch runner has, needs no pinned
file; a runtime whose model file holds no config.json, a .litertlm, is
matched to a pinned checkpoint by its size (graph/architecture.detect()).

Only what the model is lives here; how a runtime's graph binds to it is
that runtime's (graph/bindings/ for LiteRT-LM).
"""

from collections.abc import Callable
import dataclasses
import hashlib
import json
import pathlib
from typing import Any

from model_debugger.server.architectures import gemma4
from model_debugger.server.architectures import skeleton

_DATA = pathlib.Path(__file__).parent / 'data'
_MANIFEST = 'architecture.json'
# This package's directory in the repository: the server's Bazel workspace,
# src/server, holds model_debugger/ (generate.py's _PACKAGE).
REPOSITORY_PACKAGE = 'src/server/model_debugger/server/architectures'
# What writes the skeletons, and what checks them (ci/check_server.sh).
GENERATE_COMMAND = 'bazel run //model_debugger/server/architectures:generate'
CHECK_COMMAND = GENERATE_COMMAND + ' -- --check'
# Per architecture id, its definition: a text config to its skeleton.
_DEFINITIONS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    'gemma4': gemma4.build
}


@dataclasses.dataclass(frozen=True)
class Architecture:
  """A model family: its definition's file, and the files beside it.

  Attributes:
      id: Its id, its directory's name.
      name: What the user knows it by.
      spec: Its architecture.json, whole.
      directory: Its directory, which holds its skeletons and configs.
  """

  id: str
  name: str
  spec: dict[str, Any]
  directory: pathlib.Path

  def entry(self, skeleton_id: str) -> dict[str, Any]:
    """One of its skeletons' entries in its file.

    Args:
        skeleton_id: The skeleton's id.

    Returns:
        The skeleton's entry in the architecture's manifest.

    Raises:
        ValueError: It has no such skeleton.
    """
    for entry in self.spec['skeletons']:
      if entry['id'] == skeleton_id:
        return entry
    raise ValueError(f'{self.name} has no skeleton {skeleton_id}.')

  def skeleton(self, skeleton_id: str) -> dict[str, Any]:
    """One of its skeletons, checked (skeleton.load())."""
    return skeleton.load(self.directory / self.entry(skeleton_id)['file'])

  def build(self, config: dict[str, Any]) -> dict[str, Any]:
    """The skeleton its definition makes of a text config, unchecked."""
    return _DEFINITIONS[self.id](config)

  def provenance(self, skeleton_id: str) -> dict[str, Any]:
    """Where one of its skeletons comes from, by paths in the repository.

    Args:
        skeleton_id: The skeleton's id.

    Returns:
        `{generator, config}`: `generator` what writes the skeleton,
        `{path, definition, command, check, identity, derived_from}`
        (generate.py, the definition's module, the commands that write
        and check it, the architectures' files' digest (identity()),
        and the note of where the definition was ported from); `config`
        the pinned config.json it is made of, `{path, model, revision,
        source, git_blob, sha256, copied_from}`.

    Raises:
        ValueError: It has no such skeleton.
    """
    pinned = self.entry(skeleton_id)['pinned']
    generator = self.spec['generator']
    return {
        'generator': {
            'path': f'{REPOSITORY_PACKAGE}/generate.py',
            'definition': f'{REPOSITORY_PACKAGE}/{generator["definition"]}',
            'command': GENERATE_COMMAND,
            'check': CHECK_COMMAND,
            'identity': identity(),
            'derived_from': generator['derived_from'],
        },
        'config': {
            'path': f'{REPOSITORY_PACKAGE}/data/{self.id}/{pinned["file"]}',
            **{key: pinned[key] for key in pinned if key != 'file'},
        },
    }


def architectures() -> list[Architecture]:
  """Every architecture the server knows, by id."""
  found = []
  for path in sorted(_DATA.glob(f'*/{_MANIFEST}')):
    spec = json.loads(path.read_text(encoding='utf-8'))
    if spec['id'] != path.parent.name or spec['id'] not in _DEFINITIONS:
      raise ValueError(
          f'{path} is of architecture {spec["id"]!r}, which is not '
          'its directory or has no definition.'
      )
    found.append(Architecture(spec['id'], spec['name'], spec, path.parent))
  return found


def text_config(config: dict[str, Any]) -> dict[str, Any]:
  """The text model's config of a config.json: `text_config`, or it all."""
  return config.get('text_config', config)


def generate(config: dict[str, Any]) -> tuple[Architecture, dict[str, Any]]:
  """The architecture of a config.json, and the skeleton it makes, checked.

  Args:
      config: A Hugging Face config.json, parsed; of a multimodal model, its
        `text_config` is the text model's.

  Returns:
      `(architecture, skeleton)`: the architecture that defines its text
      model_type, and the skeleton it makes of the text config.

  Raises:
      ValueError: No architecture defines its text model_type, or the
        skeleton made breaks a rule of the schema (the text lists each).
  """
  text = text_config(config)
  model_type = text.get('model_type')
  known = [a for a in architectures() if a.spec['model_type'] == model_type]
  if not known:
    raise ValueError(
        f'No architecture defines model_type {model_type!r}; the known are '
        + ', '.join(a.spec['model_type'] for a in architectures())
        + '.'
    )
  found = known[0].build(text)
  errors = skeleton.check(found)
  if errors:
    raise ValueError(
        f'The {known[0].name} skeleton made of this config is not valid: '
        + '; '.join(errors)
    )
  return known[0], found


def identity() -> str:
  """A digest of the architectures' files: what they are and hold."""
  digest = hashlib.sha256()
  for path in sorted(_DATA.rglob('*.json')):
    name = path.relative_to(_DATA).as_posix().encode()
    digest.update(name + b'\0' + path.read_bytes())
  return digest.hexdigest()[:16]
