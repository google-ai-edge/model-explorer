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

"""Observing tensors, as the UI shows it: anchors, a model's scan, profiles.

What a LiteRT-LM side captures by default, every decoder layer's anchors,
the editor shows in that side's box as `tap_anchors` of GET
/api/runtime/capabilities says (anchors_capture(): the checkbox's label and
help), and, of the side's model, as GET /api/artifacts/{id}/anchors
answers (anchors()): `status` ('found', 'pending' until the model's graph
is converted, or 'none'), `text`, what is captured or why nothing is,
`points`, how many, `copy_note` as below, and per signature `{name, points,
layers, text, skipped}`, `skipped` each anchor left out with the reason,
one line per reason.

GET and POST /api/artifacts/{id}/scan answer the UI's TapScan: `status`
('idle', 'scanning', 'completed' or 'failed'), `error` once failed,
`copy_note`, what observing the model's tensors costs on disk
(prepared_models.PreparedModels.disk_note()), `max_points`, the most points
of one model a Session observes, its anchors among them, and once completed
`points`, `recommended` and `signatures`. A point is
the UI's TapPoint: where it is, what it is, whether it can be chosen and why
not (`selectable`, `reason`), and `note`, what the user should know before
choosing it: that its signature may not run in a chat, so it may capture
nothing; that its shape holds the model's KV length, which a capture holds
at the context length; that its signature outputs it already.
"""

from typing import Any

from model_debugger.server.session import anchor_points
from model_debugger.server.session import session_store
from model_debugger.server.session import tap_scans
from model_debugger.server.tap import tap_model
from model_debugger.server.tap import tap_scan

# What a Session may observe, as GET /api/runtime/capabilities lists it.
PROFILES = (
    {
        'id': session_store.TAP_PROFILE,
        'name': 'Chosen tensor outputs',
        'description': (
            'Captures the operator outputs chosen of the text model, at '
            'every step their signature runs, from a copy of the model that '
            'outputs them; with the anchors, besides them.'
        ),
    },
)
# What every LiteRT-LM side captures, as the editor's tensor capture says it.
ANCHORS_CAPTURE = {
    'name': 'Capture the anchors',
    # One line: what is captured. How (a copy of the model, prepared as the
    # Session starts) is the anchors' copy_note, which the editor folds away.
    'description': "Captures every layer's 4 anchors in prefill and decode.",
}


def anchors_capture() -> dict[str, Any]:
  """The UI's tap_anchors: what capturing the anchors is."""
  return dict(ANCHORS_CAPTURE)


def anchors(found: anchor_points.Anchors, copy_note: str) -> dict[str, Any]:
  """The UI's TapAnchors: the anchors a Session of a model captures.

  Args:
    found: The model's anchors.
    copy_note: What observing the model's tensors costs on disk.

  Returns:
    The anchors captured by a model session as a dictionary.
  """
  return {
      'status': found.status,
      'text': found.text,
      'points': len(found.point_ids),
      'copy_note': copy_note if found.point_ids else '',
      'signatures': [
          {
              'name': signature.name,
              'points': len(signature.point_ids),
              'layers': list(signature.layers),
              'text': signature.text,
              'skipped': list(signature.skipped),
          }
          for signature in found.signatures
      ],
  }


def profiles() -> list[dict[str, Any]]:
  """The UI's tap_profiles: what a Session may observe."""
  return [dict(profile) for profile in PROFILES]


def scan(state: tap_scans.State, copy_note: str) -> dict[str, Any]:
  """The UI's TapScan of a model's scan state.

  Args:
    state: Where the scan is.
    copy_note: What observing the model's tensors costs on disk.

  Returns:
    A dictionary representation of the model's scan state.
  """
  if state.status != tap_scans.COMPLETED:
    view = {
        'status': state.status,
        'copy_note': copy_note,
        'max_points': tap_model.MAX_POINTS,
    }
    if state.error:
      view['error'] = state.error
    return view
  found = state.scan
  notes = _signature_notes(found.signatures)
  return {
      'status': state.status,
      'copy_note': copy_note,
      'max_points': tap_model.MAX_POINTS,
      'signatures': [
          {
              'name': signature.name,
              'runs': signature.runs,
              'note': notes[signature.name],
          }
          for signature in found.signatures
      ],
      'points': [
          _point(point, notes[point.signature], found.kv_length)
          for point in found.points
      ],
      'recommended': list(found.recommended),
  }


def _signature_notes(
    signatures: tuple[tap_scan.Signature, ...],
) -> dict[str, Any]:
  """What choosing a point of each signature means, by signature."""
  smallest = [
      signature.name
      for signature in signatures
      if signature.runs == tap_scan.RUNS_ALWAYS and signature.name != 'decode'
  ]
  short = (
      f'a short one is prefilled by {smallest[0]} alone'
      if smallest
      else 'a short one may not need it'
  )
  notes = {}
  for signature in signatures:
    name = signature.name
    notes[name] = {
        tap_scan.RUNS_ALWAYS: '',
        tap_scan.RUNS_LONG_PROMPTS: (
            f'{name} runs only for long prompts; {short}, so this point '
            'may capture nothing.'
        ),
        tap_scan.RUNS_SPECULATIVE: (
            f'{name} runs only with speculative decoding, which cannot be '
            'enabled for now, so this point captures nothing.'
        ),
        tap_scan.RUNS_UNKNOWN: (
            f'LiteRT-LM is not known to run {name}, so this point may '
            'capture nothing.'
        ),
    }[signature.runs]
  return notes


def _point(
    point: tap_scan.Point, note: str, kv_length: int | None
) -> dict[str, Any]:
  """The UI's TapPoint."""
  notes = [note] if note else []
  if point.context_axes:
    notes.append(
        f"Its {kv_length} is the model's KV length; a capture holds it at "
        "the Session's context length."
    )
  if point.output_name is not None:
    notes.append(
        f'{point.signature} outputs it already, as {point.output_name}; '
        'it is captured without being observed.'
    )
  return {
      'id': point.id,
      'signature': point.signature,
      'op': point.op,
      'output': point.output,
      'tensor': point.tensor,
      'tensor_name': point.tensor_name,
      'shape': list(point.shape),
      'dtype': point.dtype,
      'operator': point.operator,
      'selectable': point.selectable,
      'reason': point.reason,
      'note': ' '.join(notes),
  }
