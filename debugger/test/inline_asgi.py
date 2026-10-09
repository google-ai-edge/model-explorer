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

"""In-process ASGI client for tests that evaluates analysis inline.

Requests take the production routing; only analysis skips the worker pool.

Not a test module (unittest discovers test*.py only).
"""

from fastapi.testclient import TestClient
from model_explorer_debugger.analysis import evaluate
from model_explorer_debugger.asgi import create_app
from model_explorer_debugger.store import SessionStore


class InlineAnalysis:
  """Evaluate analysis queries in the test process, not the worker pool."""

  def __init__(self, stores=()):
    self.stores = {str(store.root): store for store in stores}

  def query(self, root, path, query, payload=None, cancelled=None):
    store = self.stores.get(str(root))
    if store is None:
      store = self.stores[str(root)] = SessionStore(root)
    return evaluate(store, path, query, payload)

  def diagnostics(self):
    return {'active': 0, 'inline': True}

  def close(self):
    pass


def client(store=None, registry=None, stores=()):
  """Returns a TestClient over create_app.

  Use it as a context manager so the lifespan closes the Application.
  """
  known = [candidate for candidate in (store, *stores) if candidate is not None]
  app = create_app(store, registry, analysis=InlineAnalysis(known))
  return TestClient(app, base_url='http://127.0.0.1')
