#!/usr/bin/env python3
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

"""Preview the offline Runner renderer without exposing other repo files."""

import argparse
from http import server
import pathlib
import types
import urllib.parse

UI = pathlib.Path(__file__).resolve().parents[1] / 'src/runner/ui'
ROUTES = types.MappingProxyType({
    '/': ('index.html', 'text/html; charset=utf-8'),
    '/index.html': ('index.html', 'text/html; charset=utf-8'),
    '/debugger-logo.svg': ('debugger-logo.svg', 'image/svg+xml'),
})


class Handler(server.BaseHTTPRequestHandler):
  """Static file handler serving only files from the Runner UI directory."""

  def do_GET(self) -> None:
    path = urllib.parse.urlsplit(self.path).path
    if path == '/favicon.ico':
      self.send_response(204)
      self.end_headers()
      return
    route = ROUTES.get(path)
    if route is None:
      self.send_error(404)
      return
    filename, mime = route
    data = (UI / filename).read_bytes()
    self.send_response(200)
    self.send_header('Content-Type', mime)
    self.send_header('Content-Length', str(len(data)))
    self.send_header('Cache-Control', 'no-store')
    self.send_header('X-Content-Type-Options', 'nosniff')
    self.end_headers()
    self.wfile.write(data)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--port', type=int, default=8872)
  args = parser.parse_args()
  http_server = server.ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
  print(f'Runner renderer: http://127.0.0.1:{args.port}/', flush=True)
  http_server.serve_forever()
