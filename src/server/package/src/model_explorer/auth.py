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

"""Access token that protects the Model Explorer server API.

The server generates a random token when it starts and requires it on every API
request. The token only reaches the browser through the URL that the server
prints, opens, or embeds (it is never served by an unauthenticated endpoint), so
other web pages (e.g. via CSRF or DNS rebinding) and other machines cannot use
the API.

The token is also written to a file only readable by the current user, so that
another Model Explorer process of the same user (`--reuse_server`) can talk to
the running server.
"""

from __future__ import annotations

import hmac
import ipaddress
import os
import re
import secrets
import socket
from urllib import parse

# The HTTP header that carries the token.
AUTH_TOKEN_HEADER = 'X-Model-Explorer-Token'

# The URL query parameter that carries the token. The web app reads it from the
# page URL. API requests only accept it on the paths in
# `QUERY_TOKEN_PATH_PREFIXES`, where clients can't set headers.
AUTH_TOKEN_QUERY_PARAM = 'token'

# Path prefixes of the API endpoints that require the token. Static assets, the
# index page and `/check_health` stay public: they don't expose data and are
# needed to load the page and to discover running servers.
PROTECTED_PATH_PREFIXES = ('/api/', '/apipost/', '/apistream/')

# Path prefixes of the API endpoints that also accept the token as a query
# parameter, because `EventSource` can't send headers. Other endpoints need the
# header, which keeps the token out of request URLs and makes cross-site
# requests need a CORS preflight.
QUERY_TOKEN_PATH_PREFIXES = ('/apistream/',)

# Tokens must be URL-safe (they go into URLs and Colab's embedding script) and
# long enough to not be guessable.
_VALID_TOKEN_REGEX = re.compile(r'[A-Za-z0-9_-]{16,}')
_TOKEN_NUM_BYTES = 32
_TOKEN_FILE_DIR_MODE = 0o700
_TOKEN_FILE_MODE = 0o600


def generate_token() -> str:
  """Returns a new random access token."""
  return secrets.token_urlsafe(_TOKEN_NUM_BYTES)


def check_token_format(token: str) -> None:
  """Raises ValueError if `token` can't be used as an access token."""
  if not _VALID_TOKEN_REGEX.fullmatch(token):
    raise ValueError(
        'The access token must have at least 16 characters out of A-Z, a-z,'
        ' 0-9, "_" and "-".'
    )


def is_protected_path(path: str) -> bool:
  """Returns whether the request to the given URL path requires the token."""
  return path.startswith(PROTECTED_PATH_PREFIXES)


def accepts_query_token(path: str) -> bool:
  """Returns whether the request to the given URL path accepts `?token=`."""
  return path.startswith(QUERY_TOKEN_PATH_PREFIXES)


def is_valid_token(expected: str, provided: str | None) -> bool:
  """Returns whether `provided` matches `expected` (in constant time)."""
  if not provided:
    return False
  return hmac.compare_digest(provided.encode(), expected.encode())


def token_query_param(token: str) -> str:
  """Returns the URL query parameter (`token=...`) that carries `token`."""
  return parse.urlencode({AUTH_TOKEN_QUERY_PARAM: token})


def add_token_to_url(url: str, token: str | None) -> str:
  """Returns the given URL with the token query parameter added.

  Args:
    url: An absolute or relative URL.
    token: The token to add. The URL is returned unchanged if it is None.
  """
  if token is None:
    return url
  parts = parse.urlsplit(url)
  query = token_query_param(token)
  return parse.urlunsplit(
      parts._replace(query=f'{parts.query}&{query}' if parts.query else query)
  )


def is_local_host(host: str) -> bool:
  """Returns whether `host` refers to this machine.

  Token files are only readable on this machine, so their tokens must only be
  sent to servers on this machine.

  Args:
    host: A host name or IP address.
  """
  if host.lower() == 'localhost':
    return True
  try:
    return ipaddress.ip_address(host.strip('[]')).is_loopback
  except ValueError:
    # Not an IP address. Avoid DNS lookups, which can block for a long time.
    return host == socket.gethostname()


def _get_token_file_path(port: int) -> str:
  cache_dir = os.environ.get('XDG_CACHE_HOME') or os.path.join(
      os.path.expanduser('~'), '.cache'
  )
  return os.path.join(cache_dir, 'model_explorer', f'server-{port}.token')


def write_token_file(port: int, token: str) -> str | None:
  """Writes the token of the server on `port` to a user-private file.

  Args:
    port: The port of the server.
    token: The token of the server.

  Returns:
    The path of the file, or None if it couldn't be written (e.g. read-only home
    directory). The server works without it, except for `--reuse_server`.
  """
  path = _get_token_file_path(port)
  try:
    os.makedirs(os.path.dirname(path), mode=_TOKEN_FILE_DIR_MODE, exist_ok=True)
    # Replace any existing file (e.g. left by a server that crashed) instead of
    # writing into it, so that the new file gets the restrictive mode.
    if os.path.lexists(path):
      os.remove(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _TOKEN_FILE_MODE)
    with os.fdopen(fd, 'w') as f:
      f.write(token)
  except OSError:
    return None
  return path


def read_token_file(port: int) -> str | None:
  """Returns the token of the server on `port`, or None if not found."""
  try:
    with open(_get_token_file_path(port), 'r') as f:
      return f.read().strip() or None
  except OSError:
    return None


def remove_token_file(port: int, token: str) -> None:
  """Removes the token file of the server on `port` if it still holds `token`.

  The check avoids deleting the file of a newer server on the same port.

  Args:
    port: The port of the server.
    token: The token of the server.
  """
  if read_token_file(port) != token:
    return
  try:
    os.remove(_get_token_file_path(port))
  except OSError:
    pass
