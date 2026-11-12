# Copyright 2024 The AI Edge Model Explorer Authors.
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

import requests

from . import auth
from .config import ModelExplorerConfig


class ServerDirector:
  """A class to ask an exsiting ME server to do stuff."""

  def __init__(self, config: ModelExplorerConfig):
    self.config = config

  def update_config(self) -> bool:
    """Updates config in the running server.

    Returns:
      Whether the server accepted the update.
    """

    data = self.config.get_transferrable_data()
    server_address = (
        'http://'
        f'{self.config.reuse_server_host}:'
        f'{self.config.reuse_server_port}'
    )
    # A running server on this machine writes its access token to a
    # user-private file. Never send it to another machine.
    headers: dict[str, str] = {}
    token = (
        auth.read_token_file(self.config.reuse_server_port)
        if auth.is_local_host(self.config.reuse_server_host)
        else None
    )
    if token is not None:
      headers[auth.AUTH_TOKEN_HEADER] = token
    resp = requests.post(
        f'{server_address}/apipost/v1/update_config', json=data, headers=headers
    )
    return resp.ok
