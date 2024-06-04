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

import argparse

from . import server
from .config import ModelExplorerConfig
from .consts import DEFAULT_HOST, DEFAULT_PORT

parser = argparse.ArgumentParser(
    prog='model-explorer',
    description='A modern model graph visualizer and debugger',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('model_paths',
                    nargs='?',
                    help='comma separated model file paths')
parser.add_argument('--host',
                    default=DEFAULT_HOST,
                    help='host of the server')
parser.add_argument('--port',
                    default=DEFAULT_PORT,
                    type=int,
                    help='port of the server')
parser.add_argument('--node_data_paths',
                    help='comma separated node data file paths')
parser.add_argument('--no_open_in_browser',
                    action='store_true',
                    help='Don\'t open the web app in browser after server starts')
parser.add_argument('--extensions',
                    help='comma separated extension module names')
parser.add_argument('--cors_host',
                    help='the host of the Access-Control-Allow-Origin header')
parser.add_argument('--skip_health_check',
                    action='store_true',
                    help='Whether to skip the health check after server starts')
args = parser.parse_args()


def main():
  """Entry point for the command line version of model explorer."""

  model_paths: list[str] = []
  if args.model_paths is not None and args.model_paths != '':
    model_paths = [x.strip() for x in args.model_paths.split(',')]

  node_data_paths: list[str] = []
  if args.node_data_paths is not None and args.node_data_paths != '':
    node_data_paths = [x.strip() for x in args.node_data_paths.split(',')]

  extensions: list[str] = []
  if args.extensions is not None:
    extensions = [x.strip() for x in args.extensions.split(',')]

  # Construct config.
  config = ModelExplorerConfig()
  for model_path in model_paths:
    config.add_model_from_path(model_path)
  for node_data_path in node_data_paths:
    config.add_node_data_from_path(node_data_path)

  server.start(host=args.host,
               port=args.port,
               config=config,
               extensions=extensions,
               cors_host=args.cors_host,
               no_open_in_browser=args.no_open_in_browser,
               skip_health_check=args.skip_health_check)
