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

import json
import logging
import os
import platform
import queue
import socket
import sys
import tempfile
import threading
import time
import traceback
import webbrowser
from importlib import metadata
from time import sleep
from typing import Any, Union

import portpicker
import requests
from flask import Flask, Response, make_response, request, send_from_directory
from IPython import display
from packaging.version import parse
from termcolor import colored, cprint

from .config import ModelExplorerConfig
from .consts import (
    DEFAULT_COLAB_HEIGHT,
    DEFAULT_HOST,
    DEFAULT_PORT,
    PACKAGE_NAME,
)
from .extension_manager import ExtensionManager
from .server_directive_dispatcher import ServerDirectiveDispatcher
from .server_director import ServerDirector
from .utils import convert_adapter_response

server_directive_dispatcher = ServerDirectiveDispatcher()


def _make_json_response(obj):
  body = json.dumps(obj)
  resp = make_response(body)
  resp.headers['Content-Type'] = 'application/json'
  return resp


def _get_latest_version_from_repo(package_json_url: str) -> str:
  req = requests.get(package_json_url)
  version = parse('0')
  if req.status_code == requests.codes.ok:
    j = json.loads(req.text.encode('utf-8'))
    releases = j.get('releases', [])
    for release in releases:
      ver = parse(release)
      if not ver.is_prerelease:
        version = max(version, ver)
  return str(version)


def _get_release_from_github(version: str) -> dict:
  # Get release data through github API.
  req = requests.get(
      f'https://api.github.com/repos/google-ai-edge/model-explorer/releases/tags/model-explorer-v{version}'
  )
  req_json = json.loads(req.text.encode('utf-8'))

  # Construct the search term from platform and cpu architecture for finding
  # asset download url.
  #
  # darwin, linux, etc.
  cur_platform = sys.platform
  # x64_64, arm64, etc.
  cur_mach = platform.machine()
  if cur_mach == 'x86_64':
    cur_mach = 'x64'

  # Find the download url from assets.
  asset_search_term = f'{cur_platform}-{cur_mach}-{version}'
  asset_url = ''
  assets = req_json.get('assets', [])
  for asset in assets:
    browser_download_url = asset.get('browser_download_url', '')
    if asset_search_term in browser_download_url:
      asset_url = browser_download_url
      break

  return {
      'releaseUrl': req_json.get('html_url', ''),
      'desktopAppUrl': f'{asset_url}',
  }


def _print_yellow(x):
  return cprint(x, 'yellow')


def _check_new_version(print_msg=True):
  check_new_version_resp = {
      'version': '',
      'runningVersion': '',
      'releaseUrl': '',
      'desktopAppUrl': '',
  }
  try:
    # Get version from repo.
    repo_version = _get_latest_version_from_repo(
        f'https://pypi.python.org/pypi/{PACKAGE_NAME}/json'
    )

    if repo_version != '0':
      # Compare with the local installed version.
      installed_version = metadata.version(PACKAGE_NAME)
      check_new_version_resp['runningVersion'] = installed_version
      if parse(installed_version) < parse(repo_version):
        check_new_version_resp['version'] = repo_version
        if print_msg:
          _print_yellow(
              f'\n{PACKAGE_NAME} version {repo_version} is available, and you'
              f' are using version {installed_version}.'
          )
          _print_yellow('Consider upgrading via the following command:')
          _print_yellow(f'$ pip install -U {PACKAGE_NAME}')

        # Get the corresponding release data from github.
        github_release = _get_release_from_github(repo_version)
        releaseUrl = github_release['releaseUrl']
        check_new_version_resp['releaseUrl'] = releaseUrl
        check_new_version_resp['desktopAppUrl'] = github_release[
            'desktopAppUrl'
        ]

        if print_msg:
          _print_yellow(f'\nRelease notes: {releaseUrl}')
  except:
    pass
  finally:
    return check_new_version_resp


def _is_port_in_use(host: str, port: int) -> bool:
  try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
      return s.connect_ex((host, port)) == 0
  except socket.gaierror:
    print(
        f'"{host}" cannot be resolved. Try using IP address directly:'
        ' model-explorer --host=127.0.0.1'
    )
    sys.exit(1)


def _js(script):
  display.display(display.Javascript(script))


def start(
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
    config: Union[ModelExplorerConfig, None] = None,
    no_open_in_browser: bool = False,
    extensions: list[str] = [],
    colab_height: int = DEFAULT_COLAB_HEIGHT,
    cors_host: Union[str, None] = None,
    skip_health_check: bool = False,
):
  """Starts the local server that serves the web app.

  Args:
    host: The host of the server. Default to localhost.
    port: The port of the server. Default to 8080.
    config: the object that stores the data to be visualized.
    no_open_in_browser: Don't open the web app in browser after server starts.
    extensions: A list of extension module names. Default to empty.
    colab_height: The height of the embedded iFrame when running in colab.
    cors_host: The value of the Access-Control-Allow-Origin header. The header
        won't be present if it is None.
    skip_health_check: Whether to skip the health check after server starts.
  """

  # Don't start the server if user wants to reuse an existing server.
  if (
      config is not None
      and config.reuse_server_host != ''
      and config.reuse_server_port > 0
  ):
    director = ServerDirector(config=config)
    director.update_config()
    return

  # Check whether it is running in colab.
  colab = 'google.colab' in sys.modules or os.getenv('COLAB_RELEASE_TAG')

  # Check port in non-colab environment.
  if not colab:
    # If default port (8080) is specified, try to find an available port
    # automatically.
    #
    # - First, find a port from 8080 to 8099.
    # - If no available port from that range, pick a random port.
    if port == 8080:
      found_port = False
      for i in range(0, 20):
        port = port + i
        if not _is_port_in_use(host, port):
          found_port = True
          break
      if not found_port:
        port = portpicker.pick_unused_port()
    # If a non-default port is specified, output an error message when port
    # is used.
    else:
      if _is_port_in_use(host, port):
        print(colored(f'port {port} already in use.', 'red'))
        sys.exit(1)

  app = Flask(__name__)

  # Disable logging from werkzeug.
  #
  # Without this, flask will show a warning about using dev server (which is OK
  # in our usecase).
  logging.getLogger('werkzeug').disabled = True

  # Disable startup messages.
  cli: Any = sys.modules['flask.cli']
  cli.show_server_banner = lambda *x: None

  # Print a info message when used in colab.
  if colab:
    print('ℹ️ Please re-run the cell in each new session')
    print()

  # Load extensions.
  print('Loading extensions...')
  extension_manager = ExtensionManager(extensions)
  extension_manager.load_extensions()
  extension_metadata_list = extension_manager.get_extensions_metadata()
  num_extensions = len(extension_metadata_list)
  print(
      f'Loaded {num_extensions} extension{"" if num_extensions == 1 else "s"}:'
  )
  for extension in extension_metadata_list:
    print(f' - {extension["name"]}')

  @app.route('/api/v1/check_new_version')
  def check_new_version():
    """Checks new version."""
    return _make_json_response(_check_new_version(False))

  @app.route('/api/v1/get_extensions')
  def get_extensions():
    """Loads all adapter extensions."""
    return _make_json_response(extension_manager.get_extensions_metadata())

  # Note: using "/api/..." for POST requests is not allowed when running in
  # colab.
  @app.route('/apipost/v1/upload', methods=['POST'])
  def upload_file():
    f = request.files['file']
    file_name = f.filename if f.filename is not None else 'no_name'
    tmp_dir = tempfile.mkdtemp()
    file_path = os.path.join(tmp_dir, file_name)
    f.save(file_path)
    return _make_json_response({'path': file_path})

  # Note: using "/api/..." for POST requests is not allowed when running in
  # colab.
  @app.route('/apipost/v1/send_command', methods=['POST'])
  def send_command():
    try:
      resp = extension_manager.run_cmd(request.json)
      return _make_json_response(resp)
    except Exception as err:
      traceback.print_exc()
      return _make_json_response({'error': f'{type(err).__name__}: {str(err)}'})
    finally:
      extension_manager.cleanup(request.json)

  @app.route('/api/v1/load_graphs_json')
  def load_graphs_json():
    if config is None:
      return {}
    graph_index_str = request.args.get('graph_index')
    if graph_index_str is None:
      return {}
    graph_index = int(graph_index_str)
    return _make_json_response(
        convert_adapter_response(config.get_model_explorer_graphs(graph_index))
    )

  @app.route('/api/v1/load_node_data')
  def load_node_data():
    if config is None:
      return {}
    node_data_index_str = request.args.get('node_data_index')
    if node_data_index_str is None:
      return {}
    node_data_index = int(node_data_index_str)
    node_data = config.get_node_data(node_data_index)
    if isinstance(node_data, str):
      json_str = node_data
    else:
      json_str = node_data.to_json_string()
    return _make_json_response({'content': json_str})

  @app.route('/api/v1/read_text_file')
  def read_text_file():
    path = request.args.get('path')
    if path is None:
      return _make_json_response({'error': 'no file path provided'})
    path = os.path.expanduser(path)

    try:
      with open(path, 'r') as file:
        content = file.read()
      return _make_json_response({'content': content})
    except Exception as err:
      return _make_json_response({'error': str(err)})

  @app.route('/check_health')
  def check_health():
    """Serves check_health request."""
    return 'model_explorer_ok'

  @app.route('/apipost/v1/update_config', methods=['POST'])
  def update_config():
    # TODO(do not submit): Update confnig.
    config_data = request.json
    if config and config_data:
      config.set_transferrable_data(config_data)

      # Ask UI to refresh page with the new url.
      server_directive_dispatcher.broadcast(
          json.dumps({
              'name': 'refreshPage',
              'url': f'/?data={config.to_url_param_value()}',
          })
      )

    return ''

  @app.route('/apistream/server_director')
  def server_director_stream():
    def stream():
      directive_queue = server_directive_dispatcher.listen()
      try:
        while True:
          # Try to get a new message.
          try:
            msg = directive_queue.get(block=False)
            yield f'data: {msg}\n\n'
          except queue.Empty:
            # Ignore if there is no new messages.
            pass

          # Keep the connection alive.
          yield ': heartbeat\n\n'
          time.sleep(1)
      except:
        # The client closes the connection (i.e. close the browser tab)
        server_directive_dispatcher.remove_listener(directive_queue)

    headers = {
        'Access-Control-Allow-Origin': '*',
        'Cache-Control': 'no-cache',
    }
    return Response(
        stream(),
        headers=headers,
        content_type='text/event-stream',
        mimetype='text/event-stream',
    )

  @app.route('/')
  def send_index_html():
    """Serves index.html."""
    return send_from_directory('web_app', 'index.html')

  @app.route('/<path:path>')
  def send_static(path):
    """Serves static files."""
    return send_from_directory('web_app', path)

  @app.after_request
  def add_header(response):
    """Adds headers to all responses."""
    # Don't cache any files.
    response.headers['Cache-Control'] = 'public, max-age=0'
    if cors_host is not None:
      response.headers['Access-Control-Allow-Origin'] = cors_host
      response.headers['Access-Control-Allow-Credentials'] = 'true'
      response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

  def start_server():
    """Starts the server in non-colab environment."""
    app_thread = threading.Thread(target=lambda: app.run(host=host, port=port))
    app_thread.daemon = True
    app_thread.start()

    # Wait for server to start
    if not skip_health_check:
      while True:
        try:
          response = requests.get(f'http://{host}:{port}/')
        except:
          continue

        if response.status_code == 200:
          break

    # Server is ready.
    server_address = f'http://{host}:{port}'
    url_params: list[str] = []
    if config is not None and config.has_data_to_encode_in_url():
      url_params.append(f'data={config.to_url_param_value()}')

    if len(url_params) > 0:
      server_address = f'{server_address}/?{"&".join(url_params)}'
    print(
        f'\nStarting Model Explorer server at:\n{server_address}\n\nPress'
        ' Ctrl+C to stop.'
    )
    if not no_open_in_browser:
      webbrowser.open_new_tab(f'{server_address}')

    # Check installed version vs published version.
    threading.Thread(target=lambda: _check_new_version()).start()

    try:
      while app_thread.is_alive():
        sleep(1)
    except KeyboardInterrupt:
      print('Stopping server...')
      pass

  def embed_in_colab():
    """Embeds the UI in a colab cell."""
    # Disable scrollbar in iFrame's parent div.
    _js('google.colab.output.setIframeHeight(0, true, {maxHeight: 5000})')

    colab_port = portpicker.pick_unused_port()
    shell = """
      (async () => {
          let url = await google.colab.kernel.proxyPort(%PORT%, {"cache": true});
          if ('%HOSTED_RUNTIME%' !== 'True') {
            url = 'http://localhost:' + %PORT% + '/';
          }
          url += '?show_open_in_new_tab=1';
          if ('%DATA_PARAM%' !== '') {
            url += '&%DATA_PARAM%';
          }
          const iframe = document.createElement('iframe');
          iframe.src = url;
          iframe.setAttribute('width', '100%');
          iframe.setAttribute('height', '%HEIGHT%');
          iframe.setAttribute('frameborder', 0);
          iframe.setAttribute('style', 'border: 1px solid #ccc; box-sizing: border-box; margin-top: 12px;');
          document.body.appendChild(iframe);
        })();
    """
    data_param = ''
    if config is not None and config.has_data_to_encode_in_url():
      data_param = f'data={config.to_url_param_value()}'
    replacements = [
        ('%PORT%', f'{colab_port}'),
        ('%HOSTED_RUNTIME%', f'{colab}'),
        ('%HEIGHT%', f'{colab_height}'),
        ('%DATA_PARAM%', f'{data_param}'),
    ]
    for k, v in replacements:
      shell = shell.replace(k, v)

    threading.Thread(
        target=app.run, kwargs={'host': '::', 'port': colab_port}
    ).start()

    _js(shell)

  if colab:
    embed_in_colab()
  else:
    start_server()
