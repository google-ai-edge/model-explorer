/**
 * @license
 * Copyright 2024 The Model Explorer Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

const {app, BrowserWindow, Menu, dialog} = require('electron');
const log = require('electron-log')
const {spawn} = require('node:child_process');
const path = require('node:path');
const http = require('http');

var splashScreen = null;
var mainWindow = null;
var meServerPort = -1;
var meServerProcess = null;
var meServerReady = false;
var filePathToOpen = '';

app.on('open-file', (event, path) => {
  event.preventDefault();

  if (!mainWindow) {
    filePathToOpen = path;
    if (meServerReady) {
      createMainWindow();
    }
  } else {
    loadFiles([path]);
  }
});

app.whenReady().then(async () => {
  // Create the main window when the app is relaunched while it is still
  // running.
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createMainWindow();
    }
  });

  // Kill the server process when the app quits.
  app.on('quit', () => {
    try {
      if (meServerProcess) {
        meServerProcess.kill();
      }
    } catch {
      // Ignore error.
    }
  });

  // Quit the app when all windows are closed on non-mac platforms.
  app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
      app.quit();
    }
  });

  // Typical app startup process.
  showSplashScreen();
  createMenu();
  await startServer();
  await waitForServer();
  hideSplashScreen();
  createMainWindow();
});

function createMenu() {
  log.info('Creating menu');

  // setting up the menu with just two items
  const menu = Menu.buildFromTemplate([
    {
      label: 'Menu',
      submenu: [
        // Open file.
        {
          label: 'Open Model File',
          accelerator: 'CmdOrCtrl+O',
          click() {
            dialog
              .showOpenDialog({
                properties: ['openFile', 'multiSelections'],
              })
              .then(function (fileObj) {
                if (!fileObj.canceled && mainWindow) {
                  loadFiles(fileObj.filePaths);
                }
              })
              .catch(function (err) {
                console.error(err);
              });
          },
        },
        // Quit.
        {
          label: 'Quit Model Explorer',
          accelerator: 'CmdOrCtrl+Q',
          click() {
            app.quit();
          },
        },
      ],
    },
  ]);
  Menu.setApplicationMenu(menu);
}

function showSplashScreen() {
  log.info('Showing splash screen');

  splashScreen = new BrowserWindow({
    width: 300,
    height: 300,
    frame: false,
    alwaysOnTop: true,
    center: true,
  });
  splashScreen.loadFile('src/splash_screen.html');
}

function hideSplashScreen() {
  log.info('Hiding splash screen');
  splashScreen?.close();
}

async function startServer() {
  // Find a free port and start ME server at that port.
  log.info('Getting free port');
  meServerPort = await getFreePort();
  log.info(`Free port: ${meServerPort}`);

  const baseDir = isDev()
    ? process.cwd()
    : path.join(process.resourcesPath, 'app');
  const serverBinary = path.join(
    path.join(baseDir, 'model_explorer_server'),
    'model_explorer',
  );
  meServerProcess = spawn(serverBinary, [
    '--extensions=model_explorer_onnx',
    '--host=127.0.0.1',
    '--no_open_in_browser',
    `--port=${meServerPort}`,
  ]);
  log.info('Server process spawned');
}

async function waitForServer() {
  log.info('Waiting for server');
  return new Promise((resolve) => {
    const checkStatus = () => {
      const options = {
        hostname: '127.0.0.1',
        port: meServerPort,
        path: '/',
        method: 'GET',
      };

      const req = http.get(options, (res) => {
        if (res.statusCode === 200) {
          meServerReady = true;
          resolve();
        } else {
          // Retry every 1 second
          setTimeout(checkStatus, 1000);
        }
      });
      req.on('error', () => {
        // Retry every 1 second
        setTimeout(checkStatus, 1000);
      });
    };

    checkStatus(); // Start the initial check
  });
}

function createMainWindow() {
  log.info('Creating main window');

  mainWindow = new BrowserWindow({
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
    filePathToOpen = '';
  });

  if (filePathToOpen !== '') {
    loadFiles([filePathToOpen]);
  } else {
    mainWindow.loadURL(`http://127.0.0.1:${meServerPort}`);
  }
  mainWindow.maximize();
  mainWindow.show();

  log.info('Main windown shown');
}

async function getFreePort() {
  for (let port = 8080; port <= 65535; port++) {
    try {
      // Attempt a quick HTTP request to the port
      await new Promise((resolve, reject) => {
        const req = http.request({port}, () => {
          // Close the connection immediately
          req.destroy();
          reject(new Error('Port in use'));
        });

        req.on('error', (err) => {
          if (err.code === 'ECONNREFUSED') {
            // Port likely available
            resolve(port);
          } else {
            // Other error, might not be available
            reject(err);
          }
        });

        req.end();
      });

      // If no error, port is likely available
      return port;
    } catch (err) {
      // Port is likely in use, continue to the next one
    }
  }

  throw new Error('No available ports found');
}

function isDev() {
  return !app.isPackaged;
}

function loadFiles(paths) {
  if (!mainWindow) {
    return;
  }

  const data = {
    'models': paths.map((path) => {
      return {
        'url': path,
      };
    }),
  };
  mainWindow.loadURL(
    `http://127.0.0.1:${meServerPort}/?data=${encodeURIComponent(
      JSON.stringify(data),
    )}`,
  );
}
