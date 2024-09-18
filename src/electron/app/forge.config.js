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

const {FusesPlugin} = require('@electron-forge/plugin-fuses');
const {FuseV1Options, FuseVersion} = require('@electron/fuses');
const path = require('path');
const fs = require('node:fs/promises');

module.exports = {
  packagerConfig: {
    asar: false,
    ignore: [
      /^\/icon_sets/,
      /^\/node_modules/,
      /^\/.gitignore/,
      /^\/forge.config.js/,
    ],
    name: 'Model Explorer',
    icon: './icon_sets/app_icon',
    extraResource: [
      './icon_sets/doc_icon.icns',
    ],
    extendInfo: {
      CFBundleDocumentTypes: [
        {
          CFBundleTypeExtensions: ['tflite'],
          CFBundleTypeName: 'Model Explorer',
          CFBundleTypeRole: 'Editor',
          CFBundleTypeIconFile: 'doc_icon.icns',
          LSTypeIsPackage: true,
          LSHandlerRank: 'Default',
        },
      ],
    },
  },
  rebuildConfig: {},
  makers: [
    {
      name: '@electron-forge/maker-squirrel',
      config: {},
    },
    {
      name: '@electron-forge/maker-zip',
      platforms: ['darwin'],
    },
    // {
    //   name: "@electron-forge/maker-dmg",
    //   config: {
    //     "title": "Model Explorer",
    //     "icon": "icon_sets/app_icon.icns",
    //     "format": "ULFO"
    //   }
    // },
    {
      name: '@electron-forge/maker-deb',
      config: {},
    },
    {
      name: '@electron-forge/maker-rpm',
      config: {},
    },
  ],
  plugins: [
    // {
    //   name: '@electron-forge/plugin-auto-unpack-natives',
    //   config: {},
    // },
    // Fuses are used to enable/disable various Electron functionality
    // at package time, before code signing the application
    new FusesPlugin({
      version: FuseVersion.V1,
      [FuseV1Options.RunAsNode]: false,
      [FuseV1Options.EnableCookieEncryption]: true,
      [FuseV1Options.EnableNodeOptionsEnvironmentVariable]: false,
      [FuseV1Options.EnableNodeCliInspectArguments]: false,
      [FuseV1Options.EnableEmbeddedAsarIntegrityValidation]: false,
      [FuseV1Options.OnlyLoadAppFromAsar]: false,
    }),
  ],
  hooks: {
    postMake: async (forgeConfig, makeResults) => {
      // Rename the packaged file name from "Model Explorer-xxx.zip" to
      // "model-explorer-xxx.zip".
      for (const makeResult of makeResults) {
        const firstArtifact = makeResult.artifacts[0];
        const newName = firstArtifact.replace('Model Explorer', 'model-explorer');
        await fs.rename(firstArtifact, newName)
      }
    },
  },
};
