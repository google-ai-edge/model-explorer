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

import {Injectable} from '@angular/core';
import * as three from 'three';

import {CharInfo, FontWeight} from './common/types';

const THREE = three;

declare interface FontInfo {
  chars: CharInfo[];
  info: FontBasicInfo;
  common: CommonInfo;
  distanceField: DistanceFieldInfo;
}

declare interface FontBasicInfo {
  size: number;
}

declare interface CommonInfo {
  scaleW: number;
}

declare interface DistanceFieldInfo {
  distanceRange: number;
}

/**
 * A service to manage threejs related tasks.
 */
@Injectable({providedIn: 'root'})
export class ThreejsService {
  charsInfoRegular: Record<string, CharInfo> = {};
  charsInfoMedium: Record<string, CharInfo> = {};
  charsInfoBold: Record<string, CharInfo> = {};
  charsInfoIcons: Record<string, CharInfo> = {};
  textureRegular!: three.Texture;
  textureMedium!: three.Texture;
  textureBold!: three.Texture;
  textureIcons!: three.Texture;

  fontInfoRegular!: FontInfo;
  fontInfoMedium!: FontInfo;
  fontInfoBold!: FontInfo;
  fontInfoIcons!: FontInfo;

  readonly depsLoadedPromise: Promise<void>;

  constructor() {
    this.depsLoadedPromise = new Promise(async (resolve) => {
      await this.loadDeps();
      resolve();
    });
  }

  private async loadDeps() {

    // Run the following command to generate them.
    //
    // (Install: $ npm install msdf-bmfont-xml -g)
    //
    // $ msdf-bmfont -f json -m 512,512 -s 50 -r 6 -d 3 ./FontRegular.ttf
    let staticUrlBase =
      (window as any)['modelExplorer']?.assetFilesBaseUrl ?? 'static_files';

    const results = await Promise.all([
      this.loadFontAtals(`${staticUrlBase}/FontRegular.png`),
      this.loadFontAtals(`${staticUrlBase}/FontMedium.png`),
      this.loadFontAtals(`${staticUrlBase}/FontBold.png`),
      this.loadFontAtals(`${staticUrlBase}/icons_20240521.png`),
      this.loadFontInfo(`${staticUrlBase}/FontRegular.json`),
      this.loadFontInfo(`${staticUrlBase}/FontMedium.json`),
      this.loadFontInfo(`${staticUrlBase}/FontBold.json`),
      this.loadFontInfo(`${staticUrlBase}/icons_20240521.json`),
    ]);
    this.textureRegular = results[0];
    this.textureMedium = results[1];
    this.textureBold = results[2];
    this.textureIcons = results[3];
    this.charsInfoRegular = results[4].charsInfo;
    this.charsInfoMedium = results[5].charsInfo;
    this.charsInfoBold = results[6].charsInfo;
    this.charsInfoIcons = results[7].charsInfo;
    this.fontInfoRegular = results[4].fontInfo;
    this.fontInfoMedium = results[5].fontInfo;
    this.fontInfoBold = results[6].fontInfo;
    this.fontInfoIcons = results[7].fontInfo;
  }

  getCharsInfo(weight: FontWeight) {
    switch (weight) {
      case FontWeight.REGULAR:
        return this.charsInfoRegular;
      case FontWeight.MEDIUM:
        return this.charsInfoMedium;
      case FontWeight.BOLD:
        return this.charsInfoBold;
      case FontWeight.ICONS:
        return this.charsInfoIcons;

      default:
        return this.charsInfoRegular;
    }
  }

  getFontInfo(weight: FontWeight): FontInfo {
    switch (weight) {
      case FontWeight.REGULAR:
        return this.fontInfoRegular;
      case FontWeight.MEDIUM:
        return this.fontInfoMedium;
      case FontWeight.BOLD:
        return this.fontInfoBold;
      case FontWeight.ICONS:
        return this.fontInfoIcons;

      default:
        return this.fontInfoRegular;
    }
  }

  private async loadFontAtals(url: string): Promise<three.Texture> {
    return new Promise<three.Texture>((resolve) => {
      new THREE.TextureLoader().load(url, (texture) => {
        resolve(texture);
      });
    });
  }

  private async loadFontInfo(
    url: string,
  ): Promise<{fontInfo: FontInfo; charsInfo: Record<string, CharInfo>}> {
    const resp = await fetch(url);
    const fontInfo = (await resp.json()) as FontInfo;
    const charsInfo: Record<string, CharInfo> = {};
    for (const charInfo of fontInfo.chars) {
      charsInfo[charInfo.char] = charInfo;
    }
    return {fontInfo, charsInfo};
  }
}
