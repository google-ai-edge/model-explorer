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

import * as three from 'three';

import {FontWeight, Rect, UniformValue, WebglColor} from './common/types';
import {ThreejsService} from './threejs_service';

const THREE = three;

const VERTEX_SHADER = `
precision highp float;

// Set this to <0 to disable animation.
uniform float animationProgress;

// 0: regular. 1: medium. 2: bold.
attribute float weight;
attribute vec3 color;
attribute vec4 bound;
attribute vec4 targetBound;
attribute vec4 uvBound;
attribute float yOffset;
attribute float opacity;
attribute float angle;
attribute vec3 borderColor;
attribute float weightLevel;

varying vec2 vUv;
varying vec3 vColor;
varying float vWeight;
varying float vOpacity;
varying vec3 vBorderColor;
varying float vWeightLevel;

void main() {
  vUv = vec2(0.0, 0.0);
  vec3 pos = position;
  if (pos.x < 0.0 && pos.z < 0.0) {
    vUv = vec2(uvBound.x, uvBound.y + uvBound.w);
  } else if (pos.x > 0.0 && pos.z < 0.0) {
    vUv = vec2(uvBound.x + uvBound.z, uvBound.y + uvBound.w);
  } else if (pos.x > 0.0 && pos.z > 0.0) {
    vUv = vec2(uvBound.x + uvBound.z, uvBound.y);
  } else {
    vUv = vec2(uvBound.x, uvBound.y);
  }

  vColor = color;
  vWeight = weight;
  vOpacity = opacity;
  vBorderColor = borderColor;
  vWeightLevel = weightLevel;

  float curX = bound.x;
  float curY = bound.y;
  float curW = bound.z;
  float curH = bound.w;

  float x = curX;
  float y = curY;
  float w = curW;
  float h = curH;

  if (animationProgress >= 0.0) {
    x = curX + (targetBound.x - curX) * animationProgress;
    y = curY + (targetBound.y - curY) * animationProgress;
    w = curW + (targetBound.z - curW) * animationProgress;
    h = curH + (targetBound.w - curH) * animationProgress;
  }

  // For each vertex, move it by delta calculated below so that the final
  // rectangle's width and height match the width and height stored in "bound".
  //
  if (pos.x < 0.0) {
    pos.x = - w / 2.0;
  } else if (pos.x > 0.0) {
    pos.x = w / 2.0;
  }

  if (pos.z < 0.0) {
    pos.z = - h / 2.0;
  } else if (pos.z > 0.0) {
    pos.z = h / 2.0;
  }

  // Rotate.
  //
  float c = cos(angle);
  float s = sin(angle);
  float posX = pos.x * c + pos.z * s;
  float posZ = pos.z * c - pos.x * s;

  gl_Position = projectionMatrix * modelViewMatrix *
      vec4(posX + x, yOffset, posZ + y, 1.0);
}
`;

// https://github.com/Blatko1/awesome-msdf?tab=readme-ov-file#basic-msdf-usage
const FRAGMENT_SHADER = `
precision highp float;

uniform sampler2D textureRegular;
uniform sampler2D textureMedium;
uniform sampler2D textureBold;
uniform sampler2D textureIcons;
uniform float distanceRange;
varying vec2 vUv;
varying float vWeight;
varying vec3 vColor;
varying float vOpacity;
varying vec3 vBorderColor;
varying float vWeightLevel;

vec3 getSampleColor() {
  if (vWeight < 0.5) {
    return texture2D(textureRegular, vUv).rgb;
  }
  if (vWeight < 1.5) {
    return texture2D(textureMedium, vUv).rgb;
  }
  if (vWeight < 2.5) {
    return texture2D(textureBold, vUv).rgb;
  }
  return texture2D(textureIcons, vUv).rgb;
}

float median(float r, float g, float b) {
  return max(min(r, g), min(max(r, g), b));
}

float screenPxRange() {
  vec2 unitRange = vec2(distanceRange) / vec2(textureSize(textureRegular, 0));
  vec2 screenTexSize = vec2(1.0) / fwidth(vUv);
  return max(0.5 * dot(unitRange, screenTexSize), 1.0);
}

void main() {
  vec3 sampleColor = getSampleColor();

  float dist = median(sampleColor.r, sampleColor.g, sampleColor.b);
  vec3 color = vColor;
  float threshold = step(0.0, vBorderColor.r) * 0.35;
  float pxDist = screenPxRange() * (dist - 0.5 + threshold) + vWeightLevel - threshold;

  // Has border.
  if (vBorderColor.r >= 0.0) {
    float blur = fwidth(dist) / 2.0;
    color = mix(vBorderColor, vColor, smoothstep(0.5 - blur, 0.5 + blur, dist));
  }

  float opacity = clamp(pxDist, 0.0, 1.0);
  if (opacity > 0.001) {
    opacity *= vOpacity;
  }
  gl_FragColor = vec4(color, opacity);
}
`;

interface LabelCharSizes {
  rects: Rect[];
  minX: number;
  minZ: number;
  maxX: number;
  maxZ: number;
}

/** Data for one label text to render. */
export interface LabelData {
  id: string;
  nodeId?: string;
  label: string;
  height: number;
  hAlign: 'center' | 'left' | 'right' | '';
  vAlign: 'center' | 'top' | 'bottom' | '';
  weight: FontWeight;
  x: number;
  y: number;
  z: number;
  // Default to black.
  color?: WebglColor;
  maxWidth?: number;
  // Used in icons font where labels are used as keys to look up char info in
  // the font json file.
  treatLabelAsAWhole?: boolean;
  // Angle in radians.
  angle?: number;
  // Mode used for rendering text along edges.
  edgeTextMode?: boolean;
  // Default to (-1, -1, -1) which will disable border.
  borderColor?: WebglColor;
  // Default to 0.5. It adds/subtracts weight from `weight` above.
  weightLevel?: number;
}

/**
 * A class for generating and managing texts mesh for webgl renderer.
 */
export class WebglTexts {
  mesh?: three.Mesh;
  material: three.ShaderMaterial;

  private readonly planeGeo: three.PlaneGeometry;
  private readonly labelCharSizesCache: Record<string, LabelCharSizes> = {};
  private readonly fontSize: number;
  private readonly distanceRange: number;
  private savedBounds: Record<string, Rect> = {};
  private nodeIdToOpacityIndexRanges: Record<
    string,
    Array<{minIndex: number; maxIndex: number}>
  > = {};
  private nodeIdToColorIndexRanges: Record<
    string,
    Array<{minIndex: number; maxIndex: number}>
  > = {};
  private lastOpacityUpdateIndexRanges: Array<{
    minIndex: number;
    maxIndex: number;
  }> = [];
  private lastColorUpdateIndexRanges: Array<{
    minIndex: number;
    maxIndex: number;
  }> = [];
  private originalColors: number[] = [];
  private readonly curAnimationProgrssUniform: UniformValue = {value: -1};

  constructor(private readonly threejsService: ThreejsService) {
    this.planeGeo = new THREE.PlaneGeometry(1, 1);
    this.planeGeo.rotateX(-Math.PI / 2);

    // Use the regular font to retrieve various metadata.
    this.fontSize = this.threejsService.fontInfoRegular.info.size;
    this.distanceRange =
      this.threejsService.fontInfoRegular.distanceField.distanceRange;

    this.material = new THREE.ShaderMaterial({
      uniforms: {
        'textureRegular': {
          value: this.threejsService.textureRegular,
        },
        'textureMedium': {
          value: this.threejsService.textureMedium,
        },
        'textureBold': {
          value: this.threejsService.textureBold,
        },
        'textureIcons': {
          value: this.threejsService.textureIcons,
        },
        'distanceRange': {
          value: this.distanceRange,
        },
        'animationProgress': this.curAnimationProgrssUniform,
      },
      extensions: {
        derivatives: true,
      },
      vertexShader: VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
      transparent: true,
      alphaToCoverage: true,
    });
  }

  generateMesh(
    labels: LabelData[],
    forceNoAnimation = false,
    forceAnimation = false,
    noInitAnimation = false,
  ) {
    let index = 0;
    this.nodeIdToOpacityIndexRanges = {};
    this.nodeIdToColorIndexRanges = {};

    const weights: number[] = [];
    const colors: number[] = [];
    const bounds: number[] = [];
    const targetBounds: number[] = [];
    const yOffsets: number[] = [];
    const opacities: number[] = [];
    const uvBounds: number[] = [];
    const angles: number[] = [];
    const borderColors: number[] = [];
    const weightLevels: number[] = [];

    const needAnimation = Object.keys(this.savedBounds).length > 0;
    const savedBounds = {...this.savedBounds};
    this.savedBounds = {};

    for (const lb of labels) {
      const charsInfo = this.threejsService.getCharsInfo(lb.weight);
      const fontInfo = this.threejsService.getFontInfo(lb.weight);
      const atlasSize = fontInfo.common.scaleW;

      // Calculate unscaled rect for each character, starting at (0, 0).
      const sizes = this.getLabelSizes(
        lb.label,
        lb.weight,
        lb.height,
        lb.maxWidth,
        lb.treatLabelAsAWhole,
        lb.angle,
        lb.edgeTextMode,
      ).sizes;
      const scale = lb.height / this.fontSize;

      let startCharX = lb.x;
      let startCharZ = lb.z;
      const aSizes = lb.treatLabelAsAWhole
        ? {
            rects: [
              {
                x: 0,
                y: 0,
                width: 20,
                height: lb.height,
              },
            ],
          }
        : this.getLabelSizes('a', lb.weight, lb.height).sizes;

      const aHeight = aSizes.rects[0].height * scale;
      switch (lb.vAlign) {
        case 'top':
          startCharZ -= sizes.minZ * scale;
          break;
        case 'bottom':
          startCharZ -= sizes.maxZ * scale;
          break;
        case 'center':
          startCharZ -= ((sizes.minZ + sizes.maxZ) / 2) * scale + aHeight / 2;
          break;
        default:
          break;
      }
      switch (lb.hAlign) {
        case 'left':
          startCharX -= sizes.minX * scale;
          break;
        case 'right':
          startCharX -= sizes.maxX * scale;
          break;
        case 'center':
          startCharX -= ((sizes.minX + sizes.maxX) / 2) * scale;
          break;
        default:
          break;
      }
      const y = lb.y;
      const startCharIndex = opacities.length;
      for (let i = 0; i < sizes.rects.length; i++) {
        const rect = sizes.rects[i];
        const char = lb.treatLabelAsAWhole ? lb.label : lb.label[i];
        const charInfo = charsInfo[char] || charsInfo['?'];
        const id = `${lb.id}_${char}_${i}`;

        const charWidth = rect.width * scale;
        const charHeight = rect.height * scale;
        const charX = startCharX + rect.x * scale;
        const charZ = startCharZ + rect.y * scale;

        const curBound: Rect = {
          x: lb.edgeTextMode ? charX : charX + charWidth / 2,
          y: lb.edgeTextMode ? charZ : charZ + charHeight / 2,
          width: charWidth,
          height: charHeight,
        };
        const savedBound = savedBounds[id];
        if ((!needAnimation || forceNoAnimation) && !forceAnimation) {
          bounds.push(curBound.x, curBound.y, curBound.width, curBound.height);
        } else {
          bounds.push(
            savedBound?.x ?? curBound.x,
            savedBound?.y ?? curBound.y,
            savedBound?.width ?? (noInitAnimation ? curBound.width : 0),
            savedBound?.height ?? (noInitAnimation ? curBound.height : 0),
          );
        }
        targetBounds.push(
          curBound.x,
          curBound.y,
          curBound.width,
          curBound.height,
        );
        yOffsets.push(y);

        // UV.
        const u = charInfo.x / atlasSize;
        const v = 1 - charInfo.y / atlasSize - charInfo.height / atlasSize;
        const uvWidth = charInfo.width / atlasSize;
        const uvHeight = charInfo.height / atlasSize;
        uvBounds.push(u, v, uvWidth, uvHeight);

        // Color and font weights.
        const r = lb.color?.r ?? 0;
        const g = lb.color?.g ?? 0;
        const b = lb.color?.b ?? 0;
        colors.push(r, g, b);
        weights.push(lb.weight);
        opacities.push(1);
        angles.push(lb.angle || 0);
        borderColors.push(
          lb.borderColor?.r ?? -1,
          lb.borderColor?.g ?? -1,
          lb.borderColor?.b ?? -1,
        );
        weightLevels.push(lb.weightLevel ?? 0.5);

        this.savedBounds[id] = curBound;
        index++;
      }
      if (lb.nodeId) {
        if (!this.nodeIdToOpacityIndexRanges[lb.nodeId]) {
          this.nodeIdToOpacityIndexRanges[lb.nodeId] = [];
        }
        this.nodeIdToOpacityIndexRanges[lb.nodeId].push({
          minIndex: startCharIndex,
          maxIndex: opacities.length - 1,
        });

        if (!this.nodeIdToColorIndexRanges[lb.nodeId]) {
          this.nodeIdToColorIndexRanges[lb.nodeId] = [];
        }
        this.nodeIdToColorIndexRanges[lb.nodeId].push({
          minIndex: startCharIndex,
          maxIndex: opacities.length - 1,
        });
      }
    }
    this.originalColors = colors;

    const geometry = new THREE.InstancedBufferGeometry().copy(this.planeGeo);
    geometry.instanceCount = yOffsets.length;
    geometry.setAttribute(
      'bound',
      new THREE.InstancedBufferAttribute(new Float32Array(bounds), 4),
    );
    geometry.setAttribute(
      'targetBound',
      new THREE.InstancedBufferAttribute(new Float32Array(targetBounds), 4),
    );
    geometry.setAttribute(
      'uvBound',
      new THREE.InstancedBufferAttribute(new Float32Array(uvBounds), 4),
    );
    geometry.setAttribute(
      'yOffset',
      new THREE.InstancedBufferAttribute(new Float32Array(yOffsets), 1),
    );
    geometry.setAttribute(
      'weight',
      new THREE.InstancedBufferAttribute(new Float32Array(weights), 1),
    );
    geometry.setAttribute(
      'color',
      new THREE.InstancedBufferAttribute(new Float32Array(colors), 3),
    );
    geometry.setAttribute(
      'opacity',
      new THREE.InstancedBufferAttribute(new Float32Array(opacities), 1),
    );
    geometry.setAttribute(
      'angle',
      new THREE.InstancedBufferAttribute(new Float32Array(angles), 1),
    );
    geometry.setAttribute(
      'borderColor',
      new THREE.InstancedBufferAttribute(new Float32Array(borderColors), 3),
    );
    geometry.setAttribute(
      'weightLevel',
      new THREE.InstancedBufferAttribute(new Float32Array(weightLevels), 1),
    );

    this.mesh = new THREE.Mesh(geometry, this.material);
    this.mesh.frustumCulled = false;
  }

  updateOpacityInNode(nodeIds: string[], opacity: number) {
    if (!this.mesh) {
      return;
    }

    const opacityAttrs = this.mesh.geometry.getAttribute('opacity');
    if (nodeIds.length > 0) {
      for (const nodeId of nodeIds) {
        if (this.nodeIdToOpacityIndexRanges[nodeId]) {
          for (const {minIndex, maxIndex} of this.nodeIdToOpacityIndexRanges[
            nodeId
          ]) {
            for (let i = minIndex; i <= maxIndex; i++) {
              opacityAttrs.setX(i, opacity);
            }
            this.lastOpacityUpdateIndexRanges.push({minIndex, maxIndex});
          }
        }
      }
    }
    opacityAttrs.needsUpdate = true;
  }

  updateColorInNode(nodeIds: string[], color: three.Color) {
    if (!this.mesh) {
      return;
    }

    const colorAttrs = this.mesh.geometry.getAttribute('color');
    if (nodeIds.length > 0) {
      for (const nodeId of nodeIds) {
        if (this.nodeIdToColorIndexRanges[nodeId]) {
          for (const {minIndex, maxIndex} of this.nodeIdToColorIndexRanges[
            nodeId
          ]) {
            for (let i = minIndex; i <= maxIndex; i++) {
              colorAttrs.setXYZ(i, color.r, color.g, color.b);
            }
            this.lastColorUpdateIndexRanges.push({minIndex, maxIndex});
          }
        }
      }
    }
    colorAttrs.needsUpdate = true;
  }

  restoreOpacities() {
    if (!this.mesh) {
      return;
    }

    const opacityAttrs = this.mesh.geometry.getAttribute('opacity');
    if (this.lastOpacityUpdateIndexRanges.length >= 0) {
      for (const {minIndex, maxIndex} of this.lastOpacityUpdateIndexRanges) {
        for (let i = minIndex; i <= maxIndex; i++) {
          opacityAttrs.setX(i, 1);
        }
      }
      opacityAttrs.needsUpdate = true;
    }
    this.lastOpacityUpdateIndexRanges = [];
  }

  restoreColors() {
    if (!this.mesh) {
      return;
    }

    const colorAttrs = this.mesh.geometry.getAttribute('color');
    if (this.lastColorUpdateIndexRanges.length >= 0) {
      for (const {minIndex, maxIndex} of this.lastColorUpdateIndexRanges) {
        for (let i = minIndex; i <= maxIndex; i++) {
          colorAttrs.setXYZ(
            i,
            this.originalColors[i * 3],
            this.originalColors[i * 3 + 1],
            this.originalColors[i * 3 + 2],
          );
        }
      }
      colorAttrs.needsUpdate = true;
    }
    this.lastColorUpdateIndexRanges = [];
  }

  updateAnimationProgress(progress: number) {
    this.curAnimationProgrssUniform.value = progress;
  }

  getLabelSizes(
    label: string,
    weight: FontWeight,
    height: number,
    maxWidth?: number,
    treatLabelAsAWhole?: boolean,
    angle?: number,
    edgeTextMode?: boolean,
  ): {sizes: LabelCharSizes; updatedLabel?: string} {
    let key = this.getLabelCharSizesKey(label, weight, angle);
    let sizes = this.labelCharSizesCache[key];
    let updatedLabel: string | undefined = undefined;
    if (sizes == null) {
      sizes = this.getLabelSizesInternal(
        label,
        weight,
        treatLabelAsAWhole,
        angle,
        edgeTextMode,
      );

      // Handle max width.
      if (maxWidth != null) {
        const scale = height / this.fontSize;
        let curLabel = label;
        for (let i = 0; i < sizes.rects.length; i++) {
          const rect = sizes.rects[i];
          if ((rect.x + rect.width) * scale > maxWidth) {
            curLabel = curLabel.substring(0, i - 1);
            curLabel += '...';
            break;
          }
        }
        updatedLabel = curLabel;
        sizes = this.getLabelSizesInternal(
          curLabel,
          weight,
          treatLabelAsAWhole,
        );
        key = this.getLabelCharSizesKey(curLabel, weight);
      }
      this.labelCharSizesCache[key] = sizes;
    }
    return {sizes: this.labelCharSizesCache[key], updatedLabel};
  }

  updateLabelSizesCache(
    label: string,
    weight: FontWeight,
    sizes: LabelCharSizes,
  ) {
    const key = this.getLabelCharSizesKey(label, weight);
    this.labelCharSizesCache[key] = sizes;
  }

  getFontSize(): number {
    return this.fontSize;
  }

  clearSavedDataForAnimation() {
    this.savedBounds = {};
  }

  private getLabelCharSizesKey(
    label: string,
    weight: FontWeight,
    angle?: number,
  ) {
    return `${label}__${weight}__${angle}`;
  }

  private getLabelSizesInternal(
    label: string,
    weight: FontWeight,
    treatLabelAsAWhole?: boolean,
    angle?: number,
    edgeTextMode?: boolean,
  ): LabelCharSizes {
    const charsInfo = this.threejsService.getCharsInfo(weight);
    const rects: Rect[] = [];
    let curCharX = 0;
    let minX = Number.MAX_VALUE;
    let minZ = Number.MAX_VALUE;
    let maxX = Number.NEGATIVE_INFINITY;
    let maxZ = Number.NEGATIVE_INFINITY;
    for (const char of treatLabelAsAWhole ? [label] : label) {
      let charInfo = charsInfo[char];
      if (!charInfo) {
        charInfo = charsInfo['?'];
      }
      let charZ = charInfo.yoffset;
      if (!edgeTextMode) {
        curCharX += charInfo.xoffset;
      } else {
        curCharX +=
          Math.sin(angle || 0) * (charInfo.yoffset + charInfo.height / 2);
        charZ = Math.cos(angle || 0) * (charInfo.yoffset + charInfo.height / 2);
      }
      const width = charInfo.width;
      const height = charInfo.height;
      const rect: Rect = {x: curCharX, y: charZ, width, height};
      rects.push(rect);
      minX = Math.min(minX, rect.x);
      minZ = Math.min(minZ, rect.y);
      maxX = Math.max(maxX, rect.x + rect.width);
      maxZ = Math.max(maxZ, rect.y + rect.height);
      curCharX += charInfo.xadvance * 0.98;
    }
    return {
      rects,
      minX,
      minZ: 0,
      maxX,
      maxZ: this.fontSize,
    };
  }
}
