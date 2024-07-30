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

import {Rect, UniformValue, WebglColor} from './common/types';

const THREE = three;

const VERTEX_SHADER = `
precision highp float;

uniform float borderRadius;
// Set this to <0 to disable animation.
uniform float animationProgress;
uniform vec4 bgColorWhenFar;

attribute vec4 bound;
attribute vec4 targetBound;
attribute float yOffset;
attribute float isRounded;
attribute float borderWidth;
attribute vec3 bgColor;
attribute vec3 borderColor;
attribute float angle;
attribute float opacity;
attribute float changeColorWhenFar;

varying vec2 vUv;
varying vec2 vSize;
varying float vIsRounded;
varying float vBorderWidth;
varying vec3 vBgColor;
varying vec3 vBorderColor;
varying float vOpacity;
varying float vChangeColorWhenFar;

void main() {
  vUv = uv;
  vIsRounded = isRounded;
  vBorderWidth = borderWidth;
  vBgColor = bgColor;
  vBorderColor = borderColor;
  vOpacity = opacity;
  vChangeColorWhenFar = changeColorWhenFar;

  vec3 pos = position;
  float curX = bound.x;
  float curY = bound.y;
  float curW = bound.z;
  float curH = bound.w;

  float progress = animationProgress * step(0.0, animationProgress); 
  float x = curX + (targetBound.x - curX) * progress;
  float y = curY + (targetBound.y - curY) * progress;
  float w = curW + (targetBound.z - curW) * progress;
  float h = curH + (targetBound.w - curH) * progress;
  vSize = vec2(w, h);

  // For each vertex, move it by delta calculated below so that the final
  // rectangle's width and height match the width and height stored in "bound".
  //
  // pos.x < 0: the first () below returns -1.
  // pos.x > 0: the first () below returns 1.
  pos.x = (step(0.0, pos.x) * 2.0 - 1.0) * (w / 2.0);
  pos.z = (step(0.0, pos.z) * 2.0 - 1.0) * (h / 2.0);

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
const FRAGMENT_SHADER = `
precision highp float;

uniform float borderRadius;
uniform vec4 bgColorWhenFar;

varying vec2 vUv;
varying vec2 vSize;
varying float vIsRounded;
varying float vBorderWidth;
varying vec3 vBgColor;
varying vec3 vBorderColor;
varying float vOpacity;
varying float vChangeColorWhenFar;

// See:
// https://www.shadertoy.com/view/4tc3DX#

// Clamp [0..1] range
#define saturate(a) clamp(a, 0.0, 1.0)

// This function will make a signed distance field that says how far you are
// from the edge of the line at any point U,V.
// Pass it UVs, line end points, line thickness (x is along the line and y is
// perpendicular), How rounded the end points should be (0.0 is rectangular,
// setting rounded to thick.y will be circular).
float LineDistField(vec2 uv, vec2 pA, vec2 pB, vec2 thick, float rounded, float dashOn) {
  // Don't let it get more round than circular.
  rounded = min(thick.y, rounded);
  // midpoint
  vec2 mid = (pB + pA) * 0.5;
  // vector from point A to B
  vec2 delta = pB - pA;
  // Distance between endpoints
  float lenD = length(delta);
  // unit vector pointing in the line's direction
  vec2 unit = delta / lenD;
  // Check for when line endpoints are the same
  if (lenD < 0.0001) unit = vec2(1.0, 0.0);	// if pA and pB are same
  // Perpendicular vector to unit - also length 1.0
  vec2 perp = unit.yx * vec2(-1.0, 1.0);
  // position along line from midpoint
  float dpx = dot(unit, uv - mid);
  // distance away from line at a right angle
  float dpy = dot(perp, uv - mid);
  // Make a distance function that is 0 at the transition from black to white
  float disty = abs(dpy) - thick.y + rounded;
  float distx = abs(dpx) - lenD * 0.5 - thick.x + rounded;

  // Too tired to remember what this does. Something like rounded endpoints for distance function.
  float dist = length(vec2(max(0.0, distx), max(0.0,disty))) - rounded;
  dist = min(dist, max(distx, disty));

  return dist;
}

// This makes a line in UV units. A 1.0 thick line will span a whole 0..1 in
// UV space.
float FillLine(vec2 uv, vec2 pA, vec2 pB, vec2 thick, float rounded) {
  float df = LineDistField(uv, pA, pB, vec2(thick), rounded, 0.0);
  return saturate(df / abs(dFdy(uv).y));
}

void main() {
  // Normalize uv.
  vec2 uv = vUv;
  uv -= 0.5;
  float aspect = vSize.x / vSize.y;
  uv.x *= aspect;

  vec4 finalColor = vec4(1.0);

  float radius = mix(1.0 / vSize.y, borderRadius / vSize.y, step(0.5, vIsRounded));
  float borderWidth = vBorderWidth / vSize.y;

  // Border.
  float c = FillLine(uv,
    vec2(-0.5 * aspect, 0.0), vec2(0.5 * aspect, 0.0), vec2(0.0, 0.5), radius);
  finalColor = mix(vec4(vBorderColor.rgb, 1.0), vec4(1.0, 1.0, 1.0, 0.0), c);

  // Body.
  float c2 = FillLine(uv,
    vec2(-0.5 * aspect + borderWidth, 0.0),
    vec2(0.5 * aspect - borderWidth, 0.0), vec2(0.0, 0.5 - borderWidth),
    radius - 1.0 / vSize.y);
  finalColor = mix(vec4(vBgColor.rgb, 1.0), finalColor, c2);
  finalColor = saturate(finalColor);
  finalColor.rgb = mix(finalColor.rgb, bgColorWhenFar.rgb, bgColorWhenFar.a * vChangeColorWhenFar);

  float alpha = finalColor.w * vOpacity;
  if (alpha < 0.00001) {
    discard;
  }
  gl_FragColor = vec4(finalColor.rgb, alpha);
}
`;

/** Data for one rounded rectangle. */
export interface RoundedRectangleData {
  index: number;
  id: string;
  bound: Rect;
  isRounded: boolean;
  yOffset: number;
  bgColor: WebglColor;
  borderColor: WebglColor;
  borderWidth: number;
  opacity: number;
  nodeId?: string;
  changeColorWhenFar?: boolean;
}

/**
 * A class for generating and managing rounded rectangles mesh for webgl
 * renderer.
 */
export class WebglRoundedRectangles {
  mesh?: three.Mesh;
  meshForRayCasting?: three.InstancedMesh;
  material: three.ShaderMaterial;
  materialForRayCasting: three.MeshBasicMaterial;
  planeGeo: three.PlaneGeometry;

  private hoveredRectangelId = '';
  private curRectangles: RoundedRectangleData[] = [];
  private savedRectangles: Record<string, RoundedRectangleData> = {};
  private readonly curAnimationProgrssUniform: UniformValue = {value: -1};
  private readonly dummy = new THREE.Object3D();
  private originalBorderColors: number[] = [];
  private originalBgColors: number[] = [];
  private originalBorderWidths: number[] = [];
  private originalOpacities: number[] = [];
  private lastBorderColorUpdateRectangles: RoundedRectangleData[] = [];
  private lastBgColorUpdateRectangles: RoundedRectangleData[] = [];
  private lastBorderWidthUpdateRectangles: RoundedRectangleData[] = [];
  private lastOpacityUpdateRectangles: RoundedRectangleData[] = [];

  constructor(private readonly radius: number) {
    this.planeGeo = new THREE.PlaneGeometry(1, 1);
    this.planeGeo.rotateX(-Math.PI / 2);

    this.material = new THREE.ShaderMaterial({
      extensions: {
        derivatives: true,
      },
      uniforms: {
        'borderRadius': {value: this.radius},
        'animationProgress': this.curAnimationProgrssUniform,
        'bgColorWhenFar': {value: [0, 0, 0, 0]},
      },
      vertexShader: VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
      transparent: true,
    });
    this.materialForRayCasting = new THREE.MeshBasicMaterial({
      opacity: 0,
      transparent: true,
    });
  }

  generateMesh(
    rectangles: RoundedRectangleData[],
    createRayCastingMesh = false,
    forceNoAnimation = false,
    forceAnimation = false,
    noInitAnimation = false,
  ) {
    if (rectangles.length === 0) {
      this.savedRectangles = {};
      this.mesh = undefined;
      this.meshForRayCasting = undefined;
      return;
    }

    this.curRectangles = rectangles;

    const bounds: number[] = [];
    const yOffsets: number[] = [];
    const targetBounds: number[] = [];
    const isRounded: number[] = [];
    const borderWidths: number[] = [];
    const bgColors: number[] = [];
    const borderColors: number[] = [];
    const angles: number[] = [];
    const opacities: number[] = [];
    const changeColorWhenFarValues: number[] = [];

    const needAnimation = Object.keys(this.savedRectangles).length > 0;
    const savedRectangleBounds = {...this.savedRectangles};
    this.savedRectangles = {};

    let index = 0;
    for (let i = 0; i < rectangles.length; i++) {
      const rectangle = rectangles[i];

      const curBound = rectangle.bound;
      const savedBound = savedRectangleBounds[rectangle.id]?.bound;
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

      yOffsets.push(rectangle.yOffset);

      isRounded.push(rectangle.isRounded ? 1 : 0);
      borderWidths.push(rectangle.borderWidth);
      bgColors.push(
        rectangle.bgColor.r,
        rectangle.bgColor.g,
        rectangle.bgColor.b,
      );
      borderColors.push(
        rectangle.borderColor.r,
        rectangle.borderColor.g,
        rectangle.borderColor.b,
      );
      angles.push(0);
      opacities.push(rectangle.opacity);
      changeColorWhenFarValues.push(rectangle.changeColorWhenFar ? 1 : 0);

      this.savedRectangles[rectangle.id] = rectangle;
      index++;
    }
    this.originalBorderColors = borderColors;
    this.originalBgColors = bgColors;
    this.originalBorderWidths = borderWidths;
    this.originalOpacities = opacities;

    const geometry = new THREE.InstancedBufferGeometry().copy(this.planeGeo);
    geometry.instanceCount = rectangles.length;
    geometry.setAttribute(
      'bound',
      new THREE.InstancedBufferAttribute(new Float32Array(bounds), 4),
    );
    geometry.setAttribute(
      'targetBound',
      new THREE.InstancedBufferAttribute(new Float32Array(targetBounds), 4),
    );
    geometry.setAttribute(
      'yOffset',
      new THREE.InstancedBufferAttribute(new Float32Array(yOffsets), 1),
    );
    geometry.setAttribute(
      'isRounded',
      new THREE.InstancedBufferAttribute(new Float32Array(isRounded), 1),
    );
    geometry.setAttribute(
      'borderWidth',
      new THREE.InstancedBufferAttribute(new Float32Array(borderWidths), 1),
    );
    geometry.setAttribute(
      'bgColor',
      new THREE.InstancedBufferAttribute(new Float32Array(bgColors), 3),
    );
    geometry.setAttribute(
      'borderColor',
      new THREE.InstancedBufferAttribute(new Float32Array(borderColors), 3),
    );
    geometry.setAttribute(
      'angle',
      new THREE.InstancedBufferAttribute(new Float32Array(angles), 1),
    );
    geometry.setAttribute(
      'opacity',
      new THREE.InstancedBufferAttribute(new Float32Array(opacities), 1),
    );
    geometry.setAttribute(
      'changeColorWhenFar',
      new THREE.InstancedBufferAttribute(
        new Float32Array(changeColorWhenFarValues),
        1,
      ),
    );

    this.mesh = new THREE.Mesh(geometry, this.material);
    this.mesh.frustumCulled = false;

    // Create a instanced mesh for the purpose of ray casting.
    // It is invisible visually.
    if (createRayCastingMesh) {
      this.meshForRayCasting = new THREE.InstancedMesh(
        new THREE.BoxGeometry(1, 1, 1),
        this.materialForRayCasting,
        rectangles.length,
      );
      for (let i = 0; i < rectangles.length; i++) {
        const rectangle = rectangles[i];
        const curBound = rectangle.bound;
        this.setInstancePositionAndScale(
          this.meshForRayCasting,
          i,
          curBound.x,
          -10,
          curBound.y,
          curBound.width,
          curBound.height,
        );
      }
      this.meshForRayCasting.frustumCulled = false;
    }
  }

  raycast(
    raycaster: three.Raycaster,
    hoveredChanged: (
      hoveredRectangleId: string,
      rectangle: RoundedRectangleData,
    ) => void,
    updateCursor = true,
  ) {
    if (!this.meshForRayCasting) {
      return;
    }

    const intersects = raycaster.intersectObject(this.meshForRayCasting);
    let curHoveredRectangleId = '';
    if (intersects.length > 0) {
      const lastInstance = intersects[intersects.length - 1];
      const instanceId = lastInstance.instanceId;
      if (instanceId != null) {
        curHoveredRectangleId = this.getRectangleId(instanceId);
      }
    }
    if (this.hoveredRectangelId !== curHoveredRectangleId) {
      this.hoveredRectangelId = curHoveredRectangleId;
      if (updateCursor) {
        document.body.style.cursor =
          this.hoveredRectangelId === '' ? 'default' : 'pointer';
      }
      const rectangleData = this.savedRectangles[this.hoveredRectangelId];
      hoveredChanged(this.hoveredRectangelId, rectangleData);
    }
  }

  updateBorderColor(nodeIds: string[], color: three.Color) {
    if (!this.mesh) {
      return;
    }

    nodeIds = nodeIds.filter((nodeId) => nodeId !== '');

    const borderColorAttrs = this.mesh.geometry.getAttribute('borderColor');
    if (nodeIds.length > 0) {
      for (const nodeId of nodeIds) {
        const savedRectangle = this.savedRectangles[nodeId];
        if (!savedRectangle) {
          continue;
        }
        const index = savedRectangle.index;
        borderColorAttrs.setXYZ(index, color.r, color.g, color.b);
        savedRectangle.borderColor.r = color.r;
        savedRectangle.borderColor.g = color.g;
        savedRectangle.borderColor.b = color.b;
        this.lastBorderColorUpdateRectangles.push(savedRectangle);
      }
    }

    borderColorAttrs.needsUpdate = true;
  }

  restoreBorderColors() {
    if (!this.mesh) {
      return;
    }

    const borderColorAttrs = this.mesh.geometry.getAttribute('borderColor');
    if (this.lastBorderColorUpdateRectangles.length >= 0) {
      for (const rectangle of this.lastBorderColorUpdateRectangles) {
        const index = rectangle.index;
        borderColorAttrs.setXYZ(
          index,
          this.originalBorderColors[index * 3],
          this.originalBorderColors[index * 3 + 1],
          this.originalBorderColors[index * 3 + 2],
        );
        rectangle.borderColor.r = this.originalBorderColors[index * 3];
        rectangle.borderColor.g = this.originalBorderColors[index * 3 + 1];
        rectangle.borderColor.b = this.originalBorderColors[index * 3 + 2];
      }
      borderColorAttrs.needsUpdate = true;
    }
    this.lastBorderColorUpdateRectangles = [];
  }

  updateBgColor(
    nodeIds: string[],
    color: three.Color,
    ignoreNonWhiteBackground = false,
  ) {
    if (!this.mesh) {
      return;
    }

    nodeIds = nodeIds.filter((nodeId) => nodeId !== '');

    const bgColorAttrs = this.mesh.geometry.getAttribute('bgColor');
    if (nodeIds.length > 0) {
      for (const nodeId of nodeIds) {
        const savedRectangle = this.savedRectangles[nodeId];
        if (!savedRectangle) {
          continue;
        }
        const index = savedRectangle.index;
        if (ignoreNonWhiteBackground) {
          const originR = this.originalBgColors[index * 3];
          const originG = this.originalBgColors[index * 3 + 1];
          const originB = this.originalBgColors[index * 3 + 2];
          if (originR !== 1 || originG !== 1 || originB !== 1) {
            continue;
          }
        }
        bgColorAttrs.setXYZ(index, color.r, color.g, color.b);
        savedRectangle.bgColor.r = color.r;
        savedRectangle.bgColor.g = color.g;
        savedRectangle.bgColor.b = color.b;
        this.lastBgColorUpdateRectangles.push(savedRectangle);
      }
    }

    bgColorAttrs.needsUpdate = true;
  }

  restoreBgColors() {
    if (!this.mesh) {
      return;
    }

    const bgColorAttrs = this.mesh.geometry.getAttribute('bgColor');
    if (this.lastBgColorUpdateRectangles.length >= 0) {
      for (const rectangle of this.lastBgColorUpdateRectangles) {
        const index = rectangle.index;
        bgColorAttrs.setXYZ(
          index,
          this.originalBgColors[index * 3],
          this.originalBgColors[index * 3 + 1],
          this.originalBgColors[index * 3 + 2],
        );
        rectangle.bgColor.r = this.originalBgColors[index * 3];
        rectangle.bgColor.g = this.originalBgColors[index * 3 + 1];
        rectangle.bgColor.b = this.originalBgColors[index * 3 + 2];
      }
      bgColorAttrs.needsUpdate = true;
    }
    this.lastBgColorUpdateRectangles = [];
  }

  updateBorderWidth(nodeIds: string[], width: number) {
    if (!this.mesh) {
      return;
    }

    nodeIds = nodeIds.filter((nodeId) => nodeId !== '');

    const borderWidthAttrs = this.mesh.geometry.getAttribute('borderWidth');
    if (nodeIds.length > 0) {
      for (const nodeId of nodeIds) {
        const savedRectangle = this.savedRectangles[nodeId];
        if (!savedRectangle) {
          continue;
        }
        const index = savedRectangle.index;
        borderWidthAttrs.setX(index, width);
        savedRectangle.borderWidth = width;
        this.lastBorderWidthUpdateRectangles.push(savedRectangle);
      }
    }

    borderWidthAttrs.needsUpdate = true;
  }

  restoreBorderWidths() {
    if (!this.mesh) {
      return;
    }

    const borderWidthAttrs = this.mesh.geometry.getAttribute('borderWidth');
    if (this.lastBorderWidthUpdateRectangles.length >= 0) {
      for (const rectangle of this.lastBorderWidthUpdateRectangles) {
        const index = rectangle.index;
        borderWidthAttrs.setX(index, this.originalBorderWidths[index]);
        rectangle.borderWidth = this.originalBorderWidths[index];
      }
      borderWidthAttrs.needsUpdate = true;
    }
    this.lastBorderWidthUpdateRectangles = [];
  }

  updateOpacity(nodeIds: string[], opacity: number) {
    if (!this.mesh) {
      return;
    }

    nodeIds = nodeIds.filter((nodeId) => nodeId !== '');

    const opacityAttrs = this.mesh.geometry.getAttribute('opacity');
    if (nodeIds.length > 0) {
      for (const nodeId of nodeIds) {
        const savedRectangle = this.savedRectangles[nodeId];
        if (!savedRectangle) {
          continue;
        }
        const index = savedRectangle.index;
        opacityAttrs.setX(index, opacity);
        savedRectangle.opacity = opacity;
        this.lastOpacityUpdateRectangles.push(savedRectangle);
      }
    }

    opacityAttrs.needsUpdate = true;
  }

  restoreOpacities() {
    if (!this.mesh) {
      return;
    }

    const opacityAttrs = this.mesh.geometry.getAttribute('opacity');
    if (this.lastOpacityUpdateRectangles.length >= 0) {
      for (const rectangle of this.lastOpacityUpdateRectangles) {
        const index = rectangle.index;
        opacityAttrs.setX(index, this.originalOpacities[index]);
        rectangle.opacity = this.originalOpacities[index];
      }
      opacityAttrs.needsUpdate = true;
    }
    this.lastOpacityUpdateRectangles = [];
  }

  updateAngle(nodeId: string, angleInDegree: number) {
    if (!this.mesh) {
      return;
    }

    const angleAttr = this.mesh.geometry.getAttribute('angle');
    const rectangle = this.savedRectangles[nodeId];
    if (!rectangle) {
      return;
    }
    const angleInRadius = (angleInDegree / 180) * Math.PI;
    angleAttr.setX(rectangle.index, angleInRadius);
    angleAttr.needsUpdate = true;
  }

  updateAnimationProgress(progress: number) {
    if (!this.mesh) {
      return;
    }

    this.curAnimationProgrssUniform.value = progress;
  }

  getRectangleId(index: number): string {
    return this.curRectangles[index].id;
  }

  getNodeIndex(nodeId: string): number {
    return this.savedRectangles[nodeId]?.index ?? -1;
  }

  setBgColorWhenFar(color: WebglColor, alpha: number) {
    if (!this.mesh) {
      return;
    }

    this.material.uniforms['bgColorWhenFar'].value = [
      color.r,
      color.g,
      color.b,
      alpha,
    ];
  }

  clearSavedDataForAnimation() {
    this.savedRectangles = {};
  }

  private setInstancePositionAndScale(
    mesh: three.InstancedMesh,
    index: number,
    x: number,
    y: number,
    z: number,
    scaleWidth: number,
    scaleHeight: number,
  ) {
    this.dummy.position.set(x, y, z);
    this.dummy.scale.set(scaleWidth, 1, scaleHeight);
    this.dummy.updateMatrix();
    mesh.setMatrixAt(index, this.dummy.matrix);
    mesh.instanceMatrix.needsUpdate = true;
  }
}
