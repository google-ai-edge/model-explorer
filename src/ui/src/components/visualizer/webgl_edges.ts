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

import {WEBGL_ELEMENT_Y_FACTOR} from './common/consts';
import {ModelEdge, ModelGraph} from './common/model_graph';
import {UniformValue, WebglColor} from './common/types';

const THREE = three;

const EDGE_SEGMENT_VERTEX_SHADER = `
precision highp float;

#define PI 3.1415926535897932384626433832795

uniform float edgeWidth;
uniform float animationProgress;

attribute vec4 endPoints;
attribute vec4 targetEndPoints;
attribute vec3 color;
attribute float yOffset;

varying vec3 vColor;

float atan2(in float y, in float x) {
  bool s = (abs(x) > abs(y));
  return mix(PI/2.0 - atan(x,y), atan(y,x), s);
}

void main() {
  vColor = color;

  vec3 pos = position;

  float curStartX = endPoints.x;
  float curStartY = endPoints.y;
  float curEndX = endPoints.z;
  float curEndY = endPoints.w;
  float targetStartX = targetEndPoints.x;
  float targetStartY = targetEndPoints.y;
  float targetEndX = targetEndPoints.z;
  float targetEndY = targetEndPoints.w;

  float progress = animationProgress * step(0.0, animationProgress); 
  float startX = curStartX + (targetStartX - curStartX) * progress;
  float startY = curStartY + (targetStartY - curStartY) * progress;
  float endX = curEndX + (targetEndX - curEndX) * progress;
  float endY = curEndY + (targetEndY - curEndY) * progress;

  float length = distance(vec2(startX, startY), vec2(endX, endY));
  pos.x = (step(0.0, pos.x) * 2.0 - 1.0) * (edgeWidth / 2.0);
  pos.z = (step(0.0, pos.z) * 2.0 - 1.0) * (length / 2.0);

  float angle = PI / 2.0 - atan2(endY - startY, endX - startX);
  float c = cos(angle);
  float s = sin(angle);

  float posX = pos.x;
  float posZ = pos.z;
  pos.x = posX * c + posZ * s;
  pos.z = posZ * c - posX * s;

  float centerX = (startX + endX) / 2.0;
  float centerZ = (startY + endY) / 2.0;

  gl_Position = projectionMatrix * modelViewMatrix *
      vec4(pos.x + centerX, yOffset, pos.z + centerZ, 1.0);
}
`;

const EDGE_SEGMENT_FRAGMENT_SHADER = `
precision highp float;

varying vec3 vColor;

void main() {
  gl_FragColor = vec4(vColor, 1.0);
}
`;

const ARROW_HEAD_VERTEX_SHADER = `
precision highp float;

#define PI 3.1415926535897932384626433832795

uniform float animationProgress;

// End points of the last segment of the edge.
attribute vec4 endPoints;
attribute vec4 targetEndPoints;
attribute vec3 color;
attribute float yOffset;

varying vec3 vColor;

float atan2(in float y, in float x) {
  bool s = (abs(x) > abs(y));
  return mix(PI/2.0 - atan(x,y), atan(y,x), s);
}

void main() {
  vColor = color;

  vec3 pos = position;

  float curStartX = endPoints.x;
  float curStartY = endPoints.y;
  float curEndX = endPoints.z;
  float curEndY = endPoints.w;
  float targetStartX = targetEndPoints.x;
  float targetStartY = targetEndPoints.y;
  float targetEndX = targetEndPoints.z;
  float targetEndY = targetEndPoints.w;

  float progress = animationProgress * step(0.0, animationProgress); 
  float startX = curStartX + (targetStartX - curStartX) * progress;
  float startY = curStartY + (targetStartY - curStartY) * progress;
  float endX = curEndX + (targetEndX - curEndX) * progress;
  float endY = curEndY + (targetEndY - curEndY) * progress;

  float angle = PI - atan2(endY - startY, endX - startX) + PI / 2.0;
  float c = cos(angle);
  float s = sin(angle);

  float posX = pos.x;
  float posZ = pos.z;
  pos.x = posX * c + posZ * s;
  pos.z = posZ * c - posX * s;

  gl_Position = projectionMatrix * modelViewMatrix *
      vec4(pos.x + endX, yOffset, pos.z + endY, 1.0);
}
`;

const ARROW_HEAD_FRAGMENT_SHADER = `
precision highp float;

varying vec3 vColor;

void main() {
  gl_FragColor = vec4(vColor, 1.0);
}
`;

const ARROW_BASE_SIZE = 6;
const ARROW_HEIGHT = 6;
const ARROW_THICKNESS = 4;

interface EdgeSegment {
  index: number;
  endPoints: number[];
}

interface ArrowHead {
  index: number;
  lastSegmentEndPoints: number[];
}

/**
 * A class for generating and managing edges for webgl renderer.
 */
export class WebglEdges {
  edgesMesh?: three.Mesh;
  material: three.ShaderMaterial;
  arrowHeadMat: three.ShaderMaterial;
  arrowHeadsMesh?: three.Mesh;

  private readonly planeGeo: three.PlaneGeometry;
  private readonly arrowHeadGeometry: three.ShapeGeometry;
  private savedEdgeSegments: Record<string, EdgeSegment> = {};
  // index: edge id.
  private savedEdges: Record<string, EdgeSegment[]> = {};
  // index: edge id.
  private savedArrowHeads: Record<string, ArrowHead> = {};
  private readonly curAnimationProgrssUniform: UniformValue = {value: -1};
  private originalColors: number[] = [];
  private originalYOffsets: number[] = [];
  private originalArrowHeadYOffsets: number[] = [];
  private lastColorUpdateEdgeSegments: EdgeSegment[] = [];
  private lastColorUpdateArrowHeads: ArrowHead[] = [];
  private lastYOffsetsUpdateEdgeSegments: EdgeSegment[] = [];
  private lastYOffsetsUpdateArrowHeads: ArrowHead[] = [];

  constructor(
    private readonly color: WebglColor,
    private readonly edgeWidth: number,
    private readonly arrowScale = 1,
  ) {
    this.planeGeo = new THREE.PlaneGeometry(1, 1);
    this.planeGeo.rotateX(-Math.PI / 2);

    this.material = new THREE.ShaderMaterial({
      uniforms: {
        'edgeWidth': {value: this.edgeWidth},
        'animationProgress': this.curAnimationProgrssUniform,
      },
      vertexShader: EDGE_SEGMENT_VERTEX_SHADER,
      fragmentShader: EDGE_SEGMENT_FRAGMENT_SHADER,
      transparent: true,
    });

    // Create arrow head geo.
    const triangle = new THREE.Shape();
    const arrowBaseSize = ARROW_BASE_SIZE * arrowScale;
    const arrowHeight = ARROW_HEIGHT * arrowScale;
    const arrowThickness = ARROW_THICKNESS * arrowScale;
    triangle
      .moveTo(-arrowBaseSize / 2, -arrowHeight)
      .lineTo(0, -arrowThickness)
      .lineTo(arrowBaseSize / 2, -arrowHeight)
      .lineTo(0, 0)
      .lineTo(-arrowBaseSize / 2, -arrowHeight);
    this.arrowHeadGeometry = new THREE.ShapeGeometry(triangle);
    this.arrowHeadGeometry.rotateX(-Math.PI / 2);

    // Material for arrow head.
    this.arrowHeadMat = new THREE.ShaderMaterial({
      uniforms: {
        'animationProgress': this.curAnimationProgrssUniform,
      },
      vertexShader: ARROW_HEAD_VERTEX_SHADER,
      fragmentShader: ARROW_HEAD_FRAGMENT_SHADER,
      transparent: true,
    });
  }

  generateMesh(
    edges: Array<{edge: ModelEdge; index: number}>,
    modelGraph: ModelGraph,
    forceNoAnimation = false,
  ) {
    if (edges.length === 0) {
      this.edgesMesh = undefined;
      this.arrowHeadsMesh = undefined;
      return;
    }

    // Edge segments.
    const endPoints: number[] = [];
    const targetEndPoints: number[] = [];
    const lastSegmentEndPoints: number[] = [];
    const targetLastSegmentEndPoints: number[] = [];
    const colors: number[] = [];
    const yOffsets: number[] = [];
    const arrowHeadYOffsets: number[] = [];
    const arrowHeadColors: number[] = [];

    const newSavedEdgeSegments: Record<string, EdgeSegment> = {};
    const newSavedArrowHeads: Record<string, ArrowHead> = {};
    this.savedEdges = {};

    let segmentIndex = 0;
    let edgeIndex = 0;
    for (const {edge, index} of edges) {
      const points = edge.curvePoints || [];
      const fromNode = modelGraph.nodesById[edge.fromNodeId];
      const toNode = modelGraph.nodesById[edge.toNodeId];
      const nodeGlobalX = fromNode.globalX || 0;
      const nodeGlobalY = fromNode.globalY || 0;
      for (let i = 0; i < points.length - 1; i++) {
        const startPt = points[i];
        const endPt = points[i + 1];
        const segmentId = `${fromNode.id}__${toNode.id}___${i}`;
        const curEndpoints = [
          startPt.x + nodeGlobalX,
          startPt.y + nodeGlobalY,
          endPt.x + nodeGlobalX,
          endPt.y + nodeGlobalY,
        ];
        const savedCurEndpoints = [...curEndpoints];

        // Move the last segment inward a little bit so that it doesn't go out
        // of the arrowhead.
        if (i === points.length - 2 && points.length >= 2) {
          const f = Math.atan2(endPt.y - startPt.y, endPt.x - startPt.x);
          curEndpoints[2] -= (Math.cos(f) * ARROW_HEIGHT * this.arrowScale) / 2;
          curEndpoints[3] -= (Math.sin(f) * ARROW_HEIGHT * this.arrowScale) / 2;
        }

        const savedSegment = this.savedEdgeSegments[segmentId];
        if (forceNoAnimation) {
          endPoints.push(...curEndpoints);
        } else {
          if (!savedSegment) {
            endPoints.push(...curEndpoints);
          } else {
            endPoints.push(...savedSegment.endPoints);
          }
        }
        targetEndPoints.push(...curEndpoints);

        yOffsets.push(index * WEBGL_ELEMENT_Y_FACTOR);
        colors.push(this.color.r, this.color.g, this.color.b);
        newSavedEdgeSegments[segmentId] = {
          endPoints: curEndpoints,
          index: segmentIndex,
        };

        if (this.savedEdges[edge.id] == null) {
          this.savedEdges[edge.id] = [];
        }
        this.savedEdges[edge.id].push(newSavedEdgeSegments[segmentId]);

        // Arrowheads.
        if (i === points.length - 2) {
          const arrowHeadId = edge.id;
          const curLastSegmentEndpoints = savedCurEndpoints;
          const savedArrowHead = this.savedArrowHeads[arrowHeadId];
          if (forceNoAnimation) {
            lastSegmentEndPoints.push(...curLastSegmentEndpoints);
          } else {
            if (!savedArrowHead) {
              lastSegmentEndPoints.push(...curLastSegmentEndpoints);
            } else {
              lastSegmentEndPoints.push(...savedArrowHead.lastSegmentEndPoints);
            }
          }
          targetLastSegmentEndPoints.push(...curLastSegmentEndpoints);

          arrowHeadYOffsets.push(
            index * WEBGL_ELEMENT_Y_FACTOR + WEBGL_ELEMENT_Y_FACTOR / 2,
          );
          arrowHeadColors.push(this.color.r, this.color.g, this.color.b);
          newSavedArrowHeads[arrowHeadId] = {
            index: edgeIndex,
            lastSegmentEndPoints: curLastSegmentEndpoints,
          };
        }

        segmentIndex++;
      }
      edgeIndex++;
    }

    this.savedEdgeSegments = newSavedEdgeSegments;
    this.savedArrowHeads = newSavedArrowHeads;
    this.originalColors = colors;
    this.originalYOffsets = yOffsets;
    this.originalArrowHeadYOffsets = arrowHeadYOffsets;

    // Geo for edge segments.
    const geometry = new THREE.InstancedBufferGeometry().copy(this.planeGeo);
    geometry.instanceCount = yOffsets.length;
    geometry.setAttribute(
      'endPoints',
      new THREE.InstancedBufferAttribute(new Float32Array(endPoints), 4),
    );
    geometry.setAttribute(
      'targetEndPoints',
      new THREE.InstancedBufferAttribute(new Float32Array(targetEndPoints), 4),
    );
    geometry.setAttribute(
      'color',
      new THREE.InstancedBufferAttribute(new Float32Array(colors), 3),
    );
    geometry.setAttribute(
      'yOffset',
      new THREE.InstancedBufferAttribute(new Float32Array(yOffsets), 1),
    );
    this.edgesMesh = new THREE.Mesh(geometry, this.material);
    this.edgesMesh.frustumCulled = false;

    // Geo for arrowheads.
    const arrowHeadGeometry = new THREE.InstancedBufferGeometry().copy(
      this.arrowHeadGeometry,
    );
    arrowHeadGeometry.instanceCount = arrowHeadYOffsets.length;
    arrowHeadGeometry.setAttribute(
      'endPoints',
      new THREE.InstancedBufferAttribute(
        new Float32Array(lastSegmentEndPoints),
        4,
      ),
    );
    arrowHeadGeometry.setAttribute(
      'targetEndPoints',
      new THREE.InstancedBufferAttribute(
        new Float32Array(targetLastSegmentEndPoints),
        4,
      ),
    );
    arrowHeadGeometry.setAttribute(
      'color',
      new THREE.InstancedBufferAttribute(new Float32Array(arrowHeadColors), 3),
    );
    arrowHeadGeometry.setAttribute(
      'yOffset',
      new THREE.InstancedBufferAttribute(
        new Float32Array(arrowHeadYOffsets),
        1,
      ),
    );
    this.arrowHeadsMesh = new THREE.Mesh(arrowHeadGeometry, this.arrowHeadMat);
    this.arrowHeadsMesh.frustumCulled = false;
  }

  updateColors(edgeIds: string[], color: WebglColor) {
    if (this.edgesMesh) {
      const colorAttr = this.edgesMesh.geometry.getAttribute('color');
      if (edgeIds.length > 0) {
        for (const edgeId of edgeIds) {
          for (const segment of this.savedEdges[edgeId] || []) {
            const index = segment.index;
            colorAttr.setXYZ(index, color.r, color.g, color.b);
            this.lastColorUpdateEdgeSegments.push(segment);
          }
        }
      }
      colorAttr.needsUpdate = true;
    }

    if (this.arrowHeadsMesh) {
      const colorAttr = this.arrowHeadsMesh.geometry.getAttribute('color');
      if (edgeIds.length > 0) {
        for (const edgeId of edgeIds) {
          const arrowHead = this.savedArrowHeads[edgeId];
          const index = arrowHead.index;
          colorAttr.setXYZ(index, color.r, color.g, color.b);
          this.lastColorUpdateArrowHeads.push(arrowHead);
        }
      }
      colorAttr.needsUpdate = true;
    }
  }

  updateYOffsets(edgeIds: string[], yOffset: number) {
    if (this.edgesMesh) {
      const yOffsetsAttr = this.edgesMesh.geometry.getAttribute('yOffset');
      if (edgeIds.length > 0) {
        for (const edgeId of edgeIds) {
          for (const segment of this.savedEdges[edgeId] || []) {
            const index = segment.index;
            yOffsetsAttr.setX(index, yOffset);
            this.lastYOffsetsUpdateEdgeSegments.push(segment);
          }
        }
      }
      yOffsetsAttr.needsUpdate = true;
    }

    if (this.arrowHeadsMesh) {
      const yOffsetsAttr = this.arrowHeadsMesh.geometry.getAttribute('yOffset');
      if (edgeIds.length > 0) {
        for (const edgeId of edgeIds) {
          const arrowHead = this.savedArrowHeads[edgeId];
          if (!arrowHead) {
            continue;
          }
          const index = arrowHead.index;
          yOffsetsAttr.setX(index, yOffset);
          this.lastYOffsetsUpdateArrowHeads.push(arrowHead);
        }
      }
      yOffsetsAttr.needsUpdate = true;
    }
  }

  restoreColors() {
    if (this.edgesMesh) {
      const colorAttr = this.edgesMesh.geometry.getAttribute('color');
      if (this.lastColorUpdateEdgeSegments.length > 0) {
        for (const segment of this.lastColorUpdateEdgeSegments) {
          const index = segment.index;
          colorAttr.setXYZ(
            index,
            this.originalColors[index * 3],
            this.originalColors[index * 3 + 1],
            this.originalColors[index * 3 + 2],
          );
        }
        colorAttr.needsUpdate = true;
      }
      this.lastColorUpdateEdgeSegments = [];
    }

    if (this.arrowHeadsMesh) {
      const colorAttr = this.arrowHeadsMesh.geometry.getAttribute('color');
      if (this.lastColorUpdateArrowHeads.length > 0) {
        for (const arrowHead of this.lastColorUpdateArrowHeads) {
          const index = arrowHead.index;
          colorAttr.setXYZ(
            index,
            this.originalColors[index * 3],
            this.originalColors[index * 3 + 1],
            this.originalColors[index * 3 + 2],
          );
        }
        colorAttr.needsUpdate = true;
      }
      this.lastColorUpdateArrowHeads = [];
    }
  }

  restoreYOffsets() {
    if (this.edgesMesh) {
      const yOffsetsAttr = this.edgesMesh.geometry.getAttribute('yOffset');
      if (this.lastYOffsetsUpdateEdgeSegments.length > 0) {
        for (const segment of this.lastYOffsetsUpdateEdgeSegments) {
          const index = segment.index;
          yOffsetsAttr.setX(index, this.originalYOffsets[index]);
        }
        yOffsetsAttr.needsUpdate = true;
      }
      this.lastYOffsetsUpdateEdgeSegments = [];
    }

    if (this.arrowHeadsMesh) {
      const yOffsetsAttr = this.arrowHeadsMesh.geometry.getAttribute('yOffset');
      if (this.lastYOffsetsUpdateArrowHeads.length > 0) {
        for (const arrowHead of this.lastYOffsetsUpdateArrowHeads) {
          const index = arrowHead.index;
          yOffsetsAttr.setX(index, this.originalArrowHeadYOffsets[index]);
        }
        yOffsetsAttr.needsUpdate = true;
      }
      this.lastYOffsetsUpdateArrowHeads = [];
    }
  }

  clear() {
    if (this.edgesMesh) {
      this.edgesMesh.removeFromParent();
    }
    if (this.arrowHeadsMesh) {
      this.arrowHeadsMesh.removeFromParent();
    }
  }

  clearSavedDataForAnimation() {
    this.savedEdgeSegments = {};
    this.savedEdges = {};
    this.savedArrowHeads = {};
  }

  updateAnimationProgress(progress: number) {
    if (!this.edgesMesh) {
      return;
    }

    this.curAnimationProgrssUniform.value = progress;
  }
}
