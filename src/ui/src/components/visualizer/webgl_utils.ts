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

const THREE = three;

/** Creates a rounded rectangle geometry. */
export function createRoundedRectGeometry(
  nodeWidth: number,
  nodeHeight: number,
  radius: number,
): three.ShapeGeometry {
  const shape = createRoundedRectangleShape(
    0,
    0,
    nodeWidth,
    nodeHeight,
    radius,
  );
  const geometry = new THREE.ShapeGeometry(
    shape,
    // This number minus one controls the number of vertices on each rounded
    // corner curves.
    16,
  );
  geometry.rotateX(-Math.PI / 2);
  // Anchor at the center.
  geometry.translate(-nodeWidth / 2, 0, nodeHeight / 2);
  return geometry;
}

/** Creates a rounded rectangle shape. */
export function createRoundedRectangleShape(
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
): three.Shape {
  const roundedRectShape = new THREE.Shape();
  // Start from bottom left corner on the left edge.
  roundedRectShape.moveTo(x, y + radius);
  roundedRectShape.lineTo(x, y + height - radius);
  roundedRectShape.quadraticCurveTo(x, y + height, x + radius, y + height);
  roundedRectShape.lineTo(x + width - radius, y + height);
  roundedRectShape.quadraticCurveTo(
    x + width,
    y + height,
    x + width,
    y + height - radius,
  );
  roundedRectShape.lineTo(x + width, y + radius);
  roundedRectShape.quadraticCurveTo(x + width, y, x + width - radius, y);
  roundedRectShape.lineTo(x + radius, y);
  roundedRectShape.quadraticCurveTo(x, y, x, y + radius);
  return roundedRectShape;
}
