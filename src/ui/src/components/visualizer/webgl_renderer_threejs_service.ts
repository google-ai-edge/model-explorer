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
import * as d3 from 'd3';
import * as three from 'three';

import {ModelGraph} from './common/model_graph';
import {Point, Rect} from './common/types';
import {getHighQualityPixelRatio, IS_MAC} from './common/utils';
import {WebglRenderer} from './webgl_renderer';

const DEFAULT_FRUSTUM_SIZE = 500;
const DEFAULT_CAMERA_Y = 200;
const D3_CONSTRAINT_PADDING = 20;

const THREE = three;

/**
 * A service to handle threejs specific stuff.
 */
@Injectable()
export class WebglRendererThreejsService {
  curScale = 1;
  fps = '';
  camera!: three.OrthographicCamera;
  raycaster!: three.Raycaster;

  readonly zoom = d3.zoom();
  private webglRenderer!: WebglRenderer;
  private curTranslateX = 0;
  private curTranslateY = 0;
  private scene!: three.Scene;
  private renderer!: three.WebGLRenderer;
  private snapshotRenderer!: three.WebGLRenderer;
  private pngDownloaderRenderer!: three.WebGLRenderer;
  private savedCenterX: number | null = null;
  private savedCenterY: number | null = null;
  private resizeTimeoutRef = -1;
  private fpsStartTime = -1;
  private frames = 0;

  init(webglRenderer: WebglRenderer) {
    this.webglRenderer = webglRenderer;
  }

  setupZoomAndPan(
    rootEle: HTMLElement | SVGElement,
    minZoom = 0.1,
    maxZoom = 10,
  ) {
    const view = d3.select(rootEle as Element);

    let savedTranslateX = 0;
    let savedTranslateY = 0;
    this.zoom
      .scaleExtent([minZoom, maxZoom])
      // Constrain the translation to the graph area.
      .constrain((transform) => {
        const container = this.webglRenderer.container.nativeElement;
        const width = container.clientWidth;
        const height = container.clientHeight;
        const aspect = width / height;
        const scaledFrustumSize = DEFAULT_FRUSTUM_SIZE / transform.k;

        // Pre-calculate common terms for X transformations
        const commonXDivisor = 2 * scaledFrustumSize * aspect;
        const offsetScreenX = D3_CONSTRAINT_PADDING / transform.k;
        const offsetScreenXWidth =
          (width - D3_CONSTRAINT_PADDING) / transform.k;

        // Calculate minTransformX
        const termMinX =
          this.webglRenderer.currentMaxX +
          this.convertXFromScreenToScene(offsetScreenX) +
          scaledFrustumSize * aspect;
        const minTransformX = (-termMinX * width) / commonXDivisor + width / 2;

        // Calculate maxTransformX
        const termMaxX =
          this.webglRenderer.currentMinX +
          this.convertXFromScreenToScene(offsetScreenXWidth) +
          scaledFrustumSize * aspect;
        const maxTransformX = (-termMaxX * width) / commonXDivisor + width / 2;

        // Pre-calculate common terms for Y transformations
        const commonYDivisor = 2 * scaledFrustumSize;
        const offsetScreenY = D3_CONSTRAINT_PADDING / transform.k;
        const offsetScreenYHeight =
          (height - D3_CONSTRAINT_PADDING) / transform.k;

        // Calculate minTransformY
        const termMinY =
          -this.webglRenderer.currentMaxZ +
          this.convertZFromScreenToScene(offsetScreenY) -
          scaledFrustumSize;
        const minTransformY = (termMinY * height) / commonYDivisor + height / 2;

        // Calculate maxTransformY
        const termMaxY =
          -this.webglRenderer.currentMinZ +
          this.convertZFromScreenToScene(offsetScreenYHeight) -
          scaledFrustumSize;
        const maxTransformY = (termMaxY * height) / commonYDivisor + height / 2;

        return d3.zoomIdentity
          .translate(
            Math.min(maxTransformX, Math.max(minTransformX, transform.x)),
            Math.min(maxTransformY, Math.max(minTransformY, transform.y)),
          )
          .scale(transform.k);
      })
      .wheelDelta(
        // This controls the speed of pinch-to-zoom (and zoom by
        // ctrl+scroll).
        () => {
          return (-d3.event.deltaY * (d3.event.deltaMode ? 120 : 1)) / 150;
        },
      )
      .filter(() => {
        if (d3.event.type === 'mousedown') {
          savedTranslateX = this.curTranslateX;
          savedTranslateY = this.curTranslateY;
        }

        // Ignore right click.
        if (
          d3.event.button === 2 ||
          (IS_MAC &&
            d3.event.ctrlKey &&
            d3.event.button === 0 &&
            d3.event.type === 'mousedown')
        ) {
          return false;
        }

        if (d3.event.type === 'dblclick') {
          d3.event.stopPropagation();
          this.webglRenderer.handleDoubleClickOnGraph(
            d3.event.altKey,
            d3.event.shiftKey,
          );
          return false;
        }

        // By default, d3.zoom uses scrolling to trigger zooming. To make
        // the interactions more intuitive (and to be more consistent with
        // similar software such as Netron, Figma, etc), we disable the
        // default behavior (by returning false), and make the scrolling to
        // scroll (translate) the model graph.
        //
        // Note that in d3.zoom, the way to check if zoom is being triggered
        // by scrolling is to check that its event type is 'wheel' and
        // ctrlKey is false.
        if (d3.event.type === 'wheel' && !d3.event.ctrlKey) {
          // Scale scrolling amount by the zoom level to make the experience
          // consistent at different zoom levels.
          const factor = 0.5 / this.curScale;
          this.zoom.translateBy(
            view,
            -Number(d3.event.deltaX) * factor,
            -Number(d3.event.deltaY) * factor,
          );
          d3.event.preventDefault();
          return false;
        }

        return true;
      })
      .on('zoom', () => {
        this.handleZoom();
      })
      .on('end', () => {
        this.handleZoomEnd(savedTranslateX, savedTranslateY);
      });

    // Use the regular interpolation instead of the default d3.zoomInterpolate
    // which has some unexpected behavior.
    this.zoom.interpolate(d3.interpolate);

    view.call(this.zoom);
  }

  setupThreeJs() {
    const canvas = this.webglRenderer.canvas.nativeElement;

    // Set up THREE.js scene.
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xffffff);

    // Create a camera with orthographic projection.
    //
    // In this projection mode, an object's size in the rendered image stays
    // constant regardless of its distance from the camera. It is suitable for
    // rendering 2D scenes (such as the model graph).
    //
    // Frustum size determines the region of the scene that will appear on the
    // screen (field of view). Larger/Smaller frustum size means more/less
    // stuff to show. To prevent distortion in the final render, the aspect
    // ratio of the frustum size needs to match the content's aspect ratio.
    //
    // In `setupPanAndZoom` below, the frustum size will be used to simulate
    // zooming.
    const aspect = canvas.clientWidth / canvas.clientHeight;
    this.camera = new THREE.OrthographicCamera(
      0, // left
      2 * DEFAULT_FRUSTUM_SIZE * aspect, // right
      0, // top. Notice this value needs to be negative.
      -2 * DEFAULT_FRUSTUM_SIZE, // bottom. Notice this value needs to be
      // negative.
      0.001, // near plane,
      1000, // far plane
    );
    // The camera is looking down on the x-z plane. The distance between the
    // camera and the x-z plane doesn't matter (due to OrthographicCamera).
    this.camera.position.y = DEFAULT_CAMERA_Y;
    this.camera.lookAt(new THREE.Vector3(0, 0, 0));
    // this.camera.rotateOnAxis(new THREE.Vector3(0, 0, 1), Math.PI / 2);
    this.camera.updateMatrixWorld();
    this.camera.updateProjectionMatrix();

    // Set up renderer (using WebGL 2 behind the scene).
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      // This will enable performance mode (i.e. use high-performance graphic
      // card) on supported platforms.
      powerPreference: 'high-performance',
      precision: 'highp',
      // This will make things (especially thin lines) look better when
      // zoomed out.
      antialias: true,
      alpha: true,
    });
    const pixelRatio = getHighQualityPixelRatio();
    this.renderer.setPixelRatio(pixelRatio);
    this.renderer.setSize(canvas.clientWidth, canvas.clientHeight);

    // Set up renderer for snapshot.
    const snapshotCanvas = this.webglRenderer.snapshotCanvas.nativeElement;
    this.snapshotRenderer = new THREE.WebGLRenderer({
      canvas: snapshotCanvas,
      powerPreference: 'high-performance',
      precision: 'highp',
      antialias: true,
      alpha: true,
      // This allows us to render this canvas to other canvases.
      preserveDrawingBuffer: true,
    });
    this.snapshotRenderer.setPixelRatio(pixelRatio);

    // Render.
    //
    // Note that we don't have an explicit animation loop. We render things
    // on demand (by calling this.render).
    this.render();

    // Resize renderer to match the canvas size when resized.
    const observer = new ResizeObserver(() => {
      // Wrap in RAF for smoothness.
      requestAnimationFrame(() => {
        this.resizeRendererToDisplaySize();
      });
    });
    observer.observe(this.webglRenderer.container.nativeElement);

    this.raycaster = new THREE.Raycaster();
    this.raycaster.params.Points!.threshold = 5.5;
  }

  clearScene(objsToSkip: Array<three.Object3D | undefined> = []) {
    for (let i = this.scene.children.length - 1; i >= 0; i--) {
      const obj = this.scene.children[i] as three.Mesh;
      if (objsToSkip.includes(obj)) {
        continue;
      }
      if (obj.geometry) {
        obj.geometry.dispose();
      }
      this.scene.remove(obj);
    }
  }

  setupPngDownloaderRenderer(
    canvas: HTMLCanvasElement,
    transparentBackground: boolean,
    width: number,
    height: number,
  ) {
    if (!this.pngDownloaderRenderer) {
      this.pngDownloaderRenderer = new THREE.WebGLRenderer({
        canvas,
        powerPreference: 'high-performance',
        precision: 'highp',
        antialias: true,
        alpha: true,
        preserveDrawingBuffer: true,
      });
      this.pngDownloaderRenderer.setPixelRatio(getHighQualityPixelRatio());
    }
    if (transparentBackground) {
      this.scene.background = null;
      this.pngDownloaderRenderer.setClearColor(0x000000, 0);
    }
    this.pngDownloaderRenderer.setSize(width, height, false);
  }

  renderPngDownloader(camera: three.Camera) {
    this.pngDownloaderRenderer.render(this.scene, camera);
  }

  renderSnapshot(width: number, height: number) {
    this.snapshotRenderer.setSize(width, height, false);
    this.snapshotRenderer.render(this.scene, this.camera);
  }

  setSceneBackground(color: three.Color) {
    this.scene.background = color;
  }

  createOrthographicCamera(
    left: number,
    right: number,
    top: number,
    bottom: number,
  ): three.OrthographicCamera {
    // Create a camera used for rendering full graph for downloading.
    const camera = new THREE.OrthographicCamera(
      left,
      right,
      top,
      bottom,
      0.001, // near plane,
      1000, // far plane
    );
    // The camera is looking down on the x-z plane. The distance between the
    // camera and the x-z plane doesn't matter (due to OrthographicCamera).
    camera.position.y = DEFAULT_CAMERA_Y;
    camera.lookAt(new THREE.Vector3(0, 0, 0));
    camera.updateMatrixWorld();
    camera.updateProjectionMatrix();

    return camera;
  }

  dispose() {
    if (this.renderer) {
      this.renderer.dispose();
      this.renderer.forceContextLoss();
    }
    if (this.snapshotRenderer) {
      this.snapshotRenderer.dispose();
      this.snapshotRenderer.forceContextLoss();
    }
    if (this.pngDownloaderRenderer) {
      this.pngDownloaderRenderer.dispose();
      this.pngDownloaderRenderer.forceContextLoss();
    }
  }

  render(countFps = false) {
    if (!this.renderer || !this.scene || !this.camera) {
      return;
    }

    this.renderer.render(this.scene, this.camera);

    if (this.webglRenderer.benchmark && countFps) {
      if (this.fpsStartTime < 0) {
        this.fpsStartTime = performance.now();
      }
      this.frames += 1;
      const delta = performance.now() - this.fpsStartTime;
      if (delta > 1000) {
        this.fps = ((this.frames / delta) * 1000).toFixed(1);
        this.fpsStartTime = -1;
        this.frames = 0;
        this.webglRenderer.changeDetectorRef.markForCheck();
      }
    }

    // Uncomment the following to show the number of draw calls
    // console.log('draw call count', this.renderer.info.render.calls);
  }

  zoomFitGraph(paddingPercent = 0.9, transitionDuration = 200) {
    this.zoomFit(
      {
        x: this.webglRenderer.currentMinX,
        y: this.webglRenderer.currentMinZ,
        width: this.webglRenderer.currentMaxX - this.webglRenderer.currentMinX,
        height: this.webglRenderer.currentMaxZ - this.webglRenderer.currentMinZ,
      },
      paddingPercent,
      transitionDuration,
    );
  }

  zoomFit(
    rect: Rect,
    paddingPercent = 0.9,
    transitionDuration = 300,
    useCurScale = false,
    capScale = true,
    capMinScale = false,
    extraScaleFactor = 1,
  ) {
    if (!this.webglRenderer.container) {
      return;
    }
    const container = this.webglRenderer.container.nativeElement;

    const containerWidth = container.clientWidth * paddingPercent;
    const containerHeight = container.clientHeight * paddingPercent;
    const rectAspect = rect.width / rect.height;
    const containerAspect = containerWidth / containerHeight;
    let scale = useCurScale
      ? this.curScale
      : Math.abs(
          rectAspect > containerAspect
            ? this.convertXFromScreenToScene(containerWidth) / rect.width
            : this.convertZFromScreenToScene(containerHeight) / rect.height,
        );
    const targetMidX = rect.x + rect.width / 2;
    let targetMidZ = rect.y + rect.height / 2;

    if (!useCurScale && capScale) {
      // Max scale turns a 30 height to a maximum 45 height.
      const maxScale = this.convertZFromScreenToScene(45) / 30;
      scale = Math.min(maxScale, scale);

      if (capMinScale) {
        // Min scale turns a 30 height to a minimum 20 height.
        const minScale = this.convertZFromScreenToScene(20) / 30;
        // When the target scale is too small (<1), we move the canvas so that
        // the top edge of the target aligns with the top edge of the screen
        // instead of zooming all the way out.
        if (scale < minScale) {
          targetMidZ =
            rect.y +
            this.convertZFromScreenToScene(containerHeight / 2 - 60, minScale);
          scale = Math.max(minScale, scale);
        }
      }
    }

    this.centerViewAt(
      targetMidX,
      targetMidZ,
      scale * extraScaleFactor,
      transitionDuration,
    );
  }

  zoomFitOnNode(
    nodeId: string | undefined,
    modelGraph: ModelGraph,
    transitionDuration: number,
  ) {
    if (!nodeId) {
      setTimeout(() => {
        this.zoomFitGraph(0.9, transitionDuration);
      });
    } else {
      setTimeout(() => {
        const node = modelGraph.nodesById[nodeId];
        this.zoomFit(
          {
            x: this.webglRenderer.getNodeX(node),
            y: this.webglRenderer.getNodeY(node),
            width: this.webglRenderer.getNodeWidth(node),
            height: this.webglRenderer.getNodeHeight(node),
          },
          0.9,
          transitionDuration,
          false,
          true,
          // Cap min scale when zooming on a node.
          true,
          // Extra zoom factor.
          this.webglRenderer.appService.config()?.extraZoomFactorOnNode ?? 1,
        );
      }, 0);
    }
  }

  zoomFitOnNodes(
    nodeIds: string[],
    modelGraph: ModelGraph,
    transitionDuration: number,
  ) {
    if (nodeIds.length === 0) {
      setTimeout(() => {
        this.zoomFitGraph(0.9, transitionDuration);
      });
    } else {
      setTimeout(() => {
        let minX = Infinity;
        let maxX = -Infinity;
        let minY = Infinity;
        let maxY = -Infinity;
        for (const nodeId of nodeIds) {
          const node = modelGraph.nodesById[nodeId];
          if (node) {
            minX = Math.min(minX, this.webglRenderer.getNodeX(node));
            maxX = Math.max(
              maxX,
              this.webglRenderer.getNodeX(node) +
                this.webglRenderer.getNodeWidth(node),
            );
            minY = Math.min(minY, this.webglRenderer.getNodeY(node));
            maxY = Math.max(
              maxY,
              this.webglRenderer.getNodeY(node) +
                this.webglRenderer.getNodeHeight(node),
            );
          }
        }
        this.zoomFit(
          {
            x: minX,
            y: minY,
            width: maxX - minX,
            height: maxY - minY,
          },
          0.9,
          transitionDuration,
          false,
          true,
          // Cap min scale when zooming on a node.
          true,
          // Extra zoom factor.
          this.webglRenderer.appService.config()?.extraZoomFactorOnNode ?? 1,
        );
      }, 0);
    }
  }

  addToScene(object: three.Object3D | undefined) {
    if (object) {
      this.scene.add(object);
    }
  }

  removeFromScene(object: three.Object3D | undefined) {
    if (object) {
      this.scene.remove(object);
    }
  }

  convertXFromSceneToScreen(sceneX: number): number {
    if (!this.webglRenderer.container) {
      return 0;
    }
    const container = this.webglRenderer.container.nativeElement;

    // The following is the reverse of the calculations in
    // `setCameraFrustum` above.
    const containerWidth = container.clientWidth;
    const aspect = containerWidth / container.clientHeight;

    return (
      (sceneX / ((DEFAULT_FRUSTUM_SIZE / 1) * aspect) / -2) * containerWidth
    );
  }

  convertZFromSceneToScreen(sceneZ: number): number {
    if (!this.webglRenderer.container) {
      return 0;
    }
    const container = this.webglRenderer.container.nativeElement;

    const containerHeight = container.clientHeight;
    return (sceneZ * 1 * containerHeight) / DEFAULT_FRUSTUM_SIZE / 2;
  }

  convertXFromScreenToScene(screenX: number): number {
    if (!this.webglRenderer.container) {
      return 0;
    }
    const container = this.webglRenderer.container.nativeElement;

    const containerWidth = container.clientWidth;
    const aspect = containerWidth / container.clientHeight;
    return (
      (screenX / containerWidth) * -2 * ((DEFAULT_FRUSTUM_SIZE / 1) * aspect)
    );
  }

  convertZFromScreenToScene(screenZ: number, scale = 1): number {
    if (!this.webglRenderer.container) {
      return 0;
    }
    const container = this.webglRenderer.container.nativeElement;

    const containerHeight = container.clientHeight;
    return (screenZ * DEFAULT_FRUSTUM_SIZE * 2) / scale / containerHeight;
  }

  convertScenePosToScreen(x: number, y: number): Point {
    const container = this.webglRenderer.container.nativeElement;
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    const pos = new THREE.Vector3(x, 0, y);
    pos.project(this.camera);
    return {
      x: (pos.x * containerWidth) / 2 + containerWidth / 2,
      y: -((pos.y * containerHeight) / 2) + containerHeight / 2,
    };
  }

  convertScreenPosToScene(x: number, y: number): Point {
    const vector = new THREE.Vector3();
    vector.set(
      (x / this.webglRenderer.canvas.nativeElement.offsetWidth) * 2 - 1,
      -(y / this.webglRenderer.canvas.nativeElement.offsetHeight) * 2 + 1,
      -1,
    );
    const pt = vector.unproject(this.camera);
    return {x: pt.x, y: pt.z};
  }

  // Used by tests only.
  scrollGraphArea(deltaX: number, deltaY: number) {
    const container = this.webglRenderer.container.nativeElement;
    const view = d3.select(container as Element);
    this.zoom.translateBy(view, deltaX, deltaY);
  }

  private handleZoom() {
    this.curScale = d3.event.transform.k;
    this.curTranslateX = d3.event.transform.x;
    this.curTranslateY = d3.event.transform.y;

    requestAnimationFrame(() => {
      if (!this.camera) {
        return;
      }

      this.setCameraFrustum();

      this.webglRenderer.updateNodeBgColorWhenFar();
      this.render();
      this.webglRenderer.handleHoveredGroupNodeIconChanged();
    });
  }

  private handleZoomEnd(savedTranslateX: number, savedTranslateY: number) {
    // Treat tiny amount of translation as clicking to improve user
    // experience.
    if (d3.event.sourceEvent && d3.event.sourceEvent.type === 'mouseup') {
      const deltaX = Math.abs(this.curTranslateX - savedTranslateX);
      const deltaY = Math.abs(this.curTranslateY - savedTranslateY);
      if (deltaX >= 0 && deltaX <= 3 && deltaY >= 0 && deltaY <= 3) {
        this.webglRenderer.handleClickOnGraph(d3.event.sourceEvent.shiftKey);
      }
    }
  }

  private setCameraFrustum() {
    const container = this.webglRenderer.container.nativeElement;

    const width = container.clientWidth;
    const height = container.clientHeight;
    const aspect = width / height;

    // Without going into too much detail, the following code maps the
    // d3.zoom's translation and scale level to camera's frustum area.
    //
    // Code reference: http://bl.ocks.org/nitaku/b25e6f091e97667c6cae
    const x = this.curTranslateX - width / 2;
    const y = this.curTranslateY - height / 2;

    const scaledFrustumSize = DEFAULT_FRUSTUM_SIZE / this.curScale;
    const offsetX = (x / width) * 2 * scaledFrustumSize * aspect;
    const offsetY = (y / height) * 2 * scaledFrustumSize;

    this.camera.left = -scaledFrustumSize * aspect - offsetX;
    this.camera.right = this.camera.left + 2 * scaledFrustumSize * aspect;
    this.camera.top = scaledFrustumSize + offsetY;
    this.camera.bottom = this.camera.top - 2 * scaledFrustumSize;
    this.camera.updateProjectionMatrix();
  }

  private resizeRendererToDisplaySize(render = true) {
    const container = this.webglRenderer.container.nativeElement;

    const canvas = this.renderer.domElement;
    const width = container.clientWidth;
    const height = container.clientHeight;
    if (width === 0 || height === 0) {
      return;
    }
    const needResize = canvas.width !== width || canvas.height !== height;
    if (needResize) {
      // Calculate the current center and keep the graph centered at that point.
      if (this.savedCenterX == null && this.savedCenterY == null) {
        const {x, y} = this.convertScreenPosToScene(width / 2, height / 2);
        this.savedCenterX = x;
        this.savedCenterY = y;
      }

      this.renderer.setSize(width, height, false);
      this.webglRenderer.canvas.nativeElement.style.width = `100%`;
      this.webglRenderer.canvas.nativeElement.style.height = `100%`;
      this.setCameraFrustum();
      this.render();

      this.zoomFit(
        {
          x: this.savedCenterX!,
          y: this.savedCenterY!,
          width: 0.0000001,
          height: 0.0000001,
        },
        0.9,
        0,
        true,
      );

      // Clear the currently saved center after a short timeout.
      if (this.resizeTimeoutRef >= 0) {
        window.clearTimeout(this.resizeTimeoutRef);
      }
      this.resizeTimeoutRef = window.setTimeout(() => {
        this.savedCenterX = null;
        this.savedCenterY = null;
      }, 500);
    }
  }

  private centerViewAt(
    centerX: number,
    centerY: number,
    scale: number,
    transitionDuration = 300,
  ) {
    if (!this.webglRenderer.container) {
      return;
    }
    const container = this.webglRenderer.container.nativeElement;
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;

    // Apply transform (translation + scale) through d3.zoom.
    const aspect = containerWidth / containerHeight;
    const targetCameraLeft =
      (-2 * DEFAULT_FRUSTUM_SIZE * aspect) / 2 / scale + centerX;
    const targetCameraTop = -centerY + DEFAULT_FRUSTUM_SIZE / scale;
    const transform = d3.zoomIdentity
      .scale(scale)
      .translate(
        this.convertXFromSceneToScreen(targetCameraLeft),
        this.convertZFromSceneToScreen(targetCameraTop),
      );
    const view = d3.select(container);
    if (transitionDuration === 0) {
      // tslint:disable-next-line:no-any
      (view as any).call(this.zoom.transform, transform);
    } else {
      // tslint:disable-next-line:no-any
      (view as any)
        .transition()
        .duration(
          this.webglRenderer.appService.testMode ? 0 : transitionDuration,
        )
        .ease(d3.easeExpOut)
        .call(this.zoom.transform, transform);
    }
  }
}
