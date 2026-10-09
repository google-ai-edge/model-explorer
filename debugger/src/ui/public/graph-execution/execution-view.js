/**
 * @license
 * Copyright 2026 The AI Edge Model Explorer Authors.
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

/* The parent owns inspection/mapping state; this frame owns only graph
 * presentation. */

(() => {
  'use strict';
  const CHANNEL = 'model-debugger-execution';
  const WORKER_SCRIPT_PATH =
      'upstream/model-explorer/dist/worker.js?v=33f00615723c8d92';
  function getElement(id) {
    const element = document.getElementById(id);
    if (!element) {
      throw new Error(`Missing required execution graph element: #${id}`);
    }
    return element;
  }
  function selectedKey(data) {
    return JSON.stringify([
      data.reference,
      data.target,
      data.operation,
      data.mapping,
    ]);
  }
  function isRecordObject(value) {
    return typeof value === 'object' && value !== null;
  }
  function matchesLastHit(lastHit, payload) {
    if (!lastHit) {
      return false;
    }
    const parsed = JSON.parse(lastHit);
    if (!isRecordObject(parsed) || !isRecordObject(parsed['hit'])) {
      return false;
    }
    const hit = parsed['hit'];
    if (parsed['type'] === 'tensorSelected') {
      const selectedId =
          hit['side'] === 'ref' ? payload.reference : payload.target;
      return selectedId === hit['id'];
    }
    return JSON.stringify(payload.operation) === JSON.stringify(hit);
  }
  /**
   * Encapsulates the execution graph iframe lifecycle and bridge state.
   */
  class ExecutionViewController {
    constructor(dataAdapter) {
      this.dataAdapter = dataAdapter;
      this.host = getElement('host');
      this.tooltip = getElement('tooltip');
      this.views = new Map();
      this.latest = null;
      this.current = null;
      this.element = null;
      this.generation = 0;
      this.ready = false;
      this.initialized = false;
      this.frame = 0;
      this.desiredSelections = ['', ''];
      this.choices = {};
      this.suppressSelectionEvents = 0;
      this.hover = null;
      this.pointer = null;
      this.pointerDownCoordinates = null;
      this.metricKey = '';
      this.lastHit = '';
      this.selectionKey = '';
      this.priorSelection = '';
      this.pendingReveal = false;
      this.pendingAction = '';
      this.mappingSnapshot = null;
      this.paintedTheme = '';
      this.themeWarning = '';
      const modelExplorerGlobal = window.modelExplorer;
      modelExplorerGlobal.assetFilesBaseUrl =
          'upstream/model-explorer/dist/static_files';
      modelExplorerGlobal.workerScriptPath = WORKER_SCRIPT_PATH;
      const resizeObserver = new ResizeObserver(() => {
        this.ensureViewport();
      });
      resizeObserver.observe(this.host);
      getElement('retry').onclick = () => {
        location.reload();
      };
      window.addEventListener('message', (event) => {
        if (event.source !== parent || event.origin !== location.origin ||
            event.data?.channel !== CHANNEL) {
          return;
        }
        if (event.data.type === 'render') {
          this.render(event.data.payload).catch((error) => {
            this.initializationFailed(error);
          });
        }
        if (event.data.type === 'action' &&
            (event.data.action === 'fit' || event.data.action === 'readable')) {
          this.perform(event.data.action);
        }
      });
      window.addEventListener('pagehide', () => {
        resizeObserver.disconnect();
        if (this.frame) {
          cancelAnimationFrame(this.frame);
        }
      });
      this.post({type: 'ready'});
    }
    post(message) {
      parent.postMessage({channel: CHANNEL, ...message}, location.origin);
    }
    showStatus(text) {
      getElement('status').textContent = this.themeWarning || text;
    }
    initializationFailed(error) {
      this.generation++;
      this.ready = false;
      this.initialized = false;
      this.host.dataset['viewportReady'] = 'false';
      if (this.frame) {
        cancelAnimationFrame(this.frame);
      }
      this.frame = 0;
      getElement('status').textContent =
          'Unable to initialize execution graph. Reload to retry.';
      getElement('retry').hidden = false;
      console.error(error);
    }
    applyTheme(payload) {
      const dark = payload?.theme === 'dark';
      document.documentElement.dataset['theme'] = dark ? 'dark' : 'light';
      const theme = dark ? 'dark' : 'light';
      if (this.ready && this.element && this.current &&
          this.paintedTheme !== theme) {
        // A cached/pre-upgrade renderer can still draw the graph. Report the
        // missing theme capability instead of throwing from its processed event
        // and stalling.
        if (typeof this.element.setPaneBackground !== 'function') {
          this.themeWarning =
              'Execution theme support is outdated. Reload execution graph.';
          getElement('retry').hidden = false;
          this.showStatus('');
          return;
        }
        try {
          const setPaneBackground =
              this.element.setPaneBackground.bind(this.element);
          const painted =
              this.current.panes
                  .map(
                      (pane) => !pane.graph.nodes.length ||
                          setPaneBackground(
                              dark ? '#202124' : '#ffffff', pane.paneIndex))
                  .every(Boolean);
          if (painted) {
            this.paintedTheme = theme;
            this.themeWarning = '';
            getElement('retry').hidden = true;
          }
        } catch {
          this.themeWarning =
              'Execution theme could not be applied. Reload execution graph.';
          getElement('retry').hidden = false;
          this.showStatus('');
        }
      }
    }
    saveViewport() {
      if (!this.ready || !this.element || !this.current) {
        return;
      }
      for (const pane of this.current.panes) {
        const view = this.element.readPaneViewport(pane.paneIndex);
        if (view) {
          this.views.set(pane.graph.id, view);
        }
      }
    }
    selectedNodes() {
      if (!this.current || !this.latest) {
        return ['', ''];
      }
      const latest = this.latest;
      const current = this.current;
      const operation = latest.operation;
      if (operation && !latest.mapping) {
        return current.panes.map(
            (pane) => [...current.lookup.values()]
                          .find(
                              (hit) => hit.kind === 'operation' &&
                                  hit.side === pane.side &&
                                  hit.side === operation.side &&
                                  hit.graphId === operation.graphId &&
                                  hit.nodeId === operation.nodeId)
                          ?.rendererNodeId ||
                '');
      }
      return current.panes.map((pane) => {
        const recordId =
            pane.side === 'ref' ? latest.reference || '' : latest.target || '';
        const hit = current.recordNodes.get(recordId);
        return hit?.side === pane.side ? hit.rendererNodeId : '';
      });
    }
    syncSelection(force = false) {
      if (!this.ready || !this.initialized || !this.element || !this.current) {
        return;
      }
      const element = this.element;
      const current = this.current;
      this.desiredSelections = this.selectedNodes();
      const nextKey = JSON.stringify(this.desiredSelections);
      if (force || nextKey !== this.selectionKey) {
        const revision = ++this.suppressSelectionEvents;
        this.desiredSelections.forEach((nodeId, pane) => {
          if (current.panes[pane].graph.nodes.length) {
            element.setPaneSelection(nodeId, pane);
          }
        });
        requestAnimationFrame(() => requestAnimationFrame(() => {
                                if (this.suppressSelectionEvents === revision) {
                                  this.suppressSelectionEvents = 0;
                                }
                              }));
        this.selectionKey = nextKey;
      }
      if (this.pendingReveal) {
        const done = this.desiredSelections
                         .map(
                             (nodeId, pane) => !nodeId ||
                                 element.focusNodeViewport(nodeId, pane, {
                                   labelSize: 12.5,
                                   duration: 0,
                                 }))
                         .every(Boolean);
        if (done) {
          this.pendingReveal = false;
        }
      }
    }
    readable() {
      if (!this.ready || !this.element || !this.current) {
        return false;
      }
      const element = this.element;
      const current = this.current;
      return current.panes
          .map((pane) => {
            const entry = pane.graph.nodes.find(
                              (node) => current.lookup.get(node.id)?.kind ===
                                      'operation' &&
                                  !node.incomingEdges.length) ||
                pane.graph.nodes[0];
            return (
                !entry || element.focusNodeViewport(entry.id, pane.paneIndex, {
                  labelSize: 12.5,
                  align: 'top',
                  paddingTop: 30,
                  duration: 0,
                }));
          })
          .every(Boolean);
    }
    initialize(revision, attempt = 0) {
      this.frame = 0;
      if (revision !== this.generation || !this.ready || !this.element ||
          !this.current || !this.host.clientWidth || !this.host.clientHeight) {
        return;
      }
      const current = this.current;
      const element = this.element;
      try {
        let done = true;
        for (const pane of current.panes) {
          if (!pane.graph.nodes.length) {
            continue;
          }
          const saved = this.views.get(pane.graph.id);
          const entry =
              pane.graph.nodes.find(
                  (node) => current.lookup.get(node.id)?.kind === 'operation' &&
                      !node.incomingEdges.length) ||
              pane.graph.nodes[0];
          const restored = saved ?
              element.setPaneViewport(saved, pane.paneIndex) :
              element.focusNodeViewport(entry.id, pane.paneIndex, {
                labelSize: 12.5,
                align: 'top',
                paddingTop: 30,
                duration: 0,
              });
          done = restored && done;
        }
        if (!done && attempt < 120) {
          this.frame = requestAnimationFrame(() => {
            this.initialize(revision, attempt + 1);
          });
        } else {
          this.initialized = done;
          this.host.dataset['viewportReady'] = String(done);
          if (done) {
            this.applyTheme(this.latest);
            this.syncSelection(true);
            if (this.pendingAction) {
              const action = this.pendingAction;
              this.pendingAction = '';
              this.perform(action);
            }
          }
        }
      } catch (error) {
        this.initializationFailed(error);
      }
    }
    ensureViewport() {
      if (this.ready && !this.initialized && !this.frame) {
        this.frame = requestAnimationFrame(() => {
          this.initialize(this.generation);
        });
      }
    }
    updateData() {
      if (!this.ready || !this.element || !this.current || !this.latest) {
        return;
      }
      const nextKey = JSON.stringify([
        this.latest.metric,
        this.latest.metrics,
        this.latest.reference,
        this.latest.target,
        this.latest.theme,
      ]);
      if (nextKey === this.metricKey) {
        return;
      }
      const data = this.dataAdapter.buildNodeData(this.current, this.latest);
      for (const pane of this.current.panes) {
        this.element.addNodeDataProviderDataWithGraphIndex(
            this.latest.metric || 'Metric',
            {[pane.graph.id]: data[pane.graph.id]}, pane.paneIndex,
            pane.paneIndex === 0);
      }
      this.metricKey = nextKey;
      const metricName = this.latest.metric || '';
      const metricValue = this.latest.metrics?.[metricName];
      getElement('metric').textContent =
          `Selected pair · ${this.latest.metric || 'Metric'} ${
              this.latest.reference && this.latest.target &&
                      typeof metricValue === 'number' &&
                      Number.isFinite(metricValue) ?
                  Number(metricValue.toPrecision(5)) :
                  'Not captured'}`;
    }
    selectHit(hit, reveal = false) {
      if (!hit || !this.ready || !this.element || !this.latest) {
        return;
      }
      if (this.latest.mapping && (hit.kind !== 'tensor' || !hit.recordId)) {
        this.syncSelection(true);
        return;
      }
      const selection = hit.kind === 'tensor' && hit.recordId ?
          {type: 'tensorSelected', hit: {side: hit.side, id: hit.recordId}} :
          {
            type: 'operationSelected',
            hit: {side: hit.side, nodeId: hit.nodeId, graphId: hit.graphId},
          };
      const hitKey = JSON.stringify(selection);
      if (hitKey !== this.lastHit) {
        this.lastHit = hitKey;
        this.post(selection);
      }
      // Parent acknowledgement updates the selection. Restore any native clear
      // or mapping-disallowed operation selection without moving either camera.
      this.syncSelection(true);
      if (reveal) {
        this.element.focusNodeViewport(hit.rendererNodeId, hit.paneIndex, {
          labelSize: 12.5,
          duration: 0,
        });
      }
    }
    showTooltip() {
      if (!this.hover || !this.pointer || !this.ready) {
        this.tooltip.hidden = true;
        return;
      }
      this.tooltip.textContent =
          `${this.hover.side === 'ref' ? 'Reference' : 'Target'} · ${
              this.hover.kind === 'tensor' ?
                  'Tensor' :
                  'Operation'}\n${this.hover.label}` +
          (this.hover.kind === 'tensor' ? `\nOutput ${this.hover.outputId} · ${
                                              this.hover.recordId ?
                                                  'Captured tensor' :
                                                  this.hover.recordIds?.length ?
                                                  'Multiple captured records' :
                                                  'Not captured'}` :
                                          `\n${this.hover.nodeId}`);
      this.tooltip.hidden = false;
      const bounds = getElement('canvas').getBoundingClientRect();
      this.tooltip.style.left = `${
          Math.max(
              8,
              Math.min(
                  this.pointer.x - bounds.left + 14,
                  bounds.width - this.tooltip.offsetWidth - 8))}px`;
      this.tooltip.style.top = `${
          Math.max(
              8,
              Math.min(
                  this.pointer.y - bounds.top + 14,
                  bounds.height - this.tooltip.offsetHeight - 8))}px`;
    }
    renderLabels() {
      if (!this.current) {
        return;
      }
      const panesContainer = getElement('panes');
      panesContainer.replaceChildren();
      for (const pane of this.current.panes) {
        const label = document.createElement('div');
        const title = document.createElement('strong');
        const run = document.createElement('small');
        label.className = 'paneLabel';
        title.textContent = pane.side === 'ref' ? 'Reference' : 'Target';
        run.textContent = pane.run?.runtime || pane.run?.id || 'Not captured';
        label.append(title, run);
        if (pane.run && pane.run.graphs.length > 1) {
          const select = document.createElement('select');
          select.setAttribute(
              'aria-label', `${title.textContent} execution graph`);
          for (const graph of pane.run.graphs) {
            const option = document.createElement('option');
            option.value = graph.id;
            option.textContent = graph.id;
            select.append(option);
          }
          select.value = pane.original?.id || '';
          select.onchange = () => {
            this.choices[pane.side] = select.value;
            this.render(this.latest, true).catch((error) => {
              this.initializationFailed(error);
            });
          };
          label.append(select);
        }
        panesContainer.append(label);
      }
    }
    async render(payload, keepChoices = false) {
      this.applyTheme(payload);
      const previous = this.latest;
      let restoreViews = null;
      if (!payload || !Array.isArray(payload.details?.executions)) {
        this.generation++;
        this.ready = false;
        this.initialized = false;
        this.latest = payload;
        this.current = null;
        this.element = null;
        if (this.frame) {
          cancelAnimationFrame(this.frame);
        }
        this.frame = 0;
        this.host.replaceChildren();
        getElement('panes').replaceChildren();
        getElement('empty').hidden = false;
        this.themeWarning = '';
        getElement('retry').hidden = true;
        getElement('status').textContent = '';
        getElement('metric').textContent = '';
        return;
      }
      if (!keepChoices) {
        const sides = this.dataAdapter.executionSides(payload);
        for (const {side, selected, run} of sides) {
          const currentSelection =
              side === 'ref' ? payload.reference : payload.target;
          const previousSelection =
              side === 'ref' ? previous?.reference : previous?.target;
          if (currentSelection !== previousSelection && selected?.graph) {
            this.choices[side] = String(selected.graph);
          }
          if (!run?.graphs.some((graph) => graph.id === this.choices[side])) {
            delete this.choices[side];
          }
        }
        if (payload.operation &&
            JSON.stringify(payload.operation) !==
                JSON.stringify(previous?.operation)) {
          this.choices[payload.operation.side] = payload.operation.graphId;
        }
      }
      if (payload.mapping && !previous?.mapping && this.element &&
          this.current && previous) {
        const element = this.element;
        this.mappingSnapshot = {
          reference: previous.reference,
          target: previous.target,
          choices: {...this.choices},
          views: this.current.panes.map(
              (pane) => element.readPaneViewport(pane.paneIndex)),
        };
      }
      if (!payload.mapping && previous?.mapping && this.mappingSnapshot) {
        if (payload.reference === this.mappingSnapshot.reference &&
            payload.target === this.mappingSnapshot.target) {
          this.choices = this.mappingSnapshot.choices;
          restoreViews = this.mappingSnapshot.views;
        }
        this.mappingSnapshot = null;
      }
      this.latest = payload;
      const next = this.dataAdapter.buildExecutionGraphs(payload, this.choices);
      const nextSelection = selectedKey(payload);
      // Parent selection changes reveal the selected tensor only when it was
      // not just selected by this canvas. Metric-only updates keep the current
      // view.
      const localSelection = matchesLastHit(this.lastHit, payload);
      if (this.priorSelection !== nextSelection) {
        this.pendingReveal =
            !localSelection && !payload.mapping && !restoreViews;
        if (!localSelection) {
          this.lastHit = '';
        }
        this.priorSelection = nextSelection;
      }
      if (previous?.mapping !== payload.mapping) {
        this.lastHit = '';
      }
      if (this.element && this.current?.topologyKey === next.topologyKey) {
        const element = this.element;
        this.current = next;
        if (restoreViews) {
          restoreViews.forEach((view, pane) => {
            if (view) {
              element.setPaneViewport(view, pane);
            }
          });
        }
        this.updateData();
        this.syncSelection();
        this.ensureViewport();
        return;
      }
      this.saveViewport();
      if (restoreViews) {
        restoreViews.forEach((view) => {
          if (view) {
            this.views.set(view.graphId, view);
          }
        });
      }
      if (this.frame) {
        cancelAnimationFrame(this.frame);
      }
      this.frame = 0;
      this.current = next;
      this.ready = false;
      this.initialized = false;
      this.metricKey = '';
      this.selectionKey = '';
      this.hover = null;
      this.pointerDownCoordinates = null;
      this.tooltip.hidden = true;
      this.renderLabels();
      const count = this.current.panes.reduce(
          (total, pane) => total + pane.graph.nodes.length, 0);
      getElement('empty').hidden = count > 0;
      getElement('empty').textContent =
          'No execution graph was captured for this node.';
      this.showStatus(count ? 'Loading graph…' : 'Not captured');
      const revision = ++this.generation;
      if (!count) {
        this.host.replaceChildren();
        this.element = null;
        getElement('metric').textContent = '';
        return;
      }
      await customElements.whenDefined('model-explorer-visualizer');
      if (revision !== this.generation) {
        return;
      }
      const element = document.createElement('model-explorer-visualizer');
      this.element = element;
      this.paintedTheme = '';
      element.workerScriptPath = WORKER_SCRIPT_PATH;
      element.graphCollections = this.current.graphCollections;
      element.initialUiState = this.current.initialUiState;
      element.config = {
        hideTitleBar: true,
        hideToolBar: true,
        hideInfoPanel: true,
        hideLegends: true,
        hideEmptyNodeDataEntries: true,
        edgeColor: '#b8c1cc',
      };
      const processed = new Set();
      element.addEventListener('modelGraphProcessed', (event) => {
        if (revision !== this.generation || !this.current) {
          return;
        }
        try {
          processed.add(event.detail.paneIndex);
          if (this.current.panes.some(
                  (pane) => pane.graph.nodes.length &&
                      !processed.has(pane.paneIndex))) {
            return;
          }
          this.ready = true;
          this.applyTheme(this.latest);
          this.updateData();
          this.showStatus('Operation → output tensor');
          this.ensureViewport();
        } catch (error) {
          this.initializationFailed(error);
        }
      });
      element.addEventListener('selectedNodeChanged', (event) => {
        if (revision !== this.generation || !this.ready ||
            this.suppressSelectionEvents || !this.current) {
          return;
        }
        if (!event.detail?.nodeId) {
          this.syncSelection(true);
          return;
        }
        this.selectHit(this.current.lookup.get(event.detail.nodeId));
      });
      element.addEventListener('uiStateChanged', (event) => {
        if (revision === this.generation) {
          const fraction = event.detail?.paneStates?.[0]?.widthFraction ?? 0.5;
          getElement('panes').style.gridTemplateColumns =
              `${fraction}fr ${1 - fraction}fr`;
        }
      });
      element.addEventListener('hoveredNodeChanged', (event) => {
        if (revision === this.generation && this.current) {
          this.hover = this.current.lookup.get(event.detail?.nodeId || '');
          this.showTooltip();
        }
      });
      element.addEventListener('pointermove', (event) => {
        this.pointer = {x: event.clientX, y: event.clientY};
        this.showTooltip();
      });
      element.addEventListener('pointerleave', () => {
        this.hover = null;
        this.tooltip.hidden = true;
      });
      element.addEventListener('pointerdown', (event) => {
        this.pointerDownCoordinates = event.button === 0 &&
                event.composedPath().some(
                    (node) =>
                        node instanceof Element && node.tagName === 'CANVAS') ?
            {x: event.clientX, y: event.clientY} :
            null;
      });
      element.addEventListener('pointerup', (event) => {
        const start = this.pointerDownCoordinates;
        this.pointerDownCoordinates = null;
        if (revision === this.generation && start && event.button === 0 &&
            Math.hypot(event.clientX - start.x, event.clientY - start.y) <= 4) {
          this.selectHit(this.hover || undefined);
        }
      });
      element.addEventListener('pointercancel', () => {
        this.pointerDownCoordinates = null;
      });
      this.host.replaceChildren(element);
      const style = document.createElement('style');
      style.textContent =
          '.pane-title-container,.sync-navigation-container{display:none!important}';
      element.shadowRoot?.append(style);
      this.host.dataset['mountCount'] =
          String(Number(this.host.dataset['mountCount'] || 0) + 1);
    }
    perform(action) {
      if (!this.ready || !this.initialized || !this.element || !this.current) {
        this.pendingAction = action;
        return;
      }
      const element = this.element;
      if (action === 'fit') {
        this.current.panes.forEach((pane) => {
          if (pane.graph.nodes.length) {
            element.fitPaneGraph(pane.paneIndex);
          }
        });
      }
      if (action === 'readable') {
        this.readable();
      }
    }
  }
  function bootstrapExecutionView(dataAdapter = window.ExecutionGraphData) {
    return new ExecutionViewController(dataAdapter);
  }
  if (typeof window !== 'undefined' && typeof document !== 'undefined' &&
      document.getElementById('host') && window.ExecutionGraphData) {
    bootstrapExecutionView();
  }
})();
