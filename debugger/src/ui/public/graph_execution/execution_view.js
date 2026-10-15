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

/* The parent owns inspection/mapping state; this frame owns only graph presentation. */
(() => {
  'use strict';
  const channel = 'model-debugger-execution', $ = id => document.getElementById(id);
  const host = $('host'), tooltip = $('tooltip');
  const {buildExecutionGraphs, buildNodeData, executionSides} = window.ExecutionGraphData;
  window.modelExplorer.assetFilesBaseUrl = 'upstream/model_explorer/dist/static_files';
  window.modelExplorer.workerScriptPath = 'upstream/model_explorer/dist/worker.js?v=f857968b63b0a40f';
  let latest, current, element, generation = 0, ready = false, initialized = false, frame = 0;
  let desired = ['', ''], choices = {}, suppress = 0, hover, pointer, down, metricKey = '', lastHit = '';
  let selectionKey = '', priorSelection = '', pendingReveal = false, pendingAction = '';
  let mappingSnapshot = null, paintedTheme = '', themeWarning = '';
  const views = new Map();
  const post = message => parent.postMessage({channel, ...message}, location.origin);
  const showStatus = text => { $('status').textContent = themeWarning || text; };
  function initializationFailed(error) {
    generation++; ready = false; initialized = false;
    host.dataset.viewportReady = 'false';
    if (frame) cancelAnimationFrame(frame); frame = 0;
    $('status').textContent = 'Unable to initialize execution graph. Reload to retry.';
    $('retry').hidden = false;
    console.error(error);
  }
  function applyTheme(payload) {
    const dark = payload?.theme === 'dark';
    document.documentElement.dataset.theme = dark ? 'dark' : 'light';
    const theme = dark ? 'dark' : 'light';
    if (ready && element && paintedTheme !== theme) {
      // A cached/pre-upgrade renderer can still draw the graph. Report the missing
      // theme capability instead of throwing from its processed event and stalling.
      if (typeof element.setPaneBackground !== 'function') {
        themeWarning = 'Execution theme support is outdated. Reload execution graph.';
        $('retry').hidden = false; showStatus(''); return;
      }
      try {
        const painted = current.panes.map(pane => !pane.graph.nodes.length ||
          element.setPaneBackground(dark ? '#202124' : '#ffffff', pane.paneIndex)).every(Boolean);
        if (painted) { paintedTheme = theme; themeWarning = ''; $('retry').hidden = true; }
      } catch (_) {
        themeWarning = 'Execution theme could not be applied. Reload execution graph.';
        $('retry').hidden = false; showStatus('');
      }
    }
  }
  const selectedKey = data => JSON.stringify([data.reference, data.target, data.operation, data.mapping]);
  const saveViewport = () => {
    if (!ready || !element) return;
    current.panes.forEach(pane => { const view = element.readPaneViewport(pane.paneIndex); if (view) views.set(pane.graph.id, view); });
  };
  function selectedNodes() {
    if (latest.operation && !latest.mapping) return current.panes.map(pane => [...current.lookup.values()].find(hit =>
      hit.kind === 'operation' && hit.side === pane.side && hit.side === latest.operation.side
      && hit.graphId === latest.operation.graphId && hit.nodeId === latest.operation.nodeId)?.rendererNodeId || '');
    return current.panes.map(pane => {
      const hit = current.recordNodes.get(pane.side === 'ref' ? latest.reference : latest.target);
      return hit?.side === pane.side ? hit.rendererNodeId : '';
    });
  }
  function syncSelection(force = false) {
    if (!ready || !initialized) return;
    desired = selectedNodes();
    const nextKey = JSON.stringify(desired);
    if (force || nextKey !== selectionKey) {
      const revision = ++suppress;
      desired.forEach((nodeId, pane) => { if (current.panes[pane].graph.nodes.length) element.setPaneSelection(nodeId, pane); });
      requestAnimationFrame(() => requestAnimationFrame(() => { if (suppress === revision) suppress = 0; }));
      selectionKey = nextKey;
    }
    if (pendingReveal) {
      const done = desired.map((nodeId, pane) => !nodeId || element.focusNodeViewport(nodeId, pane, {labelSize: 12.5, duration: 0})).every(Boolean);
      if (done) pendingReveal = false;
    }
  }
  function readable() {
    if (!ready) return false;
    return current.panes.map(pane => {
      const entry = pane.graph.nodes.find(node => current.lookup.get(node.id)?.kind === 'operation' && !node.incomingEdges.length) || pane.graph.nodes[0];
      return !entry || element.focusNodeViewport(entry.id, pane.paneIndex, {labelSize: 12.5, align: 'top', paddingTop: 30, duration: 0});
    }).every(Boolean);
  }
  function initialize(revision, attempt = 0) {
    frame = 0;
    if (revision !== generation || !ready || !host.clientWidth || !host.clientHeight) return;
    try {
    let done = true;
    for (const pane of current.panes) {
      if (!pane.graph.nodes.length) continue;
      const saved = views.get(pane.graph.id);
      const entry = pane.graph.nodes.find(node => current.lookup.get(node.id)?.kind === 'operation' && !node.incomingEdges.length) || pane.graph.nodes[0];
      const restored = saved ? element.setPaneViewport(saved, pane.paneIndex) : element.focusNodeViewport(entry.id, pane.paneIndex,
        {labelSize: 12.5, align: 'top', paddingTop: 30, duration: 0});
      done = restored && done;
    }
    if (!done && attempt < 120) frame = requestAnimationFrame(() => initialize(revision, attempt + 1));
    else {
      initialized = done; host.dataset.viewportReady = String(done);
      if (done) applyTheme(latest);
      if (done) { syncSelection(true); if (pendingAction) { const action = pendingAction; pendingAction = ''; perform(action); } }
    }
    } catch (error) { initializationFailed(error); }
  }
  function ensureViewport() { if (ready && !initialized && !frame) frame = requestAnimationFrame(() => initialize(generation)); }
  const resize = new ResizeObserver(ensureViewport);
  resize.observe(host); // Defer scripts run after the actual host element exists.

  function updateData() {
    if (!ready) return;
    const nextKey = JSON.stringify([latest.metric, latest.metrics, latest.reference, latest.target, latest.theme]);
    if (nextKey === metricKey) return;
    const data = buildNodeData(current, latest);
    current.panes.forEach(pane => element.addNodeDataProviderDataWithGraphIndex(latest.metric || 'Metric',
      {[pane.graph.id]: data[pane.graph.id]}, pane.paneIndex, pane.paneIndex === 0));
    metricKey = nextKey;
    const number = latest.metrics?.[latest.metric];
    $('metric').textContent = `Selected pair · ${latest.metric || 'Metric'} ${latest.reference && latest.target && typeof number === 'number' && Number.isFinite(number) ? Number(number.toPrecision(5)) : 'Not captured'}`;
  }
  function selectHit(hit, reveal = false) {
    if (!hit || !ready) return;
    if (latest.mapping && (hit.kind !== 'tensor' || !hit.recordId)) { syncSelection(true); return; }
    const selection = hit.kind === 'tensor' && hit.recordId
      ? {type: 'tensorSelected', hit: {side: hit.side, id: hit.recordId}}
      : {type: 'operationSelected', hit: {side: hit.side, nodeId: hit.nodeId, graphId: hit.graphId}};
    const hitKey = JSON.stringify(selection);
    if (hitKey !== lastHit) { lastHit = hitKey; post(selection); }
    // Parent acknowledgement updates the selection. Restore any native clear
    // or mapping-disallowed operation selection without moving either camera.
    syncSelection(true);
    if (reveal) element.focusNodeViewport(hit.rendererNodeId, hit.paneIndex, {labelSize: 12.5, duration: 0});
  }
  function showTooltip() {
    if (!hover || !pointer || !ready) { tooltip.hidden = true; return; }
    tooltip.textContent = `${hover.side === 'ref' ? 'Reference' : 'Target'} · ${hover.kind === 'tensor' ? 'Tensor' : 'Operation'}\n${hover.label}`
      + (hover.kind === 'tensor' ? `\nOutput ${hover.outputId} · ${hover.recordId ? 'Captured tensor' : hover.recordIds.length ? 'Multiple captured records' : 'Not captured'}` : `\n${hover.nodeId}`);
    tooltip.hidden = false;
    const bounds = $('canvas').getBoundingClientRect();
    tooltip.style.left = Math.max(8, Math.min(pointer.x - bounds.left + 14, bounds.width - tooltip.offsetWidth - 8)) + 'px';
    tooltip.style.top = Math.max(8, Math.min(pointer.y - bounds.top + 14, bounds.height - tooltip.offsetHeight - 8)) + 'px';
  }
  function renderLabels() {
    $('panes').replaceChildren();
    for (const pane of current.panes) {
      const label = document.createElement('div'), title = document.createElement('strong'), run = document.createElement('small');
      label.className = 'paneLabel'; title.textContent = pane.side === 'ref' ? 'Reference' : 'Target'; run.textContent = pane.run?.runtime || pane.run?.id || 'Not captured';
      label.append(title, run);
      if (pane.run?.graphs.length > 1) {
        const select = document.createElement('select'); select.setAttribute('aria-label', `${title.textContent} execution graph`);
        for (const graph of pane.run.graphs) { const option = document.createElement('option'); option.value = graph.id; option.textContent = graph.id; select.append(option); }
        select.value = pane.original?.id || ''; select.onchange = () => { choices[pane.side] = select.value; render(latest, true); }; label.append(select);
      }
      $('panes').append(label);
    }
  }
  async function render(payload, keepChoices = false) {
    applyTheme(payload);
    const previous = latest;
    let restoreViews = null;
    if (!payload || !Array.isArray(payload.details?.executions)) {
      generation++; ready = false; initialized = false; latest = payload; current = null; element = null;
      if (frame) cancelAnimationFrame(frame); frame = 0; host.replaceChildren(); $('panes').replaceChildren();
      $('empty').hidden = false; themeWarning = ''; $('retry').hidden = true; $('status').textContent = ''; $('metric').textContent = ''; return;
    }
    if (!keepChoices) {
      const sides = executionSides(payload);
      for (const {side, selected, run} of sides) {
        const field = side === 'ref' ? 'reference' : 'target';
        if (payload[field] !== previous?.[field] && selected?.graph) choices[side] = selected.graph;
        if (!run?.graphs.some(graph => graph.id === choices[side])) delete choices[side];
      }
      if (payload.operation && JSON.stringify(payload.operation) !== JSON.stringify(previous?.operation)) choices[payload.operation.side] = payload.operation.graphId;
    }
    if (payload.mapping && !previous?.mapping && element) {
      mappingSnapshot = {reference: previous.reference, target: previous.target, choices: {...choices}, views: current.panes.map(pane => element.readPaneViewport(pane.paneIndex))};
    }
    if (!payload.mapping && previous?.mapping && mappingSnapshot) {
      if (payload.reference === mappingSnapshot.reference && payload.target === mappingSnapshot.target) {
        choices = mappingSnapshot.choices;
        restoreViews = mappingSnapshot.views;
      }
      mappingSnapshot = null;
    }
    latest = payload;
    const next = buildExecutionGraphs(payload, choices), nextSelection = selectedKey(payload);
    // Parent selection changes reveal the selected tensor only when it was not
    // just selected by this canvas. Metric-only updates keep the current view.
    const localSelection = lastHit && (() => { const hit = JSON.parse(lastHit); return hit.type === 'tensorSelected'
      ? payload[hit.hit.side === 'ref' ? 'reference' : 'target'] === hit.hit.id
      : JSON.stringify(payload.operation) === JSON.stringify(hit.hit); })();
    if (priorSelection !== nextSelection) {
      pendingReveal = !localSelection && !payload.mapping && !restoreViews;
      if (!localSelection) lastHit = '';
      priorSelection = nextSelection;
    }
    if (previous?.mapping !== payload.mapping) lastHit = '';
    if (element && current?.topologyKey === next.topologyKey) {
      current = next;
      if (restoreViews) restoreViews.forEach((view, pane) => { if (view) element.setPaneViewport(view, pane); });
      updateData(); syncSelection(); ensureViewport(); return;
    }
    saveViewport();
    if (restoreViews) restoreViews.forEach(view => { if (view) views.set(view.graphId, view); });
    if (frame) cancelAnimationFrame(frame); frame = 0;
    current = next; ready = false; initialized = false; metricKey = ''; selectionKey = ''; hover = null; down = null; tooltip.hidden = true;
    renderLabels();
    const count = current.panes.reduce((total, pane) => total + pane.graph.nodes.length, 0);
    $('empty').hidden = count > 0; $('empty').textContent = 'No execution graph was captured for this node.';
    showStatus(count ? 'Loading graph…' : 'Not captured');
    const revision = ++generation;
    if (!count) { host.replaceChildren(); element = null; $('metric').textContent = ''; return; }
    await customElements.whenDefined('model-explorer-visualizer');
    if (revision !== generation) return;
    element = document.createElement('model-explorer-visualizer');
    paintedTheme = '';
    element.graphCollections = current.graphCollections; element.initialUiState = current.initialUiState;
    element.config = {hideTitleBar: true, hideToolBar: true, hideInfoPanel: true, hideLegends: true, hideEmptyNodeDataEntries: true, edgeColor: '#b8c1cc'};
    const processed = new Set();
    element.addEventListener('modelGraphProcessed', event => {
      if (revision !== generation) return;
      try {
        processed.add(event.detail.paneIndex);
        if (current.panes.some(pane => pane.graph.nodes.length && !processed.has(pane.paneIndex))) return;
        ready = true; applyTheme(latest); updateData(); showStatus('Operation → output tensor'); ensureViewport();
      } catch (error) { initializationFailed(error); }
    });
    element.addEventListener('selectedNodeChanged', event => {
      if (revision !== generation || !ready || suppress) return;
      if (!event.detail?.nodeId) { syncSelection(true); return; }
      selectHit(current.lookup.get(event.detail.nodeId));
    });
    element.addEventListener('uiStateChanged', event => { if (revision === generation) { const fraction = event.detail?.paneStates?.[0]?.widthFraction || .5; $('panes').style.gridTemplateColumns = `${fraction}fr ${1 - fraction}fr`; } });
    element.addEventListener('hoveredNodeChanged', event => { if (revision === generation) { hover = current.lookup.get(event.detail?.nodeId); showTooltip(); } });
    element.addEventListener('pointermove', event => { pointer = {x: event.clientX, y: event.clientY}; showTooltip(); });
    element.addEventListener('pointerleave', () => { hover = null; tooltip.hidden = true; });
    element.addEventListener('pointerdown', event => { down = event.button === 0 && event.composedPath().some(node => node.tagName === 'CANVAS') ? {x: event.clientX, y: event.clientY} : null; });
    element.addEventListener('pointerup', event => { const start = down; down = null; if (revision === generation && start && event.button === 0 && Math.hypot(event.clientX - start.x, event.clientY - start.y) <= 4) selectHit(hover); });
    element.addEventListener('pointercancel', () => { down = null; });
    host.replaceChildren(element);
    const style = document.createElement('style'); style.textContent = '.pane-title-container,.sync-navigation-container{display:none!important}'; element.shadowRoot.append(style);
    host.dataset.mountCount = String(Number(host.dataset.mountCount || 0) + 1);
  }
  function perform(action) {
    if (!ready || !initialized) { pendingAction = action; return; }
    if (action === 'fit') current.panes.forEach(pane => { if (pane.graph.nodes.length) element.fitPaneGraph(pane.paneIndex); });
    if (action === 'readable') readable();
  }
  $('retry').onclick = () => location.reload();
  window.addEventListener('message', event => {
    if (event.source !== parent || event.origin !== location.origin || event.data?.channel !== channel) return;
    if (event.data.type === 'render') render(event.data.payload).catch(initializationFailed);
    if (event.data.type === 'action') perform(event.data.action);
  });
  window.addEventListener('pagehide', () => { resize.disconnect(); if (frame) cancelAnimationFrame(frame); });
  post({type: 'ready'});
})();
