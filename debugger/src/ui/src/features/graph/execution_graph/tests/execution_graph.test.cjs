#!/usr/bin/env node
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

'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const root = path.resolve(__dirname, '../../../../../public/graph-execution');
const dataSource =
    fs.readFileSync(path.join(root, 'execution-data.js'), 'utf8');
const viewSource =
    fs.readFileSync(path.join(root, 'execution-view.js'), 'utf8');

function fixture() {
  const node = {
    id: 'multi',
    label: 'MultiOutput',
    namespace: '',
    incomingEdges: [],
    outputsMetadata: [
      {id: '0', attrs: [{key: 'tensor_name', value: 'first'}]},
      {id: '1', attrs: [{key: 'tensor_name', value: 'uncaptured'}]},
    ]
  };
  const consumer = {
    id: 'consumer',
    label: 'Consume',
    namespace: '',
    incomingEdges: [
      {sourceNodeId: 'multi', sourceNodeOutputId: '1', targetNodeInputId: '0'}
    ]
  };
  const executions = ['run-r', 'run-t'].map(
      run => ({id: run, graphs: [{id: 'g', nodes: [node, consumer]}]}));
  const tensors = ['run-r', 'run-t'].map((run, index) => ({
                                           id: index ? 't0' : 'r0',
                                           run,
                                           graph: 'g',
                                           node: 'multi',
                                           output: '0',
                                           layer: 0,
                                           batch: 0,
                                           sample: null,
                                           shape: [1, 4],
                                           dtype: 'float32'
                                         }));
  return {
    details: {executions, tensors},
    reference: 'r0',
    target: 't0',
    metric: 'mse',
    metrics: {mse: .01},
    mapping: false,
    operation: null
  };
}

function harness(options = {}) {
  const elements = new Map();
  const mounts = [];
  const frames = new Map();
  const messages = [];
  const errors = [];
  const listeners = new Map();
  let nextFrame = 0;
  let reloads = 0;
  const whenDefinedCalls = [];
  class Element {
    constructor(tag = 'div') {
      Object.assign(this, {
        tagName: tag.toUpperCase(),
        children: [],
        listeners: new Map(),
        style: {},
        dataset: {},
        value: '',
        clientWidth: 900,
        clientHeight: 600,
        calls: [],
        selected: ['', ''],
        open: false
      });
    }
    addEventListener(type, listener) {
      if (!this.listeners.has(type)) {
        this.listeners.set(type, []);
      }
      this.listeners.get(type).push(listener);
    }
    emit(type, event) {
      for (const callback of this.listeners.get(type) || []) {
        callback(event);
      }
    }
    setAttribute() {}
    append(...items) {
      this.children.push(...items);
    }
    replaceChildren(...items) {
      this.children = items;
    }
    matches() {
      return this.open;
    }
    hidePopover() {
      this.open = false;
    }
    showPopover() {
      this.open = true;
    }
    focus() {}
    querySelector() {
      return this.children.find(child => child.tagName === 'BUTTON');
    }
    getBoundingClientRect() {
      return {left: 0, top: 0, width: 900, height: 600};
    }
    setPaneSelection(nodeId, pane) {
      this.selected[pane] = nodeId;
      this.calls.push(['select', nodeId, pane]);
      return true;
    }
    setPaneBackground(color, pane) {
      this.calls.push(['background', color, pane]);
      return true;
    }
    focusNodeViewport(nodeId, pane) {
      this.calls.push(['focus', nodeId, pane]);
      return true;
    }
    readPaneViewport(pane) {
      return {
        graphId: this.graphCollections[0].graphs[pane].id,
        centerX: 55,
        centerY: 77,
        pixelsPerUnit: 1
      };
    }
    setPaneViewport(view, pane) {
      this.calls.push(['restore', view, pane]);
      return true;
    }
    fitPaneGraph(pane) {
      this.calls.push(['fit', pane]);
      return true;
    }
    addNodeDataProviderDataWithGraphIndex(...args) {
      this.calls.push(['data', ...args]);
    }
  }
  const document = {
    documentElement: new Element('html'),
    getElementById(id) {
      if (!elements.has(id)) {
        elements.set(id, new Element());
      }
      return elements.get(id);
    },
    createElement(tag) {
      const element = new Element(tag);
      if (tag === 'model-explorer-visualizer') {
        element.shadowRoot = new Element();
        mounts.push(element);
      }
      return element;
    },
  };
  const parent = {postMessage: message => messages.push(message)};
  const context = {
    Element,
    document,
    parent,
    location: {
      origin: 'http://test',
      reload() {
        reloads++;
      }
    },
    modelExplorer: {},
    addEventListener(type, listener) {
      listeners.set(type, listener);
    },
    ResizeObserver: class {
      observe(element) {
        assert.ok(element instanceof Element);
      } disconnect() {}
    },
    requestAnimationFrame: callback => {
      frames.set(++nextFrame, callback);
      return nextFrame;
    },
    cancelAnimationFrame: frame => frames.delete(frame),
    customElements: {
      whenDefined: name => {
        whenDefinedCalls.push(name);
        return options.whenDefined ? options.whenDefined(name) :
                                     Promise.resolve();
      }
    },
    console: {error: error => errors.push(error)},
  };
  context.window = context;
  vm.createContext(context);
  vm.runInContext(dataSource, context);
  vm.runInContext(viewSource, context);
  const flush = (expectedErrors = 0) => {
    for (let tries = 0; frames.size && tries < 130; tries++) {
      const callbacks = [...frames.values()];
      frames.clear();
      callbacks.forEach(callback => callback());
    }
    assert.equal(frames.size, 0);
    assert.equal(errors.length, expectedErrors);
  };
  const send = async message => {
    listeners.get('message')({
      source: parent,
      origin: 'http://test',
      data: {channel: 'model-debugger-execution', ...message}
    });
    await Promise.resolve();
    await Promise.resolve();
  };
  const render = payload => send({type: 'render', payload});
  const load = () => {
    const element = mounts.at(-1);
    for (const paneIndex of [0, 1]) {
      element.emit('modelGraphProcessed', {detail: {paneIndex}});
    }
    flush();
    return element;
  };
  return {
    render,
    send,
    load,
    flush,
    mounts,
    messages,
    errors,
    whenDefinedCalls,
    reloads: () => reloads,
    document,
    data: context.ExecutionGraphData,
    get: id => document.getElementById(id)
  };
}

test(
    'real inventories retain multi-output topology; uncaptured ports never acquire tensor records',
    () => {
      const h = harness();
      const payload = fixture();
      const model = h.data.buildExecutionGraphs(payload);
      assert.equal(model.recordNodes.size, 2);
      for (const pane of model.panes) {
        assert.equal(pane.graph.nodes.length, 4);
        const outputs = [...model.lookup.values()].filter(
            hit => hit.side === pane.side && hit.kind === 'tensor');
        assert.equal(outputs.length, 2);
        assert.equal(outputs.find(hit => hit.outputId === '1').recordId, null);
        const consumer =
            pane.graph.nodes.find(node => node.label === 'Consume');
        assert.equal(
            consumer.incomingEdges[0].sourceNodeId,
            outputs.find(hit => hit.outputId === '1').rendererNodeId);
        assert.equal(
            [
              ...model.lookup.values()
            ].filter(hit => hit.nodeId === 'consumer' && hit.kind === 'tensor')
                .length,
            0);
      }
      const noRecords = h.data.buildExecutionGraphs(
          {...payload, details: {...payload.details, tensors: []}});
      assert.equal(noRecords.recordNodes.size, 0);
      assert.equal(noRecords.graphCollections[0].graphs[0].nodes.length, 4);
    });

test(
    'selected real records resolve run roles independently of executions ordering',
    () => {
      const h = harness();
      const payload = fixture();
      payload.details.executions.reverse();
      const model = h.data.buildExecutionGraphs(payload);
      assert.equal(model.panes[0].run.id, 'run-r');
      assert.equal(model.panes[1].run.id, 'run-t');
      const onlyTarget =
          h.data.buildExecutionGraphs({...payload, reference: ''});
      assert.equal(onlyTarget.panes[0].run.id, 'run-r');
      assert.equal(onlyTarget.panes[1].run.id, 'run-t');
    });

test(
    'metrics apply only to selected captured pair; operations and uncaptured tensors stay without metrics',
    () => {
      const h = harness();
      const payload = fixture();
      const model = h.data.buildExecutionGraphs(payload);
      const data = h.data.buildNodeData(model, payload);
      for (const pane of model.panes) {
        const results = data[pane.graph.id].results;
        for (const node of pane.graph.nodes) {
          const hit = model.lookup.get(node.id);
          assert.equal(
              results[node.id].value,
              hit.kind === 'operation' ? '' :
                  hit.recordId         ? .01 :
                                         '—');
        }
      }
      const missing =
          h.data.buildNodeData(model, {...payload, metrics: {mse: null}});
      assert.equal(
          missing[model.panes[0].graph.id]
              .results[model.recordNodes.get('r0').rendererNodeId]
              .value,
          '—');
    });

test(
    'metric updates preserve mount, camera and paired highlighting; blank canvas preserves inspection',
    async () => {
      const h = harness();
      const payload = fixture();
      await h.render(payload);
      const element = h.load();
      const selection = [...element.selected];
      element.calls = [];
      await h.render({...payload, metrics: {mse: .5}});
      h.flush();
      assert.equal(h.mounts.length, 1);
      assert.deepEqual(element.selected, selection);
      assert(element.calls.every(call => call[0] === 'data'));
      element.selected[0] = '';
      element.emit('selectedNodeChanged', {detail: {nodeId: ''}});
      h.flush();
      assert.deepEqual(element.selected, selection);
      assert.equal(
          h.messages.filter(message => message.type === 'tensorSelected')
              .length,
          0);
    });

test(
    'mapping allows only real tensor picks; ordinary unrecorded outputs inspect their producer operation',
    async () => {
      const h = harness();
      const payload = fixture();
      await h.render({...payload, mapping: true});
      const element = h.load();
      const model = h.data.buildExecutionGraphs(payload);
      const hits =
          [...model.lookup.values()].filter(hit => hit.side === 'target');
      for (const hit of hits.filter(hit => !hit.recordId)) {
        element.emit(
            'selectedNodeChanged', {detail: {nodeId: hit.rendererNodeId}});
        h.flush();
      }
      assert.equal(
          h.messages.filter(message => /Selected$/.test(message.type)).length,
          0);
      element.emit(
          'selectedNodeChanged',
          {detail: {nodeId: model.recordNodes.get('t0').rendererNodeId}});
      h.flush();
      assert.equal(
          h.messages.filter(message => message.type === 'tensorSelected')
              .length,
          1);
      await h.render(payload);
      h.flush();
      const output = hits.find(hit => hit.kind === 'tensor' && !hit.recordId);
      element.emit(
          'selectedNodeChanged', {detail: {nodeId: output.rendererNodeId}});
      h.flush();
      const event =
          h.messages.find(message => message.type === 'operationSelected');
      assert.equal(event.hit.nodeId, 'multi');
    });

test(
    'obsolete first load after close is ignored and reopen initializes a fresh live renderer',
    async () => {
      const h = harness();
      const payload = fixture();
      await h.render(payload);
      const old = h.mounts[0];
      await h.render({details: null});
      old.emit('modelGraphProcessed', {detail: {paneIndex: 0}});
      old.emit('modelGraphProcessed', {detail: {paneIndex: 1}});
      h.flush();
      assert.equal(old.calls.length, 0);
      await h.render(payload);
      const element = h.load();
      assert.equal(h.mounts.length, 2);
      assert.ok(element.calls.some(call => call[0] === 'focus'));
      assert.ok(element.selected.every(Boolean));
    });

test(
    'cancel mapping restores captured viewport without remount or automatic reveal',
    async () => {
      const h = harness();
      const payload = fixture();
      await h.render(payload);
      const element = h.load();
      await h.render({...payload, mapping: true});
      h.flush();
      element.calls = [];
      await h.render(payload);
      h.flush();
      assert.equal(h.mounts.length, 1);
      assert.equal(
          element.calls.filter(call => call[0] === 'restore').length, 2);
      assert.equal(element.calls.filter(call => call[0] === 'focus').length, 0);
    });

test(
    'theme changes repaint the frame and nodes without remounting or moving inspection',
    async () => {
      const h = harness();
      const payload = fixture();
      await h.render({...payload, theme: 'dark'});
      const element = h.load();
      assert.equal(h.document.documentElement.dataset.theme, 'dark');
      assert.ok(element.calls.some(
          call => call[0] === 'background' && call[1] === '#202124'));
      const selected = [...element.selected];
      const model = h.data.buildExecutionGraphs(payload);
      const darkData = h.data.buildNodeData(model, {...payload, theme: 'dark'});
      const lightData =
          h.data.buildNodeData(model, {...payload, theme: 'light'});
      const graphId = model.panes[0].graph.id;
      const nodeId = model.recordNodes.get('r0').rendererNodeId;
      assert.equal(
          darkData[graphId].results[nodeId].value,
          lightData[graphId].results[nodeId].value);
      assert.notEqual(
          darkData[graphId].results[nodeId].bgColor,
          lightData[graphId].results[nodeId].bgColor);
      element.calls = [];
      await h.render({...payload, theme: 'light'});
      h.flush();
      assert.equal(h.document.documentElement.dataset.theme, 'light');
      assert.equal(h.mounts.length, 1);
      assert.deepEqual(element.selected, selected);
      assert.equal(
          element.calls
              .filter(
                  call => ['focus', 'restore', 'fit', 'select'].includes(
                      call[0]))
              .length,
          0);
      assert.equal(
          element.calls
              .filter(call => call[0] === 'background' && call[1] === '#ffffff')
              .length,
          2);
      assert.equal(element.calls.filter(call => call[0] === 'data').length, 2);
    });


test(
    'a pre-upgrade theme bridge keeps execution usable with a visible reload action',
    async () => {
      const h = harness();
      await h.render({...fixture(), theme: 'dark'});
      h.mounts.at(-1).setPaneBackground = undefined;
      const element = h.load();
      assert.equal(h.get('host').dataset.viewportReady, 'true');
      assert.ok(element.selected.every(Boolean));
      assert.match(h.get('status').textContent, /theme support is outdated/);
      assert.doesNotMatch(h.get('status').textContent, /Loading/);
      assert.equal(h.get('retry').hidden, false);
      h.get('retry').onclick();
      assert.equal(h.reloads(), 1);
    });

test(
    'theme application errors remain recoverable without stalling graph initialization',
    async () => {
      const h = harness();
      await h.render({...fixture(), theme: 'dark'});
      h.mounts.at(-1).setPaneBackground = () => {
        throw new Error('theme failure');
      };
      const element = h.load();
      assert.ok(element.selected.every(Boolean));
      assert.equal(h.get('host').dataset.viewportReady, 'true');
      assert.match(h.get('status').textContent, /theme could not be applied/);
      assert.equal(h.get('retry').hidden, false);
    });

test(
    'first processed-event initialization errors show a retry state instead of Loading',
    async () => {
      const h = harness();
      await h.render(fixture());
      const element = h.mounts.at(-1);
      element.addNodeDataProviderDataWithGraphIndex = () => {
        throw new Error('provider failed');
      };
      for (const paneIndex of [0, 1]) {
        element.emit('modelGraphProcessed', {detail: {paneIndex}});
      }
      assert.match(
          h.get('status').textContent, /Unable to initialize execution graph/);
      assert.equal(h.get('retry').hidden, false);
      assert.equal(h.errors.length, 1);
      h.errors.length = 0;
      h.flush();
      element.emit('modelGraphProcessed', {detail: {paneIndex: 1}});
      h.flush();
      assert.match(
          h.get('status').textContent, /Unable to initialize execution graph/);
    });

test('first viewport initialization errors show a retry state', async () => {
  const h = harness();
  await h.render(fixture());
  const element = h.mounts.at(-1);
  element.focusNodeViewport = () => {
    throw new Error('viewport failed');
  };
  for (const paneIndex of [0, 1]) {
    element.emit('modelGraphProcessed', {detail: {paneIndex}});
  }
  h.flush(1);  // The initialization error is logged once and converted into the
               // visible retry state.
  assert.equal(h.errors.length, 1);
  assert.match(
      h.get('status').textContent, /Unable to initialize execution graph/);
  assert.equal(h.get('retry').hidden, false);
  h.errors.length = 0;
  h.flush();
});

test(
    'CosSim severity is metric-name case-insensitive: a perfect match is not highlighted',
    () => {
      const h = harness();
      const payload = {...fixture(), metric: 'CosSim', metrics: {CosSim: 1}};
      const model = h.data.buildExecutionGraphs(payload);
      const data = h.data.buildNodeData(model, payload);
      const node = model.recordNodes.get('r0').rendererNodeId;
      const pane = model.panes[0].graph.id;
      assert.equal(data[pane].results[node].value, 1);
      assert.equal(data[pane].results[node].bgColor, '#ffffff');
      const drift =
          h.data.buildNodeData(model, {...payload, metrics: {CosSim: .5}});
      assert.equal(drift[pane].results[node].bgColor, '#f7cfcd');
    });

test(
    'stale render while waiting for customElements.whenDefined does not mount an obsolete visualizer',
    async () => {
      let resolveDefined;
      const definedPromise = new Promise(resolve => {
        resolveDefined = resolve;
      });
      const h = harness({whenDefined: () => definedPromise});
      const pendingRender = h.render(fixture());
      assert.deepEqual(h.whenDefinedCalls, ['model-explorer-visualizer']);
      assert.equal(h.mounts.length, 0);
      await h.render({details: null});
      resolveDefined();
      await pendingRender;
      await Promise.resolve();
      assert.equal(h.mounts.length, 0);
    });

test(
    'uiStateChanged preserves explicit zero widthFraction using nullish coalescing',
    async () => {
      const h = harness();
      await h.render(fixture());
      const element = h.mounts.at(-1);
      element.emit(
          'uiStateChanged', {detail: {paneStates: [{widthFraction: 0}]}});
      assert.equal(h.get('panes').style.gridTemplateColumns, '0fr 1fr');
    });

test(
    'graph selector onchange routes async render rejections to initializationFailed',
    async () => {
      let failWhenDefined = false;
      const h = harness({
        whenDefined: () => failWhenDefined ?
            Promise.reject(new Error('whenDefined failed')) :
            Promise.resolve(),
      });
      const data = fixture();
      data.details.executions[0].graphs.push({
        id: 'alt_graph',
        name: 'alt_graph',
        nodes:
            [{id: 'alt_op', label: 'alt_op', namespace: 'model', outputs: []}]
      });
      await h.render(data);
      const select = h.get('panes').children[0].children.find(
          child => child.tagName === 'SELECT');
      assert.ok(select);
      failWhenDefined = true;
      select.value = 'alt_graph';
      select.onchange();
      await new Promise(setImmediate);
      assert.match(
          h.get('status').textContent, /Unable to initialize execution graph/);
      assert.equal(h.get('retry').hidden, false);
      assert.equal(h.errors.length, 1);
    });
