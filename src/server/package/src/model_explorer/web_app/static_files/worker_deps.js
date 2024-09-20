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

(function(f){if(typeof exports==="object"&&typeof module!=="undefined"){module.exports=f()}else if(typeof define==="function"&&define.amd){define([],f)}else{var g;if(typeof window!=="undefined"){g=window}else if(typeof global!=="undefined"){g=global}else if(typeof self!=="undefined"){g=self}else{g=this}g.dagre = f()}})(function(){var define,module,exports;return (function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){

  module.exports = {
    graphlib: require("@dagrejs/graphlib"),
  
    layout: require("./lib/layout"),
    debug: require("./lib/debug"),
    util: {
      time: require("./lib/util").time,
      notime: require("./lib/util").notime
    },
    version: require("./lib/version")
  };
  
  },{"./lib/debug":6,"./lib/layout":8,"./lib/util":27,"./lib/version":28,"@dagrejs/graphlib":29}],2:[function(require,module,exports){
  "use strict";
  
  let greedyFAS = require("./greedy-fas");
  let uniqueId = require("./util").uniqueId;
  
  module.exports = {
    run: run,
    undo: undo
  };
  
  function run(g) {
    let fas = (g.graph().acyclicer === "greedy"
      ? greedyFAS(g, weightFn(g))
      : dfsFAS(g));
    fas.forEach(e => {
      let label = g.edge(e);
      g.removeEdge(e);
      label.forwardName = e.name;
      label.reversed = true;
      g.setEdge(e.w, e.v, label, uniqueId("rev"));
    });
  
    function weightFn(g) {
      return e => {
        return g.edge(e).weight;
      };
    }
  }
  
  function dfsFAS(g) {
    let fas = [];
    let stack = {};
    let visited = {};
  
    function dfs(v) {
      if (visited.hasOwnProperty(v)) {
        return;
      }
      visited[v] = true;
      stack[v] = true;
      g.outEdges(v).forEach(e => {
        if (stack.hasOwnProperty(e.w)) {
          fas.push(e);
        } else {
          dfs(e.w);
        }
      });
      delete stack[v];
    }
  
    g.nodes().forEach(dfs);
    return fas;
  }
  
  function undo(g) {
    g.edges().forEach(e => {
      let label = g.edge(e);
      if (label.reversed) {
        g.removeEdge(e);
  
        let forwardName = label.forwardName;
        delete label.reversed;
        delete label.forwardName;
        g.setEdge(e.w, e.v, label, forwardName);
      }
    });
  }
  
  },{"./greedy-fas":7,"./util":27}],3:[function(require,module,exports){
  let util = require("./util");
  
  module.exports = addBorderSegments;
  
  function addBorderSegments(g) {
    function dfs(v) {
      let children = g.children(v);
      let node = g.node(v);
      if (children.length) {
        children.forEach(dfs);
      }
  
      if (node.hasOwnProperty("minRank")) {
        node.borderLeft = [];
        node.borderRight = [];
        for (let rank = node.minRank, maxRank = node.maxRank + 1;
          rank < maxRank;
          ++rank) {
          addBorderNode(g, "borderLeft", "_bl", v, node, rank);
          addBorderNode(g, "borderRight", "_br", v, node, rank);
        }
      }
    }
  
    g.children().forEach(dfs);
  }
  
  function addBorderNode(g, prop, prefix, sg, sgNode, rank) {
    let label = { width: 0, height: 0, rank: rank, borderType: prop };
    let prev = sgNode[prop][rank - 1];
    let curr = util.addDummyNode(g, "border", label, prefix);
    sgNode[prop][rank] = curr;
    g.setParent(curr, sg);
    if (prev) {
      g.setEdge(prev, curr, { weight: 1 });
    }
  }
  
  },{"./util":27}],4:[function(require,module,exports){
  "use strict";
  
  module.exports = {
    adjust: adjust,
    undo: undo
  };
  
  function adjust(g) {
    let rankDir = g.graph().rankdir.toLowerCase();
    if (rankDir === "lr" || rankDir === "rl") {
      swapWidthHeight(g);
    }
  }
  
  function undo(g) {
    let rankDir = g.graph().rankdir.toLowerCase();
    if (rankDir === "bt" || rankDir === "rl") {
      reverseY(g);
    }
  
    if (rankDir === "lr" || rankDir === "rl") {
      swapXY(g);
      swapWidthHeight(g);
    }
  }
  
  function swapWidthHeight(g) {
    g.nodes().forEach(v => swapWidthHeightOne(g.node(v)));
    g.edges().forEach(e => swapWidthHeightOne(g.edge(e)));
  }
  
  function swapWidthHeightOne(attrs) {
    let w = attrs.width;
    attrs.width = attrs.height;
    attrs.height = w;
  }
  
  function reverseY(g) {
    g.nodes().forEach(v => reverseYOne(g.node(v)));
  
    g.edges().forEach(e => {
      let edge = g.edge(e);
      edge.points.forEach(reverseYOne);
      if (edge.hasOwnProperty("y")) {
        reverseYOne(edge);
      }
    });
  }
  
  function reverseYOne(attrs) {
    attrs.y = -attrs.y;
  }
  
  function swapXY(g) {
    g.nodes().forEach(v => swapXYOne(g.node(v)));
  
    g.edges().forEach(e => {
      let edge = g.edge(e);
      edge.points.forEach(swapXYOne);
      if (edge.hasOwnProperty("x")) {
        swapXYOne(edge);
      }
    });
  }
  
  function swapXYOne(attrs) {
    let x = attrs.x;
    attrs.x = attrs.y;
    attrs.y = x;
  }
  
  },{}],5:[function(require,module,exports){
  /*
   * Simple doubly linked list implementation derived from Cormen, et al.,
   * "Introduction to Algorithms".
   */
  
  class List {
    constructor() {
      let sentinel = {};
      sentinel._next = sentinel._prev = sentinel;
      this._sentinel = sentinel;
    }
  
    dequeue() {
      let sentinel = this._sentinel;
      let entry = sentinel._prev;
      if (entry !== sentinel) {
        unlink(entry);
        return entry;
      }
    }
  
    enqueue(entry) {
      let sentinel = this._sentinel;
      if (entry._prev && entry._next) {
        unlink(entry);
      }
      entry._next = sentinel._next;
      sentinel._next._prev = entry;
      sentinel._next = entry;
      entry._prev = sentinel;
    }
  
    toString() {
      let strs = [];
      let sentinel = this._sentinel;
      let curr = sentinel._prev;
      while (curr !== sentinel) {
        strs.push(JSON.stringify(curr, filterOutLinks));
        curr = curr._prev;
      }
      return "[" + strs.join(", ") + "]";
    }
  }
  
  function unlink(entry) {
    entry._prev._next = entry._next;
    entry._next._prev = entry._prev;
    delete entry._next;
    delete entry._prev;
  }
  
  function filterOutLinks(k, v) {
    if (k !== "_next" && k !== "_prev") {
      return v;
    }
  }
  
  module.exports = List;
  
  },{}],6:[function(require,module,exports){
  let util = require("./util");
  let Graph = require("@dagrejs/graphlib").Graph;
  
  module.exports = {
    debugOrdering: debugOrdering
  };
  
  /* istanbul ignore next */
  function debugOrdering(g) {
    let layerMatrix = util.buildLayerMatrix(g);
  
    let h = new Graph({ compound: true, multigraph: true }).setGraph({});
  
    g.nodes().forEach(v => {
      h.setNode(v, { label: v });
      h.setParent(v, "layer" + g.node(v).rank);
    });
  
    g.edges().forEach(e => h.setEdge(e.v, e.w, {}, e.name));
  
    layerMatrix.forEach((layer, i) => {
      let layerV = "layer" + i;
      h.setNode(layerV, { rank: "same" });
      layer.reduce((u, v) => {
        h.setEdge(u, v, { style: "invis" });
        return v;
      });
    });
  
    return h;
  }
  
  },{"./util":27,"@dagrejs/graphlib":29}],7:[function(require,module,exports){
  let Graph = require("@dagrejs/graphlib").Graph;
  let List = require("./data/list");
  
  /*
   * A greedy heuristic for finding a feedback arc set for a graph. A feedback
   * arc set is a set of edges that can be removed to make a graph acyclic.
   * The algorithm comes from: P. Eades, X. Lin, and W. F. Smyth, "A fast and
   * effective heuristic for the feedback arc set problem." This implementation
   * adjusts that from the paper to allow for weighted edges.
   */
  module.exports = greedyFAS;
  
  let DEFAULT_WEIGHT_FN = () => 1;
  
  function greedyFAS(g, weightFn) {
    if (g.nodeCount() <= 1) {
      return [];
    }
    let state = buildState(g, weightFn || DEFAULT_WEIGHT_FN);
    let results = doGreedyFAS(state.graph, state.buckets, state.zeroIdx);
  
    // Expand multi-edges
    return results.flatMap(e => g.outEdges(e.v, e.w));
  }
  
  function doGreedyFAS(g, buckets, zeroIdx) {
    let results = [];
    let sources = buckets[buckets.length - 1];
    let sinks = buckets[0];
  
    let entry;
    while (g.nodeCount()) {
      while ((entry = sinks.dequeue()))   { removeNode(g, buckets, zeroIdx, entry); }
      while ((entry = sources.dequeue())) { removeNode(g, buckets, zeroIdx, entry); }
      if (g.nodeCount()) {
        for (let i = buckets.length - 2; i > 0; --i) {
          entry = buckets[i].dequeue();
          if (entry) {
            results = results.concat(removeNode(g, buckets, zeroIdx, entry, true));
            break;
          }
        }
      }
    }
  
    return results;
  }
  
  function removeNode(g, buckets, zeroIdx, entry, collectPredecessors) {
    let results = collectPredecessors ? [] : undefined;
  
    g.inEdges(entry.v).forEach(edge => {
      let weight = g.edge(edge);
      let uEntry = g.node(edge.v);
  
      if (collectPredecessors) {
        results.push({ v: edge.v, w: edge.w });
      }
  
      uEntry.out -= weight;
      assignBucket(buckets, zeroIdx, uEntry);
    });
  
    g.outEdges(entry.v).forEach(edge => {
      let weight = g.edge(edge);
      let w = edge.w;
      let wEntry = g.node(w);
      wEntry["in"] -= weight;
      assignBucket(buckets, zeroIdx, wEntry);
    });
  
    g.removeNode(entry.v);
  
    return results;
  }
  
  function buildState(g, weightFn) {
    let fasGraph = new Graph();
    let maxIn = 0;
    let maxOut = 0;
  
    g.nodes().forEach(v => {
      fasGraph.setNode(v, { v: v, "in": 0, out: 0 });
    });
  
    // Aggregate weights on nodes, but also sum the weights across multi-edges
    // into a single edge for the fasGraph.
    g.edges().forEach(e => {
      let prevWeight = fasGraph.edge(e.v, e.w) || 0;
      let weight = weightFn(e);
      let edgeWeight = prevWeight + weight;
      fasGraph.setEdge(e.v, e.w, edgeWeight);
      maxOut = Math.max(maxOut, fasGraph.node(e.v).out += weight);
      maxIn  = Math.max(maxIn,  fasGraph.node(e.w)["in"]  += weight);
    });
  
    let buckets = range(maxOut + maxIn + 3).map(() => new List());
    let zeroIdx = maxIn + 1;
  
    fasGraph.nodes().forEach(v => {
      assignBucket(buckets, zeroIdx, fasGraph.node(v));
    });
  
    return { graph: fasGraph, buckets: buckets, zeroIdx: zeroIdx };
  }
  
  function assignBucket(buckets, zeroIdx, entry) {
    if (!entry.out) {
      buckets[0].enqueue(entry);
    } else if (!entry["in"]) {
      buckets[buckets.length - 1].enqueue(entry);
    } else {
      buckets[entry.out - entry["in"] + zeroIdx].enqueue(entry);
    }
  }
  
  function range(limit) {
    const range = [];
    for (let i = 0; i < limit; i++) {
      range.push(i);
    }
  
    return range;
  }
  
  },{"./data/list":5,"@dagrejs/graphlib":29}],8:[function(require,module,exports){
  "use strict";
  
  let acyclic = require("./acyclic");
  let normalize = require("./normalize");
  let rank = require("./rank");
  let normalizeRanks = require("./util").normalizeRanks;
  let parentDummyChains = require("./parent-dummy-chains");
  let removeEmptyRanks = require("./util").removeEmptyRanks;
  let nestingGraph = require("./nesting-graph");
  let addBorderSegments = require("./add-border-segments");
  let coordinateSystem = require("./coordinate-system");
  let order = require("./order");
  let position = require("./position");
  let util = require("./util");
  let Graph = require("@dagrejs/graphlib").Graph;
  
  module.exports = layout;
  
  function layout(g, opts) {
    let time = opts && opts.debugTiming ? util.time : util.notime;
    time("layout", () => {
      let layoutGraph =
        time("  buildLayoutGraph", () => buildLayoutGraph(g));
      time("  runLayout",        () => runLayout(layoutGraph, time));
      time("  updateInputGraph", () => updateInputGraph(g, layoutGraph));
    });
  }
  
  function runLayout(g, time) {
    time("    makeSpaceForEdgeLabels", () => makeSpaceForEdgeLabels(g));
    time("    removeSelfEdges",        () => removeSelfEdges(g));
    time("    acyclic",                () => acyclic.run(g));
    time("    nestingGraph.run",       () => nestingGraph.run(g));
    time("    rank",                   () => rank(util.asNonCompoundGraph(g)));
    time("    injectEdgeLabelProxies", () => injectEdgeLabelProxies(g));
    time("    removeEmptyRanks",       () => removeEmptyRanks(g));
    time("    nestingGraph.cleanup",   () => nestingGraph.cleanup(g));
    time("    normalizeRanks",         () => normalizeRanks(g));
    time("    assignRankMinMax",       () => assignRankMinMax(g));
    time("    removeEdgeLabelProxies", () => removeEdgeLabelProxies(g));
    time("    normalize.run",          () => normalize.run(g));
    time("    parentDummyChains",      () => parentDummyChains(g));
    time("    addBorderSegments",      () => addBorderSegments(g));
    time("    order",                  () => order(g));
    time("    insertSelfEdges",        () => insertSelfEdges(g));
    time("    adjustCoordinateSystem", () => coordinateSystem.adjust(g));
    time("    position",               () => position(g));
    time("    positionSelfEdges",      () => positionSelfEdges(g));
    time("    removeBorderNodes",      () => removeBorderNodes(g));
    time("    normalize.undo",         () => normalize.undo(g));
    time("    fixupEdgeLabelCoords",   () => fixupEdgeLabelCoords(g));
    time("    undoCoordinateSystem",   () => coordinateSystem.undo(g));
    time("    translateGraph",         () => translateGraph(g));
    time("    assignNodeIntersects",   () => assignNodeIntersects(g));
    time("    reversePoints",          () => reversePointsForReversedEdges(g));
    time("    acyclic.undo",           () => acyclic.undo(g));
  }
  
  /*
   * Copies final layout information from the layout graph back to the input
   * graph. This process only copies whitelisted attributes from the layout graph
   * to the input graph, so it serves as a good place to determine what
   * attributes can influence layout.
   */
  function updateInputGraph(inputGraph, layoutGraph) {
    inputGraph.nodes().forEach(v => {
      let inputLabel = inputGraph.node(v);
      let layoutLabel = layoutGraph.node(v);
  
      if (inputLabel) {
        inputLabel.x = layoutLabel.x;
        inputLabel.y = layoutLabel.y;
        inputLabel.rank = layoutLabel.rank;
  
        if (layoutGraph.children(v).length) {
          inputLabel.width = layoutLabel.width;
          inputLabel.height = layoutLabel.height;
        }
      }
    });
  
    inputGraph.edges().forEach(e => {
      let inputLabel = inputGraph.edge(e);
      let layoutLabel = layoutGraph.edge(e);
  
      inputLabel.points = layoutLabel.points;
      if (layoutLabel.hasOwnProperty("x")) {
        inputLabel.x = layoutLabel.x;
        inputLabel.y = layoutLabel.y;
      }
    });
  
    inputGraph.graph().width = layoutGraph.graph().width;
    inputGraph.graph().height = layoutGraph.graph().height;
  }
  
  let graphNumAttrs = ["nodesep", "edgesep", "ranksep", "marginx", "marginy"];
  let graphDefaults = { ranksep: 50, edgesep: 20, nodesep: 50, rankdir: "tb" };
  let graphAttrs = ["acyclicer", "ranker", "rankdir", "align"];
  let nodeNumAttrs = ["width", "height"];
  let nodeDefaults = { width: 0, height: 0 };
  let edgeNumAttrs = ["minlen", "weight", "width", "height", "labeloffset"];
  let edgeDefaults = {
    minlen: 1, weight: 1, width: 0, height: 0,
    labeloffset: 10, labelpos: "r"
  };
  let edgeAttrs = ["labelpos"];
  
  /*
   * Constructs a new graph from the input graph, which can be used for layout.
   * This process copies only whitelisted attributes from the input graph to the
   * layout graph. Thus this function serves as a good place to determine what
   * attributes can influence layout.
   */
  function buildLayoutGraph(inputGraph) {
    let g = new Graph({ multigraph: true, compound: true });
    let graph = canonicalize(inputGraph.graph());
  
    g.setGraph(Object.assign({},
      graphDefaults,
      selectNumberAttrs(graph, graphNumAttrs),
      util.pick(graph, graphAttrs)));
  
    inputGraph.nodes().forEach(v => {
      let node = canonicalize(inputGraph.node(v));
      const newNode = selectNumberAttrs(node, nodeNumAttrs);
      Object.keys(nodeDefaults).forEach(k => {
        if (newNode[k] === undefined) {
          newNode[k] = nodeDefaults[k];
        }
      });
  
      g.setNode(v, newNode);
      g.setParent(v, inputGraph.parent(v));
    });
  
    inputGraph.edges().forEach(e => {
      let edge = canonicalize(inputGraph.edge(e));
      g.setEdge(e, Object.assign({},
        edgeDefaults,
        selectNumberAttrs(edge, edgeNumAttrs),
        util.pick(edge, edgeAttrs)));
    });
  
    return g;
  }
  
  /*
   * This idea comes from the Gansner paper: to account for edge labels in our
   * layout we split each rank in half by doubling minlen and halving ranksep.
   * Then we can place labels at these mid-points between nodes.
   *
   * We also add some minimal padding to the width to push the label for the edge
   * away from the edge itself a bit.
   */
  function makeSpaceForEdgeLabels(g) {
    let graph = g.graph();
    graph.ranksep /= 2;
    g.edges().forEach(e => {
      let edge = g.edge(e);
      edge.minlen *= 2;
      if (edge.labelpos.toLowerCase() !== "c") {
        if (graph.rankdir === "TB" || graph.rankdir === "BT") {
          edge.width += edge.labeloffset;
        } else {
          edge.height += edge.labeloffset;
        }
      }
    });
  }
  
  /*
   * Creates temporary dummy nodes that capture the rank in which each edge's
   * label is going to, if it has one of non-zero width and height. We do this
   * so that we can safely remove empty ranks while preserving balance for the
   * label's position.
   */
  function injectEdgeLabelProxies(g) {
    g.edges().forEach(e => {
      let edge = g.edge(e);
      if (edge.width && edge.height) {
        let v = g.node(e.v);
        let w = g.node(e.w);
        let label = { rank: (w.rank - v.rank) / 2 + v.rank, e: e };
        util.addDummyNode(g, "edge-proxy", label, "_ep");
      }
    });
  }
  
  function assignRankMinMax(g) {
    let maxRank = 0;
    g.nodes().forEach(v => {
      let node = g.node(v);
      if (node.borderTop) {
        node.minRank = g.node(node.borderTop).rank;
        node.maxRank = g.node(node.borderBottom).rank;
        maxRank = Math.max(maxRank, node.maxRank);
      }
    });
    g.graph().maxRank = maxRank;
  }
  
  function removeEdgeLabelProxies(g) {
    g.nodes().forEach(v => {
      let node = g.node(v);
      if (node.dummy === "edge-proxy") {
        g.edge(node.e).labelRank = node.rank;
        g.removeNode(v);
      }
    });
  }
  
  function translateGraph(g) {
    let minX = Number.POSITIVE_INFINITY;
    let maxX = 0;
    let minY = Number.POSITIVE_INFINITY;
    let maxY = 0;
    let graphLabel = g.graph();
    let marginX = graphLabel.marginx || 0;
    let marginY = graphLabel.marginy || 0;
  
    function getExtremes(attrs) {
      let x = attrs.x;
      let y = attrs.y;
      let w = attrs.width;
      let h = attrs.height;
      minX = Math.min(minX, x - w / 2);
      maxX = Math.max(maxX, x + w / 2);
      minY = Math.min(minY, y - h / 2);
      maxY = Math.max(maxY, y + h / 2);
    }
  
    g.nodes().forEach(v => getExtremes(g.node(v)));
    g.edges().forEach(e => {
      let edge = g.edge(e);
      if (edge.hasOwnProperty("x")) {
        getExtremes(edge);
      }
    });
  
    minX -= marginX;
    minY -= marginY;
  
    g.nodes().forEach(v => {
      let node = g.node(v);
      node.x -= minX;
      node.y -= minY;
    });
  
    g.edges().forEach(e => {
      let edge = g.edge(e);
      edge.points.forEach(p => {
        p.x -= minX;
        p.y -= minY;
      });
      if (edge.hasOwnProperty("x")) { edge.x -= minX; }
      if (edge.hasOwnProperty("y")) { edge.y -= minY; }
    });
  
    graphLabel.width = maxX - minX + marginX;
    graphLabel.height = maxY - minY + marginY;
  }
  
  function assignNodeIntersects(g) {
    g.edges().forEach(e => {
      let edge = g.edge(e);
      let nodeV = g.node(e.v);
      let nodeW = g.node(e.w);
      let p1, p2;
      if (!edge.points) {
        edge.points = [];
        p1 = nodeW;
        p2 = nodeV;
      } else {
        p1 = edge.points[0];
        p2 = edge.points[edge.points.length - 1];
      }
      edge.points.unshift(util.intersectRect(nodeV, p1));
      edge.points.push(util.intersectRect(nodeW, p2));
    });
  }
  
  function fixupEdgeLabelCoords(g) {
    g.edges().forEach(e => {
      let edge = g.edge(e);
      if (edge.hasOwnProperty("x")) {
        if (edge.labelpos === "l" || edge.labelpos === "r") {
          edge.width -= edge.labeloffset;
        }
        switch (edge.labelpos) {
        case "l": edge.x -= edge.width / 2 + edge.labeloffset; break;
        case "r": edge.x += edge.width / 2 + edge.labeloffset; break;
        }
      }
    });
  }
  
  function reversePointsForReversedEdges(g) {
    g.edges().forEach(e => {
      let edge = g.edge(e);
      if (edge.reversed) {
        edge.points.reverse();
      }
    });
  }
  
  function removeBorderNodes(g) {
    g.nodes().forEach(v => {
      if (g.children(v).length) {
        let node = g.node(v);
        let t = g.node(node.borderTop);
        let b = g.node(node.borderBottom);
        let l = g.node(node.borderLeft[node.borderLeft.length - 1]);
        let r = g.node(node.borderRight[node.borderRight.length - 1]);
  
        node.width = Math.abs(r.x - l.x);
        node.height = Math.abs(b.y - t.y);
        node.x = l.x + node.width / 2;
        node.y = t.y + node.height / 2;
      }
    });
  
    g.nodes().forEach(v => {
      if (g.node(v).dummy === "border") {
        g.removeNode(v);
      }
    });
  }
  
  function removeSelfEdges(g) {
    g.edges().forEach(e => {
      if (e.v === e.w) {
        var node = g.node(e.v);
        if (!node.selfEdges) {
          node.selfEdges = [];
        }
        node.selfEdges.push({ e: e, label: g.edge(e) });
        g.removeEdge(e);
      }
    });
  }
  
  function insertSelfEdges(g) {
    var layers = util.buildLayerMatrix(g);
    layers.forEach(layer => {
      var orderShift = 0;
      layer.forEach((v, i) => {
        var node = g.node(v);
        node.order = i + orderShift;
        (node.selfEdges || []).forEach(selfEdge => {
          util.addDummyNode(g, "selfedge", {
            width: selfEdge.label.width,
            height: selfEdge.label.height,
            rank: node.rank,
            order: i + (++orderShift),
            e: selfEdge.e,
            label: selfEdge.label
          }, "_se");
        });
        delete node.selfEdges;
      });
    });
  }
  
  function positionSelfEdges(g) {
    g.nodes().forEach(v => {
      var node = g.node(v);
      if (node.dummy === "selfedge") {
        var selfNode = g.node(node.e.v);
        var x = selfNode.x + selfNode.width / 2;
        var y = selfNode.y;
        var dx = node.x - x;
        var dy = selfNode.height / 2;
        g.setEdge(node.e, node.label);
        g.removeNode(v);
        node.label.points = [
          { x: x + 2 * dx / 3, y: y - dy },
          { x: x + 5 * dx / 6, y: y - dy },
          { x: x +     dx    , y: y },
          { x: x + 5 * dx / 6, y: y + dy },
          { x: x + 2 * dx / 3, y: y + dy }
        ];
        node.label.x = node.x;
        node.label.y = node.y;
      }
    });
  }
  
  function selectNumberAttrs(obj, attrs) {
    return util.mapValues(util.pick(obj, attrs), Number);
  }
  
  function canonicalize(attrs) {
    var newAttrs = {};
    if (attrs) {
      Object.entries(attrs).forEach(([k, v]) => {
        if (typeof k === "string") {
          k = k.toLowerCase();
        }
  
        newAttrs[k] = v;
      });
    }
    return newAttrs;
  }
  
  },{"./acyclic":2,"./add-border-segments":3,"./coordinate-system":4,"./nesting-graph":9,"./normalize":10,"./order":15,"./parent-dummy-chains":20,"./position":22,"./rank":24,"./util":27,"@dagrejs/graphlib":29}],9:[function(require,module,exports){
  let util = require("./util");
  
  module.exports = {
    run,
    cleanup,
  };
  
  /*
   * A nesting graph creates dummy nodes for the tops and bottoms of subgraphs,
   * adds appropriate edges to ensure that all cluster nodes are placed between
   * these boundaries, and ensures that the graph is connected.
   *
   * In addition we ensure, through the use of the minlen property, that nodes
   * and subgraph border nodes to not end up on the same rank.
   *
   * Preconditions:
   *
   *    1. Input graph is a DAG
   *    2. Nodes in the input graph has a minlen attribute
   *
   * Postconditions:
   *
   *    1. Input graph is connected.
   *    2. Dummy nodes are added for the tops and bottoms of subgraphs.
   *    3. The minlen attribute for nodes is adjusted to ensure nodes do not
   *       get placed on the same rank as subgraph border nodes.
   *
   * The nesting graph idea comes from Sander, "Layout of Compound Directed
   * Graphs."
   */
  function run(g) {
    let root = util.addDummyNode(g, "root", {}, "_root");
    let depths = treeDepths(g);
    let height = Math.max(...Object.values(depths)) - 1; // Note: depths is an Object not an array
    let nodeSep = 2 * height + 1;
  
    g.graph().nestingRoot = root;
  
    // Multiply minlen by nodeSep to align nodes on non-border ranks.
    g.edges().forEach(e => g.edge(e).minlen *= nodeSep);
  
    // Calculate a weight that is sufficient to keep subgraphs vertically compact
    let weight = sumWeights(g) + 1;
  
    // Create border nodes and link them up
    g.children().forEach(child => dfs(g, root, nodeSep, weight, height, depths, child));
  
    // Save the multiplier for node layers for later removal of empty border
    // layers.
    g.graph().nodeRankFactor = nodeSep;
  }
  
  function dfs(g, root, nodeSep, weight, height, depths, v) {
    let children = g.children(v);
    if (!children.length) {
      if (v !== root) {
        g.setEdge(root, v, { weight: 0, minlen: nodeSep });
      }
      return;
    }
  
    let top = util.addBorderNode(g, "_bt");
    let bottom = util.addBorderNode(g, "_bb");
    let label = g.node(v);
  
    g.setParent(top, v);
    label.borderTop = top;
    g.setParent(bottom, v);
    label.borderBottom = bottom;
  
    children.forEach(child => {
      dfs(g, root, nodeSep, weight, height, depths, child);
  
      let childNode = g.node(child);
      let childTop = childNode.borderTop ? childNode.borderTop : child;
      let childBottom = childNode.borderBottom ? childNode.borderBottom : child;
      let thisWeight = childNode.borderTop ? weight : 2 * weight;
      let minlen = childTop !== childBottom ? 1 : height - depths[v] + 1;
  
      g.setEdge(top, childTop, {
        weight: thisWeight,
        minlen: minlen,
        nestingEdge: true
      });
  
      g.setEdge(childBottom, bottom, {
        weight: thisWeight,
        minlen: minlen,
        nestingEdge: true
      });
    });
  
    if (!g.parent(v)) {
      g.setEdge(root, top, { weight: 0, minlen: height + depths[v] });
    }
  }
  
  function treeDepths(g) {
    var depths = {};
    function dfs(v, depth) {
      var children = g.children(v);
      if (children && children.length) {
        children.forEach(child => dfs(child, depth + 1));
      }
      depths[v] = depth;
    }
    g.children().forEach(v => dfs(v, 1));
    return depths;
  }
  
  function sumWeights(g) {
    return g.edges().reduce((acc, e) => acc + g.edge(e).weight, 0);
  }
  
  function cleanup(g) {
    var graphLabel = g.graph();
    g.removeNode(graphLabel.nestingRoot);
    delete graphLabel.nestingRoot;
    g.edges().forEach(e => {
      var edge = g.edge(e);
      if (edge.nestingEdge) {
        g.removeEdge(e);
      }
    });
  }
  
  },{"./util":27}],10:[function(require,module,exports){
  "use strict";
  
  let util = require("./util");
  
  module.exports = {
    run: run,
    undo: undo
  };
  
  /*
   * Breaks any long edges in the graph into short segments that span 1 layer
   * each. This operation is undoable with the denormalize function.
   *
   * Pre-conditions:
   *
   *    1. The input graph is a DAG.
   *    2. Each node in the graph has a "rank" property.
   *
   * Post-condition:
   *
   *    1. All edges in the graph have a length of 1.
   *    2. Dummy nodes are added where edges have been split into segments.
   *    3. The graph is augmented with a "dummyChains" attribute which contains
   *       the first dummy in each chain of dummy nodes produced.
   */
  function run(g) {
    g.graph().dummyChains = [];
    g.edges().forEach(edge => normalizeEdge(g, edge));
  }
  
  function normalizeEdge(g, e) {
    let v = e.v;
    let vRank = g.node(v).rank;
    let w = e.w;
    let wRank = g.node(w).rank;
    let name = e.name;
    let edgeLabel = g.edge(e);
    let labelRank = edgeLabel.labelRank;
  
    if (wRank === vRank + 1) return;
  
    g.removeEdge(e);
  
    let dummy, attrs, i;
    for (i = 0, ++vRank; vRank < wRank; ++i, ++vRank) {
      edgeLabel.points = [];
      attrs = {
        width: 0, height: 0,
        edgeLabel: edgeLabel, edgeObj: e,
        rank: vRank
      };
      dummy = util.addDummyNode(g, "edge", attrs, "_d");
      if (vRank === labelRank) {
        attrs.width = edgeLabel.width;
        attrs.height = edgeLabel.height;
        attrs.dummy = "edge-label";
        attrs.labelpos = edgeLabel.labelpos;
      }
      g.setEdge(v, dummy, { weight: edgeLabel.weight }, name);
      if (i === 0) {
        g.graph().dummyChains.push(dummy);
      }
      v = dummy;
    }
  
    g.setEdge(v, w, { weight: edgeLabel.weight }, name);
  }
  
  function undo(g) {
    g.graph().dummyChains.forEach(v => {
      let node = g.node(v);
      let origLabel = node.edgeLabel;
      let w;
      g.setEdge(node.edgeObj, origLabel);
      while (node.dummy) {
        w = g.successors(v)[0];
        g.removeNode(v);
        origLabel.points.push({ x: node.x, y: node.y });
        if (node.dummy === "edge-label") {
          origLabel.x = node.x;
          origLabel.y = node.y;
          origLabel.width = node.width;
          origLabel.height = node.height;
        }
        v = w;
        node = g.node(v);
      }
    });
  }
  
  },{"./util":27}],11:[function(require,module,exports){
  module.exports = addSubgraphConstraints;
  
  function addSubgraphConstraints(g, cg, vs) {
    let prev = {},
      rootPrev;
  
    vs.forEach(v => {
      let child = g.parent(v),
        parent,
        prevChild;
      while (child) {
        parent = g.parent(child);
        if (parent) {
          prevChild = prev[parent];
          prev[parent] = child;
        } else {
          prevChild = rootPrev;
          rootPrev = child;
        }
        if (prevChild && prevChild !== child) {
          cg.setEdge(prevChild, child);
          return;
        }
        child = parent;
      }
    });
  
    /*
    function dfs(v) {
      var children = v ? g.children(v) : g.children();
      if (children.length) {
        var min = Number.POSITIVE_INFINITY,
            subgraphs = [];
        children.forEach(function(child) {
          var childMin = dfs(child);
          if (g.children(child).length) {
            subgraphs.push({ v: child, order: childMin });
          }
          min = Math.min(min, childMin);
        });
        _.sortBy(subgraphs, "order").reduce(function(prev, curr) {
          cg.setEdge(prev.v, curr.v);
          return curr;
        });
        return min;
      }
      return g.node(v).order;
    }
    dfs(undefined);
    */
  }
  
  },{}],12:[function(require,module,exports){
  module.exports = barycenter;
  
  function barycenter(g, movable = []) {
    return movable.map(v => {
      let inV = g.inEdges(v);
      if (!inV.length) {
        return { v: v };
      } else {
        let result = inV.reduce((acc, e) => {
          let edge = g.edge(e),
            nodeU = g.node(e.v);
          return {
            sum: acc.sum + (edge.weight * nodeU.order),
            weight: acc.weight + edge.weight
          };
        }, { sum: 0, weight: 0 });
  
        return {
          v: v,
          barycenter: result.sum / result.weight,
          weight: result.weight
        };
      }
    });
  }
  
  
  },{}],13:[function(require,module,exports){
  let Graph = require("@dagrejs/graphlib").Graph;
  let util = require("../util");
  
  module.exports = buildLayerGraph;
  
  /*
   * Constructs a graph that can be used to sort a layer of nodes. The graph will
   * contain all base and subgraph nodes from the request layer in their original
   * hierarchy and any edges that are incident on these nodes and are of the type
   * requested by the "relationship" parameter.
   *
   * Nodes from the requested rank that do not have parents are assigned a root
   * node in the output graph, which is set in the root graph attribute. This
   * makes it easy to walk the hierarchy of movable nodes during ordering.
   *
   * Pre-conditions:
   *
   *    1. Input graph is a DAG
   *    2. Base nodes in the input graph have a rank attribute
   *    3. Subgraph nodes in the input graph has minRank and maxRank attributes
   *    4. Edges have an assigned weight
   *
   * Post-conditions:
   *
   *    1. Output graph has all nodes in the movable rank with preserved
   *       hierarchy.
   *    2. Root nodes in the movable layer are made children of the node
   *       indicated by the root attribute of the graph.
   *    3. Non-movable nodes incident on movable nodes, selected by the
   *       relationship parameter, are included in the graph (without hierarchy).
   *    4. Edges incident on movable nodes, selected by the relationship
   *       parameter, are added to the output graph.
   *    5. The weights for copied edges are aggregated as need, since the output
   *       graph is not a multi-graph.
   */
  function buildLayerGraph(g, rank, relationship) {
    let root = createRootNode(g),
      result = new Graph({ compound: true }).setGraph({ root: root })
        .setDefaultNodeLabel(v => g.node(v));
  
    g.nodes().forEach(v => {
      let node = g.node(v),
        parent = g.parent(v);
  
      if (node.rank === rank || node.minRank <= rank && rank <= node.maxRank) {
        result.setNode(v);
        result.setParent(v, parent || root);
  
        // This assumes we have only short edges!
        g[relationship](v).forEach(e => {
          let u = e.v === v ? e.w : e.v,
            edge = result.edge(u, v),
            weight = edge !== undefined ? edge.weight : 0;
          result.setEdge(u, v, { weight: g.edge(e).weight + weight });
        });
  
        if (node.hasOwnProperty("minRank")) {
          result.setNode(v, {
            borderLeft: node.borderLeft[rank],
            borderRight: node.borderRight[rank]
          });
        }
      }
    });
  
    return result;
  }
  
  function createRootNode(g) {
    var v;
    while (g.hasNode((v = util.uniqueId("_root"))));
    return v;
  }
  
  },{"../util":27,"@dagrejs/graphlib":29}],14:[function(require,module,exports){
  "use strict";
  
  let zipObject = require("../util").zipObject;
  
  module.exports = crossCount;
  
  /*
   * A function that takes a layering (an array of layers, each with an array of
   * ordererd nodes) and a graph and returns a weighted crossing count.
   *
   * Pre-conditions:
   *
   *    1. Input graph must be simple (not a multigraph), directed, and include
   *       only simple edges.
   *    2. Edges in the input graph must have assigned weights.
   *
   * Post-conditions:
   *
   *    1. The graph and layering matrix are left unchanged.
   *
   * This algorithm is derived from Barth, et al., "Bilayer Cross Counting."
   */
  function crossCount(g, layering) {
    let cc = 0;
    for (let i = 1; i < layering.length; ++i) {
      cc += twoLayerCrossCount(g, layering[i-1], layering[i]);
    }
    return cc;
  }
  
  function twoLayerCrossCount(g, northLayer, southLayer) {
    // Sort all of the edges between the north and south layers by their position
    // in the north layer and then the south. Map these edges to the position of
    // their head in the south layer.
    let southPos = zipObject(southLayer, southLayer.map((v, i) => i));
    let southEntries = northLayer.flatMap(v => {
      return g.outEdges(v).map(e => {
        return { pos: southPos[e.w], weight: g.edge(e).weight };
      }).sort((a, b) => a.pos - b.pos);
    });
  
    // Build the accumulator tree
    let firstIndex = 1;
    while (firstIndex < southLayer.length) firstIndex <<= 1;
    let treeSize = 2 * firstIndex - 1;
    firstIndex -= 1;
    let tree = new Array(treeSize).fill(0);
  
    // Calculate the weighted crossings
    let cc = 0;
    southEntries.forEach(entry => {
      let index = entry.pos + firstIndex;
      tree[index] += entry.weight;
      let weightSum = 0;
      while (index > 0) {
        if (index % 2) {
          weightSum += tree[index + 1];
        }
        index = (index - 1) >> 1;
        tree[index] += entry.weight;
      }
      cc += entry.weight * weightSum;
    });
  
    return cc;
  }
  
  },{"../util":27}],15:[function(require,module,exports){
  "use strict";
  
  let initOrder = require("./init-order");
  let crossCount = require("./cross-count");
  let sortSubgraph = require("./sort-subgraph");
  let buildLayerGraph = require("./build-layer-graph");
  let addSubgraphConstraints = require("./add-subgraph-constraints");
  let Graph = require("@dagrejs/graphlib").Graph;
  let util = require("../util");
  
  module.exports = order;
  
  /*
   * Applies heuristics to minimize edge crossings in the graph and sets the best
   * order solution as an order attribute on each node.
   *
   * Pre-conditions:
   *
   *    1. Graph must be DAG
   *    2. Graph nodes must be objects with a "rank" attribute
   *    3. Graph edges must have the "weight" attribute
   *
   * Post-conditions:
   *
   *    1. Graph nodes will have an "order" attribute based on the results of the
   *       algorithm.
   */
  function order(g, opts) {
    if (opts && typeof opts.customOrder === 'function') {
      opts.customOrder(g, order);
      return;
    }
  
    let maxRank = util.maxRank(g),
      downLayerGraphs = buildLayerGraphs(g, util.range(1, maxRank + 1), "inEdges"),
      upLayerGraphs = buildLayerGraphs(g, util.range(maxRank - 1, -1, -1), "outEdges");
  
    let layering = initOrder(g);
    assignOrder(g, layering);
  
    if (opts && opts.disableOptimalOrderHeuristic) {
      return;
    }
  
    let bestCC = Number.POSITIVE_INFINITY,
      best;
  
    for (let i = 0, lastBest = 0; lastBest < 4; ++i, ++lastBest) {
      sweepLayerGraphs(i % 2 ? downLayerGraphs : upLayerGraphs, i % 4 >= 2);
  
      layering = util.buildLayerMatrix(g);
      let cc = crossCount(g, layering);
      if (cc < bestCC) {
        lastBest = 0;
        best = Object.assign({}, layering);
        bestCC = cc;
      }
    }
  
    assignOrder(g, best);
  }
  
  function buildLayerGraphs(g, ranks, relationship) {
    return ranks.map(function(rank) {
      return buildLayerGraph(g, rank, relationship);
    });
  }
  
  function sweepLayerGraphs(layerGraphs, biasRight) {
    let cg = new Graph();
    layerGraphs.forEach(function(lg) {
      let root = lg.graph().root;
      let sorted = sortSubgraph(lg, root, cg, biasRight);
      sorted.vs.forEach((v, i) => lg.node(v).order = i);
      addSubgraphConstraints(lg, cg, sorted.vs);
    });
  }
  
  function assignOrder(g, layering) {
    Object.values(layering).forEach(layer => layer.forEach((v, i) => g.node(v).order = i));
  }
  
  },{"../util":27,"./add-subgraph-constraints":11,"./build-layer-graph":13,"./cross-count":14,"./init-order":16,"./sort-subgraph":18,"@dagrejs/graphlib":29}],16:[function(require,module,exports){
  "use strict";
  
  let util = require("../util");
  
  module.exports = initOrder;
  
  /*
   * Assigns an initial order value for each node by performing a DFS search
   * starting from nodes in the first rank. Nodes are assigned an order in their
   * rank as they are first visited.
   *
   * This approach comes from Gansner, et al., "A Technique for Drawing Directed
   * Graphs."
   *
   * Returns a layering matrix with an array per layer and each layer sorted by
   * the order of its nodes.
   */
  function initOrder(g) {
    let visited = {};
    let simpleNodes = g.nodes().filter(v => !g.children(v).length);

    // The following code would throw "maximum call stack size exceeded" error
    // when handling large graphs. Change it to using loop.
    //
    // let maxRank = Math.max(...simpleNodes.map(v => g.node(v).rank));

    let maxRank =  -Infinity;
    for (let i = 0; i < simpleNodes.length; i++) {
      const rank = g.node(simpleNodes[i]).rank;
      if (rank > maxRank) {
        maxRank = rank;
      }
    }

    let layers = util.range(maxRank + 1).map(() => []);
  
    /* 
     * The following code uses dfs to iterate nodes which will case
     * "maximum call stack size exceeded" error when handling large graphs.
     * Change it to using bfs instead.
     *
     * function dfs(v) {
     *   if (visited[v]) return;
     *   visited[v] = true;
     *   let node = g.node(v);
     *   layers[node.rank].push(v);
     *   g.successors(v).forEach(dfs);
     * }
     *
     * let orderedVs = simpleNodes.sort((a, b) => g.node(a).rank - g.node(b).rank);
     * orderedVs.forEach(dfs);
     */
  
    function bfs(startV) { 
      const queue = [startV];
  
      while (queue.length > 0) {
        const v = queue.shift();
  
        if (visited[v]) continue;
  
        visited[v] = true;
        const node = g.node(v);
        layers[node.rank].push(v);
  
        g.successors(v).forEach(neighbor => queue.push(neighbor));
      }
    }
  
    let orderedVs = simpleNodes.sort((a, b) => g.node(a).rank - g.node(b).rank);
    orderedVs.forEach(bfs); 
  
    return layers;
  }
  
  },{"../util":27}],17:[function(require,module,exports){
  "use strict";
  
  let util = require("../util");
  
  module.exports = resolveConflicts;
  
  /*
   * Given a list of entries of the form {v, barycenter, weight} and a
   * constraint graph this function will resolve any conflicts between the
   * constraint graph and the barycenters for the entries. If the barycenters for
   * an entry would violate a constraint in the constraint graph then we coalesce
   * the nodes in the conflict into a new node that respects the contraint and
   * aggregates barycenter and weight information.
   *
   * This implementation is based on the description in Forster, "A Fast and
   * Simple Hueristic for Constrained Two-Level Crossing Reduction," thought it
   * differs in some specific details.
   *
   * Pre-conditions:
   *
   *    1. Each entry has the form {v, barycenter, weight}, or if the node has
   *       no barycenter, then {v}.
   *
   * Returns:
   *
   *    A new list of entries of the form {vs, i, barycenter, weight}. The list
   *    `vs` may either be a singleton or it may be an aggregation of nodes
   *    ordered such that they do not violate constraints from the constraint
   *    graph. The property `i` is the lowest original index of any of the
   *    elements in `vs`.
   */
  function resolveConflicts(entries, cg) {
    let mappedEntries = {};
    entries.forEach((entry, i) => {
      let tmp = mappedEntries[entry.v] = {
        indegree: 0,
        "in": [],
        out: [],
        vs: [entry.v],
        i: i
      };
      if (entry.barycenter !== undefined) {
        tmp.barycenter = entry.barycenter;
        tmp.weight = entry.weight;
      }
    });
  
    cg.edges().forEach(e => {
      let entryV = mappedEntries[e.v];
      let entryW = mappedEntries[e.w];
      if (entryV !== undefined && entryW !== undefined) {
        entryW.indegree++;
        entryV.out.push(mappedEntries[e.w]);
      }
    });
  
    let sourceSet = Object.values(mappedEntries).filter(entry => !entry.indegree);
  
    return doResolveConflicts(sourceSet);
  }
  
  function doResolveConflicts(sourceSet) {
    let entries = [];
  
    function handleIn(vEntry) {
      return uEntry => {
        if (uEntry.merged) {
          return;
        }
        if (uEntry.barycenter === undefined ||
            vEntry.barycenter === undefined ||
            uEntry.barycenter >= vEntry.barycenter) {
          mergeEntries(vEntry, uEntry);
        }
      };
    }
  
    function handleOut(vEntry) {
      return wEntry => {
        wEntry["in"].push(vEntry);
        if (--wEntry.indegree === 0) {
          sourceSet.push(wEntry);
        }
      };
    }
  
    while (sourceSet.length) {
      let entry = sourceSet.pop();
      entries.push(entry);
      entry["in"].reverse().forEach(handleIn(entry));
      entry.out.forEach(handleOut(entry));
    }
  
    return entries.filter(entry => !entry.merged).map(entry => {
      return util.pick(entry, ["vs", "i", "barycenter", "weight"]);
    });
  }
  
  function mergeEntries(target, source) {
    let sum = 0;
    let weight = 0;
  
    if (target.weight) {
      sum += target.barycenter * target.weight;
      weight += target.weight;
    }
  
    if (source.weight) {
      sum += source.barycenter * source.weight;
      weight += source.weight;
    }
  
    target.vs = source.vs.concat(target.vs);
    target.barycenter = sum / weight;
    target.weight = weight;
    target.i = Math.min(source.i, target.i);
    source.merged = true;
  }
  
  },{"../util":27}],18:[function(require,module,exports){
  let barycenter = require("./barycenter");
  let resolveConflicts = require("./resolve-conflicts");
  let sort = require("./sort");
  
  module.exports = sortSubgraph;
  
  function sortSubgraph(g, v, cg, biasRight) {
    let movable = g.children(v);
    let node = g.node(v);
    let bl = node ? node.borderLeft : undefined;
    let br = node ? node.borderRight: undefined;
    let subgraphs = {};
  
    if (bl) {
      movable = movable.filter(w => w !== bl && w !== br);
    }
  
    let barycenters = barycenter(g, movable);
    barycenters.forEach(entry => {
      if (g.children(entry.v).length) {
        let subgraphResult = sortSubgraph(g, entry.v, cg, biasRight);
        subgraphs[entry.v] = subgraphResult;
        if (subgraphResult.hasOwnProperty("barycenter")) {
          mergeBarycenters(entry, subgraphResult);
        }
      }
    });
  
    let entries = resolveConflicts(barycenters, cg);
    expandSubgraphs(entries, subgraphs);
  
    let result = sort(entries, biasRight);
  
    if (bl) {
      result.vs = [bl, result.vs, br].flat(true);
      if (g.predecessors(bl).length) {
        let blPred = g.node(g.predecessors(bl)[0]),
          brPred = g.node(g.predecessors(br)[0]);
        if (!result.hasOwnProperty("barycenter")) {
          result.barycenter = 0;
          result.weight = 0;
        }
        result.barycenter = (result.barycenter * result.weight +
                             blPred.order + brPred.order) / (result.weight + 2);
        result.weight += 2;
      }
    }
  
    return result;
  }
  
  function expandSubgraphs(entries, subgraphs) {
    entries.forEach(entry => {
      entry.vs = entry.vs.flatMap(v => {
        if (subgraphs[v]) {
          return subgraphs[v].vs;
        }
        return v;
      });
    });
  }
  
  function mergeBarycenters(target, other) {
    if (target.barycenter !== undefined) {
      target.barycenter = (target.barycenter * target.weight +
                           other.barycenter * other.weight) /
                          (target.weight + other.weight);
      target.weight += other.weight;
    } else {
      target.barycenter = other.barycenter;
      target.weight = other.weight;
    }
  }
  
  },{"./barycenter":12,"./resolve-conflicts":17,"./sort":19}],19:[function(require,module,exports){
  let util = require("../util");
  
  module.exports = sort;
  
  function sort(entries, biasRight) {
    let parts = util.partition(entries, entry => {
      return entry.hasOwnProperty("barycenter");
    });
    let sortable = parts.lhs,
      unsortable = parts.rhs.sort((a, b) => b.i - a.i),
      vs = [],
      sum = 0,
      weight = 0,
      vsIndex = 0;
  
    sortable.sort(compareWithBias(!!biasRight));
  
    vsIndex = consumeUnsortable(vs, unsortable, vsIndex);
  
    sortable.forEach(entry => {
      vsIndex += entry.vs.length;
      vs.push(entry.vs);
      sum += entry.barycenter * entry.weight;
      weight += entry.weight;
      vsIndex = consumeUnsortable(vs, unsortable, vsIndex);
    });
  
    let result = { vs: vs.flat(true) };
    if (weight) {
      result.barycenter = sum / weight;
      result.weight = weight;
    }
    return result;
  }
  
  function consumeUnsortable(vs, unsortable, index) {
    let last;
    while (unsortable.length && (last = unsortable[unsortable.length - 1]).i <= index) {
      unsortable.pop();
      vs.push(last.vs);
      index++;
    }
    return index;
  }
  
  function compareWithBias(bias) {
    return (entryV, entryW) => {
      if (entryV.barycenter < entryW.barycenter) {
        return -1;
      } else if (entryV.barycenter > entryW.barycenter) {
        return 1;
      }
  
      return !bias ? entryV.i - entryW.i : entryW.i - entryV.i;
    };
  }
  
  },{"../util":27}],20:[function(require,module,exports){
  module.exports = parentDummyChains;
  
  function parentDummyChains(g) {
    let postorderNums = postorder(g);
  
    g.graph().dummyChains.forEach(v => {
      let node = g.node(v);
      let edgeObj = node.edgeObj;
      let pathData = findPath(g, postorderNums, edgeObj.v, edgeObj.w);
      let path = pathData.path;
      let lca = pathData.lca;
      let pathIdx = 0;
      let pathV = path[pathIdx];
      let ascending = true;
  
      while (v !== edgeObj.w) {
        node = g.node(v);
  
        if (ascending) {
          while ((pathV = path[pathIdx]) !== lca &&
                 g.node(pathV).maxRank < node.rank) {
            pathIdx++;
          }
  
          if (pathV === lca) {
            ascending = false;
          }
        }
  
        if (!ascending) {
          while (pathIdx < path.length - 1 &&
                 g.node(pathV = path[pathIdx + 1]).minRank <= node.rank) {
            pathIdx++;
          }
          pathV = path[pathIdx];
        }
  
        g.setParent(v, pathV);
        v = g.successors(v)[0];
      }
    });
  }
  
  // Find a path from v to w through the lowest common ancestor (LCA). Return the
  // full path and the LCA.
  function findPath(g, postorderNums, v, w) {
    let vPath = [];
    let wPath = [];
    let low = Math.min(postorderNums[v].low, postorderNums[w].low);
    let lim = Math.max(postorderNums[v].lim, postorderNums[w].lim);
    let parent;
    let lca;
  
    // Traverse up from v to find the LCA
    parent = v;
    do {
      parent = g.parent(parent);
      vPath.push(parent);
    } while (parent &&
             (postorderNums[parent].low > low || lim > postorderNums[parent].lim));
    lca = parent;
  
    // Traverse from w to LCA
    parent = w;
    while ((parent = g.parent(parent)) !== lca) {
      wPath.push(parent);
    }
  
    return { path: vPath.concat(wPath.reverse()), lca: lca };
  }
  
  function postorder(g) {
    let result = {};
    let lim = 0;
  
    function dfs(v) {
      let low = lim;
      g.children(v).forEach(dfs);
      result[v] = { low: low, lim: lim++ };
    }
    g.children().forEach(dfs);
  
    return result;
  }
  
  },{}],21:[function(require,module,exports){
  "use strict";
  
  let Graph = require("@dagrejs/graphlib").Graph;
  let util = require("../util");
  
  /*
   * This module provides coordinate assignment based on Brandes and Köpf, "Fast
   * and Simple Horizontal Coordinate Assignment."
   */
  
  module.exports = {
    positionX: positionX,
    findType1Conflicts: findType1Conflicts,
    findType2Conflicts: findType2Conflicts,
    addConflict: addConflict,
    hasConflict: hasConflict,
    verticalAlignment: verticalAlignment,
    horizontalCompaction: horizontalCompaction,
    alignCoordinates: alignCoordinates,
    findSmallestWidthAlignment: findSmallestWidthAlignment,
    balance: balance
  };
  
  /*
   * Marks all edges in the graph with a type-1 conflict with the "type1Conflict"
   * property. A type-1 conflict is one where a non-inner segment crosses an
   * inner segment. An inner segment is an edge with both incident nodes marked
   * with the "dummy" property.
   *
   * This algorithm scans layer by layer, starting with the second, for type-1
   * conflicts between the current layer and the previous layer. For each layer
   * it scans the nodes from left to right until it reaches one that is incident
   * on an inner segment. It then scans predecessors to determine if they have
   * edges that cross that inner segment. At the end a final scan is done for all
   * nodes on the current rank to see if they cross the last visited inner
   * segment.
   *
   * This algorithm (safely) assumes that a dummy node will only be incident on a
   * single node in the layers being scanned.
   */
  function findType1Conflicts(g, layering) {
    let conflicts = {};
  
    function visitLayer(prevLayer, layer) {
      let
        // last visited node in the previous layer that is incident on an inner
        // segment.
        k0 = 0,
        // Tracks the last node in this layer scanned for crossings with a type-1
        // segment.
        scanPos = 0,
        prevLayerLength = prevLayer.length,
        lastNode = layer[layer.length - 1];
  
      layer.forEach((v, i) => {
        let w = findOtherInnerSegmentNode(g, v),
          k1 = w ? g.node(w).order : prevLayerLength;
  
        if (w || v === lastNode) {
          layer.slice(scanPos, i+1).forEach(scanNode => {
            g.predecessors(scanNode).forEach(u => {
              let uLabel = g.node(u),
                uPos = uLabel.order;
              if ((uPos < k0 || k1 < uPos) &&
                  !(uLabel.dummy && g.node(scanNode).dummy)) {
                addConflict(conflicts, u, scanNode);
              }
            });
          });
          scanPos = i + 1;
          k0 = k1;
        }
      });
  
      return layer;
    }
  
    layering.length && layering.reduce(visitLayer);
  
    return conflicts;
  }
  
  function findType2Conflicts(g, layering) {
    let conflicts = {};
  
    function scan(south, southPos, southEnd, prevNorthBorder, nextNorthBorder) {
      let v;
      util.range(southPos, southEnd).forEach(i => {
        v = south[i];
        if (g.node(v).dummy) {
          g.predecessors(v).forEach(u => {
            let uNode = g.node(u);
            if (uNode.dummy &&
                (uNode.order < prevNorthBorder || uNode.order > nextNorthBorder)) {
              addConflict(conflicts, u, v);
            }
          });
        }
      });
    }
  
  
    function visitLayer(north, south) {
      let prevNorthPos = -1,
        nextNorthPos,
        southPos = 0;
  
      south.forEach((v, southLookahead) => {
        if (g.node(v).dummy === "border") {
          let predecessors = g.predecessors(v);
          if (predecessors.length) {
            nextNorthPos = g.node(predecessors[0]).order;
            scan(south, southPos, southLookahead, prevNorthPos, nextNorthPos);
            southPos = southLookahead;
            prevNorthPos = nextNorthPos;
          }
        }
        scan(south, southPos, south.length, nextNorthPos, north.length);
      });
  
      return south;
    }
  
    layering.length && layering.reduce(visitLayer);
  
    return conflicts;
  }
  
  function findOtherInnerSegmentNode(g, v) {
    if (g.node(v).dummy) {
      return g.predecessors(v).find(u => g.node(u).dummy);
    }
  }
  
  function addConflict(conflicts, v, w) {
    if (v > w) {
      let tmp = v;
      v = w;
      w = tmp;
    }
  
    let conflictsV = conflicts[v];
    if (!conflictsV) {
      conflicts[v] = conflictsV = {};
    }
    conflictsV[w] = true;
  }
  
  function hasConflict(conflicts, v, w) {
    if (v > w) {
      let tmp = v;
      v = w;
      w = tmp;
    }
    return !!conflicts[v] && conflicts[v].hasOwnProperty(w);
  }
  
  /*
   * Try to align nodes into vertical "blocks" where possible. This algorithm
   * attempts to align a node with one of its median neighbors. If the edge
   * connecting a neighbor is a type-1 conflict then we ignore that possibility.
   * If a previous node has already formed a block with a node after the node
   * we're trying to form a block with, we also ignore that possibility - our
   * blocks would be split in that scenario.
   */
  function verticalAlignment(g, layering, conflicts, neighborFn) {
    let root = {},
      align = {},
      pos = {};
  
    // We cache the position here based on the layering because the graph and
    // layering may be out of sync. The layering matrix is manipulated to
    // generate different extreme alignments.
    layering.forEach(layer => {
      layer.forEach((v, order) => {
        root[v] = v;
        align[v] = v;
        pos[v] = order;
      });
    });
  
    layering.forEach(layer => {
      let prevIdx = -1;
      layer.forEach(v => {
        let ws = neighborFn(v);
        if (ws.length) {
          ws = ws.sort((a, b) => pos[a] - pos[b]);
          let mp = (ws.length - 1) / 2;
          for (let i = Math.floor(mp), il = Math.ceil(mp); i <= il; ++i) {
            let w = ws[i];
            if (align[v] === v &&
                prevIdx < pos[w] &&
                !hasConflict(conflicts, v, w)) {
              align[w] = v;
              align[v] = root[v] = root[w];
              prevIdx = pos[w];
            }
          }
        }
      });
    });
  
    return { root: root, align: align };
  }
  
  function horizontalCompaction(g, layering, root, align, reverseSep) {
    // This portion of the algorithm differs from BK due to a number of problems.
    // Instead of their algorithm we construct a new block graph and do two
    // sweeps. The first sweep places blocks with the smallest possible
    // coordinates. The second sweep removes unused space by moving blocks to the
    // greatest coordinates without violating separation.
    let xs = {},
      blockG = buildBlockGraph(g, layering, root, reverseSep),
      borderType = reverseSep ? "borderLeft" : "borderRight";
  
    function iterate(setXsFunc, nextNodesFunc) {
      let stack = blockG.nodes();
      let elem = stack.pop();
      let visited = {};
      while (elem) {
        if (visited[elem]) {
          setXsFunc(elem);
        } else {
          visited[elem] = true;
          stack.push(elem);
          stack = stack.concat(nextNodesFunc(elem));
        }
  
        elem = stack.pop();
      }
    }
  
    // First pass, assign smallest coordinates
    function pass1(elem) {
      xs[elem] = blockG.inEdges(elem).reduce((acc, e) => {
        return Math.max(acc, xs[e.v] + blockG.edge(e));
      }, 0);
    }
  
    // Second pass, assign greatest coordinates
    function pass2(elem) {
      let min = blockG.outEdges(elem).reduce((acc, e) => {
        return Math.min(acc, xs[e.w] - blockG.edge(e));
      }, Number.POSITIVE_INFINITY);
  
      let node = g.node(elem);
      if (min !== Number.POSITIVE_INFINITY && node.borderType !== borderType) {
        xs[elem] = Math.max(xs[elem], min);
      }
    }
  
    iterate(pass1, blockG.predecessors.bind(blockG));
    iterate(pass2, blockG.successors.bind(blockG));
  
    // Assign x coordinates to all nodes
    Object.keys(align).forEach(v => xs[v] = xs[root[v]]);
  
    return xs;
  }
  
  
  function buildBlockGraph(g, layering, root, reverseSep) {
    let blockGraph = new Graph(),
      graphLabel = g.graph(),
      sepFn = sep(graphLabel.nodesep, graphLabel.edgesep, reverseSep);
  
    layering.forEach(layer => {
      let u;
      layer.forEach(v => {
        let vRoot = root[v];
        blockGraph.setNode(vRoot);
        if (u) {
          var uRoot = root[u],
            prevMax = blockGraph.edge(uRoot, vRoot);
          blockGraph.setEdge(uRoot, vRoot, Math.max(sepFn(g, v, u), prevMax || 0));
        }
        u = v;
      });
    });
  
    return blockGraph;
  }
  
  /*
   * Returns the alignment that has the smallest width of the given alignments.
   */
  function findSmallestWidthAlignment(g, xss) {
    return Object.values(xss).reduce((currentMinAndXs, xs) => {
      let max = Number.NEGATIVE_INFINITY;
      let min = Number.POSITIVE_INFINITY;
  
      Object.entries(xs).forEach(([v, x]) => {
        let halfWidth = width(g, v) / 2;
  
        max = Math.max(x + halfWidth, max);
        min = Math.min(x - halfWidth, min);
      });
  
      const newMin = max - min;
      if (newMin < currentMinAndXs[0]) {
        currentMinAndXs = [newMin, xs];
      }
      return currentMinAndXs;
    }, [Number.POSITIVE_INFINITY, null])[1];
  }
  
  /*
   * Align the coordinates of each of the layout alignments such that
   * left-biased alignments have their minimum coordinate at the same point as
   * the minimum coordinate of the smallest width alignment and right-biased
   * alignments have their maximum coordinate at the same point as the maximum
   * coordinate of the smallest width alignment.
   */
  function alignCoordinates(xss, alignTo) {
    let alignToVals = Object.values(alignTo);

    // The following code would throw "maximum call stack size exceeded" error
    // when handling large graphs. Change them to using loop.
    //
    // alignToMin = Math.min(...alignToVals),
    // alignToMax = Math.max(...alignToVals);
    let alignToMin = Infinity;
    let alignToMax = -Infinity;
    for (const v of alignToVals) {
      if (v < alignToMin) {
        alignToMin = v;
      }
      if (v > alignToMax) {
        alignToMax = v;
      }
    }
  
    ["u", "d"].forEach(vert => {
      ["l", "r"].forEach(horiz => {
        let alignment = vert + horiz,
          xs = xss[alignment];
  
        if (xs === alignTo) return;
        
        // Math.min(...) and Math.max(...) below would throw "maximum call stack
        // "size exceeded" error when handling large graphs. Change them to
        // using loop.
        let xsVals = Object.values(xs);
        let xMin = Infinity;
        let xMax = -Infinity;
        for (const v of xsVals) {
          if (v < xMin) {
            xMin = v;
          }
          if (v > xMax) {
            xMax = v;
          }
        }
  
        // let delta = alignToMin - Math.min(...xsVals);
        let delta = alignToMin - xMin;;
        if (horiz !== "l") {
          // delta = alignToMax - Math.max(...xsVals);
          delta = alignToMax - xMax;
        }
  
        if (delta) {
          xss[alignment] = util.mapValues(xs, x => x + delta);
        }
      });
    });
  }
  
  function balance(xss, align) {
    return util.mapValues(xss.ul, (num, v) => {
      if (align) {
        return xss[align.toLowerCase()][v];
      } else {
        let xs = Object.values(xss).map(xs => xs[v]).sort((a, b) => a - b);
        return (xs[1] + xs[2]) / 2;
      }
    });
  }
  
  function positionX(g) {
    let layering = util.buildLayerMatrix(g);
    let conflicts = Object.assign(
      findType1Conflicts(g, layering),
      findType2Conflicts(g, layering));
  
    let xss = {};
    let adjustedLayering;
    ["u", "d"].forEach(vert => {
      adjustedLayering = vert === "u" ? layering : Object.values(layering).reverse();
      ["l", "r"].forEach(horiz => {
        if (horiz === "r") {
          adjustedLayering = adjustedLayering.map(inner => {
            return Object.values(inner).reverse();
          });
        }
  
        let neighborFn = (vert === "u" ? g.predecessors : g.successors).bind(g);
        let align = verticalAlignment(g, adjustedLayering, conflicts, neighborFn);
        let xs = horizontalCompaction(g, adjustedLayering,
          align.root, align.align, horiz === "r");
        if (horiz === "r") {
          xs = util.mapValues(xs, x => -x);
        }
        xss[vert + horiz] = xs;
      });
    });
  
  
    let smallestWidth = findSmallestWidthAlignment(g, xss);
    alignCoordinates(xss, smallestWidth);
    return balance(xss, g.graph().align);
  }
  
  function sep(nodeSep, edgeSep, reverseSep) {
    return (g, v, w) => {
      let vLabel = g.node(v);
      let wLabel = g.node(w);
      let sum = 0;
      let delta;
  
      sum += vLabel.width / 2;
      if (vLabel.hasOwnProperty("labelpos")) {
        switch (vLabel.labelpos.toLowerCase()) {
        case "l": delta = -vLabel.width / 2; break;
        case "r": delta = vLabel.width / 2; break;
        }
      }
      if (delta) {
        sum += reverseSep ? delta : -delta;
      }
      delta = 0;
  
      sum += (vLabel.dummy ? edgeSep : nodeSep) / 2;
      sum += (wLabel.dummy ? edgeSep : nodeSep) / 2;
  
      sum += wLabel.width / 2;
      if (wLabel.hasOwnProperty("labelpos")) {
        switch (wLabel.labelpos.toLowerCase()) {
        case "l": delta = wLabel.width / 2; break;
        case "r": delta = -wLabel.width / 2; break;
        }
      }
      if (delta) {
        sum += reverseSep ? delta : -delta;
      }
      delta = 0;
  
      return sum;
    };
  }
  
  function width(g, v) {
    return g.node(v).width;
  }
  
  },{"../util":27,"@dagrejs/graphlib":29}],22:[function(require,module,exports){
  "use strict";
  
  let util = require("../util");
  let positionX = require("./bk").positionX;
  
  module.exports = position;
  
  function position(g) {
    g = util.asNonCompoundGraph(g);
  
    positionY(g);
    Object.entries(positionX(g)).forEach(([v, x]) => g.node(v).x = x);
  }
  
  function positionY(g) {
    let layering = util.buildLayerMatrix(g);
    let rankSep = g.graph().ranksep;
    let prevY = 0;
    layering.forEach(layer => {
      const maxHeight = layer.reduce((acc, v) => {
        const height = g.node(v).height;
        if (acc > height) {
          return acc;
        } else {
          return height;
        }
      }, 0);
      layer.forEach(v => g.node(v).y = prevY + maxHeight / 2);
      prevY += maxHeight + rankSep;
    });
  }
  
  
  },{"../util":27,"./bk":21}],23:[function(require,module,exports){
  "use strict";
  
  var Graph = require("@dagrejs/graphlib").Graph;
  var slack = require("./util").slack;
  
  module.exports = feasibleTree;
  
  /*
   * Constructs a spanning tree with tight edges and adjusted the input node's
   * ranks to achieve this. A tight edge is one that is has a length that matches
   * its "minlen" attribute.
   *
   * The basic structure for this function is derived from Gansner, et al., "A
   * Technique for Drawing Directed Graphs."
   *
   * Pre-conditions:
   *
   *    1. Graph must be a DAG.
   *    2. Graph must be connected.
   *    3. Graph must have at least one node.
   *    5. Graph nodes must have been previously assigned a "rank" property that
   *       respects the "minlen" property of incident edges.
   *    6. Graph edges must have a "minlen" property.
   *
   * Post-conditions:
   *
   *    - Graph nodes will have their rank adjusted to ensure that all edges are
   *      tight.
   *
   * Returns a tree (undirected graph) that is constructed using only "tight"
   * edges.
   */
  function feasibleTree(g) {
    var t = new Graph({ directed: false });
  
    // Choose arbitrary node from which to start our tree
    var start = g.nodes()[0];
    var size = g.nodeCount();
    t.setNode(start, {});
  
    var edge, delta;
    while (tightTree(t, g) < size) {
      edge = findMinSlackEdge(t, g);
      delta = t.hasNode(edge.v) ? slack(g, edge) : -slack(g, edge);
      shiftRanks(t, g, delta);
    }
  
    return t;
  }
  
  /*
   * Finds a maximal tree of tight edges and returns the number of nodes in the
   * tree.
   */
  function tightTree(t, g) {
    function dfs(v) {
      g.nodeEdges(v).forEach(e => {
        var edgeV = e.v,
          w = (v === edgeV) ? e.w : edgeV;
        if (!t.hasNode(w) && !slack(g, e)) {
          t.setNode(w, {});
          t.setEdge(v, w, {});
          dfs(w);
        }
      });
    }
  
    t.nodes().forEach(dfs);
    return t.nodeCount();
  }
  
  /*
   * Finds the edge with the smallest slack that is incident on tree and returns
   * it.
   */
  function findMinSlackEdge(t, g) {
    const edges = g.edges();
  
    return edges.reduce((acc, edge) => {
      let edgeSlack = Number.POSITIVE_INFINITY;
      if (t.hasNode(edge.v) !== t.hasNode(edge.w)) {
        edgeSlack = slack(g, edge);
      }
  
      if (edgeSlack < acc[0]) {
        return [edgeSlack, edge];
      }
  
      return acc;
    }, [Number.POSITIVE_INFINITY, null])[1];
  }
  
  function shiftRanks(t, g, delta) {
    t.nodes().forEach(v => g.node(v).rank += delta);
  }
  
  },{"./util":26,"@dagrejs/graphlib":29}],24:[function(require,module,exports){
  "use strict";
  
  var rankUtil = require("./util");
  var longestPath = rankUtil.longestPath;
  var feasibleTree = require("./feasible-tree");
  var networkSimplex = require("./network-simplex");
  
  module.exports = rank;
  
  /*
   * Assigns a rank to each node in the input graph that respects the "minlen"
   * constraint specified on edges between nodes.
   *
   * This basic structure is derived from Gansner, et al., "A Technique for
   * Drawing Directed Graphs."
   *
   * Pre-conditions:
   *
   *    1. Graph must be a connected DAG
   *    2. Graph nodes must be objects
   *    3. Graph edges must have "weight" and "minlen" attributes
   *
   * Post-conditions:
   *
   *    1. Graph nodes will have a "rank" attribute based on the results of the
   *       algorithm. Ranks can start at any index (including negative), we'll
   *       fix them up later.
   */
  function rank(g) {
    switch(g.graph().ranker) {
    case "network-simplex": networkSimplexRanker(g); break;
    case "tight-tree": tightTreeRanker(g); break;
    case "longest-path": longestPathRanker(g); break;
    default: networkSimplexRanker(g);
    }
  }
  
  // A fast and simple ranker, but results are far from optimal.
  var longestPathRanker = longestPath;
  
  function tightTreeRanker(g) {
    longestPath(g);
    feasibleTree(g);
  }
  
  function networkSimplexRanker(g) {
    networkSimplex(g);
  }
  
  },{"./feasible-tree":23,"./network-simplex":25,"./util":26}],25:[function(require,module,exports){
  "use strict";
  
  var feasibleTree = require("./feasible-tree");
  var slack = require("./util").slack;
  var initRank = require("./util").longestPath;
  var preorder = require("@dagrejs/graphlib").alg.preorder;
  var postorder = require("@dagrejs/graphlib").alg.postorder;
  var simplify = require("../util").simplify;
  
  module.exports = networkSimplex;
  
  // Expose some internals for testing purposes
  networkSimplex.initLowLimValues = initLowLimValues;
  networkSimplex.initCutValues = initCutValues;
  networkSimplex.calcCutValue = calcCutValue;
  networkSimplex.leaveEdge = leaveEdge;
  networkSimplex.enterEdge = enterEdge;
  networkSimplex.exchangeEdges = exchangeEdges;
  
  /*
   * The network simplex algorithm assigns ranks to each node in the input graph
   * and iteratively improves the ranking to reduce the length of edges.
   *
   * Preconditions:
   *
   *    1. The input graph must be a DAG.
   *    2. All nodes in the graph must have an object value.
   *    3. All edges in the graph must have "minlen" and "weight" attributes.
   *
   * Postconditions:
   *
   *    1. All nodes in the graph will have an assigned "rank" attribute that has
   *       been optimized by the network simplex algorithm. Ranks start at 0.
   *
   *
   * A rough sketch of the algorithm is as follows:
   *
   *    1. Assign initial ranks to each node. We use the longest path algorithm,
   *       which assigns ranks to the lowest position possible. In general this
   *       leads to very wide bottom ranks and unnecessarily long edges.
   *    2. Construct a feasible tight tree. A tight tree is one such that all
   *       edges in the tree have no slack (difference between length of edge
   *       and minlen for the edge). This by itself greatly improves the assigned
   *       rankings by shorting edges.
   *    3. Iteratively find edges that have negative cut values. Generally a
   *       negative cut value indicates that the edge could be removed and a new
   *       tree edge could be added to produce a more compact graph.
   *
   * Much of the algorithms here are derived from Gansner, et al., "A Technique
   * for Drawing Directed Graphs." The structure of the file roughly follows the
   * structure of the overall algorithm.
   */
  function networkSimplex(g) {
    g = simplify(g);
    initRank(g);
    var t = feasibleTree(g);
    initLowLimValues(t);
    initCutValues(t, g);
  
    var e, f;
    while ((e = leaveEdge(t))) {
      f = enterEdge(t, g, e);
      exchangeEdges(t, g, e, f);
    }
  }
  
  /*
   * Initializes cut values for all edges in the tree.
   */
  function initCutValues(t, g) {
    var vs = postorder(t, t.nodes());
    vs = vs.slice(0, vs.length - 1);
    vs.forEach(v => assignCutValue(t, g, v));
  }
  
  function assignCutValue(t, g, child) {
    var childLab = t.node(child);
    var parent = childLab.parent;
    t.edge(child, parent).cutvalue = calcCutValue(t, g, child);
  }
  
  /*
   * Given the tight tree, its graph, and a child in the graph calculate and
   * return the cut value for the edge between the child and its parent.
   */
  function calcCutValue(t, g, child) {
    var childLab = t.node(child);
    var parent = childLab.parent;
    // True if the child is on the tail end of the edge in the directed graph
    var childIsTail = true;
    // The graph's view of the tree edge we're inspecting
    var graphEdge = g.edge(child, parent);
    // The accumulated cut value for the edge between this node and its parent
    var cutValue = 0;
  
    if (!graphEdge) {
      childIsTail = false;
      graphEdge = g.edge(parent, child);
    }
  
    cutValue = graphEdge.weight;
  
    g.nodeEdges(child).forEach(e => {
      var isOutEdge = e.v === child,
        other = isOutEdge ? e.w : e.v;
  
      if (other !== parent) {
        var pointsToHead = isOutEdge === childIsTail,
          otherWeight = g.edge(e).weight;
  
        cutValue += pointsToHead ? otherWeight : -otherWeight;
        if (isTreeEdge(t, child, other)) {
          var otherCutValue = t.edge(child, other).cutvalue;
          cutValue += pointsToHead ? -otherCutValue : otherCutValue;
        }
      }
    });
  
    return cutValue;
  }
  
  function initLowLimValues(tree, root) {
    if (arguments.length < 2) {
      root = tree.nodes()[0];
    }
    // The following code would throw "maximum call stack size exceeded" error
    // when handling large graphs. Change it to using an iterative version.
    //
    // dfsAssignLowLim(tree, {}, 1, root);

    dfsAssignLowLimIterative(tree, {}, 1, root);
  }
  
  function dfsAssignLowLim(tree, visited, nextLim, v, parent) {
    var low = nextLim;
    var label = tree.node(v);
  
    visited[v] = true;
    tree.neighbors(v).forEach(w => {
      if (!visited.hasOwnProperty(w)) {
        nextLim = dfsAssignLowLim(tree, visited, nextLim, w, v);
      }
    });
  
    label.low = low;
    label.lim = nextLim++;
    if (parent) {
      label.parent = parent;
    } else {
      // TODO should be able to remove this when we incrementally update low lim
      delete label.parent;
    }
  
    return nextLim;
  }
    
  function dfsAssignLowLimIterative(tree, visited, nextLim, startNode, parent = null) {
    const stack = [];
    const lowLimStack = [];

    stack.push({ v: startNode, parent: parent, stage: 0 });

    while (stack.length > 0) {
      let { v, parent, stage } = stack.pop();
      let label = tree.node(v);

      // Stage 0 means this node is being processed for the first time
      if (stage === 0) {
        visited[v] = true;
        var low = nextLim;
        label.low = low;
        lowLimStack.push({ node: v, low });

        // Mark the node as in-process and push it back to the stack
        stack.push({ v: v, parent: parent, stage: 1 });

        // Process its neighbors
        let neighbors = tree.neighbors(v);
        for (let i = neighbors.length-1; i>=0; i--) {
          const w = neighbors[i];
          if (!visited.hasOwnProperty(w)) {
            // Push neighbor node onto the stack
            stack.push({ v: w, parent: v, stage: 0 });
          }
        }
      }

      // Stage 1 means we are returning to the node after processing all its neighbors
      else if (stage === 1) {
        // Assign limits and update parent information
        let lim = nextLim++;
        label.lim = lim;

        if (parent) {
          label.parent = parent;
        } else {
          delete label.parent;
        }

        let lowLimNode = lowLimStack.pop();
        label.low = lowLimNode.low;
      }
    }

    return nextLim;
  }
  
  function leaveEdge(tree) {
    return tree.edges().find(e => tree.edge(e).cutvalue < 0);
  }
  
  function enterEdge(t, g, edge) {
    var v = edge.v;
    var w = edge.w;
  
    // For the rest of this function we assume that v is the tail and w is the
    // head, so if we don't have this edge in the graph we should flip it to
    // match the correct orientation.
    if (!g.hasEdge(v, w)) {
      v = edge.w;
      w = edge.v;
    }
  
    var vLabel = t.node(v);
    var wLabel = t.node(w);
    var tailLabel = vLabel;
    var flip = false;
  
    // If the root is in the tail of the edge then we need to flip the logic that
    // checks for the head and tail nodes in the candidates function below.
    if (vLabel.lim > wLabel.lim) {
      tailLabel = wLabel;
      flip = true;
    }
  
    var candidates = g.edges().filter(edge => {
      return flip === isDescendant(t, t.node(edge.v), tailLabel) &&
             flip !== isDescendant(t, t.node(edge.w), tailLabel);
    });
  
    return candidates.reduce((acc, edge) => {
      if (slack(g, edge) < slack(g, acc)) {
        return edge;
      }
  
      return acc;
    });
  }
  
  function exchangeEdges(t, g, e, f) {
    var v = e.v;
    var w = e.w;
    t.removeEdge(v, w);
    t.setEdge(f.v, f.w, {});
    initLowLimValues(t);
    initCutValues(t, g);
    updateRanks(t, g);
  }
  
  function updateRanks(t, g) {
    var root = t.nodes().find(v => !g.node(v).parent);
    var vs = preorder(t, root);
    vs = vs.slice(1);
    vs.forEach(v => {
      var parent = t.node(v).parent,
        edge = g.edge(v, parent),
        flipped = false;
  
      if (!edge) {
        edge = g.edge(parent, v);
        flipped = true;
      }
  
      g.node(v).rank = g.node(parent).rank + (flipped ? edge.minlen : -edge.minlen);
    });
  }
  
  /*
   * Returns true if the edge is in the tree.
   */
  function isTreeEdge(tree, u, v) {
    return tree.hasEdge(u, v);
  }
  
  /*
   * Returns true if the specified node is descendant of the root node per the
   * assigned low and lim attributes in the tree.
   */
  function isDescendant(tree, vLabel, rootLabel) {
    return rootLabel.low <= vLabel.lim && vLabel.lim <= rootLabel.lim;
  }
  
  },{"../util":27,"./feasible-tree":23,"./util":26,"@dagrejs/graphlib":29}],26:[function(require,module,exports){
  "use strict";
  
  module.exports = {
    longestPath: longestPath,
    slack: slack
  };
  
  /*
   * Initializes ranks for the input graph using the longest path algorithm. This
   * algorithm scales well and is fast in practice, it yields rather poor
   * solutions. Nodes are pushed to the lowest layer possible, leaving the bottom
   * ranks wide and leaving edges longer than necessary. However, due to its
   * speed, this algorithm is good for getting an initial ranking that can be fed
   * into other algorithms.
   *
   * This algorithm does not normalize layers because it will be used by other
   * algorithms in most cases. If using this algorithm directly, be sure to
   * run normalize at the end.
   *
   * Pre-conditions:
   *
   *    1. Input graph is a DAG.
   *    2. Input graph node labels can be assigned properties.
   *
   * Post-conditions:
   *
   *    1. Each node will be assign an (unnormalized) "rank" property.
   */
  function longestPath(g) {
    var visited = {};
  
    function dfs(v) {
      var label = g.node(v);
      if (visited.hasOwnProperty(v)) {
        return label.rank;
      }
      visited[v] = true;
  
      var rank = Math.min(...g.outEdges(v).map(e => {
        if (e == null) {
          return Number.POSITIVE_INFINITY;
        }
  
        return dfs(e.w) - g.edge(e).minlen;
      }));
  
      if (rank === Number.POSITIVE_INFINITY) {
        rank = 0;
      }
  
      return (label.rank = rank);
    }
  
    g.sources().forEach(dfs);
  }
  
  /*
   * Returns the amount of slack for the given edge. The slack is defined as the
   * difference between the length of the edge and its minimum length.
   */
  function slack(g, e) {
    return g.node(e.w).rank - g.node(e.v).rank - g.edge(e).minlen;
  }
  
  },{}],27:[function(require,module,exports){
  /* eslint "no-console": off */
  
  "use strict";
  
  let Graph = require("@dagrejs/graphlib").Graph;
  
  module.exports = {
    addBorderNode,
    addDummyNode,
    asNonCompoundGraph,
    buildLayerMatrix,
    intersectRect,
    mapValues,
    maxRank,
    normalizeRanks,
    notime,
    partition,
    pick,
    predecessorWeights,
    range,
    removeEmptyRanks,
    simplify,
    successorWeights,
    time,
    uniqueId,
    zipObject,
  };
  
  /*
   * Adds a dummy node to the graph and return v.
   */
  function addDummyNode(g, type, attrs, name) {
    let v;
    do {
      v = uniqueId(name);
    } while (g.hasNode(v));
  
    attrs.dummy = type;
    g.setNode(v, attrs);
    return v;
  }
  
  /*
   * Returns a new graph with only simple edges. Handles aggregation of data
   * associated with multi-edges.
   */
  function simplify(g) {
    let simplified = new Graph().setGraph(g.graph());
    g.nodes().forEach(v => simplified.setNode(v, g.node(v)));
    g.edges().forEach(e => {
      let simpleLabel = simplified.edge(e.v, e.w) || { weight: 0, minlen: 1 };
      let label = g.edge(e);
      simplified.setEdge(e.v, e.w, {
        weight: simpleLabel.weight + label.weight,
        minlen: Math.max(simpleLabel.minlen, label.minlen)
      });
    });
    return simplified;
  }
  
  function asNonCompoundGraph(g) {
    let simplified = new Graph({ multigraph: g.isMultigraph() }).setGraph(g.graph());
    g.nodes().forEach(v => {
      if (!g.children(v).length) {
        simplified.setNode(v, g.node(v));
      }
    });
    g.edges().forEach(e => {
      simplified.setEdge(e, g.edge(e));
    });
    return simplified;
  }
  
  function successorWeights(g) {
    let weightMap = g.nodes().map(v => {
      let sucs = {};
      g.outEdges(v).forEach(e => {
        sucs[e.w] = (sucs[e.w] || 0) + g.edge(e).weight;
      });
      return sucs;
    });
    return zipObject(g.nodes(), weightMap);
  }
  
  function predecessorWeights(g) {
    let weightMap = g.nodes().map(v => {
      let preds = {};
      g.inEdges(v).forEach(e => {
        preds[e.v] = (preds[e.v] || 0) + g.edge(e).weight;
      });
      return preds;
    });
    return zipObject(g.nodes(), weightMap);
  }
  
  /*
   * Finds where a line starting at point ({x, y}) would intersect a rectangle
   * ({x, y, width, height}) if it were pointing at the rectangle's center.
   */
  function intersectRect(rect, point) {
    let x = rect.x;
    let y = rect.y;
  
    // Rectangle intersection algorithm from:
    // http://math.stackexchange.com/questions/108113/find-edge-between-two-boxes
    let dx = point.x - x;
    let dy = point.y - y;
    let w = rect.width / 2;
    let h = rect.height / 2;
  
    if (!dx && !dy) {
      throw new Error("Not possible to find intersection inside of the rectangle");
    }
  
    let sx, sy;
    if (Math.abs(dy) * w > Math.abs(dx) * h) {
      // Intersection is top or bottom of rect.
      if (dy < 0) {
        h = -h;
      }
      sx = h * dx / dy;
      sy = h;
    } else {
      // Intersection is left or right of rect.
      if (dx < 0) {
        w = -w;
      }
      sx = w;
      sy = w * dy / dx;
    }
  
    return { x: x + sx, y: y + sy };
  }
  
  /*
   * Given a DAG with each node assigned "rank" and "order" properties, this
   * function will produce a matrix with the ids of each node.
   */
  function buildLayerMatrix(g) {
    let layering = range(maxRank(g) + 1).map(() => []);
    g.nodes().forEach(v => {
      let node = g.node(v);
      let rank = node.rank;
      if (rank !== undefined) {
        layering[rank][node.order] = v;
      }
    });
    return layering;
  }
  
  /*
   * Adjusts the ranks for all nodes in the graph such that all nodes v have
   * rank(v) >= 0 and at least one node w has rank(w) = 0.
   */
  function normalizeRanks(g) {
    let min = Math.min(...g.nodes().map(v => {
      let rank = g.node(v).rank;
      if (rank === undefined) {
        return Number.MAX_VALUE;
      }
  
      return rank;
    }));
    g.nodes().forEach(v => {
      let node = g.node(v);
      if (node.hasOwnProperty("rank")) {
        node.rank -= min;
      }
    });
  }
  
  function removeEmptyRanks(g) {
    // Ranks may not start at 0, so we need to offset them
    let offset = Math.min(...g.nodes().map(v => g.node(v).rank));
  
    let layers = [];
    g.nodes().forEach(v => {
      let rank = g.node(v).rank - offset;
      if (!layers[rank]) {
        layers[rank] = [];
      }
      layers[rank].push(v);
    });
  
    let delta = 0;
    let nodeRankFactor = g.graph().nodeRankFactor;
    Array.from(layers).forEach((vs, i) => {
      if (vs === undefined && i % nodeRankFactor !== 0) {
        --delta;
      } else if (vs !== undefined && delta) {
        vs.forEach(v => g.node(v).rank += delta);
      }
    });
  }
  
  function addBorderNode(g, prefix, rank, order) {
    let node = {
      width: 0,
      height: 0
    };
    if (arguments.length >= 4) {
      node.rank = rank;
      node.order = order;
    }
    return addDummyNode(g, "border", node, prefix);
  }
  
  function maxRank(g) {
    /*
     * The following code would throw "maximum call stack size exceeded" error
     * when handling large graphs. Change it to using for loop instead.
     *
     *  return Math.max(...g.nodes().map(v => {
     *    let rank = g.node(v).rank;
     *    if (rank === undefined) {
     *      return Number.MIN_VALUE;
     *    }
     *    return rank;
     *  }));
     */
  
    let maxRank = Number.MIN_VALUE;
  
    for (const v of g.nodes()) {
      let rank = g.node(v).rank;
  
      if (rank === undefined) {
        continue; 
      }
  
      if (rank > maxRank) {
        maxRank = rank;
      }
    }
  
    return maxRank;
  }
  
  /*
   * Partition a collection into two groups: `lhs` and `rhs`. If the supplied
   * function returns true for an entry it goes into `lhs`. Otherwise it goes
   * into `rhs.
   */
  function partition(collection, fn) {
    let result = { lhs: [], rhs: [] };
    collection.forEach(value => {
      if (fn(value)) {
        result.lhs.push(value);
      } else {
        result.rhs.push(value);
      }
    });
    return result;
  }
  
  /*
   * Returns a new function that wraps `fn` with a timer. The wrapper logs the
   * time it takes to execute the function.
   */
  function time(name, fn) {
    let start = Date.now();
    try {
      return fn();
    } finally {
      console.log(name + " time: " + (Date.now() - start) + "ms");
    }
  }
  
  function notime(name, fn) {
    return fn();
  }
  
  let idCounter = 0;
  function uniqueId(prefix) {
    var id = ++idCounter;
    return toString(prefix) + id;
  }
  
  function range(start, limit, step = 1) {
    if (limit == null) {
      limit = start;
      start = 0;
    }
  
    let endCon = (i) => i < limit;
    if (step < 0) {
      endCon = (i) => limit < i;
    }
  
    const range = [];
    for (let i = start; endCon(i); i += step) {
      range.push(i);
    }
  
    return range;
  }
  
  function pick(source, keys) {
    const dest = {};
    for (const key of keys) {
      if (source[key] !== undefined) {
        dest[key] = source[key];
      }
    }
  
    return dest;
  }
  
  function mapValues(obj, funcOrProp) {
    let func = funcOrProp;
    if (typeof funcOrProp === 'string') {
      func = (val) => val[funcOrProp];
    }
  
    return Object.entries(obj).reduce((acc, [k, v]) => {
      acc[k] = func(v, k);
      return acc;
    }, {});
  }
  
  function zipObject(props, values) {
    return props.reduce((acc, key, i) => {
      acc[key] = values[i];
      return acc;
    }, {});
  }
  
  },{"@dagrejs/graphlib":29}],28:[function(require,module,exports){
  module.exports = "1.1.1";
  
  },{}],29:[function(require,module,exports){
  var lib = require("./lib");
  
  // dagre.
  
  module.exports = {
    Graph: lib.Graph,
    json: require("./lib/json"),
    alg: require("./lib/alg"),
    version: lib.version
  };
  
  },{"./lib":45,"./lib/alg":36,"./lib/json":46}],30:[function(require,module,exports){
  module.exports = components;
  
  function components(g) {
    var visited = {};
    var cmpts = [];
    var cmpt;
  
    function dfs(v) {
      if (visited.hasOwnProperty(v)) return;
      visited[v] = true;
      cmpt.push(v);
      g.successors(v).forEach(dfs);
      g.predecessors(v).forEach(dfs);
    }
  
    g.nodes().forEach(function(v) {
      cmpt = [];
      dfs(v);
      if (cmpt.length) {
        cmpts.push(cmpt);
      }
    });
  
    return cmpts;
  }
  
  },{}],31:[function(require,module,exports){
  module.exports = dfs;
  
  /*
   * A helper that preforms a pre- or post-order traversal on the input graph
   * and returns the nodes in the order they were visited. If the graph is
   * undirected then this algorithm will navigate using neighbors. If the graph
   * is directed then this algorithm will navigate using successors.
   *
   * If the order is not "post", it will be treated as "pre".
   */
  function dfs(g, vs, order) {
    if (!Array.isArray(vs)) {
      vs = [vs];
    }
  
    var navigation = g.isDirected() ? v => g.successors(v) : v => g.neighbors(v);
    var orderFunc = order === "post" ? postOrderDfs : preOrderDfs;
  
    var acc = [];
    var visited = {};
    vs.forEach(v => {
      if (!g.hasNode(v)) {
        throw new Error("Graph does not have node: " + v);
      }
  
      orderFunc(v, navigation, visited, acc);
    });
  
    return acc;
  }
  
  function postOrderDfs(v, navigation, visited, acc) {
    var stack = [[v, false]];
    while (stack.length > 0) {
      var curr = stack.pop();
      if (curr[1]) {
        acc.push(curr[0]);
      } else {
        if (!visited.hasOwnProperty(curr[0])) {
          visited[curr[0]] = true;
          stack.push([curr[0], true]);
          forEachRight(navigation(curr[0]), w => stack.push([w, false]));
        }
      }
    }
  }
  
  function preOrderDfs(v, navigation, visited, acc) {
    var stack = [v];
    while (stack.length > 0) {
      var curr = stack.pop();
      if (!visited.hasOwnProperty(curr)) {
        visited[curr] = true;
        acc.push(curr);
        forEachRight(navigation(curr), w => stack.push(w));
      }
    }
  }
  
  function forEachRight(array, iteratee) {
    var length = array.length;
    while (length--) {
      iteratee(array[length], length, array);
    }
  
    return array;
  }
  
  },{}],32:[function(require,module,exports){
  var dijkstra = require("./dijkstra");
  
  module.exports = dijkstraAll;
  
  function dijkstraAll(g, weightFunc, edgeFunc) {
    return g.nodes().reduce(function(acc, v) {
      acc[v] = dijkstra(g, v, weightFunc, edgeFunc);
      return acc;
    }, {});
  }
  
  },{"./dijkstra":33}],33:[function(require,module,exports){
  var PriorityQueue = require("../data/priority-queue");
  
  module.exports = dijkstra;
  
  var DEFAULT_WEIGHT_FUNC = () => 1;
  
  function dijkstra(g, source, weightFn, edgeFn) {
    return runDijkstra(g, String(source),
      weightFn || DEFAULT_WEIGHT_FUNC,
      edgeFn || function(v) { return g.outEdges(v); });
  }
  
  function runDijkstra(g, source, weightFn, edgeFn) {
    var results = {};
    var pq = new PriorityQueue();
    var v, vEntry;
  
    var updateNeighbors = function(edge) {
      var w = edge.v !== v ? edge.v : edge.w;
      var wEntry = results[w];
      var weight = weightFn(edge);
      var distance = vEntry.distance + weight;
  
      if (weight < 0) {
        throw new Error("dijkstra does not allow negative edge weights. " +
                        "Bad edge: " + edge + " Weight: " + weight);
      }
  
      if (distance < wEntry.distance) {
        wEntry.distance = distance;
        wEntry.predecessor = v;
        pq.decrease(w, distance);
      }
    };
  
    g.nodes().forEach(function(v) {
      var distance = v === source ? 0 : Number.POSITIVE_INFINITY;
      results[v] = { distance: distance };
      pq.add(v, distance);
    });
  
    while (pq.size() > 0) {
      v = pq.removeMin();
      vEntry = results[v];
      if (vEntry.distance === Number.POSITIVE_INFINITY) {
        break;
      }
  
      edgeFn(v).forEach(updateNeighbors);
    }
  
    return results;
  }
  
  },{"../data/priority-queue":43}],34:[function(require,module,exports){
  var tarjan = require("./tarjan");
  
  module.exports = findCycles;
  
  function findCycles(g) {
    return tarjan(g).filter(function(cmpt) {
      return cmpt.length > 1 || (cmpt.length === 1 && g.hasEdge(cmpt[0], cmpt[0]));
    });
  }
  
  },{"./tarjan":41}],35:[function(require,module,exports){
  module.exports = floydWarshall;
  
  var DEFAULT_WEIGHT_FUNC = () => 1;
  
  function floydWarshall(g, weightFn, edgeFn) {
    return runFloydWarshall(g,
      weightFn || DEFAULT_WEIGHT_FUNC,
      edgeFn || function(v) { return g.outEdges(v); });
  }
  
  function runFloydWarshall(g, weightFn, edgeFn) {
    var results = {};
    var nodes = g.nodes();
  
    nodes.forEach(function(v) {
      results[v] = {};
      results[v][v] = { distance: 0 };
      nodes.forEach(function(w) {
        if (v !== w) {
          results[v][w] = { distance: Number.POSITIVE_INFINITY };
        }
      });
      edgeFn(v).forEach(function(edge) {
        var w = edge.v === v ? edge.w : edge.v;
        var d = weightFn(edge);
        results[v][w] = { distance: d, predecessor: v };
      });
    });
  
    nodes.forEach(function(k) {
      var rowK = results[k];
      nodes.forEach(function(i) {
        var rowI = results[i];
        nodes.forEach(function(j) {
          var ik = rowI[k];
          var kj = rowK[j];
          var ij = rowI[j];
          var altDistance = ik.distance + kj.distance;
          if (altDistance < ij.distance) {
            ij.distance = altDistance;
            ij.predecessor = kj.predecessor;
          }
        });
      });
    });
  
    return results;
  }
  
  },{}],36:[function(require,module,exports){
  module.exports = {
    components: require("./components"),
    dijkstra: require("./dijkstra"),
    dijkstraAll: require("./dijkstra-all"),
    findCycles: require("./find-cycles"),
    floydWarshall: require("./floyd-warshall"),
    isAcyclic: require("./is-acyclic"),
    postorder: require("./postorder"),
    preorder: require("./preorder"),
    prim: require("./prim"),
    tarjan: require("./tarjan"),
    topsort: require("./topsort")
  };
  
  },{"./components":30,"./dijkstra":33,"./dijkstra-all":32,"./find-cycles":34,"./floyd-warshall":35,"./is-acyclic":37,"./postorder":38,"./preorder":39,"./prim":40,"./tarjan":41,"./topsort":42}],37:[function(require,module,exports){
  var topsort = require("./topsort");
  
  module.exports = isAcyclic;
  
  function isAcyclic(g) {
    try {
      topsort(g);
    } catch (e) {
      if (e instanceof topsort.CycleException) {
        return false;
      }
      throw e;
    }
    return true;
  }
  
  },{"./topsort":42}],38:[function(require,module,exports){
  var dfs = require("./dfs");
  
  module.exports = postorder;
  
  function postorder(g, vs) {
    return dfs(g, vs, "post");
  }
  
  },{"./dfs":31}],39:[function(require,module,exports){
  var dfs = require("./dfs");
  
  module.exports = preorder;
  
  function preorder(g, vs) {
    return dfs(g, vs, "pre");
  }
  
  },{"./dfs":31}],40:[function(require,module,exports){
  var Graph = require("../graph");
  var PriorityQueue = require("../data/priority-queue");
  
  module.exports = prim;
  
  function prim(g, weightFunc) {
    var result = new Graph();
    var parents = {};
    var pq = new PriorityQueue();
    var v;
  
    function updateNeighbors(edge) {
      var w = edge.v === v ? edge.w : edge.v;
      var pri = pq.priority(w);
      if (pri !== undefined) {
        var edgeWeight = weightFunc(edge);
        if (edgeWeight < pri) {
          parents[w] = v;
          pq.decrease(w, edgeWeight);
        }
      }
    }
  
    if (g.nodeCount() === 0) {
      return result;
    }
  
    g.nodes().forEach(function(v) {
      pq.add(v, Number.POSITIVE_INFINITY);
      result.setNode(v);
    });
  
    // Start from an arbitrary node
    pq.decrease(g.nodes()[0], 0);
  
    var init = false;
    while (pq.size() > 0) {
      v = pq.removeMin();
      if (parents.hasOwnProperty(v)) {
        result.setEdge(v, parents[v]);
      } else if (init) {
        throw new Error("Input graph is not connected: " + g);
      } else {
        init = true;
      }
  
      g.nodeEdges(v).forEach(updateNeighbors);
    }
  
    return result;
  }
  
  },{"../data/priority-queue":43,"../graph":44}],41:[function(require,module,exports){
  module.exports = tarjan;
  
  function tarjan(g) {
    var index = 0;
    var stack = [];
    var visited = {}; // node id -> { onStack, lowlink, index }
    var results = [];
  
    function dfs(v) {
      var entry = visited[v] = {
        onStack: true,
        lowlink: index,
        index: index++
      };
      stack.push(v);
  
      g.successors(v).forEach(function(w) {
        if (!visited.hasOwnProperty(w)) {
          dfs(w);
          entry.lowlink = Math.min(entry.lowlink, visited[w].lowlink);
        } else if (visited[w].onStack) {
          entry.lowlink = Math.min(entry.lowlink, visited[w].index);
        }
      });
  
      if (entry.lowlink === entry.index) {
        var cmpt = [];
        var w;
        do {
          w = stack.pop();
          visited[w].onStack = false;
          cmpt.push(w);
        } while (v !== w);
        results.push(cmpt);
      }
    }
  
    g.nodes().forEach(function(v) {
      if (!visited.hasOwnProperty(v)) {
        dfs(v);
      }
    });
  
    return results;
  }
  
  },{}],42:[function(require,module,exports){
  function topsort(g) {
    var visited = {};
    var stack = {};
    var results = [];
  
    function visit(node) {
      if (stack.hasOwnProperty(node)) {
        throw new CycleException();
      }
  
      if (!visited.hasOwnProperty(node)) {
        stack[node] = true;
        visited[node] = true;
        g.predecessors(node).forEach(visit);
        delete stack[node];
        results.push(node);
      }
    }
  
    g.sinks().forEach(visit);
  
    if (Object.keys(visited).length !== g.nodeCount()) {
      throw new CycleException();
    }
  
    return results;
  }
  
  class CycleException extends Error {
    constructor() {
      super(...arguments);
    }
  }
  
  module.exports = topsort;
  topsort.CycleException = CycleException;
  
  },{}],43:[function(require,module,exports){
  /**
   * A min-priority queue data structure. This algorithm is derived from Cormen,
   * et al., "Introduction to Algorithms". The basic idea of a min-priority
   * queue is that you can efficiently (in O(1) time) get the smallest key in
   * the queue. Adding and removing elements takes O(log n) time. A key can
   * have its priority decreased in O(log n) time.
   */
  class PriorityQueue {
    #arr = [];
    #keyIndices = {};
  
    /**
     * Returns the number of elements in the queue. Takes `O(1)` time.
     */
    size() {
      return this.#arr.length;
    }
  
    /**
     * Returns the keys that are in the queue. Takes `O(n)` time.
     */
    keys() {
      return this.#arr.map(function(x) { return x.key; });
    }
  
    /**
     * Returns `true` if **key** is in the queue and `false` if not.
     */
    has(key) {
      return this.#keyIndices.hasOwnProperty(key);
    }
  
    /**
     * Returns the priority for **key**. If **key** is not present in the queue
     * then this function returns `undefined`. Takes `O(1)` time.
     *
     * @param {Object} key
     */
    priority(key) {
      var index = this.#keyIndices[key];
      if (index !== undefined) {
        return this.#arr[index].priority;
      }
    }
  
    /**
     * Returns the key for the minimum element in this queue. If the queue is
     * empty this function throws an Error. Takes `O(1)` time.
     */
    min() {
      if (this.size() === 0) {
        throw new Error("Queue underflow");
      }
      return this.#arr[0].key;
    }
  
    /**
     * Inserts a new key into the priority queue. If the key already exists in
     * the queue this function returns `false`; otherwise it will return `true`.
     * Takes `O(n)` time.
     *
     * @param {Object} key the key to add
     * @param {Number} priority the initial priority for the key
     */
    add(key, priority) {
      var keyIndices = this.#keyIndices;
      key = String(key);
      if (!keyIndices.hasOwnProperty(key)) {
        var arr = this.#arr;
        var index = arr.length;
        keyIndices[key] = index;
        arr.push({key: key, priority: priority});
        this.#decrease(index);
        return true;
      }
      return false;
    }
  
    /**
     * Removes and returns the smallest key in the queue. Takes `O(log n)` time.
     */
    removeMin() {
      this.#swap(0, this.#arr.length - 1);
      var min = this.#arr.pop();
      delete this.#keyIndices[min.key];
      this.#heapify(0);
      return min.key;
    }
  
    /**
     * Decreases the priority for **key** to **priority**. If the new priority is
     * greater than the previous priority, this function will throw an Error.
     *
     * @param {Object} key the key for which to raise priority
     * @param {Number} priority the new priority for the key
     */
    decrease(key, priority) {
      var index = this.#keyIndices[key];
      if (priority > this.#arr[index].priority) {
        throw new Error("New priority is greater than current priority. " +
            "Key: " + key + " Old: " + this.#arr[index].priority + " New: " + priority);
      }
      this.#arr[index].priority = priority;
      this.#decrease(index);
    }
  
    #heapify(i) {
      var arr = this.#arr;
      var l = 2 * i;
      var r = l + 1;
      var largest = i;
      if (l < arr.length) {
        largest = arr[l].priority < arr[largest].priority ? l : largest;
        if (r < arr.length) {
          largest = arr[r].priority < arr[largest].priority ? r : largest;
        }
        if (largest !== i) {
          this.#swap(i, largest);
          this.#heapify(largest);
        }
      }
    }
  
    #decrease(index) {
      var arr = this.#arr;
      var priority = arr[index].priority;
      var parent;
      while (index !== 0) {
        parent = index >> 1;
        if (arr[parent].priority < priority) {
          break;
        }
        this.#swap(index, parent);
        index = parent;
      }
    }
  
    #swap(i, j) {
      var arr = this.#arr;
      var keyIndices = this.#keyIndices;
      var origArrI = arr[i];
      var origArrJ = arr[j];
      arr[i] = origArrJ;
      arr[j] = origArrI;
      keyIndices[origArrJ.key] = i;
      keyIndices[origArrI.key] = j;
    }
  }
  
  module.exports = PriorityQueue;
  
  },{}],44:[function(require,module,exports){
  "use strict";
  
  var DEFAULT_EDGE_NAME = "\x00";
  var GRAPH_NODE = "\x00";
  var EDGE_KEY_DELIM = "\x01";
  
  // Implementation notes:
  //
  //  * Node id query functions should return string ids for the nodes
  //  * Edge id query functions should return an "edgeObj", edge object, that is
  //    composed of enough information to uniquely identify an edge: {v, w, name}.
  //  * Internally we use an "edgeId", a stringified form of the edgeObj, to
  //    reference edges. This is because we need a performant way to look these
  //    edges up and, object properties, which have string keys, are the closest
  //    we're going to get to a performant hashtable in JavaScript.
  
  class Graph {
    #isDirected = true;
    #isMultigraph = false;
    #isCompound = false;
  
    // Label for the graph itself
    #label;
  
    // Defaults to be set when creating a new node
    #defaultNodeLabelFn = () => undefined;
  
    // Defaults to be set when creating a new edge
    #defaultEdgeLabelFn = () => undefined;
  
    // v -> label
    #nodes = {};
  
    // v -> edgeObj
    #in = {};
  
    // u -> v -> Number
    #preds = {};
  
    // v -> edgeObj
    #out = {};
  
    // v -> w -> Number
    #sucs = {};
  
    // e -> edgeObj
    #edgeObjs = {};
  
    // e -> label
    #edgeLabels = {};
  
    /* Number of nodes in the graph. Should only be changed by the implementation. */
    #nodeCount = 0;
  
    /* Number of edges in the graph. Should only be changed by the implementation. */
    #edgeCount = 0;
  
    #parent;
  
    #children;
  
    constructor(opts) {
      if (opts) {
        this.#isDirected = opts.hasOwnProperty("directed") ? opts.directed : true;
        this.#isMultigraph = opts.hasOwnProperty("multigraph") ? opts.multigraph : false;
        this.#isCompound = opts.hasOwnProperty("compound") ? opts.compound : false;
      }
  
      if (this.#isCompound) {
        // v -> parent
        this.#parent = {};
  
        // v -> children
        this.#children = {};
        this.#children[GRAPH_NODE] = {};
      }
    }
  
    /* === Graph functions ========= */
  
    /**
     * Whether graph was created with 'directed' flag set to true or not.
     */
    isDirected() {
      return this.#isDirected;
    }
  
    /**
     * Whether graph was created with 'multigraph' flag set to true or not.
     */
    isMultigraph() {
      return this.#isMultigraph;
    }
  
    /**
     * Whether graph was created with 'compound' flag set to true or not.
     */
    isCompound() {
      return this.#isCompound;
    }
  
    /**
     * Sets the label of the graph.
     */
    setGraph(label) {
      this.#label = label;
      return this;
    }
  
    /**
     * Gets the graph label.
     */
    graph() {
      return this.#label;
    }
  
  
    /* === Node functions ========== */
  
    /**
     * Sets the default node label. If newDefault is a function, it will be
     * invoked ach time when setting a label for a node. Otherwise, this label
     * will be assigned as default label in case if no label was specified while
     * setting a node.
     * Complexity: O(1).
     */
    setDefaultNodeLabel(newDefault) {
      this.#defaultNodeLabelFn = newDefault;
      if (typeof newDefault !== 'function') {
        this.#defaultNodeLabelFn = () => newDefault;
      }
  
      return this;
    }
  
    /**
     * Gets the number of nodes in the graph.
     * Complexity: O(1).
     */
    nodeCount() {
      return this.#nodeCount;
    }
  
    /**
     * Gets all nodes of the graph. Note, the in case of compound graph subnodes are
     * not included in list.
     * Complexity: O(1).
     */
    nodes() {
      return Object.keys(this.#nodes);
    }
  
    /**
     * Gets list of nodes without in-edges.
     * Complexity: O(|V|).
     */
    sources() {
      var self = this;
      return this.nodes().filter(v => Object.keys(self.#in[v]).length === 0);
    }
  
    /**
     * Gets list of nodes without out-edges.
     * Complexity: O(|V|).
     */
    sinks() {
      var self = this;
      return this.nodes().filter(v => Object.keys(self.#out[v]).length === 0);
    }
  
    /**
     * Invokes setNode method for each node in names list.
     * Complexity: O(|names|).
     */
    setNodes(vs, value) {
      var args = arguments;
      var self = this;
      vs.forEach(function(v) {
        if (args.length > 1) {
          self.setNode(v, value);
        } else {
          self.setNode(v);
        }
      });
      return this;
    }
  
    /**
     * Creates or updates the value for the node v in the graph. If label is supplied
     * it is set as the value for the node. If label is not supplied and the node was
     * created by this call then the default node label will be assigned.
     * Complexity: O(1).
     */
    setNode(v, value) {
      if (this.#nodes.hasOwnProperty(v)) {
        if (arguments.length > 1) {
          this.#nodes[v] = value;
        }
        return this;
      }
  
      this.#nodes[v] = arguments.length > 1 ? value : this.#defaultNodeLabelFn(v);
      if (this.#isCompound) {
        this.#parent[v] = GRAPH_NODE;
        this.#children[v] = {};
        this.#children[GRAPH_NODE][v] = true;
      }
      this.#in[v] = {};
      this.#preds[v] = {};
      this.#out[v] = {};
      this.#sucs[v] = {};
      ++this.#nodeCount;
      return this;
    }
  
    /**
     * Gets the label of node with specified name.
     * Complexity: O(|V|).
     */
    node(v) {
      return this.#nodes[v];
    }
  
    /**
     * Detects whether graph has a node with specified name or not.
     */
    hasNode(v) {
      return this.#nodes.hasOwnProperty(v);
    }
  
    /**
     * Remove the node with the name from the graph or do nothing if the node is not in
     * the graph. If the node was removed this function also removes any incident
     * edges.
     * Complexity: O(1).
     */
    removeNode(v) {
      var self = this;
      if (this.#nodes.hasOwnProperty(v)) {
        var removeEdge = e => self.removeEdge(self.#edgeObjs[e]);
        delete this.#nodes[v];
        if (this.#isCompound) {
          this.#removeFromParentsChildList(v);
          delete this.#parent[v];
          this.children(v).forEach(function(child) {
            self.setParent(child);
          });
          delete this.#children[v];
        }
        Object.keys(this.#in[v]).forEach(removeEdge);
        delete this.#in[v];
        delete this.#preds[v];
        Object.keys(this.#out[v]).forEach(removeEdge);
        delete this.#out[v];
        delete this.#sucs[v];
        --this.#nodeCount;
      }
      return this;
    }
  
    /**
     * Sets node p as a parent for node v if it is defined, or removes the
     * parent for v if p is undefined. Method throws an exception in case of
     * invoking it in context of noncompound graph.
     * Average-case complexity: O(1).
     */
    setParent(v, parent) {
      if (!this.#isCompound) {
        throw new Error("Cannot set parent in a non-compound graph");
      }
  
      if (parent === undefined) {
        parent = GRAPH_NODE;
      } else {
        // Coerce parent to string
        parent += "";
        for (var ancestor = parent; ancestor !== undefined; ancestor = this.parent(ancestor)) {
          if (ancestor === v) {
            throw new Error("Setting " + parent+ " as parent of " + v +
                " would create a cycle");
          }
        }
  
        this.setNode(parent);
      }
  
      this.setNode(v);
      this.#removeFromParentsChildList(v);
      this.#parent[v] = parent;
      this.#children[parent][v] = true;
      return this;
    }
  
    #removeFromParentsChildList(v) {
      delete this.#children[this.#parent[v]][v];
    }
  
    /**
     * Gets parent node for node v.
     * Complexity: O(1).
     */
    parent(v) {
      if (this.#isCompound) {
        var parent = this.#parent[v];
        if (parent !== GRAPH_NODE) {
          return parent;
        }
      }
    }
  
    /**
     * Gets list of direct children of node v.
     * Complexity: O(1).
     */
    children(v = GRAPH_NODE) {
      if (this.#isCompound) {
        var children = this.#children[v];
        if (children) {
          return Object.keys(children);
        }
      } else if (v === GRAPH_NODE) {
        return this.nodes();
      } else if (this.hasNode(v)) {
        return [];
      }
    }
  
    /**
     * Return all nodes that are predecessors of the specified node or undefined if node v is not in
     * the graph. Behavior is undefined for undirected graphs - use neighbors instead.
     * Complexity: O(|V|).
     */
    predecessors(v) {
      var predsV = this.#preds[v];
      if (predsV) {
        return Object.keys(predsV);
      }
    }
  
    /**
     * Return all nodes that are successors of the specified node or undefined if node v is not in
     * the graph. Behavior is undefined for undirected graphs - use neighbors instead.
     * Complexity: O(|V|).
     */
    successors(v) {
      var sucsV = this.#sucs[v];
      if (sucsV) {
        return Object.keys(sucsV);
      }
    }
  
    /**
     * Return all nodes that are predecessors or successors of the specified node or undefined if
     * node v is not in the graph.
     * Complexity: O(|V|).
     */
    neighbors(v) {
      var preds = this.predecessors(v);
      if (preds) {
        const union = new Set(preds);
        for (var succ of this.successors(v)) {
          union.add(succ);
        }
  
        return Array.from(union.values());
      }
    }
  
    isLeaf(v) {
      var neighbors;
      if (this.isDirected()) {
        neighbors = this.successors(v);
      } else {
        neighbors = this.neighbors(v);
      }
      return neighbors.length === 0;
    }
  
    /**
     * Creates new graph with nodes filtered via filter. Edges incident to rejected node
     * are also removed. In case of compound graph, if parent is rejected by filter,
     * than all its children are rejected too.
     * Average-case complexity: O(|E|+|V|).
     */
    filterNodes(filter) {
      var copy = new this.constructor({
        directed: this.#isDirected,
        multigraph: this.#isMultigraph,
        compound: this.#isCompound
      });
  
      copy.setGraph(this.graph());
  
      var self = this;
      Object.entries(this.#nodes).forEach(function([v, value]) {
        if (filter(v)) {
          copy.setNode(v, value);
        }
      });
  
      Object.values(this.#edgeObjs).forEach(function(e) {
        if (copy.hasNode(e.v) && copy.hasNode(e.w)) {
          copy.setEdge(e, self.edge(e));
        }
      });
  
      var parents = {};
      function findParent(v) {
        var parent = self.parent(v);
        if (parent === undefined || copy.hasNode(parent)) {
          parents[v] = parent;
          return parent;
        } else if (parent in parents) {
          return parents[parent];
        } else {
          return findParent(parent);
        }
      }
  
      if (this.#isCompound) {
        copy.nodes().forEach(v => copy.setParent(v, findParent(v)));
      }
  
      return copy;
    }
  
    /* === Edge functions ========== */
  
    /**
     * Sets the default edge label or factory function. This label will be
     * assigned as default label in case if no label was specified while setting
     * an edge or this function will be invoked each time when setting an edge
     * with no label specified and returned value * will be used as a label for edge.
     * Complexity: O(1).
     */
    setDefaultEdgeLabel(newDefault) {
      this.#defaultEdgeLabelFn = newDefault;
      if (typeof newDefault !== 'function') {
        this.#defaultEdgeLabelFn = () => newDefault;
      }
  
      return this;
    }
  
    /**
     * Gets the number of edges in the graph.
     * Complexity: O(1).
     */
    edgeCount() {
      return this.#edgeCount;
    }
  
    /**
     * Gets edges of the graph. In case of compound graph subgraphs are not considered.
     * Complexity: O(|E|).
     */
    edges() {
      return Object.values(this.#edgeObjs);
    }
  
    /**
     * Establish an edges path over the nodes in nodes list. If some edge is already
     * exists, it will update its label, otherwise it will create an edge between pair
     * of nodes with label provided or default label if no label provided.
     * Complexity: O(|nodes|).
     */
    setPath(vs, value) {
      var self = this;
      var args = arguments;
      vs.reduce(function(v, w) {
        if (args.length > 1) {
          self.setEdge(v, w, value);
        } else {
          self.setEdge(v, w);
        }
        return w;
      });
      return this;
    }
  
    /**
     * Creates or updates the label for the edge (v, w) with the optionally supplied
     * name. If label is supplied it is set as the value for the edge. If label is not
     * supplied and the edge was created by this call then the default edge label will
     * be assigned. The name parameter is only useful with multigraphs.
     */
    setEdge() {
      var v, w, name, value;
      var valueSpecified = false;
      var arg0 = arguments[0];
  
      if (typeof arg0 === "object" && arg0 !== null && "v" in arg0) {
        v = arg0.v;
        w = arg0.w;
        name = arg0.name;
        if (arguments.length === 2) {
          value = arguments[1];
          valueSpecified = true;
        }
      } else {
        v = arg0;
        w = arguments[1];
        name = arguments[3];
        if (arguments.length > 2) {
          value = arguments[2];
          valueSpecified = true;
        }
      }
  
      v = "" + v;
      w = "" + w;
      if (name !== undefined) {
        name = "" + name;
      }
  
      var e = edgeArgsToId(this.#isDirected, v, w, name);
      if (this.#edgeLabels.hasOwnProperty(e)) {
        if (valueSpecified) {
          this.#edgeLabels[e] = value;
        }
        return this;
      }
  
      if (name !== undefined && !this.#isMultigraph) {
        throw new Error("Cannot set a named edge when isMultigraph = false");
      }
  
      // It didn't exist, so we need to create it.
      // First ensure the nodes exist.
      this.setNode(v);
      this.setNode(w);
  
      this.#edgeLabels[e] = valueSpecified ? value : this.#defaultEdgeLabelFn(v, w, name);
  
      var edgeObj = edgeArgsToObj(this.#isDirected, v, w, name);
      // Ensure we add undirected edges in a consistent way.
      v = edgeObj.v;
      w = edgeObj.w;
  
      Object.freeze(edgeObj);
      this.#edgeObjs[e] = edgeObj;
      incrementOrInitEntry(this.#preds[w], v);
      incrementOrInitEntry(this.#sucs[v], w);
      this.#in[w][e] = edgeObj;
      this.#out[v][e] = edgeObj;
      this.#edgeCount++;
      return this;
    }
  
    /**
     * Gets the label for the specified edge.
     * Complexity: O(1).
     */
    edge(v, w, name) {
      var e = (arguments.length === 1
        ? edgeObjToId(this.#isDirected, arguments[0])
        : edgeArgsToId(this.#isDirected, v, w, name));
      return this.#edgeLabels[e];
    }
  
    /**
     * Gets the label for the specified edge and converts it to an object.
     * Complexity: O(1)
     */
    edgeAsObj() {
      const edge = this.edge(...arguments);
      if (typeof edge !== "object") {
        return {label: edge};
      }
  
      return edge;
    }
  
    /**
     * Detects whether the graph contains specified edge or not. No subgraphs are considered.
     * Complexity: O(1).
     */
    hasEdge(v, w, name) {
      var e = (arguments.length === 1
        ? edgeObjToId(this.#isDirected, arguments[0])
        : edgeArgsToId(this.#isDirected, v, w, name));
      return this.#edgeLabels.hasOwnProperty(e);
    }
  
    /**
     * Removes the specified edge from the graph. No subgraphs are considered.
     * Complexity: O(1).
     */
    removeEdge(v, w, name) {
      var e = (arguments.length === 1
        ? edgeObjToId(this.#isDirected, arguments[0])
        : edgeArgsToId(this.#isDirected, v, w, name));
      var edge = this.#edgeObjs[e];
      if (edge) {
        v = edge.v;
        w = edge.w;
        delete this.#edgeLabels[e];
        delete this.#edgeObjs[e];
        decrementOrRemoveEntry(this.#preds[w], v);
        decrementOrRemoveEntry(this.#sucs[v], w);
        delete this.#in[w][e];
        delete this.#out[v][e];
        this.#edgeCount--;
      }
      return this;
    }
  
    /**
     * Return all edges that point to the node v. Optionally filters those edges down to just those
     * coming from node u. Behavior is undefined for undirected graphs - use nodeEdges instead.
     * Complexity: O(|E|).
     */
    inEdges(v, u) {
      var inV = this.#in[v];
      if (inV) {
        var edges = Object.values(inV);
        if (!u) {
          return edges;
        }
        return edges.filter(edge => edge.v === u);
      }
    }
  
    /**
     * Return all edges that are pointed at by node v. Optionally filters those edges down to just
     * those point to w. Behavior is undefined for undirected graphs - use nodeEdges instead.
     * Complexity: O(|E|).
     */
    outEdges(v, w) {
      var outV = this.#out[v];
      if (outV) {
        var edges = Object.values(outV);
        if (!w) {
          return edges;
        }
        return edges.filter(edge => edge.w === w);
      }
    }
  
    /**
     * Returns all edges to or from node v regardless of direction. Optionally filters those edges
     * down to just those between nodes v and w regardless of direction.
     * Complexity: O(|E|).
     */
    nodeEdges(v, w) {
      var inEdges = this.inEdges(v, w);
      if (inEdges) {
        return inEdges.concat(this.outEdges(v, w));
      }
    }
  }
  
  function incrementOrInitEntry(map, k) {
    if (map[k]) {
      map[k]++;
    } else {
      map[k] = 1;
    }
  }
  
  function decrementOrRemoveEntry(map, k) {
    if (!--map[k]) { delete map[k]; }
  }
  
  function edgeArgsToId(isDirected, v_, w_, name) {
    var v = "" + v_;
    var w = "" + w_;
    if (!isDirected && v > w) {
      var tmp = v;
      v = w;
      w = tmp;
    }
    return v + EDGE_KEY_DELIM + w + EDGE_KEY_DELIM +
               (name === undefined ? DEFAULT_EDGE_NAME : name);
  }
  
  function edgeArgsToObj(isDirected, v_, w_, name) {
    var v = "" + v_;
    var w = "" + w_;
    if (!isDirected && v > w) {
      var tmp = v;
      v = w;
      w = tmp;
    }
    var edgeObj =  { v: v, w: w };
    if (name) {
      edgeObj.name = name;
    }
    return edgeObj;
  }
  
  function edgeObjToId(isDirected, edgeObj) {
    return edgeArgsToId(isDirected, edgeObj.v, edgeObj.w, edgeObj.name);
  }
  
  module.exports = Graph;
  
  },{}],45:[function(require,module,exports){
  // Includes only the "core" of graphlib
  module.exports = {
    Graph: require("./graph"),
    version: require("./version")
  };
  
  },{"./graph":44,"./version":47}],46:[function(require,module,exports){
  var Graph = require("./graph");
  
  module.exports = {
    write: write,
    read: read
  };
  
  /**
   * Creates a JSON representation of the graph that can be serialized to a string with
   * JSON.stringify. The graph can later be restored using json.read.
   */
  function write(g) {
    var json = {
      options: {
        directed: g.isDirected(),
        multigraph: g.isMultigraph(),
        compound: g.isCompound()
      },
      nodes: writeNodes(g),
      edges: writeEdges(g)
    };
  
    if (g.graph() !== undefined) {
      json.value = structuredClone(g.graph());
    }
    return json;
  }
  
  function writeNodes(g) {
    return g.nodes().map(function(v) {
      var nodeValue = g.node(v);
      var parent = g.parent(v);
      var node = { v: v };
      if (nodeValue !== undefined) {
        node.value = nodeValue;
      }
      if (parent !== undefined) {
        node.parent = parent;
      }
      return node;
    });
  }
  
  function writeEdges(g) {
    return g.edges().map(function(e) {
      var edgeValue = g.edge(e);
      var edge = { v: e.v, w: e.w };
      if (e.name !== undefined) {
        edge.name = e.name;
      }
      if (edgeValue !== undefined) {
        edge.value = edgeValue;
      }
      return edge;
    });
  }
  
  /**
   * Takes JSON as input and returns the graph representation.
   *
   * @example
   * var g2 = graphlib.json.read(JSON.parse(str));
   * g2.nodes();
   * // ['a', 'b']
   * g2.edges()
   * // [ { v: 'a', w: 'b' } ]
   */
  function read(json) {
    var g = new Graph(json.options).setGraph(json.value);
    json.nodes.forEach(function(entry) {
      g.setNode(entry.v, entry.value);
      if (entry.parent) {
        g.setParent(entry.v, entry.parent);
      }
    });
    json.edges.forEach(function(entry) {
      g.setEdge({ v: entry.v, w: entry.w, name: entry.name }, entry.value);
    });
    return g;
  }
  
  },{"./graph":44}],47:[function(require,module,exports){
  module.exports = '2.2.1';
  
  },{}]},{},[1])(1)
  });
  
  !function(t,e){"object"==typeof exports&&"undefined"!=typeof module?e(exports):"function"==typeof define&&define.amd?define(["exports"],e):e((t="undefined"!=typeof globalThis?globalThis:t||self).THREE={})}(this,(function(t){"use strict";const e="134",n=100,i=300,r=301,s=302,a=303,o=304,l=306,c=307,h=1e3,u=1001,d=1002,p=1003,m=1004,f=1005,g=1006,v=1007,y=1008,x=1009,_=1012,M=1014,b=1015,w=1016,S=1020,T=1022,E=1023,A=1026,L=1027,R=33776,C=33777,P=33778,I=33779,D=35840,N=35841,z=35842,B=35843,F=37492,O=37496,U=2300,H=2301,G=2302,k=2400,V=2401,W=2402,j=2500,q=2501,X=3e3,Y=3001,J=3007,Z=3002,Q=3004,K=3005,$=3006,tt=7680,et=35044,nt=35048,it="300 es";class rt{addEventListener(t,e){void 0===this._listeners&&(this._listeners={});const n=this._listeners;void 0===n[t]&&(n[t]=[]),-1===n[t].indexOf(e)&&n[t].push(e)}hasEventListener(t,e){if(void 0===this._listeners)return!1;const n=this._listeners;return void 0!==n[t]&&-1!==n[t].indexOf(e)}removeEventListener(t,e){if(void 0===this._listeners)return;const n=this._listeners[t];if(void 0!==n){const t=n.indexOf(e);-1!==t&&n.splice(t,1)}}dispatchEvent(t){if(void 0===this._listeners)return;const e=this._listeners[t.type];if(void 0!==e){t.target=this;const n=e.slice(0);for(let e=0,i=n.length;e<i;e++)n[e].call(this,t);t.target=null}}}let st=1234567;const at=Math.PI/180,ot=180/Math.PI,lt=[];for(let t=0;t<256;t++)lt[t]=(t<16?"0":"")+t.toString(16);const ct="undefined"!=typeof crypto&&"randomUUID"in crypto;function ht(){if(ct)return crypto.randomUUID().toUpperCase();const t=4294967295*Math.random()|0,e=4294967295*Math.random()|0,n=4294967295*Math.random()|0,i=4294967295*Math.random()|0;return(lt[255&t]+lt[t>>8&255]+lt[t>>16&255]+lt[t>>24&255]+"-"+lt[255&e]+lt[e>>8&255]+"-"+lt[e>>16&15|64]+lt[e>>24&255]+"-"+lt[63&n|128]+lt[n>>8&255]+"-"+lt[n>>16&255]+lt[n>>24&255]+lt[255&i]+lt[i>>8&255]+lt[i>>16&255]+lt[i>>24&255]).toUpperCase()}function ut(t,e,n){return Math.max(e,Math.min(n,t))}function dt(t,e){return(t%e+e)%e}function pt(t,e,n){return(1-n)*t+n*e}function mt(t){return 0==(t&t-1)&&0!==t}function ft(t){return Math.pow(2,Math.ceil(Math.log(t)/Math.LN2))}function gt(t){return Math.pow(2,Math.floor(Math.log(t)/Math.LN2))}var vt=Object.freeze({__proto__:null,DEG2RAD:at,RAD2DEG:ot,generateUUID:ht,clamp:ut,euclideanModulo:dt,mapLinear:function(t,e,n,i,r){return i+(t-e)*(r-i)/(n-e)},inverseLerp:function(t,e,n){return t!==e?(n-t)/(e-t):0},lerp:pt,damp:function(t,e,n,i){return pt(t,e,1-Math.exp(-n*i))},pingpong:function(t,e=1){return e-Math.abs(dt(t,2*e)-e)},smoothstep:function(t,e,n){return t<=e?0:t>=n?1:(t=(t-e)/(n-e))*t*(3-2*t)},smootherstep:function(t,e,n){return t<=e?0:t>=n?1:(t=(t-e)/(n-e))*t*t*(t*(6*t-15)+10)},randInt:function(t,e){return t+Math.floor(Math.random()*(e-t+1))},randFloat:function(t,e){return t+Math.random()*(e-t)},randFloatSpread:function(t){return t*(.5-Math.random())},seededRandom:function(t){return void 0!==t&&(st=t%2147483647),st=16807*st%2147483647,(st-1)/2147483646},degToRad:function(t){return t*at},radToDeg:function(t){return t*ot},isPowerOfTwo:mt,ceilPowerOfTwo:ft,floorPowerOfTwo:gt,setQuaternionFromProperEuler:function(t,e,n,i,r){const s=Math.cos,a=Math.sin,o=s(n/2),l=a(n/2),c=s((e+i)/2),h=a((e+i)/2),u=s((e-i)/2),d=a((e-i)/2),p=s((i-e)/2),m=a((i-e)/2);switch(r){case"XYX":t.set(o*h,l*u,l*d,o*c);break;case"YZY":t.set(l*d,o*h,l*u,o*c);break;case"ZXZ":t.set(l*u,l*d,o*h,o*c);break;case"XZX":t.set(o*h,l*m,l*p,o*c);break;case"YXY":t.set(l*p,o*h,l*m,o*c);break;case"ZYZ":t.set(l*m,l*p,o*h,o*c);break;default:console.warn("THREE.MathUtils: .setQuaternionFromProperEuler() encountered an unknown order: "+r)}}});class yt{constructor(t=0,e=0){this.x=t,this.y=e}get width(){return this.x}set width(t){this.x=t}get height(){return this.y}set height(t){this.y=t}set(t,e){return this.x=t,this.y=e,this}setScalar(t){return this.x=t,this.y=t,this}setX(t){return this.x=t,this}setY(t){return this.y=t,this}setComponent(t,e){switch(t){case 0:this.x=e;break;case 1:this.y=e;break;default:throw new Error("index is out of range: "+t)}return this}getComponent(t){switch(t){case 0:return this.x;case 1:return this.y;default:throw new Error("index is out of range: "+t)}}clone(){return new this.constructor(this.x,this.y)}copy(t){return this.x=t.x,this.y=t.y,this}add(t,e){return void 0!==e?(console.warn("THREE.Vector2: .add() now only accepts one argument. Use .addVectors( a, b ) instead."),this.addVectors(t,e)):(this.x+=t.x,this.y+=t.y,this)}addScalar(t){return this.x+=t,this.y+=t,this}addVectors(t,e){return this.x=t.x+e.x,this.y=t.y+e.y,this}addScaledVector(t,e){return this.x+=t.x*e,this.y+=t.y*e,this}sub(t,e){return void 0!==e?(console.warn("THREE.Vector2: .sub() now only accepts one argument. Use .subVectors( a, b ) instead."),this.subVectors(t,e)):(this.x-=t.x,this.y-=t.y,this)}subScalar(t){return this.x-=t,this.y-=t,this}subVectors(t,e){return this.x=t.x-e.x,this.y=t.y-e.y,this}multiply(t){return this.x*=t.x,this.y*=t.y,this}multiplyScalar(t){return this.x*=t,this.y*=t,this}divide(t){return this.x/=t.x,this.y/=t.y,this}divideScalar(t){return this.multiplyScalar(1/t)}applyMatrix3(t){const e=this.x,n=this.y,i=t.elements;return this.x=i[0]*e+i[3]*n+i[6],this.y=i[1]*e+i[4]*n+i[7],this}min(t){return this.x=Math.min(this.x,t.x),this.y=Math.min(this.y,t.y),this}max(t){return this.x=Math.max(this.x,t.x),this.y=Math.max(this.y,t.y),this}clamp(t,e){return this.x=Math.max(t.x,Math.min(e.x,this.x)),this.y=Math.max(t.y,Math.min(e.y,this.y)),this}clampScalar(t,e){return this.x=Math.max(t,Math.min(e,this.x)),this.y=Math.max(t,Math.min(e,this.y)),this}clampLength(t,e){const n=this.length();return this.divideScalar(n||1).multiplyScalar(Math.max(t,Math.min(e,n)))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this}roundToZero(){return this.x=this.x<0?Math.ceil(this.x):Math.floor(this.x),this.y=this.y<0?Math.ceil(this.y):Math.floor(this.y),this}negate(){return this.x=-this.x,this.y=-this.y,this}dot(t){return this.x*t.x+this.y*t.y}cross(t){return this.x*t.y-this.y*t.x}lengthSq(){return this.x*this.x+this.y*this.y}length(){return Math.sqrt(this.x*this.x+this.y*this.y)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)}normalize(){return this.divideScalar(this.length()||1)}angle(){return Math.atan2(-this.y,-this.x)+Math.PI}distanceTo(t){return Math.sqrt(this.distanceToSquared(t))}distanceToSquared(t){const e=this.x-t.x,n=this.y-t.y;return e*e+n*n}manhattanDistanceTo(t){return Math.abs(this.x-t.x)+Math.abs(this.y-t.y)}setLength(t){return this.normalize().multiplyScalar(t)}lerp(t,e){return this.x+=(t.x-this.x)*e,this.y+=(t.y-this.y)*e,this}lerpVectors(t,e,n){return this.x=t.x+(e.x-t.x)*n,this.y=t.y+(e.y-t.y)*n,this}equals(t){return t.x===this.x&&t.y===this.y}fromArray(t,e=0){return this.x=t[e],this.y=t[e+1],this}toArray(t=[],e=0){return t[e]=this.x,t[e+1]=this.y,t}fromBufferAttribute(t,e,n){return void 0!==n&&console.warn("THREE.Vector2: offset has been removed from .fromBufferAttribute()."),this.x=t.getX(e),this.y=t.getY(e),this}rotateAround(t,e){const n=Math.cos(e),i=Math.sin(e),r=this.x-t.x,s=this.y-t.y;return this.x=r*n-s*i+t.x,this.y=r*i+s*n+t.y,this}random(){return this.x=Math.random(),this.y=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y}}yt.prototype.isVector2=!0;class xt{constructor(){this.elements=[1,0,0,0,1,0,0,0,1],arguments.length>0&&console.error("THREE.Matrix3: the constructor no longer reads arguments. use .set() instead.")}set(t,e,n,i,r,s,a,o,l){const c=this.elements;return c[0]=t,c[1]=i,c[2]=a,c[3]=e,c[4]=r,c[5]=o,c[6]=n,c[7]=s,c[8]=l,this}identity(){return this.set(1,0,0,0,1,0,0,0,1),this}copy(t){const e=this.elements,n=t.elements;return e[0]=n[0],e[1]=n[1],e[2]=n[2],e[3]=n[3],e[4]=n[4],e[5]=n[5],e[6]=n[6],e[7]=n[7],e[8]=n[8],this}extractBasis(t,e,n){return t.setFromMatrix3Column(this,0),e.setFromMatrix3Column(this,1),n.setFromMatrix3Column(this,2),this}setFromMatrix4(t){const e=t.elements;return this.set(e[0],e[4],e[8],e[1],e[5],e[9],e[2],e[6],e[10]),this}multiply(t){return this.multiplyMatrices(this,t)}premultiply(t){return this.multiplyMatrices(t,this)}multiplyMatrices(t,e){const n=t.elements,i=e.elements,r=this.elements,s=n[0],a=n[3],o=n[6],l=n[1],c=n[4],h=n[7],u=n[2],d=n[5],p=n[8],m=i[0],f=i[3],g=i[6],v=i[1],y=i[4],x=i[7],_=i[2],M=i[5],b=i[8];return r[0]=s*m+a*v+o*_,r[3]=s*f+a*y+o*M,r[6]=s*g+a*x+o*b,r[1]=l*m+c*v+h*_,r[4]=l*f+c*y+h*M,r[7]=l*g+c*x+h*b,r[2]=u*m+d*v+p*_,r[5]=u*f+d*y+p*M,r[8]=u*g+d*x+p*b,this}multiplyScalar(t){const e=this.elements;return e[0]*=t,e[3]*=t,e[6]*=t,e[1]*=t,e[4]*=t,e[7]*=t,e[2]*=t,e[5]*=t,e[8]*=t,this}determinant(){const t=this.elements,e=t[0],n=t[1],i=t[2],r=t[3],s=t[4],a=t[5],o=t[6],l=t[7],c=t[8];return e*s*c-e*a*l-n*r*c+n*a*o+i*r*l-i*s*o}invert(){const t=this.elements,e=t[0],n=t[1],i=t[2],r=t[3],s=t[4],a=t[5],o=t[6],l=t[7],c=t[8],h=c*s-a*l,u=a*o-c*r,d=l*r-s*o,p=e*h+n*u+i*d;if(0===p)return this.set(0,0,0,0,0,0,0,0,0);const m=1/p;return t[0]=h*m,t[1]=(i*l-c*n)*m,t[2]=(a*n-i*s)*m,t[3]=u*m,t[4]=(c*e-i*o)*m,t[5]=(i*r-a*e)*m,t[6]=d*m,t[7]=(n*o-l*e)*m,t[8]=(s*e-n*r)*m,this}transpose(){let t;const e=this.elements;return t=e[1],e[1]=e[3],e[3]=t,t=e[2],e[2]=e[6],e[6]=t,t=e[5],e[5]=e[7],e[7]=t,this}getNormalMatrix(t){return this.setFromMatrix4(t).invert().transpose()}transposeIntoArray(t){const e=this.elements;return t[0]=e[0],t[1]=e[3],t[2]=e[6],t[3]=e[1],t[4]=e[4],t[5]=e[7],t[6]=e[2],t[7]=e[5],t[8]=e[8],this}setUvTransform(t,e,n,i,r,s,a){const o=Math.cos(r),l=Math.sin(r);return this.set(n*o,n*l,-n*(o*s+l*a)+s+t,-i*l,i*o,-i*(-l*s+o*a)+a+e,0,0,1),this}scale(t,e){const n=this.elements;return n[0]*=t,n[3]*=t,n[6]*=t,n[1]*=e,n[4]*=e,n[7]*=e,this}rotate(t){const e=Math.cos(t),n=Math.sin(t),i=this.elements,r=i[0],s=i[3],a=i[6],o=i[1],l=i[4],c=i[7];return i[0]=e*r+n*o,i[3]=e*s+n*l,i[6]=e*a+n*c,i[1]=-n*r+e*o,i[4]=-n*s+e*l,i[7]=-n*a+e*c,this}translate(t,e){const n=this.elements;return n[0]+=t*n[2],n[3]+=t*n[5],n[6]+=t*n[8],n[1]+=e*n[2],n[4]+=e*n[5],n[7]+=e*n[8],this}equals(t){const e=this.elements,n=t.elements;for(let t=0;t<9;t++)if(e[t]!==n[t])return!1;return!0}fromArray(t,e=0){for(let n=0;n<9;n++)this.elements[n]=t[n+e];return this}toArray(t=[],e=0){const n=this.elements;return t[e]=n[0],t[e+1]=n[1],t[e+2]=n[2],t[e+3]=n[3],t[e+4]=n[4],t[e+5]=n[5],t[e+6]=n[6],t[e+7]=n[7],t[e+8]=n[8],t}clone(){return(new this.constructor).fromArray(this.elements)}}function _t(t){if(0===t.length)return-1/0;let e=t[0];for(let n=1,i=t.length;n<i;++n)t[n]>e&&(e=t[n]);return e}xt.prototype.isMatrix3=!0;const Mt={Int8Array:Int8Array,Uint8Array:Uint8Array,Uint8ClampedArray:Uint8ClampedArray,Int16Array:Int16Array,Uint16Array:Uint16Array,Int32Array:Int32Array,Uint32Array:Uint32Array,Float32Array:Float32Array,Float64Array:Float64Array};function bt(t,e){return new Mt[t](e)}function wt(t){return document.createElementNS("http://www.w3.org/1999/xhtml",t)}function St(t,e=0){let n=3735928559^e,i=1103547991^e;for(let e,r=0;r<t.length;r++)e=t.charCodeAt(r),n=Math.imul(n^e,2654435761),i=Math.imul(i^e,1597334677);return n=Math.imul(n^n>>>16,2246822507)^Math.imul(i^i>>>13,3266489909),i=Math.imul(i^i>>>16,2246822507)^Math.imul(n^n>>>13,3266489909),4294967296*(2097151&i)+(n>>>0)}let Tt;class Et{static getDataURL(t){if(/^data:/i.test(t.src))return t.src;if("undefined"==typeof HTMLCanvasElement)return t.src;let e;if(t instanceof HTMLCanvasElement)e=t;else{void 0===Tt&&(Tt=wt("canvas")),Tt.width=t.width,Tt.height=t.height;const n=Tt.getContext("2d");t instanceof ImageData?n.putImageData(t,0,0):n.drawImage(t,0,0,t.width,t.height),e=Tt}return e.width>2048||e.height>2048?(console.warn("THREE.ImageUtils.getDataURL: Image converted to jpg for performance reasons",t),e.toDataURL("image/jpeg",.6)):e.toDataURL("image/png")}}let At=0;class Lt extends rt{constructor(t=Lt.DEFAULT_IMAGE,e=Lt.DEFAULT_MAPPING,n=1001,i=1001,r=1006,s=1008,a=1023,o=1009,l=1,c=3e3){super(),Object.defineProperty(this,"id",{value:At++}),this.uuid=ht(),this.name="",this.image=t,this.mipmaps=[],this.mapping=e,this.wrapS=n,this.wrapT=i,this.magFilter=r,this.minFilter=s,this.anisotropy=l,this.format=a,this.internalFormat=null,this.type=o,this.offset=new yt(0,0),this.repeat=new yt(1,1),this.center=new yt(0,0),this.rotation=0,this.matrixAutoUpdate=!0,this.matrix=new xt,this.generateMipmaps=!0,this.premultiplyAlpha=!1,this.flipY=!0,this.unpackAlignment=4,this.encoding=c,this.userData={},this.version=0,this.onUpdate=null,this.isRenderTargetTexture=!1}updateMatrix(){this.matrix.setUvTransform(this.offset.x,this.offset.y,this.repeat.x,this.repeat.y,this.rotation,this.center.x,this.center.y)}clone(){return(new this.constructor).copy(this)}copy(t){return this.name=t.name,this.image=t.image,this.mipmaps=t.mipmaps.slice(0),this.mapping=t.mapping,this.wrapS=t.wrapS,this.wrapT=t.wrapT,this.magFilter=t.magFilter,this.minFilter=t.minFilter,this.anisotropy=t.anisotropy,this.format=t.format,this.internalFormat=t.internalFormat,this.type=t.type,this.offset.copy(t.offset),this.repeat.copy(t.repeat),this.center.copy(t.center),this.rotation=t.rotation,this.matrixAutoUpdate=t.matrixAutoUpdate,this.matrix.copy(t.matrix),this.generateMipmaps=t.generateMipmaps,this.premultiplyAlpha=t.premultiplyAlpha,this.flipY=t.flipY,this.unpackAlignment=t.unpackAlignment,this.encoding=t.encoding,this.userData=JSON.parse(JSON.stringify(t.userData)),this}toJSON(t){const e=void 0===t||"string"==typeof t;if(!e&&void 0!==t.textures[this.uuid])return t.textures[this.uuid];const n={metadata:{version:4.5,type:"Texture",generator:"Texture.toJSON"},uuid:this.uuid,name:this.name,mapping:this.mapping,repeat:[this.repeat.x,this.repeat.y],offset:[this.offset.x,this.offset.y],center:[this.center.x,this.center.y],rotation:this.rotation,wrap:[this.wrapS,this.wrapT],format:this.format,type:this.type,encoding:this.encoding,minFilter:this.minFilter,magFilter:this.magFilter,anisotropy:this.anisotropy,flipY:this.flipY,premultiplyAlpha:this.premultiplyAlpha,unpackAlignment:this.unpackAlignment};if(void 0!==this.image){const i=this.image;if(void 0===i.uuid&&(i.uuid=ht()),!e&&void 0===t.images[i.uuid]){let e;if(Array.isArray(i)){e=[];for(let t=0,n=i.length;t<n;t++)i[t].isDataTexture?e.push(Rt(i[t].image)):e.push(Rt(i[t]))}else e=Rt(i);t.images[i.uuid]={uuid:i.uuid,url:e}}n.image=i.uuid}return"{}"!==JSON.stringify(this.userData)&&(n.userData=this.userData),e||(t.textures[this.uuid]=n),n}dispose(){this.dispatchEvent({type:"dispose"})}transformUv(t){if(this.mapping!==i)return t;if(t.applyMatrix3(this.matrix),t.x<0||t.x>1)switch(this.wrapS){case h:t.x=t.x-Math.floor(t.x);break;case u:t.x=t.x<0?0:1;break;case d:1===Math.abs(Math.floor(t.x)%2)?t.x=Math.ceil(t.x)-t.x:t.x=t.x-Math.floor(t.x)}if(t.y<0||t.y>1)switch(this.wrapT){case h:t.y=t.y-Math.floor(t.y);break;case u:t.y=t.y<0?0:1;break;case d:1===Math.abs(Math.floor(t.y)%2)?t.y=Math.ceil(t.y)-t.y:t.y=t.y-Math.floor(t.y)}return this.flipY&&(t.y=1-t.y),t}set needsUpdate(t){!0===t&&this.version++}}function Rt(t){return"undefined"!=typeof HTMLImageElement&&t instanceof HTMLImageElement||"undefined"!=typeof HTMLCanvasElement&&t instanceof HTMLCanvasElement||"undefined"!=typeof ImageBitmap&&t instanceof ImageBitmap?Et.getDataURL(t):t.data?{data:Array.prototype.slice.call(t.data),width:t.width,height:t.height,type:t.data.constructor.name}:(console.warn("THREE.Texture: Unable to serialize Texture."),{})}Lt.DEFAULT_IMAGE=void 0,Lt.DEFAULT_MAPPING=i,Lt.prototype.isTexture=!0;class Ct{constructor(t=0,e=0,n=0,i=1){this.x=t,this.y=e,this.z=n,this.w=i}get width(){return this.z}set width(t){this.z=t}get height(){return this.w}set height(t){this.w=t}set(t,e,n,i){return this.x=t,this.y=e,this.z=n,this.w=i,this}setScalar(t){return this.x=t,this.y=t,this.z=t,this.w=t,this}setX(t){return this.x=t,this}setY(t){return this.y=t,this}setZ(t){return this.z=t,this}setW(t){return this.w=t,this}setComponent(t,e){switch(t){case 0:this.x=e;break;case 1:this.y=e;break;case 2:this.z=e;break;case 3:this.w=e;break;default:throw new Error("index is out of range: "+t)}return this}getComponent(t){switch(t){case 0:return this.x;case 1:return this.y;case 2:return this.z;case 3:return this.w;default:throw new Error("index is out of range: "+t)}}clone(){return new this.constructor(this.x,this.y,this.z,this.w)}copy(t){return this.x=t.x,this.y=t.y,this.z=t.z,this.w=void 0!==t.w?t.w:1,this}add(t,e){return void 0!==e?(console.warn("THREE.Vector4: .add() now only accepts one argument. Use .addVectors( a, b ) instead."),this.addVectors(t,e)):(this.x+=t.x,this.y+=t.y,this.z+=t.z,this.w+=t.w,this)}addScalar(t){return this.x+=t,this.y+=t,this.z+=t,this.w+=t,this}addVectors(t,e){return this.x=t.x+e.x,this.y=t.y+e.y,this.z=t.z+e.z,this.w=t.w+e.w,this}addScaledVector(t,e){return this.x+=t.x*e,this.y+=t.y*e,this.z+=t.z*e,this.w+=t.w*e,this}sub(t,e){return void 0!==e?(console.warn("THREE.Vector4: .sub() now only accepts one argument. Use .subVectors( a, b ) instead."),this.subVectors(t,e)):(this.x-=t.x,this.y-=t.y,this.z-=t.z,this.w-=t.w,this)}subScalar(t){return this.x-=t,this.y-=t,this.z-=t,this.w-=t,this}subVectors(t,e){return this.x=t.x-e.x,this.y=t.y-e.y,this.z=t.z-e.z,this.w=t.w-e.w,this}multiply(t){return this.x*=t.x,this.y*=t.y,this.z*=t.z,this.w*=t.w,this}multiplyScalar(t){return this.x*=t,this.y*=t,this.z*=t,this.w*=t,this}applyMatrix4(t){const e=this.x,n=this.y,i=this.z,r=this.w,s=t.elements;return this.x=s[0]*e+s[4]*n+s[8]*i+s[12]*r,this.y=s[1]*e+s[5]*n+s[9]*i+s[13]*r,this.z=s[2]*e+s[6]*n+s[10]*i+s[14]*r,this.w=s[3]*e+s[7]*n+s[11]*i+s[15]*r,this}divideScalar(t){return this.multiplyScalar(1/t)}setAxisAngleFromQuaternion(t){this.w=2*Math.acos(t.w);const e=Math.sqrt(1-t.w*t.w);return e<1e-4?(this.x=1,this.y=0,this.z=0):(this.x=t.x/e,this.y=t.y/e,this.z=t.z/e),this}setAxisAngleFromRotationMatrix(t){let e,n,i,r;const s=.01,a=.1,o=t.elements,l=o[0],c=o[4],h=o[8],u=o[1],d=o[5],p=o[9],m=o[2],f=o[6],g=o[10];if(Math.abs(c-u)<s&&Math.abs(h-m)<s&&Math.abs(p-f)<s){if(Math.abs(c+u)<a&&Math.abs(h+m)<a&&Math.abs(p+f)<a&&Math.abs(l+d+g-3)<a)return this.set(1,0,0,0),this;e=Math.PI;const t=(l+1)/2,o=(d+1)/2,v=(g+1)/2,y=(c+u)/4,x=(h+m)/4,_=(p+f)/4;return t>o&&t>v?t<s?(n=0,i=.707106781,r=.707106781):(n=Math.sqrt(t),i=y/n,r=x/n):o>v?o<s?(n=.707106781,i=0,r=.707106781):(i=Math.sqrt(o),n=y/i,r=_/i):v<s?(n=.707106781,i=.707106781,r=0):(r=Math.sqrt(v),n=x/r,i=_/r),this.set(n,i,r,e),this}let v=Math.sqrt((f-p)*(f-p)+(h-m)*(h-m)+(u-c)*(u-c));return Math.abs(v)<.001&&(v=1),this.x=(f-p)/v,this.y=(h-m)/v,this.z=(u-c)/v,this.w=Math.acos((l+d+g-1)/2),this}min(t){return this.x=Math.min(this.x,t.x),this.y=Math.min(this.y,t.y),this.z=Math.min(this.z,t.z),this.w=Math.min(this.w,t.w),this}max(t){return this.x=Math.max(this.x,t.x),this.y=Math.max(this.y,t.y),this.z=Math.max(this.z,t.z),this.w=Math.max(this.w,t.w),this}clamp(t,e){return this.x=Math.max(t.x,Math.min(e.x,this.x)),this.y=Math.max(t.y,Math.min(e.y,this.y)),this.z=Math.max(t.z,Math.min(e.z,this.z)),this.w=Math.max(t.w,Math.min(e.w,this.w)),this}clampScalar(t,e){return this.x=Math.max(t,Math.min(e,this.x)),this.y=Math.max(t,Math.min(e,this.y)),this.z=Math.max(t,Math.min(e,this.z)),this.w=Math.max(t,Math.min(e,this.w)),this}clampLength(t,e){const n=this.length();return this.divideScalar(n||1).multiplyScalar(Math.max(t,Math.min(e,n)))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this.w=Math.floor(this.w),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this.w=Math.ceil(this.w),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this.w=Math.round(this.w),this}roundToZero(){return this.x=this.x<0?Math.ceil(this.x):Math.floor(this.x),this.y=this.y<0?Math.ceil(this.y):Math.floor(this.y),this.z=this.z<0?Math.ceil(this.z):Math.floor(this.z),this.w=this.w<0?Math.ceil(this.w):Math.floor(this.w),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this.w=-this.w,this}dot(t){return this.x*t.x+this.y*t.y+this.z*t.z+this.w*t.w}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)+Math.abs(this.w)}normalize(){return this.divideScalar(this.length()||1)}setLength(t){return this.normalize().multiplyScalar(t)}lerp(t,e){return this.x+=(t.x-this.x)*e,this.y+=(t.y-this.y)*e,this.z+=(t.z-this.z)*e,this.w+=(t.w-this.w)*e,this}lerpVectors(t,e,n){return this.x=t.x+(e.x-t.x)*n,this.y=t.y+(e.y-t.y)*n,this.z=t.z+(e.z-t.z)*n,this.w=t.w+(e.w-t.w)*n,this}equals(t){return t.x===this.x&&t.y===this.y&&t.z===this.z&&t.w===this.w}fromArray(t,e=0){return this.x=t[e],this.y=t[e+1],this.z=t[e+2],this.w=t[e+3],this}toArray(t=[],e=0){return t[e]=this.x,t[e+1]=this.y,t[e+2]=this.z,t[e+3]=this.w,t}fromBufferAttribute(t,e,n){return void 0!==n&&console.warn("THREE.Vector4: offset has been removed from .fromBufferAttribute()."),this.x=t.getX(e),this.y=t.getY(e),this.z=t.getZ(e),this.w=t.getW(e),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this.w=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z,yield this.w}}Ct.prototype.isVector4=!0;class Pt extends rt{constructor(t,e,n={}){super(),this.width=t,this.height=e,this.depth=1,this.scissor=new Ct(0,0,t,e),this.scissorTest=!1,this.viewport=new Ct(0,0,t,e),this.texture=new Lt(void 0,n.mapping,n.wrapS,n.wrapT,n.magFilter,n.minFilter,n.format,n.type,n.anisotropy,n.encoding),this.texture.isRenderTargetTexture=!0,this.texture.image={width:t,height:e,depth:1},this.texture.generateMipmaps=void 0!==n.generateMipmaps&&n.generateMipmaps,this.texture.internalFormat=void 0!==n.internalFormat?n.internalFormat:null,this.texture.minFilter=void 0!==n.minFilter?n.minFilter:g,this.depthBuffer=void 0===n.depthBuffer||n.depthBuffer,this.stencilBuffer=void 0!==n.stencilBuffer&&n.stencilBuffer,this.depthTexture=void 0!==n.depthTexture?n.depthTexture:null}setTexture(t){t.image={width:this.width,height:this.height,depth:this.depth},this.texture=t}setSize(t,e,n=1){this.width===t&&this.height===e&&this.depth===n||(this.width=t,this.height=e,this.depth=n,this.texture.image.width=t,this.texture.image.height=e,this.texture.image.depth=n,this.dispose()),this.viewport.set(0,0,t,e),this.scissor.set(0,0,t,e)}clone(){return(new this.constructor).copy(this)}copy(t){return this.width=t.width,this.height=t.height,this.depth=t.depth,this.viewport.copy(t.viewport),this.texture=t.texture.clone(),this.texture.image={...this.texture.image},this.depthBuffer=t.depthBuffer,this.stencilBuffer=t.stencilBuffer,this.depthTexture=t.depthTexture,this}dispose(){this.dispatchEvent({type:"dispose"})}}Pt.prototype.isWebGLRenderTarget=!0;class It extends Pt{constructor(t,e,n){super(t,e);const i=this.texture;this.texture=[];for(let t=0;t<n;t++)this.texture[t]=i.clone()}setSize(t,e,n=1){if(this.width!==t||this.height!==e||this.depth!==n){this.width=t,this.height=e,this.depth=n;for(let i=0,r=this.texture.length;i<r;i++)this.texture[i].image.width=t,this.texture[i].image.height=e,this.texture[i].image.depth=n;this.dispose()}return this.viewport.set(0,0,t,e),this.scissor.set(0,0,t,e),this}copy(t){this.dispose(),this.width=t.width,this.height=t.height,this.depth=t.depth,this.viewport.set(0,0,this.width,this.height),this.scissor.set(0,0,this.width,this.height),this.depthBuffer=t.depthBuffer,this.stencilBuffer=t.stencilBuffer,this.depthTexture=t.depthTexture,this.texture.length=0;for(let e=0,n=t.texture.length;e<n;e++)this.texture[e]=t.texture[e].clone();return this}}It.prototype.isWebGLMultipleRenderTargets=!0;class Dt extends Pt{constructor(t,e,n){super(t,e,n),this.samples=4}copy(t){return super.copy.call(this,t),this.samples=t.samples,this}}Dt.prototype.isWebGLMultisampleRenderTarget=!0;class Nt{constructor(t=0,e=0,n=0,i=1){this._x=t,this._y=e,this._z=n,this._w=i}static slerp(t,e,n,i){return console.warn("THREE.Quaternion: Static .slerp() has been deprecated. Use qm.slerpQuaternions( qa, qb, t ) instead."),n.slerpQuaternions(t,e,i)}static slerpFlat(t,e,n,i,r,s,a){let o=n[i+0],l=n[i+1],c=n[i+2],h=n[i+3];const u=r[s+0],d=r[s+1],p=r[s+2],m=r[s+3];if(0===a)return t[e+0]=o,t[e+1]=l,t[e+2]=c,void(t[e+3]=h);if(1===a)return t[e+0]=u,t[e+1]=d,t[e+2]=p,void(t[e+3]=m);if(h!==m||o!==u||l!==d||c!==p){let t=1-a;const e=o*u+l*d+c*p+h*m,n=e>=0?1:-1,i=1-e*e;if(i>Number.EPSILON){const r=Math.sqrt(i),s=Math.atan2(r,e*n);t=Math.sin(t*s)/r,a=Math.sin(a*s)/r}const r=a*n;if(o=o*t+u*r,l=l*t+d*r,c=c*t+p*r,h=h*t+m*r,t===1-a){const t=1/Math.sqrt(o*o+l*l+c*c+h*h);o*=t,l*=t,c*=t,h*=t}}t[e]=o,t[e+1]=l,t[e+2]=c,t[e+3]=h}static multiplyQuaternionsFlat(t,e,n,i,r,s){const a=n[i],o=n[i+1],l=n[i+2],c=n[i+3],h=r[s],u=r[s+1],d=r[s+2],p=r[s+3];return t[e]=a*p+c*h+o*d-l*u,t[e+1]=o*p+c*u+l*h-a*d,t[e+2]=l*p+c*d+a*u-o*h,t[e+3]=c*p-a*h-o*u-l*d,t}get x(){return this._x}set x(t){this._x=t,this._onChangeCallback()}get y(){return this._y}set y(t){this._y=t,this._onChangeCallback()}get z(){return this._z}set z(t){this._z=t,this._onChangeCallback()}get w(){return this._w}set w(t){this._w=t,this._onChangeCallback()}set(t,e,n,i){return this._x=t,this._y=e,this._z=n,this._w=i,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._w)}copy(t){return this._x=t.x,this._y=t.y,this._z=t.z,this._w=t.w,this._onChangeCallback(),this}setFromEuler(t,e){if(!t||!t.isEuler)throw new Error("THREE.Quaternion: .setFromEuler() now expects an Euler rotation rather than a Vector3 and order.");const n=t._x,i=t._y,r=t._z,s=t._order,a=Math.cos,o=Math.sin,l=a(n/2),c=a(i/2),h=a(r/2),u=o(n/2),d=o(i/2),p=o(r/2);switch(s){case"XYZ":this._x=u*c*h+l*d*p,this._y=l*d*h-u*c*p,this._z=l*c*p+u*d*h,this._w=l*c*h-u*d*p;break;case"YXZ":this._x=u*c*h+l*d*p,this._y=l*d*h-u*c*p,this._z=l*c*p-u*d*h,this._w=l*c*h+u*d*p;break;case"ZXY":this._x=u*c*h-l*d*p,this._y=l*d*h+u*c*p,this._z=l*c*p+u*d*h,this._w=l*c*h-u*d*p;break;case"ZYX":this._x=u*c*h-l*d*p,this._y=l*d*h+u*c*p,this._z=l*c*p-u*d*h,this._w=l*c*h+u*d*p;break;case"YZX":this._x=u*c*h+l*d*p,this._y=l*d*h+u*c*p,this._z=l*c*p-u*d*h,this._w=l*c*h-u*d*p;break;case"XZY":this._x=u*c*h-l*d*p,this._y=l*d*h-u*c*p,this._z=l*c*p+u*d*h,this._w=l*c*h+u*d*p;break;default:console.warn("THREE.Quaternion: .setFromEuler() encountered an unknown order: "+s)}return!1!==e&&this._onChangeCallback(),this}setFromAxisAngle(t,e){const n=e/2,i=Math.sin(n);return this._x=t.x*i,this._y=t.y*i,this._z=t.z*i,this._w=Math.cos(n),this._onChangeCallback(),this}setFromRotationMatrix(t){const e=t.elements,n=e[0],i=e[4],r=e[8],s=e[1],a=e[5],o=e[9],l=e[2],c=e[6],h=e[10],u=n+a+h;if(u>0){const t=.5/Math.sqrt(u+1);this._w=.25/t,this._x=(c-o)*t,this._y=(r-l)*t,this._z=(s-i)*t}else if(n>a&&n>h){const t=2*Math.sqrt(1+n-a-h);this._w=(c-o)/t,this._x=.25*t,this._y=(i+s)/t,this._z=(r+l)/t}else if(a>h){const t=2*Math.sqrt(1+a-n-h);this._w=(r-l)/t,this._x=(i+s)/t,this._y=.25*t,this._z=(o+c)/t}else{const t=2*Math.sqrt(1+h-n-a);this._w=(s-i)/t,this._x=(r+l)/t,this._y=(o+c)/t,this._z=.25*t}return this._onChangeCallback(),this}setFromUnitVectors(t,e){let n=t.dot(e)+1;return n<Number.EPSILON?(n=0,Math.abs(t.x)>Math.abs(t.z)?(this._x=-t.y,this._y=t.x,this._z=0,this._w=n):(this._x=0,this._y=-t.z,this._z=t.y,this._w=n)):(this._x=t.y*e.z-t.z*e.y,this._y=t.z*e.x-t.x*e.z,this._z=t.x*e.y-t.y*e.x,this._w=n),this.normalize()}angleTo(t){return 2*Math.acos(Math.abs(ut(this.dot(t),-1,1)))}rotateTowards(t,e){const n=this.angleTo(t);if(0===n)return this;const i=Math.min(1,e/n);return this.slerp(t,i),this}identity(){return this.set(0,0,0,1)}invert(){return this.conjugate()}conjugate(){return this._x*=-1,this._y*=-1,this._z*=-1,this._onChangeCallback(),this}dot(t){return this._x*t._x+this._y*t._y+this._z*t._z+this._w*t._w}lengthSq(){return this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w}length(){return Math.sqrt(this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w)}normalize(){let t=this.length();return 0===t?(this._x=0,this._y=0,this._z=0,this._w=1):(t=1/t,this._x=this._x*t,this._y=this._y*t,this._z=this._z*t,this._w=this._w*t),this._onChangeCallback(),this}multiply(t,e){return void 0!==e?(console.warn("THREE.Quaternion: .multiply() now only accepts one argument. Use .multiplyQuaternions( a, b ) instead."),this.multiplyQuaternions(t,e)):this.multiplyQuaternions(this,t)}premultiply(t){return this.multiplyQuaternions(t,this)}multiplyQuaternions(t,e){const n=t._x,i=t._y,r=t._z,s=t._w,a=e._x,o=e._y,l=e._z,c=e._w;return this._x=n*c+s*a+i*l-r*o,this._y=i*c+s*o+r*a-n*l,this._z=r*c+s*l+n*o-i*a,this._w=s*c-n*a-i*o-r*l,this._onChangeCallback(),this}slerp(t,e){if(0===e)return this;if(1===e)return this.copy(t);const n=this._x,i=this._y,r=this._z,s=this._w;let a=s*t._w+n*t._x+i*t._y+r*t._z;if(a<0?(this._w=-t._w,this._x=-t._x,this._y=-t._y,this._z=-t._z,a=-a):this.copy(t),a>=1)return this._w=s,this._x=n,this._y=i,this._z=r,this;const o=1-a*a;if(o<=Number.EPSILON){const t=1-e;return this._w=t*s+e*this._w,this._x=t*n+e*this._x,this._y=t*i+e*this._y,this._z=t*r+e*this._z,this.normalize(),this._onChangeCallback(),this}const l=Math.sqrt(o),c=Math.atan2(l,a),h=Math.sin((1-e)*c)/l,u=Math.sin(e*c)/l;return this._w=s*h+this._w*u,this._x=n*h+this._x*u,this._y=i*h+this._y*u,this._z=r*h+this._z*u,this._onChangeCallback(),this}slerpQuaternions(t,e,n){this.copy(t).slerp(e,n)}random(){const t=Math.random(),e=Math.sqrt(1-t),n=Math.sqrt(t),i=2*Math.PI*Math.random(),r=2*Math.PI*Math.random();return this.set(e*Math.cos(i),n*Math.sin(r),n*Math.cos(r),e*Math.sin(i))}equals(t){return t._x===this._x&&t._y===this._y&&t._z===this._z&&t._w===this._w}fromArray(t,e=0){return this._x=t[e],this._y=t[e+1],this._z=t[e+2],this._w=t[e+3],this._onChangeCallback(),this}toArray(t=[],e=0){return t[e]=this._x,t[e+1]=this._y,t[e+2]=this._z,t[e+3]=this._w,t}fromBufferAttribute(t,e){return this._x=t.getX(e),this._y=t.getY(e),this._z=t.getZ(e),this._w=t.getW(e),this}_onChange(t){return this._onChangeCallback=t,this}_onChangeCallback(){}}Nt.prototype.isQuaternion=!0;class zt{constructor(t=0,e=0,n=0){this.x=t,this.y=e,this.z=n}set(t,e,n){return void 0===n&&(n=this.z),this.x=t,this.y=e,this.z=n,this}setScalar(t){return this.x=t,this.y=t,this.z=t,this}setX(t){return this.x=t,this}setY(t){return this.y=t,this}setZ(t){return this.z=t,this}setComponent(t,e){switch(t){case 0:this.x=e;break;case 1:this.y=e;break;case 2:this.z=e;break;default:throw new Error("index is out of range: "+t)}return this}getComponent(t){switch(t){case 0:return this.x;case 1:return this.y;case 2:return this.z;default:throw new Error("index is out of range: "+t)}}clone(){return new this.constructor(this.x,this.y,this.z)}copy(t){return this.x=t.x,this.y=t.y,this.z=t.z,this}add(t,e){return void 0!==e?(console.warn("THREE.Vector3: .add() now only accepts one argument. Use .addVectors( a, b ) instead."),this.addVectors(t,e)):(this.x+=t.x,this.y+=t.y,this.z+=t.z,this)}addScalar(t){return this.x+=t,this.y+=t,this.z+=t,this}addVectors(t,e){return this.x=t.x+e.x,this.y=t.y+e.y,this.z=t.z+e.z,this}addScaledVector(t,e){return this.x+=t.x*e,this.y+=t.y*e,this.z+=t.z*e,this}sub(t,e){return void 0!==e?(console.warn("THREE.Vector3: .sub() now only accepts one argument. Use .subVectors( a, b ) instead."),this.subVectors(t,e)):(this.x-=t.x,this.y-=t.y,this.z-=t.z,this)}subScalar(t){return this.x-=t,this.y-=t,this.z-=t,this}subVectors(t,e){return this.x=t.x-e.x,this.y=t.y-e.y,this.z=t.z-e.z,this}multiply(t,e){return void 0!==e?(console.warn("THREE.Vector3: .multiply() now only accepts one argument. Use .multiplyVectors( a, b ) instead."),this.multiplyVectors(t,e)):(this.x*=t.x,this.y*=t.y,this.z*=t.z,this)}multiplyScalar(t){return this.x*=t,this.y*=t,this.z*=t,this}multiplyVectors(t,e){return this.x=t.x*e.x,this.y=t.y*e.y,this.z=t.z*e.z,this}applyEuler(t){return t&&t.isEuler||console.error("THREE.Vector3: .applyEuler() now expects an Euler rotation rather than a Vector3 and order."),this.applyQuaternion(Ft.setFromEuler(t))}applyAxisAngle(t,e){return this.applyQuaternion(Ft.setFromAxisAngle(t,e))}applyMatrix3(t){const e=this.x,n=this.y,i=this.z,r=t.elements;return this.x=r[0]*e+r[3]*n+r[6]*i,this.y=r[1]*e+r[4]*n+r[7]*i,this.z=r[2]*e+r[5]*n+r[8]*i,this}applyNormalMatrix(t){return this.applyMatrix3(t).normalize()}applyMatrix4(t){const e=this.x,n=this.y,i=this.z,r=t.elements,s=1/(r[3]*e+r[7]*n+r[11]*i+r[15]);return this.x=(r[0]*e+r[4]*n+r[8]*i+r[12])*s,this.y=(r[1]*e+r[5]*n+r[9]*i+r[13])*s,this.z=(r[2]*e+r[6]*n+r[10]*i+r[14])*s,this}applyQuaternion(t){const e=this.x,n=this.y,i=this.z,r=t.x,s=t.y,a=t.z,o=t.w,l=o*e+s*i-a*n,c=o*n+a*e-r*i,h=o*i+r*n-s*e,u=-r*e-s*n-a*i;return this.x=l*o+u*-r+c*-a-h*-s,this.y=c*o+u*-s+h*-r-l*-a,this.z=h*o+u*-a+l*-s-c*-r,this}project(t){return this.applyMatrix4(t.matrixWorldInverse).applyMatrix4(t.projectionMatrix)}unproject(t){return this.applyMatrix4(t.projectionMatrixInverse).applyMatrix4(t.matrixWorld)}transformDirection(t){const e=this.x,n=this.y,i=this.z,r=t.elements;return this.x=r[0]*e+r[4]*n+r[8]*i,this.y=r[1]*e+r[5]*n+r[9]*i,this.z=r[2]*e+r[6]*n+r[10]*i,this.normalize()}divide(t){return this.x/=t.x,this.y/=t.y,this.z/=t.z,this}divideScalar(t){return this.multiplyScalar(1/t)}min(t){return this.x=Math.min(this.x,t.x),this.y=Math.min(this.y,t.y),this.z=Math.min(this.z,t.z),this}max(t){return this.x=Math.max(this.x,t.x),this.y=Math.max(this.y,t.y),this.z=Math.max(this.z,t.z),this}clamp(t,e){return this.x=Math.max(t.x,Math.min(e.x,this.x)),this.y=Math.max(t.y,Math.min(e.y,this.y)),this.z=Math.max(t.z,Math.min(e.z,this.z)),this}clampScalar(t,e){return this.x=Math.max(t,Math.min(e,this.x)),this.y=Math.max(t,Math.min(e,this.y)),this.z=Math.max(t,Math.min(e,this.z)),this}clampLength(t,e){const n=this.length();return this.divideScalar(n||1).multiplyScalar(Math.max(t,Math.min(e,n)))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this}roundToZero(){return this.x=this.x<0?Math.ceil(this.x):Math.floor(this.x),this.y=this.y<0?Math.ceil(this.y):Math.floor(this.y),this.z=this.z<0?Math.ceil(this.z):Math.floor(this.z),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this}dot(t){return this.x*t.x+this.y*t.y+this.z*t.z}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)}normalize(){return this.divideScalar(this.length()||1)}setLength(t){return this.normalize().multiplyScalar(t)}lerp(t,e){return this.x+=(t.x-this.x)*e,this.y+=(t.y-this.y)*e,this.z+=(t.z-this.z)*e,this}lerpVectors(t,e,n){return this.x=t.x+(e.x-t.x)*n,this.y=t.y+(e.y-t.y)*n,this.z=t.z+(e.z-t.z)*n,this}cross(t,e){return void 0!==e?(console.warn("THREE.Vector3: .cross() now only accepts one argument. Use .crossVectors( a, b ) instead."),this.crossVectors(t,e)):this.crossVectors(this,t)}crossVectors(t,e){const n=t.x,i=t.y,r=t.z,s=e.x,a=e.y,o=e.z;return this.x=i*o-r*a,this.y=r*s-n*o,this.z=n*a-i*s,this}projectOnVector(t){const e=t.lengthSq();if(0===e)return this.set(0,0,0);const n=t.dot(this)/e;return this.copy(t).multiplyScalar(n)}projectOnPlane(t){return Bt.copy(this).projectOnVector(t),this.sub(Bt)}reflect(t){return this.sub(Bt.copy(t).multiplyScalar(2*this.dot(t)))}angleTo(t){const e=Math.sqrt(this.lengthSq()*t.lengthSq());if(0===e)return Math.PI/2;const n=this.dot(t)/e;return Math.acos(ut(n,-1,1))}distanceTo(t){return Math.sqrt(this.distanceToSquared(t))}distanceToSquared(t){const e=this.x-t.x,n=this.y-t.y,i=this.z-t.z;return e*e+n*n+i*i}manhattanDistanceTo(t){return Math.abs(this.x-t.x)+Math.abs(this.y-t.y)+Math.abs(this.z-t.z)}setFromSpherical(t){return this.setFromSphericalCoords(t.radius,t.phi,t.theta)}setFromSphericalCoords(t,e,n){const i=Math.sin(e)*t;return this.x=i*Math.sin(n),this.y=Math.cos(e)*t,this.z=i*Math.cos(n),this}setFromCylindrical(t){return this.setFromCylindricalCoords(t.radius,t.theta,t.y)}setFromCylindricalCoords(t,e,n){return this.x=t*Math.sin(e),this.y=n,this.z=t*Math.cos(e),this}setFromMatrixPosition(t){const e=t.elements;return this.x=e[12],this.y=e[13],this.z=e[14],this}setFromMatrixScale(t){const e=this.setFromMatrixColumn(t,0).length(),n=this.setFromMatrixColumn(t,1).length(),i=this.setFromMatrixColumn(t,2).length();return this.x=e,this.y=n,this.z=i,this}setFromMatrixColumn(t,e){return this.fromArray(t.elements,4*e)}setFromMatrix3Column(t,e){return this.fromArray(t.elements,3*e)}equals(t){return t.x===this.x&&t.y===this.y&&t.z===this.z}fromArray(t,e=0){return this.x=t[e],this.y=t[e+1],this.z=t[e+2],this}toArray(t=[],e=0){return t[e]=this.x,t[e+1]=this.y,t[e+2]=this.z,t}fromBufferAttribute(t,e,n){return void 0!==n&&console.warn("THREE.Vector3: offset has been removed from .fromBufferAttribute()."),this.x=t.getX(e),this.y=t.getY(e),this.z=t.getZ(e),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this}randomDirection(){const t=2*(Math.random()-.5),e=Math.random()*Math.PI*2,n=Math.sqrt(1-t**2);return this.x=n*Math.cos(e),this.y=n*Math.sin(e),this.z=t,this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z}}zt.prototype.isVector3=!0;const Bt=new zt,Ft=new Nt;class Ot{constructor(t=new zt(1/0,1/0,1/0),e=new zt(-1/0,-1/0,-1/0)){this.min=t,this.max=e}set(t,e){return this.min.copy(t),this.max.copy(e),this}setFromArray(t){let e=1/0,n=1/0,i=1/0,r=-1/0,s=-1/0,a=-1/0;for(let o=0,l=t.length;o<l;o+=3){const l=t[o],c=t[o+1],h=t[o+2];l<e&&(e=l),c<n&&(n=c),h<i&&(i=h),l>r&&(r=l),c>s&&(s=c),h>a&&(a=h)}return this.min.set(e,n,i),this.max.set(r,s,a),this}setFromBufferAttribute(t){let e=1/0,n=1/0,i=1/0,r=-1/0,s=-1/0,a=-1/0;for(let o=0,l=t.count;o<l;o++){const l=t.getX(o),c=t.getY(o),h=t.getZ(o);l<e&&(e=l),c<n&&(n=c),h<i&&(i=h),l>r&&(r=l),c>s&&(s=c),h>a&&(a=h)}return this.min.set(e,n,i),this.max.set(r,s,a),this}setFromPoints(t){this.makeEmpty();for(let e=0,n=t.length;e<n;e++)this.expandByPoint(t[e]);return this}setFromCenterAndSize(t,e){const n=Ht.copy(e).multiplyScalar(.5);return this.min.copy(t).sub(n),this.max.copy(t).add(n),this}setFromObject(t){return this.makeEmpty(),this.expandByObject(t)}clone(){return(new this.constructor).copy(this)}copy(t){return this.min.copy(t.min),this.max.copy(t.max),this}makeEmpty(){return this.min.x=this.min.y=this.min.z=1/0,this.max.x=this.max.y=this.max.z=-1/0,this}isEmpty(){return this.max.x<this.min.x||this.max.y<this.min.y||this.max.z<this.min.z}getCenter(t){return this.isEmpty()?t.set(0,0,0):t.addVectors(this.min,this.max).multiplyScalar(.5)}getSize(t){return this.isEmpty()?t.set(0,0,0):t.subVectors(this.max,this.min)}expandByPoint(t){return this.min.min(t),this.max.max(t),this}expandByVector(t){return this.min.sub(t),this.max.add(t),this}expandByScalar(t){return this.min.addScalar(-t),this.max.addScalar(t),this}expandByObject(t){t.updateWorldMatrix(!1,!1);const e=t.geometry;void 0!==e&&(null===e.boundingBox&&e.computeBoundingBox(),Gt.copy(e.boundingBox),Gt.applyMatrix4(t.matrixWorld),this.union(Gt));const n=t.children;for(let t=0,e=n.length;t<e;t++)this.expandByObject(n[t]);return this}containsPoint(t){return!(t.x<this.min.x||t.x>this.max.x||t.y<this.min.y||t.y>this.max.y||t.z<this.min.z||t.z>this.max.z)}containsBox(t){return this.min.x<=t.min.x&&t.max.x<=this.max.x&&this.min.y<=t.min.y&&t.max.y<=this.max.y&&this.min.z<=t.min.z&&t.max.z<=this.max.z}getParameter(t,e){return e.set((t.x-this.min.x)/(this.max.x-this.min.x),(t.y-this.min.y)/(this.max.y-this.min.y),(t.z-this.min.z)/(this.max.z-this.min.z))}intersectsBox(t){return!(t.max.x<this.min.x||t.min.x>this.max.x||t.max.y<this.min.y||t.min.y>this.max.y||t.max.z<this.min.z||t.min.z>this.max.z)}intersectsSphere(t){return this.clampPoint(t.center,Ht),Ht.distanceToSquared(t.center)<=t.radius*t.radius}intersectsPlane(t){let e,n;return t.normal.x>0?(e=t.normal.x*this.min.x,n=t.normal.x*this.max.x):(e=t.normal.x*this.max.x,n=t.normal.x*this.min.x),t.normal.y>0?(e+=t.normal.y*this.min.y,n+=t.normal.y*this.max.y):(e+=t.normal.y*this.max.y,n+=t.normal.y*this.min.y),t.normal.z>0?(e+=t.normal.z*this.min.z,n+=t.normal.z*this.max.z):(e+=t.normal.z*this.max.z,n+=t.normal.z*this.min.z),e<=-t.constant&&n>=-t.constant}intersectsTriangle(t){if(this.isEmpty())return!1;this.getCenter(Yt),Jt.subVectors(this.max,Yt),kt.subVectors(t.a,Yt),Vt.subVectors(t.b,Yt),Wt.subVectors(t.c,Yt),jt.subVectors(Vt,kt),qt.subVectors(Wt,Vt),Xt.subVectors(kt,Wt);let e=[0,-jt.z,jt.y,0,-qt.z,qt.y,0,-Xt.z,Xt.y,jt.z,0,-jt.x,qt.z,0,-qt.x,Xt.z,0,-Xt.x,-jt.y,jt.x,0,-qt.y,qt.x,0,-Xt.y,Xt.x,0];return!!Kt(e,kt,Vt,Wt,Jt)&&(e=[1,0,0,0,1,0,0,0,1],!!Kt(e,kt,Vt,Wt,Jt)&&(Zt.crossVectors(jt,qt),e=[Zt.x,Zt.y,Zt.z],Kt(e,kt,Vt,Wt,Jt)))}clampPoint(t,e){return e.copy(t).clamp(this.min,this.max)}distanceToPoint(t){return Ht.copy(t).clamp(this.min,this.max).sub(t).length()}getBoundingSphere(t){return this.getCenter(t.center),t.radius=.5*this.getSize(Ht).length(),t}intersect(t){return this.min.max(t.min),this.max.min(t.max),this.isEmpty()&&this.makeEmpty(),this}union(t){return this.min.min(t.min),this.max.max(t.max),this}applyMatrix4(t){return this.isEmpty()||(Ut[0].set(this.min.x,this.min.y,this.min.z).applyMatrix4(t),Ut[1].set(this.min.x,this.min.y,this.max.z).applyMatrix4(t),Ut[2].set(this.min.x,this.max.y,this.min.z).applyMatrix4(t),Ut[3].set(this.min.x,this.max.y,this.max.z).applyMatrix4(t),Ut[4].set(this.max.x,this.min.y,this.min.z).applyMatrix4(t),Ut[5].set(this.max.x,this.min.y,this.max.z).applyMatrix4(t),Ut[6].set(this.max.x,this.max.y,this.min.z).applyMatrix4(t),Ut[7].set(this.max.x,this.max.y,this.max.z).applyMatrix4(t),this.setFromPoints(Ut)),this}translate(t){return this.min.add(t),this.max.add(t),this}equals(t){return t.min.equals(this.min)&&t.max.equals(this.max)}}Ot.prototype.isBox3=!0;const Ut=[new zt,new zt,new zt,new zt,new zt,new zt,new zt,new zt],Ht=new zt,Gt=new Ot,kt=new zt,Vt=new zt,Wt=new zt,jt=new zt,qt=new zt,Xt=new zt,Yt=new zt,Jt=new zt,Zt=new zt,Qt=new zt;function Kt(t,e,n,i,r){for(let s=0,a=t.length-3;s<=a;s+=3){Qt.fromArray(t,s);const a=r.x*Math.abs(Qt.x)+r.y*Math.abs(Qt.y)+r.z*Math.abs(Qt.z),o=e.dot(Qt),l=n.dot(Qt),c=i.dot(Qt);if(Math.max(-Math.max(o,l,c),Math.min(o,l,c))>a)return!1}return!0}const $t=new Ot,te=new zt,ee=new zt,ne=new zt;class ie{constructor(t=new zt,e=-1){this.center=t,this.radius=e}set(t,e){return this.center.copy(t),this.radius=e,this}setFromPoints(t,e){const n=this.center;void 0!==e?n.copy(e):$t.setFromPoints(t).getCenter(n);let i=0;for(let e=0,r=t.length;e<r;e++)i=Math.max(i,n.distanceToSquared(t[e]));return this.radius=Math.sqrt(i),this}copy(t){return this.center.copy(t.center),this.radius=t.radius,this}isEmpty(){return this.radius<0}makeEmpty(){return this.center.set(0,0,0),this.radius=-1,this}containsPoint(t){return t.distanceToSquared(this.center)<=this.radius*this.radius}distanceToPoint(t){return t.distanceTo(this.center)-this.radius}intersectsSphere(t){const e=this.radius+t.radius;return t.center.distanceToSquared(this.center)<=e*e}intersectsBox(t){return t.intersectsSphere(this)}intersectsPlane(t){return Math.abs(t.distanceToPoint(this.center))<=this.radius}clampPoint(t,e){const n=this.center.distanceToSquared(t);return e.copy(t),n>this.radius*this.radius&&(e.sub(this.center).normalize(),e.multiplyScalar(this.radius).add(this.center)),e}getBoundingBox(t){return this.isEmpty()?(t.makeEmpty(),t):(t.set(this.center,this.center),t.expandByScalar(this.radius),t)}applyMatrix4(t){return this.center.applyMatrix4(t),this.radius=this.radius*t.getMaxScaleOnAxis(),this}translate(t){return this.center.add(t),this}expandByPoint(t){ne.subVectors(t,this.center);const e=ne.lengthSq();if(e>this.radius*this.radius){const t=Math.sqrt(e),n=.5*(t-this.radius);this.center.add(ne.multiplyScalar(n/t)),this.radius+=n}return this}union(t){return ee.subVectors(t.center,this.center).normalize().multiplyScalar(t.radius),this.expandByPoint(te.copy(t.center).add(ee)),this.expandByPoint(te.copy(t.center).sub(ee)),this}equals(t){return t.center.equals(this.center)&&t.radius===this.radius}clone(){return(new this.constructor).copy(this)}}const re=new zt,se=new zt,ae=new zt,oe=new zt,le=new zt,ce=new zt,he=new zt;class ue{constructor(t=new zt,e=new zt(0,0,-1)){this.origin=t,this.direction=e}set(t,e){return this.origin.copy(t),this.direction.copy(e),this}copy(t){return this.origin.copy(t.origin),this.direction.copy(t.direction),this}at(t,e){return e.copy(this.direction).multiplyScalar(t).add(this.origin)}lookAt(t){return this.direction.copy(t).sub(this.origin).normalize(),this}recast(t){return this.origin.copy(this.at(t,re)),this}closestPointToPoint(t,e){e.subVectors(t,this.origin);const n=e.dot(this.direction);return n<0?e.copy(this.origin):e.copy(this.direction).multiplyScalar(n).add(this.origin)}distanceToPoint(t){return Math.sqrt(this.distanceSqToPoint(t))}distanceSqToPoint(t){const e=re.subVectors(t,this.origin).dot(this.direction);return e<0?this.origin.distanceToSquared(t):(re.copy(this.direction).multiplyScalar(e).add(this.origin),re.distanceToSquared(t))}distanceSqToSegment(t,e,n,i){se.copy(t).add(e).multiplyScalar(.5),ae.copy(e).sub(t).normalize(),oe.copy(this.origin).sub(se);const r=.5*t.distanceTo(e),s=-this.direction.dot(ae),a=oe.dot(this.direction),o=-oe.dot(ae),l=oe.lengthSq(),c=Math.abs(1-s*s);let h,u,d,p;if(c>0)if(h=s*o-a,u=s*a-o,p=r*c,h>=0)if(u>=-p)if(u<=p){const t=1/c;h*=t,u*=t,d=h*(h+s*u+2*a)+u*(s*h+u+2*o)+l}else u=r,h=Math.max(0,-(s*u+a)),d=-h*h+u*(u+2*o)+l;else u=-r,h=Math.max(0,-(s*u+a)),d=-h*h+u*(u+2*o)+l;else u<=-p?(h=Math.max(0,-(-s*r+a)),u=h>0?-r:Math.min(Math.max(-r,-o),r),d=-h*h+u*(u+2*o)+l):u<=p?(h=0,u=Math.min(Math.max(-r,-o),r),d=u*(u+2*o)+l):(h=Math.max(0,-(s*r+a)),u=h>0?r:Math.min(Math.max(-r,-o),r),d=-h*h+u*(u+2*o)+l);else u=s>0?-r:r,h=Math.max(0,-(s*u+a)),d=-h*h+u*(u+2*o)+l;return n&&n.copy(this.direction).multiplyScalar(h).add(this.origin),i&&i.copy(ae).multiplyScalar(u).add(se),d}intersectSphere(t,e){re.subVectors(t.center,this.origin);const n=re.dot(this.direction),i=re.dot(re)-n*n,r=t.radius*t.radius;if(i>r)return null;const s=Math.sqrt(r-i),a=n-s,o=n+s;return a<0&&o<0?null:a<0?this.at(o,e):this.at(a,e)}intersectsSphere(t){return this.distanceSqToPoint(t.center)<=t.radius*t.radius}distanceToPlane(t){const e=t.normal.dot(this.direction);if(0===e)return 0===t.distanceToPoint(this.origin)?0:null;const n=-(this.origin.dot(t.normal)+t.constant)/e;return n>=0?n:null}intersectPlane(t,e){const n=this.distanceToPlane(t);return null===n?null:this.at(n,e)}intersectsPlane(t){const e=t.distanceToPoint(this.origin);if(0===e)return!0;return t.normal.dot(this.direction)*e<0}intersectBox(t,e){let n,i,r,s,a,o;const l=1/this.direction.x,c=1/this.direction.y,h=1/this.direction.z,u=this.origin;return l>=0?(n=(t.min.x-u.x)*l,i=(t.max.x-u.x)*l):(n=(t.max.x-u.x)*l,i=(t.min.x-u.x)*l),c>=0?(r=(t.min.y-u.y)*c,s=(t.max.y-u.y)*c):(r=(t.max.y-u.y)*c,s=(t.min.y-u.y)*c),n>s||r>i?null:((r>n||n!=n)&&(n=r),(s<i||i!=i)&&(i=s),h>=0?(a=(t.min.z-u.z)*h,o=(t.max.z-u.z)*h):(a=(t.max.z-u.z)*h,o=(t.min.z-u.z)*h),n>o||a>i?null:((a>n||n!=n)&&(n=a),(o<i||i!=i)&&(i=o),i<0?null:this.at(n>=0?n:i,e)))}intersectsBox(t){return null!==this.intersectBox(t,re)}intersectTriangle(t,e,n,i,r){le.subVectors(e,t),ce.subVectors(n,t),he.crossVectors(le,ce);let s,a=this.direction.dot(he);if(a>0){if(i)return null;s=1}else{if(!(a<0))return null;s=-1,a=-a}oe.subVectors(this.origin,t);const o=s*this.direction.dot(ce.crossVectors(oe,ce));if(o<0)return null;const l=s*this.direction.dot(le.cross(oe));if(l<0)return null;if(o+l>a)return null;const c=-s*oe.dot(he);return c<0?null:this.at(c/a,r)}applyMatrix4(t){return this.origin.applyMatrix4(t),this.direction.transformDirection(t),this}equals(t){return t.origin.equals(this.origin)&&t.direction.equals(this.direction)}clone(){return(new this.constructor).copy(this)}}class de{constructor(){this.elements=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],arguments.length>0&&console.error("THREE.Matrix4: the constructor no longer reads arguments. use .set() instead.")}set(t,e,n,i,r,s,a,o,l,c,h,u,d,p,m,f){const g=this.elements;return g[0]=t,g[4]=e,g[8]=n,g[12]=i,g[1]=r,g[5]=s,g[9]=a,g[13]=o,g[2]=l,g[6]=c,g[10]=h,g[14]=u,g[3]=d,g[7]=p,g[11]=m,g[15]=f,this}identity(){return this.set(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1),this}clone(){return(new de).fromArray(this.elements)}copy(t){const e=this.elements,n=t.elements;return e[0]=n[0],e[1]=n[1],e[2]=n[2],e[3]=n[3],e[4]=n[4],e[5]=n[5],e[6]=n[6],e[7]=n[7],e[8]=n[8],e[9]=n[9],e[10]=n[10],e[11]=n[11],e[12]=n[12],e[13]=n[13],e[14]=n[14],e[15]=n[15],this}copyPosition(t){const e=this.elements,n=t.elements;return e[12]=n[12],e[13]=n[13],e[14]=n[14],this}setFromMatrix3(t){const e=t.elements;return this.set(e[0],e[3],e[6],0,e[1],e[4],e[7],0,e[2],e[5],e[8],0,0,0,0,1),this}extractBasis(t,e,n){return t.setFromMatrixColumn(this,0),e.setFromMatrixColumn(this,1),n.setFromMatrixColumn(this,2),this}makeBasis(t,e,n){return this.set(t.x,e.x,n.x,0,t.y,e.y,n.y,0,t.z,e.z,n.z,0,0,0,0,1),this}extractRotation(t){const e=this.elements,n=t.elements,i=1/pe.setFromMatrixColumn(t,0).length(),r=1/pe.setFromMatrixColumn(t,1).length(),s=1/pe.setFromMatrixColumn(t,2).length();return e[0]=n[0]*i,e[1]=n[1]*i,e[2]=n[2]*i,e[3]=0,e[4]=n[4]*r,e[5]=n[5]*r,e[6]=n[6]*r,e[7]=0,e[8]=n[8]*s,e[9]=n[9]*s,e[10]=n[10]*s,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,this}makeRotationFromEuler(t){t&&t.isEuler||console.error("THREE.Matrix4: .makeRotationFromEuler() now expects a Euler rotation rather than a Vector3 and order.");const e=this.elements,n=t.x,i=t.y,r=t.z,s=Math.cos(n),a=Math.sin(n),o=Math.cos(i),l=Math.sin(i),c=Math.cos(r),h=Math.sin(r);if("XYZ"===t.order){const t=s*c,n=s*h,i=a*c,r=a*h;e[0]=o*c,e[4]=-o*h,e[8]=l,e[1]=n+i*l,e[5]=t-r*l,e[9]=-a*o,e[2]=r-t*l,e[6]=i+n*l,e[10]=s*o}else if("YXZ"===t.order){const t=o*c,n=o*h,i=l*c,r=l*h;e[0]=t+r*a,e[4]=i*a-n,e[8]=s*l,e[1]=s*h,e[5]=s*c,e[9]=-a,e[2]=n*a-i,e[6]=r+t*a,e[10]=s*o}else if("ZXY"===t.order){const t=o*c,n=o*h,i=l*c,r=l*h;e[0]=t-r*a,e[4]=-s*h,e[8]=i+n*a,e[1]=n+i*a,e[5]=s*c,e[9]=r-t*a,e[2]=-s*l,e[6]=a,e[10]=s*o}else if("ZYX"===t.order){const t=s*c,n=s*h,i=a*c,r=a*h;e[0]=o*c,e[4]=i*l-n,e[8]=t*l+r,e[1]=o*h,e[5]=r*l+t,e[9]=n*l-i,e[2]=-l,e[6]=a*o,e[10]=s*o}else if("YZX"===t.order){const t=s*o,n=s*l,i=a*o,r=a*l;e[0]=o*c,e[4]=r-t*h,e[8]=i*h+n,e[1]=h,e[5]=s*c,e[9]=-a*c,e[2]=-l*c,e[6]=n*h+i,e[10]=t-r*h}else if("XZY"===t.order){const t=s*o,n=s*l,i=a*o,r=a*l;e[0]=o*c,e[4]=-h,e[8]=l*c,e[1]=t*h+r,e[5]=s*c,e[9]=n*h-i,e[2]=i*h-n,e[6]=a*c,e[10]=r*h+t}return e[3]=0,e[7]=0,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,this}makeRotationFromQuaternion(t){return this.compose(fe,t,ge)}lookAt(t,e,n){const i=this.elements;return xe.subVectors(t,e),0===xe.lengthSq()&&(xe.z=1),xe.normalize(),ve.crossVectors(n,xe),0===ve.lengthSq()&&(1===Math.abs(n.z)?xe.x+=1e-4:xe.z+=1e-4,xe.normalize(),ve.crossVectors(n,xe)),ve.normalize(),ye.crossVectors(xe,ve),i[0]=ve.x,i[4]=ye.x,i[8]=xe.x,i[1]=ve.y,i[5]=ye.y,i[9]=xe.y,i[2]=ve.z,i[6]=ye.z,i[10]=xe.z,this}multiply(t,e){return void 0!==e?(console.warn("THREE.Matrix4: .multiply() now only accepts one argument. Use .multiplyMatrices( a, b ) instead."),this.multiplyMatrices(t,e)):this.multiplyMatrices(this,t)}premultiply(t){return this.multiplyMatrices(t,this)}multiplyMatrices(t,e){const n=t.elements,i=e.elements,r=this.elements,s=n[0],a=n[4],o=n[8],l=n[12],c=n[1],h=n[5],u=n[9],d=n[13],p=n[2],m=n[6],f=n[10],g=n[14],v=n[3],y=n[7],x=n[11],_=n[15],M=i[0],b=i[4],w=i[8],S=i[12],T=i[1],E=i[5],A=i[9],L=i[13],R=i[2],C=i[6],P=i[10],I=i[14],D=i[3],N=i[7],z=i[11],B=i[15];return r[0]=s*M+a*T+o*R+l*D,r[4]=s*b+a*E+o*C+l*N,r[8]=s*w+a*A+o*P+l*z,r[12]=s*S+a*L+o*I+l*B,r[1]=c*M+h*T+u*R+d*D,r[5]=c*b+h*E+u*C+d*N,r[9]=c*w+h*A+u*P+d*z,r[13]=c*S+h*L+u*I+d*B,r[2]=p*M+m*T+f*R+g*D,r[6]=p*b+m*E+f*C+g*N,r[10]=p*w+m*A+f*P+g*z,r[14]=p*S+m*L+f*I+g*B,r[3]=v*M+y*T+x*R+_*D,r[7]=v*b+y*E+x*C+_*N,r[11]=v*w+y*A+x*P+_*z,r[15]=v*S+y*L+x*I+_*B,this}multiplyScalar(t){const e=this.elements;return e[0]*=t,e[4]*=t,e[8]*=t,e[12]*=t,e[1]*=t,e[5]*=t,e[9]*=t,e[13]*=t,e[2]*=t,e[6]*=t,e[10]*=t,e[14]*=t,e[3]*=t,e[7]*=t,e[11]*=t,e[15]*=t,this}determinant(){const t=this.elements,e=t[0],n=t[4],i=t[8],r=t[12],s=t[1],a=t[5],o=t[9],l=t[13],c=t[2],h=t[6],u=t[10],d=t[14];return t[3]*(+r*o*h-i*l*h-r*a*u+n*l*u+i*a*d-n*o*d)+t[7]*(+e*o*d-e*l*u+r*s*u-i*s*d+i*l*c-r*o*c)+t[11]*(+e*l*h-e*a*d-r*s*h+n*s*d+r*a*c-n*l*c)+t[15]*(-i*a*c-e*o*h+e*a*u+i*s*h-n*s*u+n*o*c)}transpose(){const t=this.elements;let e;return e=t[1],t[1]=t[4],t[4]=e,e=t[2],t[2]=t[8],t[8]=e,e=t[6],t[6]=t[9],t[9]=e,e=t[3],t[3]=t[12],t[12]=e,e=t[7],t[7]=t[13],t[13]=e,e=t[11],t[11]=t[14],t[14]=e,this}setPosition(t,e,n){const i=this.elements;return t.isVector3?(i[12]=t.x,i[13]=t.y,i[14]=t.z):(i[12]=t,i[13]=e,i[14]=n),this}invert(){const t=this.elements,e=t[0],n=t[1],i=t[2],r=t[3],s=t[4],a=t[5],o=t[6],l=t[7],c=t[8],h=t[9],u=t[10],d=t[11],p=t[12],m=t[13],f=t[14],g=t[15],v=h*f*l-m*u*l+m*o*d-a*f*d-h*o*g+a*u*g,y=p*u*l-c*f*l-p*o*d+s*f*d+c*o*g-s*u*g,x=c*m*l-p*h*l+p*a*d-s*m*d-c*a*g+s*h*g,_=p*h*o-c*m*o-p*a*u+s*m*u+c*a*f-s*h*f,M=e*v+n*y+i*x+r*_;if(0===M)return this.set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);const b=1/M;return t[0]=v*b,t[1]=(m*u*r-h*f*r-m*i*d+n*f*d+h*i*g-n*u*g)*b,t[2]=(a*f*r-m*o*r+m*i*l-n*f*l-a*i*g+n*o*g)*b,t[3]=(h*o*r-a*u*r-h*i*l+n*u*l+a*i*d-n*o*d)*b,t[4]=y*b,t[5]=(c*f*r-p*u*r+p*i*d-e*f*d-c*i*g+e*u*g)*b,t[6]=(p*o*r-s*f*r-p*i*l+e*f*l+s*i*g-e*o*g)*b,t[7]=(s*u*r-c*o*r+c*i*l-e*u*l-s*i*d+e*o*d)*b,t[8]=x*b,t[9]=(p*h*r-c*m*r-p*n*d+e*m*d+c*n*g-e*h*g)*b,t[10]=(s*m*r-p*a*r+p*n*l-e*m*l-s*n*g+e*a*g)*b,t[11]=(c*a*r-s*h*r-c*n*l+e*h*l+s*n*d-e*a*d)*b,t[12]=_*b,t[13]=(c*m*i-p*h*i+p*n*u-e*m*u-c*n*f+e*h*f)*b,t[14]=(p*a*i-s*m*i-p*n*o+e*m*o+s*n*f-e*a*f)*b,t[15]=(s*h*i-c*a*i+c*n*o-e*h*o-s*n*u+e*a*u)*b,this}scale(t){const e=this.elements,n=t.x,i=t.y,r=t.z;return e[0]*=n,e[4]*=i,e[8]*=r,e[1]*=n,e[5]*=i,e[9]*=r,e[2]*=n,e[6]*=i,e[10]*=r,e[3]*=n,e[7]*=i,e[11]*=r,this}getMaxScaleOnAxis(){const t=this.elements,e=t[0]*t[0]+t[1]*t[1]+t[2]*t[2],n=t[4]*t[4]+t[5]*t[5]+t[6]*t[6],i=t[8]*t[8]+t[9]*t[9]+t[10]*t[10];return Math.sqrt(Math.max(e,n,i))}makeTranslation(t,e,n){return this.set(1,0,0,t,0,1,0,e,0,0,1,n,0,0,0,1),this}makeRotationX(t){const e=Math.cos(t),n=Math.sin(t);return this.set(1,0,0,0,0,e,-n,0,0,n,e,0,0,0,0,1),this}makeRotationY(t){const e=Math.cos(t),n=Math.sin(t);return this.set(e,0,n,0,0,1,0,0,-n,0,e,0,0,0,0,1),this}makeRotationZ(t){const e=Math.cos(t),n=Math.sin(t);return this.set(e,-n,0,0,n,e,0,0,0,0,1,0,0,0,0,1),this}makeRotationAxis(t,e){const n=Math.cos(e),i=Math.sin(e),r=1-n,s=t.x,a=t.y,o=t.z,l=r*s,c=r*a;return this.set(l*s+n,l*a-i*o,l*o+i*a,0,l*a+i*o,c*a+n,c*o-i*s,0,l*o-i*a,c*o+i*s,r*o*o+n,0,0,0,0,1),this}makeScale(t,e,n){return this.set(t,0,0,0,0,e,0,0,0,0,n,0,0,0,0,1),this}makeShear(t,e,n,i,r,s){return this.set(1,n,r,0,t,1,s,0,e,i,1,0,0,0,0,1),this}compose(t,e,n){const i=this.elements,r=e._x,s=e._y,a=e._z,o=e._w,l=r+r,c=s+s,h=a+a,u=r*l,d=r*c,p=r*h,m=s*c,f=s*h,g=a*h,v=o*l,y=o*c,x=o*h,_=n.x,M=n.y,b=n.z;return i[0]=(1-(m+g))*_,i[1]=(d+x)*_,i[2]=(p-y)*_,i[3]=0,i[4]=(d-x)*M,i[5]=(1-(u+g))*M,i[6]=(f+v)*M,i[7]=0,i[8]=(p+y)*b,i[9]=(f-v)*b,i[10]=(1-(u+m))*b,i[11]=0,i[12]=t.x,i[13]=t.y,i[14]=t.z,i[15]=1,this}decompose(t,e,n){const i=this.elements;let r=pe.set(i[0],i[1],i[2]).length();const s=pe.set(i[4],i[5],i[6]).length(),a=pe.set(i[8],i[9],i[10]).length();this.determinant()<0&&(r=-r),t.x=i[12],t.y=i[13],t.z=i[14],me.copy(this);const o=1/r,l=1/s,c=1/a;return me.elements[0]*=o,me.elements[1]*=o,me.elements[2]*=o,me.elements[4]*=l,me.elements[5]*=l,me.elements[6]*=l,me.elements[8]*=c,me.elements[9]*=c,me.elements[10]*=c,e.setFromRotationMatrix(me),n.x=r,n.y=s,n.z=a,this}makePerspective(t,e,n,i,r,s){void 0===s&&console.warn("THREE.Matrix4: .makePerspective() has been redefined and has a new signature. Please check the docs.");const a=this.elements,o=2*r/(e-t),l=2*r/(n-i),c=(e+t)/(e-t),h=(n+i)/(n-i),u=-(s+r)/(s-r),d=-2*s*r/(s-r);return a[0]=o,a[4]=0,a[8]=c,a[12]=0,a[1]=0,a[5]=l,a[9]=h,a[13]=0,a[2]=0,a[6]=0,a[10]=u,a[14]=d,a[3]=0,a[7]=0,a[11]=-1,a[15]=0,this}makeOrthographic(t,e,n,i,r,s){const a=this.elements,o=1/(e-t),l=1/(n-i),c=1/(s-r),h=(e+t)*o,u=(n+i)*l,d=(s+r)*c;return a[0]=2*o,a[4]=0,a[8]=0,a[12]=-h,a[1]=0,a[5]=2*l,a[9]=0,a[13]=-u,a[2]=0,a[6]=0,a[10]=-2*c,a[14]=-d,a[3]=0,a[7]=0,a[11]=0,a[15]=1,this}equals(t){const e=this.elements,n=t.elements;for(let t=0;t<16;t++)if(e[t]!==n[t])return!1;return!0}fromArray(t,e=0){for(let n=0;n<16;n++)this.elements[n]=t[n+e];return this}toArray(t=[],e=0){const n=this.elements;return t[e]=n[0],t[e+1]=n[1],t[e+2]=n[2],t[e+3]=n[3],t[e+4]=n[4],t[e+5]=n[5],t[e+6]=n[6],t[e+7]=n[7],t[e+8]=n[8],t[e+9]=n[9],t[e+10]=n[10],t[e+11]=n[11],t[e+12]=n[12],t[e+13]=n[13],t[e+14]=n[14],t[e+15]=n[15],t}}de.prototype.isMatrix4=!0;const pe=new zt,me=new de,fe=new zt(0,0,0),ge=new zt(1,1,1),ve=new zt,ye=new zt,xe=new zt,_e=new de,Me=new Nt;class be{constructor(t=0,e=0,n=0,i=be.DefaultOrder){this._x=t,this._y=e,this._z=n,this._order=i}get x(){return this._x}set x(t){this._x=t,this._onChangeCallback()}get y(){return this._y}set y(t){this._y=t,this._onChangeCallback()}get z(){return this._z}set z(t){this._z=t,this._onChangeCallback()}get order(){return this._order}set order(t){this._order=t,this._onChangeCallback()}set(t,e,n,i=this._order){return this._x=t,this._y=e,this._z=n,this._order=i,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._order)}copy(t){return this._x=t._x,this._y=t._y,this._z=t._z,this._order=t._order,this._onChangeCallback(),this}setFromRotationMatrix(t,e=this._order,n=!0){const i=t.elements,r=i[0],s=i[4],a=i[8],o=i[1],l=i[5],c=i[9],h=i[2],u=i[6],d=i[10];switch(e){case"XYZ":this._y=Math.asin(ut(a,-1,1)),Math.abs(a)<.9999999?(this._x=Math.atan2(-c,d),this._z=Math.atan2(-s,r)):(this._x=Math.atan2(u,l),this._z=0);break;case"YXZ":this._x=Math.asin(-ut(c,-1,1)),Math.abs(c)<.9999999?(this._y=Math.atan2(a,d),this._z=Math.atan2(o,l)):(this._y=Math.atan2(-h,r),this._z=0);break;case"ZXY":this._x=Math.asin(ut(u,-1,1)),Math.abs(u)<.9999999?(this._y=Math.atan2(-h,d),this._z=Math.atan2(-s,l)):(this._y=0,this._z=Math.atan2(o,r));break;case"ZYX":this._y=Math.asin(-ut(h,-1,1)),Math.abs(h)<.9999999?(this._x=Math.atan2(u,d),this._z=Math.atan2(o,r)):(this._x=0,this._z=Math.atan2(-s,l));break;case"YZX":this._z=Math.asin(ut(o,-1,1)),Math.abs(o)<.9999999?(this._x=Math.atan2(-c,l),this._y=Math.atan2(-h,r)):(this._x=0,this._y=Math.atan2(a,d));break;case"XZY":this._z=Math.asin(-ut(s,-1,1)),Math.abs(s)<.9999999?(this._x=Math.atan2(u,l),this._y=Math.atan2(a,r)):(this._x=Math.atan2(-c,d),this._y=0);break;default:console.warn("THREE.Euler: .setFromRotationMatrix() encountered an unknown order: "+e)}return this._order=e,!0===n&&this._onChangeCallback(),this}setFromQuaternion(t,e,n){return _e.makeRotationFromQuaternion(t),this.setFromRotationMatrix(_e,e,n)}setFromVector3(t,e=this._order){return this.set(t.x,t.y,t.z,e)}reorder(t){return Me.setFromEuler(this),this.setFromQuaternion(Me,t)}equals(t){return t._x===this._x&&t._y===this._y&&t._z===this._z&&t._order===this._order}fromArray(t){return this._x=t[0],this._y=t[1],this._z=t[2],void 0!==t[3]&&(this._order=t[3]),this._onChangeCallback(),this}toArray(t=[],e=0){return t[e]=this._x,t[e+1]=this._y,t[e+2]=this._z,t[e+3]=this._order,t}toVector3(t){return t?t.set(this._x,this._y,this._z):new zt(this._x,this._y,this._z)}_onChange(t){return this._onChangeCallback=t,this}_onChangeCallback(){}}be.prototype.isEuler=!0,be.DefaultOrder="XYZ",be.RotationOrders=["XYZ","YZX","ZXY","XZY","YXZ","ZYX"];class we{constructor(){this.mask=1}set(t){this.mask=1<<t|0}enable(t){this.mask|=1<<t|0}enableAll(){this.mask=-1}toggle(t){this.mask^=1<<t|0}disable(t){this.mask&=~(1<<t|0)}disableAll(){this.mask=0}test(t){return 0!=(this.mask&t.mask)}}let Se=0;const Te=new zt,Ee=new Nt,Ae=new de,Le=new zt,Re=new zt,Ce=new zt,Pe=new Nt,Ie=new zt(1,0,0),De=new zt(0,1,0),Ne=new zt(0,0,1),ze={type:"added"},Be={type:"removed"};class Fe extends rt{constructor(){super(),Object.defineProperty(this,"id",{value:Se++}),this.uuid=ht(),this.name="",this.type="Object3D",this.parent=null,this.children=[],this.up=Fe.DefaultUp.clone();const t=new zt,e=new be,n=new Nt,i=new zt(1,1,1);e._onChange((function(){n.setFromEuler(e,!1)})),n._onChange((function(){e.setFromQuaternion(n,void 0,!1)})),Object.defineProperties(this,{position:{configurable:!0,enumerable:!0,value:t},rotation:{configurable:!0,enumerable:!0,value:e},quaternion:{configurable:!0,enumerable:!0,value:n},scale:{configurable:!0,enumerable:!0,value:i},modelViewMatrix:{value:new de},normalMatrix:{value:new xt}}),this.matrix=new de,this.matrixWorld=new de,this.matrixAutoUpdate=Fe.DefaultMatrixAutoUpdate,this.matrixWorldNeedsUpdate=!1,this.layers=new we,this.visible=!0,this.castShadow=!1,this.receiveShadow=!1,this.frustumCulled=!0,this.renderOrder=0,this.animations=[],this.userData={}}onBeforeRender(){}onAfterRender(){}applyMatrix4(t){this.matrixAutoUpdate&&this.updateMatrix(),this.matrix.premultiply(t),this.matrix.decompose(this.position,this.quaternion,this.scale)}applyQuaternion(t){return this.quaternion.premultiply(t),this}setRotationFromAxisAngle(t,e){this.quaternion.setFromAxisAngle(t,e)}setRotationFromEuler(t){this.quaternion.setFromEuler(t,!0)}setRotationFromMatrix(t){this.quaternion.setFromRotationMatrix(t)}setRotationFromQuaternion(t){this.quaternion.copy(t)}rotateOnAxis(t,e){return Ee.setFromAxisAngle(t,e),this.quaternion.multiply(Ee),this}rotateOnWorldAxis(t,e){return Ee.setFromAxisAngle(t,e),this.quaternion.premultiply(Ee),this}rotateX(t){return this.rotateOnAxis(Ie,t)}rotateY(t){return this.rotateOnAxis(De,t)}rotateZ(t){return this.rotateOnAxis(Ne,t)}translateOnAxis(t,e){return Te.copy(t).applyQuaternion(this.quaternion),this.position.add(Te.multiplyScalar(e)),this}translateX(t){return this.translateOnAxis(Ie,t)}translateY(t){return this.translateOnAxis(De,t)}translateZ(t){return this.translateOnAxis(Ne,t)}localToWorld(t){return t.applyMatrix4(this.matrixWorld)}worldToLocal(t){return t.applyMatrix4(Ae.copy(this.matrixWorld).invert())}lookAt(t,e,n){t.isVector3?Le.copy(t):Le.set(t,e,n);const i=this.parent;this.updateWorldMatrix(!0,!1),Re.setFromMatrixPosition(this.matrixWorld),this.isCamera||this.isLight?Ae.lookAt(Re,Le,this.up):Ae.lookAt(Le,Re,this.up),this.quaternion.setFromRotationMatrix(Ae),i&&(Ae.extractRotation(i.matrixWorld),Ee.setFromRotationMatrix(Ae),this.quaternion.premultiply(Ee.invert()))}add(t){if(arguments.length>1){for(let t=0;t<arguments.length;t++)this.add(arguments[t]);return this}return t===this?(console.error("THREE.Object3D.add: object can't be added as a child of itself.",t),this):(t&&t.isObject3D?(null!==t.parent&&t.parent.remove(t),t.parent=this,this.children.push(t),t.dispatchEvent(ze)):console.error("THREE.Object3D.add: object not an instance of THREE.Object3D.",t),this)}remove(t){if(arguments.length>1){for(let t=0;t<arguments.length;t++)this.remove(arguments[t]);return this}const e=this.children.indexOf(t);return-1!==e&&(t.parent=null,this.children.splice(e,1),t.dispatchEvent(Be)),this}removeFromParent(){const t=this.parent;return null!==t&&t.remove(this),this}clear(){for(let t=0;t<this.children.length;t++){const e=this.children[t];e.parent=null,e.dispatchEvent(Be)}return this.children.length=0,this}attach(t){return this.updateWorldMatrix(!0,!1),Ae.copy(this.matrixWorld).invert(),null!==t.parent&&(t.parent.updateWorldMatrix(!0,!1),Ae.multiply(t.parent.matrixWorld)),t.applyMatrix4(Ae),this.add(t),t.updateWorldMatrix(!1,!0),this}getObjectById(t){return this.getObjectByProperty("id",t)}getObjectByName(t){return this.getObjectByProperty("name",t)}getObjectByProperty(t,e){if(this[t]===e)return this;for(let n=0,i=this.children.length;n<i;n++){const i=this.children[n].getObjectByProperty(t,e);if(void 0!==i)return i}}getWorldPosition(t){return this.updateWorldMatrix(!0,!1),t.setFromMatrixPosition(this.matrixWorld)}getWorldQuaternion(t){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(Re,t,Ce),t}getWorldScale(t){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(Re,Pe,t),t}getWorldDirection(t){this.updateWorldMatrix(!0,!1);const e=this.matrixWorld.elements;return t.set(e[8],e[9],e[10]).normalize()}raycast(){}traverse(t){t(this);const e=this.children;for(let n=0,i=e.length;n<i;n++)e[n].traverse(t)}traverseVisible(t){if(!1===this.visible)return;t(this);const e=this.children;for(let n=0,i=e.length;n<i;n++)e[n].traverseVisible(t)}traverseAncestors(t){const e=this.parent;null!==e&&(t(e),e.traverseAncestors(t))}updateMatrix(){this.matrix.compose(this.position,this.quaternion,this.scale),this.matrixWorldNeedsUpdate=!0}updateMatrixWorld(t){this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||t)&&(null===this.parent?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix),this.matrixWorldNeedsUpdate=!1,t=!0);const e=this.children;for(let n=0,i=e.length;n<i;n++)e[n].updateMatrixWorld(t)}updateWorldMatrix(t,e){const n=this.parent;if(!0===t&&null!==n&&n.updateWorldMatrix(!0,!1),this.matrixAutoUpdate&&this.updateMatrix(),null===this.parent?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix),!0===e){const t=this.children;for(let e=0,n=t.length;e<n;e++)t[e].updateWorldMatrix(!1,!0)}}toJSON(t){const e=void 0===t||"string"==typeof t,n={};e&&(t={geometries:{},materials:{},textures:{},images:{},shapes:{},skeletons:{},animations:{}},n.metadata={version:4.5,type:"Object",generator:"Object3D.toJSON"});const i={};function r(e,n){return void 0===e[n.uuid]&&(e[n.uuid]=n.toJSON(t)),n.uuid}if(i.uuid=this.uuid,i.type=this.type,""!==this.name&&(i.name=this.name),!0===this.castShadow&&(i.castShadow=!0),!0===this.receiveShadow&&(i.receiveShadow=!0),!1===this.visible&&(i.visible=!1),!1===this.frustumCulled&&(i.frustumCulled=!1),0!==this.renderOrder&&(i.renderOrder=this.renderOrder),"{}"!==JSON.stringify(this.userData)&&(i.userData=this.userData),i.layers=this.layers.mask,i.matrix=this.matrix.toArray(),!1===this.matrixAutoUpdate&&(i.matrixAutoUpdate=!1),this.isInstancedMesh&&(i.type="InstancedMesh",i.count=this.count,i.instanceMatrix=this.instanceMatrix.toJSON(),null!==this.instanceColor&&(i.instanceColor=this.instanceColor.toJSON())),this.isScene)this.background&&(this.background.isColor?i.background=this.background.toJSON():this.background.isTexture&&(i.background=this.background.toJSON(t).uuid)),this.environment&&this.environment.isTexture&&(i.environment=this.environment.toJSON(t).uuid);else if(this.isMesh||this.isLine||this.isPoints){i.geometry=r(t.geometries,this.geometry);const e=this.geometry.parameters;if(void 0!==e&&void 0!==e.shapes){const n=e.shapes;if(Array.isArray(n))for(let e=0,i=n.length;e<i;e++){const i=n[e];r(t.shapes,i)}else r(t.shapes,n)}}if(this.isSkinnedMesh&&(i.bindMode=this.bindMode,i.bindMatrix=this.bindMatrix.toArray(),void 0!==this.skeleton&&(r(t.skeletons,this.skeleton),i.skeleton=this.skeleton.uuid)),void 0!==this.material)if(Array.isArray(this.material)){const e=[];for(let n=0,i=this.material.length;n<i;n++)e.push(r(t.materials,this.material[n]));i.material=e}else i.material=r(t.materials,this.material);if(this.children.length>0){i.children=[];for(let e=0;e<this.children.length;e++)i.children.push(this.children[e].toJSON(t).object)}if(this.animations.length>0){i.animations=[];for(let e=0;e<this.animations.length;e++){const n=this.animations[e];i.animations.push(r(t.animations,n))}}if(e){const e=s(t.geometries),i=s(t.materials),r=s(t.textures),a=s(t.images),o=s(t.shapes),l=s(t.skeletons),c=s(t.animations);e.length>0&&(n.geometries=e),i.length>0&&(n.materials=i),r.length>0&&(n.textures=r),a.length>0&&(n.images=a),o.length>0&&(n.shapes=o),l.length>0&&(n.skeletons=l),c.length>0&&(n.animations=c)}return n.object=i,n;function s(t){const e=[];for(const n in t){const i=t[n];delete i.metadata,e.push(i)}return e}}clone(t){return(new this.constructor).copy(this,t)}copy(t,e=!0){if(this.name=t.name,this.up.copy(t.up),this.position.copy(t.position),this.rotation.order=t.rotation.order,this.quaternion.copy(t.quaternion),this.scale.copy(t.scale),this.matrix.copy(t.matrix),this.matrixWorld.copy(t.matrixWorld),this.matrixAutoUpdate=t.matrixAutoUpdate,this.matrixWorldNeedsUpdate=t.matrixWorldNeedsUpdate,this.layers.mask=t.layers.mask,this.visible=t.visible,this.castShadow=t.castShadow,this.receiveShadow=t.receiveShadow,this.frustumCulled=t.frustumCulled,this.renderOrder=t.renderOrder,this.userData=JSON.parse(JSON.stringify(t.userData)),!0===e)for(let e=0;e<t.children.length;e++){const n=t.children[e];this.add(n.clone())}return this}}Fe.DefaultUp=new zt(0,1,0),Fe.DefaultMatrixAutoUpdate=!0,Fe.prototype.isObject3D=!0;const Oe=new zt,Ue=new zt,He=new zt,Ge=new zt,ke=new zt,Ve=new zt,We=new zt,je=new zt,qe=new zt,Xe=new zt;class Ye{constructor(t=new zt,e=new zt,n=new zt){this.a=t,this.b=e,this.c=n}static getNormal(t,e,n,i){i.subVectors(n,e),Oe.subVectors(t,e),i.cross(Oe);const r=i.lengthSq();return r>0?i.multiplyScalar(1/Math.sqrt(r)):i.set(0,0,0)}static getBarycoord(t,e,n,i,r){Oe.subVectors(i,e),Ue.subVectors(n,e),He.subVectors(t,e);const s=Oe.dot(Oe),a=Oe.dot(Ue),o=Oe.dot(He),l=Ue.dot(Ue),c=Ue.dot(He),h=s*l-a*a;if(0===h)return r.set(-2,-1,-1);const u=1/h,d=(l*o-a*c)*u,p=(s*c-a*o)*u;return r.set(1-d-p,p,d)}static containsPoint(t,e,n,i){return this.getBarycoord(t,e,n,i,Ge),Ge.x>=0&&Ge.y>=0&&Ge.x+Ge.y<=1}static getUV(t,e,n,i,r,s,a,o){return this.getBarycoord(t,e,n,i,Ge),o.set(0,0),o.addScaledVector(r,Ge.x),o.addScaledVector(s,Ge.y),o.addScaledVector(a,Ge.z),o}static isFrontFacing(t,e,n,i){return Oe.subVectors(n,e),Ue.subVectors(t,e),Oe.cross(Ue).dot(i)<0}set(t,e,n){return this.a.copy(t),this.b.copy(e),this.c.copy(n),this}setFromPointsAndIndices(t,e,n,i){return this.a.copy(t[e]),this.b.copy(t[n]),this.c.copy(t[i]),this}setFromAttributeAndIndices(t,e,n,i){return this.a.fromBufferAttribute(t,e),this.b.fromBufferAttribute(t,n),this.c.fromBufferAttribute(t,i),this}clone(){return(new this.constructor).copy(this)}copy(t){return this.a.copy(t.a),this.b.copy(t.b),this.c.copy(t.c),this}getArea(){return Oe.subVectors(this.c,this.b),Ue.subVectors(this.a,this.b),.5*Oe.cross(Ue).length()}getMidpoint(t){return t.addVectors(this.a,this.b).add(this.c).multiplyScalar(1/3)}getNormal(t){return Ye.getNormal(this.a,this.b,this.c,t)}getPlane(t){return t.setFromCoplanarPoints(this.a,this.b,this.c)}getBarycoord(t,e){return Ye.getBarycoord(t,this.a,this.b,this.c,e)}getUV(t,e,n,i,r){return Ye.getUV(t,this.a,this.b,this.c,e,n,i,r)}containsPoint(t){return Ye.containsPoint(t,this.a,this.b,this.c)}isFrontFacing(t){return Ye.isFrontFacing(this.a,this.b,this.c,t)}intersectsBox(t){return t.intersectsTriangle(this)}closestPointToPoint(t,e){const n=this.a,i=this.b,r=this.c;let s,a;ke.subVectors(i,n),Ve.subVectors(r,n),je.subVectors(t,n);const o=ke.dot(je),l=Ve.dot(je);if(o<=0&&l<=0)return e.copy(n);qe.subVectors(t,i);const c=ke.dot(qe),h=Ve.dot(qe);if(c>=0&&h<=c)return e.copy(i);const u=o*h-c*l;if(u<=0&&o>=0&&c<=0)return s=o/(o-c),e.copy(n).addScaledVector(ke,s);Xe.subVectors(t,r);const d=ke.dot(Xe),p=Ve.dot(Xe);if(p>=0&&d<=p)return e.copy(r);const m=d*l-o*p;if(m<=0&&l>=0&&p<=0)return a=l/(l-p),e.copy(n).addScaledVector(Ve,a);const f=c*p-d*h;if(f<=0&&h-c>=0&&d-p>=0)return We.subVectors(r,i),a=(h-c)/(h-c+(d-p)),e.copy(i).addScaledVector(We,a);const g=1/(f+m+u);return s=m*g,a=u*g,e.copy(n).addScaledVector(ke,s).addScaledVector(Ve,a)}equals(t){return t.a.equals(this.a)&&t.b.equals(this.b)&&t.c.equals(this.c)}}let Je=0;class Ze extends rt{constructor(){super(),Object.defineProperty(this,"id",{value:Je++}),this.uuid=ht(),this.name="",this.type="Material",this.fog=!0,this.blending=1,this.side=0,this.vertexColors=!1,this.opacity=1,this.format=E,this.transparent=!1,this.blendSrc=204,this.blendDst=205,this.blendEquation=n,this.blendSrcAlpha=null,this.blendDstAlpha=null,this.blendEquationAlpha=null,this.depthFunc=3,this.depthTest=!0,this.depthWrite=!0,this.stencilWriteMask=255,this.stencilFunc=519,this.stencilRef=0,this.stencilFuncMask=255,this.stencilFail=tt,this.stencilZFail=tt,this.stencilZPass=tt,this.stencilWrite=!1,this.clippingPlanes=null,this.clipIntersection=!1,this.clipShadows=!1,this.shadowSide=null,this.colorWrite=!0,this.precision=null,this.polygonOffset=!1,this.polygonOffsetFactor=0,this.polygonOffsetUnits=0,this.dithering=!1,this.alphaToCoverage=!1,this.premultipliedAlpha=!1,this.visible=!0,this.toneMapped=!0,this.userData={},this.version=0,this._alphaTest=0}get alphaTest(){return this._alphaTest}set alphaTest(t){this._alphaTest>0!=t>0&&this.version++,this._alphaTest=t}onBuild(){}onBeforeRender(){}onBeforeCompile(){}customProgramCacheKey(){return this.onBeforeCompile.toString()}setValues(t){if(void 0!==t)for(const e in t){const n=t[e];if(void 0===n){console.warn("THREE.Material: '"+e+"' parameter is undefined.");continue}if("shading"===e){console.warn("THREE."+this.type+": .shading has been removed. Use the boolean .flatShading instead."),this.flatShading=1===n;continue}const i=this[e];void 0!==i?i&&i.isColor?i.set(n):i&&i.isVector3&&n&&n.isVector3?i.copy(n):this[e]=n:console.warn("THREE."+this.type+": '"+e+"' is not a property of this material.")}}toJSON(t){const e=void 0===t||"string"==typeof t;e&&(t={textures:{},images:{}});const n={metadata:{version:4.5,type:"Material",generator:"Material.toJSON"}};function i(t){const e=[];for(const n in t){const i=t[n];delete i.metadata,e.push(i)}return e}if(n.uuid=this.uuid,n.type=this.type,""!==this.name&&(n.name=this.name),this.color&&this.color.isColor&&(n.color=this.color.getHex()),void 0!==this.roughness&&(n.roughness=this.roughness),void 0!==this.metalness&&(n.metalness=this.metalness),void 0!==this.sheen&&(n.sheen=this.sheen),this.sheenColor&&this.sheenColor.isColor&&(n.sheenColor=this.sheenColor.getHex()),void 0!==this.sheenRoughness&&(n.sheenRoughness=this.sheenRoughness),this.emissive&&this.emissive.isColor&&(n.emissive=this.emissive.getHex()),this.emissiveIntensity&&1!==this.emissiveIntensity&&(n.emissiveIntensity=this.emissiveIntensity),this.specular&&this.specular.isColor&&(n.specular=this.specular.getHex()),void 0!==this.specularIntensity&&(n.specularIntensity=this.specularIntensity),this.specularColor&&this.specularColor.isColor&&(n.specularColor=this.specularColor.getHex()),void 0!==this.shininess&&(n.shininess=this.shininess),void 0!==this.clearcoat&&(n.clearcoat=this.clearcoat),void 0!==this.clearcoatRoughness&&(n.clearcoatRoughness=this.clearcoatRoughness),this.clearcoatMap&&this.clearcoatMap.isTexture&&(n.clearcoatMap=this.clearcoatMap.toJSON(t).uuid),this.clearcoatRoughnessMap&&this.clearcoatRoughnessMap.isTexture&&(n.clearcoatRoughnessMap=this.clearcoatRoughnessMap.toJSON(t).uuid),this.clearcoatNormalMap&&this.clearcoatNormalMap.isTexture&&(n.clearcoatNormalMap=this.clearcoatNormalMap.toJSON(t).uuid,n.clearcoatNormalScale=this.clearcoatNormalScale.toArray()),this.map&&this.map.isTexture&&(n.map=this.map.toJSON(t).uuid),this.matcap&&this.matcap.isTexture&&(n.matcap=this.matcap.toJSON(t).uuid),this.alphaMap&&this.alphaMap.isTexture&&(n.alphaMap=this.alphaMap.toJSON(t).uuid),this.lightMap&&this.lightMap.isTexture&&(n.lightMap=this.lightMap.toJSON(t).uuid,n.lightMapIntensity=this.lightMapIntensity),this.aoMap&&this.aoMap.isTexture&&(n.aoMap=this.aoMap.toJSON(t).uuid,n.aoMapIntensity=this.aoMapIntensity),this.bumpMap&&this.bumpMap.isTexture&&(n.bumpMap=this.bumpMap.toJSON(t).uuid,n.bumpScale=this.bumpScale),this.normalMap&&this.normalMap.isTexture&&(n.normalMap=this.normalMap.toJSON(t).uuid,n.normalMapType=this.normalMapType,n.normalScale=this.normalScale.toArray()),this.displacementMap&&this.displacementMap.isTexture&&(n.displacementMap=this.displacementMap.toJSON(t).uuid,n.displacementScale=this.displacementScale,n.displacementBias=this.displacementBias),this.roughnessMap&&this.roughnessMap.isTexture&&(n.roughnessMap=this.roughnessMap.toJSON(t).uuid),this.metalnessMap&&this.metalnessMap.isTexture&&(n.metalnessMap=this.metalnessMap.toJSON(t).uuid),this.emissiveMap&&this.emissiveMap.isTexture&&(n.emissiveMap=this.emissiveMap.toJSON(t).uuid),this.specularMap&&this.specularMap.isTexture&&(n.specularMap=this.specularMap.toJSON(t).uuid),this.specularIntensityMap&&this.specularIntensityMap.isTexture&&(n.specularIntensityMap=this.specularIntensityMap.toJSON(t).uuid),this.specularColorMap&&this.specularColorMap.isTexture&&(n.specularColorMap=this.specularColorMap.toJSON(t).uuid),this.envMap&&this.envMap.isTexture&&(n.envMap=this.envMap.toJSON(t).uuid,void 0!==this.combine&&(n.combine=this.combine)),void 0!==this.envMapIntensity&&(n.envMapIntensity=this.envMapIntensity),void 0!==this.reflectivity&&(n.reflectivity=this.reflectivity),void 0!==this.refractionRatio&&(n.refractionRatio=this.refractionRatio),this.gradientMap&&this.gradientMap.isTexture&&(n.gradientMap=this.gradientMap.toJSON(t).uuid),void 0!==this.transmission&&(n.transmission=this.transmission),this.transmissionMap&&this.transmissionMap.isTexture&&(n.transmissionMap=this.transmissionMap.toJSON(t).uuid),void 0!==this.thickness&&(n.thickness=this.thickness),this.thicknessMap&&this.thicknessMap.isTexture&&(n.thicknessMap=this.thicknessMap.toJSON(t).uuid),void 0!==this.attenuationDistance&&(n.attenuationDistance=this.attenuationDistance),void 0!==this.attenuationColor&&(n.attenuationColor=this.attenuationColor.getHex()),void 0!==this.size&&(n.size=this.size),null!==this.shadowSide&&(n.shadowSide=this.shadowSide),void 0!==this.sizeAttenuation&&(n.sizeAttenuation=this.sizeAttenuation),1!==this.blending&&(n.blending=this.blending),0!==this.side&&(n.side=this.side),this.vertexColors&&(n.vertexColors=!0),this.opacity<1&&(n.opacity=this.opacity),this.format!==E&&(n.format=this.format),!0===this.transparent&&(n.transparent=this.transparent),n.depthFunc=this.depthFunc,n.depthTest=this.depthTest,n.depthWrite=this.depthWrite,n.colorWrite=this.colorWrite,n.stencilWrite=this.stencilWrite,n.stencilWriteMask=this.stencilWriteMask,n.stencilFunc=this.stencilFunc,n.stencilRef=this.stencilRef,n.stencilFuncMask=this.stencilFuncMask,n.stencilFail=this.stencilFail,n.stencilZFail=this.stencilZFail,n.stencilZPass=this.stencilZPass,this.rotation&&0!==this.rotation&&(n.rotation=this.rotation),!0===this.polygonOffset&&(n.polygonOffset=!0),0!==this.polygonOffsetFactor&&(n.polygonOffsetFactor=this.polygonOffsetFactor),0!==this.polygonOffsetUnits&&(n.polygonOffsetUnits=this.polygonOffsetUnits),this.linewidth&&1!==this.linewidth&&(n.linewidth=this.linewidth),void 0!==this.dashSize&&(n.dashSize=this.dashSize),void 0!==this.gapSize&&(n.gapSize=this.gapSize),void 0!==this.scale&&(n.scale=this.scale),!0===this.dithering&&(n.dithering=!0),this.alphaTest>0&&(n.alphaTest=this.alphaTest),!0===this.alphaToCoverage&&(n.alphaToCoverage=this.alphaToCoverage),!0===this.premultipliedAlpha&&(n.premultipliedAlpha=this.premultipliedAlpha),!0===this.wireframe&&(n.wireframe=this.wireframe),this.wireframeLinewidth>1&&(n.wireframeLinewidth=this.wireframeLinewidth),"round"!==this.wireframeLinecap&&(n.wireframeLinecap=this.wireframeLinecap),"round"!==this.wireframeLinejoin&&(n.wireframeLinejoin=this.wireframeLinejoin),!0===this.flatShading&&(n.flatShading=this.flatShading),!1===this.visible&&(n.visible=!1),!1===this.toneMapped&&(n.toneMapped=!1),"{}"!==JSON.stringify(this.userData)&&(n.userData=this.userData),e){const e=i(t.textures),r=i(t.images);e.length>0&&(n.textures=e),r.length>0&&(n.images=r)}return n}clone(){return(new this.constructor).copy(this)}copy(t){this.name=t.name,this.fog=t.fog,this.blending=t.blending,this.side=t.side,this.vertexColors=t.vertexColors,this.opacity=t.opacity,this.format=t.format,this.transparent=t.transparent,this.blendSrc=t.blendSrc,this.blendDst=t.blendDst,this.blendEquation=t.blendEquation,this.blendSrcAlpha=t.blendSrcAlpha,this.blendDstAlpha=t.blendDstAlpha,this.blendEquationAlpha=t.blendEquationAlpha,this.depthFunc=t.depthFunc,this.depthTest=t.depthTest,this.depthWrite=t.depthWrite,this.stencilWriteMask=t.stencilWriteMask,this.stencilFunc=t.stencilFunc,this.stencilRef=t.stencilRef,this.stencilFuncMask=t.stencilFuncMask,this.stencilFail=t.stencilFail,this.stencilZFail=t.stencilZFail,this.stencilZPass=t.stencilZPass,this.stencilWrite=t.stencilWrite;const e=t.clippingPlanes;let n=null;if(null!==e){const t=e.length;n=new Array(t);for(let i=0;i!==t;++i)n[i]=e[i].clone()}return this.clippingPlanes=n,this.clipIntersection=t.clipIntersection,this.clipShadows=t.clipShadows,this.shadowSide=t.shadowSide,this.colorWrite=t.colorWrite,this.precision=t.precision,this.polygonOffset=t.polygonOffset,this.polygonOffsetFactor=t.polygonOffsetFactor,this.polygonOffsetUnits=t.polygonOffsetUnits,this.dithering=t.dithering,this.alphaTest=t.alphaTest,this.alphaToCoverage=t.alphaToCoverage,this.premultipliedAlpha=t.premultipliedAlpha,this.visible=t.visible,this.toneMapped=t.toneMapped,this.userData=JSON.parse(JSON.stringify(t.userData)),this}dispose(){this.dispatchEvent({type:"dispose"})}set needsUpdate(t){!0===t&&this.version++}}Ze.prototype.isMaterial=!0;const Qe={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074},Ke={h:0,s:0,l:0},$e={h:0,s:0,l:0};function tn(t,e,n){return n<0&&(n+=1),n>1&&(n-=1),n<1/6?t+6*(e-t)*n:n<.5?e:n<2/3?t+6*(e-t)*(2/3-n):t}function en(t){return t<.04045?.0773993808*t:Math.pow(.9478672986*t+.0521327014,2.4)}function nn(t){return t<.0031308?12.92*t:1.055*Math.pow(t,.41666)-.055}class rn{constructor(t,e,n){return void 0===e&&void 0===n?this.set(t):this.setRGB(t,e,n)}set(t){return t&&t.isColor?this.copy(t):"number"==typeof t?this.setHex(t):"string"==typeof t&&this.setStyle(t),this}setScalar(t){return this.r=t,this.g=t,this.b=t,this}setHex(t){return t=Math.floor(t),this.r=(t>>16&255)/255,this.g=(t>>8&255)/255,this.b=(255&t)/255,this}setRGB(t,e,n){return this.r=t,this.g=e,this.b=n,this}setHSL(t,e,n){if(t=dt(t,1),e=ut(e,0,1),n=ut(n,0,1),0===e)this.r=this.g=this.b=n;else{const i=n<=.5?n*(1+e):n+e-n*e,r=2*n-i;this.r=tn(r,i,t+1/3),this.g=tn(r,i,t),this.b=tn(r,i,t-1/3)}return this}setStyle(t){function e(e){void 0!==e&&parseFloat(e)<1&&console.warn("THREE.Color: Alpha component of "+t+" will be ignored.")}let n;if(n=/^((?:rgb|hsl)a?)\(([^\)]*)\)/.exec(t)){let t;const i=n[1],r=n[2];switch(i){case"rgb":case"rgba":if(t=/^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(r))return this.r=Math.min(255,parseInt(t[1],10))/255,this.g=Math.min(255,parseInt(t[2],10))/255,this.b=Math.min(255,parseInt(t[3],10))/255,e(t[4]),this;if(t=/^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(r))return this.r=Math.min(100,parseInt(t[1],10))/100,this.g=Math.min(100,parseInt(t[2],10))/100,this.b=Math.min(100,parseInt(t[3],10))/100,e(t[4]),this;break;case"hsl":case"hsla":if(t=/^\s*(\d*\.?\d+)\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(r)){const n=parseFloat(t[1])/360,i=parseInt(t[2],10)/100,r=parseInt(t[3],10)/100;return e(t[4]),this.setHSL(n,i,r)}}}else if(n=/^\#([A-Fa-f\d]+)$/.exec(t)){const t=n[1],e=t.length;if(3===e)return this.r=parseInt(t.charAt(0)+t.charAt(0),16)/255,this.g=parseInt(t.charAt(1)+t.charAt(1),16)/255,this.b=parseInt(t.charAt(2)+t.charAt(2),16)/255,this;if(6===e)return this.r=parseInt(t.charAt(0)+t.charAt(1),16)/255,this.g=parseInt(t.charAt(2)+t.charAt(3),16)/255,this.b=parseInt(t.charAt(4)+t.charAt(5),16)/255,this}return t&&t.length>0?this.setColorName(t):this}setColorName(t){const e=Qe[t.toLowerCase()];return void 0!==e?this.setHex(e):console.warn("THREE.Color: Unknown color "+t),this}clone(){return new this.constructor(this.r,this.g,this.b)}copy(t){return this.r=t.r,this.g=t.g,this.b=t.b,this}copyGammaToLinear(t,e=2){return this.r=Math.pow(t.r,e),this.g=Math.pow(t.g,e),this.b=Math.pow(t.b,e),this}copyLinearToGamma(t,e=2){const n=e>0?1/e:1;return this.r=Math.pow(t.r,n),this.g=Math.pow(t.g,n),this.b=Math.pow(t.b,n),this}convertGammaToLinear(t){return this.copyGammaToLinear(this,t),this}convertLinearToGamma(t){return this.copyLinearToGamma(this,t),this}copySRGBToLinear(t){return this.r=en(t.r),this.g=en(t.g),this.b=en(t.b),this}copyLinearToSRGB(t){return this.r=nn(t.r),this.g=nn(t.g),this.b=nn(t.b),this}convertSRGBToLinear(){return this.copySRGBToLinear(this),this}convertLinearToSRGB(){return this.copyLinearToSRGB(this),this}getHex(){return 255*this.r<<16^255*this.g<<8^255*this.b<<0}getHexString(){return("000000"+this.getHex().toString(16)).slice(-6)}getHSL(t){const e=this.r,n=this.g,i=this.b,r=Math.max(e,n,i),s=Math.min(e,n,i);let a,o;const l=(s+r)/2;if(s===r)a=0,o=0;else{const t=r-s;switch(o=l<=.5?t/(r+s):t/(2-r-s),r){case e:a=(n-i)/t+(n<i?6:0);break;case n:a=(i-e)/t+2;break;case i:a=(e-n)/t+4}a/=6}return t.h=a,t.s=o,t.l=l,t}getStyle(){return"rgb("+(255*this.r|0)+","+(255*this.g|0)+","+(255*this.b|0)+")"}offsetHSL(t,e,n){return this.getHSL(Ke),Ke.h+=t,Ke.s+=e,Ke.l+=n,this.setHSL(Ke.h,Ke.s,Ke.l),this}add(t){return this.r+=t.r,this.g+=t.g,this.b+=t.b,this}addColors(t,e){return this.r=t.r+e.r,this.g=t.g+e.g,this.b=t.b+e.b,this}addScalar(t){return this.r+=t,this.g+=t,this.b+=t,this}sub(t){return this.r=Math.max(0,this.r-t.r),this.g=Math.max(0,this.g-t.g),this.b=Math.max(0,this.b-t.b),this}multiply(t){return this.r*=t.r,this.g*=t.g,this.b*=t.b,this}multiplyScalar(t){return this.r*=t,this.g*=t,this.b*=t,this}lerp(t,e){return this.r+=(t.r-this.r)*e,this.g+=(t.g-this.g)*e,this.b+=(t.b-this.b)*e,this}lerpColors(t,e,n){return this.r=t.r+(e.r-t.r)*n,this.g=t.g+(e.g-t.g)*n,this.b=t.b+(e.b-t.b)*n,this}lerpHSL(t,e){this.getHSL(Ke),t.getHSL($e);const n=pt(Ke.h,$e.h,e),i=pt(Ke.s,$e.s,e),r=pt(Ke.l,$e.l,e);return this.setHSL(n,i,r),this}equals(t){return t.r===this.r&&t.g===this.g&&t.b===this.b}fromArray(t,e=0){return this.r=t[e],this.g=t[e+1],this.b=t[e+2],this}toArray(t=[],e=0){return t[e]=this.r,t[e+1]=this.g,t[e+2]=this.b,t}fromBufferAttribute(t,e){return this.r=t.getX(e),this.g=t.getY(e),this.b=t.getZ(e),!0===t.normalized&&(this.r/=255,this.g/=255,this.b/=255),this}toJSON(){return this.getHex()}}rn.NAMES=Qe,rn.prototype.isColor=!0,rn.prototype.r=1,rn.prototype.g=1,rn.prototype.b=1;class sn extends Ze{constructor(t){super(),this.type="MeshBasicMaterial",this.color=new rn(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.combine=0,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.setValues(t)}copy(t){return super.copy(t),this.color.copy(t.color),this.map=t.map,this.lightMap=t.lightMap,this.lightMapIntensity=t.lightMapIntensity,this.aoMap=t.aoMap,this.aoMapIntensity=t.aoMapIntensity,this.specularMap=t.specularMap,this.alphaMap=t.alphaMap,this.envMap=t.envMap,this.combine=t.combine,this.reflectivity=t.reflectivity,this.refractionRatio=t.refractionRatio,this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this.wireframeLinecap=t.wireframeLinecap,this.wireframeLinejoin=t.wireframeLinejoin,this}}sn.prototype.isMeshBasicMaterial=!0;const an=new zt,on=new yt;class ln{constructor(t,e,n){if(Array.isArray(t))throw new TypeError("THREE.BufferAttribute: array should be a Typed Array.");this.name="",this.array=t,this.itemSize=e,this.count=void 0!==t?t.length/e:0,this.normalized=!0===n,this.usage=et,this.updateRange={offset:0,count:-1},this.version=0}onUploadCallback(){}set needsUpdate(t){!0===t&&this.version++}setUsage(t){return this.usage=t,this}copy(t){return this.name=t.name,this.array=new t.array.constructor(t.array),this.itemSize=t.itemSize,this.count=t.count,this.normalized=t.normalized,this.usage=t.usage,this}copyAt(t,e,n){t*=this.itemSize,n*=e.itemSize;for(let i=0,r=this.itemSize;i<r;i++)this.array[t+i]=e.array[n+i];return this}copyArray(t){return this.array.set(t),this}copyColorsArray(t){const e=this.array;let n=0;for(let i=0,r=t.length;i<r;i++){let r=t[i];void 0===r&&(console.warn("THREE.BufferAttribute.copyColorsArray(): color is undefined",i),r=new rn),e[n++]=r.r,e[n++]=r.g,e[n++]=r.b}return this}copyVector2sArray(t){const e=this.array;let n=0;for(let i=0,r=t.length;i<r;i++){let r=t[i];void 0===r&&(console.warn("THREE.BufferAttribute.copyVector2sArray(): vector is undefined",i),r=new yt),e[n++]=r.x,e[n++]=r.y}return this}copyVector3sArray(t){const e=this.array;let n=0;for(let i=0,r=t.length;i<r;i++){let r=t[i];void 0===r&&(console.warn("THREE.BufferAttribute.copyVector3sArray(): vector is undefined",i),r=new zt),e[n++]=r.x,e[n++]=r.y,e[n++]=r.z}return this}copyVector4sArray(t){const e=this.array;let n=0;for(let i=0,r=t.length;i<r;i++){let r=t[i];void 0===r&&(console.warn("THREE.BufferAttribute.copyVector4sArray(): vector is undefined",i),r=new Ct),e[n++]=r.x,e[n++]=r.y,e[n++]=r.z,e[n++]=r.w}return this}applyMatrix3(t){if(2===this.itemSize)for(let e=0,n=this.count;e<n;e++)on.fromBufferAttribute(this,e),on.applyMatrix3(t),this.setXY(e,on.x,on.y);else if(3===this.itemSize)for(let e=0,n=this.count;e<n;e++)an.fromBufferAttribute(this,e),an.applyMatrix3(t),this.setXYZ(e,an.x,an.y,an.z);return this}applyMatrix4(t){for(let e=0,n=this.count;e<n;e++)an.x=this.getX(e),an.y=this.getY(e),an.z=this.getZ(e),an.applyMatrix4(t),this.setXYZ(e,an.x,an.y,an.z);return this}applyNormalMatrix(t){for(let e=0,n=this.count;e<n;e++)an.x=this.getX(e),an.y=this.getY(e),an.z=this.getZ(e),an.applyNormalMatrix(t),this.setXYZ(e,an.x,an.y,an.z);return this}transformDirection(t){for(let e=0,n=this.count;e<n;e++)an.x=this.getX(e),an.y=this.getY(e),an.z=this.getZ(e),an.transformDirection(t),this.setXYZ(e,an.x,an.y,an.z);return this}set(t,e=0){return this.array.set(t,e),this}getX(t){return this.array[t*this.itemSize]}setX(t,e){return this.array[t*this.itemSize]=e,this}getY(t){return this.array[t*this.itemSize+1]}setY(t,e){return this.array[t*this.itemSize+1]=e,this}getZ(t){return this.array[t*this.itemSize+2]}setZ(t,e){return this.array[t*this.itemSize+2]=e,this}getW(t){return this.array[t*this.itemSize+3]}setW(t,e){return this.array[t*this.itemSize+3]=e,this}setXY(t,e,n){return t*=this.itemSize,this.array[t+0]=e,this.array[t+1]=n,this}setXYZ(t,e,n,i){return t*=this.itemSize,this.array[t+0]=e,this.array[t+1]=n,this.array[t+2]=i,this}setXYZW(t,e,n,i,r){return t*=this.itemSize,this.array[t+0]=e,this.array[t+1]=n,this.array[t+2]=i,this.array[t+3]=r,this}onUpload(t){return this.onUploadCallback=t,this}clone(){return new this.constructor(this.array,this.itemSize).copy(this)}toJSON(){const t={itemSize:this.itemSize,type:this.array.constructor.name,array:Array.prototype.slice.call(this.array),normalized:this.normalized};return""!==this.name&&(t.name=this.name),this.usage!==et&&(t.usage=this.usage),0===this.updateRange.offset&&-1===this.updateRange.count||(t.updateRange=this.updateRange),t}}ln.prototype.isBufferAttribute=!0;class cn extends ln{constructor(t,e,n){super(new Int8Array(t),e,n)}}class hn extends ln{constructor(t,e,n){super(new Uint8Array(t),e,n)}}class un extends ln{constructor(t,e,n){super(new Uint8ClampedArray(t),e,n)}}class dn extends ln{constructor(t,e,n){super(new Int16Array(t),e,n)}}class pn extends ln{constructor(t,e,n){super(new Uint16Array(t),e,n)}}class mn extends ln{constructor(t,e,n){super(new Int32Array(t),e,n)}}class fn extends ln{constructor(t,e,n){super(new Uint32Array(t),e,n)}}class gn extends ln{constructor(t,e,n){super(new Uint16Array(t),e,n)}}gn.prototype.isFloat16BufferAttribute=!0;class vn extends ln{constructor(t,e,n){super(new Float32Array(t),e,n)}}class yn extends ln{constructor(t,e,n){super(new Float64Array(t),e,n)}}let xn=0;const _n=new de,Mn=new Fe,bn=new zt,wn=new Ot,Sn=new Ot,Tn=new zt;class En extends rt{constructor(){super(),Object.defineProperty(this,"id",{value:xn++}),this.uuid=ht(),this.name="",this.type="BufferGeometry",this.index=null,this.attributes={},this.morphAttributes={},this.morphTargetsRelative=!1,this.groups=[],this.boundingBox=null,this.boundingSphere=null,this.drawRange={start:0,count:1/0},this.userData={}}getIndex(){return this.index}setIndex(t){return Array.isArray(t)?this.index=new(_t(t)>65535?fn:pn)(t,1):this.index=t,this}getAttribute(t){return this.attributes[t]}setAttribute(t,e){return this.attributes[t]=e,this}deleteAttribute(t){return delete this.attributes[t],this}hasAttribute(t){return void 0!==this.attributes[t]}addGroup(t,e,n=0){this.groups.push({start:t,count:e,materialIndex:n})}clearGroups(){this.groups=[]}setDrawRange(t,e){this.drawRange.start=t,this.drawRange.count=e}applyMatrix4(t){const e=this.attributes.position;void 0!==e&&(e.applyMatrix4(t),e.needsUpdate=!0);const n=this.attributes.normal;if(void 0!==n){const e=(new xt).getNormalMatrix(t);n.applyNormalMatrix(e),n.needsUpdate=!0}const i=this.attributes.tangent;return void 0!==i&&(i.transformDirection(t),i.needsUpdate=!0),null!==this.boundingBox&&this.computeBoundingBox(),null!==this.boundingSphere&&this.computeBoundingSphere(),this}applyQuaternion(t){return _n.makeRotationFromQuaternion(t),this.applyMatrix4(_n),this}rotateX(t){return _n.makeRotationX(t),this.applyMatrix4(_n),this}rotateY(t){return _n.makeRotationY(t),this.applyMatrix4(_n),this}rotateZ(t){return _n.makeRotationZ(t),this.applyMatrix4(_n),this}translate(t,e,n){return _n.makeTranslation(t,e,n),this.applyMatrix4(_n),this}scale(t,e,n){return _n.makeScale(t,e,n),this.applyMatrix4(_n),this}lookAt(t){return Mn.lookAt(t),Mn.updateMatrix(),this.applyMatrix4(Mn.matrix),this}center(){return this.computeBoundingBox(),this.boundingBox.getCenter(bn).negate(),this.translate(bn.x,bn.y,bn.z),this}setFromPoints(t){const e=[];for(let n=0,i=t.length;n<i;n++){const i=t[n];e.push(i.x,i.y,i.z||0)}return this.setAttribute("position",new vn(e,3)),this}computeBoundingBox(){null===this.boundingBox&&(this.boundingBox=new Ot);const t=this.attributes.position,e=this.morphAttributes.position;if(t&&t.isGLBufferAttribute)return console.error('THREE.BufferGeometry.computeBoundingBox(): GLBufferAttribute requires a manual bounding box. Alternatively set "mesh.frustumCulled" to "false".',this),void this.boundingBox.set(new zt(-1/0,-1/0,-1/0),new zt(1/0,1/0,1/0));if(void 0!==t){if(this.boundingBox.setFromBufferAttribute(t),e)for(let t=0,n=e.length;t<n;t++){const n=e[t];wn.setFromBufferAttribute(n),this.morphTargetsRelative?(Tn.addVectors(this.boundingBox.min,wn.min),this.boundingBox.expandByPoint(Tn),Tn.addVectors(this.boundingBox.max,wn.max),this.boundingBox.expandByPoint(Tn)):(this.boundingBox.expandByPoint(wn.min),this.boundingBox.expandByPoint(wn.max))}}else this.boundingBox.makeEmpty();(isNaN(this.boundingBox.min.x)||isNaN(this.boundingBox.min.y)||isNaN(this.boundingBox.min.z))&&console.error('THREE.BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The "position" attribute is likely to have NaN values.',this)}computeBoundingSphere(){null===this.boundingSphere&&(this.boundingSphere=new ie);const t=this.attributes.position,e=this.morphAttributes.position;if(t&&t.isGLBufferAttribute)return console.error('THREE.BufferGeometry.computeBoundingSphere(): GLBufferAttribute requires a manual bounding sphere. Alternatively set "mesh.frustumCulled" to "false".',this),void this.boundingSphere.set(new zt,1/0);if(t){const n=this.boundingSphere.center;if(wn.setFromBufferAttribute(t),e)for(let t=0,n=e.length;t<n;t++){const n=e[t];Sn.setFromBufferAttribute(n),this.morphTargetsRelative?(Tn.addVectors(wn.min,Sn.min),wn.expandByPoint(Tn),Tn.addVectors(wn.max,Sn.max),wn.expandByPoint(Tn)):(wn.expandByPoint(Sn.min),wn.expandByPoint(Sn.max))}wn.getCenter(n);let i=0;for(let e=0,r=t.count;e<r;e++)Tn.fromBufferAttribute(t,e),i=Math.max(i,n.distanceToSquared(Tn));if(e)for(let r=0,s=e.length;r<s;r++){const s=e[r],a=this.morphTargetsRelative;for(let e=0,r=s.count;e<r;e++)Tn.fromBufferAttribute(s,e),a&&(bn.fromBufferAttribute(t,e),Tn.add(bn)),i=Math.max(i,n.distanceToSquared(Tn))}this.boundingSphere.radius=Math.sqrt(i),isNaN(this.boundingSphere.radius)&&console.error('THREE.BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.',this)}}computeTangents(){const t=this.index,e=this.attributes;if(null===t||void 0===e.position||void 0===e.normal||void 0===e.uv)return void console.error("THREE.BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)");const n=t.array,i=e.position.array,r=e.normal.array,s=e.uv.array,a=i.length/3;void 0===e.tangent&&this.setAttribute("tangent",new ln(new Float32Array(4*a),4));const o=e.tangent.array,l=[],c=[];for(let t=0;t<a;t++)l[t]=new zt,c[t]=new zt;const h=new zt,u=new zt,d=new zt,p=new yt,m=new yt,f=new yt,g=new zt,v=new zt;function y(t,e,n){h.fromArray(i,3*t),u.fromArray(i,3*e),d.fromArray(i,3*n),p.fromArray(s,2*t),m.fromArray(s,2*e),f.fromArray(s,2*n),u.sub(h),d.sub(h),m.sub(p),f.sub(p);const r=1/(m.x*f.y-f.x*m.y);isFinite(r)&&(g.copy(u).multiplyScalar(f.y).addScaledVector(d,-m.y).multiplyScalar(r),v.copy(d).multiplyScalar(m.x).addScaledVector(u,-f.x).multiplyScalar(r),l[t].add(g),l[e].add(g),l[n].add(g),c[t].add(v),c[e].add(v),c[n].add(v))}let x=this.groups;0===x.length&&(x=[{start:0,count:n.length}]);for(let t=0,e=x.length;t<e;++t){const e=x[t],i=e.start;for(let t=i,r=i+e.count;t<r;t+=3)y(n[t+0],n[t+1],n[t+2])}const _=new zt,M=new zt,b=new zt,w=new zt;function S(t){b.fromArray(r,3*t),w.copy(b);const e=l[t];_.copy(e),_.sub(b.multiplyScalar(b.dot(e))).normalize(),M.crossVectors(w,e);const n=M.dot(c[t])<0?-1:1;o[4*t]=_.x,o[4*t+1]=_.y,o[4*t+2]=_.z,o[4*t+3]=n}for(let t=0,e=x.length;t<e;++t){const e=x[t],i=e.start;for(let t=i,r=i+e.count;t<r;t+=3)S(n[t+0]),S(n[t+1]),S(n[t+2])}}computeVertexNormals(){const t=this.index,e=this.getAttribute("position");if(void 0!==e){let n=this.getAttribute("normal");if(void 0===n)n=new ln(new Float32Array(3*e.count),3),this.setAttribute("normal",n);else for(let t=0,e=n.count;t<e;t++)n.setXYZ(t,0,0,0);const i=new zt,r=new zt,s=new zt,a=new zt,o=new zt,l=new zt,c=new zt,h=new zt;if(t)for(let u=0,d=t.count;u<d;u+=3){const d=t.getX(u+0),p=t.getX(u+1),m=t.getX(u+2);i.fromBufferAttribute(e,d),r.fromBufferAttribute(e,p),s.fromBufferAttribute(e,m),c.subVectors(s,r),h.subVectors(i,r),c.cross(h),a.fromBufferAttribute(n,d),o.fromBufferAttribute(n,p),l.fromBufferAttribute(n,m),a.add(c),o.add(c),l.add(c),n.setXYZ(d,a.x,a.y,a.z),n.setXYZ(p,o.x,o.y,o.z),n.setXYZ(m,l.x,l.y,l.z)}else for(let t=0,a=e.count;t<a;t+=3)i.fromBufferAttribute(e,t+0),r.fromBufferAttribute(e,t+1),s.fromBufferAttribute(e,t+2),c.subVectors(s,r),h.subVectors(i,r),c.cross(h),n.setXYZ(t+0,c.x,c.y,c.z),n.setXYZ(t+1,c.x,c.y,c.z),n.setXYZ(t+2,c.x,c.y,c.z);this.normalizeNormals(),n.needsUpdate=!0}}merge(t,e){if(!t||!t.isBufferGeometry)return void console.error("THREE.BufferGeometry.merge(): geometry not an instance of THREE.BufferGeometry.",t);void 0===e&&(e=0,console.warn("THREE.BufferGeometry.merge(): Overwriting original geometry, starting at offset=0. Use BufferGeometryUtils.mergeBufferGeometries() for lossless merge."));const n=this.attributes;for(const i in n){if(void 0===t.attributes[i])continue;const r=n[i].array,s=t.attributes[i],a=s.array,o=s.itemSize*e,l=Math.min(a.length,r.length-o);for(let t=0,e=o;t<l;t++,e++)r[e]=a[t]}return this}normalizeNormals(){const t=this.attributes.normal;for(let e=0,n=t.count;e<n;e++)Tn.fromBufferAttribute(t,e),Tn.normalize(),t.setXYZ(e,Tn.x,Tn.y,Tn.z)}toNonIndexed(){function t(t,e){const n=t.array,i=t.itemSize,r=t.normalized,s=new n.constructor(e.length*i);let a=0,o=0;for(let r=0,l=e.length;r<l;r++){a=t.isInterleavedBufferAttribute?e[r]*t.data.stride+t.offset:e[r]*i;for(let t=0;t<i;t++)s[o++]=n[a++]}return new ln(s,i,r)}if(null===this.index)return console.warn("THREE.BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed."),this;const e=new En,n=this.index.array,i=this.attributes;for(const r in i){const s=t(i[r],n);e.setAttribute(r,s)}const r=this.morphAttributes;for(const i in r){const s=[],a=r[i];for(let e=0,i=a.length;e<i;e++){const i=t(a[e],n);s.push(i)}e.morphAttributes[i]=s}e.morphTargetsRelative=this.morphTargetsRelative;const s=this.groups;for(let t=0,n=s.length;t<n;t++){const n=s[t];e.addGroup(n.start,n.count,n.materialIndex)}return e}toJSON(){const t={metadata:{version:4.5,type:"BufferGeometry",generator:"BufferGeometry.toJSON"}};if(t.uuid=this.uuid,t.type=this.type,""!==this.name&&(t.name=this.name),Object.keys(this.userData).length>0&&(t.userData=this.userData),void 0!==this.parameters){const e=this.parameters;for(const n in e)void 0!==e[n]&&(t[n]=e[n]);return t}t.data={attributes:{}};const e=this.index;null!==e&&(t.data.index={type:e.array.constructor.name,array:Array.prototype.slice.call(e.array)});const n=this.attributes;for(const e in n){const i=n[e];t.data.attributes[e]=i.toJSON(t.data)}const i={};let r=!1;for(const e in this.morphAttributes){const n=this.morphAttributes[e],s=[];for(let e=0,i=n.length;e<i;e++){const i=n[e];s.push(i.toJSON(t.data))}s.length>0&&(i[e]=s,r=!0)}r&&(t.data.morphAttributes=i,t.data.morphTargetsRelative=this.morphTargetsRelative);const s=this.groups;s.length>0&&(t.data.groups=JSON.parse(JSON.stringify(s)));const a=this.boundingSphere;return null!==a&&(t.data.boundingSphere={center:a.center.toArray(),radius:a.radius}),t}clone(){return(new this.constructor).copy(this)}copy(t){this.index=null,this.attributes={},this.morphAttributes={},this.groups=[],this.boundingBox=null,this.boundingSphere=null;const e={};this.name=t.name;const n=t.index;null!==n&&this.setIndex(n.clone(e));const i=t.attributes;for(const t in i){const n=i[t];this.setAttribute(t,n.clone(e))}const r=t.morphAttributes;for(const t in r){const n=[],i=r[t];for(let t=0,r=i.length;t<r;t++)n.push(i[t].clone(e));this.morphAttributes[t]=n}this.morphTargetsRelative=t.morphTargetsRelative;const s=t.groups;for(let t=0,e=s.length;t<e;t++){const e=s[t];this.addGroup(e.start,e.count,e.materialIndex)}const a=t.boundingBox;null!==a&&(this.boundingBox=a.clone());const o=t.boundingSphere;return null!==o&&(this.boundingSphere=o.clone()),this.drawRange.start=t.drawRange.start,this.drawRange.count=t.drawRange.count,this.userData=t.userData,void 0!==t.parameters&&(this.parameters=Object.assign({},t.parameters)),this}dispose(){this.dispatchEvent({type:"dispose"})}}En.prototype.isBufferGeometry=!0;const An=new de,Ln=new ue,Rn=new ie,Cn=new zt,Pn=new zt,In=new zt,Dn=new zt,Nn=new zt,zn=new zt,Bn=new zt,Fn=new zt,On=new zt,Un=new yt,Hn=new yt,Gn=new yt,kn=new zt,Vn=new zt;class Wn extends Fe{constructor(t=new En,e=new sn){super(),this.type="Mesh",this.geometry=t,this.material=e,this.updateMorphTargets()}copy(t){return super.copy(t),void 0!==t.morphTargetInfluences&&(this.morphTargetInfluences=t.morphTargetInfluences.slice()),void 0!==t.morphTargetDictionary&&(this.morphTargetDictionary=Object.assign({},t.morphTargetDictionary)),this.material=t.material,this.geometry=t.geometry,this}updateMorphTargets(){const t=this.geometry;if(t.isBufferGeometry){const e=t.morphAttributes,n=Object.keys(e);if(n.length>0){const t=e[n[0]];if(void 0!==t){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let e=0,n=t.length;e<n;e++){const n=t[e].name||String(e);this.morphTargetInfluences.push(0),this.morphTargetDictionary[n]=e}}}}else{const e=t.morphTargets;void 0!==e&&e.length>0&&console.error("THREE.Mesh.updateMorphTargets() no longer supports THREE.Geometry. Use THREE.BufferGeometry instead.")}}raycast(t,e){const n=this.geometry,i=this.material,r=this.matrixWorld;if(void 0===i)return;if(null===n.boundingSphere&&n.computeBoundingSphere(),Rn.copy(n.boundingSphere),Rn.applyMatrix4(r),!1===t.ray.intersectsSphere(Rn))return;if(An.copy(r).invert(),Ln.copy(t.ray).applyMatrix4(An),null!==n.boundingBox&&!1===Ln.intersectsBox(n.boundingBox))return;let s;if(n.isBufferGeometry){const r=n.index,a=n.attributes.position,o=n.morphAttributes.position,l=n.morphTargetsRelative,c=n.attributes.uv,h=n.attributes.uv2,u=n.groups,d=n.drawRange;if(null!==r)if(Array.isArray(i))for(let n=0,p=u.length;n<p;n++){const p=u[n],m=i[p.materialIndex];for(let n=Math.max(p.start,d.start),i=Math.min(r.count,Math.min(p.start+p.count,d.start+d.count));n<i;n+=3){const i=r.getX(n),u=r.getX(n+1),d=r.getX(n+2);s=jn(this,m,t,Ln,a,o,l,c,h,i,u,d),s&&(s.faceIndex=Math.floor(n/3),s.face.materialIndex=p.materialIndex,e.push(s))}}else{for(let n=Math.max(0,d.start),u=Math.min(r.count,d.start+d.count);n<u;n+=3){const u=r.getX(n),d=r.getX(n+1),p=r.getX(n+2);s=jn(this,i,t,Ln,a,o,l,c,h,u,d,p),s&&(s.faceIndex=Math.floor(n/3),e.push(s))}}else if(void 0!==a)if(Array.isArray(i))for(let n=0,r=u.length;n<r;n++){const r=u[n],p=i[r.materialIndex];for(let n=Math.max(r.start,d.start),i=Math.min(a.count,Math.min(r.start+r.count,d.start+d.count));n<i;n+=3){s=jn(this,p,t,Ln,a,o,l,c,h,n,n+1,n+2),s&&(s.faceIndex=Math.floor(n/3),s.face.materialIndex=r.materialIndex,e.push(s))}}else{for(let n=Math.max(0,d.start),r=Math.min(a.count,d.start+d.count);n<r;n+=3){s=jn(this,i,t,Ln,a,o,l,c,h,n,n+1,n+2),s&&(s.faceIndex=Math.floor(n/3),e.push(s))}}}else n.isGeometry&&console.error("THREE.Mesh.raycast() no longer supports THREE.Geometry. Use THREE.BufferGeometry instead.")}}function jn(t,e,n,i,r,s,a,o,l,c,h,u){Cn.fromBufferAttribute(r,c),Pn.fromBufferAttribute(r,h),In.fromBufferAttribute(r,u);const d=t.morphTargetInfluences;if(s&&d){Bn.set(0,0,0),Fn.set(0,0,0),On.set(0,0,0);for(let t=0,e=s.length;t<e;t++){const e=d[t],n=s[t];0!==e&&(Dn.fromBufferAttribute(n,c),Nn.fromBufferAttribute(n,h),zn.fromBufferAttribute(n,u),a?(Bn.addScaledVector(Dn,e),Fn.addScaledVector(Nn,e),On.addScaledVector(zn,e)):(Bn.addScaledVector(Dn.sub(Cn),e),Fn.addScaledVector(Nn.sub(Pn),e),On.addScaledVector(zn.sub(In),e)))}Cn.add(Bn),Pn.add(Fn),In.add(On)}t.isSkinnedMesh&&(t.boneTransform(c,Cn),t.boneTransform(h,Pn),t.boneTransform(u,In));const p=function(t,e,n,i,r,s,a,o){let l;if(l=1===e.side?i.intersectTriangle(a,s,r,!0,o):i.intersectTriangle(r,s,a,2!==e.side,o),null===l)return null;Vn.copy(o),Vn.applyMatrix4(t.matrixWorld);const c=n.ray.origin.distanceTo(Vn);return c<n.near||c>n.far?null:{distance:c,point:Vn.clone(),object:t}}(t,e,n,i,Cn,Pn,In,kn);if(p){o&&(Un.fromBufferAttribute(o,c),Hn.fromBufferAttribute(o,h),Gn.fromBufferAttribute(o,u),p.uv=Ye.getUV(kn,Cn,Pn,In,Un,Hn,Gn,new yt)),l&&(Un.fromBufferAttribute(l,c),Hn.fromBufferAttribute(l,h),Gn.fromBufferAttribute(l,u),p.uv2=Ye.getUV(kn,Cn,Pn,In,Un,Hn,Gn,new yt));const t={a:c,b:h,c:u,normal:new zt,materialIndex:0};Ye.getNormal(Cn,Pn,In,t.normal),p.face=t}return p}Wn.prototype.isMesh=!0;class qn extends En{constructor(t=1,e=1,n=1,i=1,r=1,s=1){super(),this.type="BoxGeometry",this.parameters={width:t,height:e,depth:n,widthSegments:i,heightSegments:r,depthSegments:s};const a=this;i=Math.floor(i),r=Math.floor(r),s=Math.floor(s);const o=[],l=[],c=[],h=[];let u=0,d=0;function p(t,e,n,i,r,s,p,m,f,g,v){const y=s/f,x=p/g,_=s/2,M=p/2,b=m/2,w=f+1,S=g+1;let T=0,E=0;const A=new zt;for(let s=0;s<S;s++){const a=s*x-M;for(let o=0;o<w;o++){const u=o*y-_;A[t]=u*i,A[e]=a*r,A[n]=b,l.push(A.x,A.y,A.z),A[t]=0,A[e]=0,A[n]=m>0?1:-1,c.push(A.x,A.y,A.z),h.push(o/f),h.push(1-s/g),T+=1}}for(let t=0;t<g;t++)for(let e=0;e<f;e++){const n=u+e+w*t,i=u+e+w*(t+1),r=u+(e+1)+w*(t+1),s=u+(e+1)+w*t;o.push(n,i,s),o.push(i,r,s),E+=6}a.addGroup(d,E,v),d+=E,u+=T}p("z","y","x",-1,-1,n,e,t,s,r,0),p("z","y","x",1,-1,n,e,-t,s,r,1),p("x","z","y",1,1,t,n,e,i,s,2),p("x","z","y",1,-1,t,n,-e,i,s,3),p("x","y","z",1,-1,t,e,n,i,r,4),p("x","y","z",-1,-1,t,e,-n,i,r,5),this.setIndex(o),this.setAttribute("position",new vn(l,3)),this.setAttribute("normal",new vn(c,3)),this.setAttribute("uv",new vn(h,2))}static fromJSON(t){return new qn(t.width,t.height,t.depth,t.widthSegments,t.heightSegments,t.depthSegments)}}function Xn(t){const e={};for(const n in t){e[n]={};for(const i in t[n]){const r=t[n][i];r&&(r.isColor||r.isMatrix3||r.isMatrix4||r.isVector2||r.isVector3||r.isVector4||r.isTexture||r.isQuaternion)?e[n][i]=r.clone():Array.isArray(r)?e[n][i]=r.slice():e[n][i]=r}}return e}function Yn(t){const e={};for(let n=0;n<t.length;n++){const i=Xn(t[n]);for(const t in i)e[t]=i[t]}return e}const Jn={clone:Xn,merge:Yn};class Zn extends Ze{constructor(t){super(),this.type="ShaderMaterial",this.defines={},this.uniforms={},this.vertexShader="void main() {\n\tgl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );\n}",this.fragmentShader="void main() {\n\tgl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );\n}",this.linewidth=1,this.wireframe=!1,this.wireframeLinewidth=1,this.fog=!1,this.lights=!1,this.clipping=!1,this.extensions={derivatives:!1,fragDepth:!1,drawBuffers:!1,shaderTextureLOD:!1},this.defaultAttributeValues={color:[1,1,1],uv:[0,0],uv2:[0,0]},this.index0AttributeName=void 0,this.uniformsNeedUpdate=!1,this.glslVersion=null,void 0!==t&&(void 0!==t.attributes&&console.error("THREE.ShaderMaterial: attributes should now be defined in THREE.BufferGeometry instead."),this.setValues(t))}copy(t){return super.copy(t),this.fragmentShader=t.fragmentShader,this.vertexShader=t.vertexShader,this.uniforms=Xn(t.uniforms),this.defines=Object.assign({},t.defines),this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this.lights=t.lights,this.clipping=t.clipping,this.extensions=Object.assign({},t.extensions),this.glslVersion=t.glslVersion,this}toJSON(t){const e=super.toJSON(t);e.glslVersion=this.glslVersion,e.uniforms={};for(const n in this.uniforms){const i=this.uniforms[n].value;i&&i.isTexture?e.uniforms[n]={type:"t",value:i.toJSON(t).uuid}:i&&i.isColor?e.uniforms[n]={type:"c",value:i.getHex()}:i&&i.isVector2?e.uniforms[n]={type:"v2",value:i.toArray()}:i&&i.isVector3?e.uniforms[n]={type:"v3",value:i.toArray()}:i&&i.isVector4?e.uniforms[n]={type:"v4",value:i.toArray()}:i&&i.isMatrix3?e.uniforms[n]={type:"m3",value:i.toArray()}:i&&i.isMatrix4?e.uniforms[n]={type:"m4",value:i.toArray()}:e.uniforms[n]={value:i}}Object.keys(this.defines).length>0&&(e.defines=this.defines),e.vertexShader=this.vertexShader,e.fragmentShader=this.fragmentShader;const n={};for(const t in this.extensions)!0===this.extensions[t]&&(n[t]=!0);return Object.keys(n).length>0&&(e.extensions=n),e}}Zn.prototype.isShaderMaterial=!0;class Qn extends Fe{constructor(){super(),this.type="Camera",this.matrixWorldInverse=new de,this.projectionMatrix=new de,this.projectionMatrixInverse=new de}copy(t,e){return super.copy(t,e),this.matrixWorldInverse.copy(t.matrixWorldInverse),this.projectionMatrix.copy(t.projectionMatrix),this.projectionMatrixInverse.copy(t.projectionMatrixInverse),this}getWorldDirection(t){this.updateWorldMatrix(!0,!1);const e=this.matrixWorld.elements;return t.set(-e[8],-e[9],-e[10]).normalize()}updateMatrixWorld(t){super.updateMatrixWorld(t),this.matrixWorldInverse.copy(this.matrixWorld).invert()}updateWorldMatrix(t,e){super.updateWorldMatrix(t,e),this.matrixWorldInverse.copy(this.matrixWorld).invert()}clone(){return(new this.constructor).copy(this)}}Qn.prototype.isCamera=!0;class Kn extends Qn{constructor(t=50,e=1,n=.1,i=2e3){super(),this.type="PerspectiveCamera",this.fov=t,this.zoom=1,this.near=n,this.far=i,this.focus=10,this.aspect=e,this.view=null,this.filmGauge=35,this.filmOffset=0,this.updateProjectionMatrix()}copy(t,e){return super.copy(t,e),this.fov=t.fov,this.zoom=t.zoom,this.near=t.near,this.far=t.far,this.focus=t.focus,this.aspect=t.aspect,this.view=null===t.view?null:Object.assign({},t.view),this.filmGauge=t.filmGauge,this.filmOffset=t.filmOffset,this}setFocalLength(t){const e=.5*this.getFilmHeight()/t;this.fov=2*ot*Math.atan(e),this.updateProjectionMatrix()}getFocalLength(){const t=Math.tan(.5*at*this.fov);return.5*this.getFilmHeight()/t}getEffectiveFOV(){return 2*ot*Math.atan(Math.tan(.5*at*this.fov)/this.zoom)}getFilmWidth(){return this.filmGauge*Math.min(this.aspect,1)}getFilmHeight(){return this.filmGauge/Math.max(this.aspect,1)}setViewOffset(t,e,n,i,r,s){this.aspect=t/e,null===this.view&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=t,this.view.fullHeight=e,this.view.offsetX=n,this.view.offsetY=i,this.view.width=r,this.view.height=s,this.updateProjectionMatrix()}clearViewOffset(){null!==this.view&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const t=this.near;let e=t*Math.tan(.5*at*this.fov)/this.zoom,n=2*e,i=this.aspect*n,r=-.5*i;const s=this.view;if(null!==this.view&&this.view.enabled){const t=s.fullWidth,a=s.fullHeight;r+=s.offsetX*i/t,e-=s.offsetY*n/a,i*=s.width/t,n*=s.height/a}const a=this.filmOffset;0!==a&&(r+=t*a/this.getFilmWidth()),this.projectionMatrix.makePerspective(r,r+i,e,e-n,t,this.far),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(t){const e=super.toJSON(t);return e.object.fov=this.fov,e.object.zoom=this.zoom,e.object.near=this.near,e.object.far=this.far,e.object.focus=this.focus,e.object.aspect=this.aspect,null!==this.view&&(e.object.view=Object.assign({},this.view)),e.object.filmGauge=this.filmGauge,e.object.filmOffset=this.filmOffset,e}}Kn.prototype.isPerspectiveCamera=!0;const $n=90;class ti extends Fe{constructor(t,e,n){if(super(),this.type="CubeCamera",!0!==n.isWebGLCubeRenderTarget)return void console.error("THREE.CubeCamera: The constructor now expects an instance of WebGLCubeRenderTarget as third parameter.");this.renderTarget=n;const i=new Kn($n,1,t,e);i.layers=this.layers,i.up.set(0,-1,0),i.lookAt(new zt(1,0,0)),this.add(i);const r=new Kn($n,1,t,e);r.layers=this.layers,r.up.set(0,-1,0),r.lookAt(new zt(-1,0,0)),this.add(r);const s=new Kn($n,1,t,e);s.layers=this.layers,s.up.set(0,0,1),s.lookAt(new zt(0,1,0)),this.add(s);const a=new Kn($n,1,t,e);a.layers=this.layers,a.up.set(0,0,-1),a.lookAt(new zt(0,-1,0)),this.add(a);const o=new Kn($n,1,t,e);o.layers=this.layers,o.up.set(0,-1,0),o.lookAt(new zt(0,0,1)),this.add(o);const l=new Kn($n,1,t,e);l.layers=this.layers,l.up.set(0,-1,0),l.lookAt(new zt(0,0,-1)),this.add(l)}update(t,e){null===this.parent&&this.updateMatrixWorld();const n=this.renderTarget,[i,r,s,a,o,l]=this.children,c=t.xr.enabled,h=t.getRenderTarget();t.xr.enabled=!1;const u=n.texture.generateMipmaps;n.texture.generateMipmaps=!1,t.setRenderTarget(n,0),t.render(e,i),t.setRenderTarget(n,1),t.render(e,r),t.setRenderTarget(n,2),t.render(e,s),t.setRenderTarget(n,3),t.render(e,a),t.setRenderTarget(n,4),t.render(e,o),n.texture.generateMipmaps=u,t.setRenderTarget(n,5),t.render(e,l),t.setRenderTarget(h),t.xr.enabled=c}}class ei extends Lt{constructor(t,e,n,i,s,a,o,l,c,h){super(t=void 0!==t?t:[],e=void 0!==e?e:r,n,i,s,a,o,l,c,h),this.flipY=!1}get images(){return this.image}set images(t){this.image=t}}ei.prototype.isCubeTexture=!0;class ni extends Pt{constructor(t,e,n){Number.isInteger(e)&&(console.warn("THREE.WebGLCubeRenderTarget: constructor signature is now WebGLCubeRenderTarget( size, options )"),e=n),super(t,t,e),e=e||{},this.texture=new ei(void 0,e.mapping,e.wrapS,e.wrapT,e.magFilter,e.minFilter,e.format,e.type,e.anisotropy,e.encoding),this.texture.isRenderTargetTexture=!0,this.texture.generateMipmaps=void 0!==e.generateMipmaps&&e.generateMipmaps,this.texture.minFilter=void 0!==e.minFilter?e.minFilter:g,this.texture._needsFlipEnvMap=!1}fromEquirectangularTexture(t,e){this.texture.type=e.type,this.texture.format=E,this.texture.encoding=e.encoding,this.texture.generateMipmaps=e.generateMipmaps,this.texture.minFilter=e.minFilter,this.texture.magFilter=e.magFilter;const n={uniforms:{tEquirect:{value:null}},vertexShader:"\n\n\t\t\t\tvarying vec3 vWorldDirection;\n\n\t\t\t\tvec3 transformDirection( in vec3 dir, in mat4 matrix ) {\n\n\t\t\t\t\treturn normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );\n\n\t\t\t\t}\n\n\t\t\t\tvoid main() {\n\n\t\t\t\t\tvWorldDirection = transformDirection( position, modelMatrix );\n\n\t\t\t\t\t#include <begin_vertex>\n\t\t\t\t\t#include <project_vertex>\n\n\t\t\t\t}\n\t\t\t",fragmentShader:"\n\n\t\t\t\tuniform sampler2D tEquirect;\n\n\t\t\t\tvarying vec3 vWorldDirection;\n\n\t\t\t\t#include <common>\n\n\t\t\t\tvoid main() {\n\n\t\t\t\t\tvec3 direction = normalize( vWorldDirection );\n\n\t\t\t\t\tvec2 sampleUV = equirectUv( direction );\n\n\t\t\t\t\tgl_FragColor = texture2D( tEquirect, sampleUV );\n\n\t\t\t\t}\n\t\t\t"},i=new qn(5,5,5),r=new Zn({name:"CubemapFromEquirect",uniforms:Xn(n.uniforms),vertexShader:n.vertexShader,fragmentShader:n.fragmentShader,side:1,blending:0});r.uniforms.tEquirect.value=e;const s=new Wn(i,r),a=e.minFilter;e.minFilter===y&&(e.minFilter=g);return new ti(1,10,this).update(t,s),e.minFilter=a,s.geometry.dispose(),s.material.dispose(),this}clear(t,e,n,i){const r=t.getRenderTarget();for(let r=0;r<6;r++)t.setRenderTarget(this,r),t.clear(e,n,i);t.setRenderTarget(r)}}ni.prototype.isWebGLCubeRenderTarget=!0;const ii=new zt,ri=new zt,si=new xt;class ai{constructor(t=new zt(1,0,0),e=0){this.normal=t,this.constant=e}set(t,e){return this.normal.copy(t),this.constant=e,this}setComponents(t,e,n,i){return this.normal.set(t,e,n),this.constant=i,this}setFromNormalAndCoplanarPoint(t,e){return this.normal.copy(t),this.constant=-e.dot(this.normal),this}setFromCoplanarPoints(t,e,n){const i=ii.subVectors(n,e).cross(ri.subVectors(t,e)).normalize();return this.setFromNormalAndCoplanarPoint(i,t),this}copy(t){return this.normal.copy(t.normal),this.constant=t.constant,this}normalize(){const t=1/this.normal.length();return this.normal.multiplyScalar(t),this.constant*=t,this}negate(){return this.constant*=-1,this.normal.negate(),this}distanceToPoint(t){return this.normal.dot(t)+this.constant}distanceToSphere(t){return this.distanceToPoint(t.center)-t.radius}projectPoint(t,e){return e.copy(this.normal).multiplyScalar(-this.distanceToPoint(t)).add(t)}intersectLine(t,e){const n=t.delta(ii),i=this.normal.dot(n);if(0===i)return 0===this.distanceToPoint(t.start)?e.copy(t.start):null;const r=-(t.start.dot(this.normal)+this.constant)/i;return r<0||r>1?null:e.copy(n).multiplyScalar(r).add(t.start)}intersectsLine(t){const e=this.distanceToPoint(t.start),n=this.distanceToPoint(t.end);return e<0&&n>0||n<0&&e>0}intersectsBox(t){return t.intersectsPlane(this)}intersectsSphere(t){return t.intersectsPlane(this)}coplanarPoint(t){return t.copy(this.normal).multiplyScalar(-this.constant)}applyMatrix4(t,e){const n=e||si.getNormalMatrix(t),i=this.coplanarPoint(ii).applyMatrix4(t),r=this.normal.applyMatrix3(n).normalize();return this.constant=-i.dot(r),this}translate(t){return this.constant-=t.dot(this.normal),this}equals(t){return t.normal.equals(this.normal)&&t.constant===this.constant}clone(){return(new this.constructor).copy(this)}}ai.prototype.isPlane=!0;const oi=new ie,li=new zt;class ci{constructor(t=new ai,e=new ai,n=new ai,i=new ai,r=new ai,s=new ai){this.planes=[t,e,n,i,r,s]}set(t,e,n,i,r,s){const a=this.planes;return a[0].copy(t),a[1].copy(e),a[2].copy(n),a[3].copy(i),a[4].copy(r),a[5].copy(s),this}copy(t){const e=this.planes;for(let n=0;n<6;n++)e[n].copy(t.planes[n]);return this}setFromProjectionMatrix(t){const e=this.planes,n=t.elements,i=n[0],r=n[1],s=n[2],a=n[3],o=n[4],l=n[5],c=n[6],h=n[7],u=n[8],d=n[9],p=n[10],m=n[11],f=n[12],g=n[13],v=n[14],y=n[15];return e[0].setComponents(a-i,h-o,m-u,y-f).normalize(),e[1].setComponents(a+i,h+o,m+u,y+f).normalize(),e[2].setComponents(a+r,h+l,m+d,y+g).normalize(),e[3].setComponents(a-r,h-l,m-d,y-g).normalize(),e[4].setComponents(a-s,h-c,m-p,y-v).normalize(),e[5].setComponents(a+s,h+c,m+p,y+v).normalize(),this}intersectsObject(t){const e=t.geometry;return null===e.boundingSphere&&e.computeBoundingSphere(),oi.copy(e.boundingSphere).applyMatrix4(t.matrixWorld),this.intersectsSphere(oi)}intersectsSprite(t){return oi.center.set(0,0,0),oi.radius=.7071067811865476,oi.applyMatrix4(t.matrixWorld),this.intersectsSphere(oi)}intersectsSphere(t){const e=this.planes,n=t.center,i=-t.radius;for(let t=0;t<6;t++){if(e[t].distanceToPoint(n)<i)return!1}return!0}intersectsBox(t){const e=this.planes;for(let n=0;n<6;n++){const i=e[n];if(li.x=i.normal.x>0?t.max.x:t.min.x,li.y=i.normal.y>0?t.max.y:t.min.y,li.z=i.normal.z>0?t.max.z:t.min.z,i.distanceToPoint(li)<0)return!1}return!0}containsPoint(t){const e=this.planes;for(let n=0;n<6;n++)if(e[n].distanceToPoint(t)<0)return!1;return!0}clone(){return(new this.constructor).copy(this)}}function hi(){let t=null,e=!1,n=null,i=null;function r(e,s){n(e,s),i=t.requestAnimationFrame(r)}return{start:function(){!0!==e&&null!==n&&(i=t.requestAnimationFrame(r),e=!0)},stop:function(){t.cancelAnimationFrame(i),e=!1},setAnimationLoop:function(t){n=t},setContext:function(e){t=e}}}function ui(t,e){const n=e.isWebGL2,i=new WeakMap;return{get:function(t){return t.isInterleavedBufferAttribute&&(t=t.data),i.get(t)},remove:function(e){e.isInterleavedBufferAttribute&&(e=e.data);const n=i.get(e);n&&(t.deleteBuffer(n.buffer),i.delete(e))},update:function(e,r){if(e.isGLBufferAttribute){const t=i.get(e);return void((!t||t.version<e.version)&&i.set(e,{buffer:e.buffer,type:e.type,bytesPerElement:e.elementSize,version:e.version}))}e.isInterleavedBufferAttribute&&(e=e.data);const s=i.get(e);void 0===s?i.set(e,function(e,i){const r=e.array,s=e.usage,a=t.createBuffer();t.bindBuffer(i,a),t.bufferData(i,r,s),e.onUploadCallback();let o=5126;return r instanceof Float32Array?o=5126:r instanceof Float64Array?console.warn("THREE.WebGLAttributes: Unsupported data buffer format: Float64Array."):r instanceof Uint16Array?e.isFloat16BufferAttribute?n?o=5131:console.warn("THREE.WebGLAttributes: Usage of Float16BufferAttribute requires WebGL2."):o=5123:r instanceof Int16Array?o=5122:r instanceof Uint32Array?o=5125:r instanceof Int32Array?o=5124:r instanceof Int8Array?o=5120:(r instanceof Uint8Array||r instanceof Uint8ClampedArray)&&(o=5121),{buffer:a,type:o,bytesPerElement:r.BYTES_PER_ELEMENT,version:e.version}}(e,r)):s.version<e.version&&(!function(e,i,r){const s=i.array,a=i.updateRange;t.bindBuffer(r,e),-1===a.count?t.bufferSubData(r,0,s):(n?t.bufferSubData(r,a.offset*s.BYTES_PER_ELEMENT,s,a.offset,a.count):t.bufferSubData(r,a.offset*s.BYTES_PER_ELEMENT,s.subarray(a.offset,a.offset+a.count)),a.count=-1)}(s.buffer,e,r),s.version=e.version)}}}class di extends En{constructor(t=1,e=1,n=1,i=1){super(),this.type="PlaneGeometry",this.parameters={width:t,height:e,widthSegments:n,heightSegments:i};const r=t/2,s=e/2,a=Math.floor(n),o=Math.floor(i),l=a+1,c=o+1,h=t/a,u=e/o,d=[],p=[],m=[],f=[];for(let t=0;t<c;t++){const e=t*u-s;for(let n=0;n<l;n++){const i=n*h-r;p.push(i,-e,0),m.push(0,0,1),f.push(n/a),f.push(1-t/o)}}for(let t=0;t<o;t++)for(let e=0;e<a;e++){const n=e+l*t,i=e+l*(t+1),r=e+1+l*(t+1),s=e+1+l*t;d.push(n,i,s),d.push(i,r,s)}this.setIndex(d),this.setAttribute("position",new vn(p,3)),this.setAttribute("normal",new vn(m,3)),this.setAttribute("uv",new vn(f,2))}static fromJSON(t){return new di(t.width,t.height,t.widthSegments,t.heightSegments)}}const pi={alphamap_fragment:"#ifdef USE_ALPHAMAP\n\tdiffuseColor.a *= texture2D( alphaMap, vUv ).g;\n#endif",alphamap_pars_fragment:"#ifdef USE_ALPHAMAP\n\tuniform sampler2D alphaMap;\n#endif",alphatest_fragment:"#ifdef USE_ALPHATEST\n\tif ( diffuseColor.a < alphaTest ) discard;\n#endif",alphatest_pars_fragment:"#ifdef USE_ALPHATEST\n\tuniform float alphaTest;\n#endif",aomap_fragment:"#ifdef USE_AOMAP\n\tfloat ambientOcclusion = ( texture2D( aoMap, vUv2 ).r - 1.0 ) * aoMapIntensity + 1.0;\n\treflectedLight.indirectDiffuse *= ambientOcclusion;\n\t#if defined( USE_ENVMAP ) && defined( STANDARD )\n\t\tfloat dotNV = saturate( dot( geometry.normal, geometry.viewDir ) );\n\t\treflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.roughness );\n\t#endif\n#endif",aomap_pars_fragment:"#ifdef USE_AOMAP\n\tuniform sampler2D aoMap;\n\tuniform float aoMapIntensity;\n#endif",begin_vertex:"vec3 transformed = vec3( position );",beginnormal_vertex:"vec3 objectNormal = vec3( normal );\n#ifdef USE_TANGENT\n\tvec3 objectTangent = vec3( tangent.xyz );\n#endif",bsdfs:"vec3 BRDF_Lambert( const in vec3 diffuseColor ) {\n\treturn RECIPROCAL_PI * diffuseColor;\n}\nvec3 F_Schlick( const in vec3 f0, const in float f90, const in float dotVH ) {\n\tfloat fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );\n\treturn f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );\n}\nfloat V_GGX_SmithCorrelated( const in float alpha, const in float dotNL, const in float dotNV ) {\n\tfloat a2 = pow2( alpha );\n\tfloat gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );\n\tfloat gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );\n\treturn 0.5 / max( gv + gl, EPSILON );\n}\nfloat D_GGX( const in float alpha, const in float dotNH ) {\n\tfloat a2 = pow2( alpha );\n\tfloat denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0;\n\treturn RECIPROCAL_PI * a2 / pow2( denom );\n}\nvec3 BRDF_GGX( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in vec3 f0, const in float f90, const in float roughness ) {\n\tfloat alpha = pow2( roughness );\n\tvec3 halfDir = normalize( lightDir + viewDir );\n\tfloat dotNL = saturate( dot( normal, lightDir ) );\n\tfloat dotNV = saturate( dot( normal, viewDir ) );\n\tfloat dotNH = saturate( dot( normal, halfDir ) );\n\tfloat dotVH = saturate( dot( viewDir, halfDir ) );\n\tvec3 F = F_Schlick( f0, f90, dotVH );\n\tfloat V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );\n\tfloat D = D_GGX( alpha, dotNH );\n\treturn F * ( V * D );\n}\nvec2 LTC_Uv( const in vec3 N, const in vec3 V, const in float roughness ) {\n\tconst float LUT_SIZE = 64.0;\n\tconst float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;\n\tconst float LUT_BIAS = 0.5 / LUT_SIZE;\n\tfloat dotNV = saturate( dot( N, V ) );\n\tvec2 uv = vec2( roughness, sqrt( 1.0 - dotNV ) );\n\tuv = uv * LUT_SCALE + LUT_BIAS;\n\treturn uv;\n}\nfloat LTC_ClippedSphereFormFactor( const in vec3 f ) {\n\tfloat l = length( f );\n\treturn max( ( l * l + f.z ) / ( l + 1.0 ), 0.0 );\n}\nvec3 LTC_EdgeVectorFormFactor( const in vec3 v1, const in vec3 v2 ) {\n\tfloat x = dot( v1, v2 );\n\tfloat y = abs( x );\n\tfloat a = 0.8543985 + ( 0.4965155 + 0.0145206 * y ) * y;\n\tfloat b = 3.4175940 + ( 4.1616724 + y ) * y;\n\tfloat v = a / b;\n\tfloat theta_sintheta = ( x > 0.0 ) ? v : 0.5 * inversesqrt( max( 1.0 - x * x, 1e-7 ) ) - v;\n\treturn cross( v1, v2 ) * theta_sintheta;\n}\nvec3 LTC_Evaluate( const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[ 4 ] ) {\n\tvec3 v1 = rectCoords[ 1 ] - rectCoords[ 0 ];\n\tvec3 v2 = rectCoords[ 3 ] - rectCoords[ 0 ];\n\tvec3 lightNormal = cross( v1, v2 );\n\tif( dot( lightNormal, P - rectCoords[ 0 ] ) < 0.0 ) return vec3( 0.0 );\n\tvec3 T1, T2;\n\tT1 = normalize( V - N * dot( V, N ) );\n\tT2 = - cross( N, T1 );\n\tmat3 mat = mInv * transposeMat3( mat3( T1, T2, N ) );\n\tvec3 coords[ 4 ];\n\tcoords[ 0 ] = mat * ( rectCoords[ 0 ] - P );\n\tcoords[ 1 ] = mat * ( rectCoords[ 1 ] - P );\n\tcoords[ 2 ] = mat * ( rectCoords[ 2 ] - P );\n\tcoords[ 3 ] = mat * ( rectCoords[ 3 ] - P );\n\tcoords[ 0 ] = normalize( coords[ 0 ] );\n\tcoords[ 1 ] = normalize( coords[ 1 ] );\n\tcoords[ 2 ] = normalize( coords[ 2 ] );\n\tcoords[ 3 ] = normalize( coords[ 3 ] );\n\tvec3 vectorFormFactor = vec3( 0.0 );\n\tvectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );\n\tvectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );\n\tvectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );\n\tvectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );\n\tfloat result = LTC_ClippedSphereFormFactor( vectorFormFactor );\n\treturn vec3( result );\n}\nfloat G_BlinnPhong_Implicit( ) {\n\treturn 0.25;\n}\nfloat D_BlinnPhong( const in float shininess, const in float dotNH ) {\n\treturn RECIPROCAL_PI * ( shininess * 0.5 + 1.0 ) * pow( dotNH, shininess );\n}\nvec3 BRDF_BlinnPhong( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in vec3 specularColor, const in float shininess ) {\n\tvec3 halfDir = normalize( lightDir + viewDir );\n\tfloat dotNH = saturate( dot( normal, halfDir ) );\n\tfloat dotVH = saturate( dot( viewDir, halfDir ) );\n\tvec3 F = F_Schlick( specularColor, 1.0, dotVH );\n\tfloat G = G_BlinnPhong_Implicit( );\n\tfloat D = D_BlinnPhong( shininess, dotNH );\n\treturn F * ( G * D );\n}\n#if defined( USE_SHEEN )\nfloat D_Charlie( float roughness, float dotNH ) {\n\tfloat alpha = pow2( roughness );\n\tfloat invAlpha = 1.0 / alpha;\n\tfloat cos2h = dotNH * dotNH;\n\tfloat sin2h = max( 1.0 - cos2h, 0.0078125 );\n\treturn ( 2.0 + invAlpha ) * pow( sin2h, invAlpha * 0.5 ) / ( 2.0 * PI );\n}\nfloat V_Neubelt( float dotNV, float dotNL ) {\n\treturn saturate( 1.0 / ( 4.0 * ( dotNL + dotNV - dotNL * dotNV ) ) );\n}\nvec3 BRDF_Sheen( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, vec3 sheenColor, const in float sheenRoughness ) {\n\tvec3 halfDir = normalize( lightDir + viewDir );\n\tfloat dotNL = saturate( dot( normal, lightDir ) );\n\tfloat dotNV = saturate( dot( normal, viewDir ) );\n\tfloat dotNH = saturate( dot( normal, halfDir ) );\n\tfloat D = D_Charlie( sheenRoughness, dotNH );\n\tfloat V = V_Neubelt( dotNV, dotNL );\n\treturn sheenColor * ( D * V );\n}\n#endif",bumpmap_pars_fragment:"#ifdef USE_BUMPMAP\n\tuniform sampler2D bumpMap;\n\tuniform float bumpScale;\n\tvec2 dHdxy_fwd() {\n\t\tvec2 dSTdx = dFdx( vUv );\n\t\tvec2 dSTdy = dFdy( vUv );\n\t\tfloat Hll = bumpScale * texture2D( bumpMap, vUv ).x;\n\t\tfloat dBx = bumpScale * texture2D( bumpMap, vUv + dSTdx ).x - Hll;\n\t\tfloat dBy = bumpScale * texture2D( bumpMap, vUv + dSTdy ).x - Hll;\n\t\treturn vec2( dBx, dBy );\n\t}\n\tvec3 perturbNormalArb( vec3 surf_pos, vec3 surf_norm, vec2 dHdxy, float faceDirection ) {\n\t\tvec3 vSigmaX = vec3( dFdx( surf_pos.x ), dFdx( surf_pos.y ), dFdx( surf_pos.z ) );\n\t\tvec3 vSigmaY = vec3( dFdy( surf_pos.x ), dFdy( surf_pos.y ), dFdy( surf_pos.z ) );\n\t\tvec3 vN = surf_norm;\n\t\tvec3 R1 = cross( vSigmaY, vN );\n\t\tvec3 R2 = cross( vN, vSigmaX );\n\t\tfloat fDet = dot( vSigmaX, R1 ) * faceDirection;\n\t\tvec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );\n\t\treturn normalize( abs( fDet ) * surf_norm - vGrad );\n\t}\n#endif",clipping_planes_fragment:"#if NUM_CLIPPING_PLANES > 0\n\tvec4 plane;\n\t#pragma unroll_loop_start\n\tfor ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {\n\t\tplane = clippingPlanes[ i ];\n\t\tif ( dot( vClipPosition, plane.xyz ) > plane.w ) discard;\n\t}\n\t#pragma unroll_loop_end\n\t#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES\n\t\tbool clipped = true;\n\t\t#pragma unroll_loop_start\n\t\tfor ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {\n\t\t\tplane = clippingPlanes[ i ];\n\t\t\tclipped = ( dot( vClipPosition, plane.xyz ) > plane.w ) && clipped;\n\t\t}\n\t\t#pragma unroll_loop_end\n\t\tif ( clipped ) discard;\n\t#endif\n#endif",clipping_planes_pars_fragment:"#if NUM_CLIPPING_PLANES > 0\n\tvarying vec3 vClipPosition;\n\tuniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];\n#endif",clipping_planes_pars_vertex:"#if NUM_CLIPPING_PLANES > 0\n\tvarying vec3 vClipPosition;\n#endif",clipping_planes_vertex:"#if NUM_CLIPPING_PLANES > 0\n\tvClipPosition = - mvPosition.xyz;\n#endif",color_fragment:"#if defined( USE_COLOR_ALPHA )\n\tdiffuseColor *= vColor;\n#elif defined( USE_COLOR )\n\tdiffuseColor.rgb *= vColor;\n#endif",color_pars_fragment:"#if defined( USE_COLOR_ALPHA )\n\tvarying vec4 vColor;\n#elif defined( USE_COLOR )\n\tvarying vec3 vColor;\n#endif",color_pars_vertex:"#if defined( USE_COLOR_ALPHA )\n\tvarying vec4 vColor;\n#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR )\n\tvarying vec3 vColor;\n#endif",color_vertex:"#if defined( USE_COLOR_ALPHA )\n\tvColor = vec4( 1.0 );\n#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR )\n\tvColor = vec3( 1.0 );\n#endif\n#ifdef USE_COLOR\n\tvColor *= color;\n#endif\n#ifdef USE_INSTANCING_COLOR\n\tvColor.xyz *= instanceColor.xyz;\n#endif",common:"#define PI 3.141592653589793\n#define PI2 6.283185307179586\n#define PI_HALF 1.5707963267948966\n#define RECIPROCAL_PI 0.3183098861837907\n#define RECIPROCAL_PI2 0.15915494309189535\n#define EPSILON 1e-6\n#ifndef saturate\n#define saturate( a ) clamp( a, 0.0, 1.0 )\n#endif\n#define whiteComplement( a ) ( 1.0 - saturate( a ) )\nfloat pow2( const in float x ) { return x*x; }\nfloat pow3( const in float x ) { return x*x*x; }\nfloat pow4( const in float x ) { float x2 = x*x; return x2*x2; }\nfloat max3( const in vec3 v ) { return max( max( v.x, v.y ), v.z ); }\nfloat average( const in vec3 color ) { return dot( color, vec3( 0.3333 ) ); }\nhighp float rand( const in vec2 uv ) {\n\tconst highp float a = 12.9898, b = 78.233, c = 43758.5453;\n\thighp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );\n\treturn fract( sin( sn ) * c );\n}\n#ifdef HIGH_PRECISION\n\tfloat precisionSafeLength( vec3 v ) { return length( v ); }\n#else\n\tfloat precisionSafeLength( vec3 v ) {\n\t\tfloat maxComponent = max3( abs( v ) );\n\t\treturn length( v / maxComponent ) * maxComponent;\n\t}\n#endif\nstruct IncidentLight {\n\tvec3 color;\n\tvec3 direction;\n\tbool visible;\n};\nstruct ReflectedLight {\n\tvec3 directDiffuse;\n\tvec3 directSpecular;\n\tvec3 indirectDiffuse;\n\tvec3 indirectSpecular;\n};\nstruct GeometricContext {\n\tvec3 position;\n\tvec3 normal;\n\tvec3 viewDir;\n#ifdef USE_CLEARCOAT\n\tvec3 clearcoatNormal;\n#endif\n};\nvec3 transformDirection( in vec3 dir, in mat4 matrix ) {\n\treturn normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );\n}\nvec3 inverseTransformDirection( in vec3 dir, in mat4 matrix ) {\n\treturn normalize( ( vec4( dir, 0.0 ) * matrix ).xyz );\n}\nmat3 transposeMat3( const in mat3 m ) {\n\tmat3 tmp;\n\ttmp[ 0 ] = vec3( m[ 0 ].x, m[ 1 ].x, m[ 2 ].x );\n\ttmp[ 1 ] = vec3( m[ 0 ].y, m[ 1 ].y, m[ 2 ].y );\n\ttmp[ 2 ] = vec3( m[ 0 ].z, m[ 1 ].z, m[ 2 ].z );\n\treturn tmp;\n}\nfloat linearToRelativeLuminance( const in vec3 color ) {\n\tvec3 weights = vec3( 0.2126, 0.7152, 0.0722 );\n\treturn dot( weights, color.rgb );\n}\nbool isPerspectiveMatrix( mat4 m ) {\n\treturn m[ 2 ][ 3 ] == - 1.0;\n}\nvec2 equirectUv( in vec3 dir ) {\n\tfloat u = atan( dir.z, dir.x ) * RECIPROCAL_PI2 + 0.5;\n\tfloat v = asin( clamp( dir.y, - 1.0, 1.0 ) ) * RECIPROCAL_PI + 0.5;\n\treturn vec2( u, v );\n}",cube_uv_reflection_fragment:"#ifdef ENVMAP_TYPE_CUBE_UV\n\t#define cubeUV_maxMipLevel 8.0\n\t#define cubeUV_minMipLevel 4.0\n\t#define cubeUV_maxTileSize 256.0\n\t#define cubeUV_minTileSize 16.0\n\tfloat getFace( vec3 direction ) {\n\t\tvec3 absDirection = abs( direction );\n\t\tfloat face = - 1.0;\n\t\tif ( absDirection.x > absDirection.z ) {\n\t\t\tif ( absDirection.x > absDirection.y )\n\t\t\t\tface = direction.x > 0.0 ? 0.0 : 3.0;\n\t\t\telse\n\t\t\t\tface = direction.y > 0.0 ? 1.0 : 4.0;\n\t\t} else {\n\t\t\tif ( absDirection.z > absDirection.y )\n\t\t\t\tface = direction.z > 0.0 ? 2.0 : 5.0;\n\t\t\telse\n\t\t\t\tface = direction.y > 0.0 ? 1.0 : 4.0;\n\t\t}\n\t\treturn face;\n\t}\n\tvec2 getUV( vec3 direction, float face ) {\n\t\tvec2 uv;\n\t\tif ( face == 0.0 ) {\n\t\t\tuv = vec2( direction.z, direction.y ) / abs( direction.x );\n\t\t} else if ( face == 1.0 ) {\n\t\t\tuv = vec2( - direction.x, - direction.z ) / abs( direction.y );\n\t\t} else if ( face == 2.0 ) {\n\t\t\tuv = vec2( - direction.x, direction.y ) / abs( direction.z );\n\t\t} else if ( face == 3.0 ) {\n\t\t\tuv = vec2( - direction.z, direction.y ) / abs( direction.x );\n\t\t} else if ( face == 4.0 ) {\n\t\t\tuv = vec2( - direction.x, direction.z ) / abs( direction.y );\n\t\t} else {\n\t\t\tuv = vec2( direction.x, direction.y ) / abs( direction.z );\n\t\t}\n\t\treturn 0.5 * ( uv + 1.0 );\n\t}\n\tvec3 bilinearCubeUV( sampler2D envMap, vec3 direction, float mipInt ) {\n\t\tfloat face = getFace( direction );\n\t\tfloat filterInt = max( cubeUV_minMipLevel - mipInt, 0.0 );\n\t\tmipInt = max( mipInt, cubeUV_minMipLevel );\n\t\tfloat faceSize = exp2( mipInt );\n\t\tfloat texelSize = 1.0 / ( 3.0 * cubeUV_maxTileSize );\n\t\tvec2 uv = getUV( direction, face ) * ( faceSize - 1.0 );\n\t\tvec2 f = fract( uv );\n\t\tuv += 0.5 - f;\n\t\tif ( face > 2.0 ) {\n\t\t\tuv.y += faceSize;\n\t\t\tface -= 3.0;\n\t\t}\n\t\tuv.x += face * faceSize;\n\t\tif ( mipInt < cubeUV_maxMipLevel ) {\n\t\t\tuv.y += 2.0 * cubeUV_maxTileSize;\n\t\t}\n\t\tuv.y += filterInt * 2.0 * cubeUV_minTileSize;\n\t\tuv.x += 3.0 * max( 0.0, cubeUV_maxTileSize - 2.0 * faceSize );\n\t\tuv *= texelSize;\n\t\tvec3 tl = envMapTexelToLinear( texture2D( envMap, uv ) ).rgb;\n\t\tuv.x += texelSize;\n\t\tvec3 tr = envMapTexelToLinear( texture2D( envMap, uv ) ).rgb;\n\t\tuv.y += texelSize;\n\t\tvec3 br = envMapTexelToLinear( texture2D( envMap, uv ) ).rgb;\n\t\tuv.x -= texelSize;\n\t\tvec3 bl = envMapTexelToLinear( texture2D( envMap, uv ) ).rgb;\n\t\tvec3 tm = mix( tl, tr, f.x );\n\t\tvec3 bm = mix( bl, br, f.x );\n\t\treturn mix( tm, bm, f.y );\n\t}\n\t#define r0 1.0\n\t#define v0 0.339\n\t#define m0 - 2.0\n\t#define r1 0.8\n\t#define v1 0.276\n\t#define m1 - 1.0\n\t#define r4 0.4\n\t#define v4 0.046\n\t#define m4 2.0\n\t#define r5 0.305\n\t#define v5 0.016\n\t#define m5 3.0\n\t#define r6 0.21\n\t#define v6 0.0038\n\t#define m6 4.0\n\tfloat roughnessToMip( float roughness ) {\n\t\tfloat mip = 0.0;\n\t\tif ( roughness >= r1 ) {\n\t\t\tmip = ( r0 - roughness ) * ( m1 - m0 ) / ( r0 - r1 ) + m0;\n\t\t} else if ( roughness >= r4 ) {\n\t\t\tmip = ( r1 - roughness ) * ( m4 - m1 ) / ( r1 - r4 ) + m1;\n\t\t} else if ( roughness >= r5 ) {\n\t\t\tmip = ( r4 - roughness ) * ( m5 - m4 ) / ( r4 - r5 ) + m4;\n\t\t} else if ( roughness >= r6 ) {\n\t\t\tmip = ( r5 - roughness ) * ( m6 - m5 ) / ( r5 - r6 ) + m5;\n\t\t} else {\n\t\t\tmip = - 2.0 * log2( 1.16 * roughness );\t\t}\n\t\treturn mip;\n\t}\n\tvec4 textureCubeUV( sampler2D envMap, vec3 sampleDir, float roughness ) {\n\t\tfloat mip = clamp( roughnessToMip( roughness ), m0, cubeUV_maxMipLevel );\n\t\tfloat mipF = fract( mip );\n\t\tfloat mipInt = floor( mip );\n\t\tvec3 color0 = bilinearCubeUV( envMap, sampleDir, mipInt );\n\t\tif ( mipF == 0.0 ) {\n\t\t\treturn vec4( color0, 1.0 );\n\t\t} else {\n\t\t\tvec3 color1 = bilinearCubeUV( envMap, sampleDir, mipInt + 1.0 );\n\t\t\treturn vec4( mix( color0, color1, mipF ), 1.0 );\n\t\t}\n\t}\n#endif",defaultnormal_vertex:"vec3 transformedNormal = objectNormal;\n#ifdef USE_INSTANCING\n\tmat3 m = mat3( instanceMatrix );\n\ttransformedNormal /= vec3( dot( m[ 0 ], m[ 0 ] ), dot( m[ 1 ], m[ 1 ] ), dot( m[ 2 ], m[ 2 ] ) );\n\ttransformedNormal = m * transformedNormal;\n#endif\ntransformedNormal = normalMatrix * transformedNormal;\n#ifdef FLIP_SIDED\n\ttransformedNormal = - transformedNormal;\n#endif\n#ifdef USE_TANGENT\n\tvec3 transformedTangent = ( modelViewMatrix * vec4( objectTangent, 0.0 ) ).xyz;\n\t#ifdef FLIP_SIDED\n\t\ttransformedTangent = - transformedTangent;\n\t#endif\n#endif",displacementmap_pars_vertex:"#ifdef USE_DISPLACEMENTMAP\n\tuniform sampler2D displacementMap;\n\tuniform float displacementScale;\n\tuniform float displacementBias;\n#endif",displacementmap_vertex:"#ifdef USE_DISPLACEMENTMAP\n\ttransformed += normalize( objectNormal ) * ( texture2D( displacementMap, vUv ).x * displacementScale + displacementBias );\n#endif",emissivemap_fragment:"#ifdef USE_EMISSIVEMAP\n\tvec4 emissiveColor = texture2D( emissiveMap, vUv );\n\temissiveColor.rgb = emissiveMapTexelToLinear( emissiveColor ).rgb;\n\ttotalEmissiveRadiance *= emissiveColor.rgb;\n#endif",emissivemap_pars_fragment:"#ifdef USE_EMISSIVEMAP\n\tuniform sampler2D emissiveMap;\n#endif",encodings_fragment:"gl_FragColor = linearToOutputTexel( gl_FragColor );",encodings_pars_fragment:"\nvec4 LinearToLinear( in vec4 value ) {\n\treturn value;\n}\nvec4 GammaToLinear( in vec4 value, in float gammaFactor ) {\n\treturn vec4( pow( value.rgb, vec3( gammaFactor ) ), value.a );\n}\nvec4 LinearToGamma( in vec4 value, in float gammaFactor ) {\n\treturn vec4( pow( value.rgb, vec3( 1.0 / gammaFactor ) ), value.a );\n}\nvec4 sRGBToLinear( in vec4 value ) {\n\treturn vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.a );\n}\nvec4 LinearTosRGB( in vec4 value ) {\n\treturn vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );\n}\nvec4 RGBEToLinear( in vec4 value ) {\n\treturn vec4( value.rgb * exp2( value.a * 255.0 - 128.0 ), 1.0 );\n}\nvec4 LinearToRGBE( in vec4 value ) {\n\tfloat maxComponent = max( max( value.r, value.g ), value.b );\n\tfloat fExp = clamp( ceil( log2( maxComponent ) ), -128.0, 127.0 );\n\treturn vec4( value.rgb / exp2( fExp ), ( fExp + 128.0 ) / 255.0 );\n}\nvec4 RGBMToLinear( in vec4 value, in float maxRange ) {\n\treturn vec4( value.rgb * value.a * maxRange, 1.0 );\n}\nvec4 LinearToRGBM( in vec4 value, in float maxRange ) {\n\tfloat maxRGB = max( value.r, max( value.g, value.b ) );\n\tfloat M = clamp( maxRGB / maxRange, 0.0, 1.0 );\n\tM = ceil( M * 255.0 ) / 255.0;\n\treturn vec4( value.rgb / ( M * maxRange ), M );\n}\nvec4 RGBDToLinear( in vec4 value, in float maxRange ) {\n\treturn vec4( value.rgb * ( ( maxRange / 255.0 ) / value.a ), 1.0 );\n}\nvec4 LinearToRGBD( in vec4 value, in float maxRange ) {\n\tfloat maxRGB = max( value.r, max( value.g, value.b ) );\n\tfloat D = max( maxRange / maxRGB, 1.0 );\n\tD = clamp( floor( D ) / 255.0, 0.0, 1.0 );\n\treturn vec4( value.rgb * ( D * ( 255.0 / maxRange ) ), D );\n}\nconst mat3 cLogLuvM = mat3( 0.2209, 0.3390, 0.4184, 0.1138, 0.6780, 0.7319, 0.0102, 0.1130, 0.2969 );\nvec4 LinearToLogLuv( in vec4 value ) {\n\tvec3 Xp_Y_XYZp = cLogLuvM * value.rgb;\n\tXp_Y_XYZp = max( Xp_Y_XYZp, vec3( 1e-6, 1e-6, 1e-6 ) );\n\tvec4 vResult;\n\tvResult.xy = Xp_Y_XYZp.xy / Xp_Y_XYZp.z;\n\tfloat Le = 2.0 * log2(Xp_Y_XYZp.y) + 127.0;\n\tvResult.w = fract( Le );\n\tvResult.z = ( Le - ( floor( vResult.w * 255.0 ) ) / 255.0 ) / 255.0;\n\treturn vResult;\n}\nconst mat3 cLogLuvInverseM = mat3( 6.0014, -2.7008, -1.7996, -1.3320, 3.1029, -5.7721, 0.3008, -1.0882, 5.6268 );\nvec4 LogLuvToLinear( in vec4 value ) {\n\tfloat Le = value.z * 255.0 + value.w;\n\tvec3 Xp_Y_XYZp;\n\tXp_Y_XYZp.y = exp2( ( Le - 127.0 ) / 2.0 );\n\tXp_Y_XYZp.z = Xp_Y_XYZp.y / value.y;\n\tXp_Y_XYZp.x = value.x * Xp_Y_XYZp.z;\n\tvec3 vRGB = cLogLuvInverseM * Xp_Y_XYZp.rgb;\n\treturn vec4( max( vRGB, 0.0 ), 1.0 );\n}",envmap_fragment:"#ifdef USE_ENVMAP\n\t#ifdef ENV_WORLDPOS\n\t\tvec3 cameraToFrag;\n\t\tif ( isOrthographic ) {\n\t\t\tcameraToFrag = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );\n\t\t} else {\n\t\t\tcameraToFrag = normalize( vWorldPosition - cameraPosition );\n\t\t}\n\t\tvec3 worldNormal = inverseTransformDirection( normal, viewMatrix );\n\t\t#ifdef ENVMAP_MODE_REFLECTION\n\t\t\tvec3 reflectVec = reflect( cameraToFrag, worldNormal );\n\t\t#else\n\t\t\tvec3 reflectVec = refract( cameraToFrag, worldNormal, refractionRatio );\n\t\t#endif\n\t#else\n\t\tvec3 reflectVec = vReflect;\n\t#endif\n\t#ifdef ENVMAP_TYPE_CUBE\n\t\tvec4 envColor = textureCube( envMap, vec3( flipEnvMap * reflectVec.x, reflectVec.yz ) );\n\t\tenvColor = envMapTexelToLinear( envColor );\n\t#elif defined( ENVMAP_TYPE_CUBE_UV )\n\t\tvec4 envColor = textureCubeUV( envMap, reflectVec, 0.0 );\n\t#else\n\t\tvec4 envColor = vec4( 0.0 );\n\t#endif\n\t#ifdef ENVMAP_BLENDING_MULTIPLY\n\t\toutgoingLight = mix( outgoingLight, outgoingLight * envColor.xyz, specularStrength * reflectivity );\n\t#elif defined( ENVMAP_BLENDING_MIX )\n\t\toutgoingLight = mix( outgoingLight, envColor.xyz, specularStrength * reflectivity );\n\t#elif defined( ENVMAP_BLENDING_ADD )\n\t\toutgoingLight += envColor.xyz * specularStrength * reflectivity;\n\t#endif\n#endif",envmap_common_pars_fragment:"#ifdef USE_ENVMAP\n\tuniform float envMapIntensity;\n\tuniform float flipEnvMap;\n\tuniform int maxMipLevel;\n\t#ifdef ENVMAP_TYPE_CUBE\n\t\tuniform samplerCube envMap;\n\t#else\n\t\tuniform sampler2D envMap;\n\t#endif\n\t\n#endif",envmap_pars_fragment:"#ifdef USE_ENVMAP\n\tuniform float reflectivity;\n\t#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG )\n\t\t#define ENV_WORLDPOS\n\t#endif\n\t#ifdef ENV_WORLDPOS\n\t\tvarying vec3 vWorldPosition;\n\t\tuniform float refractionRatio;\n\t#else\n\t\tvarying vec3 vReflect;\n\t#endif\n#endif",envmap_pars_vertex:"#ifdef USE_ENVMAP\n\t#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) ||defined( PHONG )\n\t\t#define ENV_WORLDPOS\n\t#endif\n\t#ifdef ENV_WORLDPOS\n\t\t\n\t\tvarying vec3 vWorldPosition;\n\t#else\n\t\tvarying vec3 vReflect;\n\t\tuniform float refractionRatio;\n\t#endif\n#endif",envmap_physical_pars_fragment:"#if defined( USE_ENVMAP )\n\t#ifdef ENVMAP_MODE_REFRACTION\n\t\tuniform float refractionRatio;\n\t#endif\n\tvec3 getIBLIrradiance( const in vec3 normal ) {\n\t\t#if defined( ENVMAP_TYPE_CUBE_UV )\n\t\t\tvec3 worldNormal = inverseTransformDirection( normal, viewMatrix );\n\t\t\tvec4 envMapColor = textureCubeUV( envMap, worldNormal, 1.0 );\n\t\t\treturn PI * envMapColor.rgb * envMapIntensity;\n\t\t#else\n\t\t\treturn vec3( 0.0 );\n\t\t#endif\n\t}\n\tvec3 getIBLRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {\n\t\t#if defined( ENVMAP_TYPE_CUBE_UV )\n\t\t\tvec3 reflectVec;\n\t\t\t#ifdef ENVMAP_MODE_REFLECTION\n\t\t\t\treflectVec = reflect( - viewDir, normal );\n\t\t\t\treflectVec = normalize( mix( reflectVec, normal, roughness * roughness) );\n\t\t\t#else\n\t\t\t\treflectVec = refract( - viewDir, normal, refractionRatio );\n\t\t\t#endif\n\t\t\treflectVec = inverseTransformDirection( reflectVec, viewMatrix );\n\t\t\tvec4 envMapColor = textureCubeUV( envMap, reflectVec, roughness );\n\t\t\treturn envMapColor.rgb * envMapIntensity;\n\t\t#else\n\t\t\treturn vec3( 0.0 );\n\t\t#endif\n\t}\n#endif",envmap_vertex:"#ifdef USE_ENVMAP\n\t#ifdef ENV_WORLDPOS\n\t\tvWorldPosition = worldPosition.xyz;\n\t#else\n\t\tvec3 cameraToVertex;\n\t\tif ( isOrthographic ) {\n\t\t\tcameraToVertex = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );\n\t\t} else {\n\t\t\tcameraToVertex = normalize( worldPosition.xyz - cameraPosition );\n\t\t}\n\t\tvec3 worldNormal = inverseTransformDirection( transformedNormal, viewMatrix );\n\t\t#ifdef ENVMAP_MODE_REFLECTION\n\t\t\tvReflect = reflect( cameraToVertex, worldNormal );\n\t\t#else\n\t\t\tvReflect = refract( cameraToVertex, worldNormal, refractionRatio );\n\t\t#endif\n\t#endif\n#endif",fog_vertex:"#ifdef USE_FOG\n\tvFogDepth = - mvPosition.z;\n#endif",fog_pars_vertex:"#ifdef USE_FOG\n\tvarying float vFogDepth;\n#endif",fog_fragment:"#ifdef USE_FOG\n\t#ifdef FOG_EXP2\n\t\tfloat fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );\n\t#else\n\t\tfloat fogFactor = smoothstep( fogNear, fogFar, vFogDepth );\n\t#endif\n\tgl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );\n#endif",fog_pars_fragment:"#ifdef USE_FOG\n\tuniform vec3 fogColor;\n\tvarying float vFogDepth;\n\t#ifdef FOG_EXP2\n\t\tuniform float fogDensity;\n\t#else\n\t\tuniform float fogNear;\n\t\tuniform float fogFar;\n\t#endif\n#endif",gradientmap_pars_fragment:"#ifdef USE_GRADIENTMAP\n\tuniform sampler2D gradientMap;\n#endif\nvec3 getGradientIrradiance( vec3 normal, vec3 lightDirection ) {\n\tfloat dotNL = dot( normal, lightDirection );\n\tvec2 coord = vec2( dotNL * 0.5 + 0.5, 0.0 );\n\t#ifdef USE_GRADIENTMAP\n\t\treturn texture2D( gradientMap, coord ).rgb;\n\t#else\n\t\treturn ( coord.x < 0.7 ) ? vec3( 0.7 ) : vec3( 1.0 );\n\t#endif\n}",lightmap_fragment:"#ifdef USE_LIGHTMAP\n\tvec4 lightMapTexel = texture2D( lightMap, vUv2 );\n\tvec3 lightMapIrradiance = lightMapTexelToLinear( lightMapTexel ).rgb * lightMapIntensity;\n\t#ifndef PHYSICALLY_CORRECT_LIGHTS\n\t\tlightMapIrradiance *= PI;\n\t#endif\n\treflectedLight.indirectDiffuse += lightMapIrradiance;\n#endif",lightmap_pars_fragment:"#ifdef USE_LIGHTMAP\n\tuniform sampler2D lightMap;\n\tuniform float lightMapIntensity;\n#endif",lights_lambert_vertex:"vec3 diffuse = vec3( 1.0 );\nGeometricContext geometry;\ngeometry.position = mvPosition.xyz;\ngeometry.normal = normalize( transformedNormal );\ngeometry.viewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( -mvPosition.xyz );\nGeometricContext backGeometry;\nbackGeometry.position = geometry.position;\nbackGeometry.normal = -geometry.normal;\nbackGeometry.viewDir = geometry.viewDir;\nvLightFront = vec3( 0.0 );\nvIndirectFront = vec3( 0.0 );\n#ifdef DOUBLE_SIDED\n\tvLightBack = vec3( 0.0 );\n\tvIndirectBack = vec3( 0.0 );\n#endif\nIncidentLight directLight;\nfloat dotNL;\nvec3 directLightColor_Diffuse;\nvIndirectFront += getAmbientLightIrradiance( ambientLightColor );\nvIndirectFront += getLightProbeIrradiance( lightProbe, geometry.normal );\n#ifdef DOUBLE_SIDED\n\tvIndirectBack += getAmbientLightIrradiance( ambientLightColor );\n\tvIndirectBack += getLightProbeIrradiance( lightProbe, backGeometry.normal );\n#endif\n#if NUM_POINT_LIGHTS > 0\n\t#pragma unroll_loop_start\n\tfor ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {\n\t\tgetPointLightInfo( pointLights[ i ], geometry, directLight );\n\t\tdotNL = dot( geometry.normal, directLight.direction );\n\t\tdirectLightColor_Diffuse = directLight.color;\n\t\tvLightFront += saturate( dotNL ) * directLightColor_Diffuse;\n\t\t#ifdef DOUBLE_SIDED\n\t\t\tvLightBack += saturate( - dotNL ) * directLightColor_Diffuse;\n\t\t#endif\n\t}\n\t#pragma unroll_loop_end\n#endif\n#if NUM_SPOT_LIGHTS > 0\n\t#pragma unroll_loop_start\n\tfor ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {\n\t\tgetSpotLightInfo( spotLights[ i ], geometry, directLight );\n\t\tdotNL = dot( geometry.normal, directLight.direction );\n\t\tdirectLightColor_Diffuse = directLight.color;\n\t\tvLightFront += saturate( dotNL ) * directLightColor_Diffuse;\n\t\t#ifdef DOUBLE_SIDED\n\t\t\tvLightBack += saturate( - dotNL ) * directLightColor_Diffuse;\n\t\t#endif\n\t}\n\t#pragma unroll_loop_end\n#endif\n#if NUM_DIR_LIGHTS > 0\n\t#pragma unroll_loop_start\n\tfor ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {\n\t\tgetDirectionalLightInfo( directionalLights[ i ], geometry, directLight );\n\t\tdotNL = dot( geometry.normal, directLight.direction );\n\t\tdirectLightColor_Diffuse = directLight.color;\n\t\tvLightFront += saturate( dotNL ) * directLightColor_Diffuse;\n\t\t#ifdef DOUBLE_SIDED\n\t\t\tvLightBack += saturate( - dotNL ) * directLightColor_Diffuse;\n\t\t#endif\n\t}\n\t#pragma unroll_loop_end\n#endif\n#if NUM_HEMI_LIGHTS > 0\n\t#pragma unroll_loop_start\n\tfor ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {\n\t\tvIndirectFront += getHemisphereLightIrradiance( hemisphereLights[ i ], geometry.normal );\n\t\t#ifdef DOUBLE_SIDED\n\t\t\tvIndirectBack += getHemisphereLightIrradiance( hemisphereLights[ i ], backGeometry.normal );\n\t\t#endif\n\t}\n\t#pragma unroll_loop_end\n#endif",lights_pars_begin:"uniform bool receiveShadow;\nuniform vec3 ambientLightColor;\nuniform vec3 lightProbe[ 9 ];\nvec3 shGetIrradianceAt( in vec3 normal, in vec3 shCoefficients[ 9 ] ) {\n\tfloat x = normal.x, y = normal.y, z = normal.z;\n\tvec3 result = shCoefficients[ 0 ] * 0.886227;\n\tresult += shCoefficients[ 1 ] * 2.0 * 0.511664 * y;\n\tresult += shCoefficients[ 2 ] * 2.0 * 0.511664 * z;\n\tresult += shCoefficients[ 3 ] * 2.0 * 0.511664 * x;\n\tresult += shCoefficients[ 4 ] * 2.0 * 0.429043 * x * y;\n\tresult += shCoefficients[ 5 ] * 2.0 * 0.429043 * y * z;\n\tresult += shCoefficients[ 6 ] * ( 0.743125 * z * z - 0.247708 );\n\tresult += shCoefficients[ 7 ] * 2.0 * 0.429043 * x * z;\n\tresult += shCoefficients[ 8 ] * 0.429043 * ( x * x - y * y );\n\treturn result;\n}\nvec3 getLightProbeIrradiance( const in vec3 lightProbe[ 9 ], const in vec3 normal ) {\n\tvec3 worldNormal = inverseTransformDirection( normal, viewMatrix );\n\tvec3 irradiance = shGetIrradianceAt( worldNormal, lightProbe );\n\treturn irradiance;\n}\nvec3 getAmbientLightIrradiance( const in vec3 ambientLightColor ) {\n\tvec3 irradiance = ambientLightColor;\n\treturn irradiance;\n}\nfloat getDistanceAttenuation( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {\n\t#if defined ( PHYSICALLY_CORRECT_LIGHTS )\n\t\tfloat distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );\n\t\tif ( cutoffDistance > 0.0 ) {\n\t\t\tdistanceFalloff *= pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );\n\t\t}\n\t\treturn distanceFalloff;\n\t#else\n\t\tif ( cutoffDistance > 0.0 && decayExponent > 0.0 ) {\n\t\t\treturn pow( saturate( - lightDistance / cutoffDistance + 1.0 ), decayExponent );\n\t\t}\n\t\treturn 1.0;\n\t#endif\n}\nfloat getSpotAttenuation( const in float coneCosine, const in float penumbraCosine, const in float angleCosine ) {\n\treturn smoothstep( coneCosine, penumbraCosine, angleCosine );\n}\n#if NUM_DIR_LIGHTS > 0\n\tstruct DirectionalLight {\n\t\tvec3 direction;\n\t\tvec3 color;\n\t};\n\tuniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];\n\tvoid getDirectionalLightInfo( const in DirectionalLight directionalLight, const in GeometricContext geometry, out IncidentLight light ) {\n\t\tlight.color = directionalLight.color;\n\t\tlight.direction = directionalLight.direction;\n\t\tlight.visible = true;\n\t}\n#endif\n#if NUM_POINT_LIGHTS > 0\n\tstruct PointLight {\n\t\tvec3 position;\n\t\tvec3 color;\n\t\tfloat distance;\n\t\tfloat decay;\n\t};\n\tuniform PointLight pointLights[ NUM_POINT_LIGHTS ];\n\tvoid getPointLightInfo( const in PointLight pointLight, const in GeometricContext geometry, out IncidentLight light ) {\n\t\tvec3 lVector = pointLight.position - geometry.position;\n\t\tlight.direction = normalize( lVector );\n\t\tfloat lightDistance = length( lVector );\n\t\tlight.color = pointLight.color;\n\t\tlight.color *= getDistanceAttenuation( lightDistance, pointLight.distance, pointLight.decay );\n\t\tlight.visible = ( light.color != vec3( 0.0 ) );\n\t}\n#endif\n#if NUM_SPOT_LIGHTS > 0\n\tstruct SpotLight {\n\t\tvec3 position;\n\t\tvec3 direction;\n\t\tvec3 color;\n\t\tfloat distance;\n\t\tfloat decay;\n\t\tfloat coneCos;\n\t\tfloat penumbraCos;\n\t};\n\tuniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];\n\tvoid getSpotLightInfo( const in SpotLight spotLight, const in GeometricContext geometry, out IncidentLight light ) {\n\t\tvec3 lVector = spotLight.position - geometry.position;\n\t\tlight.direction = normalize( lVector );\n\t\tfloat angleCos = dot( light.direction, spotLight.direction );\n\t\tfloat spotAttenuation = getSpotAttenuation( spotLight.coneCos, spotLight.penumbraCos, angleCos );\n\t\tif ( spotAttenuation > 0.0 ) {\n\t\t\tfloat lightDistance = length( lVector );\n\t\t\tlight.color = spotLight.color * spotAttenuation;\n\t\t\tlight.color *= getDistanceAttenuation( lightDistance, spotLight.distance, spotLight.decay );\n\t\t\tlight.visible = ( light.color != vec3( 0.0 ) );\n\t\t} else {\n\t\t\tlight.color = vec3( 0.0 );\n\t\t\tlight.visible = false;\n\t\t}\n\t}\n#endif\n#if NUM_RECT_AREA_LIGHTS > 0\n\tstruct RectAreaLight {\n\t\tvec3 color;\n\t\tvec3 position;\n\t\tvec3 halfWidth;\n\t\tvec3 halfHeight;\n\t};\n\tuniform sampler2D ltc_1;\tuniform sampler2D ltc_2;\n\tuniform RectAreaLight rectAreaLights[ NUM_RECT_AREA_LIGHTS ];\n#endif\n#if NUM_HEMI_LIGHTS > 0\n\tstruct HemisphereLight {\n\t\tvec3 direction;\n\t\tvec3 skyColor;\n\t\tvec3 groundColor;\n\t};\n\tuniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];\n\tvec3 getHemisphereLightIrradiance( const in HemisphereLight hemiLight, const in vec3 normal ) {\n\t\tfloat dotNL = dot( normal, hemiLight.direction );\n\t\tfloat hemiDiffuseWeight = 0.5 * dotNL + 0.5;\n\t\tvec3 irradiance = mix( hemiLight.groundColor, hemiLight.skyColor, hemiDiffuseWeight );\n\t\treturn irradiance;\n\t}\n#endif",lights_toon_fragment:"ToonMaterial material;\nmaterial.diffuseColor = diffuseColor.rgb;",lights_toon_pars_fragment:"varying vec3 vViewPosition;\nstruct ToonMaterial {\n\tvec3 diffuseColor;\n};\nvoid RE_Direct_Toon( const in IncidentLight directLight, const in GeometricContext geometry, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {\n\tvec3 irradiance = getGradientIrradiance( geometry.normal, directLight.direction ) * directLight.color;\n\treflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );\n}\nvoid RE_IndirectDiffuse_Toon( const in vec3 irradiance, const in GeometricContext geometry, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {\n\treflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );\n}\n#define RE_Direct\t\t\t\tRE_Direct_Toon\n#define RE_IndirectDiffuse\t\tRE_IndirectDiffuse_Toon\n#define Material_LightProbeLOD( material )\t(0)",lights_phong_fragment:"BlinnPhongMaterial material;\nmaterial.diffuseColor = diffuseColor.rgb;\nmaterial.specularColor = specular;\nmaterial.specularShininess = shininess;\nmaterial.specularStrength = specularStrength;",lights_phong_pars_fragment:"varying vec3 vViewPosition;\nstruct BlinnPhongMaterial {\n\tvec3 diffuseColor;\n\tvec3 specularColor;\n\tfloat specularShininess;\n\tfloat specularStrength;\n};\nvoid RE_Direct_BlinnPhong( const in IncidentLight directLight, const in GeometricContext geometry, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {\n\tfloat dotNL = saturate( dot( geometry.normal, directLight.direction ) );\n\tvec3 irradiance = dotNL * directLight.color;\n\treflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );\n\treflectedLight.directSpecular += irradiance * BRDF_BlinnPhong( directLight.direction, geometry.viewDir, geometry.normal, material.specularColor, material.specularShininess ) * material.specularStrength;\n}\nvoid RE_IndirectDiffuse_BlinnPhong( const in vec3 irradiance, const in GeometricContext geometry, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {\n\treflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );\n}\n#define RE_Direct\t\t\t\tRE_Direct_BlinnPhong\n#define RE_IndirectDiffuse\t\tRE_IndirectDiffuse_BlinnPhong\n#define Material_LightProbeLOD( material )\t(0)",lights_physical_fragment:"PhysicalMaterial material;\nmaterial.diffuseColor = diffuseColor.rgb * ( 1.0 - metalnessFactor );\nvec3 dxy = max( abs( dFdx( geometryNormal ) ), abs( dFdy( geometryNormal ) ) );\nfloat geometryRoughness = max( max( dxy.x, dxy.y ), dxy.z );\nmaterial.roughness = max( roughnessFactor, 0.0525 );material.roughness += geometryRoughness;\nmaterial.roughness = min( material.roughness, 1.0 );\n#ifdef IOR\n\t#ifdef SPECULAR\n\t\tfloat specularIntensityFactor = specularIntensity;\n\t\tvec3 specularColorFactor = specularColor;\n\t\t#ifdef USE_SPECULARINTENSITYMAP\n\t\t\tspecularIntensityFactor *= texture2D( specularIntensityMap, vUv ).a;\n\t\t#endif\n\t\t#ifdef USE_SPECULARCOLORMAP\n\t\t\tspecularColorFactor *= specularColorMapTexelToLinear( texture2D( specularColorMap, vUv ) ).rgb;\n\t\t#endif\n\t\tmaterial.specularF90 = mix( specularIntensityFactor, 1.0, metalnessFactor );\n\t#else\n\t\tfloat specularIntensityFactor = 1.0;\n\t\tvec3 specularColorFactor = vec3( 1.0 );\n\t\tmaterial.specularF90 = 1.0;\n\t#endif\n\tmaterial.specularColor = mix( min( pow2( ( ior - 1.0 ) / ( ior + 1.0 ) ) * specularColorFactor, vec3( 1.0 ) ) * specularIntensityFactor, diffuseColor.rgb, metalnessFactor );\n#else\n\tmaterial.specularColor = mix( vec3( 0.04 ), diffuseColor.rgb, metalnessFactor );\n\tmaterial.specularF90 = 1.0;\n#endif\n#ifdef USE_CLEARCOAT\n\tmaterial.clearcoat = clearcoat;\n\tmaterial.clearcoatRoughness = clearcoatRoughness;\n\tmaterial.clearcoatF0 = vec3( 0.04 );\n\tmaterial.clearcoatF90 = 1.0;\n\t#ifdef USE_CLEARCOATMAP\n\t\tmaterial.clearcoat *= texture2D( clearcoatMap, vUv ).x;\n\t#endif\n\t#ifdef USE_CLEARCOAT_ROUGHNESSMAP\n\t\tmaterial.clearcoatRoughness *= texture2D( clearcoatRoughnessMap, vUv ).y;\n\t#endif\n\tmaterial.clearcoat = saturate( material.clearcoat );\tmaterial.clearcoatRoughness = max( material.clearcoatRoughness, 0.0525 );\n\tmaterial.clearcoatRoughness += geometryRoughness;\n\tmaterial.clearcoatRoughness = min( material.clearcoatRoughness, 1.0 );\n#endif\n#ifdef USE_SHEEN\n\tmaterial.sheenColor = sheenColor;\n\t#ifdef USE_SHEENCOLORMAP\n\t\tmaterial.sheenColor *= sheenColorMapTexelToLinear( texture2D( sheenColorMap, vUv ) ).rgb;\n\t#endif\n\tmaterial.sheenRoughness = clamp( sheenRoughness, 0.07, 1.0 );\n\t#ifdef USE_SHEENROUGHNESSMAP\n\t\tmaterial.sheenRoughness *= texture2D( sheenRoughnessMap, vUv ).a;\n\t#endif\n#endif",lights_physical_pars_fragment:"struct PhysicalMaterial {\n\tvec3 diffuseColor;\n\tfloat roughness;\n\tvec3 specularColor;\n\tfloat specularF90;\n\t#ifdef USE_CLEARCOAT\n\t\tfloat clearcoat;\n\t\tfloat clearcoatRoughness;\n\t\tvec3 clearcoatF0;\n\t\tfloat clearcoatF90;\n\t#endif\n\t#ifdef USE_SHEEN\n\t\tvec3 sheenColor;\n\t\tfloat sheenRoughness;\n\t#endif\n};\nvec3 clearcoatSpecular = vec3( 0.0 );\nvec2 DFGApprox( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {\n\tfloat dotNV = saturate( dot( normal, viewDir ) );\n\tconst vec4 c0 = vec4( - 1, - 0.0275, - 0.572, 0.022 );\n\tconst vec4 c1 = vec4( 1, 0.0425, 1.04, - 0.04 );\n\tvec4 r = roughness * c0 + c1;\n\tfloat a004 = min( r.x * r.x, exp2( - 9.28 * dotNV ) ) * r.x + r.y;\n\tvec2 fab = vec2( - 1.04, 1.04 ) * a004 + r.zw;\n\treturn fab;\n}\nvec3 EnvironmentBRDF( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness ) {\n\tvec2 fab = DFGApprox( normal, viewDir, roughness );\n\treturn specularColor * fab.x + specularF90 * fab.y;\n}\nvoid computeMultiscattering( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {\n\tvec2 fab = DFGApprox( normal, viewDir, roughness );\n\tvec3 FssEss = specularColor * fab.x + specularF90 * fab.y;\n\tfloat Ess = fab.x + fab.y;\n\tfloat Ems = 1.0 - Ess;\n\tvec3 Favg = specularColor + ( 1.0 - specularColor ) * 0.047619;\tvec3 Fms = FssEss * Favg / ( 1.0 - Ems * Favg );\n\tsingleScatter += FssEss;\n\tmultiScatter += Fms * Ems;\n}\n#if NUM_RECT_AREA_LIGHTS > 0\n\tvoid RE_Direct_RectArea_Physical( const in RectAreaLight rectAreaLight, const in GeometricContext geometry, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {\n\t\tvec3 normal = geometry.normal;\n\t\tvec3 viewDir = geometry.viewDir;\n\t\tvec3 position = geometry.position;\n\t\tvec3 lightPos = rectAreaLight.position;\n\t\tvec3 halfWidth = rectAreaLight.halfWidth;\n\t\tvec3 halfHeight = rectAreaLight.halfHeight;\n\t\tvec3 lightColor = rectAreaLight.color;\n\t\tfloat roughness = material.roughness;\n\t\tvec3 rectCoords[ 4 ];\n\t\trectCoords[ 0 ] = lightPos + halfWidth - halfHeight;\t\trectCoords[ 1 ] = lightPos - halfWidth - halfHeight;\n\t\trectCoords[ 2 ] = lightPos - halfWidth + halfHeight;\n\t\trectCoords[ 3 ] = lightPos + halfWidth + halfHeight;\n\t\tvec2 uv = LTC_Uv( normal, viewDir, roughness );\n\t\tvec4 t1 = texture2D( ltc_1, uv );\n\t\tvec4 t2 = texture2D( ltc_2, uv );\n\t\tmat3 mInv = mat3(\n\t\t\tvec3( t1.x, 0, t1.y ),\n\t\t\tvec3(\t\t0, 1,\t\t0 ),\n\t\t\tvec3( t1.z, 0, t1.w )\n\t\t);\n\t\tvec3 fresnel = ( material.specularColor * t2.x + ( vec3( 1.0 ) - material.specularColor ) * t2.y );\n\t\treflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate( normal, viewDir, position, mInv, rectCoords );\n\t\treflectedLight.directDiffuse += lightColor * material.diffuseColor * LTC_Evaluate( normal, viewDir, position, mat3( 1.0 ), rectCoords );\n\t}\n#endif\nvoid RE_Direct_Physical( const in IncidentLight directLight, const in GeometricContext geometry, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {\n\tfloat dotNL = saturate( dot( geometry.normal, directLight.direction ) );\n\tvec3 irradiance = dotNL * directLight.color;\n\t#ifdef USE_CLEARCOAT\n\t\tfloat dotNLcc = saturate( dot( geometry.clearcoatNormal, directLight.direction ) );\n\t\tvec3 ccIrradiance = dotNLcc * directLight.color;\n\t\tclearcoatSpecular += ccIrradiance * BRDF_GGX( directLight.direction, geometry.viewDir, geometry.clearcoatNormal, material.clearcoatF0, material.clearcoatF90, material.clearcoatRoughness );\n\t#endif\n\t#ifdef USE_SHEEN\n\t\treflectedLight.directSpecular += irradiance * BRDF_Sheen( directLight.direction, geometry.viewDir, geometry.normal, material.sheenColor, material.sheenRoughness );\n\t#endif\n\treflectedLight.directSpecular += irradiance * BRDF_GGX( directLight.direction, geometry.viewDir, geometry.normal, material.specularColor, material.specularF90, material.roughness );\n\treflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );\n}\nvoid RE_IndirectDiffuse_Physical( const in vec3 irradiance, const in GeometricContext geometry, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {\n\treflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );\n}\nvoid RE_IndirectSpecular_Physical( const in vec3 radiance, const in vec3 irradiance, const in vec3 clearcoatRadiance, const in GeometricContext geometry, const in PhysicalMaterial material, inout ReflectedLight reflectedLight) {\n\t#ifdef USE_CLEARCOAT\n\t\tclearcoatSpecular += clearcoatRadiance * EnvironmentBRDF( geometry.clearcoatNormal, geometry.viewDir, material.clearcoatF0, material.clearcoatF90, material.clearcoatRoughness );\n\t#endif\n\tvec3 singleScattering = vec3( 0.0 );\n\tvec3 multiScattering = vec3( 0.0 );\n\tvec3 cosineWeightedIrradiance = irradiance * RECIPROCAL_PI;\n\tcomputeMultiscattering( geometry.normal, geometry.viewDir, material.specularColor, material.specularF90, material.roughness, singleScattering, multiScattering );\n\tvec3 diffuse = material.diffuseColor * ( 1.0 - ( singleScattering + multiScattering ) );\n\treflectedLight.indirectSpecular += radiance * singleScattering;\n\treflectedLight.indirectSpecular += multiScattering * cosineWeightedIrradiance;\n\treflectedLight.indirectDiffuse += diffuse * cosineWeightedIrradiance;\n}\n#define RE_Direct\t\t\t\tRE_Direct_Physical\n#define RE_Direct_RectArea\t\tRE_Direct_RectArea_Physical\n#define RE_IndirectDiffuse\t\tRE_IndirectDiffuse_Physical\n#define RE_IndirectSpecular\t\tRE_IndirectSpecular_Physical\nfloat computeSpecularOcclusion( const in float dotNV, const in float ambientOcclusion, const in float roughness ) {\n\treturn saturate( pow( dotNV + ambientOcclusion, exp2( - 16.0 * roughness - 1.0 ) ) - 1.0 + ambientOcclusion );\n}",lights_fragment_begin:"\nGeometricContext geometry;\ngeometry.position = - vViewPosition;\ngeometry.normal = normal;\ngeometry.viewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );\n#ifdef USE_CLEARCOAT\n\tgeometry.clearcoatNormal = clearcoatNormal;\n#endif\nIncidentLight directLight;\n#if ( NUM_POINT_LIGHTS > 0 ) && defined( RE_Direct )\n\tPointLight pointLight;\n\t#if defined( USE_SHADOWMAP ) && NUM_POINT_LIGHT_SHADOWS > 0\n\tPointLightShadow pointLightShadow;\n\t#endif\n\t#pragma unroll_loop_start\n\tfor ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {\n\t\tpointLight = pointLights[ i ];\n\t\tgetPointLightInfo( pointLight, geometry, directLight );\n\t\t#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_POINT_LIGHT_SHADOWS )\n\t\tpointLightShadow = pointLightShadows[ i ];\n\t\tdirectLight.color *= all( bvec2( directLight.visible, receiveShadow ) ) ? getPointShadow( pointShadowMap[ i ], pointLightShadow.shadowMapSize, pointLightShadow.shadowBias, pointLightShadow.shadowRadius, vPointShadowCoord[ i ], pointLightShadow.shadowCameraNear, pointLightShadow.shadowCameraFar ) : 1.0;\n\t\t#endif\n\t\tRE_Direct( directLight, geometry, material, reflectedLight );\n\t}\n\t#pragma unroll_loop_end\n#endif\n#if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )\n\tSpotLight spotLight;\n\t#if defined( USE_SHADOWMAP ) && NUM_SPOT_LIGHT_SHADOWS > 0\n\tSpotLightShadow spotLightShadow;\n\t#endif\n\t#pragma unroll_loop_start\n\tfor ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {\n\t\tspotLight = spotLights[ i ];\n\t\tgetSpotLightInfo( spotLight, geometry, directLight );\n\t\t#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )\n\t\tspotLightShadow = spotLightShadows[ i ];\n\t\tdirectLight.color *= all( bvec2( directLight.visible, receiveShadow ) ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotShadowCoord[ i ] ) : 1.0;\n\t\t#endif\n\t\tRE_Direct( directLight, geometry, material, reflectedLight );\n\t}\n\t#pragma unroll_loop_end\n#endif\n#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )\n\tDirectionalLight directionalLight;\n\t#if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0\n\tDirectionalLightShadow directionalLightShadow;\n\t#endif\n\t#pragma unroll_loop_start\n\tfor ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {\n\t\tdirectionalLight = directionalLights[ i ];\n\t\tgetDirectionalLightInfo( directionalLight, geometry, directLight );\n\t\t#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )\n\t\tdirectionalLightShadow = directionalLightShadows[ i ];\n\t\tdirectLight.color *= all( bvec2( directLight.visible, receiveShadow ) ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;\n\t\t#endif\n\t\tRE_Direct( directLight, geometry, material, reflectedLight );\n\t}\n\t#pragma unroll_loop_end\n#endif\n#if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea )\n\tRectAreaLight rectAreaLight;\n\t#pragma unroll_loop_start\n\tfor ( int i = 0; i < NUM_RECT_AREA_LIGHTS; i ++ ) {\n\t\trectAreaLight = rectAreaLights[ i ];\n\t\tRE_Direct_RectArea( rectAreaLight, geometry, material, reflectedLight );\n\t}\n\t#pragma unroll_loop_end\n#endif\n#if defined( RE_IndirectDiffuse )\n\tvec3 iblIrradiance = vec3( 0.0 );\n\tvec3 irradiance = getAmbientLightIrradiance( ambientLightColor );\n\tirradiance += getLightProbeIrradiance( lightProbe, geometry.normal );\n\t#if ( NUM_HEMI_LIGHTS > 0 )\n\t\t#pragma unroll_loop_start\n\t\tfor ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {\n\t\t\tirradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], geometry.normal );\n\t\t}\n\t\t#pragma unroll_loop_end\n\t#endif\n#endif\n#if defined( RE_IndirectSpecular )\n\tvec3 radiance = vec3( 0.0 );\n\tvec3 clearcoatRadiance = vec3( 0.0 );\n#endif",lights_fragment_maps:"#if defined( RE_IndirectDiffuse )\n\t#ifdef USE_LIGHTMAP\n\t\tvec4 lightMapTexel = texture2D( lightMap, vUv2 );\n\t\tvec3 lightMapIrradiance = lightMapTexelToLinear( lightMapTexel ).rgb * lightMapIntensity;\n\t\t#ifndef PHYSICALLY_CORRECT_LIGHTS\n\t\t\tlightMapIrradiance *= PI;\n\t\t#endif\n\t\tirradiance += lightMapIrradiance;\n\t#endif\n\t#if defined( USE_ENVMAP ) && defined( STANDARD ) && defined( ENVMAP_TYPE_CUBE_UV )\n\t\tiblIrradiance += getIBLIrradiance( geometry.normal );\n\t#endif\n#endif\n#if defined( USE_ENVMAP ) && defined( RE_IndirectSpecular )\n\tradiance += getIBLRadiance( geometry.viewDir, geometry.normal, material.roughness );\n\t#ifdef USE_CLEARCOAT\n\t\tclearcoatRadiance += getIBLRadiance( geometry.viewDir, geometry.clearcoatNormal, material.clearcoatRoughness );\n\t#endif\n#endif",lights_fragment_end:"#if defined( RE_IndirectDiffuse )\n\tRE_IndirectDiffuse( irradiance, geometry, material, reflectedLight );\n#endif\n#if defined( RE_IndirectSpecular )\n\tRE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometry, material, reflectedLight );\n#endif",logdepthbuf_fragment:"#if defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT )\n\tgl_FragDepthEXT = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;\n#endif",logdepthbuf_pars_fragment:"#if defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT )\n\tuniform float logDepthBufFC;\n\tvarying float vFragDepth;\n\tvarying float vIsPerspective;\n#endif",logdepthbuf_pars_vertex:"#ifdef USE_LOGDEPTHBUF\n\t#ifdef USE_LOGDEPTHBUF_EXT\n\t\tvarying float vFragDepth;\n\t\tvarying float vIsPerspective;\n\t#else\n\t\tuniform float logDepthBufFC;\n\t#endif\n#endif",logdepthbuf_vertex:"#ifdef USE_LOGDEPTHBUF\n\t#ifdef USE_LOGDEPTHBUF_EXT\n\t\tvFragDepth = 1.0 + gl_Position.w;\n\t\tvIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );\n\t#else\n\t\tif ( isPerspectiveMatrix( projectionMatrix ) ) {\n\t\t\tgl_Position.z = log2( max( EPSILON, gl_Position.w + 1.0 ) ) * logDepthBufFC - 1.0;\n\t\t\tgl_Position.z *= gl_Position.w;\n\t\t}\n\t#endif\n#endif",map_fragment:"#ifdef USE_MAP\n\tvec4 texelColor = texture2D( map, vUv );\n\ttexelColor = mapTexelToLinear( texelColor );\n\tdiffuseColor *= texelColor;\n#endif",map_pars_fragment:"#ifdef USE_MAP\n\tuniform sampler2D map;\n#endif",map_particle_fragment:"#if defined( USE_MAP ) || defined( USE_ALPHAMAP )\n\tvec2 uv = ( uvTransform * vec3( gl_PointCoord.x, 1.0 - gl_PointCoord.y, 1 ) ).xy;\n#endif\n#ifdef USE_MAP\n\tvec4 mapTexel = texture2D( map, uv );\n\tdiffuseColor *= mapTexelToLinear( mapTexel );\n#endif\n#ifdef USE_ALPHAMAP\n\tdiffuseColor.a *= texture2D( alphaMap, uv ).g;\n#endif",map_particle_pars_fragment:"#if defined( USE_MAP ) || defined( USE_ALPHAMAP )\n\tuniform mat3 uvTransform;\n#endif\n#ifdef USE_MAP\n\tuniform sampler2D map;\n#endif\n#ifdef USE_ALPHAMAP\n\tuniform sampler2D alphaMap;\n#endif",metalnessmap_fragment:"float metalnessFactor = metalness;\n#ifdef USE_METALNESSMAP\n\tvec4 texelMetalness = texture2D( metalnessMap, vUv );\n\tmetalnessFactor *= texelMetalness.b;\n#endif",metalnessmap_pars_fragment:"#ifdef USE_METALNESSMAP\n\tuniform sampler2D metalnessMap;\n#endif",morphnormal_vertex:"#ifdef USE_MORPHNORMALS\n\tobjectNormal *= morphTargetBaseInfluence;\n\t#ifdef MORPHTARGETS_TEXTURE\n\t\tfor ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {\n\t\t\tif ( morphTargetInfluences[ i ] > 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1, 2 ) * morphTargetInfluences[ i ];\n\t\t}\n\t#else\n\t\tobjectNormal += morphNormal0 * morphTargetInfluences[ 0 ];\n\t\tobjectNormal += morphNormal1 * morphTargetInfluences[ 1 ];\n\t\tobjectNormal += morphNormal2 * morphTargetInfluences[ 2 ];\n\t\tobjectNormal += morphNormal3 * morphTargetInfluences[ 3 ];\n\t#endif\n#endif",morphtarget_pars_vertex:"#ifdef USE_MORPHTARGETS\n\tuniform float morphTargetBaseInfluence;\n\t#ifdef MORPHTARGETS_TEXTURE\n\t\tuniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];\n\t\tuniform sampler2DArray morphTargetsTexture;\n\t\tuniform vec2 morphTargetsTextureSize;\n\t\tvec3 getMorph( const in int vertexIndex, const in int morphTargetIndex, const in int offset, const in int stride ) {\n\t\t\tfloat texelIndex = float( vertexIndex * stride + offset );\n\t\t\tfloat y = floor( texelIndex / morphTargetsTextureSize.x );\n\t\t\tfloat x = texelIndex - y * morphTargetsTextureSize.x;\n\t\t\tvec3 morphUV = vec3( ( x + 0.5 ) / morphTargetsTextureSize.x, y / morphTargetsTextureSize.y, morphTargetIndex );\n\t\t\treturn texture( morphTargetsTexture, morphUV ).xyz;\n\t\t}\n\t#else\n\t\t#ifndef USE_MORPHNORMALS\n\t\t\tuniform float morphTargetInfluences[ 8 ];\n\t\t#else\n\t\t\tuniform float morphTargetInfluences[ 4 ];\n\t\t#endif\n\t#endif\n#endif",morphtarget_vertex:"#ifdef USE_MORPHTARGETS\n\ttransformed *= morphTargetBaseInfluence;\n\t#ifdef MORPHTARGETS_TEXTURE\n\t\tfor ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {\n\t\t\t#ifndef USE_MORPHNORMALS\n\t\t\t\tif ( morphTargetInfluences[ i ] > 0.0 ) transformed += getMorph( gl_VertexID, i, 0, 1 ) * morphTargetInfluences[ i ];\n\t\t\t#else\n\t\t\t\tif ( morphTargetInfluences[ i ] > 0.0 ) transformed += getMorph( gl_VertexID, i, 0, 2 ) * morphTargetInfluences[ i ];\n\t\t\t#endif\n\t\t}\n\t#else\n\t\ttransformed += morphTarget0 * morphTargetInfluences[ 0 ];\n\t\ttransformed += morphTarget1 * morphTargetInfluences[ 1 ];\n\t\ttransformed += morphTarget2 * morphTargetInfluences[ 2 ];\n\t\ttransformed += morphTarget3 * morphTargetInfluences[ 3 ];\n\t\t#ifndef USE_MORPHNORMALS\n\t\t\ttransformed += morphTarget4 * morphTargetInfluences[ 4 ];\n\t\t\ttransformed += morphTarget5 * morphTargetInfluences[ 5 ];\n\t\t\ttransformed += morphTarget6 * morphTargetInfluences[ 6 ];\n\t\t\ttransformed += morphTarget7 * morphTargetInfluences[ 7 ];\n\t\t#endif\n\t#endif\n#endif",normal_fragment_begin:"float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;\n#ifdef FLAT_SHADED\n\tvec3 fdx = vec3( dFdx( vViewPosition.x ), dFdx( vViewPosition.y ), dFdx( vViewPosition.z ) );\n\tvec3 fdy = vec3( dFdy( vViewPosition.x ), dFdy( vViewPosition.y ), dFdy( vViewPosition.z ) );\n\tvec3 normal = normalize( cross( fdx, fdy ) );\n#else\n\tvec3 normal = normalize( vNormal );\n\t#ifdef DOUBLE_SIDED\n\t\tnormal = normal * faceDirection;\n\t#endif\n\t#ifdef USE_TANGENT\n\t\tvec3 tangent = normalize( vTangent );\n\t\tvec3 bitangent = normalize( vBitangent );\n\t\t#ifdef DOUBLE_SIDED\n\t\t\ttangent = tangent * faceDirection;\n\t\t\tbitangent = bitangent * faceDirection;\n\t\t#endif\n\t\t#if defined( TANGENTSPACE_NORMALMAP ) || defined( USE_CLEARCOAT_NORMALMAP )\n\t\t\tmat3 vTBN = mat3( tangent, bitangent, normal );\n\t\t#endif\n\t#endif\n#endif\nvec3 geometryNormal = normal;",normal_fragment_maps:"#ifdef OBJECTSPACE_NORMALMAP\n\tnormal = texture2D( normalMap, vUv ).xyz * 2.0 - 1.0;\n\t#ifdef FLIP_SIDED\n\t\tnormal = - normal;\n\t#endif\n\t#ifdef DOUBLE_SIDED\n\t\tnormal = normal * faceDirection;\n\t#endif\n\tnormal = normalize( normalMatrix * normal );\n#elif defined( TANGENTSPACE_NORMALMAP )\n\tvec3 mapN = texture2D( normalMap, vUv ).xyz * 2.0 - 1.0;\n\tmapN.xy *= normalScale;\n\t#ifdef USE_TANGENT\n\t\tnormal = normalize( vTBN * mapN );\n\t#else\n\t\tnormal = perturbNormal2Arb( - vViewPosition, normal, mapN, faceDirection );\n\t#endif\n#elif defined( USE_BUMPMAP )\n\tnormal = perturbNormalArb( - vViewPosition, normal, dHdxy_fwd(), faceDirection );\n#endif",normal_pars_fragment:"#ifndef FLAT_SHADED\n\tvarying vec3 vNormal;\n\t#ifdef USE_TANGENT\n\t\tvarying vec3 vTangent;\n\t\tvarying vec3 vBitangent;\n\t#endif\n#endif",normal_pars_vertex:"#ifndef FLAT_SHADED\n\tvarying vec3 vNormal;\n\t#ifdef USE_TANGENT\n\t\tvarying vec3 vTangent;\n\t\tvarying vec3 vBitangent;\n\t#endif\n#endif",normal_vertex:"#ifndef FLAT_SHADED\n\tvNormal = normalize( transformedNormal );\n\t#ifdef USE_TANGENT\n\t\tvTangent = normalize( transformedTangent );\n\t\tvBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );\n\t#endif\n#endif",normalmap_pars_fragment:"#ifdef USE_NORMALMAP\n\tuniform sampler2D normalMap;\n\tuniform vec2 normalScale;\n#endif\n#ifdef OBJECTSPACE_NORMALMAP\n\tuniform mat3 normalMatrix;\n#endif\n#if ! defined ( USE_TANGENT ) && ( defined ( TANGENTSPACE_NORMALMAP ) || defined ( USE_CLEARCOAT_NORMALMAP ) )\n\tvec3 perturbNormal2Arb( vec3 eye_pos, vec3 surf_norm, vec3 mapN, float faceDirection ) {\n\t\tvec3 q0 = vec3( dFdx( eye_pos.x ), dFdx( eye_pos.y ), dFdx( eye_pos.z ) );\n\t\tvec3 q1 = vec3( dFdy( eye_pos.x ), dFdy( eye_pos.y ), dFdy( eye_pos.z ) );\n\t\tvec2 st0 = dFdx( vUv.st );\n\t\tvec2 st1 = dFdy( vUv.st );\n\t\tvec3 N = surf_norm;\n\t\tvec3 q1perp = cross( q1, N );\n\t\tvec3 q0perp = cross( N, q0 );\n\t\tvec3 T = q1perp * st0.x + q0perp * st1.x;\n\t\tvec3 B = q1perp * st0.y + q0perp * st1.y;\n\t\tfloat det = max( dot( T, T ), dot( B, B ) );\n\t\tfloat scale = ( det == 0.0 ) ? 0.0 : faceDirection * inversesqrt( det );\n\t\treturn normalize( T * ( mapN.x * scale ) + B * ( mapN.y * scale ) + N * mapN.z );\n\t}\n#endif",clearcoat_normal_fragment_begin:"#ifdef USE_CLEARCOAT\n\tvec3 clearcoatNormal = geometryNormal;\n#endif",clearcoat_normal_fragment_maps:"#ifdef USE_CLEARCOAT_NORMALMAP\n\tvec3 clearcoatMapN = texture2D( clearcoatNormalMap, vUv ).xyz * 2.0 - 1.0;\n\tclearcoatMapN.xy *= clearcoatNormalScale;\n\t#ifdef USE_TANGENT\n\t\tclearcoatNormal = normalize( vTBN * clearcoatMapN );\n\t#else\n\t\tclearcoatNormal = perturbNormal2Arb( - vViewPosition, clearcoatNormal, clearcoatMapN, faceDirection );\n\t#endif\n#endif",clearcoat_pars_fragment:"#ifdef USE_CLEARCOATMAP\n\tuniform sampler2D clearcoatMap;\n#endif\n#ifdef USE_CLEARCOAT_ROUGHNESSMAP\n\tuniform sampler2D clearcoatRoughnessMap;\n#endif\n#ifdef USE_CLEARCOAT_NORMALMAP\n\tuniform sampler2D clearcoatNormalMap;\n\tuniform vec2 clearcoatNormalScale;\n#endif",output_fragment:"#ifdef OPAQUE\ndiffuseColor.a = 1.0;\n#endif\n#ifdef USE_TRANSMISSION\ndiffuseColor.a *= transmissionAlpha + 0.1;\n#endif\ngl_FragColor = vec4( outgoingLight, diffuseColor.a );",packing:"vec3 packNormalToRGB( const in vec3 normal ) {\n\treturn normalize( normal ) * 0.5 + 0.5;\n}\nvec3 unpackRGBToNormal( const in vec3 rgb ) {\n\treturn 2.0 * rgb.xyz - 1.0;\n}\nconst float PackUpscale = 256. / 255.;const float UnpackDownscale = 255. / 256.;\nconst vec3 PackFactors = vec3( 256. * 256. * 256., 256. * 256., 256. );\nconst vec4 UnpackFactors = UnpackDownscale / vec4( PackFactors, 1. );\nconst float ShiftRight8 = 1. / 256.;\nvec4 packDepthToRGBA( const in float v ) {\n\tvec4 r = vec4( fract( v * PackFactors ), v );\n\tr.yzw -= r.xyz * ShiftRight8;\treturn r * PackUpscale;\n}\nfloat unpackRGBAToDepth( const in vec4 v ) {\n\treturn dot( v, UnpackFactors );\n}\nvec4 pack2HalfToRGBA( vec2 v ) {\n\tvec4 r = vec4( v.x, fract( v.x * 255.0 ), v.y, fract( v.y * 255.0 ) );\n\treturn vec4( r.x - r.y / 255.0, r.y, r.z - r.w / 255.0, r.w );\n}\nvec2 unpackRGBATo2Half( vec4 v ) {\n\treturn vec2( v.x + ( v.y / 255.0 ), v.z + ( v.w / 255.0 ) );\n}\nfloat viewZToOrthographicDepth( const in float viewZ, const in float near, const in float far ) {\n\treturn ( viewZ + near ) / ( near - far );\n}\nfloat orthographicDepthToViewZ( const in float linearClipZ, const in float near, const in float far ) {\n\treturn linearClipZ * ( near - far ) - near;\n}\nfloat viewZToPerspectiveDepth( const in float viewZ, const in float near, const in float far ) {\n\treturn ( ( near + viewZ ) * far ) / ( ( far - near ) * viewZ );\n}\nfloat perspectiveDepthToViewZ( const in float invClipZ, const in float near, const in float far ) {\n\treturn ( near * far ) / ( ( far - near ) * invClipZ - far );\n}",premultiplied_alpha_fragment:"#ifdef PREMULTIPLIED_ALPHA\n\tgl_FragColor.rgb *= gl_FragColor.a;\n#endif",project_vertex:"vec4 mvPosition = vec4( transformed, 1.0 );\n#ifdef USE_INSTANCING\n\tmvPosition = instanceMatrix * mvPosition;\n#endif\nmvPosition = modelViewMatrix * mvPosition;\ngl_Position = projectionMatrix * mvPosition;",dithering_fragment:"#ifdef DITHERING\n\tgl_FragColor.rgb = dithering( gl_FragColor.rgb );\n#endif",dithering_pars_fragment:"#ifdef DITHERING\n\tvec3 dithering( vec3 color ) {\n\t\tfloat grid_position = rand( gl_FragCoord.xy );\n\t\tvec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );\n\t\tdither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );\n\t\treturn color + dither_shift_RGB;\n\t}\n#endif",roughnessmap_fragment:"float roughnessFactor = roughness;\n#ifdef USE_ROUGHNESSMAP\n\tvec4 texelRoughness = texture2D( roughnessMap, vUv );\n\troughnessFactor *= texelRoughness.g;\n#endif",roughnessmap_pars_fragment:"#ifdef USE_ROUGHNESSMAP\n\tuniform sampler2D roughnessMap;\n#endif",shadowmap_pars_fragment:"#ifdef USE_SHADOWMAP\n\t#if NUM_DIR_LIGHT_SHADOWS > 0\n\t\tuniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];\n\t\tvarying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];\n\t\tstruct DirectionalLightShadow {\n\t\t\tfloat shadowBias;\n\t\t\tfloat shadowNormalBias;\n\t\t\tfloat shadowRadius;\n\t\t\tvec2 shadowMapSize;\n\t\t};\n\t\tuniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];\n\t#endif\n\t#if NUM_SPOT_LIGHT_SHADOWS > 0\n\t\tuniform sampler2D spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];\n\t\tvarying vec4 vSpotShadowCoord[ NUM_SPOT_LIGHT_SHADOWS ];\n\t\tstruct SpotLightShadow {\n\t\t\tfloat shadowBias;\n\t\t\tfloat shadowNormalBias;\n\t\t\tfloat shadowRadius;\n\t\t\tvec2 shadowMapSize;\n\t\t};\n\t\tuniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];\n\t#endif\n\t#if NUM_POINT_LIGHT_SHADOWS > 0\n\t\tuniform sampler2D pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];\n\t\tvarying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];\n\t\tstruct PointLightShadow {\n\t\t\tfloat shadowBias;\n\t\t\tfloat shadowNormalBias;\n\t\t\tfloat shadowRadius;\n\t\t\tvec2 shadowMapSize;\n\t\t\tfloat shadowCameraNear;\n\t\t\tfloat shadowCameraFar;\n\t\t};\n\t\tuniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];\n\t#endif\n\tfloat texture2DCompare( sampler2D depths, vec2 uv, float compare ) {\n\t\treturn step( compare, unpackRGBAToDepth( texture2D( depths, uv ) ) );\n\t}\n\tvec2 texture2DDistribution( sampler2D shadow, vec2 uv ) {\n\t\treturn unpackRGBATo2Half( texture2D( shadow, uv ) );\n\t}\n\tfloat VSMShadow (sampler2D shadow, vec2 uv, float compare ){\n\t\tfloat occlusion = 1.0;\n\t\tvec2 distribution = texture2DDistribution( shadow, uv );\n\t\tfloat hard_shadow = step( compare , distribution.x );\n\t\tif (hard_shadow != 1.0 ) {\n\t\t\tfloat distance = compare - distribution.x ;\n\t\t\tfloat variance = max( 0.00000, distribution.y * distribution.y );\n\t\t\tfloat softness_probability = variance / (variance + distance * distance );\t\t\tsoftness_probability = clamp( ( softness_probability - 0.3 ) / ( 0.95 - 0.3 ), 0.0, 1.0 );\t\t\tocclusion = clamp( max( hard_shadow, softness_probability ), 0.0, 1.0 );\n\t\t}\n\t\treturn occlusion;\n\t}\n\tfloat getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowBias, float shadowRadius, vec4 shadowCoord ) {\n\t\tfloat shadow = 1.0;\n\t\tshadowCoord.xyz /= shadowCoord.w;\n\t\tshadowCoord.z += shadowBias;\n\t\tbvec4 inFrustumVec = bvec4 ( shadowCoord.x >= 0.0, shadowCoord.x <= 1.0, shadowCoord.y >= 0.0, shadowCoord.y <= 1.0 );\n\t\tbool inFrustum = all( inFrustumVec );\n\t\tbvec2 frustumTestVec = bvec2( inFrustum, shadowCoord.z <= 1.0 );\n\t\tbool frustumTest = all( frustumTestVec );\n\t\tif ( frustumTest ) {\n\t\t#if defined( SHADOWMAP_TYPE_PCF )\n\t\t\tvec2 texelSize = vec2( 1.0 ) / shadowMapSize;\n\t\t\tfloat dx0 = - texelSize.x * shadowRadius;\n\t\t\tfloat dy0 = - texelSize.y * shadowRadius;\n\t\t\tfloat dx1 = + texelSize.x * shadowRadius;\n\t\t\tfloat dy1 = + texelSize.y * shadowRadius;\n\t\t\tfloat dx2 = dx0 / 2.0;\n\t\t\tfloat dy2 = dy0 / 2.0;\n\t\t\tfloat dx3 = dx1 / 2.0;\n\t\t\tfloat dy3 = dy1 / 2.0;\n\t\t\tshadow = (\n\t\t\t\ttexture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy0 ), shadowCoord.z ) +\n\t\t\t\ttexture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy0 ), shadowCoord.z ) +\n\t\t\t\ttexture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy0 ), shadowCoord.z ) +\n\t\t\t\ttexture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, dy2 ), shadowCoord.z ) +\n\t\t\t\ttexture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy2 ), shadowCoord.z ) +\n\t\t\t\ttexture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, dy2 ), shadowCoord.z ) +\n\t\t\t\ttexture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, 0.0 ), shadowCoord.z ) +\n\t\t\t\ttexture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, 0.0 ), shadowCoord.z ) +\n\t\t\t\ttexture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z ) +\n\t\t\t\ttexture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, 0.0 ), shadowCoord.z ) +\n\t\t\t\ttexture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, 0.0 ), shadowCoord.z ) +\n\t\t\t\ttexture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, dy3 ), shadowCoord.z ) +\n\t\t\t\ttexture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy3 ), shadowCoord.z ) +\n\t\t\t\ttexture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, dy3 ), shadowCoord.z ) +\n\t\t\t\ttexture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy1 ), shadowCoord.z ) +\n\t\t\t\ttexture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy1 ), shadowCoord.z ) +\n\t\t\t\ttexture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy1 ), shadowCoord.z )\n\t\t\t) * ( 1.0 / 17.0 );\n\t\t#elif defined( SHADOWMAP_TYPE_PCF_SOFT )\n\t\t\tvec2 texelSize = vec2( 1.0 ) / shadowMapSize;\n\t\t\tfloat dx = texelSize.x;\n\t\t\tfloat dy = texelSize.y;\n\t\t\tvec2 uv = shadowCoord.xy;\n\t\t\tvec2 f = fract( uv * shadowMapSize + 0.5 );\n\t\t\tuv -= f * texelSize;\n\t\t\tshadow = (\n\t\t\t\ttexture2DCompare( shadowMap, uv, shadowCoord.z ) +\n\t\t\t\ttexture2DCompare( shadowMap, uv + vec2( dx, 0.0 ), shadowCoord.z ) +\n\t\t\t\ttexture2DCompare( shadowMap, uv + vec2( 0.0, dy ), shadowCoord.z ) +\n\t\t\t\ttexture2DCompare( shadowMap, uv + texelSize, shadowCoord.z ) +\n\t\t\t\tmix( texture2DCompare( shadowMap, uv + vec2( -dx, 0.0 ), shadowCoord.z ), \n\t\t\t\t\t texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, 0.0 ), shadowCoord.z ),\n\t\t\t\t\t f.x ) +\n\t\t\t\tmix( texture2DCompare( shadowMap, uv + vec2( -dx, dy ), shadowCoord.z ), \n\t\t\t\t\t texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, dy ), shadowCoord.z ),\n\t\t\t\t\t f.x ) +\n\t\t\t\tmix( texture2DCompare( shadowMap, uv + vec2( 0.0, -dy ), shadowCoord.z ), \n\t\t\t\t\t texture2DCompare( shadowMap, uv + vec2( 0.0, 2.0 * dy ), shadowCoord.z ),\n\t\t\t\t\t f.y ) +\n\t\t\t\tmix( texture2DCompare( shadowMap, uv + vec2( dx, -dy ), shadowCoord.z ), \n\t\t\t\t\t texture2DCompare( shadowMap, uv + vec2( dx, 2.0 * dy ), shadowCoord.z ),\n\t\t\t\t\t f.y ) +\n\t\t\t\tmix( mix( texture2DCompare( shadowMap, uv + vec2( -dx, -dy ), shadowCoord.z ), \n\t\t\t\t\t\t\ttexture2DCompare( shadowMap, uv + vec2( 2.0 * dx, -dy ), shadowCoord.z ),\n\t\t\t\t\t\t\tf.x ),\n\t\t\t\t\t mix( texture2DCompare( shadowMap, uv + vec2( -dx, 2.0 * dy ), shadowCoord.z ), \n\t\t\t\t\t\t\ttexture2DCompare( shadowMap, uv + vec2( 2.0 * dx, 2.0 * dy ), shadowCoord.z ),\n\t\t\t\t\t\t\tf.x ),\n\t\t\t\t\t f.y )\n\t\t\t) * ( 1.0 / 9.0 );\n\t\t#elif defined( SHADOWMAP_TYPE_VSM )\n\t\t\tshadow = VSMShadow( shadowMap, shadowCoord.xy, shadowCoord.z );\n\t\t#else\n\t\t\tshadow = texture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z );\n\t\t#endif\n\t\t}\n\t\treturn shadow;\n\t}\n\tvec2 cubeToUV( vec3 v, float texelSizeY ) {\n\t\tvec3 absV = abs( v );\n\t\tfloat scaleToCube = 1.0 / max( absV.x, max( absV.y, absV.z ) );\n\t\tabsV *= scaleToCube;\n\t\tv *= scaleToCube * ( 1.0 - 2.0 * texelSizeY );\n\t\tvec2 planar = v.xy;\n\t\tfloat almostATexel = 1.5 * texelSizeY;\n\t\tfloat almostOne = 1.0 - almostATexel;\n\t\tif ( absV.z >= almostOne ) {\n\t\t\tif ( v.z > 0.0 )\n\t\t\t\tplanar.x = 4.0 - v.x;\n\t\t} else if ( absV.x >= almostOne ) {\n\t\t\tfloat signX = sign( v.x );\n\t\t\tplanar.x = v.z * signX + 2.0 * signX;\n\t\t} else if ( absV.y >= almostOne ) {\n\t\t\tfloat signY = sign( v.y );\n\t\t\tplanar.x = v.x + 2.0 * signY + 2.0;\n\t\t\tplanar.y = v.z * signY - 2.0;\n\t\t}\n\t\treturn vec2( 0.125, 0.25 ) * planar + vec2( 0.375, 0.75 );\n\t}\n\tfloat getPointShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {\n\t\tvec2 texelSize = vec2( 1.0 ) / ( shadowMapSize * vec2( 4.0, 2.0 ) );\n\t\tvec3 lightToPosition = shadowCoord.xyz;\n\t\tfloat dp = ( length( lightToPosition ) - shadowCameraNear ) / ( shadowCameraFar - shadowCameraNear );\t\tdp += shadowBias;\n\t\tvec3 bd3D = normalize( lightToPosition );\n\t\t#if defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_PCF_SOFT ) || defined( SHADOWMAP_TYPE_VSM )\n\t\t\tvec2 offset = vec2( - 1, 1 ) * shadowRadius * texelSize.y;\n\t\t\treturn (\n\t\t\t\ttexture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyy, texelSize.y ), dp ) +\n\t\t\t\ttexture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyy, texelSize.y ), dp ) +\n\t\t\t\ttexture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyx, texelSize.y ), dp ) +\n\t\t\t\ttexture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyx, texelSize.y ), dp ) +\n\t\t\t\ttexture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp ) +\n\t\t\t\ttexture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxy, texelSize.y ), dp ) +\n\t\t\t\ttexture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxy, texelSize.y ), dp ) +\n\t\t\t\ttexture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxx, texelSize.y ), dp ) +\n\t\t\t\ttexture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxx, texelSize.y ), dp )\n\t\t\t) * ( 1.0 / 9.0 );\n\t\t#else\n\t\t\treturn texture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp );\n\t\t#endif\n\t}\n#endif",shadowmap_pars_vertex:"#ifdef USE_SHADOWMAP\n\t#if NUM_DIR_LIGHT_SHADOWS > 0\n\t\tuniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHT_SHADOWS ];\n\t\tvarying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];\n\t\tstruct DirectionalLightShadow {\n\t\t\tfloat shadowBias;\n\t\t\tfloat shadowNormalBias;\n\t\t\tfloat shadowRadius;\n\t\t\tvec2 shadowMapSize;\n\t\t};\n\t\tuniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];\n\t#endif\n\t#if NUM_SPOT_LIGHT_SHADOWS > 0\n\t\tuniform mat4 spotShadowMatrix[ NUM_SPOT_LIGHT_SHADOWS ];\n\t\tvarying vec4 vSpotShadowCoord[ NUM_SPOT_LIGHT_SHADOWS ];\n\t\tstruct SpotLightShadow {\n\t\t\tfloat shadowBias;\n\t\t\tfloat shadowNormalBias;\n\t\t\tfloat shadowRadius;\n\t\t\tvec2 shadowMapSize;\n\t\t};\n\t\tuniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];\n\t#endif\n\t#if NUM_POINT_LIGHT_SHADOWS > 0\n\t\tuniform mat4 pointShadowMatrix[ NUM_POINT_LIGHT_SHADOWS ];\n\t\tvarying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];\n\t\tstruct PointLightShadow {\n\t\t\tfloat shadowBias;\n\t\t\tfloat shadowNormalBias;\n\t\t\tfloat shadowRadius;\n\t\t\tvec2 shadowMapSize;\n\t\t\tfloat shadowCameraNear;\n\t\t\tfloat shadowCameraFar;\n\t\t};\n\t\tuniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];\n\t#endif\n#endif",shadowmap_vertex:"#ifdef USE_SHADOWMAP\n\t#if NUM_DIR_LIGHT_SHADOWS > 0 || NUM_SPOT_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0\n\t\tvec3 shadowWorldNormal = inverseTransformDirection( transformedNormal, viewMatrix );\n\t\tvec4 shadowWorldPosition;\n\t#endif\n\t#if NUM_DIR_LIGHT_SHADOWS > 0\n\t#pragma unroll_loop_start\n\tfor ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {\n\t\tshadowWorldPosition = worldPosition + vec4( shadowWorldNormal * directionalLightShadows[ i ].shadowNormalBias, 0 );\n\t\tvDirectionalShadowCoord[ i ] = directionalShadowMatrix[ i ] * shadowWorldPosition;\n\t}\n\t#pragma unroll_loop_end\n\t#endif\n\t#if NUM_SPOT_LIGHT_SHADOWS > 0\n\t#pragma unroll_loop_start\n\tfor ( int i = 0; i < NUM_SPOT_LIGHT_SHADOWS; i ++ ) {\n\t\tshadowWorldPosition = worldPosition + vec4( shadowWorldNormal * spotLightShadows[ i ].shadowNormalBias, 0 );\n\t\tvSpotShadowCoord[ i ] = spotShadowMatrix[ i ] * shadowWorldPosition;\n\t}\n\t#pragma unroll_loop_end\n\t#endif\n\t#if NUM_POINT_LIGHT_SHADOWS > 0\n\t#pragma unroll_loop_start\n\tfor ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {\n\t\tshadowWorldPosition = worldPosition + vec4( shadowWorldNormal * pointLightShadows[ i ].shadowNormalBias, 0 );\n\t\tvPointShadowCoord[ i ] = pointShadowMatrix[ i ] * shadowWorldPosition;\n\t}\n\t#pragma unroll_loop_end\n\t#endif\n#endif",shadowmask_pars_fragment:"float getShadowMask() {\n\tfloat shadow = 1.0;\n\t#ifdef USE_SHADOWMAP\n\t#if NUM_DIR_LIGHT_SHADOWS > 0\n\tDirectionalLightShadow directionalLight;\n\t#pragma unroll_loop_start\n\tfor ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {\n\t\tdirectionalLight = directionalLightShadows[ i ];\n\t\tshadow *= receiveShadow ? getShadow( directionalShadowMap[ i ], directionalLight.shadowMapSize, directionalLight.shadowBias, directionalLight.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;\n\t}\n\t#pragma unroll_loop_end\n\t#endif\n\t#if NUM_SPOT_LIGHT_SHADOWS > 0\n\tSpotLightShadow spotLight;\n\t#pragma unroll_loop_start\n\tfor ( int i = 0; i < NUM_SPOT_LIGHT_SHADOWS; i ++ ) {\n\t\tspotLight = spotLightShadows[ i ];\n\t\tshadow *= receiveShadow ? getShadow( spotShadowMap[ i ], spotLight.shadowMapSize, spotLight.shadowBias, spotLight.shadowRadius, vSpotShadowCoord[ i ] ) : 1.0;\n\t}\n\t#pragma unroll_loop_end\n\t#endif\n\t#if NUM_POINT_LIGHT_SHADOWS > 0\n\tPointLightShadow pointLight;\n\t#pragma unroll_loop_start\n\tfor ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {\n\t\tpointLight = pointLightShadows[ i ];\n\t\tshadow *= receiveShadow ? getPointShadow( pointShadowMap[ i ], pointLight.shadowMapSize, pointLight.shadowBias, pointLight.shadowRadius, vPointShadowCoord[ i ], pointLight.shadowCameraNear, pointLight.shadowCameraFar ) : 1.0;\n\t}\n\t#pragma unroll_loop_end\n\t#endif\n\t#endif\n\treturn shadow;\n}",skinbase_vertex:"#ifdef USE_SKINNING\n\tmat4 boneMatX = getBoneMatrix( skinIndex.x );\n\tmat4 boneMatY = getBoneMatrix( skinIndex.y );\n\tmat4 boneMatZ = getBoneMatrix( skinIndex.z );\n\tmat4 boneMatW = getBoneMatrix( skinIndex.w );\n#endif",skinning_pars_vertex:"#ifdef USE_SKINNING\n\tuniform mat4 bindMatrix;\n\tuniform mat4 bindMatrixInverse;\n\t#ifdef BONE_TEXTURE\n\t\tuniform highp sampler2D boneTexture;\n\t\tuniform int boneTextureSize;\n\t\tmat4 getBoneMatrix( const in float i ) {\n\t\t\tfloat j = i * 4.0;\n\t\t\tfloat x = mod( j, float( boneTextureSize ) );\n\t\t\tfloat y = floor( j / float( boneTextureSize ) );\n\t\t\tfloat dx = 1.0 / float( boneTextureSize );\n\t\t\tfloat dy = 1.0 / float( boneTextureSize );\n\t\t\ty = dy * ( y + 0.5 );\n\t\t\tvec4 v1 = texture2D( boneTexture, vec2( dx * ( x + 0.5 ), y ) );\n\t\t\tvec4 v2 = texture2D( boneTexture, vec2( dx * ( x + 1.5 ), y ) );\n\t\t\tvec4 v3 = texture2D( boneTexture, vec2( dx * ( x + 2.5 ), y ) );\n\t\t\tvec4 v4 = texture2D( boneTexture, vec2( dx * ( x + 3.5 ), y ) );\n\t\t\tmat4 bone = mat4( v1, v2, v3, v4 );\n\t\t\treturn bone;\n\t\t}\n\t#else\n\t\tuniform mat4 boneMatrices[ MAX_BONES ];\n\t\tmat4 getBoneMatrix( const in float i ) {\n\t\t\tmat4 bone = boneMatrices[ int(i) ];\n\t\t\treturn bone;\n\t\t}\n\t#endif\n#endif",skinning_vertex:"#ifdef USE_SKINNING\n\tvec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );\n\tvec4 skinned = vec4( 0.0 );\n\tskinned += boneMatX * skinVertex * skinWeight.x;\n\tskinned += boneMatY * skinVertex * skinWeight.y;\n\tskinned += boneMatZ * skinVertex * skinWeight.z;\n\tskinned += boneMatW * skinVertex * skinWeight.w;\n\ttransformed = ( bindMatrixInverse * skinned ).xyz;\n#endif",skinnormal_vertex:"#ifdef USE_SKINNING\n\tmat4 skinMatrix = mat4( 0.0 );\n\tskinMatrix += skinWeight.x * boneMatX;\n\tskinMatrix += skinWeight.y * boneMatY;\n\tskinMatrix += skinWeight.z * boneMatZ;\n\tskinMatrix += skinWeight.w * boneMatW;\n\tskinMatrix = bindMatrixInverse * skinMatrix * bindMatrix;\n\tobjectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;\n\t#ifdef USE_TANGENT\n\t\tobjectTangent = vec4( skinMatrix * vec4( objectTangent, 0.0 ) ).xyz;\n\t#endif\n#endif",specularmap_fragment:"float specularStrength;\n#ifdef USE_SPECULARMAP\n\tvec4 texelSpecular = texture2D( specularMap, vUv );\n\tspecularStrength = texelSpecular.r;\n#else\n\tspecularStrength = 1.0;\n#endif",specularmap_pars_fragment:"#ifdef USE_SPECULARMAP\n\tuniform sampler2D specularMap;\n#endif",tonemapping_fragment:"#if defined( TONE_MAPPING )\n\tgl_FragColor.rgb = toneMapping( gl_FragColor.rgb );\n#endif",tonemapping_pars_fragment:"#ifndef saturate\n#define saturate( a ) clamp( a, 0.0, 1.0 )\n#endif\nuniform float toneMappingExposure;\nvec3 LinearToneMapping( vec3 color ) {\n\treturn toneMappingExposure * color;\n}\nvec3 ReinhardToneMapping( vec3 color ) {\n\tcolor *= toneMappingExposure;\n\treturn saturate( color / ( vec3( 1.0 ) + color ) );\n}\nvec3 OptimizedCineonToneMapping( vec3 color ) {\n\tcolor *= toneMappingExposure;\n\tcolor = max( vec3( 0.0 ), color - 0.004 );\n\treturn pow( ( color * ( 6.2 * color + 0.5 ) ) / ( color * ( 6.2 * color + 1.7 ) + 0.06 ), vec3( 2.2 ) );\n}\nvec3 RRTAndODTFit( vec3 v ) {\n\tvec3 a = v * ( v + 0.0245786 ) - 0.000090537;\n\tvec3 b = v * ( 0.983729 * v + 0.4329510 ) + 0.238081;\n\treturn a / b;\n}\nvec3 ACESFilmicToneMapping( vec3 color ) {\n\tconst mat3 ACESInputMat = mat3(\n\t\tvec3( 0.59719, 0.07600, 0.02840 ),\t\tvec3( 0.35458, 0.90834, 0.13383 ),\n\t\tvec3( 0.04823, 0.01566, 0.83777 )\n\t);\n\tconst mat3 ACESOutputMat = mat3(\n\t\tvec3(\t1.60475, -0.10208, -0.00327 ),\t\tvec3( -0.53108,\t1.10813, -0.07276 ),\n\t\tvec3( -0.07367, -0.00605,\t1.07602 )\n\t);\n\tcolor *= toneMappingExposure / 0.6;\n\tcolor = ACESInputMat * color;\n\tcolor = RRTAndODTFit( color );\n\tcolor = ACESOutputMat * color;\n\treturn saturate( color );\n}\nvec3 CustomToneMapping( vec3 color ) { return color; }",transmission_fragment:"#ifdef USE_TRANSMISSION\n\tfloat transmissionAlpha = 1.0;\n\tfloat transmissionFactor = transmission;\n\tfloat thicknessFactor = thickness;\n\t#ifdef USE_TRANSMISSIONMAP\n\t\ttransmissionFactor *= texture2D( transmissionMap, vUv ).r;\n\t#endif\n\t#ifdef USE_THICKNESSMAP\n\t\tthicknessFactor *= texture2D( thicknessMap, vUv ).g;\n\t#endif\n\tvec3 pos = vWorldPosition;\n\tvec3 v = normalize( cameraPosition - pos );\n\tvec3 n = inverseTransformDirection( normal, viewMatrix );\n\tvec4 transmission = getIBLVolumeRefraction(\n\t\tn, v, roughnessFactor, material.diffuseColor, material.specularColor, material.specularF90,\n\t\tpos, modelMatrix, viewMatrix, projectionMatrix, ior, thicknessFactor,\n\t\tattenuationColor, attenuationDistance );\n\ttotalDiffuse = mix( totalDiffuse, transmission.rgb, transmissionFactor );\n\ttransmissionAlpha = mix( transmissionAlpha, transmission.a, transmissionFactor );\n#endif",transmission_pars_fragment:"#ifdef USE_TRANSMISSION\n\tuniform float transmission;\n\tuniform float thickness;\n\tuniform float attenuationDistance;\n\tuniform vec3 attenuationColor;\n\t#ifdef USE_TRANSMISSIONMAP\n\t\tuniform sampler2D transmissionMap;\n\t#endif\n\t#ifdef USE_THICKNESSMAP\n\t\tuniform sampler2D thicknessMap;\n\t#endif\n\tuniform vec2 transmissionSamplerSize;\n\tuniform sampler2D transmissionSamplerMap;\n\tuniform mat4 modelMatrix;\n\tuniform mat4 projectionMatrix;\n\tvarying vec3 vWorldPosition;\n\tvec3 getVolumeTransmissionRay( vec3 n, vec3 v, float thickness, float ior, mat4 modelMatrix ) {\n\t\tvec3 refractionVector = refract( - v, normalize( n ), 1.0 / ior );\n\t\tvec3 modelScale;\n\t\tmodelScale.x = length( vec3( modelMatrix[ 0 ].xyz ) );\n\t\tmodelScale.y = length( vec3( modelMatrix[ 1 ].xyz ) );\n\t\tmodelScale.z = length( vec3( modelMatrix[ 2 ].xyz ) );\n\t\treturn normalize( refractionVector ) * thickness * modelScale;\n\t}\n\tfloat applyIorToRoughness( float roughness, float ior ) {\n\t\treturn roughness * clamp( ior * 2.0 - 2.0, 0.0, 1.0 );\n\t}\n\tvec4 getTransmissionSample( vec2 fragCoord, float roughness, float ior ) {\n\t\tfloat framebufferLod = log2( transmissionSamplerSize.x ) * applyIorToRoughness( roughness, ior );\n\t\t#ifdef TEXTURE_LOD_EXT\n\t\t\treturn texture2DLodEXT( transmissionSamplerMap, fragCoord.xy, framebufferLod );\n\t\t#else\n\t\t\treturn texture2D( transmissionSamplerMap, fragCoord.xy, framebufferLod );\n\t\t#endif\n\t}\n\tvec3 applyVolumeAttenuation( vec3 radiance, float transmissionDistance, vec3 attenuationColor, float attenuationDistance ) {\n\t\tif ( attenuationDistance == 0.0 ) {\n\t\t\treturn radiance;\n\t\t} else {\n\t\t\tvec3 attenuationCoefficient = -log( attenuationColor ) / attenuationDistance;\n\t\t\tvec3 transmittance = exp( - attenuationCoefficient * transmissionDistance );\t\t\treturn transmittance * radiance;\n\t\t}\n\t}\n\tvec4 getIBLVolumeRefraction( vec3 n, vec3 v, float roughness, vec3 diffuseColor, vec3 specularColor, float specularF90,\n\t\tvec3 position, mat4 modelMatrix, mat4 viewMatrix, mat4 projMatrix, float ior, float thickness,\n\t\tvec3 attenuationColor, float attenuationDistance ) {\n\t\tvec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, ior, modelMatrix );\n\t\tvec3 refractedRayExit = position + transmissionRay;\n\t\tvec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );\n\t\tvec2 refractionCoords = ndcPos.xy / ndcPos.w;\n\t\trefractionCoords += 1.0;\n\t\trefractionCoords /= 2.0;\n\t\tvec4 transmittedLight = getTransmissionSample( refractionCoords, roughness, ior );\n\t\tvec3 attenuatedColor = applyVolumeAttenuation( transmittedLight.rgb, length( transmissionRay ), attenuationColor, attenuationDistance );\n\t\tvec3 F = EnvironmentBRDF( n, v, specularColor, specularF90, roughness );\n\t\treturn vec4( ( 1.0 - F ) * attenuatedColor * diffuseColor, transmittedLight.a );\n\t}\n#endif",uv_pars_fragment:"#if ( defined( USE_UV ) && ! defined( UVS_VERTEX_ONLY ) )\n\tvarying vec2 vUv;\n#endif",uv_pars_vertex:"#ifdef USE_UV\n\t#ifdef UVS_VERTEX_ONLY\n\t\tvec2 vUv;\n\t#else\n\t\tvarying vec2 vUv;\n\t#endif\n\tuniform mat3 uvTransform;\n#endif",uv_vertex:"#ifdef USE_UV\n\tvUv = ( uvTransform * vec3( uv, 1 ) ).xy;\n#endif",uv2_pars_fragment:"#if defined( USE_LIGHTMAP ) || defined( USE_AOMAP )\n\tvarying vec2 vUv2;\n#endif",uv2_pars_vertex:"#if defined( USE_LIGHTMAP ) || defined( USE_AOMAP )\n\tattribute vec2 uv2;\n\tvarying vec2 vUv2;\n\tuniform mat3 uv2Transform;\n#endif",uv2_vertex:"#if defined( USE_LIGHTMAP ) || defined( USE_AOMAP )\n\tvUv2 = ( uv2Transform * vec3( uv2, 1 ) ).xy;\n#endif",worldpos_vertex:"#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION )\n\tvec4 worldPosition = vec4( transformed, 1.0 );\n\t#ifdef USE_INSTANCING\n\t\tworldPosition = instanceMatrix * worldPosition;\n\t#endif\n\tworldPosition = modelMatrix * worldPosition;\n#endif",background_vert:"varying vec2 vUv;\nuniform mat3 uvTransform;\nvoid main() {\n\tvUv = ( uvTransform * vec3( uv, 1 ) ).xy;\n\tgl_Position = vec4( position.xy, 1.0, 1.0 );\n}",background_frag:"uniform sampler2D t2D;\nvarying vec2 vUv;\nvoid main() {\n\tvec4 texColor = texture2D( t2D, vUv );\n\tgl_FragColor = mapTexelToLinear( texColor );\n\t#include <tonemapping_fragment>\n\t#include <encodings_fragment>\n}",cube_vert:"varying vec3 vWorldDirection;\n#include <common>\nvoid main() {\n\tvWorldDirection = transformDirection( position, modelMatrix );\n\t#include <begin_vertex>\n\t#include <project_vertex>\n\tgl_Position.z = gl_Position.w;\n}",cube_frag:"#include <envmap_common_pars_fragment>\nuniform float opacity;\nvarying vec3 vWorldDirection;\n#include <cube_uv_reflection_fragment>\nvoid main() {\n\tvec3 vReflect = vWorldDirection;\n\t#include <envmap_fragment>\n\tgl_FragColor = envColor;\n\tgl_FragColor.a *= opacity;\n\t#include <tonemapping_fragment>\n\t#include <encodings_fragment>\n}",depth_vert:"#include <common>\n#include <uv_pars_vertex>\n#include <displacementmap_pars_vertex>\n#include <morphtarget_pars_vertex>\n#include <skinning_pars_vertex>\n#include <logdepthbuf_pars_vertex>\n#include <clipping_planes_pars_vertex>\nvarying vec2 vHighPrecisionZW;\nvoid main() {\n\t#include <uv_vertex>\n\t#include <skinbase_vertex>\n\t#ifdef USE_DISPLACEMENTMAP\n\t\t#include <beginnormal_vertex>\n\t\t#include <morphnormal_vertex>\n\t\t#include <skinnormal_vertex>\n\t#endif\n\t#include <begin_vertex>\n\t#include <morphtarget_vertex>\n\t#include <skinning_vertex>\n\t#include <displacementmap_vertex>\n\t#include <project_vertex>\n\t#include <logdepthbuf_vertex>\n\t#include <clipping_planes_vertex>\n\tvHighPrecisionZW = gl_Position.zw;\n}",depth_frag:"#if DEPTH_PACKING == 3200\n\tuniform float opacity;\n#endif\n#include <common>\n#include <packing>\n#include <uv_pars_fragment>\n#include <map_pars_fragment>\n#include <alphamap_pars_fragment>\n#include <alphatest_pars_fragment>\n#include <logdepthbuf_pars_fragment>\n#include <clipping_planes_pars_fragment>\nvarying vec2 vHighPrecisionZW;\nvoid main() {\n\t#include <clipping_planes_fragment>\n\tvec4 diffuseColor = vec4( 1.0 );\n\t#if DEPTH_PACKING == 3200\n\t\tdiffuseColor.a = opacity;\n\t#endif\n\t#include <map_fragment>\n\t#include <alphamap_fragment>\n\t#include <alphatest_fragment>\n\t#include <logdepthbuf_fragment>\n\tfloat fragCoordZ = 0.5 * vHighPrecisionZW[0] / vHighPrecisionZW[1] + 0.5;\n\t#if DEPTH_PACKING == 3200\n\t\tgl_FragColor = vec4( vec3( 1.0 - fragCoordZ ), opacity );\n\t#elif DEPTH_PACKING == 3201\n\t\tgl_FragColor = packDepthToRGBA( fragCoordZ );\n\t#endif\n}",distanceRGBA_vert:"#define DISTANCE\nvarying vec3 vWorldPosition;\n#include <common>\n#include <uv_pars_vertex>\n#include <displacementmap_pars_vertex>\n#include <morphtarget_pars_vertex>\n#include <skinning_pars_vertex>\n#include <clipping_planes_pars_vertex>\nvoid main() {\n\t#include <uv_vertex>\n\t#include <skinbase_vertex>\n\t#ifdef USE_DISPLACEMENTMAP\n\t\t#include <beginnormal_vertex>\n\t\t#include <morphnormal_vertex>\n\t\t#include <skinnormal_vertex>\n\t#endif\n\t#include <begin_vertex>\n\t#include <morphtarget_vertex>\n\t#include <skinning_vertex>\n\t#include <displacementmap_vertex>\n\t#include <project_vertex>\n\t#include <worldpos_vertex>\n\t#include <clipping_planes_vertex>\n\tvWorldPosition = worldPosition.xyz;\n}",distanceRGBA_frag:"#define DISTANCE\nuniform vec3 referencePosition;\nuniform float nearDistance;\nuniform float farDistance;\nvarying vec3 vWorldPosition;\n#include <common>\n#include <packing>\n#include <uv_pars_fragment>\n#include <map_pars_fragment>\n#include <alphamap_pars_fragment>\n#include <alphatest_pars_fragment>\n#include <clipping_planes_pars_fragment>\nvoid main () {\n\t#include <clipping_planes_fragment>\n\tvec4 diffuseColor = vec4( 1.0 );\n\t#include <map_fragment>\n\t#include <alphamap_fragment>\n\t#include <alphatest_fragment>\n\tfloat dist = length( vWorldPosition - referencePosition );\n\tdist = ( dist - nearDistance ) / ( farDistance - nearDistance );\n\tdist = saturate( dist );\n\tgl_FragColor = packDepthToRGBA( dist );\n}",equirect_vert:"varying vec3 vWorldDirection;\n#include <common>\nvoid main() {\n\tvWorldDirection = transformDirection( position, modelMatrix );\n\t#include <begin_vertex>\n\t#include <project_vertex>\n}",equirect_frag:"uniform sampler2D tEquirect;\nvarying vec3 vWorldDirection;\n#include <common>\nvoid main() {\n\tvec3 direction = normalize( vWorldDirection );\n\tvec2 sampleUV = equirectUv( direction );\n\tvec4 texColor = texture2D( tEquirect, sampleUV );\n\tgl_FragColor = mapTexelToLinear( texColor );\n\t#include <tonemapping_fragment>\n\t#include <encodings_fragment>\n}",linedashed_vert:"uniform float scale;\nattribute float lineDistance;\nvarying float vLineDistance;\n#include <common>\n#include <color_pars_vertex>\n#include <fog_pars_vertex>\n#include <morphtarget_pars_vertex>\n#include <logdepthbuf_pars_vertex>\n#include <clipping_planes_pars_vertex>\nvoid main() {\n\tvLineDistance = scale * lineDistance;\n\t#include <color_vertex>\n\t#include <begin_vertex>\n\t#include <morphtarget_vertex>\n\t#include <project_vertex>\n\t#include <logdepthbuf_vertex>\n\t#include <clipping_planes_vertex>\n\t#include <fog_vertex>\n}",linedashed_frag:"uniform vec3 diffuse;\nuniform float opacity;\nuniform float dashSize;\nuniform float totalSize;\nvarying float vLineDistance;\n#include <common>\n#include <color_pars_fragment>\n#include <fog_pars_fragment>\n#include <logdepthbuf_pars_fragment>\n#include <clipping_planes_pars_fragment>\nvoid main() {\n\t#include <clipping_planes_fragment>\n\tif ( mod( vLineDistance, totalSize ) > dashSize ) {\n\t\tdiscard;\n\t}\n\tvec3 outgoingLight = vec3( 0.0 );\n\tvec4 diffuseColor = vec4( diffuse, opacity );\n\t#include <logdepthbuf_fragment>\n\t#include <color_fragment>\n\toutgoingLight = diffuseColor.rgb;\n\t#include <output_fragment>\n\t#include <tonemapping_fragment>\n\t#include <encodings_fragment>\n\t#include <fog_fragment>\n\t#include <premultiplied_alpha_fragment>\n}",meshbasic_vert:"#include <common>\n#include <uv_pars_vertex>\n#include <uv2_pars_vertex>\n#include <envmap_pars_vertex>\n#include <color_pars_vertex>\n#include <fog_pars_vertex>\n#include <morphtarget_pars_vertex>\n#include <skinning_pars_vertex>\n#include <logdepthbuf_pars_vertex>\n#include <clipping_planes_pars_vertex>\nvoid main() {\n\t#include <uv_vertex>\n\t#include <uv2_vertex>\n\t#include <color_vertex>\n\t#if defined ( USE_ENVMAP ) || defined ( USE_SKINNING )\n\t\t#include <beginnormal_vertex>\n\t\t#include <morphnormal_vertex>\n\t\t#include <skinbase_vertex>\n\t\t#include <skinnormal_vertex>\n\t\t#include <defaultnormal_vertex>\n\t#endif\n\t#include <begin_vertex>\n\t#include <morphtarget_vertex>\n\t#include <skinning_vertex>\n\t#include <project_vertex>\n\t#include <logdepthbuf_vertex>\n\t#include <clipping_planes_vertex>\n\t#include <worldpos_vertex>\n\t#include <envmap_vertex>\n\t#include <fog_vertex>\n}",meshbasic_frag:"uniform vec3 diffuse;\nuniform float opacity;\n#ifndef FLAT_SHADED\n\tvarying vec3 vNormal;\n#endif\n#include <common>\n#include <dithering_pars_fragment>\n#include <color_pars_fragment>\n#include <uv_pars_fragment>\n#include <uv2_pars_fragment>\n#include <map_pars_fragment>\n#include <alphamap_pars_fragment>\n#include <alphatest_pars_fragment>\n#include <aomap_pars_fragment>\n#include <lightmap_pars_fragment>\n#include <envmap_common_pars_fragment>\n#include <envmap_pars_fragment>\n#include <cube_uv_reflection_fragment>\n#include <fog_pars_fragment>\n#include <specularmap_pars_fragment>\n#include <logdepthbuf_pars_fragment>\n#include <clipping_planes_pars_fragment>\nvoid main() {\n\t#include <clipping_planes_fragment>\n\tvec4 diffuseColor = vec4( diffuse, opacity );\n\t#include <logdepthbuf_fragment>\n\t#include <map_fragment>\n\t#include <color_fragment>\n\t#include <alphamap_fragment>\n\t#include <alphatest_fragment>\n\t#include <specularmap_fragment>\n\tReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );\n\t#ifdef USE_LIGHTMAP\n\t\tvec4 lightMapTexel= texture2D( lightMap, vUv2 );\n\t\treflectedLight.indirectDiffuse += lightMapTexelToLinear( lightMapTexel ).rgb * lightMapIntensity;\n\t#else\n\t\treflectedLight.indirectDiffuse += vec3( 1.0 );\n\t#endif\n\t#include <aomap_fragment>\n\treflectedLight.indirectDiffuse *= diffuseColor.rgb;\n\tvec3 outgoingLight = reflectedLight.indirectDiffuse;\n\t#include <envmap_fragment>\n\t#include <output_fragment>\n\t#include <tonemapping_fragment>\n\t#include <encodings_fragment>\n\t#include <fog_fragment>\n\t#include <premultiplied_alpha_fragment>\n\t#include <dithering_fragment>\n}",meshlambert_vert:"#define LAMBERT\nvarying vec3 vLightFront;\nvarying vec3 vIndirectFront;\n#ifdef DOUBLE_SIDED\n\tvarying vec3 vLightBack;\n\tvarying vec3 vIndirectBack;\n#endif\n#include <common>\n#include <uv_pars_vertex>\n#include <uv2_pars_vertex>\n#include <envmap_pars_vertex>\n#include <bsdfs>\n#include <lights_pars_begin>\n#include <color_pars_vertex>\n#include <fog_pars_vertex>\n#include <morphtarget_pars_vertex>\n#include <skinning_pars_vertex>\n#include <shadowmap_pars_vertex>\n#include <logdepthbuf_pars_vertex>\n#include <clipping_planes_pars_vertex>\nvoid main() {\n\t#include <uv_vertex>\n\t#include <uv2_vertex>\n\t#include <color_vertex>\n\t#include <beginnormal_vertex>\n\t#include <morphnormal_vertex>\n\t#include <skinbase_vertex>\n\t#include <skinnormal_vertex>\n\t#include <defaultnormal_vertex>\n\t#include <begin_vertex>\n\t#include <morphtarget_vertex>\n\t#include <skinning_vertex>\n\t#include <project_vertex>\n\t#include <logdepthbuf_vertex>\n\t#include <clipping_planes_vertex>\n\t#include <worldpos_vertex>\n\t#include <envmap_vertex>\n\t#include <lights_lambert_vertex>\n\t#include <shadowmap_vertex>\n\t#include <fog_vertex>\n}",meshlambert_frag:"uniform vec3 diffuse;\nuniform vec3 emissive;\nuniform float opacity;\nvarying vec3 vLightFront;\nvarying vec3 vIndirectFront;\n#ifdef DOUBLE_SIDED\n\tvarying vec3 vLightBack;\n\tvarying vec3 vIndirectBack;\n#endif\n#include <common>\n#include <packing>\n#include <dithering_pars_fragment>\n#include <color_pars_fragment>\n#include <uv_pars_fragment>\n#include <uv2_pars_fragment>\n#include <map_pars_fragment>\n#include <alphamap_pars_fragment>\n#include <alphatest_pars_fragment>\n#include <aomap_pars_fragment>\n#include <lightmap_pars_fragment>\n#include <emissivemap_pars_fragment>\n#include <envmap_common_pars_fragment>\n#include <envmap_pars_fragment>\n#include <cube_uv_reflection_fragment>\n#include <bsdfs>\n#include <lights_pars_begin>\n#include <fog_pars_fragment>\n#include <shadowmap_pars_fragment>\n#include <shadowmask_pars_fragment>\n#include <specularmap_pars_fragment>\n#include <logdepthbuf_pars_fragment>\n#include <clipping_planes_pars_fragment>\nvoid main() {\n\t#include <clipping_planes_fragment>\n\tvec4 diffuseColor = vec4( diffuse, opacity );\n\tReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );\n\tvec3 totalEmissiveRadiance = emissive;\n\t#include <logdepthbuf_fragment>\n\t#include <map_fragment>\n\t#include <color_fragment>\n\t#include <alphamap_fragment>\n\t#include <alphatest_fragment>\n\t#include <specularmap_fragment>\n\t#include <emissivemap_fragment>\n\t#ifdef DOUBLE_SIDED\n\t\treflectedLight.indirectDiffuse += ( gl_FrontFacing ) ? vIndirectFront : vIndirectBack;\n\t#else\n\t\treflectedLight.indirectDiffuse += vIndirectFront;\n\t#endif\n\t#include <lightmap_fragment>\n\treflectedLight.indirectDiffuse *= BRDF_Lambert( diffuseColor.rgb );\n\t#ifdef DOUBLE_SIDED\n\t\treflectedLight.directDiffuse = ( gl_FrontFacing ) ? vLightFront : vLightBack;\n\t#else\n\t\treflectedLight.directDiffuse = vLightFront;\n\t#endif\n\treflectedLight.directDiffuse *= BRDF_Lambert( diffuseColor.rgb ) * getShadowMask();\n\t#include <aomap_fragment>\n\tvec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;\n\t#include <envmap_fragment>\n\t#include <output_fragment>\n\t#include <tonemapping_fragment>\n\t#include <encodings_fragment>\n\t#include <fog_fragment>\n\t#include <premultiplied_alpha_fragment>\n\t#include <dithering_fragment>\n}",meshmatcap_vert:"#define MATCAP\nvarying vec3 vViewPosition;\n#include <common>\n#include <uv_pars_vertex>\n#include <color_pars_vertex>\n#include <displacementmap_pars_vertex>\n#include <fog_pars_vertex>\n#include <normal_pars_vertex>\n#include <morphtarget_pars_vertex>\n#include <skinning_pars_vertex>\n#include <logdepthbuf_pars_vertex>\n#include <clipping_planes_pars_vertex>\nvoid main() {\n\t#include <uv_vertex>\n\t#include <color_vertex>\n\t#include <beginnormal_vertex>\n\t#include <morphnormal_vertex>\n\t#include <skinbase_vertex>\n\t#include <skinnormal_vertex>\n\t#include <defaultnormal_vertex>\n\t#include <normal_vertex>\n\t#include <begin_vertex>\n\t#include <morphtarget_vertex>\n\t#include <skinning_vertex>\n\t#include <displacementmap_vertex>\n\t#include <project_vertex>\n\t#include <logdepthbuf_vertex>\n\t#include <clipping_planes_vertex>\n\t#include <fog_vertex>\n\tvViewPosition = - mvPosition.xyz;\n}",meshmatcap_frag:"#define MATCAP\nuniform vec3 diffuse;\nuniform float opacity;\nuniform sampler2D matcap;\nvarying vec3 vViewPosition;\n#include <common>\n#include <dithering_pars_fragment>\n#include <color_pars_fragment>\n#include <uv_pars_fragment>\n#include <map_pars_fragment>\n#include <alphamap_pars_fragment>\n#include <alphatest_pars_fragment>\n#include <fog_pars_fragment>\n#include <normal_pars_fragment>\n#include <bumpmap_pars_fragment>\n#include <normalmap_pars_fragment>\n#include <logdepthbuf_pars_fragment>\n#include <clipping_planes_pars_fragment>\nvoid main() {\n\t#include <clipping_planes_fragment>\n\tvec4 diffuseColor = vec4( diffuse, opacity );\n\t#include <logdepthbuf_fragment>\n\t#include <map_fragment>\n\t#include <color_fragment>\n\t#include <alphamap_fragment>\n\t#include <alphatest_fragment>\n\t#include <normal_fragment_begin>\n\t#include <normal_fragment_maps>\n\tvec3 viewDir = normalize( vViewPosition );\n\tvec3 x = normalize( vec3( viewDir.z, 0.0, - viewDir.x ) );\n\tvec3 y = cross( viewDir, x );\n\tvec2 uv = vec2( dot( x, normal ), dot( y, normal ) ) * 0.495 + 0.5;\n\t#ifdef USE_MATCAP\n\t\tvec4 matcapColor = texture2D( matcap, uv );\n\t\tmatcapColor = matcapTexelToLinear( matcapColor );\n\t#else\n\t\tvec4 matcapColor = vec4( 1.0 );\n\t#endif\n\tvec3 outgoingLight = diffuseColor.rgb * matcapColor.rgb;\n\t#include <output_fragment>\n\t#include <tonemapping_fragment>\n\t#include <encodings_fragment>\n\t#include <fog_fragment>\n\t#include <premultiplied_alpha_fragment>\n\t#include <dithering_fragment>\n}",meshnormal_vert:"#define NORMAL\n#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( TANGENTSPACE_NORMALMAP )\n\tvarying vec3 vViewPosition;\n#endif\n#include <common>\n#include <uv_pars_vertex>\n#include <displacementmap_pars_vertex>\n#include <normal_pars_vertex>\n#include <morphtarget_pars_vertex>\n#include <skinning_pars_vertex>\n#include <logdepthbuf_pars_vertex>\n#include <clipping_planes_pars_vertex>\nvoid main() {\n\t#include <uv_vertex>\n\t#include <beginnormal_vertex>\n\t#include <morphnormal_vertex>\n\t#include <skinbase_vertex>\n\t#include <skinnormal_vertex>\n\t#include <defaultnormal_vertex>\n\t#include <normal_vertex>\n\t#include <begin_vertex>\n\t#include <morphtarget_vertex>\n\t#include <skinning_vertex>\n\t#include <displacementmap_vertex>\n\t#include <project_vertex>\n\t#include <logdepthbuf_vertex>\n\t#include <clipping_planes_vertex>\n#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( TANGENTSPACE_NORMALMAP )\n\tvViewPosition = - mvPosition.xyz;\n#endif\n}",meshnormal_frag:"#define NORMAL\nuniform float opacity;\n#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( TANGENTSPACE_NORMALMAP )\n\tvarying vec3 vViewPosition;\n#endif\n#include <packing>\n#include <uv_pars_fragment>\n#include <normal_pars_fragment>\n#include <bumpmap_pars_fragment>\n#include <normalmap_pars_fragment>\n#include <logdepthbuf_pars_fragment>\n#include <clipping_planes_pars_fragment>\nvoid main() {\n\t#include <clipping_planes_fragment>\n\t#include <logdepthbuf_fragment>\n\t#include <normal_fragment_begin>\n\t#include <normal_fragment_maps>\n\tgl_FragColor = vec4( packNormalToRGB( normal ), opacity );\n}",meshphong_vert:"#define PHONG\nvarying vec3 vViewPosition;\n#include <common>\n#include <uv_pars_vertex>\n#include <uv2_pars_vertex>\n#include <displacementmap_pars_vertex>\n#include <envmap_pars_vertex>\n#include <color_pars_vertex>\n#include <fog_pars_vertex>\n#include <normal_pars_vertex>\n#include <morphtarget_pars_vertex>\n#include <skinning_pars_vertex>\n#include <shadowmap_pars_vertex>\n#include <logdepthbuf_pars_vertex>\n#include <clipping_planes_pars_vertex>\nvoid main() {\n\t#include <uv_vertex>\n\t#include <uv2_vertex>\n\t#include <color_vertex>\n\t#include <beginnormal_vertex>\n\t#include <morphnormal_vertex>\n\t#include <skinbase_vertex>\n\t#include <skinnormal_vertex>\n\t#include <defaultnormal_vertex>\n\t#include <normal_vertex>\n\t#include <begin_vertex>\n\t#include <morphtarget_vertex>\n\t#include <skinning_vertex>\n\t#include <displacementmap_vertex>\n\t#include <project_vertex>\n\t#include <logdepthbuf_vertex>\n\t#include <clipping_planes_vertex>\n\tvViewPosition = - mvPosition.xyz;\n\t#include <worldpos_vertex>\n\t#include <envmap_vertex>\n\t#include <shadowmap_vertex>\n\t#include <fog_vertex>\n}",meshphong_frag:"#define PHONG\nuniform vec3 diffuse;\nuniform vec3 emissive;\nuniform vec3 specular;\nuniform float shininess;\nuniform float opacity;\n#include <common>\n#include <packing>\n#include <dithering_pars_fragment>\n#include <color_pars_fragment>\n#include <uv_pars_fragment>\n#include <uv2_pars_fragment>\n#include <map_pars_fragment>\n#include <alphamap_pars_fragment>\n#include <alphatest_pars_fragment>\n#include <aomap_pars_fragment>\n#include <lightmap_pars_fragment>\n#include <emissivemap_pars_fragment>\n#include <envmap_common_pars_fragment>\n#include <envmap_pars_fragment>\n#include <cube_uv_reflection_fragment>\n#include <fog_pars_fragment>\n#include <bsdfs>\n#include <lights_pars_begin>\n#include <normal_pars_fragment>\n#include <lights_phong_pars_fragment>\n#include <shadowmap_pars_fragment>\n#include <bumpmap_pars_fragment>\n#include <normalmap_pars_fragment>\n#include <specularmap_pars_fragment>\n#include <logdepthbuf_pars_fragment>\n#include <clipping_planes_pars_fragment>\nvoid main() {\n\t#include <clipping_planes_fragment>\n\tvec4 diffuseColor = vec4( diffuse, opacity );\n\tReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );\n\tvec3 totalEmissiveRadiance = emissive;\n\t#include <logdepthbuf_fragment>\n\t#include <map_fragment>\n\t#include <color_fragment>\n\t#include <alphamap_fragment>\n\t#include <alphatest_fragment>\n\t#include <specularmap_fragment>\n\t#include <normal_fragment_begin>\n\t#include <normal_fragment_maps>\n\t#include <emissivemap_fragment>\n\t#include <lights_phong_fragment>\n\t#include <lights_fragment_begin>\n\t#include <lights_fragment_maps>\n\t#include <lights_fragment_end>\n\t#include <aomap_fragment>\n\tvec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;\n\t#include <envmap_fragment>\n\t#include <output_fragment>\n\t#include <tonemapping_fragment>\n\t#include <encodings_fragment>\n\t#include <fog_fragment>\n\t#include <premultiplied_alpha_fragment>\n\t#include <dithering_fragment>\n}",meshphysical_vert:"#define STANDARD\nvarying vec3 vViewPosition;\n#ifdef USE_TRANSMISSION\n\tvarying vec3 vWorldPosition;\n#endif\n#include <common>\n#include <uv_pars_vertex>\n#include <uv2_pars_vertex>\n#include <displacementmap_pars_vertex>\n#include <color_pars_vertex>\n#include <fog_pars_vertex>\n#include <normal_pars_vertex>\n#include <morphtarget_pars_vertex>\n#include <skinning_pars_vertex>\n#include <shadowmap_pars_vertex>\n#include <logdepthbuf_pars_vertex>\n#include <clipping_planes_pars_vertex>\nvoid main() {\n\t#include <uv_vertex>\n\t#include <uv2_vertex>\n\t#include <color_vertex>\n\t#include <beginnormal_vertex>\n\t#include <morphnormal_vertex>\n\t#include <skinbase_vertex>\n\t#include <skinnormal_vertex>\n\t#include <defaultnormal_vertex>\n\t#include <normal_vertex>\n\t#include <begin_vertex>\n\t#include <morphtarget_vertex>\n\t#include <skinning_vertex>\n\t#include <displacementmap_vertex>\n\t#include <project_vertex>\n\t#include <logdepthbuf_vertex>\n\t#include <clipping_planes_vertex>\n\tvViewPosition = - mvPosition.xyz;\n\t#include <worldpos_vertex>\n\t#include <shadowmap_vertex>\n\t#include <fog_vertex>\n#ifdef USE_TRANSMISSION\n\tvWorldPosition = worldPosition.xyz;\n#endif\n}",meshphysical_frag:"#define STANDARD\n#ifdef PHYSICAL\n\t#define IOR\n\t#define SPECULAR\n#endif\nuniform vec3 diffuse;\nuniform vec3 emissive;\nuniform float roughness;\nuniform float metalness;\nuniform float opacity;\n#ifdef IOR\n\tuniform float ior;\n#endif\n#ifdef SPECULAR\n\tuniform float specularIntensity;\n\tuniform vec3 specularColor;\n\t#ifdef USE_SPECULARINTENSITYMAP\n\t\tuniform sampler2D specularIntensityMap;\n\t#endif\n\t#ifdef USE_SPECULARCOLORMAP\n\t\tuniform sampler2D specularColorMap;\n\t#endif\n#endif\n#ifdef USE_CLEARCOAT\n\tuniform float clearcoat;\n\tuniform float clearcoatRoughness;\n#endif\n#ifdef USE_SHEEN\n\tuniform vec3 sheenColor;\n\tuniform float sheenRoughness;\n\t#ifdef USE_SHEENCOLORMAP\n\t\tuniform sampler2D sheenColorMap;\n\t#endif\n\t#ifdef USE_SHEENROUGHNESSMAP\n\t\tuniform sampler2D sheenRoughnessMap;\n\t#endif\n#endif\nvarying vec3 vViewPosition;\n#include <common>\n#include <packing>\n#include <dithering_pars_fragment>\n#include <color_pars_fragment>\n#include <uv_pars_fragment>\n#include <uv2_pars_fragment>\n#include <map_pars_fragment>\n#include <alphamap_pars_fragment>\n#include <alphatest_pars_fragment>\n#include <aomap_pars_fragment>\n#include <lightmap_pars_fragment>\n#include <emissivemap_pars_fragment>\n#include <bsdfs>\n#include <cube_uv_reflection_fragment>\n#include <envmap_common_pars_fragment>\n#include <envmap_physical_pars_fragment>\n#include <fog_pars_fragment>\n#include <lights_pars_begin>\n#include <normal_pars_fragment>\n#include <lights_physical_pars_fragment>\n#include <transmission_pars_fragment>\n#include <shadowmap_pars_fragment>\n#include <bumpmap_pars_fragment>\n#include <normalmap_pars_fragment>\n#include <clearcoat_pars_fragment>\n#include <roughnessmap_pars_fragment>\n#include <metalnessmap_pars_fragment>\n#include <logdepthbuf_pars_fragment>\n#include <clipping_planes_pars_fragment>\nvoid main() {\n\t#include <clipping_planes_fragment>\n\tvec4 diffuseColor = vec4( diffuse, opacity );\n\tReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );\n\tvec3 totalEmissiveRadiance = emissive;\n\t#include <logdepthbuf_fragment>\n\t#include <map_fragment>\n\t#include <color_fragment>\n\t#include <alphamap_fragment>\n\t#include <alphatest_fragment>\n\t#include <roughnessmap_fragment>\n\t#include <metalnessmap_fragment>\n\t#include <normal_fragment_begin>\n\t#include <normal_fragment_maps>\n\t#include <clearcoat_normal_fragment_begin>\n\t#include <clearcoat_normal_fragment_maps>\n\t#include <emissivemap_fragment>\n\t#include <lights_physical_fragment>\n\t#include <lights_fragment_begin>\n\t#include <lights_fragment_maps>\n\t#include <lights_fragment_end>\n\t#include <aomap_fragment>\n\tvec3 totalDiffuse = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;\n\tvec3 totalSpecular = reflectedLight.directSpecular + reflectedLight.indirectSpecular;\n\t#include <transmission_fragment>\n\tvec3 outgoingLight = totalDiffuse + totalSpecular + totalEmissiveRadiance;\n\t#ifdef USE_CLEARCOAT\n\t\tfloat dotNVcc = saturate( dot( geometry.clearcoatNormal, geometry.viewDir ) );\n\t\tvec3 Fcc = F_Schlick( material.clearcoatF0, material.clearcoatF90, dotNVcc );\n\t\toutgoingLight = outgoingLight * ( 1.0 - clearcoat * Fcc ) + clearcoatSpecular * clearcoat;\n\t#endif\n\t#include <output_fragment>\n\t#include <tonemapping_fragment>\n\t#include <encodings_fragment>\n\t#include <fog_fragment>\n\t#include <premultiplied_alpha_fragment>\n\t#include <dithering_fragment>\n}",meshtoon_vert:"#define TOON\nvarying vec3 vViewPosition;\n#include <common>\n#include <uv_pars_vertex>\n#include <uv2_pars_vertex>\n#include <displacementmap_pars_vertex>\n#include <color_pars_vertex>\n#include <fog_pars_vertex>\n#include <normal_pars_vertex>\n#include <morphtarget_pars_vertex>\n#include <skinning_pars_vertex>\n#include <shadowmap_pars_vertex>\n#include <logdepthbuf_pars_vertex>\n#include <clipping_planes_pars_vertex>\nvoid main() {\n\t#include <uv_vertex>\n\t#include <uv2_vertex>\n\t#include <color_vertex>\n\t#include <beginnormal_vertex>\n\t#include <morphnormal_vertex>\n\t#include <skinbase_vertex>\n\t#include <skinnormal_vertex>\n\t#include <defaultnormal_vertex>\n\t#include <normal_vertex>\n\t#include <begin_vertex>\n\t#include <morphtarget_vertex>\n\t#include <skinning_vertex>\n\t#include <displacementmap_vertex>\n\t#include <project_vertex>\n\t#include <logdepthbuf_vertex>\n\t#include <clipping_planes_vertex>\n\tvViewPosition = - mvPosition.xyz;\n\t#include <worldpos_vertex>\n\t#include <shadowmap_vertex>\n\t#include <fog_vertex>\n}",meshtoon_frag:"#define TOON\nuniform vec3 diffuse;\nuniform vec3 emissive;\nuniform float opacity;\n#include <common>\n#include <packing>\n#include <dithering_pars_fragment>\n#include <color_pars_fragment>\n#include <uv_pars_fragment>\n#include <uv2_pars_fragment>\n#include <map_pars_fragment>\n#include <alphamap_pars_fragment>\n#include <alphatest_pars_fragment>\n#include <aomap_pars_fragment>\n#include <lightmap_pars_fragment>\n#include <emissivemap_pars_fragment>\n#include <gradientmap_pars_fragment>\n#include <fog_pars_fragment>\n#include <bsdfs>\n#include <lights_pars_begin>\n#include <normal_pars_fragment>\n#include <lights_toon_pars_fragment>\n#include <shadowmap_pars_fragment>\n#include <bumpmap_pars_fragment>\n#include <normalmap_pars_fragment>\n#include <logdepthbuf_pars_fragment>\n#include <clipping_planes_pars_fragment>\nvoid main() {\n\t#include <clipping_planes_fragment>\n\tvec4 diffuseColor = vec4( diffuse, opacity );\n\tReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );\n\tvec3 totalEmissiveRadiance = emissive;\n\t#include <logdepthbuf_fragment>\n\t#include <map_fragment>\n\t#include <color_fragment>\n\t#include <alphamap_fragment>\n\t#include <alphatest_fragment>\n\t#include <normal_fragment_begin>\n\t#include <normal_fragment_maps>\n\t#include <emissivemap_fragment>\n\t#include <lights_toon_fragment>\n\t#include <lights_fragment_begin>\n\t#include <lights_fragment_maps>\n\t#include <lights_fragment_end>\n\t#include <aomap_fragment>\n\tvec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;\n\t#include <output_fragment>\n\t#include <tonemapping_fragment>\n\t#include <encodings_fragment>\n\t#include <fog_fragment>\n\t#include <premultiplied_alpha_fragment>\n\t#include <dithering_fragment>\n}",points_vert:"uniform float size;\nuniform float scale;\n#include <common>\n#include <color_pars_vertex>\n#include <fog_pars_vertex>\n#include <morphtarget_pars_vertex>\n#include <logdepthbuf_pars_vertex>\n#include <clipping_planes_pars_vertex>\nvoid main() {\n\t#include <color_vertex>\n\t#include <begin_vertex>\n\t#include <morphtarget_vertex>\n\t#include <project_vertex>\n\tgl_PointSize = size;\n\t#ifdef USE_SIZEATTENUATION\n\t\tbool isPerspective = isPerspectiveMatrix( projectionMatrix );\n\t\tif ( isPerspective ) gl_PointSize *= ( scale / - mvPosition.z );\n\t#endif\n\t#include <logdepthbuf_vertex>\n\t#include <clipping_planes_vertex>\n\t#include <worldpos_vertex>\n\t#include <fog_vertex>\n}",points_frag:"uniform vec3 diffuse;\nuniform float opacity;\n#include <common>\n#include <color_pars_fragment>\n#include <map_particle_pars_fragment>\n#include <alphatest_pars_fragment>\n#include <fog_pars_fragment>\n#include <logdepthbuf_pars_fragment>\n#include <clipping_planes_pars_fragment>\nvoid main() {\n\t#include <clipping_planes_fragment>\n\tvec3 outgoingLight = vec3( 0.0 );\n\tvec4 diffuseColor = vec4( diffuse, opacity );\n\t#include <logdepthbuf_fragment>\n\t#include <map_particle_fragment>\n\t#include <color_fragment>\n\t#include <alphatest_fragment>\n\toutgoingLight = diffuseColor.rgb;\n\t#include <output_fragment>\n\t#include <tonemapping_fragment>\n\t#include <encodings_fragment>\n\t#include <fog_fragment>\n\t#include <premultiplied_alpha_fragment>\n}",shadow_vert:"#include <common>\n#include <fog_pars_vertex>\n#include <morphtarget_pars_vertex>\n#include <skinning_pars_vertex>\n#include <shadowmap_pars_vertex>\nvoid main() {\n\t#include <beginnormal_vertex>\n\t#include <morphnormal_vertex>\n\t#include <skinbase_vertex>\n\t#include <skinnormal_vertex>\n\t#include <defaultnormal_vertex>\n\t#include <begin_vertex>\n\t#include <morphtarget_vertex>\n\t#include <skinning_vertex>\n\t#include <project_vertex>\n\t#include <worldpos_vertex>\n\t#include <shadowmap_vertex>\n\t#include <fog_vertex>\n}",shadow_frag:"uniform vec3 color;\nuniform float opacity;\n#include <common>\n#include <packing>\n#include <fog_pars_fragment>\n#include <bsdfs>\n#include <lights_pars_begin>\n#include <shadowmap_pars_fragment>\n#include <shadowmask_pars_fragment>\nvoid main() {\n\tgl_FragColor = vec4( color, opacity * ( 1.0 - getShadowMask() ) );\n\t#include <tonemapping_fragment>\n\t#include <encodings_fragment>\n\t#include <fog_fragment>\n}",sprite_vert:"uniform float rotation;\nuniform vec2 center;\n#include <common>\n#include <uv_pars_vertex>\n#include <fog_pars_vertex>\n#include <logdepthbuf_pars_vertex>\n#include <clipping_planes_pars_vertex>\nvoid main() {\n\t#include <uv_vertex>\n\tvec4 mvPosition = modelViewMatrix * vec4( 0.0, 0.0, 0.0, 1.0 );\n\tvec2 scale;\n\tscale.x = length( vec3( modelMatrix[ 0 ].x, modelMatrix[ 0 ].y, modelMatrix[ 0 ].z ) );\n\tscale.y = length( vec3( modelMatrix[ 1 ].x, modelMatrix[ 1 ].y, modelMatrix[ 1 ].z ) );\n\t#ifndef USE_SIZEATTENUATION\n\t\tbool isPerspective = isPerspectiveMatrix( projectionMatrix );\n\t\tif ( isPerspective ) scale *= - mvPosition.z;\n\t#endif\n\tvec2 alignedPosition = ( position.xy - ( center - vec2( 0.5 ) ) ) * scale;\n\tvec2 rotatedPosition;\n\trotatedPosition.x = cos( rotation ) * alignedPosition.x - sin( rotation ) * alignedPosition.y;\n\trotatedPosition.y = sin( rotation ) * alignedPosition.x + cos( rotation ) * alignedPosition.y;\n\tmvPosition.xy += rotatedPosition;\n\tgl_Position = projectionMatrix * mvPosition;\n\t#include <logdepthbuf_vertex>\n\t#include <clipping_planes_vertex>\n\t#include <fog_vertex>\n}",sprite_frag:"uniform vec3 diffuse;\nuniform float opacity;\n#include <common>\n#include <uv_pars_fragment>\n#include <map_pars_fragment>\n#include <alphamap_pars_fragment>\n#include <alphatest_pars_fragment>\n#include <fog_pars_fragment>\n#include <logdepthbuf_pars_fragment>\n#include <clipping_planes_pars_fragment>\nvoid main() {\n\t#include <clipping_planes_fragment>\n\tvec3 outgoingLight = vec3( 0.0 );\n\tvec4 diffuseColor = vec4( diffuse, opacity );\n\t#include <logdepthbuf_fragment>\n\t#include <map_fragment>\n\t#include <alphamap_fragment>\n\t#include <alphatest_fragment>\n\toutgoingLight = diffuseColor.rgb;\n\t#include <output_fragment>\n\t#include <tonemapping_fragment>\n\t#include <encodings_fragment>\n\t#include <fog_fragment>\n}"},mi={common:{diffuse:{value:new rn(16777215)},opacity:{value:1},map:{value:null},uvTransform:{value:new xt},uv2Transform:{value:new xt},alphaMap:{value:null},alphaTest:{value:0}},specularmap:{specularMap:{value:null}},envmap:{envMap:{value:null},flipEnvMap:{value:-1},reflectivity:{value:1},ior:{value:1.5},refractionRatio:{value:.98},maxMipLevel:{value:0}},aomap:{aoMap:{value:null},aoMapIntensity:{value:1}},lightmap:{lightMap:{value:null},lightMapIntensity:{value:1}},emissivemap:{emissiveMap:{value:null}},bumpmap:{bumpMap:{value:null},bumpScale:{value:1}},normalmap:{normalMap:{value:null},normalScale:{value:new yt(1,1)}},displacementmap:{displacementMap:{value:null},displacementScale:{value:1},displacementBias:{value:0}},roughnessmap:{roughnessMap:{value:null}},metalnessmap:{metalnessMap:{value:null}},gradientmap:{gradientMap:{value:null}},fog:{fogDensity:{value:25e-5},fogNear:{value:1},fogFar:{value:2e3},fogColor:{value:new rn(16777215)}},lights:{ambientLightColor:{value:[]},lightProbe:{value:[]},directionalLights:{value:[],properties:{direction:{},color:{}}},directionalLightShadows:{value:[],properties:{shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},directionalShadowMap:{value:[]},directionalShadowMatrix:{value:[]},spotLights:{value:[],properties:{color:{},position:{},direction:{},distance:{},coneCos:{},penumbraCos:{},decay:{}}},spotLightShadows:{value:[],properties:{shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},spotShadowMap:{value:[]},spotShadowMatrix:{value:[]},pointLights:{value:[],properties:{color:{},position:{},decay:{},distance:{}}},pointLightShadows:{value:[],properties:{shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{},shadowCameraNear:{},shadowCameraFar:{}}},pointShadowMap:{value:[]},pointShadowMatrix:{value:[]},hemisphereLights:{value:[],properties:{direction:{},skyColor:{},groundColor:{}}},rectAreaLights:{value:[],properties:{color:{},position:{},width:{},height:{}}},ltc_1:{value:null},ltc_2:{value:null}},points:{diffuse:{value:new rn(16777215)},opacity:{value:1},size:{value:1},scale:{value:1},map:{value:null},alphaMap:{value:null},alphaTest:{value:0},uvTransform:{value:new xt}},sprite:{diffuse:{value:new rn(16777215)},opacity:{value:1},center:{value:new yt(.5,.5)},rotation:{value:0},map:{value:null},alphaMap:{value:null},alphaTest:{value:0},uvTransform:{value:new xt}}},fi={basic:{uniforms:Yn([mi.common,mi.specularmap,mi.envmap,mi.aomap,mi.lightmap,mi.fog]),vertexShader:pi.meshbasic_vert,fragmentShader:pi.meshbasic_frag},lambert:{uniforms:Yn([mi.common,mi.specularmap,mi.envmap,mi.aomap,mi.lightmap,mi.emissivemap,mi.fog,mi.lights,{emissive:{value:new rn(0)}}]),vertexShader:pi.meshlambert_vert,fragmentShader:pi.meshlambert_frag},phong:{uniforms:Yn([mi.common,mi.specularmap,mi.envmap,mi.aomap,mi.lightmap,mi.emissivemap,mi.bumpmap,mi.normalmap,mi.displacementmap,mi.fog,mi.lights,{emissive:{value:new rn(0)},specular:{value:new rn(1118481)},shininess:{value:30}}]),vertexShader:pi.meshphong_vert,fragmentShader:pi.meshphong_frag},standard:{uniforms:Yn([mi.common,mi.envmap,mi.aomap,mi.lightmap,mi.emissivemap,mi.bumpmap,mi.normalmap,mi.displacementmap,mi.roughnessmap,mi.metalnessmap,mi.fog,mi.lights,{emissive:{value:new rn(0)},roughness:{value:1},metalness:{value:0},envMapIntensity:{value:1}}]),vertexShader:pi.meshphysical_vert,fragmentShader:pi.meshphysical_frag},toon:{uniforms:Yn([mi.common,mi.aomap,mi.lightmap,mi.emissivemap,mi.bumpmap,mi.normalmap,mi.displacementmap,mi.gradientmap,mi.fog,mi.lights,{emissive:{value:new rn(0)}}]),vertexShader:pi.meshtoon_vert,fragmentShader:pi.meshtoon_frag},matcap:{uniforms:Yn([mi.common,mi.bumpmap,mi.normalmap,mi.displacementmap,mi.fog,{matcap:{value:null}}]),vertexShader:pi.meshmatcap_vert,fragmentShader:pi.meshmatcap_frag},points:{uniforms:Yn([mi.points,mi.fog]),vertexShader:pi.points_vert,fragmentShader:pi.points_frag},dashed:{uniforms:Yn([mi.common,mi.fog,{scale:{value:1},dashSize:{value:1},totalSize:{value:2}}]),vertexShader:pi.linedashed_vert,fragmentShader:pi.linedashed_frag},depth:{uniforms:Yn([mi.common,mi.displacementmap]),vertexShader:pi.depth_vert,fragmentShader:pi.depth_frag},normal:{uniforms:Yn([mi.common,mi.bumpmap,mi.normalmap,mi.displacementmap,{opacity:{value:1}}]),vertexShader:pi.meshnormal_vert,fragmentShader:pi.meshnormal_frag},sprite:{uniforms:Yn([mi.sprite,mi.fog]),vertexShader:pi.sprite_vert,fragmentShader:pi.sprite_frag},background:{uniforms:{uvTransform:{value:new xt},t2D:{value:null}},vertexShader:pi.background_vert,fragmentShader:pi.background_frag},cube:{uniforms:Yn([mi.envmap,{opacity:{value:1}}]),vertexShader:pi.cube_vert,fragmentShader:pi.cube_frag},equirect:{uniforms:{tEquirect:{value:null}},vertexShader:pi.equirect_vert,fragmentShader:pi.equirect_frag},distanceRGBA:{uniforms:Yn([mi.common,mi.displacementmap,{referencePosition:{value:new zt},nearDistance:{value:1},farDistance:{value:1e3}}]),vertexShader:pi.distanceRGBA_vert,fragmentShader:pi.distanceRGBA_frag},shadow:{uniforms:Yn([mi.lights,mi.fog,{color:{value:new rn(0)},opacity:{value:1}}]),vertexShader:pi.shadow_vert,fragmentShader:pi.shadow_frag}};function gi(t,e,n,i,r){const s=new rn(0);let a,o,c=0,h=null,u=0,d=null;function p(t,e){n.buffers.color.setClear(t.r,t.g,t.b,e,r)}return{getClearColor:function(){return s},setClearColor:function(t,e=1){s.set(t),c=e,p(s,c)},getClearAlpha:function(){return c},setClearAlpha:function(t){c=t,p(s,c)},render:function(n,r){let m=!1,f=!0===r.isScene?r.background:null;f&&f.isTexture&&(f=e.get(f));const g=t.xr,v=g.getSession&&g.getSession();v&&"additive"===v.environmentBlendMode&&(f=null),null===f?p(s,c):f&&f.isColor&&(p(f,1),m=!0),(t.autoClear||m)&&t.clear(t.autoClearColor,t.autoClearDepth,t.autoClearStencil),f&&(f.isCubeTexture||f.mapping===l)?(void 0===o&&(o=new Wn(new qn(1,1,1),new Zn({name:"BackgroundCubeMaterial",uniforms:Xn(fi.cube.uniforms),vertexShader:fi.cube.vertexShader,fragmentShader:fi.cube.fragmentShader,side:1,depthTest:!1,depthWrite:!1,fog:!1})),o.geometry.deleteAttribute("normal"),o.geometry.deleteAttribute("uv"),o.onBeforeRender=function(t,e,n){this.matrixWorld.copyPosition(n.matrixWorld)},Object.defineProperty(o.material,"envMap",{get:function(){return this.uniforms.envMap.value}}),i.update(o)),o.material.uniforms.envMap.value=f,o.material.uniforms.flipEnvMap.value=f.isCubeTexture&&!1===f.isRenderTargetTexture?-1:1,h===f&&u===f.version&&d===t.toneMapping||(o.material.needsUpdate=!0,h=f,u=f.version,d=t.toneMapping),n.unshift(o,o.geometry,o.material,0,0,null)):f&&f.isTexture&&(void 0===a&&(a=new Wn(new di(2,2),new Zn({name:"BackgroundMaterial",uniforms:Xn(fi.background.uniforms),vertexShader:fi.background.vertexShader,fragmentShader:fi.background.fragmentShader,side:0,depthTest:!1,depthWrite:!1,fog:!1})),a.geometry.deleteAttribute("normal"),Object.defineProperty(a.material,"map",{get:function(){return this.uniforms.t2D.value}}),i.update(a)),a.material.uniforms.t2D.value=f,!0===f.matrixAutoUpdate&&f.updateMatrix(),a.material.uniforms.uvTransform.value.copy(f.matrix),h===f&&u===f.version&&d===t.toneMapping||(a.material.needsUpdate=!0,h=f,u=f.version,d=t.toneMapping),n.unshift(a,a.geometry,a.material,0,0,null))}}}function vi(t,e,n,i){const r=t.getParameter(34921),s=i.isWebGL2?null:e.get("OES_vertex_array_object"),a=i.isWebGL2||null!==s,o={},l=d(null);let c=l;function h(e){return i.isWebGL2?t.bindVertexArray(e):s.bindVertexArrayOES(e)}function u(e){return i.isWebGL2?t.deleteVertexArray(e):s.deleteVertexArrayOES(e)}function d(t){const e=[],n=[],i=[];for(let t=0;t<r;t++)e[t]=0,n[t]=0,i[t]=0;return{geometry:null,program:null,wireframe:!1,newAttributes:e,enabledAttributes:n,attributeDivisors:i,object:t,attributes:{},index:null}}function p(){const t=c.newAttributes;for(let e=0,n=t.length;e<n;e++)t[e]=0}function m(t){f(t,0)}function f(n,r){const s=c.newAttributes,a=c.enabledAttributes,o=c.attributeDivisors;if(s[n]=1,0===a[n]&&(t.enableVertexAttribArray(n),a[n]=1),o[n]!==r){(i.isWebGL2?t:e.get("ANGLE_instanced_arrays"))[i.isWebGL2?"vertexAttribDivisor":"vertexAttribDivisorANGLE"](n,r),o[n]=r}}function g(){const e=c.newAttributes,n=c.enabledAttributes;for(let i=0,r=n.length;i<r;i++)n[i]!==e[i]&&(t.disableVertexAttribArray(i),n[i]=0)}function v(e,n,r,s,a,o){!0!==i.isWebGL2||5124!==r&&5125!==r?t.vertexAttribPointer(e,n,r,s,a,o):t.vertexAttribIPointer(e,n,r,a,o)}function y(){x(),c!==l&&(c=l,h(c.object))}function x(){l.geometry=null,l.program=null,l.wireframe=!1}return{setup:function(r,l,u,y,x){let _=!1;if(a){const e=function(e,n,r){const a=!0===r.wireframe;let l=o[e.id];void 0===l&&(l={},o[e.id]=l);let c=l[n.id];void 0===c&&(c={},l[n.id]=c);let h=c[a];void 0===h&&(h=d(i.isWebGL2?t.createVertexArray():s.createVertexArrayOES()),c[a]=h);return h}(y,u,l);c!==e&&(c=e,h(c.object)),_=function(t,e){const n=c.attributes,i=t.attributes;let r=0;for(const t in i){const e=n[t],s=i[t];if(void 0===e)return!0;if(e.attribute!==s)return!0;if(e.data!==s.data)return!0;r++}return c.attributesNum!==r||c.index!==e}(y,x),_&&function(t,e){const n={},i=t.attributes;let r=0;for(const t in i){const e=i[t],s={};s.attribute=e,e.data&&(s.data=e.data),n[t]=s,r++}c.attributes=n,c.attributesNum=r,c.index=e}(y,x)}else{const t=!0===l.wireframe;c.geometry===y.id&&c.program===u.id&&c.wireframe===t||(c.geometry=y.id,c.program=u.id,c.wireframe=t,_=!0)}!0===r.isInstancedMesh&&(_=!0),null!==x&&n.update(x,34963),_&&(!function(r,s,a,o){if(!1===i.isWebGL2&&(r.isInstancedMesh||o.isInstancedBufferGeometry)&&null===e.get("ANGLE_instanced_arrays"))return;p();const l=o.attributes,c=a.getAttributes(),h=s.defaultAttributeValues;for(const e in c){const i=c[e];if(i.location>=0){let s=l[e];if(void 0===s&&("instanceMatrix"===e&&r.instanceMatrix&&(s=r.instanceMatrix),"instanceColor"===e&&r.instanceColor&&(s=r.instanceColor)),void 0!==s){const e=s.normalized,a=s.itemSize,l=n.get(s);if(void 0===l)continue;const c=l.buffer,h=l.type,u=l.bytesPerElement;if(s.isInterleavedBufferAttribute){const n=s.data,l=n.stride,d=s.offset;if(n&&n.isInstancedInterleavedBuffer){for(let t=0;t<i.locationSize;t++)f(i.location+t,n.meshPerAttribute);!0!==r.isInstancedMesh&&void 0===o._maxInstanceCount&&(o._maxInstanceCount=n.meshPerAttribute*n.count)}else for(let t=0;t<i.locationSize;t++)m(i.location+t);t.bindBuffer(34962,c);for(let t=0;t<i.locationSize;t++)v(i.location+t,a/i.locationSize,h,e,l*u,(d+a/i.locationSize*t)*u)}else{if(s.isInstancedBufferAttribute){for(let t=0;t<i.locationSize;t++)f(i.location+t,s.meshPerAttribute);!0!==r.isInstancedMesh&&void 0===o._maxInstanceCount&&(o._maxInstanceCount=s.meshPerAttribute*s.count)}else for(let t=0;t<i.locationSize;t++)m(i.location+t);t.bindBuffer(34962,c);for(let t=0;t<i.locationSize;t++)v(i.location+t,a/i.locationSize,h,e,a*u,a/i.locationSize*t*u)}}else if(void 0!==h){const n=h[e];if(void 0!==n)switch(n.length){case 2:t.vertexAttrib2fv(i.location,n);break;case 3:t.vertexAttrib3fv(i.location,n);break;case 4:t.vertexAttrib4fv(i.location,n);break;default:t.vertexAttrib1fv(i.location,n)}}}}g()}(r,l,u,y),null!==x&&t.bindBuffer(34963,n.get(x).buffer))},reset:y,resetDefaultState:x,dispose:function(){y();for(const t in o){const e=o[t];for(const t in e){const n=e[t];for(const t in n)u(n[t].object),delete n[t];delete e[t]}delete o[t]}},releaseStatesOfGeometry:function(t){if(void 0===o[t.id])return;const e=o[t.id];for(const t in e){const n=e[t];for(const t in n)u(n[t].object),delete n[t];delete e[t]}delete o[t.id]},releaseStatesOfProgram:function(t){for(const e in o){const n=o[e];if(void 0===n[t.id])continue;const i=n[t.id];for(const t in i)u(i[t].object),delete i[t];delete n[t.id]}},initAttributes:p,enableAttribute:m,disableUnusedAttributes:g}}function yi(t,e,n,i){const r=i.isWebGL2;let s;this.setMode=function(t){s=t},this.render=function(e,i){t.drawArrays(s,e,i),n.update(i,s,1)},this.renderInstances=function(i,a,o){if(0===o)return;let l,c;if(r)l=t,c="drawArraysInstanced";else if(l=e.get("ANGLE_instanced_arrays"),c="drawArraysInstancedANGLE",null===l)return void console.error("THREE.WebGLBufferRenderer: using THREE.InstancedBufferGeometry but hardware does not support extension ANGLE_instanced_arrays.");l[c](s,i,a,o),n.update(a,s,o)}}function xi(t,e,n){let i;function r(e){if("highp"===e){if(t.getShaderPrecisionFormat(35633,36338).precision>0&&t.getShaderPrecisionFormat(35632,36338).precision>0)return"highp";e="mediump"}return"mediump"===e&&t.getShaderPrecisionFormat(35633,36337).precision>0&&t.getShaderPrecisionFormat(35632,36337).precision>0?"mediump":"lowp"}const s="undefined"!=typeof WebGL2RenderingContext&&t instanceof WebGL2RenderingContext||"undefined"!=typeof WebGL2ComputeRenderingContext&&t instanceof WebGL2ComputeRenderingContext;let a=void 0!==n.precision?n.precision:"highp";const o=r(a);o!==a&&(console.warn("THREE.WebGLRenderer:",a,"not supported, using",o,"instead."),a=o);const l=s||e.has("WEBGL_draw_buffers"),c=!0===n.logarithmicDepthBuffer,h=t.getParameter(34930),u=t.getParameter(35660),d=t.getParameter(3379),p=t.getParameter(34076),m=t.getParameter(34921),f=t.getParameter(36347),g=t.getParameter(36348),v=t.getParameter(36349),y=u>0,x=s||e.has("OES_texture_float");return{isWebGL2:s,drawBuffers:l,getMaxAnisotropy:function(){if(void 0!==i)return i;if(!0===e.has("EXT_texture_filter_anisotropic")){const n=e.get("EXT_texture_filter_anisotropic");i=t.getParameter(n.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else i=0;return i},getMaxPrecision:r,precision:a,logarithmicDepthBuffer:c,maxTextures:h,maxVertexTextures:u,maxTextureSize:d,maxCubemapSize:p,maxAttributes:m,maxVertexUniforms:f,maxVaryings:g,maxFragmentUniforms:v,vertexTextures:y,floatFragmentTextures:x,floatVertexTextures:y&&x,maxSamples:s?t.getParameter(36183):0}}function _i(t){const e=this;let n=null,i=0,r=!1,s=!1;const a=new ai,o=new xt,l={value:null,needsUpdate:!1};function c(){l.value!==n&&(l.value=n,l.needsUpdate=i>0),e.numPlanes=i,e.numIntersection=0}function h(t,n,i,r){const s=null!==t?t.length:0;let c=null;if(0!==s){if(c=l.value,!0!==r||null===c){const e=i+4*s,r=n.matrixWorldInverse;o.getNormalMatrix(r),(null===c||c.length<e)&&(c=new Float32Array(e));for(let e=0,n=i;e!==s;++e,n+=4)a.copy(t[e]).applyMatrix4(r,o),a.normal.toArray(c,n),c[n+3]=a.constant}l.value=c,l.needsUpdate=!0}return e.numPlanes=s,e.numIntersection=0,c}this.uniform=l,this.numPlanes=0,this.numIntersection=0,this.init=function(t,e,s){const a=0!==t.length||e||0!==i||r;return r=e,n=h(t,s,0),i=t.length,a},this.beginShadows=function(){s=!0,h(null)},this.endShadows=function(){s=!1,c()},this.setState=function(e,a,o){const u=e.clippingPlanes,d=e.clipIntersection,p=e.clipShadows,m=t.get(e);if(!r||null===u||0===u.length||s&&!p)s?h(null):c();else{const t=s?0:i,e=4*t;let r=m.clippingState||null;l.value=r,r=h(u,a,e,o);for(let t=0;t!==e;++t)r[t]=n[t];m.clippingState=r,this.numIntersection=d?this.numPlanes:0,this.numPlanes+=t}}}function Mi(t){let e=new WeakMap;function n(t,e){return e===a?t.mapping=r:e===o&&(t.mapping=s),t}function i(t){const n=t.target;n.removeEventListener("dispose",i);const r=e.get(n);void 0!==r&&(e.delete(n),r.dispose())}return{get:function(r){if(r&&r.isTexture&&!1===r.isRenderTargetTexture){const s=r.mapping;if(s===a||s===o){if(e.has(r)){return n(e.get(r).texture,r.mapping)}{const s=r.image;if(s&&s.height>0){const a=t.getRenderTarget(),o=new ni(s.height/2);return o.fromEquirectangularTexture(t,r),e.set(r,o),t.setRenderTarget(a),r.addEventListener("dispose",i),n(o.texture,r.mapping)}return null}}}return r},dispose:function(){e=new WeakMap}}}fi.physical={uniforms:Yn([fi.standard.uniforms,{clearcoat:{value:0},clearcoatMap:{value:null},clearcoatRoughness:{value:0},clearcoatRoughnessMap:{value:null},clearcoatNormalScale:{value:new yt(1,1)},clearcoatNormalMap:{value:null},sheen:{value:0},sheenColor:{value:new rn(0)},sheenColorMap:{value:null},sheenRoughness:{value:0},sheenRoughnessMap:{value:null},transmission:{value:0},transmissionMap:{value:null},transmissionSamplerSize:{value:new yt},transmissionSamplerMap:{value:null},thickness:{value:0},thicknessMap:{value:null},attenuationDistance:{value:0},attenuationColor:{value:new rn(0)},specularIntensity:{value:0},specularIntensityMap:{value:null},specularColor:{value:new rn(1,1,1)},specularColorMap:{value:null}}]),vertexShader:pi.meshphysical_vert,fragmentShader:pi.meshphysical_frag};class bi extends Qn{constructor(t=-1,e=1,n=1,i=-1,r=.1,s=2e3){super(),this.type="OrthographicCamera",this.zoom=1,this.view=null,this.left=t,this.right=e,this.top=n,this.bottom=i,this.near=r,this.far=s,this.updateProjectionMatrix()}copy(t,e){return super.copy(t,e),this.left=t.left,this.right=t.right,this.top=t.top,this.bottom=t.bottom,this.near=t.near,this.far=t.far,this.zoom=t.zoom,this.view=null===t.view?null:Object.assign({},t.view),this}setViewOffset(t,e,n,i,r,s){null===this.view&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=t,this.view.fullHeight=e,this.view.offsetX=n,this.view.offsetY=i,this.view.width=r,this.view.height=s,this.updateProjectionMatrix()}clearViewOffset(){null!==this.view&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const t=(this.right-this.left)/(2*this.zoom),e=(this.top-this.bottom)/(2*this.zoom),n=(this.right+this.left)/2,i=(this.top+this.bottom)/2;let r=n-t,s=n+t,a=i+e,o=i-e;if(null!==this.view&&this.view.enabled){const t=(this.right-this.left)/this.view.fullWidth/this.zoom,e=(this.top-this.bottom)/this.view.fullHeight/this.zoom;r+=t*this.view.offsetX,s=r+t*this.view.width,a-=e*this.view.offsetY,o=a-e*this.view.height}this.projectionMatrix.makeOrthographic(r,s,a,o,this.near,this.far),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(t){const e=super.toJSON(t);return e.object.zoom=this.zoom,e.object.left=this.left,e.object.right=this.right,e.object.top=this.top,e.object.bottom=this.bottom,e.object.near=this.near,e.object.far=this.far,null!==this.view&&(e.object.view=Object.assign({},this.view)),e}}bi.prototype.isOrthographicCamera=!0;class wi extends Zn{constructor(t){super(t),this.type="RawShaderMaterial"}}wi.prototype.isRawShaderMaterial=!0;const Si=Math.pow(2,8),Ti=[.125,.215,.35,.446,.526,.582],Ei=5+Ti.length,Ai=20,Li={[X]:0,[Y]:1,[Z]:2,[Q]:3,[K]:4,[$]:5,[J]:6},Ri=new bi,{_lodPlanes:Ci,_sizeLods:Pi,_sigmas:Ii}=Hi(),Di=new rn;let Ni=null;const zi=(1+Math.sqrt(5))/2,Bi=1/zi,Fi=[new zt(1,1,1),new zt(-1,1,1),new zt(1,1,-1),new zt(-1,1,-1),new zt(0,zi,Bi),new zt(0,zi,-Bi),new zt(Bi,0,zi),new zt(-Bi,0,zi),new zt(zi,Bi,0),new zt(-zi,Bi,0)];class Oi{constructor(t){this._renderer=t,this._pingPongRenderTarget=null,this._blurMaterial=function(t){const e=new Float32Array(t),n=new zt(0,1,0);return new wi({name:"SphericalGaussianBlur",defines:{n:t},uniforms:{envMap:{value:null},samples:{value:1},weights:{value:e},latitudinal:{value:!1},dTheta:{value:0},mipInt:{value:0},poleAxis:{value:n},inputEncoding:{value:Li[3e3]},outputEncoding:{value:Li[3e3]}},vertexShader:ji(),fragmentShader:`\n\n\t\t\tprecision mediump float;\n\t\t\tprecision mediump int;\n\n\t\t\tvarying vec3 vOutputDirection;\n\n\t\t\tuniform sampler2D envMap;\n\t\t\tuniform int samples;\n\t\t\tuniform float weights[ n ];\n\t\t\tuniform bool latitudinal;\n\t\t\tuniform float dTheta;\n\t\t\tuniform float mipInt;\n\t\t\tuniform vec3 poleAxis;\n\n\t\t\t${qi()}\n\n\t\t\t#define ENVMAP_TYPE_CUBE_UV\n\t\t\t#include <cube_uv_reflection_fragment>\n\n\t\t\tvec3 getSample( float theta, vec3 axis ) {\n\n\t\t\t\tfloat cosTheta = cos( theta );\n\t\t\t\t// Rodrigues' axis-angle rotation\n\t\t\t\tvec3 sampleDirection = vOutputDirection * cosTheta\n\t\t\t\t\t+ cross( axis, vOutputDirection ) * sin( theta )\n\t\t\t\t\t+ axis * dot( axis, vOutputDirection ) * ( 1.0 - cosTheta );\n\n\t\t\t\treturn bilinearCubeUV( envMap, sampleDirection, mipInt );\n\n\t\t\t}\n\n\t\t\tvoid main() {\n\n\t\t\t\tvec3 axis = latitudinal ? poleAxis : cross( poleAxis, vOutputDirection );\n\n\t\t\t\tif ( all( equal( axis, vec3( 0.0 ) ) ) ) {\n\n\t\t\t\t\taxis = vec3( vOutputDirection.z, 0.0, - vOutputDirection.x );\n\n\t\t\t\t}\n\n\t\t\t\taxis = normalize( axis );\n\n\t\t\t\tgl_FragColor = vec4( 0.0, 0.0, 0.0, 1.0 );\n\t\t\t\tgl_FragColor.rgb += weights[ 0 ] * getSample( 0.0, axis );\n\n\t\t\t\tfor ( int i = 1; i < n; i++ ) {\n\n\t\t\t\t\tif ( i >= samples ) {\n\n\t\t\t\t\t\tbreak;\n\n\t\t\t\t\t}\n\n\t\t\t\t\tfloat theta = dTheta * float( i );\n\t\t\t\t\tgl_FragColor.rgb += weights[ i ] * getSample( -1.0 * theta, axis );\n\t\t\t\t\tgl_FragColor.rgb += weights[ i ] * getSample( theta, axis );\n\n\t\t\t\t}\n\n\t\t\t\tgl_FragColor = linearToOutputTexel( gl_FragColor );\n\n\t\t\t}\n\t\t`,blending:0,depthTest:!1,depthWrite:!1})}(Ai),this._equirectShader=null,this._cubemapShader=null,this._compileMaterial(this._blurMaterial)}fromScene(t,e=0,n=.1,i=100){Ni=this._renderer.getRenderTarget();const r=this._allocateTargets();return this._sceneToCubeUV(t,n,i,r),e>0&&this._blur(r,0,0,e),this._applyPMREM(r),this._cleanup(r),r}fromEquirectangular(t){return this._fromTexture(t)}fromCubemap(t){return this._fromTexture(t)}compileCubemapShader(){null===this._cubemapShader&&(this._cubemapShader=Wi(),this._compileMaterial(this._cubemapShader))}compileEquirectangularShader(){null===this._equirectShader&&(this._equirectShader=Vi(),this._compileMaterial(this._equirectShader))}dispose(){this._blurMaterial.dispose(),null!==this._cubemapShader&&this._cubemapShader.dispose(),null!==this._equirectShader&&this._equirectShader.dispose();for(let t=0;t<Ci.length;t++)Ci[t].dispose()}_cleanup(t){this._pingPongRenderTarget.dispose(),this._renderer.setRenderTarget(Ni),t.scissorTest=!1,ki(t,0,0,t.width,t.height)}_fromTexture(t){Ni=this._renderer.getRenderTarget();const e=this._allocateTargets(t);return this._textureToCubeUV(t,e),this._applyPMREM(e),this._cleanup(e),e}_allocateTargets(t){const e={magFilter:p,minFilter:p,generateMipmaps:!1,type:x,format:1023,encoding:Ui(t)?t.encoding:Z,depthBuffer:!1},n=Gi(e);return n.depthBuffer=!t,this._pingPongRenderTarget=Gi(e),n}_compileMaterial(t){const e=new Wn(Ci[0],t);this._renderer.compile(e,Ri)}_sceneToCubeUV(t,e,n,i){const r=new Kn(90,1,e,n),s=[1,-1,1,1,1,1],a=[1,1,1,-1,-1,-1],o=this._renderer,l=o.autoClear,c=o.outputEncoding,h=o.toneMapping;o.getClearColor(Di),o.toneMapping=0,o.outputEncoding=X,o.autoClear=!1;const u=new sn({name:"PMREM.Background",side:1,depthWrite:!1,depthTest:!1}),d=new Wn(new qn,u);let p=!1;const m=t.background;m?m.isColor&&(u.color.copy(m),t.background=null,p=!0):(u.color.copy(Di),p=!0);for(let e=0;e<6;e++){const n=e%3;0==n?(r.up.set(0,s[e],0),r.lookAt(a[e],0,0)):1==n?(r.up.set(0,0,s[e]),r.lookAt(0,a[e],0)):(r.up.set(0,s[e],0),r.lookAt(0,0,a[e])),ki(i,n*Si,e>2?Si:0,Si,Si),o.setRenderTarget(i),p&&o.render(d,r),o.render(t,r)}d.geometry.dispose(),d.material.dispose(),o.toneMapping=h,o.outputEncoding=c,o.autoClear=l,t.background=m}_setEncoding(t,e){!0===this._renderer.capabilities.isWebGL2&&e.format===E&&e.type===x&&e.encoding===Y?t.value=Li[3e3]:t.value=Li[e.encoding]}_textureToCubeUV(t,e){const n=this._renderer,i=t.mapping===r||t.mapping===s;i?null==this._cubemapShader&&(this._cubemapShader=Wi()):null==this._equirectShader&&(this._equirectShader=Vi());const a=i?this._cubemapShader:this._equirectShader,o=new Wn(Ci[0],a),l=a.uniforms;l.envMap.value=t,i||l.texelSize.value.set(1/t.image.width,1/t.image.height),this._setEncoding(l.inputEncoding,t),this._setEncoding(l.outputEncoding,e.texture),ki(e,0,0,3*Si,2*Si),n.setRenderTarget(e),n.render(o,Ri)}_applyPMREM(t){const e=this._renderer,n=e.autoClear;e.autoClear=!1;for(let e=1;e<Ei;e++){const n=Math.sqrt(Ii[e]*Ii[e]-Ii[e-1]*Ii[e-1]),i=Fi[(e-1)%Fi.length];this._blur(t,e-1,e,n,i)}e.autoClear=n}_blur(t,e,n,i,r){const s=this._pingPongRenderTarget;this._halfBlur(t,s,e,n,i,"latitudinal",r),this._halfBlur(s,t,n,n,i,"longitudinal",r)}_halfBlur(t,e,n,i,r,s,a){const o=this._renderer,l=this._blurMaterial;"latitudinal"!==s&&"longitudinal"!==s&&console.error("blur direction must be either latitudinal or longitudinal!");const c=new Wn(Ci[i],l),h=l.uniforms,u=Pi[n]-1,d=isFinite(r)?Math.PI/(2*u):2*Math.PI/39,p=r/d,m=isFinite(r)?1+Math.floor(3*p):Ai;m>Ai&&console.warn(`sigmaRadians, ${r}, is too large and will clip, as it requested ${m} samples when the maximum is set to 20`);const f=[];let g=0;for(let t=0;t<Ai;++t){const e=t/p,n=Math.exp(-e*e/2);f.push(n),0==t?g+=n:t<m&&(g+=2*n)}for(let t=0;t<f.length;t++)f[t]=f[t]/g;h.envMap.value=t.texture,h.samples.value=m,h.weights.value=f,h.latitudinal.value="latitudinal"===s,a&&(h.poleAxis.value=a),h.dTheta.value=d,h.mipInt.value=8-n,this._setEncoding(h.inputEncoding,t.texture),this._setEncoding(h.outputEncoding,t.texture);const v=Pi[i];ki(e,3*Math.max(0,Si-2*v),(0===i?0:2*Si)+2*v*(i>4?i-8+4:0),3*v,2*v),o.setRenderTarget(e),o.render(c,Ri)}}function Ui(t){return void 0!==t&&t.type===x&&(t.encoding===X||t.encoding===Y||t.encoding===J)}function Hi(){const t=[],e=[],n=[];let i=8;for(let r=0;r<Ei;r++){const s=Math.pow(2,i);e.push(s);let a=1/s;r>4?a=Ti[r-8+4-1]:0==r&&(a=0),n.push(a);const o=1/(s-1),l=-o/2,c=1+o/2,h=[l,l,c,l,c,c,l,l,c,c,l,c],u=6,d=6,p=3,m=2,f=1,g=new Float32Array(p*d*u),v=new Float32Array(m*d*u),y=new Float32Array(f*d*u);for(let t=0;t<u;t++){const e=t%3*2/3-1,n=t>2?0:-1,i=[e,n,0,e+2/3,n,0,e+2/3,n+1,0,e,n,0,e+2/3,n+1,0,e,n+1,0];g.set(i,p*d*t),v.set(h,m*d*t);const r=[t,t,t,t,t,t];y.set(r,f*d*t)}const x=new En;x.setAttribute("position",new ln(g,p)),x.setAttribute("uv",new ln(v,m)),x.setAttribute("faceIndex",new ln(y,f)),t.push(x),i>4&&i--}return{_lodPlanes:t,_sizeLods:e,_sigmas:n}}function Gi(t){const e=new Pt(3*Si,3*Si,t);return e.texture.mapping=l,e.texture.name="PMREM.cubeUv",e.scissorTest=!0,e}function ki(t,e,n,i,r){t.viewport.set(e,n,i,r),t.scissor.set(e,n,i,r)}function Vi(){const t=new yt(1,1);return new wi({name:"EquirectangularToCubeUV",uniforms:{envMap:{value:null},texelSize:{value:t},inputEncoding:{value:Li[3e3]},outputEncoding:{value:Li[3e3]}},vertexShader:ji(),fragmentShader:`\n\n\t\t\tprecision mediump float;\n\t\t\tprecision mediump int;\n\n\t\t\tvarying vec3 vOutputDirection;\n\n\t\t\tuniform sampler2D envMap;\n\t\t\tuniform vec2 texelSize;\n\n\t\t\t${qi()}\n\n\t\t\t#include <common>\n\n\t\t\tvoid main() {\n\n\t\t\t\tgl_FragColor = vec4( 0.0, 0.0, 0.0, 1.0 );\n\n\t\t\t\tvec3 outputDirection = normalize( vOutputDirection );\n\t\t\t\tvec2 uv = equirectUv( outputDirection );\n\n\t\t\t\tvec2 f = fract( uv / texelSize - 0.5 );\n\t\t\t\tuv -= f * texelSize;\n\t\t\t\tvec3 tl = envMapTexelToLinear( texture2D ( envMap, uv ) ).rgb;\n\t\t\t\tuv.x += texelSize.x;\n\t\t\t\tvec3 tr = envMapTexelToLinear( texture2D ( envMap, uv ) ).rgb;\n\t\t\t\tuv.y += texelSize.y;\n\t\t\t\tvec3 br = envMapTexelToLinear( texture2D ( envMap, uv ) ).rgb;\n\t\t\t\tuv.x -= texelSize.x;\n\t\t\t\tvec3 bl = envMapTexelToLinear( texture2D ( envMap, uv ) ).rgb;\n\n\t\t\t\tvec3 tm = mix( tl, tr, f.x );\n\t\t\t\tvec3 bm = mix( bl, br, f.x );\n\t\t\t\tgl_FragColor.rgb = mix( tm, bm, f.y );\n\n\t\t\t\tgl_FragColor = linearToOutputTexel( gl_FragColor );\n\n\t\t\t}\n\t\t`,blending:0,depthTest:!1,depthWrite:!1})}function Wi(){return new wi({name:"CubemapToCubeUV",uniforms:{envMap:{value:null},inputEncoding:{value:Li[3e3]},outputEncoding:{value:Li[3e3]}},vertexShader:ji(),fragmentShader:`\n\n\t\t\tprecision mediump float;\n\t\t\tprecision mediump int;\n\n\t\t\tvarying vec3 vOutputDirection;\n\n\t\t\tuniform samplerCube envMap;\n\n\t\t\t${qi()}\n\n\t\t\tvoid main() {\n\n\t\t\t\tgl_FragColor = vec4( 0.0, 0.0, 0.0, 1.0 );\n\t\t\t\tgl_FragColor.rgb = envMapTexelToLinear( textureCube( envMap, vec3( - vOutputDirection.x, vOutputDirection.yz ) ) ).rgb;\n\t\t\t\tgl_FragColor = linearToOutputTexel( gl_FragColor );\n\n\t\t\t}\n\t\t`,blending:0,depthTest:!1,depthWrite:!1})}function ji(){return"\n\n\t\tprecision mediump float;\n\t\tprecision mediump int;\n\n\t\tattribute vec3 position;\n\t\tattribute vec2 uv;\n\t\tattribute float faceIndex;\n\n\t\tvarying vec3 vOutputDirection;\n\n\t\t// RH coordinate system; PMREM face-indexing convention\n\t\tvec3 getDirection( vec2 uv, float face ) {\n\n\t\t\tuv = 2.0 * uv - 1.0;\n\n\t\t\tvec3 direction = vec3( uv, 1.0 );\n\n\t\t\tif ( face == 0.0 ) {\n\n\t\t\t\tdirection = direction.zyx; // ( 1, v, u ) pos x\n\n\t\t\t} else if ( face == 1.0 ) {\n\n\t\t\t\tdirection = direction.xzy;\n\t\t\t\tdirection.xz *= -1.0; // ( -u, 1, -v ) pos y\n\n\t\t\t} else if ( face == 2.0 ) {\n\n\t\t\t\tdirection.x *= -1.0; // ( -u, v, 1 ) pos z\n\n\t\t\t} else if ( face == 3.0 ) {\n\n\t\t\t\tdirection = direction.zyx;\n\t\t\t\tdirection.xz *= -1.0; // ( -1, v, -u ) neg x\n\n\t\t\t} else if ( face == 4.0 ) {\n\n\t\t\t\tdirection = direction.xzy;\n\t\t\t\tdirection.xy *= -1.0; // ( -u, -1, v ) neg y\n\n\t\t\t} else if ( face == 5.0 ) {\n\n\t\t\t\tdirection.z *= -1.0; // ( u, v, -1 ) neg z\n\n\t\t\t}\n\n\t\t\treturn direction;\n\n\t\t}\n\n\t\tvoid main() {\n\n\t\t\tvOutputDirection = getDirection( uv, faceIndex );\n\t\t\tgl_Position = vec4( position, 1.0 );\n\n\t\t}\n\t"}function qi(){return"\n\n\t\tuniform int inputEncoding;\n\t\tuniform int outputEncoding;\n\n\t\t#include <encodings_pars_fragment>\n\n\t\tvec4 inputTexelToLinear( vec4 value ) {\n\n\t\t\tif ( inputEncoding == 0 ) {\n\n\t\t\t\treturn value;\n\n\t\t\t} else if ( inputEncoding == 1 ) {\n\n\t\t\t\treturn sRGBToLinear( value );\n\n\t\t\t} else if ( inputEncoding == 2 ) {\n\n\t\t\t\treturn RGBEToLinear( value );\n\n\t\t\t} else if ( inputEncoding == 3 ) {\n\n\t\t\t\treturn RGBMToLinear( value, 7.0 );\n\n\t\t\t} else if ( inputEncoding == 4 ) {\n\n\t\t\t\treturn RGBMToLinear( value, 16.0 );\n\n\t\t\t} else if ( inputEncoding == 5 ) {\n\n\t\t\t\treturn RGBDToLinear( value, 256.0 );\n\n\t\t\t} else {\n\n\t\t\t\treturn GammaToLinear( value, 2.2 );\n\n\t\t\t}\n\n\t\t}\n\n\t\tvec4 linearToOutputTexel( vec4 value ) {\n\n\t\t\tif ( outputEncoding == 0 ) {\n\n\t\t\t\treturn value;\n\n\t\t\t} else if ( outputEncoding == 1 ) {\n\n\t\t\t\treturn LinearTosRGB( value );\n\n\t\t\t} else if ( outputEncoding == 2 ) {\n\n\t\t\t\treturn LinearToRGBE( value );\n\n\t\t\t} else if ( outputEncoding == 3 ) {\n\n\t\t\t\treturn LinearToRGBM( value, 7.0 );\n\n\t\t\t} else if ( outputEncoding == 4 ) {\n\n\t\t\t\treturn LinearToRGBM( value, 16.0 );\n\n\t\t\t} else if ( outputEncoding == 5 ) {\n\n\t\t\t\treturn LinearToRGBD( value, 256.0 );\n\n\t\t\t} else {\n\n\t\t\t\treturn LinearToGamma( value, 2.2 );\n\n\t\t\t}\n\n\t\t}\n\n\t\tvec4 envMapTexelToLinear( vec4 color ) {\n\n\t\t\treturn inputTexelToLinear( color );\n\n\t\t}\n\t"}function Xi(t){let e=new WeakMap,n=null;function i(t){const n=t.target;n.removeEventListener("dispose",i);const r=e.get(n);void 0!==r&&(e.delete(n),r.dispose())}return{get:function(l){if(l&&l.isTexture&&!1===l.isRenderTargetTexture){const c=l.mapping,h=c===a||c===o,u=c===r||c===s;if(h||u){if(e.has(l))return e.get(l).texture;{const r=l.image;if(h&&r&&r.height>0||u&&r&&function(t){let e=0;const n=6;for(let i=0;i<n;i++)void 0!==t[i]&&e++;return e===n}(r)){const r=t.getRenderTarget();null===n&&(n=new Oi(t));const s=h?n.fromEquirectangular(l):n.fromCubemap(l);return e.set(l,s),t.setRenderTarget(r),l.addEventListener("dispose",i),s.texture}return null}}}return l},dispose:function(){e=new WeakMap,null!==n&&(n.dispose(),n=null)}}}function Yi(t){const e={};function n(n){if(void 0!==e[n])return e[n];let i;switch(n){case"WEBGL_depth_texture":i=t.getExtension("WEBGL_depth_texture")||t.getExtension("MOZ_WEBGL_depth_texture")||t.getExtension("WEBKIT_WEBGL_depth_texture");break;case"EXT_texture_filter_anisotropic":i=t.getExtension("EXT_texture_filter_anisotropic")||t.getExtension("MOZ_EXT_texture_filter_anisotropic")||t.getExtension("WEBKIT_EXT_texture_filter_anisotropic");break;case"WEBGL_compressed_texture_s3tc":i=t.getExtension("WEBGL_compressed_texture_s3tc")||t.getExtension("MOZ_WEBGL_compressed_texture_s3tc")||t.getExtension("WEBKIT_WEBGL_compressed_texture_s3tc");break;case"WEBGL_compressed_texture_pvrtc":i=t.getExtension("WEBGL_compressed_texture_pvrtc")||t.getExtension("WEBKIT_WEBGL_compressed_texture_pvrtc");break;default:i=t.getExtension(n)}return e[n]=i,i}return{has:function(t){return null!==n(t)},init:function(t){t.isWebGL2?n("EXT_color_buffer_float"):(n("WEBGL_depth_texture"),n("OES_texture_float"),n("OES_texture_half_float"),n("OES_texture_half_float_linear"),n("OES_standard_derivatives"),n("OES_element_index_uint"),n("OES_vertex_array_object"),n("ANGLE_instanced_arrays")),n("OES_texture_float_linear"),n("EXT_color_buffer_half_float")},get:function(t){const e=n(t);return null===e&&console.warn("THREE.WebGLRenderer: "+t+" extension not supported."),e}}}function Ji(t,e,n,i){const r={},s=new WeakMap;function a(t){const o=t.target;null!==o.index&&e.remove(o.index);for(const t in o.attributes)e.remove(o.attributes[t]);o.removeEventListener("dispose",a),delete r[o.id];const l=s.get(o);l&&(e.remove(l),s.delete(o)),i.releaseStatesOfGeometry(o),!0===o.isInstancedBufferGeometry&&delete o._maxInstanceCount,n.memory.geometries--}function o(t){const n=[],i=t.index,r=t.attributes.position;let a=0;if(null!==i){const t=i.array;a=i.version;for(let e=0,i=t.length;e<i;e+=3){const i=t[e+0],r=t[e+1],s=t[e+2];n.push(i,r,r,s,s,i)}}else{const t=r.array;a=r.version;for(let e=0,i=t.length/3-1;e<i;e+=3){const t=e+0,i=e+1,r=e+2;n.push(t,i,i,r,r,t)}}const o=new(_t(n)>65535?fn:pn)(n,1);o.version=a;const l=s.get(t);l&&e.remove(l),s.set(t,o)}return{get:function(t,e){return!0===r[e.id]||(e.addEventListener("dispose",a),r[e.id]=!0,n.memory.geometries++),e},update:function(t){const n=t.attributes;for(const t in n)e.update(n[t],34962);const i=t.morphAttributes;for(const t in i){const n=i[t];for(let t=0,i=n.length;t<i;t++)e.update(n[t],34962)}},getWireframeAttribute:function(t){const e=s.get(t);if(e){const n=t.index;null!==n&&e.version<n.version&&o(t)}else o(t);return s.get(t)}}}function Zi(t,e,n,i){const r=i.isWebGL2;let s,a,o;this.setMode=function(t){s=t},this.setIndex=function(t){a=t.type,o=t.bytesPerElement},this.render=function(e,i){t.drawElements(s,i,a,e*o),n.update(i,s,1)},this.renderInstances=function(i,l,c){if(0===c)return;let h,u;if(r)h=t,u="drawElementsInstanced";else if(h=e.get("ANGLE_instanced_arrays"),u="drawElementsInstancedANGLE",null===h)return void console.error("THREE.WebGLIndexedBufferRenderer: using THREE.InstancedBufferGeometry but hardware does not support extension ANGLE_instanced_arrays.");h[u](s,l,a,i*o,c),n.update(l,s,c)}}function Qi(t){const e={frame:0,calls:0,triangles:0,points:0,lines:0};return{memory:{geometries:0,textures:0},render:e,programs:null,autoReset:!0,reset:function(){e.frame++,e.calls=0,e.triangles=0,e.points=0,e.lines=0},update:function(t,n,i){switch(e.calls++,n){case 4:e.triangles+=i*(t/3);break;case 1:e.lines+=i*(t/2);break;case 3:e.lines+=i*(t-1);break;case 2:e.lines+=i*t;break;case 0:e.points+=i*t;break;default:console.error("THREE.WebGLInfo: Unknown draw mode:",n)}}}}class Ki extends Lt{constructor(t=null,e=1,n=1,i=1){super(null),this.image={data:t,width:e,height:n,depth:i},this.magFilter=p,this.minFilter=p,this.wrapR=u,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1,this.needsUpdate=!0}}function $i(t,e){return t[0]-e[0]}function tr(t,e){return Math.abs(e[1])-Math.abs(t[1])}function er(t,e){let n=1;const i=e.isInterleavedBufferAttribute?e.data.array:e.array;i instanceof Int8Array?n=127:i instanceof Int16Array?n=32767:i instanceof Int32Array?n=2147483647:console.error("THREE.WebGLMorphtargets: Unsupported morph attribute data type: ",i),t.divideScalar(n)}function nr(t,e,n){const i={},r=new Float32Array(8),s=new WeakMap,a=new zt,o=[];for(let t=0;t<8;t++)o[t]=[t,0];return{update:function(l,c,h,u){const d=l.morphTargetInfluences;if(!0===e.isWebGL2){const i=c.morphAttributes.position.length;let r=s.get(c);if(void 0===r||r.count!==i){void 0!==r&&r.texture.dispose();const t=void 0!==c.morphAttributes.normal,n=c.morphAttributes.position,o=c.morphAttributes.normal||[],l=!0===t?2:1;let h=c.attributes.position.count*l,u=1;h>e.maxTextureSize&&(u=Math.ceil(h/e.maxTextureSize),h=e.maxTextureSize);const d=new Float32Array(h*u*4*i),p=new Ki(d,h,u,i);p.format=E,p.type=b;const m=4*l;for(let e=0;e<i;e++){const i=n[e],r=o[e],s=h*u*4*e;for(let e=0;e<i.count;e++){a.fromBufferAttribute(i,e),!0===i.normalized&&er(a,i);const n=e*m;d[s+n+0]=a.x,d[s+n+1]=a.y,d[s+n+2]=a.z,d[s+n+3]=0,!0===t&&(a.fromBufferAttribute(r,e),!0===r.normalized&&er(a,r),d[s+n+4]=a.x,d[s+n+5]=a.y,d[s+n+6]=a.z,d[s+n+7]=0)}}r={count:i,texture:p,size:new yt(h,u)},s.set(c,r)}let o=0;for(let t=0;t<d.length;t++)o+=d[t];const l=c.morphTargetsRelative?1:1-o;u.getUniforms().setValue(t,"morphTargetBaseInfluence",l),u.getUniforms().setValue(t,"morphTargetInfluences",d),u.getUniforms().setValue(t,"morphTargetsTexture",r.texture,n),u.getUniforms().setValue(t,"morphTargetsTextureSize",r.size)}else{const e=void 0===d?0:d.length;let n=i[c.id];if(void 0===n||n.length!==e){n=[];for(let t=0;t<e;t++)n[t]=[t,0];i[c.id]=n}for(let t=0;t<e;t++){const e=n[t];e[0]=t,e[1]=d[t]}n.sort(tr);for(let t=0;t<8;t++)t<e&&n[t][1]?(o[t][0]=n[t][0],o[t][1]=n[t][1]):(o[t][0]=Number.MAX_SAFE_INTEGER,o[t][1]=0);o.sort($i);const s=c.morphAttributes.position,a=c.morphAttributes.normal;let l=0;for(let t=0;t<8;t++){const e=o[t],n=e[0],i=e[1];n!==Number.MAX_SAFE_INTEGER&&i?(s&&c.getAttribute("morphTarget"+t)!==s[n]&&c.setAttribute("morphTarget"+t,s[n]),a&&c.getAttribute("morphNormal"+t)!==a[n]&&c.setAttribute("morphNormal"+t,a[n]),r[t]=i,l+=i):(s&&!0===c.hasAttribute("morphTarget"+t)&&c.deleteAttribute("morphTarget"+t),a&&!0===c.hasAttribute("morphNormal"+t)&&c.deleteAttribute("morphNormal"+t),r[t]=0)}const h=c.morphTargetsRelative?1:1-l;u.getUniforms().setValue(t,"morphTargetBaseInfluence",h),u.getUniforms().setValue(t,"morphTargetInfluences",r)}}}}function ir(t,e,n,i){let r=new WeakMap;function s(t){const e=t.target;e.removeEventListener("dispose",s),n.remove(e.instanceMatrix),null!==e.instanceColor&&n.remove(e.instanceColor)}return{update:function(t){const a=i.render.frame,o=t.geometry,l=e.get(t,o);return r.get(l)!==a&&(e.update(l),r.set(l,a)),t.isInstancedMesh&&(!1===t.hasEventListener("dispose",s)&&t.addEventListener("dispose",s),n.update(t.instanceMatrix,34962),null!==t.instanceColor&&n.update(t.instanceColor,34962)),l},dispose:function(){r=new WeakMap}}}Ki.prototype.isDataTexture2DArray=!0;class rr extends Lt{constructor(t=null,e=1,n=1,i=1){super(null),this.image={data:t,width:e,height:n,depth:i},this.magFilter=p,this.minFilter=p,this.wrapR=u,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1,this.needsUpdate=!0}}rr.prototype.isDataTexture3D=!0;const sr=new Lt,ar=new Ki,or=new rr,lr=new ei,cr=[],hr=[],ur=new Float32Array(16),dr=new Float32Array(9),pr=new Float32Array(4);function mr(t,e,n){const i=t[0];if(i<=0||i>0)return t;const r=e*n;let s=cr[r];if(void 0===s&&(s=new Float32Array(r),cr[r]=s),0!==e){i.toArray(s,0);for(let i=1,r=0;i!==e;++i)r+=n,t[i].toArray(s,r)}return s}function fr(t,e){if(t.length!==e.length)return!1;for(let n=0,i=t.length;n<i;n++)if(t[n]!==e[n])return!1;return!0}function gr(t,e){for(let n=0,i=e.length;n<i;n++)t[n]=e[n]}function vr(t,e){let n=hr[e];void 0===n&&(n=new Int32Array(e),hr[e]=n);for(let i=0;i!==e;++i)n[i]=t.allocateTextureUnit();return n}function yr(t,e){const n=this.cache;n[0]!==e&&(t.uniform1f(this.addr,e),n[0]=e)}function xr(t,e){const n=this.cache;if(void 0!==e.x)n[0]===e.x&&n[1]===e.y||(t.uniform2f(this.addr,e.x,e.y),n[0]=e.x,n[1]=e.y);else{if(fr(n,e))return;t.uniform2fv(this.addr,e),gr(n,e)}}function _r(t,e){const n=this.cache;if(void 0!==e.x)n[0]===e.x&&n[1]===e.y&&n[2]===e.z||(t.uniform3f(this.addr,e.x,e.y,e.z),n[0]=e.x,n[1]=e.y,n[2]=e.z);else if(void 0!==e.r)n[0]===e.r&&n[1]===e.g&&n[2]===e.b||(t.uniform3f(this.addr,e.r,e.g,e.b),n[0]=e.r,n[1]=e.g,n[2]=e.b);else{if(fr(n,e))return;t.uniform3fv(this.addr,e),gr(n,e)}}function Mr(t,e){const n=this.cache;if(void 0!==e.x)n[0]===e.x&&n[1]===e.y&&n[2]===e.z&&n[3]===e.w||(t.uniform4f(this.addr,e.x,e.y,e.z,e.w),n[0]=e.x,n[1]=e.y,n[2]=e.z,n[3]=e.w);else{if(fr(n,e))return;t.uniform4fv(this.addr,e),gr(n,e)}}function br(t,e){const n=this.cache,i=e.elements;if(void 0===i){if(fr(n,e))return;t.uniformMatrix2fv(this.addr,!1,e),gr(n,e)}else{if(fr(n,i))return;pr.set(i),t.uniformMatrix2fv(this.addr,!1,pr),gr(n,i)}}function wr(t,e){const n=this.cache,i=e.elements;if(void 0===i){if(fr(n,e))return;t.uniformMatrix3fv(this.addr,!1,e),gr(n,e)}else{if(fr(n,i))return;dr.set(i),t.uniformMatrix3fv(this.addr,!1,dr),gr(n,i)}}function Sr(t,e){const n=this.cache,i=e.elements;if(void 0===i){if(fr(n,e))return;t.uniformMatrix4fv(this.addr,!1,e),gr(n,e)}else{if(fr(n,i))return;ur.set(i),t.uniformMatrix4fv(this.addr,!1,ur),gr(n,i)}}function Tr(t,e){const n=this.cache;n[0]!==e&&(t.uniform1i(this.addr,e),n[0]=e)}function Er(t,e){const n=this.cache;fr(n,e)||(t.uniform2iv(this.addr,e),gr(n,e))}function Ar(t,e){const n=this.cache;fr(n,e)||(t.uniform3iv(this.addr,e),gr(n,e))}function Lr(t,e){const n=this.cache;fr(n,e)||(t.uniform4iv(this.addr,e),gr(n,e))}function Rr(t,e){const n=this.cache;n[0]!==e&&(t.uniform1ui(this.addr,e),n[0]=e)}function Cr(t,e){const n=this.cache;fr(n,e)||(t.uniform2uiv(this.addr,e),gr(n,e))}function Pr(t,e){const n=this.cache;fr(n,e)||(t.uniform3uiv(this.addr,e),gr(n,e))}function Ir(t,e){const n=this.cache;fr(n,e)||(t.uniform4uiv(this.addr,e),gr(n,e))}function Dr(t,e,n){const i=this.cache,r=n.allocateTextureUnit();i[0]!==r&&(t.uniform1i(this.addr,r),i[0]=r),n.safeSetTexture2D(e||sr,r)}function Nr(t,e,n){const i=this.cache,r=n.allocateTextureUnit();i[0]!==r&&(t.uniform1i(this.addr,r),i[0]=r),n.setTexture3D(e||or,r)}function zr(t,e,n){const i=this.cache,r=n.allocateTextureUnit();i[0]!==r&&(t.uniform1i(this.addr,r),i[0]=r),n.safeSetTextureCube(e||lr,r)}function Br(t,e,n){const i=this.cache,r=n.allocateTextureUnit();i[0]!==r&&(t.uniform1i(this.addr,r),i[0]=r),n.setTexture2DArray(e||ar,r)}function Fr(t,e){t.uniform1fv(this.addr,e)}function Or(t,e){const n=mr(e,this.size,2);t.uniform2fv(this.addr,n)}function Ur(t,e){const n=mr(e,this.size,3);t.uniform3fv(this.addr,n)}function Hr(t,e){const n=mr(e,this.size,4);t.uniform4fv(this.addr,n)}function Gr(t,e){const n=mr(e,this.size,4);t.uniformMatrix2fv(this.addr,!1,n)}function kr(t,e){const n=mr(e,this.size,9);t.uniformMatrix3fv(this.addr,!1,n)}function Vr(t,e){const n=mr(e,this.size,16);t.uniformMatrix4fv(this.addr,!1,n)}function Wr(t,e){t.uniform1iv(this.addr,e)}function jr(t,e){t.uniform2iv(this.addr,e)}function qr(t,e){t.uniform3iv(this.addr,e)}function Xr(t,e){t.uniform4iv(this.addr,e)}function Yr(t,e){t.uniform1uiv(this.addr,e)}function Jr(t,e){t.uniform2uiv(this.addr,e)}function Zr(t,e){t.uniform3uiv(this.addr,e)}function Qr(t,e){t.uniform4uiv(this.addr,e)}function Kr(t,e,n){const i=e.length,r=vr(n,i);t.uniform1iv(this.addr,r);for(let t=0;t!==i;++t)n.safeSetTexture2D(e[t]||sr,r[t])}function $r(t,e,n){const i=e.length,r=vr(n,i);t.uniform1iv(this.addr,r);for(let t=0;t!==i;++t)n.safeSetTextureCube(e[t]||lr,r[t])}function ts(t,e,n){this.id=t,this.addr=n,this.cache=[],this.setValue=function(t){switch(t){case 5126:return yr;case 35664:return xr;case 35665:return _r;case 35666:return Mr;case 35674:return br;case 35675:return wr;case 35676:return Sr;case 5124:case 35670:return Tr;case 35667:case 35671:return Er;case 35668:case 35672:return Ar;case 35669:case 35673:return Lr;case 5125:return Rr;case 36294:return Cr;case 36295:return Pr;case 36296:return Ir;case 35678:case 36198:case 36298:case 36306:case 35682:return Dr;case 35679:case 36299:case 36307:return Nr;case 35680:case 36300:case 36308:case 36293:return zr;case 36289:case 36303:case 36311:case 36292:return Br}}(e.type)}function es(t,e,n){this.id=t,this.addr=n,this.cache=[],this.size=e.size,this.setValue=function(t){switch(t){case 5126:return Fr;case 35664:return Or;case 35665:return Ur;case 35666:return Hr;case 35674:return Gr;case 35675:return kr;case 35676:return Vr;case 5124:case 35670:return Wr;case 35667:case 35671:return jr;case 35668:case 35672:return qr;case 35669:case 35673:return Xr;case 5125:return Yr;case 36294:return Jr;case 36295:return Zr;case 36296:return Qr;case 35678:case 36198:case 36298:case 36306:case 35682:return Kr;case 35680:case 36300:case 36308:case 36293:return $r}}(e.type)}function ns(t){this.id=t,this.seq=[],this.map={}}es.prototype.updateCache=function(t){const e=this.cache;t instanceof Float32Array&&e.length!==t.length&&(this.cache=new Float32Array(t.length)),gr(e,t)},ns.prototype.setValue=function(t,e,n){const i=this.seq;for(let r=0,s=i.length;r!==s;++r){const s=i[r];s.setValue(t,e[s.id],n)}};const is=/(\w+)(\])?(\[|\.)?/g;function rs(t,e){t.seq.push(e),t.map[e.id]=e}function ss(t,e,n){const i=t.name,r=i.length;for(is.lastIndex=0;;){const s=is.exec(i),a=is.lastIndex;let o=s[1];const l="]"===s[2],c=s[3];if(l&&(o|=0),void 0===c||"["===c&&a+2===r){rs(n,void 0===c?new ts(o,t,e):new es(o,t,e));break}{let t=n.map[o];void 0===t&&(t=new ns(o),rs(n,t)),n=t}}}function as(t,e){this.seq=[],this.map={};const n=t.getProgramParameter(e,35718);for(let i=0;i<n;++i){const n=t.getActiveUniform(e,i);ss(n,t.getUniformLocation(e,n.name),this)}}function os(t,e,n){const i=t.createShader(e);return t.shaderSource(i,n),t.compileShader(i),i}as.prototype.setValue=function(t,e,n,i){const r=this.map[e];void 0!==r&&r.setValue(t,n,i)},as.prototype.setOptional=function(t,e,n){const i=e[n];void 0!==i&&this.setValue(t,n,i)},as.upload=function(t,e,n,i){for(let r=0,s=e.length;r!==s;++r){const s=e[r],a=n[s.id];!1!==a.needsUpdate&&s.setValue(t,a.value,i)}},as.seqWithValue=function(t,e){const n=[];for(let i=0,r=t.length;i!==r;++i){const r=t[i];r.id in e&&n.push(r)}return n};let ls=0;function cs(t){switch(t){case X:return["Linear","( value )"];case Y:return["sRGB","( value )"];case Z:return["RGBE","( value )"];case Q:return["RGBM","( value, 7.0 )"];case K:return["RGBM","( value, 16.0 )"];case $:return["RGBD","( value, 256.0 )"];case J:return["Gamma","( value, float( GAMMA_FACTOR ) )"];case 3003:return["LogLuv","( value )"];default:return console.warn("THREE.WebGLProgram: Unsupported encoding:",t),["Linear","( value )"]}}function hs(t,e,n){const i=t.getShaderParameter(e,35713),r=t.getShaderInfoLog(e).trim();return i&&""===r?"":n.toUpperCase()+"\n\n"+r+"\n\n"+function(t){const e=t.split("\n");for(let t=0;t<e.length;t++)e[t]=t+1+": "+e[t];return e.join("\n")}(t.getShaderSource(e))}function us(t,e){const n=cs(e);return"vec4 "+t+"( vec4 value ) { return "+n[0]+"ToLinear"+n[1]+"; }"}function ds(t,e){const n=cs(e);return"vec4 "+t+"( vec4 value ) { return LinearTo"+n[0]+n[1]+"; }"}function ps(t,e){let n;switch(e){case 1:n="Linear";break;case 2:n="Reinhard";break;case 3:n="OptimizedCineon";break;case 4:n="ACESFilmic";break;case 5:n="Custom";break;default:console.warn("THREE.WebGLProgram: Unsupported toneMapping:",e),n="Linear"}return"vec3 "+t+"( vec3 color ) { return "+n+"ToneMapping( color ); }"}function ms(t){return""!==t}function fs(t,e){return t.replace(/NUM_DIR_LIGHTS/g,e.numDirLights).replace(/NUM_SPOT_LIGHTS/g,e.numSpotLights).replace(/NUM_RECT_AREA_LIGHTS/g,e.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g,e.numPointLights).replace(/NUM_HEMI_LIGHTS/g,e.numHemiLights).replace(/NUM_DIR_LIGHT_SHADOWS/g,e.numDirLightShadows).replace(/NUM_SPOT_LIGHT_SHADOWS/g,e.numSpotLightShadows).replace(/NUM_POINT_LIGHT_SHADOWS/g,e.numPointLightShadows)}function gs(t,e){return t.replace(/NUM_CLIPPING_PLANES/g,e.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g,e.numClippingPlanes-e.numClipIntersection)}const vs=/^[ \t]*#include +<([\w\d./]+)>/gm;function ys(t){return t.replace(vs,xs)}function xs(t,e){const n=pi[e];if(void 0===n)throw new Error("Can not resolve #include <"+e+">");return ys(n)}const _s=/#pragma unroll_loop[\s]+?for \( int i \= (\d+)\; i < (\d+)\; i \+\+ \) \{([\s\S]+?)(?=\})\}/g,Ms=/#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;function bs(t){return t.replace(Ms,Ss).replace(_s,ws)}function ws(t,e,n,i){return console.warn("WebGLProgram: #pragma unroll_loop shader syntax is deprecated. Please use #pragma unroll_loop_start syntax instead."),Ss(t,e,n,i)}function Ss(t,e,n,i){let r="";for(let t=parseInt(e);t<parseInt(n);t++)r+=i.replace(/\[\s*i\s*\]/g,"[ "+t+" ]").replace(/UNROLLED_LOOP_INDEX/g,t);return r}function Ts(t){let e="precision "+t.precision+" float;\nprecision "+t.precision+" int;";return"highp"===t.precision?e+="\n#define HIGH_PRECISION":"mediump"===t.precision?e+="\n#define MEDIUM_PRECISION":"lowp"===t.precision&&(e+="\n#define LOW_PRECISION"),e}function Es(t,e,n,i){const a=t.getContext(),o=n.defines;let h=n.vertexShader,u=n.fragmentShader;const d=function(t){let e="SHADOWMAP_TYPE_BASIC";return 1===t.shadowMapType?e="SHADOWMAP_TYPE_PCF":2===t.shadowMapType?e="SHADOWMAP_TYPE_PCF_SOFT":3===t.shadowMapType&&(e="SHADOWMAP_TYPE_VSM"),e}(n),p=function(t){let e="ENVMAP_TYPE_CUBE";if(t.envMap)switch(t.envMapMode){case r:case s:e="ENVMAP_TYPE_CUBE";break;case l:case c:e="ENVMAP_TYPE_CUBE_UV"}return e}(n),m=function(t){let e="ENVMAP_MODE_REFLECTION";if(t.envMap)switch(t.envMapMode){case s:case c:e="ENVMAP_MODE_REFRACTION"}return e}(n),f=function(t){let e="ENVMAP_BLENDING_NONE";if(t.envMap)switch(t.combine){case 0:e="ENVMAP_BLENDING_MULTIPLY";break;case 1:e="ENVMAP_BLENDING_MIX";break;case 2:e="ENVMAP_BLENDING_ADD"}return e}(n),g=t.gammaFactor>0?t.gammaFactor:1,v=n.isWebGL2?"":function(t){return[t.extensionDerivatives||t.envMapCubeUV||t.bumpMap||t.tangentSpaceNormalMap||t.clearcoatNormalMap||t.flatShading||"physical"===t.shaderID?"#extension GL_OES_standard_derivatives : enable":"",(t.extensionFragDepth||t.logarithmicDepthBuffer)&&t.rendererExtensionFragDepth?"#extension GL_EXT_frag_depth : enable":"",t.extensionDrawBuffers&&t.rendererExtensionDrawBuffers?"#extension GL_EXT_draw_buffers : require":"",(t.extensionShaderTextureLOD||t.envMap||t.transmission)&&t.rendererExtensionShaderTextureLod?"#extension GL_EXT_shader_texture_lod : enable":""].filter(ms).join("\n")}(n),y=function(t){const e=[];for(const n in t){const i=t[n];!1!==i&&e.push("#define "+n+" "+i)}return e.join("\n")}(o),x=a.createProgram();let _,M,b=n.glslVersion?"#version "+n.glslVersion+"\n":"";n.isRawShaderMaterial?(_=[y].filter(ms).join("\n"),_.length>0&&(_+="\n"),M=[v,y].filter(ms).join("\n"),M.length>0&&(M+="\n")):(_=[Ts(n),"#define SHADER_NAME "+n.shaderName,y,n.instancing?"#define USE_INSTANCING":"",n.instancingColor?"#define USE_INSTANCING_COLOR":"",n.supportsVertexTextures?"#define VERTEX_TEXTURES":"","#define GAMMA_FACTOR "+g,"#define MAX_BONES "+n.maxBones,n.useFog&&n.fog?"#define USE_FOG":"",n.useFog&&n.fogExp2?"#define FOG_EXP2":"",n.map?"#define USE_MAP":"",n.envMap?"#define USE_ENVMAP":"",n.envMap?"#define "+m:"",n.lightMap?"#define USE_LIGHTMAP":"",n.aoMap?"#define USE_AOMAP":"",n.emissiveMap?"#define USE_EMISSIVEMAP":"",n.bumpMap?"#define USE_BUMPMAP":"",n.normalMap?"#define USE_NORMALMAP":"",n.normalMap&&n.objectSpaceNormalMap?"#define OBJECTSPACE_NORMALMAP":"",n.normalMap&&n.tangentSpaceNormalMap?"#define TANGENTSPACE_NORMALMAP":"",n.clearcoatMap?"#define USE_CLEARCOATMAP":"",n.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",n.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",n.displacementMap&&n.supportsVertexTextures?"#define USE_DISPLACEMENTMAP":"",n.specularMap?"#define USE_SPECULARMAP":"",n.specularIntensityMap?"#define USE_SPECULARINTENSITYMAP":"",n.specularColorMap?"#define USE_SPECULARCOLORMAP":"",n.roughnessMap?"#define USE_ROUGHNESSMAP":"",n.metalnessMap?"#define USE_METALNESSMAP":"",n.alphaMap?"#define USE_ALPHAMAP":"",n.transmission?"#define USE_TRANSMISSION":"",n.transmissionMap?"#define USE_TRANSMISSIONMAP":"",n.thicknessMap?"#define USE_THICKNESSMAP":"",n.sheenColorMap?"#define USE_SHEENCOLORMAP":"",n.sheenRoughnessMap?"#define USE_SHEENROUGHNESSMAP":"",n.vertexTangents?"#define USE_TANGENT":"",n.vertexColors?"#define USE_COLOR":"",n.vertexAlphas?"#define USE_COLOR_ALPHA":"",n.vertexUvs?"#define USE_UV":"",n.uvsVertexOnly?"#define UVS_VERTEX_ONLY":"",n.flatShading?"#define FLAT_SHADED":"",n.skinning?"#define USE_SKINNING":"",n.useVertexTexture?"#define BONE_TEXTURE":"",n.morphTargets?"#define USE_MORPHTARGETS":"",n.morphNormals&&!1===n.flatShading?"#define USE_MORPHNORMALS":"",n.morphTargets&&n.isWebGL2?"#define MORPHTARGETS_TEXTURE":"",n.morphTargets&&n.isWebGL2?"#define MORPHTARGETS_COUNT "+n.morphTargetsCount:"",n.doubleSided?"#define DOUBLE_SIDED":"",n.flipSided?"#define FLIP_SIDED":"",n.shadowMapEnabled?"#define USE_SHADOWMAP":"",n.shadowMapEnabled?"#define "+d:"",n.sizeAttenuation?"#define USE_SIZEATTENUATION":"",n.logarithmicDepthBuffer?"#define USE_LOGDEPTHBUF":"",n.logarithmicDepthBuffer&&n.rendererExtensionFragDepth?"#define USE_LOGDEPTHBUF_EXT":"","uniform mat4 modelMatrix;","uniform mat4 modelViewMatrix;","uniform mat4 projectionMatrix;","uniform mat4 viewMatrix;","uniform mat3 normalMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;","#ifdef USE_INSTANCING","\tattribute mat4 instanceMatrix;","#endif","#ifdef USE_INSTANCING_COLOR","\tattribute vec3 instanceColor;","#endif","attribute vec3 position;","attribute vec3 normal;","attribute vec2 uv;","#ifdef USE_TANGENT","\tattribute vec4 tangent;","#endif","#if defined( USE_COLOR_ALPHA )","\tattribute vec4 color;","#elif defined( USE_COLOR )","\tattribute vec3 color;","#endif","#if ( defined( USE_MORPHTARGETS ) && ! defined( MORPHTARGETS_TEXTURE ) )","\tattribute vec3 morphTarget0;","\tattribute vec3 morphTarget1;","\tattribute vec3 morphTarget2;","\tattribute vec3 morphTarget3;","\t#ifdef USE_MORPHNORMALS","\t\tattribute vec3 morphNormal0;","\t\tattribute vec3 morphNormal1;","\t\tattribute vec3 morphNormal2;","\t\tattribute vec3 morphNormal3;","\t#else","\t\tattribute vec3 morphTarget4;","\t\tattribute vec3 morphTarget5;","\t\tattribute vec3 morphTarget6;","\t\tattribute vec3 morphTarget7;","\t#endif","#endif","#ifdef USE_SKINNING","\tattribute vec4 skinIndex;","\tattribute vec4 skinWeight;","#endif","\n"].filter(ms).join("\n"),M=[v,Ts(n),"#define SHADER_NAME "+n.shaderName,y,"#define GAMMA_FACTOR "+g,n.useFog&&n.fog?"#define USE_FOG":"",n.useFog&&n.fogExp2?"#define FOG_EXP2":"",n.map?"#define USE_MAP":"",n.matcap?"#define USE_MATCAP":"",n.envMap?"#define USE_ENVMAP":"",n.envMap?"#define "+p:"",n.envMap?"#define "+m:"",n.envMap?"#define "+f:"",n.lightMap?"#define USE_LIGHTMAP":"",n.aoMap?"#define USE_AOMAP":"",n.emissiveMap?"#define USE_EMISSIVEMAP":"",n.bumpMap?"#define USE_BUMPMAP":"",n.normalMap?"#define USE_NORMALMAP":"",n.normalMap&&n.objectSpaceNormalMap?"#define OBJECTSPACE_NORMALMAP":"",n.normalMap&&n.tangentSpaceNormalMap?"#define TANGENTSPACE_NORMALMAP":"",n.clearcoat?"#define USE_CLEARCOAT":"",n.clearcoatMap?"#define USE_CLEARCOATMAP":"",n.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",n.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",n.specularMap?"#define USE_SPECULARMAP":"",n.specularIntensityMap?"#define USE_SPECULARINTENSITYMAP":"",n.specularColorMap?"#define USE_SPECULARCOLORMAP":"",n.roughnessMap?"#define USE_ROUGHNESSMAP":"",n.metalnessMap?"#define USE_METALNESSMAP":"",n.alphaMap?"#define USE_ALPHAMAP":"",n.alphaTest?"#define USE_ALPHATEST":"",n.sheen?"#define USE_SHEEN":"",n.sheenColorMap?"#define USE_SHEENCOLORMAP":"",n.sheenRoughnessMap?"#define USE_SHEENROUGHNESSMAP":"",n.transmission?"#define USE_TRANSMISSION":"",n.transmissionMap?"#define USE_TRANSMISSIONMAP":"",n.thicknessMap?"#define USE_THICKNESSMAP":"",n.vertexTangents?"#define USE_TANGENT":"",n.vertexColors||n.instancingColor?"#define USE_COLOR":"",n.vertexAlphas?"#define USE_COLOR_ALPHA":"",n.vertexUvs?"#define USE_UV":"",n.uvsVertexOnly?"#define UVS_VERTEX_ONLY":"",n.gradientMap?"#define USE_GRADIENTMAP":"",n.flatShading?"#define FLAT_SHADED":"",n.doubleSided?"#define DOUBLE_SIDED":"",n.flipSided?"#define FLIP_SIDED":"",n.shadowMapEnabled?"#define USE_SHADOWMAP":"",n.shadowMapEnabled?"#define "+d:"",n.premultipliedAlpha?"#define PREMULTIPLIED_ALPHA":"",n.physicallyCorrectLights?"#define PHYSICALLY_CORRECT_LIGHTS":"",n.logarithmicDepthBuffer?"#define USE_LOGDEPTHBUF":"",n.logarithmicDepthBuffer&&n.rendererExtensionFragDepth?"#define USE_LOGDEPTHBUF_EXT":"",(n.extensionShaderTextureLOD||n.envMap)&&n.rendererExtensionShaderTextureLod?"#define TEXTURE_LOD_EXT":"","uniform mat4 viewMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;",0!==n.toneMapping?"#define TONE_MAPPING":"",0!==n.toneMapping?pi.tonemapping_pars_fragment:"",0!==n.toneMapping?ps("toneMapping",n.toneMapping):"",n.dithering?"#define DITHERING":"",n.format===T?"#define OPAQUE":"",pi.encodings_pars_fragment,n.map?us("mapTexelToLinear",n.mapEncoding):"",n.matcap?us("matcapTexelToLinear",n.matcapEncoding):"",n.envMap?us("envMapTexelToLinear",n.envMapEncoding):"",n.emissiveMap?us("emissiveMapTexelToLinear",n.emissiveMapEncoding):"",n.specularColorMap?us("specularColorMapTexelToLinear",n.specularColorMapEncoding):"",n.sheenColorMap?us("sheenColorMapTexelToLinear",n.sheenColorMapEncoding):"",n.lightMap?us("lightMapTexelToLinear",n.lightMapEncoding):"",ds("linearToOutputTexel",n.outputEncoding),n.depthPacking?"#define DEPTH_PACKING "+n.depthPacking:"","\n"].filter(ms).join("\n")),h=ys(h),h=fs(h,n),h=gs(h,n),u=ys(u),u=fs(u,n),u=gs(u,n),h=bs(h),u=bs(u),n.isWebGL2&&!0!==n.isRawShaderMaterial&&(b="#version 300 es\n",_=["precision mediump sampler2DArray;","#define attribute in","#define varying out","#define texture2D texture"].join("\n")+"\n"+_,M=["#define varying in",n.glslVersion===it?"":"out highp vec4 pc_fragColor;",n.glslVersion===it?"":"#define gl_FragColor pc_fragColor","#define gl_FragDepthEXT gl_FragDepth","#define texture2D texture","#define textureCube texture","#define texture2DProj textureProj","#define texture2DLodEXT textureLod","#define texture2DProjLodEXT textureProjLod","#define textureCubeLodEXT textureLod","#define texture2DGradEXT textureGrad","#define texture2DProjGradEXT textureProjGrad","#define textureCubeGradEXT textureGrad"].join("\n")+"\n"+M);const w=b+M+u,S=os(a,35633,b+_+h),E=os(a,35632,w);if(a.attachShader(x,S),a.attachShader(x,E),void 0!==n.index0AttributeName?a.bindAttribLocation(x,0,n.index0AttributeName):!0===n.morphTargets&&a.bindAttribLocation(x,0,"position"),a.linkProgram(x),t.debug.checkShaderErrors){const t=a.getProgramInfoLog(x).trim(),e=a.getShaderInfoLog(S).trim(),n=a.getShaderInfoLog(E).trim();let i=!0,r=!0;if(!1===a.getProgramParameter(x,35714)){i=!1;const e=hs(a,S,"vertex"),n=hs(a,E,"fragment");console.error("THREE.WebGLProgram: Shader Error "+a.getError()+" - VALIDATE_STATUS "+a.getProgramParameter(x,35715)+"\n\nProgram Info Log: "+t+"\n"+e+"\n"+n)}else""!==t?console.warn("THREE.WebGLProgram: Program Info Log:",t):""!==e&&""!==n||(r=!1);r&&(this.diagnostics={runnable:i,programLog:t,vertexShader:{log:e,prefix:_},fragmentShader:{log:n,prefix:M}})}let A,L;return a.deleteShader(S),a.deleteShader(E),this.getUniforms=function(){return void 0===A&&(A=new as(a,x)),A},this.getAttributes=function(){return void 0===L&&(L=function(t,e){const n={},i=t.getProgramParameter(e,35721);for(let r=0;r<i;r++){const i=t.getActiveAttrib(e,r),s=i.name;let a=1;35674===i.type&&(a=2),35675===i.type&&(a=3),35676===i.type&&(a=4),n[s]={type:i.type,location:t.getAttribLocation(e,s),locationSize:a}}return n}(a,x)),L},this.destroy=function(){i.releaseStatesOfProgram(this),a.deleteProgram(x),this.program=void 0},this.name=n.shaderName,this.id=ls++,this.cacheKey=e,this.usedTimes=1,this.program=x,this.vertexShader=S,this.fragmentShader=E,this}function As(t,e,n,i,r,s,a){const o=[],h=r.isWebGL2,u=r.logarithmicDepthBuffer,d=r.floatVertexTextures,p=r.maxVertexUniforms,m=r.vertexTextures;let f=r.precision;const g={MeshDepthMaterial:"depth",MeshDistanceMaterial:"distanceRGBA",MeshNormalMaterial:"normal",MeshBasicMaterial:"basic",MeshLambertMaterial:"lambert",MeshPhongMaterial:"phong",MeshToonMaterial:"toon",MeshStandardMaterial:"physical",MeshPhysicalMaterial:"physical",MeshMatcapMaterial:"matcap",LineBasicMaterial:"basic",LineDashedMaterial:"dashed",PointsMaterial:"points",ShadowMaterial:"shadow",SpriteMaterial:"sprite"},v=["precision","isWebGL2","supportsVertexTextures","outputEncoding","instancing","instancingColor","map","mapEncoding","matcap","matcapEncoding","envMap","envMapMode","envMapEncoding","envMapCubeUV","lightMap","lightMapEncoding","aoMap","emissiveMap","emissiveMapEncoding","bumpMap","normalMap","objectSpaceNormalMap","tangentSpaceNormalMap","clearcoat","clearcoatMap","clearcoatRoughnessMap","clearcoatNormalMap","displacementMap","specularMap",,"roughnessMap","metalnessMap","gradientMap","alphaMap","alphaTest","combine","vertexColors","vertexAlphas","vertexTangents","vertexUvs","uvsVertexOnly","fog","useFog","fogExp2","flatShading","sizeAttenuation","logarithmicDepthBuffer","skinning","maxBones","useVertexTexture","morphTargets","morphNormals","morphTargetsCount","premultipliedAlpha","numDirLights","numPointLights","numSpotLights","numHemiLights","numRectAreaLights","numDirLightShadows","numPointLightShadows","numSpotLightShadows","shadowMapEnabled","shadowMapType","toneMapping","physicallyCorrectLights","doubleSided","flipSided","numClippingPlanes","numClipIntersection","depthPacking","dithering","format","specularIntensityMap","specularColorMap","specularColorMapEncoding","transmission","transmissionMap","thicknessMap","sheen","sheenColorMap","sheenColorMapEncoding","sheenRoughnessMap"];function y(t){let e;return t&&t.isTexture?e=t.encoding:t&&t.isWebGLRenderTarget?(console.warn("THREE.WebGLPrograms.getTextureEncodingFromMap: don't use render targets as textures. Use their .texture property instead."),e=t.texture.encoding):e=X,h&&t&&t.isTexture&&t.format===E&&t.type===x&&t.encoding===Y&&(e=X),e}return{getParameters:function(s,o,v,x,_){const M=x.fog,b=s.isMeshStandardMaterial?x.environment:null,w=(s.isMeshStandardMaterial?n:e).get(s.envMap||b),S=g[s.type],T=_.isSkinnedMesh?function(t){const e=t.skeleton.bones;if(d)return 1024;{const t=p,n=Math.floor((t-20)/4),i=Math.min(n,e.length);return i<e.length?(console.warn("THREE.WebGLRenderer: Skeleton has "+e.length+" bones. This GPU supports "+i+"."),0):i}}(_):0;let E,A;if(null!==s.precision&&(f=r.getMaxPrecision(s.precision),f!==s.precision&&console.warn("THREE.WebGLProgram.getParameters:",s.precision,"not supported, using",f,"instead.")),S){const t=fi[S];E=t.vertexShader,A=t.fragmentShader}else E=s.vertexShader,A=s.fragmentShader;const L=t.getRenderTarget(),R=s.alphaTest>0,C=s.clearcoat>0;return{isWebGL2:h,shaderID:S,shaderName:s.type,vertexShader:E,fragmentShader:A,defines:s.defines,isRawShaderMaterial:!0===s.isRawShaderMaterial,glslVersion:s.glslVersion,precision:f,instancing:!0===_.isInstancedMesh,instancingColor:!0===_.isInstancedMesh&&null!==_.instanceColor,supportsVertexTextures:m,outputEncoding:null!==L?y(L.texture):t.outputEncoding,map:!!s.map,mapEncoding:y(s.map),matcap:!!s.matcap,matcapEncoding:y(s.matcap),envMap:!!w,envMapMode:w&&w.mapping,envMapEncoding:y(w),envMapCubeUV:!!w&&(w.mapping===l||w.mapping===c),lightMap:!!s.lightMap,lightMapEncoding:y(s.lightMap),aoMap:!!s.aoMap,emissiveMap:!!s.emissiveMap,emissiveMapEncoding:y(s.emissiveMap),bumpMap:!!s.bumpMap,normalMap:!!s.normalMap,objectSpaceNormalMap:1===s.normalMapType,tangentSpaceNormalMap:0===s.normalMapType,clearcoat:C,clearcoatMap:C&&!!s.clearcoatMap,clearcoatRoughnessMap:C&&!!s.clearcoatRoughnessMap,clearcoatNormalMap:C&&!!s.clearcoatNormalMap,displacementMap:!!s.displacementMap,roughnessMap:!!s.roughnessMap,metalnessMap:!!s.metalnessMap,specularMap:!!s.specularMap,specularIntensityMap:!!s.specularIntensityMap,specularColorMap:!!s.specularColorMap,specularColorMapEncoding:y(s.specularColorMap),alphaMap:!!s.alphaMap,alphaTest:R,gradientMap:!!s.gradientMap,sheen:s.sheen>0,sheenColorMap:!!s.sheenColorMap,sheenColorMapEncoding:y(s.sheenColorMap),sheenRoughnessMap:!!s.sheenRoughnessMap,transmission:s.transmission>0,transmissionMap:!!s.transmissionMap,thicknessMap:!!s.thicknessMap,combine:s.combine,vertexTangents:!!s.normalMap&&!!_.geometry&&!!_.geometry.attributes.tangent,vertexColors:s.vertexColors,vertexAlphas:!0===s.vertexColors&&!!_.geometry&&!!_.geometry.attributes.color&&4===_.geometry.attributes.color.itemSize,vertexUvs:!!s.map||!!s.bumpMap||!!s.normalMap||!!s.specularMap||!!s.alphaMap||!!s.emissiveMap||!!s.roughnessMap||!!s.metalnessMap||!!s.clearcoatMap||!!s.clearcoatRoughnessMap||!!s.clearcoatNormalMap||!!s.displacementMap||!!s.transmissionMap||!!s.thicknessMap||!!s.specularIntensityMap||!!s.specularColorMap||!!s.sheenColorMap||s.sheenRoughnessMap,uvsVertexOnly:!(s.map||s.bumpMap||s.normalMap||s.specularMap||s.alphaMap||s.emissiveMap||s.roughnessMap||s.metalnessMap||s.clearcoatNormalMap||s.transmission>0||s.transmissionMap||s.thicknessMap||s.specularIntensityMap||s.specularColorMap||!!s.sheen>0||s.sheenColorMap||s.sheenRoughnessMap||!s.displacementMap),fog:!!M,useFog:s.fog,fogExp2:M&&M.isFogExp2,flatShading:!!s.flatShading,sizeAttenuation:s.sizeAttenuation,logarithmicDepthBuffer:u,skinning:!0===_.isSkinnedMesh&&T>0,maxBones:T,useVertexTexture:d,morphTargets:!!_.geometry&&!!_.geometry.morphAttributes.position,morphNormals:!!_.geometry&&!!_.geometry.morphAttributes.normal,morphTargetsCount:_.geometry&&_.geometry.morphAttributes.position?_.geometry.morphAttributes.position.length:0,numDirLights:o.directional.length,numPointLights:o.point.length,numSpotLights:o.spot.length,numRectAreaLights:o.rectArea.length,numHemiLights:o.hemi.length,numDirLightShadows:o.directionalShadowMap.length,numPointLightShadows:o.pointShadowMap.length,numSpotLightShadows:o.spotShadowMap.length,numClippingPlanes:a.numPlanes,numClipIntersection:a.numIntersection,format:s.format,dithering:s.dithering,shadowMapEnabled:t.shadowMap.enabled&&v.length>0,shadowMapType:t.shadowMap.type,toneMapping:s.toneMapped?t.toneMapping:0,physicallyCorrectLights:t.physicallyCorrectLights,premultipliedAlpha:s.premultipliedAlpha,doubleSided:2===s.side,flipSided:1===s.side,depthPacking:void 0!==s.depthPacking&&s.depthPacking,index0AttributeName:s.index0AttributeName,extensionDerivatives:s.extensions&&s.extensions.derivatives,extensionFragDepth:s.extensions&&s.extensions.fragDepth,extensionDrawBuffers:s.extensions&&s.extensions.drawBuffers,extensionShaderTextureLOD:s.extensions&&s.extensions.shaderTextureLOD,rendererExtensionFragDepth:h||i.has("EXT_frag_depth"),rendererExtensionDrawBuffers:h||i.has("WEBGL_draw_buffers"),rendererExtensionShaderTextureLod:h||i.has("EXT_shader_texture_lod"),customProgramCacheKey:s.customProgramCacheKey()}},getProgramCacheKey:function(e){const n=[];if(e.shaderID?n.push(e.shaderID):(n.push(St(e.fragmentShader)),n.push(St(e.vertexShader))),void 0!==e.defines)for(const t in e.defines)n.push(t),n.push(e.defines[t]);if(!1===e.isRawShaderMaterial){for(let t=0;t<v.length;t++)n.push(e[v[t]]);n.push(t.outputEncoding),n.push(t.gammaFactor)}return n.push(e.customProgramCacheKey),n.join()},getUniforms:function(t){const e=g[t.type];let n;if(e){const t=fi[e];n=Jn.clone(t.uniforms)}else n=t.uniforms;return n},acquireProgram:function(e,n){let i;for(let t=0,e=o.length;t<e;t++){const e=o[t];if(e.cacheKey===n){i=e,++i.usedTimes;break}}return void 0===i&&(i=new Es(t,n,e,s),o.push(i)),i},releaseProgram:function(t){if(0==--t.usedTimes){const e=o.indexOf(t);o[e]=o[o.length-1],o.pop(),t.destroy()}},programs:o}}function Ls(){let t=new WeakMap;return{get:function(e){let n=t.get(e);return void 0===n&&(n={},t.set(e,n)),n},remove:function(e){t.delete(e)},update:function(e,n,i){t.get(e)[n]=i},dispose:function(){t=new WeakMap}}}function Rs(t,e){return t.groupOrder!==e.groupOrder?t.groupOrder-e.groupOrder:t.renderOrder!==e.renderOrder?t.renderOrder-e.renderOrder:t.program!==e.program?t.program.id-e.program.id:t.material.id!==e.material.id?t.material.id-e.material.id:t.z!==e.z?t.z-e.z:t.id-e.id}function Cs(t,e){return t.groupOrder!==e.groupOrder?t.groupOrder-e.groupOrder:t.renderOrder!==e.renderOrder?t.renderOrder-e.renderOrder:t.z!==e.z?e.z-t.z:t.id-e.id}function Ps(t){const e=[];let n=0;const i=[],r=[],s=[],a={id:-1};function o(i,r,s,o,l,c){let h=e[n];const u=t.get(s);return void 0===h?(h={id:i.id,object:i,geometry:r,material:s,program:u.program||a,groupOrder:o,renderOrder:i.renderOrder,z:l,group:c},e[n]=h):(h.id=i.id,h.object=i,h.geometry=r,h.material=s,h.program=u.program||a,h.groupOrder=o,h.renderOrder=i.renderOrder,h.z=l,h.group=c),n++,h}return{opaque:i,transmissive:r,transparent:s,init:function(){n=0,i.length=0,r.length=0,s.length=0},push:function(t,e,n,a,l,c){const h=o(t,e,n,a,l,c);n.transmission>0?r.push(h):!0===n.transparent?s.push(h):i.push(h)},unshift:function(t,e,n,a,l,c){const h=o(t,e,n,a,l,c);n.transmission>0?r.unshift(h):!0===n.transparent?s.unshift(h):i.unshift(h)},finish:function(){for(let t=n,i=e.length;t<i;t++){const n=e[t];if(null===n.id)break;n.id=null,n.object=null,n.geometry=null,n.material=null,n.program=null,n.group=null}},sort:function(t,e){i.length>1&&i.sort(t||Rs),r.length>1&&r.sort(e||Cs),s.length>1&&s.sort(e||Cs)}}}function Is(t){let e=new WeakMap;return{get:function(n,i){let r;return!1===e.has(n)?(r=new Ps(t),e.set(n,[r])):i>=e.get(n).length?(r=new Ps(t),e.get(n).push(r)):r=e.get(n)[i],r},dispose:function(){e=new WeakMap}}}function Ds(){const t={};return{get:function(e){if(void 0!==t[e.id])return t[e.id];let n;switch(e.type){case"DirectionalLight":n={direction:new zt,color:new rn};break;case"SpotLight":n={position:new zt,direction:new zt,color:new rn,distance:0,coneCos:0,penumbraCos:0,decay:0};break;case"PointLight":n={position:new zt,color:new rn,distance:0,decay:0};break;case"HemisphereLight":n={direction:new zt,skyColor:new rn,groundColor:new rn};break;case"RectAreaLight":n={color:new rn,position:new zt,halfWidth:new zt,halfHeight:new zt}}return t[e.id]=n,n}}}let Ns=0;function zs(t,e){return(e.castShadow?1:0)-(t.castShadow?1:0)}function Bs(t,e){const n=new Ds,i=function(){const t={};return{get:function(e){if(void 0!==t[e.id])return t[e.id];let n;switch(e.type){case"DirectionalLight":case"SpotLight":n={shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new yt};break;case"PointLight":n={shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new yt,shadowCameraNear:1,shadowCameraFar:1e3}}return t[e.id]=n,n}}}(),r={version:0,hash:{directionalLength:-1,pointLength:-1,spotLength:-1,rectAreaLength:-1,hemiLength:-1,numDirectionalShadows:-1,numPointShadows:-1,numSpotShadows:-1},ambient:[0,0,0],probe:[],directional:[],directionalShadow:[],directionalShadowMap:[],directionalShadowMatrix:[],spot:[],spotShadow:[],spotShadowMap:[],spotShadowMatrix:[],rectArea:[],rectAreaLTC1:null,rectAreaLTC2:null,point:[],pointShadow:[],pointShadowMap:[],pointShadowMatrix:[],hemi:[]};for(let t=0;t<9;t++)r.probe.push(new zt);const s=new zt,a=new de,o=new de;return{setup:function(s,a){let o=0,l=0,c=0;for(let t=0;t<9;t++)r.probe[t].set(0,0,0);let h=0,u=0,d=0,p=0,m=0,f=0,g=0,v=0;s.sort(zs);const y=!0!==a?Math.PI:1;for(let t=0,e=s.length;t<e;t++){const e=s[t],a=e.color,x=e.intensity,_=e.distance,M=e.shadow&&e.shadow.map?e.shadow.map.texture:null;if(e.isAmbientLight)o+=a.r*x*y,l+=a.g*x*y,c+=a.b*x*y;else if(e.isLightProbe)for(let t=0;t<9;t++)r.probe[t].addScaledVector(e.sh.coefficients[t],x);else if(e.isDirectionalLight){const t=n.get(e);if(t.color.copy(e.color).multiplyScalar(e.intensity*y),e.castShadow){const t=e.shadow,n=i.get(e);n.shadowBias=t.bias,n.shadowNormalBias=t.normalBias,n.shadowRadius=t.radius,n.shadowMapSize=t.mapSize,r.directionalShadow[h]=n,r.directionalShadowMap[h]=M,r.directionalShadowMatrix[h]=e.shadow.matrix,f++}r.directional[h]=t,h++}else if(e.isSpotLight){const t=n.get(e);if(t.position.setFromMatrixPosition(e.matrixWorld),t.color.copy(a).multiplyScalar(x*y),t.distance=_,t.coneCos=Math.cos(e.angle),t.penumbraCos=Math.cos(e.angle*(1-e.penumbra)),t.decay=e.decay,e.castShadow){const t=e.shadow,n=i.get(e);n.shadowBias=t.bias,n.shadowNormalBias=t.normalBias,n.shadowRadius=t.radius,n.shadowMapSize=t.mapSize,r.spotShadow[d]=n,r.spotShadowMap[d]=M,r.spotShadowMatrix[d]=e.shadow.matrix,v++}r.spot[d]=t,d++}else if(e.isRectAreaLight){const t=n.get(e);t.color.copy(a).multiplyScalar(x),t.halfWidth.set(.5*e.width,0,0),t.halfHeight.set(0,.5*e.height,0),r.rectArea[p]=t,p++}else if(e.isPointLight){const t=n.get(e);if(t.color.copy(e.color).multiplyScalar(e.intensity*y),t.distance=e.distance,t.decay=e.decay,e.castShadow){const t=e.shadow,n=i.get(e);n.shadowBias=t.bias,n.shadowNormalBias=t.normalBias,n.shadowRadius=t.radius,n.shadowMapSize=t.mapSize,n.shadowCameraNear=t.camera.near,n.shadowCameraFar=t.camera.far,r.pointShadow[u]=n,r.pointShadowMap[u]=M,r.pointShadowMatrix[u]=e.shadow.matrix,g++}r.point[u]=t,u++}else if(e.isHemisphereLight){const t=n.get(e);t.skyColor.copy(e.color).multiplyScalar(x*y),t.groundColor.copy(e.groundColor).multiplyScalar(x*y),r.hemi[m]=t,m++}}p>0&&(e.isWebGL2||!0===t.has("OES_texture_float_linear")?(r.rectAreaLTC1=mi.LTC_FLOAT_1,r.rectAreaLTC2=mi.LTC_FLOAT_2):!0===t.has("OES_texture_half_float_linear")?(r.rectAreaLTC1=mi.LTC_HALF_1,r.rectAreaLTC2=mi.LTC_HALF_2):console.error("THREE.WebGLRenderer: Unable to use RectAreaLight. Missing WebGL extensions.")),r.ambient[0]=o,r.ambient[1]=l,r.ambient[2]=c;const x=r.hash;x.directionalLength===h&&x.pointLength===u&&x.spotLength===d&&x.rectAreaLength===p&&x.hemiLength===m&&x.numDirectionalShadows===f&&x.numPointShadows===g&&x.numSpotShadows===v||(r.directional.length=h,r.spot.length=d,r.rectArea.length=p,r.point.length=u,r.hemi.length=m,r.directionalShadow.length=f,r.directionalShadowMap.length=f,r.pointShadow.length=g,r.pointShadowMap.length=g,r.spotShadow.length=v,r.spotShadowMap.length=v,r.directionalShadowMatrix.length=f,r.pointShadowMatrix.length=g,r.spotShadowMatrix.length=v,x.directionalLength=h,x.pointLength=u,x.spotLength=d,x.rectAreaLength=p,x.hemiLength=m,x.numDirectionalShadows=f,x.numPointShadows=g,x.numSpotShadows=v,r.version=Ns++)},setupView:function(t,e){let n=0,i=0,l=0,c=0,h=0;const u=e.matrixWorldInverse;for(let e=0,d=t.length;e<d;e++){const d=t[e];if(d.isDirectionalLight){const t=r.directional[n];t.direction.setFromMatrixPosition(d.matrixWorld),s.setFromMatrixPosition(d.target.matrixWorld),t.direction.sub(s),t.direction.transformDirection(u),n++}else if(d.isSpotLight){const t=r.spot[l];t.position.setFromMatrixPosition(d.matrixWorld),t.position.applyMatrix4(u),t.direction.setFromMatrixPosition(d.matrixWorld),s.setFromMatrixPosition(d.target.matrixWorld),t.direction.sub(s),t.direction.transformDirection(u),l++}else if(d.isRectAreaLight){const t=r.rectArea[c];t.position.setFromMatrixPosition(d.matrixWorld),t.position.applyMatrix4(u),o.identity(),a.copy(d.matrixWorld),a.premultiply(u),o.extractRotation(a),t.halfWidth.set(.5*d.width,0,0),t.halfHeight.set(0,.5*d.height,0),t.halfWidth.applyMatrix4(o),t.halfHeight.applyMatrix4(o),c++}else if(d.isPointLight){const t=r.point[i];t.position.setFromMatrixPosition(d.matrixWorld),t.position.applyMatrix4(u),i++}else if(d.isHemisphereLight){const t=r.hemi[h];t.direction.setFromMatrixPosition(d.matrixWorld),t.direction.transformDirection(u),t.direction.normalize(),h++}}},state:r}}function Fs(t,e){const n=new Bs(t,e),i=[],r=[];return{init:function(){i.length=0,r.length=0},state:{lightsArray:i,shadowsArray:r,lights:n},setupLights:function(t){n.setup(i,t)},setupLightsView:function(t){n.setupView(i,t)},pushLight:function(t){i.push(t)},pushShadow:function(t){r.push(t)}}}function Os(t,e){let n=new WeakMap;return{get:function(i,r=0){let s;return!1===n.has(i)?(s=new Fs(t,e),n.set(i,[s])):r>=n.get(i).length?(s=new Fs(t,e),n.get(i).push(s)):s=n.get(i)[r],s},dispose:function(){n=new WeakMap}}}class Us extends Ze{constructor(t){super(),this.type="MeshDepthMaterial",this.depthPacking=3200,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.fog=!1,this.setValues(t)}copy(t){return super.copy(t),this.depthPacking=t.depthPacking,this.map=t.map,this.alphaMap=t.alphaMap,this.displacementMap=t.displacementMap,this.displacementScale=t.displacementScale,this.displacementBias=t.displacementBias,this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this}}Us.prototype.isMeshDepthMaterial=!0;class Hs extends Ze{constructor(t){super(),this.type="MeshDistanceMaterial",this.referencePosition=new zt,this.nearDistance=1,this.farDistance=1e3,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.fog=!1,this.setValues(t)}copy(t){return super.copy(t),this.referencePosition.copy(t.referencePosition),this.nearDistance=t.nearDistance,this.farDistance=t.farDistance,this.map=t.map,this.alphaMap=t.alphaMap,this.displacementMap=t.displacementMap,this.displacementScale=t.displacementScale,this.displacementBias=t.displacementBias,this}}Hs.prototype.isMeshDistanceMaterial=!0;function Gs(t,e,n){let i=new ci;const r=new yt,s=new yt,a=new Ct,o=new Us({depthPacking:3201}),l=new Hs,c={},h=n.maxTextureSize,u={0:1,1:0,2:2},d=new Zn({defines:{VSM_SAMPLES:8},uniforms:{shadow_pass:{value:null},resolution:{value:new yt},radius:{value:4}},vertexShader:"void main() {\n\tgl_Position = vec4( position, 1.0 );\n}",fragmentShader:"uniform sampler2D shadow_pass;\nuniform vec2 resolution;\nuniform float radius;\n#include <packing>\nvoid main() {\n\tconst float samples = float( VSM_SAMPLES );\n\tfloat mean = 0.0;\n\tfloat squared_mean = 0.0;\n\tfloat uvStride = samples <= 1.0 ? 0.0 : 2.0 / ( samples - 1.0 );\n\tfloat uvStart = samples <= 1.0 ? 0.0 : - 1.0;\n\tfor ( float i = 0.0; i < samples; i ++ ) {\n\t\tfloat uvOffset = uvStart + i * uvStride;\n\t\t#ifdef HORIZONTAL_PASS\n\t\t\tvec2 distribution = unpackRGBATo2Half( texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( uvOffset, 0.0 ) * radius ) / resolution ) );\n\t\t\tmean += distribution.x;\n\t\t\tsquared_mean += distribution.y * distribution.y + distribution.x * distribution.x;\n\t\t#else\n\t\t\tfloat depth = unpackRGBAToDepth( texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( 0.0, uvOffset ) * radius ) / resolution ) );\n\t\t\tmean += depth;\n\t\t\tsquared_mean += depth * depth;\n\t\t#endif\n\t}\n\tmean = mean / samples;\n\tsquared_mean = squared_mean / samples;\n\tfloat std_dev = sqrt( squared_mean - mean * mean );\n\tgl_FragColor = pack2HalfToRGBA( vec2( mean, std_dev ) );\n}"}),m=d.clone();m.defines.HORIZONTAL_PASS=1;const f=new En;f.setAttribute("position",new ln(new Float32Array([-1,-1,.5,3,-1,.5,-1,3,.5]),3));const v=new Wn(f,d),y=this;function x(n,i){const r=e.update(v);d.defines.VSM_SAMPLES!==n.blurSamples&&(d.defines.VSM_SAMPLES=n.blurSamples,m.defines.VSM_SAMPLES=n.blurSamples,d.needsUpdate=!0,m.needsUpdate=!0),d.uniforms.shadow_pass.value=n.map.texture,d.uniforms.resolution.value=n.mapSize,d.uniforms.radius.value=n.radius,t.setRenderTarget(n.mapPass),t.clear(),t.renderBufferDirect(i,null,r,d,v,null),m.uniforms.shadow_pass.value=n.mapPass.texture,m.uniforms.resolution.value=n.mapSize,m.uniforms.radius.value=n.radius,t.setRenderTarget(n.map),t.clear(),t.renderBufferDirect(i,null,r,m,v,null)}function _(e,n,i,r,s,a,h){let d=null;const p=!0===r.isPointLight?e.customDistanceMaterial:e.customDepthMaterial;if(d=void 0!==p?p:!0===r.isPointLight?l:o,t.localClippingEnabled&&!0===i.clipShadows&&0!==i.clippingPlanes.length||i.displacementMap&&0!==i.displacementScale||i.alphaMap&&i.alphaTest>0){const t=d.uuid,e=i.uuid;let n=c[t];void 0===n&&(n={},c[t]=n);let r=n[e];void 0===r&&(r=d.clone(),n[e]=r),d=r}return d.visible=i.visible,d.wireframe=i.wireframe,d.side=3===h?null!==i.shadowSide?i.shadowSide:i.side:null!==i.shadowSide?i.shadowSide:u[i.side],d.alphaMap=i.alphaMap,d.alphaTest=i.alphaTest,d.clipShadows=i.clipShadows,d.clippingPlanes=i.clippingPlanes,d.clipIntersection=i.clipIntersection,d.displacementMap=i.displacementMap,d.displacementScale=i.displacementScale,d.displacementBias=i.displacementBias,d.wireframeLinewidth=i.wireframeLinewidth,d.linewidth=i.linewidth,!0===r.isPointLight&&!0===d.isMeshDistanceMaterial&&(d.referencePosition.setFromMatrixPosition(r.matrixWorld),d.nearDistance=s,d.farDistance=a),d}function M(n,r,s,a,o){if(!1===n.visible)return;if(n.layers.test(r.layers)&&(n.isMesh||n.isLine||n.isPoints)&&(n.castShadow||n.receiveShadow&&3===o)&&(!n.frustumCulled||i.intersectsObject(n))){n.modelViewMatrix.multiplyMatrices(s.matrixWorldInverse,n.matrixWorld);const i=e.update(n),r=n.material;if(Array.isArray(r)){const e=i.groups;for(let l=0,c=e.length;l<c;l++){const c=e[l],h=r[c.materialIndex];if(h&&h.visible){const e=_(n,0,h,a,s.near,s.far,o);t.renderBufferDirect(s,null,i,e,n,c)}}}else if(r.visible){const e=_(n,0,r,a,s.near,s.far,o);t.renderBufferDirect(s,null,i,e,n,null)}}const l=n.children;for(let t=0,e=l.length;t<e;t++)M(l[t],r,s,a,o)}this.enabled=!1,this.autoUpdate=!0,this.needsUpdate=!1,this.type=1,this.render=function(e,n,o){if(!1===y.enabled)return;if(!1===y.autoUpdate&&!1===y.needsUpdate)return;if(0===e.length)return;const l=t.getRenderTarget(),c=t.getActiveCubeFace(),u=t.getActiveMipmapLevel(),d=t.state;d.setBlending(0),d.buffers.color.setClear(1,1,1,1),d.buffers.depth.setTest(!0),d.setScissorTest(!1);for(let l=0,c=e.length;l<c;l++){const c=e[l],u=c.shadow;if(void 0===u){console.warn("THREE.WebGLShadowMap:",c,"has no shadow.");continue}if(!1===u.autoUpdate&&!1===u.needsUpdate)continue;r.copy(u.mapSize);const m=u.getFrameExtents();if(r.multiply(m),s.copy(u.mapSize),(r.x>h||r.y>h)&&(r.x>h&&(s.x=Math.floor(h/m.x),r.x=s.x*m.x,u.mapSize.x=s.x),r.y>h&&(s.y=Math.floor(h/m.y),r.y=s.y*m.y,u.mapSize.y=s.y)),null===u.map&&!u.isPointLightShadow&&3===this.type){const t={minFilter:g,magFilter:g,format:E};u.map=new Pt(r.x,r.y,t),u.map.texture.name=c.name+".shadowMap",u.mapPass=new Pt(r.x,r.y,t),u.camera.updateProjectionMatrix()}if(null===u.map){const t={minFilter:p,magFilter:p,format:E};u.map=new Pt(r.x,r.y,t),u.map.texture.name=c.name+".shadowMap",u.camera.updateProjectionMatrix()}t.setRenderTarget(u.map),t.clear();const f=u.getViewportCount();for(let t=0;t<f;t++){const e=u.getViewport(t);a.set(s.x*e.x,s.y*e.y,s.x*e.z,s.y*e.w),d.viewport(a),u.updateMatrices(c,t),i=u.getFrustum(),M(n,o,u.camera,c,this.type)}u.isPointLightShadow||3!==this.type||x(u,o),u.needsUpdate=!1}y.needsUpdate=!1,t.setRenderTarget(l,c,u)}}function ks(t,e,i){const r=i.isWebGL2;const s=new function(){let e=!1;const n=new Ct;let i=null;const r=new Ct(0,0,0,0);return{setMask:function(n){i===n||e||(t.colorMask(n,n,n,n),i=n)},setLocked:function(t){e=t},setClear:function(e,i,s,a,o){!0===o&&(e*=a,i*=a,s*=a),n.set(e,i,s,a),!1===r.equals(n)&&(t.clearColor(e,i,s,a),r.copy(n))},reset:function(){e=!1,i=null,r.set(-1,0,0,0)}}},a=new function(){let e=!1,n=null,i=null,r=null;return{setTest:function(t){t?O(2929):U(2929)},setMask:function(i){n===i||e||(t.depthMask(i),n=i)},setFunc:function(e){if(i!==e){if(e)switch(e){case 0:t.depthFunc(512);break;case 1:t.depthFunc(519);break;case 2:t.depthFunc(513);break;default:t.depthFunc(515);break;case 4:t.depthFunc(514);break;case 5:t.depthFunc(518);break;case 6:t.depthFunc(516);break;case 7:t.depthFunc(517)}else t.depthFunc(515);i=e}},setLocked:function(t){e=t},setClear:function(e){r!==e&&(t.clearDepth(e),r=e)},reset:function(){e=!1,n=null,i=null,r=null}}},o=new function(){let e=!1,n=null,i=null,r=null,s=null,a=null,o=null,l=null,c=null;return{setTest:function(t){e||(t?O(2960):U(2960))},setMask:function(i){n===i||e||(t.stencilMask(i),n=i)},setFunc:function(e,n,a){i===e&&r===n&&s===a||(t.stencilFunc(e,n,a),i=e,r=n,s=a)},setOp:function(e,n,i){a===e&&o===n&&l===i||(t.stencilOp(e,n,i),a=e,o=n,l=i)},setLocked:function(t){e=t},setClear:function(e){c!==e&&(t.clearStencil(e),c=e)},reset:function(){e=!1,n=null,i=null,r=null,s=null,a=null,o=null,l=null,c=null}}};let l={},c=null,h={},u=null,d=!1,p=null,m=null,f=null,g=null,v=null,y=null,x=null,_=!1,M=null,b=null,w=null,S=null,T=null;const E=t.getParameter(35661);let A=!1,L=0;const R=t.getParameter(7938);-1!==R.indexOf("WebGL")?(L=parseFloat(/^WebGL (\d)/.exec(R)[1]),A=L>=1):-1!==R.indexOf("OpenGL ES")&&(L=parseFloat(/^OpenGL ES (\d)/.exec(R)[1]),A=L>=2);let C=null,P={};const I=t.getParameter(3088),D=t.getParameter(2978),N=(new Ct).fromArray(I),z=(new Ct).fromArray(D);function B(e,n,i){const r=new Uint8Array(4),s=t.createTexture();t.bindTexture(e,s),t.texParameteri(e,10241,9728),t.texParameteri(e,10240,9728);for(let e=0;e<i;e++)t.texImage2D(n+e,0,6408,1,1,0,6408,5121,r);return s}const F={};function O(e){!0!==l[e]&&(t.enable(e),l[e]=!0)}function U(e){!1!==l[e]&&(t.disable(e),l[e]=!1)}F[3553]=B(3553,3553,1),F[34067]=B(34067,34069,6),s.setClear(0,0,0,1),a.setClear(1),o.setClear(0),O(2929),a.setFunc(3),V(!1),W(1),O(2884),k(0);const H={[n]:32774,101:32778,102:32779};if(r)H[103]=32775,H[104]=32776;else{const t=e.get("EXT_blend_minmax");null!==t&&(H[103]=t.MIN_EXT,H[104]=t.MAX_EXT)}const G={200:0,201:1,202:768,204:770,210:776,208:774,206:772,203:769,205:771,209:775,207:773};function k(e,i,r,s,a,o,l,c){if(0!==e){if(!1===d&&(O(3042),d=!0),5===e)a=a||i,o=o||r,l=l||s,i===m&&a===v||(t.blendEquationSeparate(H[i],H[a]),m=i,v=a),r===f&&s===g&&o===y&&l===x||(t.blendFuncSeparate(G[r],G[s],G[o],G[l]),f=r,g=s,y=o,x=l),p=e,_=null;else if(e!==p||c!==_){if(m===n&&v===n||(t.blendEquation(32774),m=n,v=n),c)switch(e){case 1:t.blendFuncSeparate(1,771,1,771);break;case 2:t.blendFunc(1,1);break;case 3:t.blendFuncSeparate(0,0,769,771);break;case 4:t.blendFuncSeparate(0,768,0,770);break;default:console.error("THREE.WebGLState: Invalid blending: ",e)}else switch(e){case 1:t.blendFuncSeparate(770,771,1,771);break;case 2:t.blendFunc(770,1);break;case 3:t.blendFunc(0,769);break;case 4:t.blendFunc(0,768);break;default:console.error("THREE.WebGLState: Invalid blending: ",e)}f=null,g=null,y=null,x=null,p=e,_=c}}else!0===d&&(U(3042),d=!1)}function V(e){M!==e&&(e?t.frontFace(2304):t.frontFace(2305),M=e)}function W(e){0!==e?(O(2884),e!==b&&(1===e?t.cullFace(1029):2===e?t.cullFace(1028):t.cullFace(1032))):U(2884),b=e}function j(e,n,i){e?(O(32823),S===n&&T===i||(t.polygonOffset(n,i),S=n,T=i)):U(32823)}function q(e){void 0===e&&(e=33984+E-1),C!==e&&(t.activeTexture(e),C=e)}return{buffers:{color:s,depth:a,stencil:o},enable:O,disable:U,bindFramebuffer:function(e,n){return null===n&&null!==c&&(n=c),h[e]!==n&&(t.bindFramebuffer(e,n),h[e]=n,r&&(36009===e&&(h[36160]=n),36160===e&&(h[36009]=n)),!0)},bindXRFramebuffer:function(e){e!==c&&(t.bindFramebuffer(36160,e),c=e)},useProgram:function(e){return u!==e&&(t.useProgram(e),u=e,!0)},setBlending:k,setMaterial:function(t,e){2===t.side?U(2884):O(2884);let n=1===t.side;e&&(n=!n),V(n),1===t.blending&&!1===t.transparent?k(0):k(t.blending,t.blendEquation,t.blendSrc,t.blendDst,t.blendEquationAlpha,t.blendSrcAlpha,t.blendDstAlpha,t.premultipliedAlpha),a.setFunc(t.depthFunc),a.setTest(t.depthTest),a.setMask(t.depthWrite),s.setMask(t.colorWrite);const i=t.stencilWrite;o.setTest(i),i&&(o.setMask(t.stencilWriteMask),o.setFunc(t.stencilFunc,t.stencilRef,t.stencilFuncMask),o.setOp(t.stencilFail,t.stencilZFail,t.stencilZPass)),j(t.polygonOffset,t.polygonOffsetFactor,t.polygonOffsetUnits),!0===t.alphaToCoverage?O(32926):U(32926)},setFlipSided:V,setCullFace:W,setLineWidth:function(e){e!==w&&(A&&t.lineWidth(e),w=e)},setPolygonOffset:j,setScissorTest:function(t){t?O(3089):U(3089)},activeTexture:q,bindTexture:function(e,n){null===C&&q();let i=P[C];void 0===i&&(i={type:void 0,texture:void 0},P[C]=i),i.type===e&&i.texture===n||(t.bindTexture(e,n||F[e]),i.type=e,i.texture=n)},unbindTexture:function(){const e=P[C];void 0!==e&&void 0!==e.type&&(t.bindTexture(e.type,null),e.type=void 0,e.texture=void 0)},compressedTexImage2D:function(){try{t.compressedTexImage2D.apply(t,arguments)}catch(t){console.error("THREE.WebGLState:",t)}},texImage2D:function(){try{t.texImage2D.apply(t,arguments)}catch(t){console.error("THREE.WebGLState:",t)}},texImage3D:function(){try{t.texImage3D.apply(t,arguments)}catch(t){console.error("THREE.WebGLState:",t)}},scissor:function(e){!1===N.equals(e)&&(t.scissor(e.x,e.y,e.z,e.w),N.copy(e))},viewport:function(e){!1===z.equals(e)&&(t.viewport(e.x,e.y,e.z,e.w),z.copy(e))},reset:function(){t.disable(3042),t.disable(2884),t.disable(2929),t.disable(32823),t.disable(3089),t.disable(2960),t.disable(32926),t.blendEquation(32774),t.blendFunc(1,0),t.blendFuncSeparate(1,0,1,0),t.colorMask(!0,!0,!0,!0),t.clearColor(0,0,0,0),t.depthMask(!0),t.depthFunc(513),t.clearDepth(1),t.stencilMask(4294967295),t.stencilFunc(519,0,4294967295),t.stencilOp(7680,7680,7680),t.clearStencil(0),t.cullFace(1029),t.frontFace(2305),t.polygonOffset(0,0),t.activeTexture(33984),t.bindFramebuffer(36160,null),!0===r&&(t.bindFramebuffer(36009,null),t.bindFramebuffer(36008,null)),t.useProgram(null),t.lineWidth(1),t.scissor(0,0,t.canvas.width,t.canvas.height),t.viewport(0,0,t.canvas.width,t.canvas.height),l={},C=null,P={},c=null,h={},u=null,d=!1,p=null,m=null,f=null,g=null,v=null,y=null,x=null,_=!1,M=null,b=null,w=null,S=null,T=null,N.set(0,0,t.canvas.width,t.canvas.height),z.set(0,0,t.canvas.width,t.canvas.height),s.reset(),a.reset(),o.reset()}}}function Vs(t,e,n,i,r,s,a){const o=r.isWebGL2,l=r.maxTextures,c=r.maxCubemapSize,x=r.maxTextureSize,R=r.maxSamples,C=new WeakMap;let P,I=!1;try{I="undefined"!=typeof OffscreenCanvas&&null!==new OffscreenCanvas(1,1).getContext("2d")}catch(t){}function D(t,e){return I?new OffscreenCanvas(t,e):wt("canvas")}function N(t,e,n,i){let r=1;if((t.width>i||t.height>i)&&(r=i/Math.max(t.width,t.height)),r<1||!0===e){if("undefined"!=typeof HTMLImageElement&&t instanceof HTMLImageElement||"undefined"!=typeof HTMLCanvasElement&&t instanceof HTMLCanvasElement||"undefined"!=typeof ImageBitmap&&t instanceof ImageBitmap){const i=e?gt:Math.floor,s=i(r*t.width),a=i(r*t.height);void 0===P&&(P=D(s,a));const o=n?D(s,a):P;o.width=s,o.height=a;return o.getContext("2d").drawImage(t,0,0,s,a),console.warn("THREE.WebGLRenderer: Texture has been resized from ("+t.width+"x"+t.height+") to ("+s+"x"+a+")."),o}return"data"in t&&console.warn("THREE.WebGLRenderer: Image in DataTexture is too big ("+t.width+"x"+t.height+")."),t}return t}function z(t){return mt(t.width)&&mt(t.height)}function B(t,e){return t.generateMipmaps&&e&&t.minFilter!==p&&t.minFilter!==g}function F(e,n,r,s,a=1){t.generateMipmap(e);i.get(n).__maxMipLevel=Math.log2(Math.max(r,s,a))}function O(n,i,r,s){if(!1===o)return i;if(null!==n){if(void 0!==t[n])return t[n];console.warn("THREE.WebGLRenderer: Attempt to use non-existing WebGL internal format '"+n+"'")}let a=i;return 6403===i&&(5126===r&&(a=33326),5131===r&&(a=33325),5121===r&&(a=33321)),6407===i&&(5126===r&&(a=34837),5131===r&&(a=34843),5121===r&&(a=32849)),6408===i&&(5126===r&&(a=34836),5131===r&&(a=34842),5121===r&&(a=s===Y?35907:32856)),33325!==a&&33326!==a&&34842!==a&&34836!==a||e.get("EXT_color_buffer_float"),a}function U(t){return t===p||t===m||t===f?9728:9729}function H(e){const n=e.target;n.removeEventListener("dispose",H),function(e){const n=i.get(e);if(void 0===n.__webglInit)return;t.deleteTexture(n.__webglTexture),i.remove(e)}(n),n.isVideoTexture&&C.delete(n),a.memory.textures--}function G(e){const n=e.target;n.removeEventListener("dispose",G),function(e){const n=e.texture,r=i.get(e),s=i.get(n);if(!e)return;void 0!==s.__webglTexture&&(t.deleteTexture(s.__webglTexture),a.memory.textures--);e.depthTexture&&e.depthTexture.dispose();if(e.isWebGLCubeRenderTarget)for(let e=0;e<6;e++)t.deleteFramebuffer(r.__webglFramebuffer[e]),r.__webglDepthbuffer&&t.deleteRenderbuffer(r.__webglDepthbuffer[e]);else t.deleteFramebuffer(r.__webglFramebuffer),r.__webglDepthbuffer&&t.deleteRenderbuffer(r.__webglDepthbuffer),r.__webglMultisampledFramebuffer&&t.deleteFramebuffer(r.__webglMultisampledFramebuffer),r.__webglColorRenderbuffer&&t.deleteRenderbuffer(r.__webglColorRenderbuffer),r.__webglDepthRenderbuffer&&t.deleteRenderbuffer(r.__webglDepthRenderbuffer);if(e.isWebGLMultipleRenderTargets)for(let e=0,r=n.length;e<r;e++){const r=i.get(n[e]);r.__webglTexture&&(t.deleteTexture(r.__webglTexture),a.memory.textures--),i.remove(n[e])}i.remove(n),i.remove(e)}(n)}let k=0;function V(t,e){const r=i.get(t);if(t.isVideoTexture&&function(t){const e=a.render.frame;C.get(t)!==e&&(C.set(t,e),t.update())}(t),t.version>0&&r.__version!==t.version){const n=t.image;if(void 0===n)console.warn("THREE.WebGLRenderer: Texture marked for update but image is undefined");else{if(!1!==n.complete)return void Z(r,t,e);console.warn("THREE.WebGLRenderer: Texture marked for update but image is incomplete")}}n.activeTexture(33984+e),n.bindTexture(3553,r.__webglTexture)}function W(e,r){const a=i.get(e);e.version>0&&a.__version!==e.version?function(e,i,r){if(6!==i.image.length)return;J(e,i),n.activeTexture(33984+r),n.bindTexture(34067,e.__webglTexture),t.pixelStorei(37440,i.flipY),t.pixelStorei(37441,i.premultiplyAlpha),t.pixelStorei(3317,i.unpackAlignment),t.pixelStorei(37443,0);const a=i&&(i.isCompressedTexture||i.image[0].isCompressedTexture),l=i.image[0]&&i.image[0].isDataTexture,h=[];for(let t=0;t<6;t++)h[t]=a||l?l?i.image[t].image:i.image[t]:N(i.image[t],!1,!0,c);const u=h[0],d=z(u)||o,p=s.convert(i.format),m=s.convert(i.type),f=O(i.internalFormat,p,m,i.encoding);let g;if(X(34067,i,d),a){for(let t=0;t<6;t++){g=h[t].mipmaps;for(let e=0;e<g.length;e++){const r=g[e];i.format!==E&&i.format!==T?null!==p?n.compressedTexImage2D(34069+t,e,f,r.width,r.height,0,r.data):console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .setTextureCube()"):n.texImage2D(34069+t,e,f,r.width,r.height,0,p,m,r.data)}}e.__maxMipLevel=g.length-1}else{g=i.mipmaps;for(let t=0;t<6;t++)if(l){n.texImage2D(34069+t,0,f,h[t].width,h[t].height,0,p,m,h[t].data);for(let e=0;e<g.length;e++){const i=g[e].image[t].image;n.texImage2D(34069+t,e+1,f,i.width,i.height,0,p,m,i.data)}}else{n.texImage2D(34069+t,0,f,p,m,h[t]);for(let e=0;e<g.length;e++){const i=g[e];n.texImage2D(34069+t,e+1,f,p,m,i.image[t])}}e.__maxMipLevel=g.length}B(i,d)&&F(34067,i,u.width,u.height);e.__version=i.version,i.onUpdate&&i.onUpdate(i)}(a,e,r):(n.activeTexture(33984+r),n.bindTexture(34067,a.__webglTexture))}const j={[h]:10497,[u]:33071,[d]:33648},q={[p]:9728,[m]:9984,[f]:9986,[g]:9729,[v]:9985,[y]:9987};function X(n,s,a){if(a?(t.texParameteri(n,10242,j[s.wrapS]),t.texParameteri(n,10243,j[s.wrapT]),32879!==n&&35866!==n||t.texParameteri(n,32882,j[s.wrapR]),t.texParameteri(n,10240,q[s.magFilter]),t.texParameteri(n,10241,q[s.minFilter])):(t.texParameteri(n,10242,33071),t.texParameteri(n,10243,33071),32879!==n&&35866!==n||t.texParameteri(n,32882,33071),s.wrapS===u&&s.wrapT===u||console.warn("THREE.WebGLRenderer: Texture is not power of two. Texture.wrapS and Texture.wrapT should be set to THREE.ClampToEdgeWrapping."),t.texParameteri(n,10240,U(s.magFilter)),t.texParameteri(n,10241,U(s.minFilter)),s.minFilter!==p&&s.minFilter!==g&&console.warn("THREE.WebGLRenderer: Texture is not power of two. Texture.minFilter should be set to THREE.NearestFilter or THREE.LinearFilter.")),!0===e.has("EXT_texture_filter_anisotropic")){const a=e.get("EXT_texture_filter_anisotropic");if(s.type===b&&!1===e.has("OES_texture_float_linear"))return;if(!1===o&&s.type===w&&!1===e.has("OES_texture_half_float_linear"))return;(s.anisotropy>1||i.get(s).__currentAnisotropy)&&(t.texParameterf(n,a.TEXTURE_MAX_ANISOTROPY_EXT,Math.min(s.anisotropy,r.getMaxAnisotropy())),i.get(s).__currentAnisotropy=s.anisotropy)}}function J(e,n){void 0===e.__webglInit&&(e.__webglInit=!0,n.addEventListener("dispose",H),e.__webglTexture=t.createTexture(),a.memory.textures++)}function Z(e,i,r){let a=3553;i.isDataTexture2DArray&&(a=35866),i.isDataTexture3D&&(a=32879),J(e,i),n.activeTexture(33984+r),n.bindTexture(a,e.__webglTexture),t.pixelStorei(37440,i.flipY),t.pixelStorei(37441,i.premultiplyAlpha),t.pixelStorei(3317,i.unpackAlignment),t.pixelStorei(37443,0);const l=function(t){return!o&&(t.wrapS!==u||t.wrapT!==u||t.minFilter!==p&&t.minFilter!==g)}(i)&&!1===z(i.image),c=N(i.image,l,!1,x),h=z(c)||o,d=s.convert(i.format);let m,f=s.convert(i.type),v=O(i.internalFormat,d,f,i.encoding);X(a,i,h);const y=i.mipmaps;if(i.isDepthTexture)v=6402,o?v=i.type===b?36012:i.type===M?33190:i.type===S?35056:33189:i.type===b&&console.error("WebGLRenderer: Floating point depth texture requires WebGL2."),i.format===A&&6402===v&&i.type!==_&&i.type!==M&&(console.warn("THREE.WebGLRenderer: Use UnsignedShortType or UnsignedIntType for DepthFormat DepthTexture."),i.type=_,f=s.convert(i.type)),i.format===L&&6402===v&&(v=34041,i.type!==S&&(console.warn("THREE.WebGLRenderer: Use UnsignedInt248Type for DepthStencilFormat DepthTexture."),i.type=S,f=s.convert(i.type))),n.texImage2D(3553,0,v,c.width,c.height,0,d,f,null);else if(i.isDataTexture)if(y.length>0&&h){for(let t=0,e=y.length;t<e;t++)m=y[t],n.texImage2D(3553,t,v,m.width,m.height,0,d,f,m.data);i.generateMipmaps=!1,e.__maxMipLevel=y.length-1}else n.texImage2D(3553,0,v,c.width,c.height,0,d,f,c.data),e.__maxMipLevel=0;else if(i.isCompressedTexture){for(let t=0,e=y.length;t<e;t++)m=y[t],i.format!==E&&i.format!==T?null!==d?n.compressedTexImage2D(3553,t,v,m.width,m.height,0,m.data):console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()"):n.texImage2D(3553,t,v,m.width,m.height,0,d,f,m.data);e.__maxMipLevel=y.length-1}else if(i.isDataTexture2DArray)n.texImage3D(35866,0,v,c.width,c.height,c.depth,0,d,f,c.data),e.__maxMipLevel=0;else if(i.isDataTexture3D)n.texImage3D(32879,0,v,c.width,c.height,c.depth,0,d,f,c.data),e.__maxMipLevel=0;else if(y.length>0&&h){for(let t=0,e=y.length;t<e;t++)m=y[t],n.texImage2D(3553,t,v,d,f,m);i.generateMipmaps=!1,e.__maxMipLevel=y.length-1}else n.texImage2D(3553,0,v,d,f,c),e.__maxMipLevel=0;B(i,h)&&F(a,i,c.width,c.height),e.__version=i.version,i.onUpdate&&i.onUpdate(i)}function Q(e,r,a,o,l){const c=s.convert(a.format),h=s.convert(a.type),u=O(a.internalFormat,c,h,a.encoding);32879===l||35866===l?n.texImage3D(l,0,u,r.width,r.height,r.depth,0,c,h,null):n.texImage2D(l,0,u,r.width,r.height,0,c,h,null),n.bindFramebuffer(36160,e),t.framebufferTexture2D(36160,o,l,i.get(a).__webglTexture,0),n.bindFramebuffer(36160,null)}function K(e,n,i){if(t.bindRenderbuffer(36161,e),n.depthBuffer&&!n.stencilBuffer){let r=33189;if(i){const e=n.depthTexture;e&&e.isDepthTexture&&(e.type===b?r=36012:e.type===M&&(r=33190));const i=tt(n);t.renderbufferStorageMultisample(36161,i,r,n.width,n.height)}else t.renderbufferStorage(36161,r,n.width,n.height);t.framebufferRenderbuffer(36160,36096,36161,e)}else if(n.depthBuffer&&n.stencilBuffer){if(i){const e=tt(n);t.renderbufferStorageMultisample(36161,e,35056,n.width,n.height)}else t.renderbufferStorage(36161,34041,n.width,n.height);t.framebufferRenderbuffer(36160,33306,36161,e)}else{const e=!0===n.isWebGLMultipleRenderTargets?n.texture[0]:n.texture,r=s.convert(e.format),a=s.convert(e.type),o=O(e.internalFormat,r,a,e.encoding);if(i){const e=tt(n);t.renderbufferStorageMultisample(36161,e,o,n.width,n.height)}else t.renderbufferStorage(36161,o,n.width,n.height)}t.bindRenderbuffer(36161,null)}function $(e){const r=i.get(e),s=!0===e.isWebGLCubeRenderTarget;if(e.depthTexture){if(s)throw new Error("target.depthTexture not supported in Cube render targets");!function(e,r){if(r&&r.isWebGLCubeRenderTarget)throw new Error("Depth Texture with cube render targets is not supported");if(n.bindFramebuffer(36160,e),!r.depthTexture||!r.depthTexture.isDepthTexture)throw new Error("renderTarget.depthTexture must be an instance of THREE.DepthTexture");i.get(r.depthTexture).__webglTexture&&r.depthTexture.image.width===r.width&&r.depthTexture.image.height===r.height||(r.depthTexture.image.width=r.width,r.depthTexture.image.height=r.height,r.depthTexture.needsUpdate=!0),V(r.depthTexture,0);const s=i.get(r.depthTexture).__webglTexture;if(r.depthTexture.format===A)t.framebufferTexture2D(36160,36096,3553,s,0);else{if(r.depthTexture.format!==L)throw new Error("Unknown depthTexture format");t.framebufferTexture2D(36160,33306,3553,s,0)}}(r.__webglFramebuffer,e)}else if(s){r.__webglDepthbuffer=[];for(let i=0;i<6;i++)n.bindFramebuffer(36160,r.__webglFramebuffer[i]),r.__webglDepthbuffer[i]=t.createRenderbuffer(),K(r.__webglDepthbuffer[i],e,!1)}else n.bindFramebuffer(36160,r.__webglFramebuffer),r.__webglDepthbuffer=t.createRenderbuffer(),K(r.__webglDepthbuffer,e,!1);n.bindFramebuffer(36160,null)}function tt(t){return o&&t.isWebGLMultisampleRenderTarget?Math.min(R,t.samples):0}let et=!1,nt=!1;this.allocateTextureUnit=function(){const t=k;return t>=l&&console.warn("THREE.WebGLTextures: Trying to use "+t+" texture units while this GPU supports only "+l),k+=1,t},this.resetTextureUnits=function(){k=0},this.setTexture2D=V,this.setTexture2DArray=function(t,e){const r=i.get(t);t.version>0&&r.__version!==t.version?Z(r,t,e):(n.activeTexture(33984+e),n.bindTexture(35866,r.__webglTexture))},this.setTexture3D=function(t,e){const r=i.get(t);t.version>0&&r.__version!==t.version?Z(r,t,e):(n.activeTexture(33984+e),n.bindTexture(32879,r.__webglTexture))},this.setTextureCube=W,this.setupRenderTarget=function(e){const l=e.texture,c=i.get(e),h=i.get(l);e.addEventListener("dispose",G),!0!==e.isWebGLMultipleRenderTargets&&(h.__webglTexture=t.createTexture(),h.__version=l.version,a.memory.textures++);const u=!0===e.isWebGLCubeRenderTarget,d=!0===e.isWebGLMultipleRenderTargets,p=!0===e.isWebGLMultisampleRenderTarget,m=l.isDataTexture3D||l.isDataTexture2DArray,f=z(e)||o;if(!o||l.format!==T||l.type!==b&&l.type!==w||(l.format=E,console.warn("THREE.WebGLRenderer: Rendering to textures with RGB format is not supported. Using RGBA format instead.")),u){c.__webglFramebuffer=[];for(let e=0;e<6;e++)c.__webglFramebuffer[e]=t.createFramebuffer()}else if(c.__webglFramebuffer=t.createFramebuffer(),d)if(r.drawBuffers){const n=e.texture;for(let e=0,r=n.length;e<r;e++){const r=i.get(n[e]);void 0===r.__webglTexture&&(r.__webglTexture=t.createTexture(),a.memory.textures++)}}else console.warn("THREE.WebGLRenderer: WebGLMultipleRenderTargets can only be used with WebGL2 or WEBGL_draw_buffers extension.");else if(p)if(o){c.__webglMultisampledFramebuffer=t.createFramebuffer(),c.__webglColorRenderbuffer=t.createRenderbuffer(),t.bindRenderbuffer(36161,c.__webglColorRenderbuffer);const i=s.convert(l.format),r=s.convert(l.type),a=O(l.internalFormat,i,r,l.encoding),o=tt(e);t.renderbufferStorageMultisample(36161,o,a,e.width,e.height),n.bindFramebuffer(36160,c.__webglMultisampledFramebuffer),t.framebufferRenderbuffer(36160,36064,36161,c.__webglColorRenderbuffer),t.bindRenderbuffer(36161,null),e.depthBuffer&&(c.__webglDepthRenderbuffer=t.createRenderbuffer(),K(c.__webglDepthRenderbuffer,e,!0)),n.bindFramebuffer(36160,null)}else console.warn("THREE.WebGLRenderer: WebGLMultisampleRenderTarget can only be used with WebGL2.");if(u){n.bindTexture(34067,h.__webglTexture),X(34067,l,f);for(let t=0;t<6;t++)Q(c.__webglFramebuffer[t],e,l,36064,34069+t);B(l,f)&&F(34067,l,e.width,e.height),n.unbindTexture()}else if(d){const t=e.texture;for(let r=0,s=t.length;r<s;r++){const s=t[r],a=i.get(s);n.bindTexture(3553,a.__webglTexture),X(3553,s,f),Q(c.__webglFramebuffer,e,s,36064+r,3553),B(s,f)&&F(3553,s,e.width,e.height)}n.unbindTexture()}else{let t=3553;if(m)if(o){t=l.isDataTexture3D?32879:35866}else console.warn("THREE.DataTexture3D and THREE.DataTexture2DArray only supported with WebGL2.");n.bindTexture(t,h.__webglTexture),X(t,l,f),Q(c.__webglFramebuffer,e,l,36064,t),B(l,f)&&F(t,l,e.width,e.height,e.depth),n.unbindTexture()}e.depthBuffer&&$(e)},this.updateRenderTargetMipmap=function(t){const e=z(t)||o,r=!0===t.isWebGLMultipleRenderTargets?t.texture:[t.texture];for(let s=0,a=r.length;s<a;s++){const a=r[s];if(B(a,e)){const e=t.isWebGLCubeRenderTarget?34067:3553,r=i.get(a).__webglTexture;n.bindTexture(e,r),F(e,a,t.width,t.height),n.unbindTexture()}}},this.updateMultisampleRenderTarget=function(e){if(e.isWebGLMultisampleRenderTarget)if(o){const r=e.width,s=e.height;let a=16384;e.depthBuffer&&(a|=256),e.stencilBuffer&&(a|=1024);const o=i.get(e);n.bindFramebuffer(36008,o.__webglMultisampledFramebuffer),n.bindFramebuffer(36009,o.__webglFramebuffer),t.blitFramebuffer(0,0,r,s,0,0,r,s,a,9728),n.bindFramebuffer(36008,null),n.bindFramebuffer(36009,o.__webglMultisampledFramebuffer)}else console.warn("THREE.WebGLRenderer: WebGLMultisampleRenderTarget can only be used with WebGL2.")},this.safeSetTexture2D=function(t,e){t&&t.isWebGLRenderTarget&&(!1===et&&(console.warn("THREE.WebGLTextures.safeSetTexture2D: don't use render targets as textures. Use their .texture property instead."),et=!0),t=t.texture),V(t,e)},this.safeSetTextureCube=function(t,e){t&&t.isWebGLCubeRenderTarget&&(!1===nt&&(console.warn("THREE.WebGLTextures.safeSetTextureCube: don't use cube render targets as textures. Use their .texture property instead."),nt=!0),t=t.texture),W(t,e)}}function Ws(t,e,n){const i=n.isWebGL2;return{convert:function(t){let n;if(t===x)return 5121;if(1017===t)return 32819;if(1018===t)return 32820;if(1019===t)return 33635;if(1010===t)return 5120;if(1011===t)return 5122;if(t===_)return 5123;if(1013===t)return 5124;if(t===M)return 5125;if(t===b)return 5126;if(t===w)return i?5131:(n=e.get("OES_texture_half_float"),null!==n?n.HALF_FLOAT_OES:null);if(1021===t)return 6406;if(t===T)return 6407;if(t===E)return 6408;if(1024===t)return 6409;if(1025===t)return 6410;if(t===A)return 6402;if(t===L)return 34041;if(1028===t)return 6403;if(1029===t)return 36244;if(1030===t)return 33319;if(1031===t)return 33320;if(1032===t)return 36248;if(1033===t)return 36249;if(t===R||t===C||t===P||t===I){if(n=e.get("WEBGL_compressed_texture_s3tc"),null===n)return null;if(t===R)return n.COMPRESSED_RGB_S3TC_DXT1_EXT;if(t===C)return n.COMPRESSED_RGBA_S3TC_DXT1_EXT;if(t===P)return n.COMPRESSED_RGBA_S3TC_DXT3_EXT;if(t===I)return n.COMPRESSED_RGBA_S3TC_DXT5_EXT}if(t===D||t===N||t===z||t===B){if(n=e.get("WEBGL_compressed_texture_pvrtc"),null===n)return null;if(t===D)return n.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;if(t===N)return n.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;if(t===z)return n.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;if(t===B)return n.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG}if(36196===t)return n=e.get("WEBGL_compressed_texture_etc1"),null!==n?n.COMPRESSED_RGB_ETC1_WEBGL:null;if((t===F||t===O)&&(n=e.get("WEBGL_compressed_texture_etc"),null!==n)){if(t===F)return n.COMPRESSED_RGB8_ETC2;if(t===O)return n.COMPRESSED_RGBA8_ETC2_EAC}return 37808===t||37809===t||37810===t||37811===t||37812===t||37813===t||37814===t||37815===t||37816===t||37817===t||37818===t||37819===t||37820===t||37821===t||37840===t||37841===t||37842===t||37843===t||37844===t||37845===t||37846===t||37847===t||37848===t||37849===t||37850===t||37851===t||37852===t||37853===t?(n=e.get("WEBGL_compressed_texture_astc"),null!==n?t:null):36492===t?(n=e.get("EXT_texture_compression_bptc"),null!==n?t:null):t===S?i?34042:(n=e.get("WEBGL_depth_texture"),null!==n?n.UNSIGNED_INT_24_8_WEBGL:null):void 0}}}class js extends Kn{constructor(t=[]){super(),this.cameras=t}}js.prototype.isArrayCamera=!0;class qs extends Fe{constructor(){super(),this.type="Group"}}qs.prototype.isGroup=!0;const Xs={type:"move"};class Ys{constructor(){this._targetRay=null,this._grip=null,this._hand=null}getHandSpace(){return null===this._hand&&(this._hand=new qs,this._hand.matrixAutoUpdate=!1,this._hand.visible=!1,this._hand.joints={},this._hand.inputState={pinching:!1}),this._hand}getTargetRaySpace(){return null===this._targetRay&&(this._targetRay=new qs,this._targetRay.matrixAutoUpdate=!1,this._targetRay.visible=!1,this._targetRay.hasLinearVelocity=!1,this._targetRay.linearVelocity=new zt,this._targetRay.hasAngularVelocity=!1,this._targetRay.angularVelocity=new zt),this._targetRay}getGripSpace(){return null===this._grip&&(this._grip=new qs,this._grip.matrixAutoUpdate=!1,this._grip.visible=!1,this._grip.hasLinearVelocity=!1,this._grip.linearVelocity=new zt,this._grip.hasAngularVelocity=!1,this._grip.angularVelocity=new zt),this._grip}dispatchEvent(t){return null!==this._targetRay&&this._targetRay.dispatchEvent(t),null!==this._grip&&this._grip.dispatchEvent(t),null!==this._hand&&this._hand.dispatchEvent(t),this}disconnect(t){return this.dispatchEvent({type:"disconnected",data:t}),null!==this._targetRay&&(this._targetRay.visible=!1),null!==this._grip&&(this._grip.visible=!1),null!==this._hand&&(this._hand.visible=!1),this}update(t,e,n){let i=null,r=null,s=null;const a=this._targetRay,o=this._grip,l=this._hand;if(t&&"visible-blurred"!==e.session.visibilityState)if(null!==a&&(i=e.getPose(t.targetRaySpace,n),null!==i&&(a.matrix.fromArray(i.transform.matrix),a.matrix.decompose(a.position,a.rotation,a.scale),i.linearVelocity?(a.hasLinearVelocity=!0,a.linearVelocity.copy(i.linearVelocity)):a.hasLinearVelocity=!1,i.angularVelocity?(a.hasAngularVelocity=!0,a.angularVelocity.copy(i.angularVelocity)):a.hasAngularVelocity=!1,this.dispatchEvent(Xs))),l&&t.hand){s=!0;for(const i of t.hand.values()){const t=e.getJointPose(i,n);if(void 0===l.joints[i.jointName]){const t=new qs;t.matrixAutoUpdate=!1,t.visible=!1,l.joints[i.jointName]=t,l.add(t)}const r=l.joints[i.jointName];null!==t&&(r.matrix.fromArray(t.transform.matrix),r.matrix.decompose(r.position,r.rotation,r.scale),r.jointRadius=t.radius),r.visible=null!==t}const i=l.joints["index-finger-tip"],r=l.joints["thumb-tip"],a=i.position.distanceTo(r.position),o=.02,c=.005;l.inputState.pinching&&a>o+c?(l.inputState.pinching=!1,this.dispatchEvent({type:"pinchend",handedness:t.handedness,target:this})):!l.inputState.pinching&&a<=o-c&&(l.inputState.pinching=!0,this.dispatchEvent({type:"pinchstart",handedness:t.handedness,target:this}))}else null!==o&&t.gripSpace&&(r=e.getPose(t.gripSpace,n),null!==r&&(o.matrix.fromArray(r.transform.matrix),o.matrix.decompose(o.position,o.rotation,o.scale),r.linearVelocity?(o.hasLinearVelocity=!0,o.linearVelocity.copy(r.linearVelocity)):o.hasLinearVelocity=!1,r.angularVelocity?(o.hasAngularVelocity=!0,o.angularVelocity.copy(r.angularVelocity)):o.hasAngularVelocity=!1));return null!==a&&(a.visible=null!==i),null!==o&&(o.visible=null!==r),null!==l&&(l.visible=null!==s),this}}class Js extends rt{constructor(t,e){super();const n=this,i=t.state;let r=null,s=1,a=null,o="local-floor",l=null,c=null,h=null,u=null,d=null,p=!1,m=null,f=null,g=null,v=null,y=null,x=null;const _=[],M=new Map,b=new Kn;b.layers.enable(1),b.viewport=new Ct;const w=new Kn;w.layers.enable(2),w.viewport=new Ct;const S=[b,w],T=new js;T.layers.enable(1),T.layers.enable(2);let E=null,A=null;function L(t){const e=M.get(t.inputSource);e&&e.dispatchEvent({type:t.type,data:t.inputSource})}function R(){M.forEach((function(t,e){t.disconnect(e)})),M.clear(),E=null,A=null,i.bindXRFramebuffer(null),t.setRenderTarget(t.getRenderTarget()),h&&e.deleteFramebuffer(h),m&&e.deleteFramebuffer(m),f&&e.deleteRenderbuffer(f),g&&e.deleteRenderbuffer(g),h=null,m=null,f=null,g=null,d=null,u=null,c=null,r=null,z.stop(),n.isPresenting=!1,n.dispatchEvent({type:"sessionend"})}function C(t){const e=r.inputSources;for(let t=0;t<_.length;t++)M.set(e[t],_[t]);for(let e=0;e<t.removed.length;e++){const n=t.removed[e],i=M.get(n);i&&(i.dispatchEvent({type:"disconnected",data:n}),M.delete(n))}for(let e=0;e<t.added.length;e++){const n=t.added[e],i=M.get(n);i&&i.dispatchEvent({type:"connected",data:n})}}this.cameraAutoUpdate=!0,this.enabled=!1,this.isPresenting=!1,this.getController=function(t){let e=_[t];return void 0===e&&(e=new Ys,_[t]=e),e.getTargetRaySpace()},this.getControllerGrip=function(t){let e=_[t];return void 0===e&&(e=new Ys,_[t]=e),e.getGripSpace()},this.getHand=function(t){let e=_[t];return void 0===e&&(e=new Ys,_[t]=e),e.getHandSpace()},this.setFramebufferScaleFactor=function(t){s=t,!0===n.isPresenting&&console.warn("THREE.WebXRManager: Cannot change framebuffer scale while presenting.")},this.setReferenceSpaceType=function(t){o=t,!0===n.isPresenting&&console.warn("THREE.WebXRManager: Cannot change reference space type while presenting.")},this.getReferenceSpace=function(){return a},this.getBaseLayer=function(){return null!==u?u:d},this.getBinding=function(){return c},this.getFrame=function(){return v},this.getSession=function(){return r},this.setSession=async function(t){if(r=t,null!==r){r.addEventListener("select",L),r.addEventListener("selectstart",L),r.addEventListener("selectend",L),r.addEventListener("squeeze",L),r.addEventListener("squeezestart",L),r.addEventListener("squeezeend",L),r.addEventListener("end",R),r.addEventListener("inputsourceschange",C);const t=e.getContextAttributes();if(!0!==t.xrCompatible&&await e.makeXRCompatible(),void 0===r.renderState.layers){const n={antialias:t.antialias,alpha:t.alpha,depth:t.depth,stencil:t.stencil,framebufferScaleFactor:s};d=new XRWebGLLayer(r,e,n),r.updateRenderState({baseLayer:d})}else if(e instanceof WebGLRenderingContext){const n={antialias:!0,alpha:t.alpha,depth:t.depth,stencil:t.stencil,framebufferScaleFactor:s};d=new XRWebGLLayer(r,e,n),r.updateRenderState({layers:[d]})}else{p=t.antialias;let n=null;t.depth&&(x=256,t.stencil&&(x|=1024),y=t.stencil?33306:36096,n=t.stencil?35056:33190);const a={colorFormat:t.alpha?32856:32849,depthFormat:n,scaleFactor:s};c=new XRWebGLBinding(r,e),u=c.createProjectionLayer(a),h=e.createFramebuffer(),r.updateRenderState({layers:[u]}),p&&(m=e.createFramebuffer(),f=e.createRenderbuffer(),e.bindRenderbuffer(36161,f),e.renderbufferStorageMultisample(36161,4,32856,u.textureWidth,u.textureHeight),i.bindFramebuffer(36160,m),e.framebufferRenderbuffer(36160,36064,36161,f),e.bindRenderbuffer(36161,null),null!==n&&(g=e.createRenderbuffer(),e.bindRenderbuffer(36161,g),e.renderbufferStorageMultisample(36161,4,n,u.textureWidth,u.textureHeight),e.framebufferRenderbuffer(36160,y,36161,g),e.bindRenderbuffer(36161,null)),i.bindFramebuffer(36160,null))}a=await r.requestReferenceSpace(o),z.setContext(r),z.start(),n.isPresenting=!0,n.dispatchEvent({type:"sessionstart"})}};const P=new zt,I=new zt;function D(t,e){null===e?t.matrixWorld.copy(t.matrix):t.matrixWorld.multiplyMatrices(e.matrixWorld,t.matrix),t.matrixWorldInverse.copy(t.matrixWorld).invert()}this.updateCamera=function(t){if(null===r)return;T.near=w.near=b.near=t.near,T.far=w.far=b.far=t.far,E===T.near&&A===T.far||(r.updateRenderState({depthNear:T.near,depthFar:T.far}),E=T.near,A=T.far);const e=t.parent,n=T.cameras;D(T,e);for(let t=0;t<n.length;t++)D(n[t],e);T.matrixWorld.decompose(T.position,T.quaternion,T.scale),t.position.copy(T.position),t.quaternion.copy(T.quaternion),t.scale.copy(T.scale),t.matrix.copy(T.matrix),t.matrixWorld.copy(T.matrixWorld);const i=t.children;for(let t=0,e=i.length;t<e;t++)i[t].updateMatrixWorld(!0);2===n.length?function(t,e,n){P.setFromMatrixPosition(e.matrixWorld),I.setFromMatrixPosition(n.matrixWorld);const i=P.distanceTo(I),r=e.projectionMatrix.elements,s=n.projectionMatrix.elements,a=r[14]/(r[10]-1),o=r[14]/(r[10]+1),l=(r[9]+1)/r[5],c=(r[9]-1)/r[5],h=(r[8]-1)/r[0],u=(s[8]+1)/s[0],d=a*h,p=a*u,m=i/(-h+u),f=m*-h;e.matrixWorld.decompose(t.position,t.quaternion,t.scale),t.translateX(f),t.translateZ(m),t.matrixWorld.compose(t.position,t.quaternion,t.scale),t.matrixWorldInverse.copy(t.matrixWorld).invert();const g=a+m,v=o+m,y=d-f,x=p+(i-f),_=l*o/v*g,M=c*o/v*g;t.projectionMatrix.makePerspective(y,x,_,M,g,v)}(T,b,w):T.projectionMatrix.copy(b.projectionMatrix)},this.getCamera=function(){return T},this.getFoveation=function(){return null!==u?u.fixedFoveation:null!==d?d.fixedFoveation:void 0},this.setFoveation=function(t){null!==u&&(u.fixedFoveation=t),null!==d&&void 0!==d.fixedFoveation&&(d.fixedFoveation=t)};let N=null;const z=new hi;z.setAnimationLoop((function(t,n){if(l=n.getViewerPose(a),v=n,null!==l){const t=l.views;null!==d&&i.bindXRFramebuffer(d.framebuffer);let n=!1;t.length!==T.cameras.length&&(T.cameras.length=0,n=!0);for(let r=0;r<t.length;r++){const s=t[r];let a=null;if(null!==d)a=d.getViewport(s);else{const t=c.getViewSubImage(u,s);i.bindXRFramebuffer(h),void 0!==t.depthStencilTexture&&e.framebufferTexture2D(36160,y,3553,t.depthStencilTexture,0),e.framebufferTexture2D(36160,36064,3553,t.colorTexture,0),a=t.viewport}const o=S[r];o.matrix.fromArray(s.transform.matrix),o.projectionMatrix.fromArray(s.projectionMatrix),o.viewport.set(a.x,a.y,a.width,a.height),0===r&&T.matrix.copy(o.matrix),!0===n&&T.cameras.push(o)}p&&(i.bindXRFramebuffer(m),null!==x&&e.clear(x))}const s=r.inputSources;for(let t=0;t<_.length;t++){const e=_[t],i=s[t];e.update(i,n,a)}if(N&&N(t,n),p){const t=u.textureWidth,n=u.textureHeight;i.bindFramebuffer(36008,m),i.bindFramebuffer(36009,h),e.invalidateFramebuffer(36008,[y]),e.invalidateFramebuffer(36009,[y]),e.blitFramebuffer(0,0,t,n,0,0,t,n,16384,9728),e.invalidateFramebuffer(36008,[36064]),i.bindFramebuffer(36008,null),i.bindFramebuffer(36009,null),i.bindFramebuffer(36160,m)}v=null})),this.setAnimationLoop=function(t){N=t},this.dispose=function(){}}}function Zs(t){function e(e,n){e.opacity.value=n.opacity,n.color&&e.diffuse.value.copy(n.color),n.emissive&&e.emissive.value.copy(n.emissive).multiplyScalar(n.emissiveIntensity),n.map&&(e.map.value=n.map),n.alphaMap&&(e.alphaMap.value=n.alphaMap),n.specularMap&&(e.specularMap.value=n.specularMap),n.alphaTest>0&&(e.alphaTest.value=n.alphaTest);const i=t.get(n).envMap;if(i){e.envMap.value=i,e.flipEnvMap.value=i.isCubeTexture&&!1===i.isRenderTargetTexture?-1:1,e.reflectivity.value=n.reflectivity,e.ior.value=n.ior,e.refractionRatio.value=n.refractionRatio;const r=t.get(i).__maxMipLevel;void 0!==r&&(e.maxMipLevel.value=r)}let r,s;n.lightMap&&(e.lightMap.value=n.lightMap,e.lightMapIntensity.value=n.lightMapIntensity),n.aoMap&&(e.aoMap.value=n.aoMap,e.aoMapIntensity.value=n.aoMapIntensity),n.map?r=n.map:n.specularMap?r=n.specularMap:n.displacementMap?r=n.displacementMap:n.normalMap?r=n.normalMap:n.bumpMap?r=n.bumpMap:n.roughnessMap?r=n.roughnessMap:n.metalnessMap?r=n.metalnessMap:n.alphaMap?r=n.alphaMap:n.emissiveMap?r=n.emissiveMap:n.clearcoatMap?r=n.clearcoatMap:n.clearcoatNormalMap?r=n.clearcoatNormalMap:n.clearcoatRoughnessMap?r=n.clearcoatRoughnessMap:n.specularIntensityMap?r=n.specularIntensityMap:n.specularColorMap?r=n.specularColorMap:n.transmissionMap?r=n.transmissionMap:n.thicknessMap?r=n.thicknessMap:n.sheenColorMap?r=n.sheenColorMap:n.sheenRoughnessMap&&(r=n.sheenRoughnessMap),void 0!==r&&(r.isWebGLRenderTarget&&(r=r.texture),!0===r.matrixAutoUpdate&&r.updateMatrix(),e.uvTransform.value.copy(r.matrix)),n.aoMap?s=n.aoMap:n.lightMap&&(s=n.lightMap),void 0!==s&&(s.isWebGLRenderTarget&&(s=s.texture),!0===s.matrixAutoUpdate&&s.updateMatrix(),e.uv2Transform.value.copy(s.matrix))}function n(e,n){e.roughness.value=n.roughness,e.metalness.value=n.metalness,n.roughnessMap&&(e.roughnessMap.value=n.roughnessMap),n.metalnessMap&&(e.metalnessMap.value=n.metalnessMap),n.emissiveMap&&(e.emissiveMap.value=n.emissiveMap),n.bumpMap&&(e.bumpMap.value=n.bumpMap,e.bumpScale.value=n.bumpScale,1===n.side&&(e.bumpScale.value*=-1)),n.normalMap&&(e.normalMap.value=n.normalMap,e.normalScale.value.copy(n.normalScale),1===n.side&&e.normalScale.value.negate()),n.displacementMap&&(e.displacementMap.value=n.displacementMap,e.displacementScale.value=n.displacementScale,e.displacementBias.value=n.displacementBias);t.get(n).envMap&&(e.envMapIntensity.value=n.envMapIntensity)}return{refreshFogUniforms:function(t,e){t.fogColor.value.copy(e.color),e.isFog?(t.fogNear.value=e.near,t.fogFar.value=e.far):e.isFogExp2&&(t.fogDensity.value=e.density)},refreshMaterialUniforms:function(t,i,r,s,a){i.isMeshBasicMaterial?e(t,i):i.isMeshLambertMaterial?(e(t,i),function(t,e){e.emissiveMap&&(t.emissiveMap.value=e.emissiveMap)}(t,i)):i.isMeshToonMaterial?(e(t,i),function(t,e){e.gradientMap&&(t.gradientMap.value=e.gradientMap);e.emissiveMap&&(t.emissiveMap.value=e.emissiveMap);e.bumpMap&&(t.bumpMap.value=e.bumpMap,t.bumpScale.value=e.bumpScale,1===e.side&&(t.bumpScale.value*=-1));e.normalMap&&(t.normalMap.value=e.normalMap,t.normalScale.value.copy(e.normalScale),1===e.side&&t.normalScale.value.negate());e.displacementMap&&(t.displacementMap.value=e.displacementMap,t.displacementScale.value=e.displacementScale,t.displacementBias.value=e.displacementBias)}(t,i)):i.isMeshPhongMaterial?(e(t,i),function(t,e){t.specular.value.copy(e.specular),t.shininess.value=Math.max(e.shininess,1e-4),e.emissiveMap&&(t.emissiveMap.value=e.emissiveMap);e.bumpMap&&(t.bumpMap.value=e.bumpMap,t.bumpScale.value=e.bumpScale,1===e.side&&(t.bumpScale.value*=-1));e.normalMap&&(t.normalMap.value=e.normalMap,t.normalScale.value.copy(e.normalScale),1===e.side&&t.normalScale.value.negate());e.displacementMap&&(t.displacementMap.value=e.displacementMap,t.displacementScale.value=e.displacementScale,t.displacementBias.value=e.displacementBias)}(t,i)):i.isMeshStandardMaterial?(e(t,i),i.isMeshPhysicalMaterial?function(t,e,i){n(t,e),t.ior.value=e.ior,e.sheen>0&&(t.sheenColor.value.copy(e.sheenColor).multiplyScalar(e.sheen),t.sheenRoughness.value=e.sheenRoughness,e.sheenColorMap&&(t.sheenColorMap.value=e.sheenColorMap),e.sheenRoughnessMap&&(t.sheenRoughnessMap.value=e.sheenRoughnessMap));e.clearcoat>0&&(t.clearcoat.value=e.clearcoat,t.clearcoatRoughness.value=e.clearcoatRoughness,e.clearcoatMap&&(t.clearcoatMap.value=e.clearcoatMap),e.clearcoatRoughnessMap&&(t.clearcoatRoughnessMap.value=e.clearcoatRoughnessMap),e.clearcoatNormalMap&&(t.clearcoatNormalScale.value.copy(e.clearcoatNormalScale),t.clearcoatNormalMap.value=e.clearcoatNormalMap,1===e.side&&t.clearcoatNormalScale.value.negate()));e.transmission>0&&(t.transmission.value=e.transmission,t.transmissionSamplerMap.value=i.texture,t.transmissionSamplerSize.value.set(i.width,i.height),e.transmissionMap&&(t.transmissionMap.value=e.transmissionMap),t.thickness.value=e.thickness,e.thicknessMap&&(t.thicknessMap.value=e.thicknessMap),t.attenuationDistance.value=e.attenuationDistance,t.attenuationColor.value.copy(e.attenuationColor));t.specularIntensity.value=e.specularIntensity,t.specularColor.value.copy(e.specularColor),e.specularIntensityMap&&(t.specularIntensityMap.value=e.specularIntensityMap);e.specularColorMap&&(t.specularColorMap.value=e.specularColorMap)}(t,i,a):n(t,i)):i.isMeshMatcapMaterial?(e(t,i),function(t,e){e.matcap&&(t.matcap.value=e.matcap);e.bumpMap&&(t.bumpMap.value=e.bumpMap,t.bumpScale.value=e.bumpScale,1===e.side&&(t.bumpScale.value*=-1));e.normalMap&&(t.normalMap.value=e.normalMap,t.normalScale.value.copy(e.normalScale),1===e.side&&t.normalScale.value.negate());e.displacementMap&&(t.displacementMap.value=e.displacementMap,t.displacementScale.value=e.displacementScale,t.displacementBias.value=e.displacementBias)}(t,i)):i.isMeshDepthMaterial?(e(t,i),function(t,e){e.displacementMap&&(t.displacementMap.value=e.displacementMap,t.displacementScale.value=e.displacementScale,t.displacementBias.value=e.displacementBias)}(t,i)):i.isMeshDistanceMaterial?(e(t,i),function(t,e){e.displacementMap&&(t.displacementMap.value=e.displacementMap,t.displacementScale.value=e.displacementScale,t.displacementBias.value=e.displacementBias);t.referencePosition.value.copy(e.referencePosition),t.nearDistance.value=e.nearDistance,t.farDistance.value=e.farDistance}(t,i)):i.isMeshNormalMaterial?(e(t,i),function(t,e){e.bumpMap&&(t.bumpMap.value=e.bumpMap,t.bumpScale.value=e.bumpScale,1===e.side&&(t.bumpScale.value*=-1));e.normalMap&&(t.normalMap.value=e.normalMap,t.normalScale.value.copy(e.normalScale),1===e.side&&t.normalScale.value.negate());e.displacementMap&&(t.displacementMap.value=e.displacementMap,t.displacementScale.value=e.displacementScale,t.displacementBias.value=e.displacementBias)}(t,i)):i.isLineBasicMaterial?(function(t,e){t.diffuse.value.copy(e.color),t.opacity.value=e.opacity}(t,i),i.isLineDashedMaterial&&function(t,e){t.dashSize.value=e.dashSize,t.totalSize.value=e.dashSize+e.gapSize,t.scale.value=e.scale}(t,i)):i.isPointsMaterial?function(t,e,n,i){t.diffuse.value.copy(e.color),t.opacity.value=e.opacity,t.size.value=e.size*n,t.scale.value=.5*i,e.map&&(t.map.value=e.map);e.alphaMap&&(t.alphaMap.value=e.alphaMap);e.alphaTest>0&&(t.alphaTest.value=e.alphaTest);let r;e.map?r=e.map:e.alphaMap&&(r=e.alphaMap);void 0!==r&&(!0===r.matrixAutoUpdate&&r.updateMatrix(),t.uvTransform.value.copy(r.matrix))}(t,i,r,s):i.isSpriteMaterial?function(t,e){t.diffuse.value.copy(e.color),t.opacity.value=e.opacity,t.rotation.value=e.rotation,e.map&&(t.map.value=e.map);e.alphaMap&&(t.alphaMap.value=e.alphaMap);e.alphaTest>0&&(t.alphaTest.value=e.alphaTest);let n;e.map?n=e.map:e.alphaMap&&(n=e.alphaMap);void 0!==n&&(!0===n.matrixAutoUpdate&&n.updateMatrix(),t.uvTransform.value.copy(n.matrix))}(t,i):i.isShadowMaterial?(t.color.value.copy(i.color),t.opacity.value=i.opacity):i.isShaderMaterial&&(i.uniformsNeedUpdate=!1)}}}function Qs(t={}){const e=void 0!==t.canvas?t.canvas:function(){const t=wt("canvas");return t.style.display="block",t}(),n=void 0!==t.context?t.context:null,i=void 0!==t.alpha&&t.alpha,r=void 0===t.depth||t.depth,s=void 0===t.stencil||t.stencil,a=void 0!==t.antialias&&t.antialias,o=void 0===t.premultipliedAlpha||t.premultipliedAlpha,l=void 0!==t.preserveDrawingBuffer&&t.preserveDrawingBuffer,c=void 0!==t.powerPreference?t.powerPreference:"default",h=void 0!==t.failIfMajorPerformanceCaveat&&t.failIfMajorPerformanceCaveat;let d=null,m=null;const f=[],g=[];this.domElement=e,this.debug={checkShaderErrors:!0},this.autoClear=!0,this.autoClearColor=!0,this.autoClearDepth=!0,this.autoClearStencil=!0,this.sortObjects=!0,this.clippingPlanes=[],this.localClippingEnabled=!1,this.gammaFactor=2,this.outputEncoding=X,this.physicallyCorrectLights=!1,this.toneMapping=0,this.toneMappingExposure=1;const v=this;let _=!1,M=0,S=0,T=null,A=-1,L=null;const R=new Ct,C=new Ct;let P=null,I=e.width,D=e.height,N=1,z=null,B=null;const F=new Ct(0,0,I,D),O=new Ct(0,0,I,D);let U=!1;const H=[],G=new ci;let k=!1,V=!1,W=null;const j=new de,q=new zt,Y={background:null,fog:null,environment:null,overrideMaterial:null,isScene:!0};function J(){return null===T?N:1}let Z,Q,K,$,tt,et,nt,it,rt,st,at,ot,lt,ct,ht,ut,dt,pt,mt,ft,gt,vt,yt,xt=n;function _t(t,n){for(let i=0;i<t.length;i++){const r=t[i],s=e.getContext(r,n);if(null!==s)return s}return null}try{const t={alpha:i,depth:r,stencil:s,antialias:a,premultipliedAlpha:o,preserveDrawingBuffer:l,powerPreference:c,failIfMajorPerformanceCaveat:h};if(e.addEventListener("webglcontextlost",St,!1),e.addEventListener("webglcontextrestored",Tt,!1),null===xt){const e=["webgl2","webgl","experimental-webgl"];if(!0===v.isWebGL1Renderer&&e.shift(),xt=_t(e,t),null===xt)throw _t(e)?new Error("Error creating WebGL context with your selected attributes."):new Error("Error creating WebGL context.")}void 0===xt.getShaderPrecisionFormat&&(xt.getShaderPrecisionFormat=function(){return{rangeMin:1,rangeMax:1,precision:1}})}catch(t){throw console.error("THREE.WebGLRenderer: "+t.message),t}function Mt(){Z=new Yi(xt),Q=new xi(xt,Z,t),Z.init(Q),vt=new Ws(xt,Z,Q),K=new ks(xt,Z,Q),H[0]=1029,$=new Qi(xt),tt=new Ls,et=new Vs(xt,Z,K,tt,Q,vt,$),nt=new Mi(v),it=new Xi(v),rt=new ui(xt,Q),yt=new vi(xt,Z,rt,Q),st=new Ji(xt,rt,$,yt),at=new ir(xt,st,rt,$),mt=new nr(xt,Q,et),ut=new _i(tt),ot=new As(v,nt,it,Z,Q,yt,ut),lt=new Zs(tt),ct=new Is(tt),ht=new Os(Z,Q),pt=new gi(v,nt,K,at,o),dt=new Gs(v,at,Q),ft=new yi(xt,Z,$,Q),gt=new Zi(xt,Z,$,Q),$.programs=ot.programs,v.capabilities=Q,v.extensions=Z,v.properties=tt,v.renderLists=ct,v.shadowMap=dt,v.state=K,v.info=$}Mt();const bt=new Js(v,xt);function St(t){t.preventDefault(),console.log("THREE.WebGLRenderer: Context Lost."),_=!0}function Tt(){console.log("THREE.WebGLRenderer: Context Restored."),_=!1;const t=$.autoReset,e=dt.enabled,n=dt.autoUpdate,i=dt.needsUpdate,r=dt.type;Mt(),$.autoReset=t,dt.enabled=e,dt.autoUpdate=n,dt.needsUpdate=i,dt.type=r}function Et(t){const e=t.target;e.removeEventListener("dispose",Et),function(t){(function(t){const e=tt.get(t).programs;void 0!==e&&e.forEach((function(t){ot.releaseProgram(t)}))})(t),tt.remove(t)}(e)}this.xr=bt,this.getContext=function(){return xt},this.getContextAttributes=function(){return xt.getContextAttributes()},this.forceContextLoss=function(){const t=Z.get("WEBGL_lose_context");t&&t.loseContext()},this.forceContextRestore=function(){const t=Z.get("WEBGL_lose_context");t&&t.restoreContext()},this.getPixelRatio=function(){return N},this.setPixelRatio=function(t){void 0!==t&&(N=t,this.setSize(I,D,!1))},this.getSize=function(t){return t.set(I,D)},this.setSize=function(t,n,i){bt.isPresenting?console.warn("THREE.WebGLRenderer: Can't change size while VR device is presenting."):(I=t,D=n,e.width=Math.floor(t*N),e.height=Math.floor(n*N),!1!==i&&(e.style.width=t+"px",e.style.height=n+"px"),this.setViewport(0,0,t,n))},this.getDrawingBufferSize=function(t){return t.set(I*N,D*N).floor()},this.setDrawingBufferSize=function(t,n,i){I=t,D=n,N=i,e.width=Math.floor(t*i),e.height=Math.floor(n*i),this.setViewport(0,0,t,n)},this.getCurrentViewport=function(t){return t.copy(R)},this.getViewport=function(t){return t.copy(F)},this.setViewport=function(t,e,n,i){t.isVector4?F.set(t.x,t.y,t.z,t.w):F.set(t,e,n,i),K.viewport(R.copy(F).multiplyScalar(N).floor())},this.getScissor=function(t){return t.copy(O)},this.setScissor=function(t,e,n,i){t.isVector4?O.set(t.x,t.y,t.z,t.w):O.set(t,e,n,i),K.scissor(C.copy(O).multiplyScalar(N).floor())},this.getScissorTest=function(){return U},this.setScissorTest=function(t){K.setScissorTest(U=t)},this.setOpaqueSort=function(t){z=t},this.setTransparentSort=function(t){B=t},this.getClearColor=function(t){return t.copy(pt.getClearColor())},this.setClearColor=function(){pt.setClearColor.apply(pt,arguments)},this.getClearAlpha=function(){return pt.getClearAlpha()},this.setClearAlpha=function(){pt.setClearAlpha.apply(pt,arguments)},this.clear=function(t,e,n){let i=0;(void 0===t||t)&&(i|=16384),(void 0===e||e)&&(i|=256),(void 0===n||n)&&(i|=1024),xt.clear(i)},this.clearColor=function(){this.clear(!0,!1,!1)},this.clearDepth=function(){this.clear(!1,!0,!1)},this.clearStencil=function(){this.clear(!1,!1,!0)},this.dispose=function(){e.removeEventListener("webglcontextlost",St,!1),e.removeEventListener("webglcontextrestored",Tt,!1),ct.dispose(),ht.dispose(),tt.dispose(),nt.dispose(),it.dispose(),at.dispose(),yt.dispose(),bt.dispose(),bt.removeEventListener("sessionstart",Lt),bt.removeEventListener("sessionend",Rt),W&&(W.dispose(),W=null),It.stop()},this.renderBufferDirect=function(t,e,n,i,r,s){null===e&&(e=Y);const a=r.isMesh&&r.matrixWorld.determinant()<0,o=function(t,e,n,i,r){!0!==e.isScene&&(e=Y);et.resetTextureUnits();const s=e.fog,a=i.isMeshStandardMaterial?e.environment:null,o=null===T?v.outputEncoding:T.texture.encoding,l=(i.isMeshStandardMaterial?it:nt).get(i.envMap||a),c=!0===i.vertexColors&&!!n.attributes.color&&4===n.attributes.color.itemSize,h=!!i.normalMap&&!!n.attributes.tangent,u=!!n.morphAttributes.position,d=!!n.morphAttributes.normal,p=n.morphAttributes.position?n.morphAttributes.position.length:0,f=tt.get(i),g=m.state.lights;if(!0===k&&(!0===V||t!==L)){const e=t===L&&i.id===A;ut.setState(i,t,e)}let y=!1;i.version===f.__version?f.needsLights&&f.lightsStateVersion!==g.state.version||f.outputEncoding!==o||r.isInstancedMesh&&!1===f.instancing?y=!0:r.isInstancedMesh||!0!==f.instancing?r.isSkinnedMesh&&!1===f.skinning?y=!0:r.isSkinnedMesh||!0!==f.skinning?f.envMap!==l||i.fog&&f.fog!==s?y=!0:void 0===f.numClippingPlanes||f.numClippingPlanes===ut.numPlanes&&f.numIntersection===ut.numIntersection?(f.vertexAlphas!==c||f.vertexTangents!==h||f.morphTargets!==u||f.morphNormals!==d||!0===Q.isWebGL2&&f.morphTargetsCount!==p)&&(y=!0):y=!0:y=!0:y=!0:(y=!0,f.__version=i.version);let x=f.currentProgram;!0===y&&(x=Ut(i,e,r));let _=!1,M=!1,b=!1;const w=x.getUniforms(),S=f.uniforms;K.useProgram(x.program)&&(_=!0,M=!0,b=!0);i.id!==A&&(A=i.id,M=!0);if(_||L!==t){if(w.setValue(xt,"projectionMatrix",t.projectionMatrix),Q.logarithmicDepthBuffer&&w.setValue(xt,"logDepthBufFC",2/(Math.log(t.far+1)/Math.LN2)),L!==t&&(L=t,M=!0,b=!0),i.isShaderMaterial||i.isMeshPhongMaterial||i.isMeshToonMaterial||i.isMeshStandardMaterial||i.envMap){const e=w.map.cameraPosition;void 0!==e&&e.setValue(xt,q.setFromMatrixPosition(t.matrixWorld))}(i.isMeshPhongMaterial||i.isMeshToonMaterial||i.isMeshLambertMaterial||i.isMeshBasicMaterial||i.isMeshStandardMaterial||i.isShaderMaterial)&&w.setValue(xt,"isOrthographic",!0===t.isOrthographicCamera),(i.isMeshPhongMaterial||i.isMeshToonMaterial||i.isMeshLambertMaterial||i.isMeshBasicMaterial||i.isMeshStandardMaterial||i.isShaderMaterial||i.isShadowMaterial||r.isSkinnedMesh)&&w.setValue(xt,"viewMatrix",t.matrixWorldInverse)}if(r.isSkinnedMesh){w.setOptional(xt,r,"bindMatrix"),w.setOptional(xt,r,"bindMatrixInverse");const t=r.skeleton;t&&(Q.floatVertexTextures?(null===t.boneTexture&&t.computeBoneTexture(),w.setValue(xt,"boneTexture",t.boneTexture,et),w.setValue(xt,"boneTextureSize",t.boneTextureSize)):w.setOptional(xt,t,"boneMatrices"))}!n||void 0===n.morphAttributes.position&&void 0===n.morphAttributes.normal||mt.update(r,n,i,x);(M||f.receiveShadow!==r.receiveShadow)&&(f.receiveShadow=r.receiveShadow,w.setValue(xt,"receiveShadow",r.receiveShadow));M&&(w.setValue(xt,"toneMappingExposure",v.toneMappingExposure),f.needsLights&&(R=b,(E=S).ambientLightColor.needsUpdate=R,E.lightProbe.needsUpdate=R,E.directionalLights.needsUpdate=R,E.directionalLightShadows.needsUpdate=R,E.pointLights.needsUpdate=R,E.pointLightShadows.needsUpdate=R,E.spotLights.needsUpdate=R,E.spotLightShadows.needsUpdate=R,E.rectAreaLights.needsUpdate=R,E.hemisphereLights.needsUpdate=R),s&&i.fog&&lt.refreshFogUniforms(S,s),lt.refreshMaterialUniforms(S,i,N,D,W),as.upload(xt,f.uniformsList,S,et));var E,R;i.isShaderMaterial&&!0===i.uniformsNeedUpdate&&(as.upload(xt,f.uniformsList,S,et),i.uniformsNeedUpdate=!1);i.isSpriteMaterial&&w.setValue(xt,"center",r.center);return w.setValue(xt,"modelViewMatrix",r.modelViewMatrix),w.setValue(xt,"normalMatrix",r.normalMatrix),w.setValue(xt,"modelMatrix",r.matrixWorld),x}(t,e,n,i,r);K.setMaterial(i,a);let l=n.index;const c=n.attributes.position;if(null===l){if(void 0===c||0===c.count)return}else if(0===l.count)return;let h,u=1;!0===i.wireframe&&(l=st.getWireframeAttribute(n),u=2),yt.setup(r,i,o,n,l);let d=ft;null!==l&&(h=rt.get(l),d=gt,d.setIndex(h));const p=null!==l?l.count:c.count,f=n.drawRange.start*u,g=n.drawRange.count*u,y=null!==s?s.start*u:0,x=null!==s?s.count*u:1/0,_=Math.max(f,y),M=Math.min(p,f+g,y+x)-1,b=Math.max(0,M-_+1);if(0!==b){if(r.isMesh)!0===i.wireframe?(K.setLineWidth(i.wireframeLinewidth*J()),d.setMode(1)):d.setMode(4);else if(r.isLine){let t=i.linewidth;void 0===t&&(t=1),K.setLineWidth(t*J()),r.isLineSegments?d.setMode(1):r.isLineLoop?d.setMode(2):d.setMode(3)}else r.isPoints?d.setMode(0):r.isSprite&&d.setMode(4);if(r.isInstancedMesh)d.renderInstances(_,b,r.count);else if(n.isInstancedBufferGeometry){const t=Math.min(n.instanceCount,n._maxInstanceCount);d.renderInstances(_,b,t)}else d.render(_,b)}},this.compile=function(t,e){m=ht.get(t),m.init(),g.push(m),t.traverseVisible((function(t){t.isLight&&t.layers.test(e.layers)&&(m.pushLight(t),t.castShadow&&m.pushShadow(t))})),m.setupLights(v.physicallyCorrectLights),t.traverse((function(e){const n=e.material;if(n)if(Array.isArray(n))for(let i=0;i<n.length;i++){Ut(n[i],t,e)}else Ut(n,t,e)})),g.pop(),m=null};let At=null;function Lt(){It.stop()}function Rt(){It.start()}const It=new hi;function Nt(t,e,n,i){if(!1===t.visible)return;if(t.layers.test(e.layers))if(t.isGroup)n=t.renderOrder;else if(t.isLOD)!0===t.autoUpdate&&t.update(e);else if(t.isLight)m.pushLight(t),t.castShadow&&m.pushShadow(t);else if(t.isSprite){if(!t.frustumCulled||G.intersectsSprite(t)){i&&q.setFromMatrixPosition(t.matrixWorld).applyMatrix4(j);const e=at.update(t),r=t.material;r.visible&&d.push(t,e,r,n,q.z,null)}}else if((t.isMesh||t.isLine||t.isPoints)&&(t.isSkinnedMesh&&t.skeleton.frame!==$.render.frame&&(t.skeleton.update(),t.skeleton.frame=$.render.frame),!t.frustumCulled||G.intersectsObject(t))){i&&q.setFromMatrixPosition(t.matrixWorld).applyMatrix4(j);const e=at.update(t),r=t.material;if(Array.isArray(r)){const i=e.groups;for(let s=0,a=i.length;s<a;s++){const a=i[s],o=r[a.materialIndex];o&&o.visible&&d.push(t,e,o,n,q.z,a)}}else r.visible&&d.push(t,e,r,n,q.z,null)}const r=t.children;for(let t=0,s=r.length;t<s;t++)Nt(r[t],e,n,i)}function Bt(t,e,n,i){const r=t.opaque,s=t.transmissive,o=t.transparent;m.setupLightsView(n),s.length>0&&function(t,e,n){if(null===W){const t=!0===a&&!0===Q.isWebGL2;W=new(t?Dt:Pt)(1024,1024,{generateMipmaps:!0,type:null!==vt.convert(w)?w:x,minFilter:y,magFilter:p,wrapS:u,wrapT:u})}const i=v.getRenderTarget();v.setRenderTarget(W),v.clear();const r=v.toneMapping;v.toneMapping=0,Ft(t,e,n),v.toneMapping=r,et.updateMultisampleRenderTarget(W),et.updateRenderTargetMipmap(W),v.setRenderTarget(i)}(r,e,n),i&&K.viewport(R.copy(i)),r.length>0&&Ft(r,e,n),s.length>0&&Ft(s,e,n),o.length>0&&Ft(o,e,n)}function Ft(t,e,n){const i=!0===e.isScene?e.overrideMaterial:null;for(let r=0,s=t.length;r<s;r++){const s=t[r],a=s.object,o=s.geometry,l=null===i?s.material:i,c=s.group;a.layers.test(n.layers)&&Ot(a,e,n,o,l,c)}}function Ot(t,e,n,i,r,s){t.onBeforeRender(v,e,n,i,r,s),t.modelViewMatrix.multiplyMatrices(n.matrixWorldInverse,t.matrixWorld),t.normalMatrix.getNormalMatrix(t.modelViewMatrix),r.onBeforeRender(v,e,n,i,t,s),!0===r.transparent&&2===r.side?(r.side=1,r.needsUpdate=!0,v.renderBufferDirect(n,e,i,r,t,s),r.side=0,r.needsUpdate=!0,v.renderBufferDirect(n,e,i,r,t,s),r.side=2):v.renderBufferDirect(n,e,i,r,t,s),t.onAfterRender(v,e,n,i,r,s)}function Ut(t,e,n){!0!==e.isScene&&(e=Y);const i=tt.get(t),r=m.state.lights,s=m.state.shadowsArray,a=r.state.version,o=ot.getParameters(t,r.state,s,e,n),l=ot.getProgramCacheKey(o);let c=i.programs;i.environment=t.isMeshStandardMaterial?e.environment:null,i.fog=e.fog,i.envMap=(t.isMeshStandardMaterial?it:nt).get(t.envMap||i.environment),void 0===c&&(t.addEventListener("dispose",Et),c=new Map,i.programs=c);let h=c.get(l);if(void 0!==h){if(i.currentProgram===h&&i.lightsStateVersion===a)return Ht(t,o),h}else o.uniforms=ot.getUniforms(t),t.onBuild(n,o,v),t.onBeforeCompile(o,v),h=ot.acquireProgram(o,l),c.set(l,h),i.uniforms=o.uniforms;const u=i.uniforms;(t.isShaderMaterial||t.isRawShaderMaterial)&&!0!==t.clipping||(u.clippingPlanes=ut.uniform),Ht(t,o),i.needsLights=function(t){return t.isMeshLambertMaterial||t.isMeshToonMaterial||t.isMeshPhongMaterial||t.isMeshStandardMaterial||t.isShadowMaterial||t.isShaderMaterial&&!0===t.lights}(t),i.lightsStateVersion=a,i.needsLights&&(u.ambientLightColor.value=r.state.ambient,u.lightProbe.value=r.state.probe,u.directionalLights.value=r.state.directional,u.directionalLightShadows.value=r.state.directionalShadow,u.spotLights.value=r.state.spot,u.spotLightShadows.value=r.state.spotShadow,u.rectAreaLights.value=r.state.rectArea,u.ltc_1.value=r.state.rectAreaLTC1,u.ltc_2.value=r.state.rectAreaLTC2,u.pointLights.value=r.state.point,u.pointLightShadows.value=r.state.pointShadow,u.hemisphereLights.value=r.state.hemi,u.directionalShadowMap.value=r.state.directionalShadowMap,u.directionalShadowMatrix.value=r.state.directionalShadowMatrix,u.spotShadowMap.value=r.state.spotShadowMap,u.spotShadowMatrix.value=r.state.spotShadowMatrix,u.pointShadowMap.value=r.state.pointShadowMap,u.pointShadowMatrix.value=r.state.pointShadowMatrix);const d=h.getUniforms(),p=as.seqWithValue(d.seq,u);return i.currentProgram=h,i.uniformsList=p,h}function Ht(t,e){const n=tt.get(t);n.outputEncoding=e.outputEncoding,n.instancing=e.instancing,n.skinning=e.skinning,n.morphTargets=e.morphTargets,n.morphNormals=e.morphNormals,n.morphTargetsCount=e.morphTargetsCount,n.numClippingPlanes=e.numClippingPlanes,n.numIntersection=e.numClipIntersection,n.vertexAlphas=e.vertexAlphas,n.vertexTangents=e.vertexTangents}It.setAnimationLoop((function(t){At&&At(t)})),"undefined"!=typeof window&&It.setContext(window),this.setAnimationLoop=function(t){At=t,bt.setAnimationLoop(t),null===t?It.stop():It.start()},bt.addEventListener("sessionstart",Lt),bt.addEventListener("sessionend",Rt),this.render=function(t,e){if(void 0!==e&&!0!==e.isCamera)return void console.error("THREE.WebGLRenderer.render: camera is not an instance of THREE.Camera.");if(!0===_)return;!0===t.autoUpdate&&t.updateMatrixWorld(),null===e.parent&&e.updateMatrixWorld(),!0===bt.enabled&&!0===bt.isPresenting&&(!0===bt.cameraAutoUpdate&&bt.updateCamera(e),e=bt.getCamera()),!0===t.isScene&&t.onBeforeRender(v,t,e,T),m=ht.get(t,g.length),m.init(),g.push(m),j.multiplyMatrices(e.projectionMatrix,e.matrixWorldInverse),G.setFromProjectionMatrix(j),V=this.localClippingEnabled,k=ut.init(this.clippingPlanes,V,e),d=ct.get(t,f.length),d.init(),f.push(d),Nt(t,e,0,v.sortObjects),d.finish(),!0===v.sortObjects&&d.sort(z,B),!0===k&&ut.beginShadows();const n=m.state.shadowsArray;if(dt.render(n,t,e),!0===k&&ut.endShadows(),!0===this.info.autoReset&&this.info.reset(),pt.render(d,t),m.setupLights(v.physicallyCorrectLights),e.isArrayCamera){const n=e.cameras;for(let e=0,i=n.length;e<i;e++){const i=n[e];Bt(d,t,i,i.viewport)}}else Bt(d,t,e);null!==T&&(et.updateMultisampleRenderTarget(T),et.updateRenderTargetMipmap(T)),!0===t.isScene&&t.onAfterRender(v,t,e),K.buffers.depth.setTest(!0),K.buffers.depth.setMask(!0),K.buffers.color.setMask(!0),K.setPolygonOffset(!1),yt.resetDefaultState(),A=-1,L=null,g.pop(),m=g.length>0?g[g.length-1]:null,f.pop(),d=f.length>0?f[f.length-1]:null},this.getActiveCubeFace=function(){return M},this.getActiveMipmapLevel=function(){return S},this.getRenderTarget=function(){return T},this.setRenderTarget=function(t,e=0,n=0){T=t,M=e,S=n,t&&void 0===tt.get(t).__webglFramebuffer&&et.setupRenderTarget(t);let i=null,r=!1,s=!1;if(t){const n=t.texture;(n.isDataTexture3D||n.isDataTexture2DArray)&&(s=!0);const a=tt.get(t).__webglFramebuffer;t.isWebGLCubeRenderTarget?(i=a[e],r=!0):i=t.isWebGLMultisampleRenderTarget?tt.get(t).__webglMultisampledFramebuffer:a,R.copy(t.viewport),C.copy(t.scissor),P=t.scissorTest}else R.copy(F).multiplyScalar(N).floor(),C.copy(O).multiplyScalar(N).floor(),P=U;if(K.bindFramebuffer(36160,i)&&Q.drawBuffers){let e=!1;if(t)if(t.isWebGLMultipleRenderTargets){const n=t.texture;if(H.length!==n.length||36064!==H[0]){for(let t=0,e=n.length;t<e;t++)H[t]=36064+t;H.length=n.length,e=!0}}else 1===H.length&&36064===H[0]||(H[0]=36064,H.length=1,e=!0);else 1===H.length&&1029===H[0]||(H[0]=1029,H.length=1,e=!0);e&&(Q.isWebGL2?xt.drawBuffers(H):Z.get("WEBGL_draw_buffers").drawBuffersWEBGL(H))}if(K.viewport(R),K.scissor(C),K.setScissorTest(P),r){const i=tt.get(t.texture);xt.framebufferTexture2D(36160,36064,34069+e,i.__webglTexture,n)}else if(s){const i=tt.get(t.texture),r=e||0;xt.framebufferTextureLayer(36160,36064,i.__webglTexture,n||0,r)}A=-1},this.readRenderTargetPixels=function(t,e,n,i,r,s,a){if(!t||!t.isWebGLRenderTarget)return void console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");let o=tt.get(t).__webglFramebuffer;if(t.isWebGLCubeRenderTarget&&void 0!==a&&(o=o[a]),o){K.bindFramebuffer(36160,o);try{const a=t.texture,o=a.format,l=a.type;if(o!==E&&vt.convert(o)!==xt.getParameter(35739))return void console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or implementation defined format.");const c=l===w&&(Z.has("EXT_color_buffer_half_float")||Q.isWebGL2&&Z.has("EXT_color_buffer_float"));if(!(l===x||vt.convert(l)===xt.getParameter(35738)||l===b&&(Q.isWebGL2||Z.has("OES_texture_float")||Z.has("WEBGL_color_buffer_float"))||c))return void console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not in UnsignedByteType or implementation defined type.");36053===xt.checkFramebufferStatus(36160)?e>=0&&e<=t.width-i&&n>=0&&n<=t.height-r&&xt.readPixels(e,n,i,r,vt.convert(o),vt.convert(l),s):console.error("THREE.WebGLRenderer.readRenderTargetPixels: readPixels from renderTarget failed. Framebuffer not complete.")}finally{const t=null!==T?tt.get(T).__webglFramebuffer:null;K.bindFramebuffer(36160,t)}}},this.copyFramebufferToTexture=function(t,e,n=0){const i=Math.pow(2,-n),r=Math.floor(e.image.width*i),s=Math.floor(e.image.height*i);let a=vt.convert(e.format);Q.isWebGL2&&(6407===a&&(a=32849),6408===a&&(a=32856)),et.setTexture2D(e,0),xt.copyTexImage2D(3553,n,a,t.x,t.y,r,s,0),K.unbindTexture()},this.copyTextureToTexture=function(t,e,n,i=0){const r=e.image.width,s=e.image.height,a=vt.convert(n.format),o=vt.convert(n.type);et.setTexture2D(n,0),xt.pixelStorei(37440,n.flipY),xt.pixelStorei(37441,n.premultiplyAlpha),xt.pixelStorei(3317,n.unpackAlignment),e.isDataTexture?xt.texSubImage2D(3553,i,t.x,t.y,r,s,a,o,e.image.data):e.isCompressedTexture?xt.compressedTexSubImage2D(3553,i,t.x,t.y,e.mipmaps[0].width,e.mipmaps[0].height,a,e.mipmaps[0].data):xt.texSubImage2D(3553,i,t.x,t.y,a,o,e.image),0===i&&n.generateMipmaps&&xt.generateMipmap(3553),K.unbindTexture()},this.copyTextureToTexture3D=function(t,e,n,i,r=0){if(v.isWebGL1Renderer)return void console.warn("THREE.WebGLRenderer.copyTextureToTexture3D: can only be used with WebGL2.");const s=t.max.x-t.min.x+1,a=t.max.y-t.min.y+1,o=t.max.z-t.min.z+1,l=vt.convert(i.format),c=vt.convert(i.type);let h;if(i.isDataTexture3D)et.setTexture3D(i,0),h=32879;else{if(!i.isDataTexture2DArray)return void console.warn("THREE.WebGLRenderer.copyTextureToTexture3D: only supports THREE.DataTexture3D and THREE.DataTexture2DArray.");et.setTexture2DArray(i,0),h=35866}xt.pixelStorei(37440,i.flipY),xt.pixelStorei(37441,i.premultiplyAlpha),xt.pixelStorei(3317,i.unpackAlignment);const u=xt.getParameter(3314),d=xt.getParameter(32878),p=xt.getParameter(3316),m=xt.getParameter(3315),f=xt.getParameter(32877),g=n.isCompressedTexture?n.mipmaps[0]:n.image;xt.pixelStorei(3314,g.width),xt.pixelStorei(32878,g.height),xt.pixelStorei(3316,t.min.x),xt.pixelStorei(3315,t.min.y),xt.pixelStorei(32877,t.min.z),n.isDataTexture||n.isDataTexture3D?xt.texSubImage3D(h,r,e.x,e.y,e.z,s,a,o,l,c,g.data):n.isCompressedTexture?(console.warn("THREE.WebGLRenderer.copyTextureToTexture3D: untested support for compressed srcTexture."),xt.compressedTexSubImage3D(h,r,e.x,e.y,e.z,s,a,o,l,g.data)):xt.texSubImage3D(h,r,e.x,e.y,e.z,s,a,o,l,c,g),xt.pixelStorei(3314,u),xt.pixelStorei(32878,d),xt.pixelStorei(3316,p),xt.pixelStorei(3315,m),xt.pixelStorei(32877,f),0===r&&i.generateMipmaps&&xt.generateMipmap(h),K.unbindTexture()},this.initTexture=function(t){et.setTexture2D(t,0),K.unbindTexture()},this.resetState=function(){M=0,S=0,T=null,K.reset(),yt.reset()},"undefined"!=typeof __THREE_DEVTOOLS__&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}Qs.prototype.isWebGLRenderer=!0;class Ks extends Qs{}Ks.prototype.isWebGL1Renderer=!0;class $s{constructor(t,e=25e-5){this.name="",this.color=new rn(t),this.density=e}clone(){return new $s(this.color,this.density)}toJSON(){return{type:"FogExp2",color:this.color.getHex(),density:this.density}}}$s.prototype.isFogExp2=!0;class ta{constructor(t,e=1,n=1e3){this.name="",this.color=new rn(t),this.near=e,this.far=n}clone(){return new ta(this.color,this.near,this.far)}toJSON(){return{type:"Fog",color:this.color.getHex(),near:this.near,far:this.far}}}ta.prototype.isFog=!0;class ea extends Fe{constructor(){super(),this.type="Scene",this.background=null,this.environment=null,this.fog=null,this.overrideMaterial=null,this.autoUpdate=!0,"undefined"!=typeof __THREE_DEVTOOLS__&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}copy(t,e){return super.copy(t,e),null!==t.background&&(this.background=t.background.clone()),null!==t.environment&&(this.environment=t.environment.clone()),null!==t.fog&&(this.fog=t.fog.clone()),null!==t.overrideMaterial&&(this.overrideMaterial=t.overrideMaterial.clone()),this.autoUpdate=t.autoUpdate,this.matrixAutoUpdate=t.matrixAutoUpdate,this}toJSON(t){const e=super.toJSON(t);return null!==this.fog&&(e.object.fog=this.fog.toJSON()),e}}ea.prototype.isScene=!0;class na{constructor(t,e){this.array=t,this.stride=e,this.count=void 0!==t?t.length/e:0,this.usage=et,this.updateRange={offset:0,count:-1},this.version=0,this.uuid=ht()}onUploadCallback(){}set needsUpdate(t){!0===t&&this.version++}setUsage(t){return this.usage=t,this}copy(t){return this.array=new t.array.constructor(t.array),this.count=t.count,this.stride=t.stride,this.usage=t.usage,this}copyAt(t,e,n){t*=this.stride,n*=e.stride;for(let i=0,r=this.stride;i<r;i++)this.array[t+i]=e.array[n+i];return this}set(t,e=0){return this.array.set(t,e),this}clone(t){void 0===t.arrayBuffers&&(t.arrayBuffers={}),void 0===this.array.buffer._uuid&&(this.array.buffer._uuid=ht()),void 0===t.arrayBuffers[this.array.buffer._uuid]&&(t.arrayBuffers[this.array.buffer._uuid]=this.array.slice(0).buffer);const e=new this.array.constructor(t.arrayBuffers[this.array.buffer._uuid]),n=new this.constructor(e,this.stride);return n.setUsage(this.usage),n}onUpload(t){return this.onUploadCallback=t,this}toJSON(t){return void 0===t.arrayBuffers&&(t.arrayBuffers={}),void 0===this.array.buffer._uuid&&(this.array.buffer._uuid=ht()),void 0===t.arrayBuffers[this.array.buffer._uuid]&&(t.arrayBuffers[this.array.buffer._uuid]=Array.prototype.slice.call(new Uint32Array(this.array.buffer))),{uuid:this.uuid,buffer:this.array.buffer._uuid,type:this.array.constructor.name,stride:this.stride}}}na.prototype.isInterleavedBuffer=!0;const ia=new zt;class ra{constructor(t,e,n,i=!1){this.name="",this.data=t,this.itemSize=e,this.offset=n,this.normalized=!0===i}get count(){return this.data.count}get array(){return this.data.array}set needsUpdate(t){this.data.needsUpdate=t}applyMatrix4(t){for(let e=0,n=this.data.count;e<n;e++)ia.x=this.getX(e),ia.y=this.getY(e),ia.z=this.getZ(e),ia.applyMatrix4(t),this.setXYZ(e,ia.x,ia.y,ia.z);return this}applyNormalMatrix(t){for(let e=0,n=this.count;e<n;e++)ia.x=this.getX(e),ia.y=this.getY(e),ia.z=this.getZ(e),ia.applyNormalMatrix(t),this.setXYZ(e,ia.x,ia.y,ia.z);return this}transformDirection(t){for(let e=0,n=this.count;e<n;e++)ia.x=this.getX(e),ia.y=this.getY(e),ia.z=this.getZ(e),ia.transformDirection(t),this.setXYZ(e,ia.x,ia.y,ia.z);return this}setX(t,e){return this.data.array[t*this.data.stride+this.offset]=e,this}setY(t,e){return this.data.array[t*this.data.stride+this.offset+1]=e,this}setZ(t,e){return this.data.array[t*this.data.stride+this.offset+2]=e,this}setW(t,e){return this.data.array[t*this.data.stride+this.offset+3]=e,this}getX(t){return this.data.array[t*this.data.stride+this.offset]}getY(t){return this.data.array[t*this.data.stride+this.offset+1]}getZ(t){return this.data.array[t*this.data.stride+this.offset+2]}getW(t){return this.data.array[t*this.data.stride+this.offset+3]}setXY(t,e,n){return t=t*this.data.stride+this.offset,this.data.array[t+0]=e,this.data.array[t+1]=n,this}setXYZ(t,e,n,i){return t=t*this.data.stride+this.offset,this.data.array[t+0]=e,this.data.array[t+1]=n,this.data.array[t+2]=i,this}setXYZW(t,e,n,i,r){return t=t*this.data.stride+this.offset,this.data.array[t+0]=e,this.data.array[t+1]=n,this.data.array[t+2]=i,this.data.array[t+3]=r,this}clone(t){if(void 0===t){console.log("THREE.InterleavedBufferAttribute.clone(): Cloning an interlaved buffer attribute will deinterleave buffer data.");const t=[];for(let e=0;e<this.count;e++){const n=e*this.data.stride+this.offset;for(let e=0;e<this.itemSize;e++)t.push(this.data.array[n+e])}return new ln(new this.array.constructor(t),this.itemSize,this.normalized)}return void 0===t.interleavedBuffers&&(t.interleavedBuffers={}),void 0===t.interleavedBuffers[this.data.uuid]&&(t.interleavedBuffers[this.data.uuid]=this.data.clone(t)),new ra(t.interleavedBuffers[this.data.uuid],this.itemSize,this.offset,this.normalized)}toJSON(t){if(void 0===t){console.log("THREE.InterleavedBufferAttribute.toJSON(): Serializing an interlaved buffer attribute will deinterleave buffer data.");const t=[];for(let e=0;e<this.count;e++){const n=e*this.data.stride+this.offset;for(let e=0;e<this.itemSize;e++)t.push(this.data.array[n+e])}return{itemSize:this.itemSize,type:this.array.constructor.name,array:t,normalized:this.normalized}}return void 0===t.interleavedBuffers&&(t.interleavedBuffers={}),void 0===t.interleavedBuffers[this.data.uuid]&&(t.interleavedBuffers[this.data.uuid]=this.data.toJSON(t)),{isInterleavedBufferAttribute:!0,itemSize:this.itemSize,data:this.data.uuid,offset:this.offset,normalized:this.normalized}}}ra.prototype.isInterleavedBufferAttribute=!0;class sa extends Ze{constructor(t){super(),this.type="SpriteMaterial",this.color=new rn(16777215),this.map=null,this.alphaMap=null,this.rotation=0,this.sizeAttenuation=!0,this.transparent=!0,this.setValues(t)}copy(t){return super.copy(t),this.color.copy(t.color),this.map=t.map,this.alphaMap=t.alphaMap,this.rotation=t.rotation,this.sizeAttenuation=t.sizeAttenuation,this}}let aa;sa.prototype.isSpriteMaterial=!0;const oa=new zt,la=new zt,ca=new zt,ha=new yt,ua=new yt,da=new de,pa=new zt,ma=new zt,fa=new zt,ga=new yt,va=new yt,ya=new yt;class xa extends Fe{constructor(t){if(super(),this.type="Sprite",void 0===aa){aa=new En;const t=new Float32Array([-.5,-.5,0,0,0,.5,-.5,0,1,0,.5,.5,0,1,1,-.5,.5,0,0,1]),e=new na(t,5);aa.setIndex([0,1,2,0,2,3]),aa.setAttribute("position",new ra(e,3,0,!1)),aa.setAttribute("uv",new ra(e,2,3,!1))}this.geometry=aa,this.material=void 0!==t?t:new sa,this.center=new yt(.5,.5)}raycast(t,e){null===t.camera&&console.error('THREE.Sprite: "Raycaster.camera" needs to be set in order to raycast against sprites.'),la.setFromMatrixScale(this.matrixWorld),da.copy(t.camera.matrixWorld),this.modelViewMatrix.multiplyMatrices(t.camera.matrixWorldInverse,this.matrixWorld),ca.setFromMatrixPosition(this.modelViewMatrix),t.camera.isPerspectiveCamera&&!1===this.material.sizeAttenuation&&la.multiplyScalar(-ca.z);const n=this.material.rotation;let i,r;0!==n&&(r=Math.cos(n),i=Math.sin(n));const s=this.center;_a(pa.set(-.5,-.5,0),ca,s,la,i,r),_a(ma.set(.5,-.5,0),ca,s,la,i,r),_a(fa.set(.5,.5,0),ca,s,la,i,r),ga.set(0,0),va.set(1,0),ya.set(1,1);let a=t.ray.intersectTriangle(pa,ma,fa,!1,oa);if(null===a&&(_a(ma.set(-.5,.5,0),ca,s,la,i,r),va.set(0,1),a=t.ray.intersectTriangle(pa,fa,ma,!1,oa),null===a))return;const o=t.ray.origin.distanceTo(oa);o<t.near||o>t.far||e.push({distance:o,point:oa.clone(),uv:Ye.getUV(oa,pa,ma,fa,ga,va,ya,new yt),face:null,object:this})}copy(t){return super.copy(t),void 0!==t.center&&this.center.copy(t.center),this.material=t.material,this}}function _a(t,e,n,i,r,s){ha.subVectors(t,n).addScalar(.5).multiply(i),void 0!==r?(ua.x=s*ha.x-r*ha.y,ua.y=r*ha.x+s*ha.y):ua.copy(ha),t.copy(e),t.x+=ua.x,t.y+=ua.y,t.applyMatrix4(da)}xa.prototype.isSprite=!0;const Ma=new zt,ba=new zt;class wa extends Fe{constructor(){super(),this._currentLevel=0,this.type="LOD",Object.defineProperties(this,{levels:{enumerable:!0,value:[]},isLOD:{value:!0}}),this.autoUpdate=!0}copy(t){super.copy(t,!1);const e=t.levels;for(let t=0,n=e.length;t<n;t++){const n=e[t];this.addLevel(n.object.clone(),n.distance)}return this.autoUpdate=t.autoUpdate,this}addLevel(t,e=0){e=Math.abs(e);const n=this.levels;let i;for(i=0;i<n.length&&!(e<n[i].distance);i++);return n.splice(i,0,{distance:e,object:t}),this.add(t),this}getCurrentLevel(){return this._currentLevel}getObjectForDistance(t){const e=this.levels;if(e.length>0){let n,i;for(n=1,i=e.length;n<i&&!(t<e[n].distance);n++);return e[n-1].object}return null}raycast(t,e){if(this.levels.length>0){Ma.setFromMatrixPosition(this.matrixWorld);const n=t.ray.origin.distanceTo(Ma);this.getObjectForDistance(n).raycast(t,e)}}update(t){const e=this.levels;if(e.length>1){Ma.setFromMatrixPosition(t.matrixWorld),ba.setFromMatrixPosition(this.matrixWorld);const n=Ma.distanceTo(ba)/t.zoom;let i,r;for(e[0].object.visible=!0,i=1,r=e.length;i<r&&n>=e[i].distance;i++)e[i-1].object.visible=!1,e[i].object.visible=!0;for(this._currentLevel=i-1;i<r;i++)e[i].object.visible=!1}}toJSON(t){const e=super.toJSON(t);!1===this.autoUpdate&&(e.object.autoUpdate=!1),e.object.levels=[];const n=this.levels;for(let t=0,i=n.length;t<i;t++){const i=n[t];e.object.levels.push({object:i.object.uuid,distance:i.distance})}return e}}const Sa=new zt,Ta=new Ct,Ea=new Ct,Aa=new zt,La=new de;class Ra extends Wn{constructor(t,e){super(t,e),this.type="SkinnedMesh",this.bindMode="attached",this.bindMatrix=new de,this.bindMatrixInverse=new de}copy(t){return super.copy(t),this.bindMode=t.bindMode,this.bindMatrix.copy(t.bindMatrix),this.bindMatrixInverse.copy(t.bindMatrixInverse),this.skeleton=t.skeleton,this}bind(t,e){this.skeleton=t,void 0===e&&(this.updateMatrixWorld(!0),this.skeleton.calculateInverses(),e=this.matrixWorld),this.bindMatrix.copy(e),this.bindMatrixInverse.copy(e).invert()}pose(){this.skeleton.pose()}normalizeSkinWeights(){const t=new Ct,e=this.geometry.attributes.skinWeight;for(let n=0,i=e.count;n<i;n++){t.x=e.getX(n),t.y=e.getY(n),t.z=e.getZ(n),t.w=e.getW(n);const i=1/t.manhattanLength();i!==1/0?t.multiplyScalar(i):t.set(1,0,0,0),e.setXYZW(n,t.x,t.y,t.z,t.w)}}updateMatrixWorld(t){super.updateMatrixWorld(t),"attached"===this.bindMode?this.bindMatrixInverse.copy(this.matrixWorld).invert():"detached"===this.bindMode?this.bindMatrixInverse.copy(this.bindMatrix).invert():console.warn("THREE.SkinnedMesh: Unrecognized bindMode: "+this.bindMode)}boneTransform(t,e){const n=this.skeleton,i=this.geometry;Ta.fromBufferAttribute(i.attributes.skinIndex,t),Ea.fromBufferAttribute(i.attributes.skinWeight,t),Sa.copy(e).applyMatrix4(this.bindMatrix),e.set(0,0,0);for(let t=0;t<4;t++){const i=Ea.getComponent(t);if(0!==i){const r=Ta.getComponent(t);La.multiplyMatrices(n.bones[r].matrixWorld,n.boneInverses[r]),e.addScaledVector(Aa.copy(Sa).applyMatrix4(La),i)}}return e.applyMatrix4(this.bindMatrixInverse)}}Ra.prototype.isSkinnedMesh=!0;class Ca extends Fe{constructor(){super(),this.type="Bone"}}Ca.prototype.isBone=!0;class Pa extends Lt{constructor(t=null,e=1,n=1,i,r,s,a,o,l=1003,c=1003,h,u){super(null,s,a,o,l,c,i,r,h,u),this.image={data:t,width:e,height:n},this.magFilter=l,this.minFilter=c,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1,this.needsUpdate=!0}}Pa.prototype.isDataTexture=!0;const Ia=new de,Da=new de;class Na{constructor(t=[],e=[]){this.uuid=ht(),this.bones=t.slice(0),this.boneInverses=e,this.boneMatrices=null,this.boneTexture=null,this.boneTextureSize=0,this.frame=-1,this.init()}init(){const t=this.bones,e=this.boneInverses;if(this.boneMatrices=new Float32Array(16*t.length),0===e.length)this.calculateInverses();else if(t.length!==e.length){console.warn("THREE.Skeleton: Number of inverse bone matrices does not match amount of bones."),this.boneInverses=[];for(let t=0,e=this.bones.length;t<e;t++)this.boneInverses.push(new de)}}calculateInverses(){this.boneInverses.length=0;for(let t=0,e=this.bones.length;t<e;t++){const e=new de;this.bones[t]&&e.copy(this.bones[t].matrixWorld).invert(),this.boneInverses.push(e)}}pose(){for(let t=0,e=this.bones.length;t<e;t++){const e=this.bones[t];e&&e.matrixWorld.copy(this.boneInverses[t]).invert()}for(let t=0,e=this.bones.length;t<e;t++){const e=this.bones[t];e&&(e.parent&&e.parent.isBone?(e.matrix.copy(e.parent.matrixWorld).invert(),e.matrix.multiply(e.matrixWorld)):e.matrix.copy(e.matrixWorld),e.matrix.decompose(e.position,e.quaternion,e.scale))}}update(){const t=this.bones,e=this.boneInverses,n=this.boneMatrices,i=this.boneTexture;for(let i=0,r=t.length;i<r;i++){const r=t[i]?t[i].matrixWorld:Da;Ia.multiplyMatrices(r,e[i]),Ia.toArray(n,16*i)}null!==i&&(i.needsUpdate=!0)}clone(){return new Na(this.bones,this.boneInverses)}computeBoneTexture(){let t=Math.sqrt(4*this.bones.length);t=ft(t),t=Math.max(t,4);const e=new Float32Array(t*t*4);e.set(this.boneMatrices);const n=new Pa(e,t,t,E,b);return this.boneMatrices=e,this.boneTexture=n,this.boneTextureSize=t,this}getBoneByName(t){for(let e=0,n=this.bones.length;e<n;e++){const n=this.bones[e];if(n.name===t)return n}}dispose(){null!==this.boneTexture&&(this.boneTexture.dispose(),this.boneTexture=null)}fromJSON(t,e){this.uuid=t.uuid;for(let n=0,i=t.bones.length;n<i;n++){const i=t.bones[n];let r=e[i];void 0===r&&(console.warn("THREE.Skeleton: No bone found with UUID:",i),r=new Ca),this.bones.push(r),this.boneInverses.push((new de).fromArray(t.boneInverses[n]))}return this.init(),this}toJSON(){const t={metadata:{version:4.5,type:"Skeleton",generator:"Skeleton.toJSON"},bones:[],boneInverses:[]};t.uuid=this.uuid;const e=this.bones,n=this.boneInverses;for(let i=0,r=e.length;i<r;i++){const r=e[i];t.bones.push(r.uuid);const s=n[i];t.boneInverses.push(s.toArray())}return t}}class za extends ln{constructor(t,e,n,i=1){"number"==typeof n&&(i=n,n=!1,console.error("THREE.InstancedBufferAttribute: The constructor now expects normalized as the third argument.")),super(t,e,n),this.meshPerAttribute=i}copy(t){return super.copy(t),this.meshPerAttribute=t.meshPerAttribute,this}toJSON(){const t=super.toJSON();return t.meshPerAttribute=this.meshPerAttribute,t.isInstancedBufferAttribute=!0,t}}za.prototype.isInstancedBufferAttribute=!0;const Ba=new de,Fa=new de,Oa=[],Ua=new Wn;class Ha extends Wn{constructor(t,e,n){super(t,e),this.instanceMatrix=new za(new Float32Array(16*n),16),this.instanceColor=null,this.count=n,this.frustumCulled=!1}copy(t){return super.copy(t),this.instanceMatrix.copy(t.instanceMatrix),null!==t.instanceColor&&(this.instanceColor=t.instanceColor.clone()),this.count=t.count,this}getColorAt(t,e){e.fromArray(this.instanceColor.array,3*t)}getMatrixAt(t,e){e.fromArray(this.instanceMatrix.array,16*t)}raycast(t,e){const n=this.matrixWorld,i=this.count;if(Ua.geometry=this.geometry,Ua.material=this.material,void 0!==Ua.material)for(let r=0;r<i;r++){this.getMatrixAt(r,Ba),Fa.multiplyMatrices(n,Ba),Ua.matrixWorld=Fa,Ua.raycast(t,Oa);for(let t=0,n=Oa.length;t<n;t++){const n=Oa[t];n.instanceId=r,n.object=this,e.push(n)}Oa.length=0}}setColorAt(t,e){null===this.instanceColor&&(this.instanceColor=new za(new Float32Array(3*this.instanceMatrix.count),3)),e.toArray(this.instanceColor.array,3*t)}setMatrixAt(t,e){e.toArray(this.instanceMatrix.array,16*t)}updateMorphTargets(){}dispose(){this.dispatchEvent({type:"dispose"})}}Ha.prototype.isInstancedMesh=!0;class Ga extends Ze{constructor(t){super(),this.type="LineBasicMaterial",this.color=new rn(16777215),this.linewidth=1,this.linecap="round",this.linejoin="round",this.setValues(t)}copy(t){return super.copy(t),this.color.copy(t.color),this.linewidth=t.linewidth,this.linecap=t.linecap,this.linejoin=t.linejoin,this}}Ga.prototype.isLineBasicMaterial=!0;const ka=new zt,Va=new zt,Wa=new de,ja=new ue,qa=new ie;class Xa extends Fe{constructor(t=new En,e=new Ga){super(),this.type="Line",this.geometry=t,this.material=e,this.updateMorphTargets()}copy(t){return super.copy(t),this.material=t.material,this.geometry=t.geometry,this}computeLineDistances(){const t=this.geometry;if(t.isBufferGeometry)if(null===t.index){const e=t.attributes.position,n=[0];for(let t=1,i=e.count;t<i;t++)ka.fromBufferAttribute(e,t-1),Va.fromBufferAttribute(e,t),n[t]=n[t-1],n[t]+=ka.distanceTo(Va);t.setAttribute("lineDistance",new vn(n,1))}else console.warn("THREE.Line.computeLineDistances(): Computation only possible with non-indexed BufferGeometry.");else t.isGeometry&&console.error("THREE.Line.computeLineDistances() no longer supports THREE.Geometry. Use THREE.BufferGeometry instead.");return this}raycast(t,e){const n=this.geometry,i=this.matrixWorld,r=t.params.Line.threshold,s=n.drawRange;if(null===n.boundingSphere&&n.computeBoundingSphere(),qa.copy(n.boundingSphere),qa.applyMatrix4(i),qa.radius+=r,!1===t.ray.intersectsSphere(qa))return;Wa.copy(i).invert(),ja.copy(t.ray).applyMatrix4(Wa);const a=r/((this.scale.x+this.scale.y+this.scale.z)/3),o=a*a,l=new zt,c=new zt,h=new zt,u=new zt,d=this.isLineSegments?2:1;if(n.isBufferGeometry){const i=n.index,r=n.attributes.position;if(null!==i){for(let n=Math.max(0,s.start),a=Math.min(i.count,s.start+s.count)-1;n<a;n+=d){const s=i.getX(n),a=i.getX(n+1);l.fromBufferAttribute(r,s),c.fromBufferAttribute(r,a);if(ja.distanceSqToSegment(l,c,u,h)>o)continue;u.applyMatrix4(this.matrixWorld);const d=t.ray.origin.distanceTo(u);d<t.near||d>t.far||e.push({distance:d,point:h.clone().applyMatrix4(this.matrixWorld),index:n,face:null,faceIndex:null,object:this})}}else{for(let n=Math.max(0,s.start),i=Math.min(r.count,s.start+s.count)-1;n<i;n+=d){l.fromBufferAttribute(r,n),c.fromBufferAttribute(r,n+1);if(ja.distanceSqToSegment(l,c,u,h)>o)continue;u.applyMatrix4(this.matrixWorld);const i=t.ray.origin.distanceTo(u);i<t.near||i>t.far||e.push({distance:i,point:h.clone().applyMatrix4(this.matrixWorld),index:n,face:null,faceIndex:null,object:this})}}}else n.isGeometry&&console.error("THREE.Line.raycast() no longer supports THREE.Geometry. Use THREE.BufferGeometry instead.")}updateMorphTargets(){const t=this.geometry;if(t.isBufferGeometry){const e=t.morphAttributes,n=Object.keys(e);if(n.length>0){const t=e[n[0]];if(void 0!==t){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let e=0,n=t.length;e<n;e++){const n=t[e].name||String(e);this.morphTargetInfluences.push(0),this.morphTargetDictionary[n]=e}}}}else{const e=t.morphTargets;void 0!==e&&e.length>0&&console.error("THREE.Line.updateMorphTargets() does not support THREE.Geometry. Use THREE.BufferGeometry instead.")}}}Xa.prototype.isLine=!0;const Ya=new zt,Ja=new zt;class Za extends Xa{constructor(t,e){super(t,e),this.type="LineSegments"}computeLineDistances(){const t=this.geometry;if(t.isBufferGeometry)if(null===t.index){const e=t.attributes.position,n=[];for(let t=0,i=e.count;t<i;t+=2)Ya.fromBufferAttribute(e,t),Ja.fromBufferAttribute(e,t+1),n[t]=0===t?0:n[t-1],n[t+1]=n[t]+Ya.distanceTo(Ja);t.setAttribute("lineDistance",new vn(n,1))}else console.warn("THREE.LineSegments.computeLineDistances(): Computation only possible with non-indexed BufferGeometry.");else t.isGeometry&&console.error("THREE.LineSegments.computeLineDistances() no longer supports THREE.Geometry. Use THREE.BufferGeometry instead.");return this}}Za.prototype.isLineSegments=!0;class Qa extends Xa{constructor(t,e){super(t,e),this.type="LineLoop"}}Qa.prototype.isLineLoop=!0;class Ka extends Ze{constructor(t){super(),this.type="PointsMaterial",this.color=new rn(16777215),this.map=null,this.alphaMap=null,this.size=1,this.sizeAttenuation=!0,this.setValues(t)}copy(t){return super.copy(t),this.color.copy(t.color),this.map=t.map,this.alphaMap=t.alphaMap,this.size=t.size,this.sizeAttenuation=t.sizeAttenuation,this}}Ka.prototype.isPointsMaterial=!0;const $a=new de,to=new ue,eo=new ie,no=new zt;class io extends Fe{constructor(t=new En,e=new Ka){super(),this.type="Points",this.geometry=t,this.material=e,this.updateMorphTargets()}copy(t){return super.copy(t),this.material=t.material,this.geometry=t.geometry,this}raycast(t,e){const n=this.geometry,i=this.matrixWorld,r=t.params.Points.threshold,s=n.drawRange;if(null===n.boundingSphere&&n.computeBoundingSphere(),eo.copy(n.boundingSphere),eo.applyMatrix4(i),eo.radius+=r,!1===t.ray.intersectsSphere(eo))return;$a.copy(i).invert(),to.copy(t.ray).applyMatrix4($a);const a=r/((this.scale.x+this.scale.y+this.scale.z)/3),o=a*a;if(n.isBufferGeometry){const r=n.index,a=n.attributes.position;if(null!==r){for(let n=Math.max(0,s.start),l=Math.min(r.count,s.start+s.count);n<l;n++){const s=r.getX(n);no.fromBufferAttribute(a,s),ro(no,s,o,i,t,e,this)}}else{for(let n=Math.max(0,s.start),r=Math.min(a.count,s.start+s.count);n<r;n++)no.fromBufferAttribute(a,n),ro(no,n,o,i,t,e,this)}}else console.error("THREE.Points.raycast() no longer supports THREE.Geometry. Use THREE.BufferGeometry instead.")}updateMorphTargets(){const t=this.geometry;if(t.isBufferGeometry){const e=t.morphAttributes,n=Object.keys(e);if(n.length>0){const t=e[n[0]];if(void 0!==t){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let e=0,n=t.length;e<n;e++){const n=t[e].name||String(e);this.morphTargetInfluences.push(0),this.morphTargetDictionary[n]=e}}}}else{const e=t.morphTargets;void 0!==e&&e.length>0&&console.error("THREE.Points.updateMorphTargets() does not support THREE.Geometry. Use THREE.BufferGeometry instead.")}}}function ro(t,e,n,i,r,s,a){const o=to.distanceSqToPoint(t);if(o<n){const n=new zt;to.closestPointToPoint(t,n),n.applyMatrix4(i);const l=r.ray.origin.distanceTo(n);if(l<r.near||l>r.far)return;s.push({distance:l,distanceToRay:Math.sqrt(o),point:n,index:e,face:null,object:a})}}io.prototype.isPoints=!0;class so extends Lt{constructor(t,e,n,i,r,s,a,o,l){super(t,e,n,i,r,s,a,o,l),this.format=void 0!==a?a:T,this.minFilter=void 0!==s?s:g,this.magFilter=void 0!==r?r:g,this.generateMipmaps=!1;const c=this;"requestVideoFrameCallback"in t&&t.requestVideoFrameCallback((function e(){c.needsUpdate=!0,t.requestVideoFrameCallback(e)}))}clone(){return new this.constructor(this.image).copy(this)}update(){const t=this.image;!1==="requestVideoFrameCallback"in t&&t.readyState>=t.HAVE_CURRENT_DATA&&(this.needsUpdate=!0)}}so.prototype.isVideoTexture=!0;class ao extends Lt{constructor(t,e,n,i,r,s,a,o,l,c,h,u){super(null,s,a,o,l,c,i,r,h,u),this.image={width:e,height:n},this.mipmaps=t,this.flipY=!1,this.generateMipmaps=!1}}ao.prototype.isCompressedTexture=!0;class oo extends Lt{constructor(t,e,n,i,r,s,a,o,l){super(t,e,n,i,r,s,a,o,l),this.needsUpdate=!0}}oo.prototype.isCanvasTexture=!0;class lo extends Lt{constructor(t,e,n,i,r,s,a,o,l,c){if((c=void 0!==c?c:A)!==A&&c!==L)throw new Error("DepthTexture format must be either THREE.DepthFormat or THREE.DepthStencilFormat");void 0===n&&c===A&&(n=_),void 0===n&&c===L&&(n=S),super(null,i,r,s,a,o,c,n,l),this.image={width:t,height:e},this.magFilter=void 0!==a?a:p,this.minFilter=void 0!==o?o:p,this.flipY=!1,this.generateMipmaps=!1}}lo.prototype.isDepthTexture=!0;class co extends En{constructor(t=1,e=8,n=0,i=2*Math.PI){super(),this.type="CircleGeometry",this.parameters={radius:t,segments:e,thetaStart:n,thetaLength:i},e=Math.max(3,e);const r=[],s=[],a=[],o=[],l=new zt,c=new yt;s.push(0,0,0),a.push(0,0,1),o.push(.5,.5);for(let r=0,h=3;r<=e;r++,h+=3){const u=n+r/e*i;l.x=t*Math.cos(u),l.y=t*Math.sin(u),s.push(l.x,l.y,l.z),a.push(0,0,1),c.x=(s[h]/t+1)/2,c.y=(s[h+1]/t+1)/2,o.push(c.x,c.y)}for(let t=1;t<=e;t++)r.push(t,t+1,0);this.setIndex(r),this.setAttribute("position",new vn(s,3)),this.setAttribute("normal",new vn(a,3)),this.setAttribute("uv",new vn(o,2))}static fromJSON(t){return new co(t.radius,t.segments,t.thetaStart,t.thetaLength)}}class ho extends En{constructor(t=1,e=1,n=1,i=8,r=1,s=!1,a=0,o=2*Math.PI){super(),this.type="CylinderGeometry",this.parameters={radiusTop:t,radiusBottom:e,height:n,radialSegments:i,heightSegments:r,openEnded:s,thetaStart:a,thetaLength:o};const l=this;i=Math.floor(i),r=Math.floor(r);const c=[],h=[],u=[],d=[];let p=0;const m=[],f=n/2;let g=0;function v(n){const r=p,s=new yt,m=new zt;let v=0;const y=!0===n?t:e,x=!0===n?1:-1;for(let t=1;t<=i;t++)h.push(0,f*x,0),u.push(0,x,0),d.push(.5,.5),p++;const _=p;for(let t=0;t<=i;t++){const e=t/i*o+a,n=Math.cos(e),r=Math.sin(e);m.x=y*r,m.y=f*x,m.z=y*n,h.push(m.x,m.y,m.z),u.push(0,x,0),s.x=.5*n+.5,s.y=.5*r*x+.5,d.push(s.x,s.y),p++}for(let t=0;t<i;t++){const e=r+t,i=_+t;!0===n?c.push(i,i+1,e):c.push(i+1,i,e),v+=3}l.addGroup(g,v,!0===n?1:2),g+=v}!function(){const s=new zt,v=new zt;let y=0;const x=(e-t)/n;for(let l=0;l<=r;l++){const c=[],g=l/r,y=g*(e-t)+t;for(let t=0;t<=i;t++){const e=t/i,r=e*o+a,l=Math.sin(r),m=Math.cos(r);v.x=y*l,v.y=-g*n+f,v.z=y*m,h.push(v.x,v.y,v.z),s.set(l,x,m).normalize(),u.push(s.x,s.y,s.z),d.push(e,1-g),c.push(p++)}m.push(c)}for(let t=0;t<i;t++)for(let e=0;e<r;e++){const n=m[e][t],i=m[e+1][t],r=m[e+1][t+1],s=m[e][t+1];c.push(n,i,s),c.push(i,r,s),y+=6}l.addGroup(g,y,0),g+=y}(),!1===s&&(t>0&&v(!0),e>0&&v(!1)),this.setIndex(c),this.setAttribute("position",new vn(h,3)),this.setAttribute("normal",new vn(u,3)),this.setAttribute("uv",new vn(d,2))}static fromJSON(t){return new ho(t.radiusTop,t.radiusBottom,t.height,t.radialSegments,t.heightSegments,t.openEnded,t.thetaStart,t.thetaLength)}}class uo extends ho{constructor(t=1,e=1,n=8,i=1,r=!1,s=0,a=2*Math.PI){super(0,t,e,n,i,r,s,a),this.type="ConeGeometry",this.parameters={radius:t,height:e,radialSegments:n,heightSegments:i,openEnded:r,thetaStart:s,thetaLength:a}}static fromJSON(t){return new uo(t.radius,t.height,t.radialSegments,t.heightSegments,t.openEnded,t.thetaStart,t.thetaLength)}}class po extends En{constructor(t=[],e=[],n=1,i=0){super(),this.type="PolyhedronGeometry",this.parameters={vertices:t,indices:e,radius:n,detail:i};const r=[],s=[];function a(t,e,n,i){const r=i+1,s=[];for(let i=0;i<=r;i++){s[i]=[];const a=t.clone().lerp(n,i/r),o=e.clone().lerp(n,i/r),l=r-i;for(let t=0;t<=l;t++)s[i][t]=0===t&&i===r?a:a.clone().lerp(o,t/l)}for(let t=0;t<r;t++)for(let e=0;e<2*(r-t)-1;e++){const n=Math.floor(e/2);e%2==0?(o(s[t][n+1]),o(s[t+1][n]),o(s[t][n])):(o(s[t][n+1]),o(s[t+1][n+1]),o(s[t+1][n]))}}function o(t){r.push(t.x,t.y,t.z)}function l(e,n){const i=3*e;n.x=t[i+0],n.y=t[i+1],n.z=t[i+2]}function c(t,e,n,i){i<0&&1===t.x&&(s[e]=t.x-1),0===n.x&&0===n.z&&(s[e]=i/2/Math.PI+.5)}function h(t){return Math.atan2(t.z,-t.x)}!function(t){const n=new zt,i=new zt,r=new zt;for(let s=0;s<e.length;s+=3)l(e[s+0],n),l(e[s+1],i),l(e[s+2],r),a(n,i,r,t)}(i),function(t){const e=new zt;for(let n=0;n<r.length;n+=3)e.x=r[n+0],e.y=r[n+1],e.z=r[n+2],e.normalize().multiplyScalar(t),r[n+0]=e.x,r[n+1]=e.y,r[n+2]=e.z}(n),function(){const t=new zt;for(let n=0;n<r.length;n+=3){t.x=r[n+0],t.y=r[n+1],t.z=r[n+2];const i=h(t)/2/Math.PI+.5,a=(e=t,Math.atan2(-e.y,Math.sqrt(e.x*e.x+e.z*e.z))/Math.PI+.5);s.push(i,1-a)}var e;(function(){const t=new zt,e=new zt,n=new zt,i=new zt,a=new yt,o=new yt,l=new yt;for(let u=0,d=0;u<r.length;u+=9,d+=6){t.set(r[u+0],r[u+1],r[u+2]),e.set(r[u+3],r[u+4],r[u+5]),n.set(r[u+6],r[u+7],r[u+8]),a.set(s[d+0],s[d+1]),o.set(s[d+2],s[d+3]),l.set(s[d+4],s[d+5]),i.copy(t).add(e).add(n).divideScalar(3);const p=h(i);c(a,d+0,t,p),c(o,d+2,e,p),c(l,d+4,n,p)}})(),function(){for(let t=0;t<s.length;t+=6){const e=s[t+0],n=s[t+2],i=s[t+4],r=Math.max(e,n,i),a=Math.min(e,n,i);r>.9&&a<.1&&(e<.2&&(s[t+0]+=1),n<.2&&(s[t+2]+=1),i<.2&&(s[t+4]+=1))}}()}(),this.setAttribute("position",new vn(r,3)),this.setAttribute("normal",new vn(r.slice(),3)),this.setAttribute("uv",new vn(s,2)),0===i?this.computeVertexNormals():this.normalizeNormals()}static fromJSON(t){return new po(t.vertices,t.indices,t.radius,t.details)}}class mo extends po{constructor(t=1,e=0){const n=(1+Math.sqrt(5))/2,i=1/n;super([-1,-1,-1,-1,-1,1,-1,1,-1,-1,1,1,1,-1,-1,1,-1,1,1,1,-1,1,1,1,0,-i,-n,0,-i,n,0,i,-n,0,i,n,-i,-n,0,-i,n,0,i,-n,0,i,n,0,-n,0,-i,n,0,-i,-n,0,i,n,0,i],[3,11,7,3,7,15,3,15,13,7,19,17,7,17,6,7,6,15,17,4,8,17,8,10,17,10,6,8,0,16,8,16,2,8,2,10,0,12,1,0,1,18,0,18,16,6,10,2,6,2,13,6,13,15,2,16,18,2,18,3,2,3,13,18,1,9,18,9,11,18,11,3,4,14,12,4,12,0,4,0,8,11,9,5,11,5,19,11,19,7,19,5,14,19,14,4,19,4,17,1,12,14,1,14,5,1,5,9],t,e),this.type="DodecahedronGeometry",this.parameters={radius:t,detail:e}}static fromJSON(t){return new mo(t.radius,t.detail)}}const fo=new zt,go=new zt,vo=new zt,yo=new Ye;class xo extends En{constructor(t=null,e=1){if(super(),this.type="EdgesGeometry",this.parameters={geometry:t,thresholdAngle:e},null!==t){const n=4,i=Math.pow(10,n),r=Math.cos(at*e),s=t.getIndex(),a=t.getAttribute("position"),o=s?s.count:a.count,l=[0,0,0],c=["a","b","c"],h=new Array(3),u={},d=[];for(let t=0;t<o;t+=3){s?(l[0]=s.getX(t),l[1]=s.getX(t+1),l[2]=s.getX(t+2)):(l[0]=t,l[1]=t+1,l[2]=t+2);const{a:e,b:n,c:o}=yo;if(e.fromBufferAttribute(a,l[0]),n.fromBufferAttribute(a,l[1]),o.fromBufferAttribute(a,l[2]),yo.getNormal(vo),h[0]=`${Math.round(e.x*i)},${Math.round(e.y*i)},${Math.round(e.z*i)}`,h[1]=`${Math.round(n.x*i)},${Math.round(n.y*i)},${Math.round(n.z*i)}`,h[2]=`${Math.round(o.x*i)},${Math.round(o.y*i)},${Math.round(o.z*i)}`,h[0]!==h[1]&&h[1]!==h[2]&&h[2]!==h[0])for(let t=0;t<3;t++){const e=(t+1)%3,n=h[t],i=h[e],s=yo[c[t]],a=yo[c[e]],o=`${n}_${i}`,p=`${i}_${n}`;p in u&&u[p]?(vo.dot(u[p].normal)<=r&&(d.push(s.x,s.y,s.z),d.push(a.x,a.y,a.z)),u[p]=null):o in u||(u[o]={index0:l[t],index1:l[e],normal:vo.clone()})}}for(const t in u)if(u[t]){const{index0:e,index1:n}=u[t];fo.fromBufferAttribute(a,e),go.fromBufferAttribute(a,n),d.push(fo.x,fo.y,fo.z),d.push(go.x,go.y,go.z)}this.setAttribute("position",new vn(d,3))}}}class _o{constructor(){this.type="Curve",this.arcLengthDivisions=200}getPoint(){return console.warn("THREE.Curve: .getPoint() not implemented."),null}getPointAt(t,e){const n=this.getUtoTmapping(t);return this.getPoint(n,e)}getPoints(t=5){const e=[];for(let n=0;n<=t;n++)e.push(this.getPoint(n/t));return e}getSpacedPoints(t=5){const e=[];for(let n=0;n<=t;n++)e.push(this.getPointAt(n/t));return e}getLength(){const t=this.getLengths();return t[t.length-1]}getLengths(t=this.arcLengthDivisions){if(this.cacheArcLengths&&this.cacheArcLengths.length===t+1&&!this.needsUpdate)return this.cacheArcLengths;this.needsUpdate=!1;const e=[];let n,i=this.getPoint(0),r=0;e.push(0);for(let s=1;s<=t;s++)n=this.getPoint(s/t),r+=n.distanceTo(i),e.push(r),i=n;return this.cacheArcLengths=e,e}updateArcLengths(){this.needsUpdate=!0,this.getLengths()}getUtoTmapping(t,e){const n=this.getLengths();let i=0;const r=n.length;let s;s=e||t*n[r-1];let a,o=0,l=r-1;for(;o<=l;)if(i=Math.floor(o+(l-o)/2),a=n[i]-s,a<0)o=i+1;else{if(!(a>0)){l=i;break}l=i-1}if(i=l,n[i]===s)return i/(r-1);const c=n[i];return(i+(s-c)/(n[i+1]-c))/(r-1)}getTangent(t,e){const n=1e-4;let i=t-n,r=t+n;i<0&&(i=0),r>1&&(r=1);const s=this.getPoint(i),a=this.getPoint(r),o=e||(s.isVector2?new yt:new zt);return o.copy(a).sub(s).normalize(),o}getTangentAt(t,e){const n=this.getUtoTmapping(t);return this.getTangent(n,e)}computeFrenetFrames(t,e){const n=new zt,i=[],r=[],s=[],a=new zt,o=new de;for(let e=0;e<=t;e++){const n=e/t;i[e]=this.getTangentAt(n,new zt)}r[0]=new zt,s[0]=new zt;let l=Number.MAX_VALUE;const c=Math.abs(i[0].x),h=Math.abs(i[0].y),u=Math.abs(i[0].z);c<=l&&(l=c,n.set(1,0,0)),h<=l&&(l=h,n.set(0,1,0)),u<=l&&n.set(0,0,1),a.crossVectors(i[0],n).normalize(),r[0].crossVectors(i[0],a),s[0].crossVectors(i[0],r[0]);for(let e=1;e<=t;e++){if(r[e]=r[e-1].clone(),s[e]=s[e-1].clone(),a.crossVectors(i[e-1],i[e]),a.length()>Number.EPSILON){a.normalize();const t=Math.acos(ut(i[e-1].dot(i[e]),-1,1));r[e].applyMatrix4(o.makeRotationAxis(a,t))}s[e].crossVectors(i[e],r[e])}if(!0===e){let e=Math.acos(ut(r[0].dot(r[t]),-1,1));e/=t,i[0].dot(a.crossVectors(r[0],r[t]))>0&&(e=-e);for(let n=1;n<=t;n++)r[n].applyMatrix4(o.makeRotationAxis(i[n],e*n)),s[n].crossVectors(i[n],r[n])}return{tangents:i,normals:r,binormals:s}}clone(){return(new this.constructor).copy(this)}copy(t){return this.arcLengthDivisions=t.arcLengthDivisions,this}toJSON(){const t={metadata:{version:4.5,type:"Curve",generator:"Curve.toJSON"}};return t.arcLengthDivisions=this.arcLengthDivisions,t.type=this.type,t}fromJSON(t){return this.arcLengthDivisions=t.arcLengthDivisions,this}}class Mo extends _o{constructor(t=0,e=0,n=1,i=1,r=0,s=2*Math.PI,a=!1,o=0){super(),this.type="EllipseCurve",this.aX=t,this.aY=e,this.xRadius=n,this.yRadius=i,this.aStartAngle=r,this.aEndAngle=s,this.aClockwise=a,this.aRotation=o}getPoint(t,e){const n=e||new yt,i=2*Math.PI;let r=this.aEndAngle-this.aStartAngle;const s=Math.abs(r)<Number.EPSILON;for(;r<0;)r+=i;for(;r>i;)r-=i;r<Number.EPSILON&&(r=s?0:i),!0!==this.aClockwise||s||(r===i?r=-i:r-=i);const a=this.aStartAngle+t*r;let o=this.aX+this.xRadius*Math.cos(a),l=this.aY+this.yRadius*Math.sin(a);if(0!==this.aRotation){const t=Math.cos(this.aRotation),e=Math.sin(this.aRotation),n=o-this.aX,i=l-this.aY;o=n*t-i*e+this.aX,l=n*e+i*t+this.aY}return n.set(o,l)}copy(t){return super.copy(t),this.aX=t.aX,this.aY=t.aY,this.xRadius=t.xRadius,this.yRadius=t.yRadius,this.aStartAngle=t.aStartAngle,this.aEndAngle=t.aEndAngle,this.aClockwise=t.aClockwise,this.aRotation=t.aRotation,this}toJSON(){const t=super.toJSON();return t.aX=this.aX,t.aY=this.aY,t.xRadius=this.xRadius,t.yRadius=this.yRadius,t.aStartAngle=this.aStartAngle,t.aEndAngle=this.aEndAngle,t.aClockwise=this.aClockwise,t.aRotation=this.aRotation,t}fromJSON(t){return super.fromJSON(t),this.aX=t.aX,this.aY=t.aY,this.xRadius=t.xRadius,this.yRadius=t.yRadius,this.aStartAngle=t.aStartAngle,this.aEndAngle=t.aEndAngle,this.aClockwise=t.aClockwise,this.aRotation=t.aRotation,this}}Mo.prototype.isEllipseCurve=!0;class bo extends Mo{constructor(t,e,n,i,r,s){super(t,e,n,n,i,r,s),this.type="ArcCurve"}}function wo(){let t=0,e=0,n=0,i=0;function r(r,s,a,o){t=r,e=a,n=-3*r+3*s-2*a-o,i=2*r-2*s+a+o}return{initCatmullRom:function(t,e,n,i,s){r(e,n,s*(n-t),s*(i-e))},initNonuniformCatmullRom:function(t,e,n,i,s,a,o){let l=(e-t)/s-(n-t)/(s+a)+(n-e)/a,c=(n-e)/a-(i-e)/(a+o)+(i-n)/o;l*=a,c*=a,r(e,n,l,c)},calc:function(r){const s=r*r;return t+e*r+n*s+i*(s*r)}}}bo.prototype.isArcCurve=!0;const So=new zt,To=new wo,Eo=new wo,Ao=new wo;class Lo extends _o{constructor(t=[],e=!1,n="centripetal",i=.5){super(),this.type="CatmullRomCurve3",this.points=t,this.closed=e,this.curveType=n,this.tension=i}getPoint(t,e=new zt){const n=e,i=this.points,r=i.length,s=(r-(this.closed?0:1))*t;let a,o,l=Math.floor(s),c=s-l;this.closed?l+=l>0?0:(Math.floor(Math.abs(l)/r)+1)*r:0===c&&l===r-1&&(l=r-2,c=1),this.closed||l>0?a=i[(l-1)%r]:(So.subVectors(i[0],i[1]).add(i[0]),a=So);const h=i[l%r],u=i[(l+1)%r];if(this.closed||l+2<r?o=i[(l+2)%r]:(So.subVectors(i[r-1],i[r-2]).add(i[r-1]),o=So),"centripetal"===this.curveType||"chordal"===this.curveType){const t="chordal"===this.curveType?.5:.25;let e=Math.pow(a.distanceToSquared(h),t),n=Math.pow(h.distanceToSquared(u),t),i=Math.pow(u.distanceToSquared(o),t);n<1e-4&&(n=1),e<1e-4&&(e=n),i<1e-4&&(i=n),To.initNonuniformCatmullRom(a.x,h.x,u.x,o.x,e,n,i),Eo.initNonuniformCatmullRom(a.y,h.y,u.y,o.y,e,n,i),Ao.initNonuniformCatmullRom(a.z,h.z,u.z,o.z,e,n,i)}else"catmullrom"===this.curveType&&(To.initCatmullRom(a.x,h.x,u.x,o.x,this.tension),Eo.initCatmullRom(a.y,h.y,u.y,o.y,this.tension),Ao.initCatmullRom(a.z,h.z,u.z,o.z,this.tension));return n.set(To.calc(c),Eo.calc(c),Ao.calc(c)),n}copy(t){super.copy(t),this.points=[];for(let e=0,n=t.points.length;e<n;e++){const n=t.points[e];this.points.push(n.clone())}return this.closed=t.closed,this.curveType=t.curveType,this.tension=t.tension,this}toJSON(){const t=super.toJSON();t.points=[];for(let e=0,n=this.points.length;e<n;e++){const n=this.points[e];t.points.push(n.toArray())}return t.closed=this.closed,t.curveType=this.curveType,t.tension=this.tension,t}fromJSON(t){super.fromJSON(t),this.points=[];for(let e=0,n=t.points.length;e<n;e++){const n=t.points[e];this.points.push((new zt).fromArray(n))}return this.closed=t.closed,this.curveType=t.curveType,this.tension=t.tension,this}}function Ro(t,e,n,i,r){const s=.5*(i-e),a=.5*(r-n),o=t*t;return(2*n-2*i+s+a)*(t*o)+(-3*n+3*i-2*s-a)*o+s*t+n}function Co(t,e,n,i){return function(t,e){const n=1-t;return n*n*e}(t,e)+function(t,e){return 2*(1-t)*t*e}(t,n)+function(t,e){return t*t*e}(t,i)}function Po(t,e,n,i,r){return function(t,e){const n=1-t;return n*n*n*e}(t,e)+function(t,e){const n=1-t;return 3*n*n*t*e}(t,n)+function(t,e){return 3*(1-t)*t*t*e}(t,i)+function(t,e){return t*t*t*e}(t,r)}Lo.prototype.isCatmullRomCurve3=!0;class Io extends _o{constructor(t=new yt,e=new yt,n=new yt,i=new yt){super(),this.type="CubicBezierCurve",this.v0=t,this.v1=e,this.v2=n,this.v3=i}getPoint(t,e=new yt){const n=e,i=this.v0,r=this.v1,s=this.v2,a=this.v3;return n.set(Po(t,i.x,r.x,s.x,a.x),Po(t,i.y,r.y,s.y,a.y)),n}copy(t){return super.copy(t),this.v0.copy(t.v0),this.v1.copy(t.v1),this.v2.copy(t.v2),this.v3.copy(t.v3),this}toJSON(){const t=super.toJSON();return t.v0=this.v0.toArray(),t.v1=this.v1.toArray(),t.v2=this.v2.toArray(),t.v3=this.v3.toArray(),t}fromJSON(t){return super.fromJSON(t),this.v0.fromArray(t.v0),this.v1.fromArray(t.v1),this.v2.fromArray(t.v2),this.v3.fromArray(t.v3),this}}Io.prototype.isCubicBezierCurve=!0;class Do extends _o{constructor(t=new zt,e=new zt,n=new zt,i=new zt){super(),this.type="CubicBezierCurve3",this.v0=t,this.v1=e,this.v2=n,this.v3=i}getPoint(t,e=new zt){const n=e,i=this.v0,r=this.v1,s=this.v2,a=this.v3;return n.set(Po(t,i.x,r.x,s.x,a.x),Po(t,i.y,r.y,s.y,a.y),Po(t,i.z,r.z,s.z,a.z)),n}copy(t){return super.copy(t),this.v0.copy(t.v0),this.v1.copy(t.v1),this.v2.copy(t.v2),this.v3.copy(t.v3),this}toJSON(){const t=super.toJSON();return t.v0=this.v0.toArray(),t.v1=this.v1.toArray(),t.v2=this.v2.toArray(),t.v3=this.v3.toArray(),t}fromJSON(t){return super.fromJSON(t),this.v0.fromArray(t.v0),this.v1.fromArray(t.v1),this.v2.fromArray(t.v2),this.v3.fromArray(t.v3),this}}Do.prototype.isCubicBezierCurve3=!0;class No extends _o{constructor(t=new yt,e=new yt){super(),this.type="LineCurve",this.v1=t,this.v2=e}getPoint(t,e=new yt){const n=e;return 1===t?n.copy(this.v2):(n.copy(this.v2).sub(this.v1),n.multiplyScalar(t).add(this.v1)),n}getPointAt(t,e){return this.getPoint(t,e)}getTangent(t,e){const n=e||new yt;return n.copy(this.v2).sub(this.v1).normalize(),n}copy(t){return super.copy(t),this.v1.copy(t.v1),this.v2.copy(t.v2),this}toJSON(){const t=super.toJSON();return t.v1=this.v1.toArray(),t.v2=this.v2.toArray(),t}fromJSON(t){return super.fromJSON(t),this.v1.fromArray(t.v1),this.v2.fromArray(t.v2),this}}No.prototype.isLineCurve=!0;class zo extends _o{constructor(t=new zt,e=new zt){super(),this.type="LineCurve3",this.isLineCurve3=!0,this.v1=t,this.v2=e}getPoint(t,e=new zt){const n=e;return 1===t?n.copy(this.v2):(n.copy(this.v2).sub(this.v1),n.multiplyScalar(t).add(this.v1)),n}getPointAt(t,e){return this.getPoint(t,e)}copy(t){return super.copy(t),this.v1.copy(t.v1),this.v2.copy(t.v2),this}toJSON(){const t=super.toJSON();return t.v1=this.v1.toArray(),t.v2=this.v2.toArray(),t}fromJSON(t){return super.fromJSON(t),this.v1.fromArray(t.v1),this.v2.fromArray(t.v2),this}}class Bo extends _o{constructor(t=new yt,e=new yt,n=new yt){super(),this.type="QuadraticBezierCurve",this.v0=t,this.v1=e,this.v2=n}getPoint(t,e=new yt){const n=e,i=this.v0,r=this.v1,s=this.v2;return n.set(Co(t,i.x,r.x,s.x),Co(t,i.y,r.y,s.y)),n}copy(t){return super.copy(t),this.v0.copy(t.v0),this.v1.copy(t.v1),this.v2.copy(t.v2),this}toJSON(){const t=super.toJSON();return t.v0=this.v0.toArray(),t.v1=this.v1.toArray(),t.v2=this.v2.toArray(),t}fromJSON(t){return super.fromJSON(t),this.v0.fromArray(t.v0),this.v1.fromArray(t.v1),this.v2.fromArray(t.v2),this}}Bo.prototype.isQuadraticBezierCurve=!0;class Fo extends _o{constructor(t=new zt,e=new zt,n=new zt){super(),this.type="QuadraticBezierCurve3",this.v0=t,this.v1=e,this.v2=n}getPoint(t,e=new zt){const n=e,i=this.v0,r=this.v1,s=this.v2;return n.set(Co(t,i.x,r.x,s.x),Co(t,i.y,r.y,s.y),Co(t,i.z,r.z,s.z)),n}copy(t){return super.copy(t),this.v0.copy(t.v0),this.v1.copy(t.v1),this.v2.copy(t.v2),this}toJSON(){const t=super.toJSON();return t.v0=this.v0.toArray(),t.v1=this.v1.toArray(),t.v2=this.v2.toArray(),t}fromJSON(t){return super.fromJSON(t),this.v0.fromArray(t.v0),this.v1.fromArray(t.v1),this.v2.fromArray(t.v2),this}}Fo.prototype.isQuadraticBezierCurve3=!0;class Oo extends _o{constructor(t=[]){super(),this.type="SplineCurve",this.points=t}getPoint(t,e=new yt){const n=e,i=this.points,r=(i.length-1)*t,s=Math.floor(r),a=r-s,o=i[0===s?s:s-1],l=i[s],c=i[s>i.length-2?i.length-1:s+1],h=i[s>i.length-3?i.length-1:s+2];return n.set(Ro(a,o.x,l.x,c.x,h.x),Ro(a,o.y,l.y,c.y,h.y)),n}copy(t){super.copy(t),this.points=[];for(let e=0,n=t.points.length;e<n;e++){const n=t.points[e];this.points.push(n.clone())}return this}toJSON(){const t=super.toJSON();t.points=[];for(let e=0,n=this.points.length;e<n;e++){const n=this.points[e];t.points.push(n.toArray())}return t}fromJSON(t){super.fromJSON(t),this.points=[];for(let e=0,n=t.points.length;e<n;e++){const n=t.points[e];this.points.push((new yt).fromArray(n))}return this}}Oo.prototype.isSplineCurve=!0;var Uo=Object.freeze({__proto__:null,ArcCurve:bo,CatmullRomCurve3:Lo,CubicBezierCurve:Io,CubicBezierCurve3:Do,EllipseCurve:Mo,LineCurve:No,LineCurve3:zo,QuadraticBezierCurve:Bo,QuadraticBezierCurve3:Fo,SplineCurve:Oo});class Ho extends _o{constructor(){super(),this.type="CurvePath",this.curves=[],this.autoClose=!1}add(t){this.curves.push(t)}closePath(){const t=this.curves[0].getPoint(0),e=this.curves[this.curves.length-1].getPoint(1);t.equals(e)||this.curves.push(new No(e,t))}getPoint(t,e){const n=t*this.getLength(),i=this.getCurveLengths();let r=0;for(;r<i.length;){if(i[r]>=n){const t=i[r]-n,s=this.curves[r],a=s.getLength(),o=0===a?0:1-t/a;return s.getPointAt(o,e)}r++}return null}getLength(){const t=this.getCurveLengths();return t[t.length-1]}updateArcLengths(){this.needsUpdate=!0,this.cacheLengths=null,this.getCurveLengths()}getCurveLengths(){if(this.cacheLengths&&this.cacheLengths.length===this.curves.length)return this.cacheLengths;const t=[];let e=0;for(let n=0,i=this.curves.length;n<i;n++)e+=this.curves[n].getLength(),t.push(e);return this.cacheLengths=t,t}getSpacedPoints(t=40){const e=[];for(let n=0;n<=t;n++)e.push(this.getPoint(n/t));return this.autoClose&&e.push(e[0]),e}getPoints(t=12){const e=[];let n;for(let i=0,r=this.curves;i<r.length;i++){const s=r[i],a=s&&s.isEllipseCurve?2*t:s&&(s.isLineCurve||s.isLineCurve3)?1:s&&s.isSplineCurve?t*s.points.length:t,o=s.getPoints(a);for(let t=0;t<o.length;t++){const i=o[t];n&&n.equals(i)||(e.push(i),n=i)}}return this.autoClose&&e.length>1&&!e[e.length-1].equals(e[0])&&e.push(e[0]),e}copy(t){super.copy(t),this.curves=[];for(let e=0,n=t.curves.length;e<n;e++){const n=t.curves[e];this.curves.push(n.clone())}return this.autoClose=t.autoClose,this}toJSON(){const t=super.toJSON();t.autoClose=this.autoClose,t.curves=[];for(let e=0,n=this.curves.length;e<n;e++){const n=this.curves[e];t.curves.push(n.toJSON())}return t}fromJSON(t){super.fromJSON(t),this.autoClose=t.autoClose,this.curves=[];for(let e=0,n=t.curves.length;e<n;e++){const n=t.curves[e];this.curves.push((new Uo[n.type]).fromJSON(n))}return this}}class Go extends Ho{constructor(t){super(),this.type="Path",this.currentPoint=new yt,t&&this.setFromPoints(t)}setFromPoints(t){this.moveTo(t[0].x,t[0].y);for(let e=1,n=t.length;e<n;e++)this.lineTo(t[e].x,t[e].y);return this}moveTo(t,e){return this.currentPoint.set(t,e),this}lineTo(t,e){const n=new No(this.currentPoint.clone(),new yt(t,e));return this.curves.push(n),this.currentPoint.set(t,e),this}quadraticCurveTo(t,e,n,i){const r=new Bo(this.currentPoint.clone(),new yt(t,e),new yt(n,i));return this.curves.push(r),this.currentPoint.set(n,i),this}bezierCurveTo(t,e,n,i,r,s){const a=new Io(this.currentPoint.clone(),new yt(t,e),new yt(n,i),new yt(r,s));return this.curves.push(a),this.currentPoint.set(r,s),this}splineThru(t){const e=[this.currentPoint.clone()].concat(t),n=new Oo(e);return this.curves.push(n),this.currentPoint.copy(t[t.length-1]),this}arc(t,e,n,i,r,s){const a=this.currentPoint.x,o=this.currentPoint.y;return this.absarc(t+a,e+o,n,i,r,s),this}absarc(t,e,n,i,r,s){return this.absellipse(t,e,n,n,i,r,s),this}ellipse(t,e,n,i,r,s,a,o){const l=this.currentPoint.x,c=this.currentPoint.y;return this.absellipse(t+l,e+c,n,i,r,s,a,o),this}absellipse(t,e,n,i,r,s,a,o){const l=new Mo(t,e,n,i,r,s,a,o);if(this.curves.length>0){const t=l.getPoint(0);t.equals(this.currentPoint)||this.lineTo(t.x,t.y)}this.curves.push(l);const c=l.getPoint(1);return this.currentPoint.copy(c),this}copy(t){return super.copy(t),this.currentPoint.copy(t.currentPoint),this}toJSON(){const t=super.toJSON();return t.currentPoint=this.currentPoint.toArray(),t}fromJSON(t){return super.fromJSON(t),this.currentPoint.fromArray(t.currentPoint),this}}class ko extends Go{constructor(t){super(t),this.uuid=ht(),this.type="Shape",this.holes=[]}getPointsHoles(t){const e=[];for(let n=0,i=this.holes.length;n<i;n++)e[n]=this.holes[n].getPoints(t);return e}extractPoints(t){return{shape:this.getPoints(t),holes:this.getPointsHoles(t)}}copy(t){super.copy(t),this.holes=[];for(let e=0,n=t.holes.length;e<n;e++){const n=t.holes[e];this.holes.push(n.clone())}return this}toJSON(){const t=super.toJSON();t.uuid=this.uuid,t.holes=[];for(let e=0,n=this.holes.length;e<n;e++){const n=this.holes[e];t.holes.push(n.toJSON())}return t}fromJSON(t){super.fromJSON(t),this.uuid=t.uuid,this.holes=[];for(let e=0,n=t.holes.length;e<n;e++){const n=t.holes[e];this.holes.push((new Go).fromJSON(n))}return this}}const Vo=function(t,e,n=2){const i=e&&e.length,r=i?e[0]*n:t.length;let s=Wo(t,0,r,n,!0);const a=[];if(!s||s.next===s.prev)return a;let o,l,c,h,u,d,p;if(i&&(s=function(t,e,n,i){const r=[];let s,a,o,l,c;for(s=0,a=e.length;s<a;s++)o=e[s]*i,l=s<a-1?e[s+1]*i:t.length,c=Wo(t,o,l,i,!1),c===c.next&&(c.steiner=!0),r.push(el(c));for(r.sort(Qo),s=0;s<r.length;s++)Ko(r[s],n),n=jo(n,n.next);return n}(t,e,s,n)),t.length>80*n){o=c=t[0],l=h=t[1];for(let e=n;e<r;e+=n)u=t[e],d=t[e+1],u<o&&(o=u),d<l&&(l=d),u>c&&(c=u),d>h&&(h=d);p=Math.max(c-o,h-l),p=0!==p?1/p:0}return qo(s,a,n,o,l,p),a};function Wo(t,e,n,i,r){let s,a;if(r===function(t,e,n,i){let r=0;for(let s=e,a=n-i;s<n;s+=i)r+=(t[a]-t[s])*(t[s+1]+t[a+1]),a=s;return r}(t,e,n,i)>0)for(s=e;s<n;s+=i)a=ul(s,t[s],t[s+1],a);else for(s=n-i;s>=e;s-=i)a=ul(s,t[s],t[s+1],a);return a&&sl(a,a.next)&&(dl(a),a=a.next),a}function jo(t,e){if(!t)return t;e||(e=t);let n,i=t;do{if(n=!1,i.steiner||!sl(i,i.next)&&0!==rl(i.prev,i,i.next))i=i.next;else{if(dl(i),i=e=i.prev,i===i.next)break;n=!0}}while(n||i!==e);return e}function qo(t,e,n,i,r,s,a){if(!t)return;!a&&s&&function(t,e,n,i){let r=t;do{null===r.z&&(r.z=tl(r.x,r.y,e,n,i)),r.prevZ=r.prev,r.nextZ=r.next,r=r.next}while(r!==t);r.prevZ.nextZ=null,r.prevZ=null,function(t){let e,n,i,r,s,a,o,l,c=1;do{for(n=t,t=null,s=null,a=0;n;){for(a++,i=n,o=0,e=0;e<c&&(o++,i=i.nextZ,i);e++);for(l=c;o>0||l>0&&i;)0!==o&&(0===l||!i||n.z<=i.z)?(r=n,n=n.nextZ,o--):(r=i,i=i.nextZ,l--),s?s.nextZ=r:t=r,r.prevZ=s,s=r;n=i}s.nextZ=null,c*=2}while(a>1)}(r)}(t,i,r,s);let o,l,c=t;for(;t.prev!==t.next;)if(o=t.prev,l=t.next,s?Yo(t,i,r,s):Xo(t))e.push(o.i/n),e.push(t.i/n),e.push(l.i/n),dl(t),t=l.next,c=l.next;else if((t=l)===c){a?1===a?qo(t=Jo(jo(t),e,n),e,n,i,r,s,2):2===a&&Zo(t,e,n,i,r,s):qo(jo(t),e,n,i,r,s,1);break}}function Xo(t){const e=t.prev,n=t,i=t.next;if(rl(e,n,i)>=0)return!1;let r=t.next.next;for(;r!==t.prev;){if(nl(e.x,e.y,n.x,n.y,i.x,i.y,r.x,r.y)&&rl(r.prev,r,r.next)>=0)return!1;r=r.next}return!0}function Yo(t,e,n,i){const r=t.prev,s=t,a=t.next;if(rl(r,s,a)>=0)return!1;const o=r.x<s.x?r.x<a.x?r.x:a.x:s.x<a.x?s.x:a.x,l=r.y<s.y?r.y<a.y?r.y:a.y:s.y<a.y?s.y:a.y,c=r.x>s.x?r.x>a.x?r.x:a.x:s.x>a.x?s.x:a.x,h=r.y>s.y?r.y>a.y?r.y:a.y:s.y>a.y?s.y:a.y,u=tl(o,l,e,n,i),d=tl(c,h,e,n,i);let p=t.prevZ,m=t.nextZ;for(;p&&p.z>=u&&m&&m.z<=d;){if(p!==t.prev&&p!==t.next&&nl(r.x,r.y,s.x,s.y,a.x,a.y,p.x,p.y)&&rl(p.prev,p,p.next)>=0)return!1;if(p=p.prevZ,m!==t.prev&&m!==t.next&&nl(r.x,r.y,s.x,s.y,a.x,a.y,m.x,m.y)&&rl(m.prev,m,m.next)>=0)return!1;m=m.nextZ}for(;p&&p.z>=u;){if(p!==t.prev&&p!==t.next&&nl(r.x,r.y,s.x,s.y,a.x,a.y,p.x,p.y)&&rl(p.prev,p,p.next)>=0)return!1;p=p.prevZ}for(;m&&m.z<=d;){if(m!==t.prev&&m!==t.next&&nl(r.x,r.y,s.x,s.y,a.x,a.y,m.x,m.y)&&rl(m.prev,m,m.next)>=0)return!1;m=m.nextZ}return!0}function Jo(t,e,n){let i=t;do{const r=i.prev,s=i.next.next;!sl(r,s)&&al(r,i,i.next,s)&&cl(r,s)&&cl(s,r)&&(e.push(r.i/n),e.push(i.i/n),e.push(s.i/n),dl(i),dl(i.next),i=t=s),i=i.next}while(i!==t);return jo(i)}function Zo(t,e,n,i,r,s){let a=t;do{let t=a.next.next;for(;t!==a.prev;){if(a.i!==t.i&&il(a,t)){let o=hl(a,t);return a=jo(a,a.next),o=jo(o,o.next),qo(a,e,n,i,r,s),void qo(o,e,n,i,r,s)}t=t.next}a=a.next}while(a!==t)}function Qo(t,e){return t.x-e.x}function Ko(t,e){if(e=function(t,e){let n=e;const i=t.x,r=t.y;let s,a=-1/0;do{if(r<=n.y&&r>=n.next.y&&n.next.y!==n.y){const t=n.x+(r-n.y)*(n.next.x-n.x)/(n.next.y-n.y);if(t<=i&&t>a){if(a=t,t===i){if(r===n.y)return n;if(r===n.next.y)return n.next}s=n.x<n.next.x?n:n.next}}n=n.next}while(n!==e);if(!s)return null;if(i===a)return s;const o=s,l=s.x,c=s.y;let h,u=1/0;n=s;do{i>=n.x&&n.x>=l&&i!==n.x&&nl(r<c?i:a,r,l,c,r<c?a:i,r,n.x,n.y)&&(h=Math.abs(r-n.y)/(i-n.x),cl(n,t)&&(h<u||h===u&&(n.x>s.x||n.x===s.x&&$o(s,n)))&&(s=n,u=h)),n=n.next}while(n!==o);return s}(t,e),e){const n=hl(e,t);jo(e,e.next),jo(n,n.next)}}function $o(t,e){return rl(t.prev,t,e.prev)<0&&rl(e.next,t,t.next)<0}function tl(t,e,n,i,r){return(t=1431655765&((t=858993459&((t=252645135&((t=16711935&((t=32767*(t-n)*r)|t<<8))|t<<4))|t<<2))|t<<1))|(e=1431655765&((e=858993459&((e=252645135&((e=16711935&((e=32767*(e-i)*r)|e<<8))|e<<4))|e<<2))|e<<1))<<1}function el(t){let e=t,n=t;do{(e.x<n.x||e.x===n.x&&e.y<n.y)&&(n=e),e=e.next}while(e!==t);return n}function nl(t,e,n,i,r,s,a,o){return(r-a)*(e-o)-(t-a)*(s-o)>=0&&(t-a)*(i-o)-(n-a)*(e-o)>=0&&(n-a)*(s-o)-(r-a)*(i-o)>=0}function il(t,e){return t.next.i!==e.i&&t.prev.i!==e.i&&!function(t,e){let n=t;do{if(n.i!==t.i&&n.next.i!==t.i&&n.i!==e.i&&n.next.i!==e.i&&al(n,n.next,t,e))return!0;n=n.next}while(n!==t);return!1}(t,e)&&(cl(t,e)&&cl(e,t)&&function(t,e){let n=t,i=!1;const r=(t.x+e.x)/2,s=(t.y+e.y)/2;do{n.y>s!=n.next.y>s&&n.next.y!==n.y&&r<(n.next.x-n.x)*(s-n.y)/(n.next.y-n.y)+n.x&&(i=!i),n=n.next}while(n!==t);return i}(t,e)&&(rl(t.prev,t,e.prev)||rl(t,e.prev,e))||sl(t,e)&&rl(t.prev,t,t.next)>0&&rl(e.prev,e,e.next)>0)}function rl(t,e,n){return(e.y-t.y)*(n.x-e.x)-(e.x-t.x)*(n.y-e.y)}function sl(t,e){return t.x===e.x&&t.y===e.y}function al(t,e,n,i){const r=ll(rl(t,e,n)),s=ll(rl(t,e,i)),a=ll(rl(n,i,t)),o=ll(rl(n,i,e));return r!==s&&a!==o||(!(0!==r||!ol(t,n,e))||(!(0!==s||!ol(t,i,e))||(!(0!==a||!ol(n,t,i))||!(0!==o||!ol(n,e,i)))))}function ol(t,e,n){return e.x<=Math.max(t.x,n.x)&&e.x>=Math.min(t.x,n.x)&&e.y<=Math.max(t.y,n.y)&&e.y>=Math.min(t.y,n.y)}function ll(t){return t>0?1:t<0?-1:0}function cl(t,e){return rl(t.prev,t,t.next)<0?rl(t,e,t.next)>=0&&rl(t,t.prev,e)>=0:rl(t,e,t.prev)<0||rl(t,t.next,e)<0}function hl(t,e){const n=new pl(t.i,t.x,t.y),i=new pl(e.i,e.x,e.y),r=t.next,s=e.prev;return t.next=e,e.prev=t,n.next=r,r.prev=n,i.next=n,n.prev=i,s.next=i,i.prev=s,i}function ul(t,e,n,i){const r=new pl(t,e,n);return i?(r.next=i.next,r.prev=i,i.next.prev=r,i.next=r):(r.prev=r,r.next=r),r}function dl(t){t.next.prev=t.prev,t.prev.next=t.next,t.prevZ&&(t.prevZ.nextZ=t.nextZ),t.nextZ&&(t.nextZ.prevZ=t.prevZ)}function pl(t,e,n){this.i=t,this.x=e,this.y=n,this.prev=null,this.next=null,this.z=null,this.prevZ=null,this.nextZ=null,this.steiner=!1}class ml{static area(t){const e=t.length;let n=0;for(let i=e-1,r=0;r<e;i=r++)n+=t[i].x*t[r].y-t[r].x*t[i].y;return.5*n}static isClockWise(t){return ml.area(t)<0}static triangulateShape(t,e){const n=[],i=[],r=[];fl(t),gl(n,t);let s=t.length;e.forEach(fl);for(let t=0;t<e.length;t++)i.push(s),s+=e[t].length,gl(n,e[t]);const a=Vo(n,i);for(let t=0;t<a.length;t+=3)r.push(a.slice(t,t+3));return r}}function fl(t){const e=t.length;e>2&&t[e-1].equals(t[0])&&t.pop()}function gl(t,e){for(let n=0;n<e.length;n++)t.push(e[n].x),t.push(e[n].y)}class vl extends En{constructor(t=new ko([new yt(.5,.5),new yt(-.5,.5),new yt(-.5,-.5),new yt(.5,-.5)]),e={}){super(),this.type="ExtrudeGeometry",this.parameters={shapes:t,options:e},t=Array.isArray(t)?t:[t];const n=this,i=[],r=[];for(let e=0,n=t.length;e<n;e++){s(t[e])}function s(t){const s=[],a=void 0!==e.curveSegments?e.curveSegments:12,o=void 0!==e.steps?e.steps:1;let l=void 0!==e.depth?e.depth:1,c=void 0===e.bevelEnabled||e.bevelEnabled,h=void 0!==e.bevelThickness?e.bevelThickness:.2,u=void 0!==e.bevelSize?e.bevelSize:h-.1,d=void 0!==e.bevelOffset?e.bevelOffset:0,p=void 0!==e.bevelSegments?e.bevelSegments:3;const m=e.extrudePath,f=void 0!==e.UVGenerator?e.UVGenerator:yl;void 0!==e.amount&&(console.warn("THREE.ExtrudeBufferGeometry: amount has been renamed to depth."),l=e.amount);let g,v,y,x,_,M=!1;m&&(g=m.getSpacedPoints(o),M=!0,c=!1,v=m.computeFrenetFrames(o,!1),y=new zt,x=new zt,_=new zt),c||(p=0,h=0,u=0,d=0);const b=t.extractPoints(a);let w=b.shape;const S=b.holes;if(!ml.isClockWise(w)){w=w.reverse();for(let t=0,e=S.length;t<e;t++){const e=S[t];ml.isClockWise(e)&&(S[t]=e.reverse())}}const T=ml.triangulateShape(w,S),E=w;for(let t=0,e=S.length;t<e;t++){const e=S[t];w=w.concat(e)}function A(t,e,n){return e||console.error("THREE.ExtrudeGeometry: vec does not exist"),e.clone().multiplyScalar(n).add(t)}const L=w.length,R=T.length;function C(t,e,n){let i,r,s;const a=t.x-e.x,o=t.y-e.y,l=n.x-t.x,c=n.y-t.y,h=a*a+o*o,u=a*c-o*l;if(Math.abs(u)>Number.EPSILON){const u=Math.sqrt(h),d=Math.sqrt(l*l+c*c),p=e.x-o/u,m=e.y+a/u,f=((n.x-c/d-p)*c-(n.y+l/d-m)*l)/(a*c-o*l);i=p+a*f-t.x,r=m+o*f-t.y;const g=i*i+r*r;if(g<=2)return new yt(i,r);s=Math.sqrt(g/2)}else{let t=!1;a>Number.EPSILON?l>Number.EPSILON&&(t=!0):a<-Number.EPSILON?l<-Number.EPSILON&&(t=!0):Math.sign(o)===Math.sign(c)&&(t=!0),t?(i=-o,r=a,s=Math.sqrt(h)):(i=a,r=o,s=Math.sqrt(h/2))}return new yt(i/s,r/s)}const P=[];for(let t=0,e=E.length,n=e-1,i=t+1;t<e;t++,n++,i++)n===e&&(n=0),i===e&&(i=0),P[t]=C(E[t],E[n],E[i]);const I=[];let D,N=P.concat();for(let t=0,e=S.length;t<e;t++){const e=S[t];D=[];for(let t=0,n=e.length,i=n-1,r=t+1;t<n;t++,i++,r++)i===n&&(i=0),r===n&&(r=0),D[t]=C(e[t],e[i],e[r]);I.push(D),N=N.concat(D)}for(let t=0;t<p;t++){const e=t/p,n=h*Math.cos(e*Math.PI/2),i=u*Math.sin(e*Math.PI/2)+d;for(let t=0,e=E.length;t<e;t++){const e=A(E[t],P[t],i);F(e.x,e.y,-n)}for(let t=0,e=S.length;t<e;t++){const e=S[t];D=I[t];for(let t=0,r=e.length;t<r;t++){const r=A(e[t],D[t],i);F(r.x,r.y,-n)}}}const z=u+d;for(let t=0;t<L;t++){const e=c?A(w[t],N[t],z):w[t];M?(x.copy(v.normals[0]).multiplyScalar(e.x),y.copy(v.binormals[0]).multiplyScalar(e.y),_.copy(g[0]).add(x).add(y),F(_.x,_.y,_.z)):F(e.x,e.y,0)}for(let t=1;t<=o;t++)for(let e=0;e<L;e++){const n=c?A(w[e],N[e],z):w[e];M?(x.copy(v.normals[t]).multiplyScalar(n.x),y.copy(v.binormals[t]).multiplyScalar(n.y),_.copy(g[t]).add(x).add(y),F(_.x,_.y,_.z)):F(n.x,n.y,l/o*t)}for(let t=p-1;t>=0;t--){const e=t/p,n=h*Math.cos(e*Math.PI/2),i=u*Math.sin(e*Math.PI/2)+d;for(let t=0,e=E.length;t<e;t++){const e=A(E[t],P[t],i);F(e.x,e.y,l+n)}for(let t=0,e=S.length;t<e;t++){const e=S[t];D=I[t];for(let t=0,r=e.length;t<r;t++){const r=A(e[t],D[t],i);M?F(r.x,r.y+g[o-1].y,g[o-1].x+n):F(r.x,r.y,l+n)}}}function B(t,e){let n=t.length;for(;--n>=0;){const i=n;let r=n-1;r<0&&(r=t.length-1);for(let t=0,n=o+2*p;t<n;t++){const n=L*t,s=L*(t+1);U(e+i+n,e+r+n,e+r+s,e+i+s)}}}function F(t,e,n){s.push(t),s.push(e),s.push(n)}function O(t,e,r){H(t),H(e),H(r);const s=i.length/3,a=f.generateTopUV(n,i,s-3,s-2,s-1);G(a[0]),G(a[1]),G(a[2])}function U(t,e,r,s){H(t),H(e),H(s),H(e),H(r),H(s);const a=i.length/3,o=f.generateSideWallUV(n,i,a-6,a-3,a-2,a-1);G(o[0]),G(o[1]),G(o[3]),G(o[1]),G(o[2]),G(o[3])}function H(t){i.push(s[3*t+0]),i.push(s[3*t+1]),i.push(s[3*t+2])}function G(t){r.push(t.x),r.push(t.y)}!function(){const t=i.length/3;if(c){let t=0,e=L*t;for(let t=0;t<R;t++){const n=T[t];O(n[2]+e,n[1]+e,n[0]+e)}t=o+2*p,e=L*t;for(let t=0;t<R;t++){const n=T[t];O(n[0]+e,n[1]+e,n[2]+e)}}else{for(let t=0;t<R;t++){const e=T[t];O(e[2],e[1],e[0])}for(let t=0;t<R;t++){const e=T[t];O(e[0]+L*o,e[1]+L*o,e[2]+L*o)}}n.addGroup(t,i.length/3-t,0)}(),function(){const t=i.length/3;let e=0;B(E,e),e+=E.length;for(let t=0,n=S.length;t<n;t++){const n=S[t];B(n,e),e+=n.length}n.addGroup(t,i.length/3-t,1)}()}this.setAttribute("position",new vn(i,3)),this.setAttribute("uv",new vn(r,2)),this.computeVertexNormals()}toJSON(){const t=super.toJSON();return function(t,e,n){if(n.shapes=[],Array.isArray(t))for(let e=0,i=t.length;e<i;e++){const i=t[e];n.shapes.push(i.uuid)}else n.shapes.push(t.uuid);void 0!==e.extrudePath&&(n.options.extrudePath=e.extrudePath.toJSON());return n}(this.parameters.shapes,this.parameters.options,t)}static fromJSON(t,e){const n=[];for(let i=0,r=t.shapes.length;i<r;i++){const r=e[t.shapes[i]];n.push(r)}const i=t.options.extrudePath;return void 0!==i&&(t.options.extrudePath=(new Uo[i.type]).fromJSON(i)),new vl(n,t.options)}}const yl={generateTopUV:function(t,e,n,i,r){const s=e[3*n],a=e[3*n+1],o=e[3*i],l=e[3*i+1],c=e[3*r],h=e[3*r+1];return[new yt(s,a),new yt(o,l),new yt(c,h)]},generateSideWallUV:function(t,e,n,i,r,s){const a=e[3*n],o=e[3*n+1],l=e[3*n+2],c=e[3*i],h=e[3*i+1],u=e[3*i+2],d=e[3*r],p=e[3*r+1],m=e[3*r+2],f=e[3*s],g=e[3*s+1],v=e[3*s+2];return Math.abs(o-h)<Math.abs(a-c)?[new yt(a,1-l),new yt(c,1-u),new yt(d,1-m),new yt(f,1-v)]:[new yt(o,1-l),new yt(h,1-u),new yt(p,1-m),new yt(g,1-v)]}};class xl extends po{constructor(t=1,e=0){const n=(1+Math.sqrt(5))/2;super([-1,n,0,1,n,0,-1,-n,0,1,-n,0,0,-1,n,0,1,n,0,-1,-n,0,1,-n,n,0,-1,n,0,1,-n,0,-1,-n,0,1],[0,11,5,0,5,1,0,1,7,0,7,10,0,10,11,1,5,9,5,11,4,11,10,2,10,7,6,7,1,8,3,9,4,3,4,2,3,2,6,3,6,8,3,8,9,4,9,5,2,4,11,6,2,10,8,6,7,9,8,1],t,e),this.type="IcosahedronGeometry",this.parameters={radius:t,detail:e}}static fromJSON(t){return new xl(t.radius,t.detail)}}class _l extends En{constructor(t=[new yt(0,.5),new yt(.5,0),new yt(0,-.5)],e=12,n=0,i=2*Math.PI){super(),this.type="LatheGeometry",this.parameters={points:t,segments:e,phiStart:n,phiLength:i},e=Math.floor(e),i=ut(i,0,2*Math.PI);const r=[],s=[],a=[],o=1/e,l=new zt,c=new yt;for(let r=0;r<=e;r++){const h=n+r*o*i,u=Math.sin(h),d=Math.cos(h);for(let n=0;n<=t.length-1;n++)l.x=t[n].x*u,l.y=t[n].y,l.z=t[n].x*d,s.push(l.x,l.y,l.z),c.x=r/e,c.y=n/(t.length-1),a.push(c.x,c.y)}for(let n=0;n<e;n++)for(let e=0;e<t.length-1;e++){const i=e+n*t.length,s=i,a=i+t.length,o=i+t.length+1,l=i+1;r.push(s,a,l),r.push(a,o,l)}if(this.setIndex(r),this.setAttribute("position",new vn(s,3)),this.setAttribute("uv",new vn(a,2)),this.computeVertexNormals(),i===2*Math.PI){const n=this.attributes.normal.array,i=new zt,r=new zt,s=new zt,a=e*t.length*3;for(let e=0,o=0;e<t.length;e++,o+=3)i.x=n[o+0],i.y=n[o+1],i.z=n[o+2],r.x=n[a+o+0],r.y=n[a+o+1],r.z=n[a+o+2],s.addVectors(i,r).normalize(),n[o+0]=n[a+o+0]=s.x,n[o+1]=n[a+o+1]=s.y,n[o+2]=n[a+o+2]=s.z}}static fromJSON(t){return new _l(t.points,t.segments,t.phiStart,t.phiLength)}}class Ml extends po{constructor(t=1,e=0){super([1,0,0,-1,0,0,0,1,0,0,-1,0,0,0,1,0,0,-1],[0,2,4,0,4,3,0,3,5,0,5,2,1,2,5,1,5,3,1,3,4,1,4,2],t,e),this.type="OctahedronGeometry",this.parameters={radius:t,detail:e}}static fromJSON(t){return new Ml(t.radius,t.detail)}}class bl extends En{constructor(t=.5,e=1,n=8,i=1,r=0,s=2*Math.PI){super(),this.type="RingGeometry",this.parameters={innerRadius:t,outerRadius:e,thetaSegments:n,phiSegments:i,thetaStart:r,thetaLength:s},n=Math.max(3,n);const a=[],o=[],l=[],c=[];let h=t;const u=(e-t)/(i=Math.max(1,i)),d=new zt,p=new yt;for(let t=0;t<=i;t++){for(let t=0;t<=n;t++){const i=r+t/n*s;d.x=h*Math.cos(i),d.y=h*Math.sin(i),o.push(d.x,d.y,d.z),l.push(0,0,1),p.x=(d.x/e+1)/2,p.y=(d.y/e+1)/2,c.push(p.x,p.y)}h+=u}for(let t=0;t<i;t++){const e=t*(n+1);for(let t=0;t<n;t++){const i=t+e,r=i,s=i+n+1,o=i+n+2,l=i+1;a.push(r,s,l),a.push(s,o,l)}}this.setIndex(a),this.setAttribute("position",new vn(o,3)),this.setAttribute("normal",new vn(l,3)),this.setAttribute("uv",new vn(c,2))}static fromJSON(t){return new bl(t.innerRadius,t.outerRadius,t.thetaSegments,t.phiSegments,t.thetaStart,t.thetaLength)}}class wl extends En{constructor(t=new ko([new yt(0,.5),new yt(-.5,-.5),new yt(.5,-.5)]),e=12){super(),this.type="ShapeGeometry",this.parameters={shapes:t,curveSegments:e};const n=[],i=[],r=[],s=[];let a=0,o=0;if(!1===Array.isArray(t))l(t);else for(let e=0;e<t.length;e++)l(t[e]),this.addGroup(a,o,e),a+=o,o=0;function l(t){const a=i.length/3,l=t.extractPoints(e);let c=l.shape;const h=l.holes;!1===ml.isClockWise(c)&&(c=c.reverse());for(let t=0,e=h.length;t<e;t++){const e=h[t];!0===ml.isClockWise(e)&&(h[t]=e.reverse())}const u=ml.triangulateShape(c,h);for(let t=0,e=h.length;t<e;t++){const e=h[t];c=c.concat(e)}for(let t=0,e=c.length;t<e;t++){const e=c[t];i.push(e.x,e.y,0),r.push(0,0,1),s.push(e.x,e.y)}for(let t=0,e=u.length;t<e;t++){const e=u[t],i=e[0]+a,r=e[1]+a,s=e[2]+a;n.push(i,r,s),o+=3}}this.setIndex(n),this.setAttribute("position",new vn(i,3)),this.setAttribute("normal",new vn(r,3)),this.setAttribute("uv",new vn(s,2))}toJSON(){const t=super.toJSON();return function(t,e){if(e.shapes=[],Array.isArray(t))for(let n=0,i=t.length;n<i;n++){const i=t[n];e.shapes.push(i.uuid)}else e.shapes.push(t.uuid);return e}(this.parameters.shapes,t)}static fromJSON(t,e){const n=[];for(let i=0,r=t.shapes.length;i<r;i++){const r=e[t.shapes[i]];n.push(r)}return new wl(n,t.curveSegments)}}class Sl extends En{constructor(t=1,e=32,n=16,i=0,r=2*Math.PI,s=0,a=Math.PI){super(),this.type="SphereGeometry",this.parameters={radius:t,widthSegments:e,heightSegments:n,phiStart:i,phiLength:r,thetaStart:s,thetaLength:a},e=Math.max(3,Math.floor(e)),n=Math.max(2,Math.floor(n));const o=Math.min(s+a,Math.PI);let l=0;const c=[],h=new zt,u=new zt,d=[],p=[],m=[],f=[];for(let d=0;d<=n;d++){const g=[],v=d/n;let y=0;0==d&&0==s?y=.5/e:d==n&&o==Math.PI&&(y=-.5/e);for(let n=0;n<=e;n++){const o=n/e;h.x=-t*Math.cos(i+o*r)*Math.sin(s+v*a),h.y=t*Math.cos(s+v*a),h.z=t*Math.sin(i+o*r)*Math.sin(s+v*a),p.push(h.x,h.y,h.z),u.copy(h).normalize(),m.push(u.x,u.y,u.z),f.push(o+y,1-v),g.push(l++)}c.push(g)}for(let t=0;t<n;t++)for(let i=0;i<e;i++){const e=c[t][i+1],r=c[t][i],a=c[t+1][i],l=c[t+1][i+1];(0!==t||s>0)&&d.push(e,r,l),(t!==n-1||o<Math.PI)&&d.push(r,a,l)}this.setIndex(d),this.setAttribute("position",new vn(p,3)),this.setAttribute("normal",new vn(m,3)),this.setAttribute("uv",new vn(f,2))}static fromJSON(t){return new Sl(t.radius,t.widthSegments,t.heightSegments,t.phiStart,t.phiLength,t.thetaStart,t.thetaLength)}}class Tl extends po{constructor(t=1,e=0){super([1,1,1,-1,-1,1,-1,1,-1,1,-1,-1],[2,1,0,0,3,2,1,3,0,2,3,1],t,e),this.type="TetrahedronGeometry",this.parameters={radius:t,detail:e}}static fromJSON(t){return new Tl(t.radius,t.detail)}}class El extends En{constructor(t=1,e=.4,n=8,i=6,r=2*Math.PI){super(),this.type="TorusGeometry",this.parameters={radius:t,tube:e,radialSegments:n,tubularSegments:i,arc:r},n=Math.floor(n),i=Math.floor(i);const s=[],a=[],o=[],l=[],c=new zt,h=new zt,u=new zt;for(let s=0;s<=n;s++)for(let d=0;d<=i;d++){const p=d/i*r,m=s/n*Math.PI*2;h.x=(t+e*Math.cos(m))*Math.cos(p),h.y=(t+e*Math.cos(m))*Math.sin(p),h.z=e*Math.sin(m),a.push(h.x,h.y,h.z),c.x=t*Math.cos(p),c.y=t*Math.sin(p),u.subVectors(h,c).normalize(),o.push(u.x,u.y,u.z),l.push(d/i),l.push(s/n)}for(let t=1;t<=n;t++)for(let e=1;e<=i;e++){const n=(i+1)*t+e-1,r=(i+1)*(t-1)+e-1,a=(i+1)*(t-1)+e,o=(i+1)*t+e;s.push(n,r,o),s.push(r,a,o)}this.setIndex(s),this.setAttribute("position",new vn(a,3)),this.setAttribute("normal",new vn(o,3)),this.setAttribute("uv",new vn(l,2))}static fromJSON(t){return new El(t.radius,t.tube,t.radialSegments,t.tubularSegments,t.arc)}}class Al extends En{constructor(t=1,e=.4,n=64,i=8,r=2,s=3){super(),this.type="TorusKnotGeometry",this.parameters={radius:t,tube:e,tubularSegments:n,radialSegments:i,p:r,q:s},n=Math.floor(n),i=Math.floor(i);const a=[],o=[],l=[],c=[],h=new zt,u=new zt,d=new zt,p=new zt,m=new zt,f=new zt,g=new zt;for(let a=0;a<=n;++a){const y=a/n*r*Math.PI*2;v(y,r,s,t,d),v(y+.01,r,s,t,p),f.subVectors(p,d),g.addVectors(p,d),m.crossVectors(f,g),g.crossVectors(m,f),m.normalize(),g.normalize();for(let t=0;t<=i;++t){const r=t/i*Math.PI*2,s=-e*Math.cos(r),p=e*Math.sin(r);h.x=d.x+(s*g.x+p*m.x),h.y=d.y+(s*g.y+p*m.y),h.z=d.z+(s*g.z+p*m.z),o.push(h.x,h.y,h.z),u.subVectors(h,d).normalize(),l.push(u.x,u.y,u.z),c.push(a/n),c.push(t/i)}}for(let t=1;t<=n;t++)for(let e=1;e<=i;e++){const n=(i+1)*(t-1)+(e-1),r=(i+1)*t+(e-1),s=(i+1)*t+e,o=(i+1)*(t-1)+e;a.push(n,r,o),a.push(r,s,o)}function v(t,e,n,i,r){const s=Math.cos(t),a=Math.sin(t),o=n/e*t,l=Math.cos(o);r.x=i*(2+l)*.5*s,r.y=i*(2+l)*a*.5,r.z=i*Math.sin(o)*.5}this.setIndex(a),this.setAttribute("position",new vn(o,3)),this.setAttribute("normal",new vn(l,3)),this.setAttribute("uv",new vn(c,2))}static fromJSON(t){return new Al(t.radius,t.tube,t.tubularSegments,t.radialSegments,t.p,t.q)}}class Ll extends En{constructor(t=new Fo(new zt(-1,-1,0),new zt(-1,1,0),new zt(1,1,0)),e=64,n=1,i=8,r=!1){super(),this.type="TubeGeometry",this.parameters={path:t,tubularSegments:e,radius:n,radialSegments:i,closed:r};const s=t.computeFrenetFrames(e,r);this.tangents=s.tangents,this.normals=s.normals,this.binormals=s.binormals;const a=new zt,o=new zt,l=new yt;let c=new zt;const h=[],u=[],d=[],p=[];function m(r){c=t.getPointAt(r/e,c);const l=s.normals[r],d=s.binormals[r];for(let t=0;t<=i;t++){const e=t/i*Math.PI*2,r=Math.sin(e),s=-Math.cos(e);o.x=s*l.x+r*d.x,o.y=s*l.y+r*d.y,o.z=s*l.z+r*d.z,o.normalize(),u.push(o.x,o.y,o.z),a.x=c.x+n*o.x,a.y=c.y+n*o.y,a.z=c.z+n*o.z,h.push(a.x,a.y,a.z)}}!function(){for(let t=0;t<e;t++)m(t);m(!1===r?e:0),function(){for(let t=0;t<=e;t++)for(let n=0;n<=i;n++)l.x=t/e,l.y=n/i,d.push(l.x,l.y)}(),function(){for(let t=1;t<=e;t++)for(let e=1;e<=i;e++){const n=(i+1)*(t-1)+(e-1),r=(i+1)*t+(e-1),s=(i+1)*t+e,a=(i+1)*(t-1)+e;p.push(n,r,a),p.push(r,s,a)}}()}(),this.setIndex(p),this.setAttribute("position",new vn(h,3)),this.setAttribute("normal",new vn(u,3)),this.setAttribute("uv",new vn(d,2))}toJSON(){const t=super.toJSON();return t.path=this.parameters.path.toJSON(),t}static fromJSON(t){return new Ll((new Uo[t.path.type]).fromJSON(t.path),t.tubularSegments,t.radius,t.radialSegments,t.closed)}}class Rl extends En{constructor(t=null){if(super(),this.type="WireframeGeometry",this.parameters={geometry:t},null!==t){const e=[],n=new Set,i=new zt,r=new zt;if(null!==t.index){const s=t.attributes.position,a=t.index;let o=t.groups;0===o.length&&(o=[{start:0,count:a.count,materialIndex:0}]);for(let t=0,l=o.length;t<l;++t){const l=o[t],c=l.start;for(let t=c,o=c+l.count;t<o;t+=3)for(let o=0;o<3;o++){const l=a.getX(t+o),c=a.getX(t+(o+1)%3);i.fromBufferAttribute(s,l),r.fromBufferAttribute(s,c),!0===Cl(i,r,n)&&(e.push(i.x,i.y,i.z),e.push(r.x,r.y,r.z))}}}else{const s=t.attributes.position;for(let t=0,a=s.count/3;t<a;t++)for(let a=0;a<3;a++){const o=3*t+a,l=3*t+(a+1)%3;i.fromBufferAttribute(s,o),r.fromBufferAttribute(s,l),!0===Cl(i,r,n)&&(e.push(i.x,i.y,i.z),e.push(r.x,r.y,r.z))}}this.setAttribute("position",new vn(e,3))}}}function Cl(t,e,n){const i=`${t.x},${t.y},${t.z}-${e.x},${e.y},${e.z}`,r=`${e.x},${e.y},${e.z}-${t.x},${t.y},${t.z}`;return!0!==n.has(i)&&!0!==n.has(r)&&(n.add(i,r),!0)}var Pl=Object.freeze({__proto__:null,BoxGeometry:qn,BoxBufferGeometry:qn,CircleGeometry:co,CircleBufferGeometry:co,ConeGeometry:uo,ConeBufferGeometry:uo,CylinderGeometry:ho,CylinderBufferGeometry:ho,DodecahedronGeometry:mo,DodecahedronBufferGeometry:mo,EdgesGeometry:xo,ExtrudeGeometry:vl,ExtrudeBufferGeometry:vl,IcosahedronGeometry:xl,IcosahedronBufferGeometry:xl,LatheGeometry:_l,LatheBufferGeometry:_l,OctahedronGeometry:Ml,OctahedronBufferGeometry:Ml,PlaneGeometry:di,PlaneBufferGeometry:di,PolyhedronGeometry:po,PolyhedronBufferGeometry:po,RingGeometry:bl,RingBufferGeometry:bl,ShapeGeometry:wl,ShapeBufferGeometry:wl,SphereGeometry:Sl,SphereBufferGeometry:Sl,TetrahedronGeometry:Tl,TetrahedronBufferGeometry:Tl,TorusGeometry:El,TorusBufferGeometry:El,TorusKnotGeometry:Al,TorusKnotBufferGeometry:Al,TubeGeometry:Ll,TubeBufferGeometry:Ll,WireframeGeometry:Rl});class Il extends Ze{constructor(t){super(),this.type="ShadowMaterial",this.color=new rn(0),this.transparent=!0,this.setValues(t)}copy(t){return super.copy(t),this.color.copy(t.color),this}}Il.prototype.isShadowMaterial=!0;class Dl extends Ze{constructor(t){super(),this.defines={STANDARD:""},this.type="MeshStandardMaterial",this.color=new rn(16777215),this.roughness=1,this.metalness=0,this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.emissive=new rn(0),this.emissiveIntensity=1,this.emissiveMap=null,this.bumpMap=null,this.bumpScale=1,this.normalMap=null,this.normalMapType=0,this.normalScale=new yt(1,1),this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.roughnessMap=null,this.metalnessMap=null,this.alphaMap=null,this.envMap=null,this.envMapIntensity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.flatShading=!1,this.setValues(t)}copy(t){return super.copy(t),this.defines={STANDARD:""},this.color.copy(t.color),this.roughness=t.roughness,this.metalness=t.metalness,this.map=t.map,this.lightMap=t.lightMap,this.lightMapIntensity=t.lightMapIntensity,this.aoMap=t.aoMap,this.aoMapIntensity=t.aoMapIntensity,this.emissive.copy(t.emissive),this.emissiveMap=t.emissiveMap,this.emissiveIntensity=t.emissiveIntensity,this.bumpMap=t.bumpMap,this.bumpScale=t.bumpScale,this.normalMap=t.normalMap,this.normalMapType=t.normalMapType,this.normalScale.copy(t.normalScale),this.displacementMap=t.displacementMap,this.displacementScale=t.displacementScale,this.displacementBias=t.displacementBias,this.roughnessMap=t.roughnessMap,this.metalnessMap=t.metalnessMap,this.alphaMap=t.alphaMap,this.envMap=t.envMap,this.envMapIntensity=t.envMapIntensity,this.refractionRatio=t.refractionRatio,this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this.wireframeLinecap=t.wireframeLinecap,this.wireframeLinejoin=t.wireframeLinejoin,this.flatShading=t.flatShading,this}}Dl.prototype.isMeshStandardMaterial=!0;class Nl extends Dl{constructor(t){super(),this.defines={STANDARD:"",PHYSICAL:""},this.type="MeshPhysicalMaterial",this.clearcoatMap=null,this.clearcoatRoughness=0,this.clearcoatRoughnessMap=null,this.clearcoatNormalScale=new yt(1,1),this.clearcoatNormalMap=null,this.ior=1.5,Object.defineProperty(this,"reflectivity",{get:function(){return ut(2.5*(this.ior-1)/(this.ior+1),0,1)},set:function(t){this.ior=(1+.4*t)/(1-.4*t)}}),this.sheenColor=new rn(0),this.sheenColorMap=null,this.sheenRoughness=1,this.sheenRoughnessMap=null,this.transmissionMap=null,this.thickness=.01,this.thicknessMap=null,this.attenuationDistance=0,this.attenuationColor=new rn(1,1,1),this.specularIntensity=1,this.specularIntensityMap=null,this.specularColor=new rn(1,1,1),this.specularColorMap=null,this._sheen=0,this._clearcoat=0,this._transmission=0,this.setValues(t)}get sheen(){return this._sheen}set sheen(t){this._sheen>0!=t>0&&this.version++,this._sheen=t}get clearcoat(){return this._clearcoat}set clearcoat(t){this._clearcoat>0!=t>0&&this.version++,this._clearcoat=t}get transmission(){return this._transmission}set transmission(t){this._transmission>0!=t>0&&this.version++,this._transmission=t}copy(t){return super.copy(t),this.defines={STANDARD:"",PHYSICAL:""},this.clearcoat=t.clearcoat,this.clearcoatMap=t.clearcoatMap,this.clearcoatRoughness=t.clearcoatRoughness,this.clearcoatRoughnessMap=t.clearcoatRoughnessMap,this.clearcoatNormalMap=t.clearcoatNormalMap,this.clearcoatNormalScale.copy(t.clearcoatNormalScale),this.ior=t.ior,this.sheen=t.sheen,this.sheenColor.copy(t.sheenColor),this.sheenColorMap=t.sheenColorMap,this.sheenRoughness=t.sheenRoughness,this.sheenRoughnessMap=t.sheenRoughnessMap,this.transmission=t.transmission,this.transmissionMap=t.transmissionMap,this.thickness=t.thickness,this.thicknessMap=t.thicknessMap,this.attenuationDistance=t.attenuationDistance,this.attenuationColor.copy(t.attenuationColor),this.specularIntensity=t.specularIntensity,this.specularIntensityMap=t.specularIntensityMap,this.specularColor.copy(t.specularColor),this.specularColorMap=t.specularColorMap,this}}Nl.prototype.isMeshPhysicalMaterial=!0;class zl extends Ze{constructor(t){super(),this.type="MeshPhongMaterial",this.color=new rn(16777215),this.specular=new rn(1118481),this.shininess=30,this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.emissive=new rn(0),this.emissiveIntensity=1,this.emissiveMap=null,this.bumpMap=null,this.bumpScale=1,this.normalMap=null,this.normalMapType=0,this.normalScale=new yt(1,1),this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.combine=0,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.flatShading=!1,this.setValues(t)}copy(t){return super.copy(t),this.color.copy(t.color),this.specular.copy(t.specular),this.shininess=t.shininess,this.map=t.map,this.lightMap=t.lightMap,this.lightMapIntensity=t.lightMapIntensity,this.aoMap=t.aoMap,this.aoMapIntensity=t.aoMapIntensity,this.emissive.copy(t.emissive),this.emissiveMap=t.emissiveMap,this.emissiveIntensity=t.emissiveIntensity,this.bumpMap=t.bumpMap,this.bumpScale=t.bumpScale,this.normalMap=t.normalMap,this.normalMapType=t.normalMapType,this.normalScale.copy(t.normalScale),this.displacementMap=t.displacementMap,this.displacementScale=t.displacementScale,this.displacementBias=t.displacementBias,this.specularMap=t.specularMap,this.alphaMap=t.alphaMap,this.envMap=t.envMap,this.combine=t.combine,this.reflectivity=t.reflectivity,this.refractionRatio=t.refractionRatio,this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this.wireframeLinecap=t.wireframeLinecap,this.wireframeLinejoin=t.wireframeLinejoin,this.flatShading=t.flatShading,this}}zl.prototype.isMeshPhongMaterial=!0;class Bl extends Ze{constructor(t){super(),this.defines={TOON:""},this.type="MeshToonMaterial",this.color=new rn(16777215),this.map=null,this.gradientMap=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.emissive=new rn(0),this.emissiveIntensity=1,this.emissiveMap=null,this.bumpMap=null,this.bumpScale=1,this.normalMap=null,this.normalMapType=0,this.normalScale=new yt(1,1),this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.alphaMap=null,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.setValues(t)}copy(t){return super.copy(t),this.color.copy(t.color),this.map=t.map,this.gradientMap=t.gradientMap,this.lightMap=t.lightMap,this.lightMapIntensity=t.lightMapIntensity,this.aoMap=t.aoMap,this.aoMapIntensity=t.aoMapIntensity,this.emissive.copy(t.emissive),this.emissiveMap=t.emissiveMap,this.emissiveIntensity=t.emissiveIntensity,this.bumpMap=t.bumpMap,this.bumpScale=t.bumpScale,this.normalMap=t.normalMap,this.normalMapType=t.normalMapType,this.normalScale.copy(t.normalScale),this.displacementMap=t.displacementMap,this.displacementScale=t.displacementScale,this.displacementBias=t.displacementBias,this.alphaMap=t.alphaMap,this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this.wireframeLinecap=t.wireframeLinecap,this.wireframeLinejoin=t.wireframeLinejoin,this}}Bl.prototype.isMeshToonMaterial=!0;class Fl extends Ze{constructor(t){super(),this.type="MeshNormalMaterial",this.bumpMap=null,this.bumpScale=1,this.normalMap=null,this.normalMapType=0,this.normalScale=new yt(1,1),this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.fog=!1,this.flatShading=!1,this.setValues(t)}copy(t){return super.copy(t),this.bumpMap=t.bumpMap,this.bumpScale=t.bumpScale,this.normalMap=t.normalMap,this.normalMapType=t.normalMapType,this.normalScale.copy(t.normalScale),this.displacementMap=t.displacementMap,this.displacementScale=t.displacementScale,this.displacementBias=t.displacementBias,this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this.flatShading=t.flatShading,this}}Fl.prototype.isMeshNormalMaterial=!0;class Ol extends Ze{constructor(t){super(),this.type="MeshLambertMaterial",this.color=new rn(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.emissive=new rn(0),this.emissiveIntensity=1,this.emissiveMap=null,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.combine=0,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.setValues(t)}copy(t){return super.copy(t),this.color.copy(t.color),this.map=t.map,this.lightMap=t.lightMap,this.lightMapIntensity=t.lightMapIntensity,this.aoMap=t.aoMap,this.aoMapIntensity=t.aoMapIntensity,this.emissive.copy(t.emissive),this.emissiveMap=t.emissiveMap,this.emissiveIntensity=t.emissiveIntensity,this.specularMap=t.specularMap,this.alphaMap=t.alphaMap,this.envMap=t.envMap,this.combine=t.combine,this.reflectivity=t.reflectivity,this.refractionRatio=t.refractionRatio,this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this.wireframeLinecap=t.wireframeLinecap,this.wireframeLinejoin=t.wireframeLinejoin,this}}Ol.prototype.isMeshLambertMaterial=!0;class Ul extends Ze{constructor(t){super(),this.defines={MATCAP:""},this.type="MeshMatcapMaterial",this.color=new rn(16777215),this.matcap=null,this.map=null,this.bumpMap=null,this.bumpScale=1,this.normalMap=null,this.normalMapType=0,this.normalScale=new yt(1,1),this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.alphaMap=null,this.flatShading=!1,this.setValues(t)}copy(t){return super.copy(t),this.defines={MATCAP:""},this.color.copy(t.color),this.matcap=t.matcap,this.map=t.map,this.bumpMap=t.bumpMap,this.bumpScale=t.bumpScale,this.normalMap=t.normalMap,this.normalMapType=t.normalMapType,this.normalScale.copy(t.normalScale),this.displacementMap=t.displacementMap,this.displacementScale=t.displacementScale,this.displacementBias=t.displacementBias,this.alphaMap=t.alphaMap,this.flatShading=t.flatShading,this}}Ul.prototype.isMeshMatcapMaterial=!0;class Hl extends Ga{constructor(t){super(),this.type="LineDashedMaterial",this.scale=1,this.dashSize=3,this.gapSize=1,this.setValues(t)}copy(t){return super.copy(t),this.scale=t.scale,this.dashSize=t.dashSize,this.gapSize=t.gapSize,this}}Hl.prototype.isLineDashedMaterial=!0;var Gl=Object.freeze({__proto__:null,ShadowMaterial:Il,SpriteMaterial:sa,RawShaderMaterial:wi,ShaderMaterial:Zn,PointsMaterial:Ka,MeshPhysicalMaterial:Nl,MeshStandardMaterial:Dl,MeshPhongMaterial:zl,MeshToonMaterial:Bl,MeshNormalMaterial:Fl,MeshLambertMaterial:Ol,MeshDepthMaterial:Us,MeshDistanceMaterial:Hs,MeshBasicMaterial:sn,MeshMatcapMaterial:Ul,LineDashedMaterial:Hl,LineBasicMaterial:Ga,Material:Ze});const kl={arraySlice:function(t,e,n){return kl.isTypedArray(t)?new t.constructor(t.subarray(e,void 0!==n?n:t.length)):t.slice(e,n)},convertArray:function(t,e,n){return!t||!n&&t.constructor===e?t:"number"==typeof e.BYTES_PER_ELEMENT?new e(t):Array.prototype.slice.call(t)},isTypedArray:function(t){return ArrayBuffer.isView(t)&&!(t instanceof DataView)},getKeyframeOrder:function(t){const e=t.length,n=new Array(e);for(let t=0;t!==e;++t)n[t]=t;return n.sort((function(e,n){return t[e]-t[n]})),n},sortedArray:function(t,e,n){const i=t.length,r=new t.constructor(i);for(let s=0,a=0;a!==i;++s){const i=n[s]*e;for(let n=0;n!==e;++n)r[a++]=t[i+n]}return r},flattenJSON:function(t,e,n,i){let r=1,s=t[0];for(;void 0!==s&&void 0===s[i];)s=t[r++];if(void 0===s)return;let a=s[i];if(void 0!==a)if(Array.isArray(a))do{a=s[i],void 0!==a&&(e.push(s.time),n.push.apply(n,a)),s=t[r++]}while(void 0!==s);else if(void 0!==a.toArray)do{a=s[i],void 0!==a&&(e.push(s.time),a.toArray(n,n.length)),s=t[r++]}while(void 0!==s);else do{a=s[i],void 0!==a&&(e.push(s.time),n.push(a)),s=t[r++]}while(void 0!==s)},subclip:function(t,e,n,i,r=30){const s=t.clone();s.name=e;const a=[];for(let t=0;t<s.tracks.length;++t){const e=s.tracks[t],o=e.getValueSize(),l=[],c=[];for(let t=0;t<e.times.length;++t){const s=e.times[t]*r;if(!(s<n||s>=i)){l.push(e.times[t]);for(let n=0;n<o;++n)c.push(e.values[t*o+n])}}0!==l.length&&(e.times=kl.convertArray(l,e.times.constructor),e.values=kl.convertArray(c,e.values.constructor),a.push(e))}s.tracks=a;let o=1/0;for(let t=0;t<s.tracks.length;++t)o>s.tracks[t].times[0]&&(o=s.tracks[t].times[0]);for(let t=0;t<s.tracks.length;++t)s.tracks[t].shift(-1*o);return s.resetDuration(),s},makeClipAdditive:function(t,e=0,n=t,i=30){i<=0&&(i=30);const r=n.tracks.length,s=e/i;for(let e=0;e<r;++e){const i=n.tracks[e],r=i.ValueTypeName;if("bool"===r||"string"===r)continue;const a=t.tracks.find((function(t){return t.name===i.name&&t.ValueTypeName===r}));if(void 0===a)continue;let o=0;const l=i.getValueSize();i.createInterpolant.isInterpolantFactoryMethodGLTFCubicSpline&&(o=l/3);let c=0;const h=a.getValueSize();a.createInterpolant.isInterpolantFactoryMethodGLTFCubicSpline&&(c=h/3);const u=i.times.length-1;let d;if(s<=i.times[0]){const t=o,e=l-o;d=kl.arraySlice(i.values,t,e)}else if(s>=i.times[u]){const t=u*l+o,e=t+l-o;d=kl.arraySlice(i.values,t,e)}else{const t=i.createInterpolant(),e=o,n=l-o;t.evaluate(s),d=kl.arraySlice(t.resultBuffer,e,n)}if("quaternion"===r){(new Nt).fromArray(d).normalize().conjugate().toArray(d)}const p=a.times.length;for(let t=0;t<p;++t){const e=t*h+c;if("quaternion"===r)Nt.multiplyQuaternionsFlat(a.values,e,d,0,a.values,e);else{const t=h-2*c;for(let n=0;n<t;++n)a.values[e+n]-=d[n]}}}return t.blendMode=q,t}};class Vl{constructor(t,e,n,i){this.parameterPositions=t,this._cachedIndex=0,this.resultBuffer=void 0!==i?i:new e.constructor(n),this.sampleValues=e,this.valueSize=n,this.settings=null,this.DefaultSettings_={}}evaluate(t){const e=this.parameterPositions;let n=this._cachedIndex,i=e[n],r=e[n-1];t:{e:{let s;n:{i:if(!(t<i)){for(let s=n+2;;){if(void 0===i){if(t<r)break i;return n=e.length,this._cachedIndex=n,this.afterEnd_(n-1,t,r)}if(n===s)break;if(r=i,i=e[++n],t<i)break e}s=e.length;break n}if(t>=r)break t;{const a=e[1];t<a&&(n=2,r=a);for(let s=n-2;;){if(void 0===r)return this._cachedIndex=0,this.beforeStart_(0,t,i);if(n===s)break;if(i=r,r=e[--n-1],t>=r)break e}s=n,n=0}}for(;n<s;){const i=n+s>>>1;t<e[i]?s=i:n=i+1}if(i=e[n],r=e[n-1],void 0===r)return this._cachedIndex=0,this.beforeStart_(0,t,i);if(void 0===i)return n=e.length,this._cachedIndex=n,this.afterEnd_(n-1,r,t)}this._cachedIndex=n,this.intervalChanged_(n,r,i)}return this.interpolate_(n,r,t,i)}getSettings_(){return this.settings||this.DefaultSettings_}copySampleValue_(t){const e=this.resultBuffer,n=this.sampleValues,i=this.valueSize,r=t*i;for(let t=0;t!==i;++t)e[t]=n[r+t];return e}interpolate_(){throw new Error("call to abstract method")}intervalChanged_(){}}Vl.prototype.beforeStart_=Vl.prototype.copySampleValue_,Vl.prototype.afterEnd_=Vl.prototype.copySampleValue_;class Wl extends Vl{constructor(t,e,n,i){super(t,e,n,i),this._weightPrev=-0,this._offsetPrev=-0,this._weightNext=-0,this._offsetNext=-0,this.DefaultSettings_={endingStart:k,endingEnd:k}}intervalChanged_(t,e,n){const i=this.parameterPositions;let r=t-2,s=t+1,a=i[r],o=i[s];if(void 0===a)switch(this.getSettings_().endingStart){case V:r=t,a=2*e-n;break;case W:r=i.length-2,a=e+i[r]-i[r+1];break;default:r=t,a=n}if(void 0===o)switch(this.getSettings_().endingEnd){case V:s=t,o=2*n-e;break;case W:s=1,o=n+i[1]-i[0];break;default:s=t-1,o=e}const l=.5*(n-e),c=this.valueSize;this._weightPrev=l/(e-a),this._weightNext=l/(o-n),this._offsetPrev=r*c,this._offsetNext=s*c}interpolate_(t,e,n,i){const r=this.resultBuffer,s=this.sampleValues,a=this.valueSize,o=t*a,l=o-a,c=this._offsetPrev,h=this._offsetNext,u=this._weightPrev,d=this._weightNext,p=(n-e)/(i-e),m=p*p,f=m*p,g=-u*f+2*u*m-u*p,v=(1+u)*f+(-1.5-2*u)*m+(-.5+u)*p+1,y=(-1-d)*f+(1.5+d)*m+.5*p,x=d*f-d*m;for(let t=0;t!==a;++t)r[t]=g*s[c+t]+v*s[l+t]+y*s[o+t]+x*s[h+t];return r}}class jl extends Vl{constructor(t,e,n,i){super(t,e,n,i)}interpolate_(t,e,n,i){const r=this.resultBuffer,s=this.sampleValues,a=this.valueSize,o=t*a,l=o-a,c=(n-e)/(i-e),h=1-c;for(let t=0;t!==a;++t)r[t]=s[l+t]*h+s[o+t]*c;return r}}class ql extends Vl{constructor(t,e,n,i){super(t,e,n,i)}interpolate_(t){return this.copySampleValue_(t-1)}}class Xl{constructor(t,e,n,i){if(void 0===t)throw new Error("THREE.KeyframeTrack: track name is undefined");if(void 0===e||0===e.length)throw new Error("THREE.KeyframeTrack: no keyframes in track named "+t);this.name=t,this.times=kl.convertArray(e,this.TimeBufferType),this.values=kl.convertArray(n,this.ValueBufferType),this.setInterpolation(i||this.DefaultInterpolation)}static toJSON(t){const e=t.constructor;let n;if(e.toJSON!==this.toJSON)n=e.toJSON(t);else{n={name:t.name,times:kl.convertArray(t.times,Array),values:kl.convertArray(t.values,Array)};const e=t.getInterpolation();e!==t.DefaultInterpolation&&(n.interpolation=e)}return n.type=t.ValueTypeName,n}InterpolantFactoryMethodDiscrete(t){return new ql(this.times,this.values,this.getValueSize(),t)}InterpolantFactoryMethodLinear(t){return new jl(this.times,this.values,this.getValueSize(),t)}InterpolantFactoryMethodSmooth(t){return new Wl(this.times,this.values,this.getValueSize(),t)}setInterpolation(t){let e;switch(t){case U:e=this.InterpolantFactoryMethodDiscrete;break;case H:e=this.InterpolantFactoryMethodLinear;break;case G:e=this.InterpolantFactoryMethodSmooth}if(void 0===e){const e="unsupported interpolation for "+this.ValueTypeName+" keyframe track named "+this.name;if(void 0===this.createInterpolant){if(t===this.DefaultInterpolation)throw new Error(e);this.setInterpolation(this.DefaultInterpolation)}return console.warn("THREE.KeyframeTrack:",e),this}return this.createInterpolant=e,this}getInterpolation(){switch(this.createInterpolant){case this.InterpolantFactoryMethodDiscrete:return U;case this.InterpolantFactoryMethodLinear:return H;case this.InterpolantFactoryMethodSmooth:return G}}getValueSize(){return this.values.length/this.times.length}shift(t){if(0!==t){const e=this.times;for(let n=0,i=e.length;n!==i;++n)e[n]+=t}return this}scale(t){if(1!==t){const e=this.times;for(let n=0,i=e.length;n!==i;++n)e[n]*=t}return this}trim(t,e){const n=this.times,i=n.length;let r=0,s=i-1;for(;r!==i&&n[r]<t;)++r;for(;-1!==s&&n[s]>e;)--s;if(++s,0!==r||s!==i){r>=s&&(s=Math.max(s,1),r=s-1);const t=this.getValueSize();this.times=kl.arraySlice(n,r,s),this.values=kl.arraySlice(this.values,r*t,s*t)}return this}validate(){let t=!0;const e=this.getValueSize();e-Math.floor(e)!=0&&(console.error("THREE.KeyframeTrack: Invalid value size in track.",this),t=!1);const n=this.times,i=this.values,r=n.length;0===r&&(console.error("THREE.KeyframeTrack: Track is empty.",this),t=!1);let s=null;for(let e=0;e!==r;e++){const i=n[e];if("number"==typeof i&&isNaN(i)){console.error("THREE.KeyframeTrack: Time is not a valid number.",this,e,i),t=!1;break}if(null!==s&&s>i){console.error("THREE.KeyframeTrack: Out of order keys.",this,e,i,s),t=!1;break}s=i}if(void 0!==i&&kl.isTypedArray(i))for(let e=0,n=i.length;e!==n;++e){const n=i[e];if(isNaN(n)){console.error("THREE.KeyframeTrack: Value is not a valid number.",this,e,n),t=!1;break}}return t}optimize(){const t=kl.arraySlice(this.times),e=kl.arraySlice(this.values),n=this.getValueSize(),i=this.getInterpolation()===G,r=t.length-1;let s=1;for(let a=1;a<r;++a){let r=!1;const o=t[a];if(o!==t[a+1]&&(1!==a||o!==t[0]))if(i)r=!0;else{const t=a*n,i=t-n,s=t+n;for(let a=0;a!==n;++a){const n=e[t+a];if(n!==e[i+a]||n!==e[s+a]){r=!0;break}}}if(r){if(a!==s){t[s]=t[a];const i=a*n,r=s*n;for(let t=0;t!==n;++t)e[r+t]=e[i+t]}++s}}if(r>0){t[s]=t[r];for(let t=r*n,i=s*n,a=0;a!==n;++a)e[i+a]=e[t+a];++s}return s!==t.length?(this.times=kl.arraySlice(t,0,s),this.values=kl.arraySlice(e,0,s*n)):(this.times=t,this.values=e),this}clone(){const t=kl.arraySlice(this.times,0),e=kl.arraySlice(this.values,0),n=new(0,this.constructor)(this.name,t,e);return n.createInterpolant=this.createInterpolant,n}}Xl.prototype.TimeBufferType=Float32Array,Xl.prototype.ValueBufferType=Float32Array,Xl.prototype.DefaultInterpolation=H;class Yl extends Xl{}Yl.prototype.ValueTypeName="bool",Yl.prototype.ValueBufferType=Array,Yl.prototype.DefaultInterpolation=U,Yl.prototype.InterpolantFactoryMethodLinear=void 0,Yl.prototype.InterpolantFactoryMethodSmooth=void 0;class Jl extends Xl{}Jl.prototype.ValueTypeName="color";class Zl extends Xl{}Zl.prototype.ValueTypeName="number";class Ql extends Vl{constructor(t,e,n,i){super(t,e,n,i)}interpolate_(t,e,n,i){const r=this.resultBuffer,s=this.sampleValues,a=this.valueSize,o=(n-e)/(i-e);let l=t*a;for(let t=l+a;l!==t;l+=4)Nt.slerpFlat(r,0,s,l-a,s,l,o);return r}}class Kl extends Xl{InterpolantFactoryMethodLinear(t){return new Ql(this.times,this.values,this.getValueSize(),t)}}Kl.prototype.ValueTypeName="quaternion",Kl.prototype.DefaultInterpolation=H,Kl.prototype.InterpolantFactoryMethodSmooth=void 0;class $l extends Xl{}$l.prototype.ValueTypeName="string",$l.prototype.ValueBufferType=Array,$l.prototype.DefaultInterpolation=U,$l.prototype.InterpolantFactoryMethodLinear=void 0,$l.prototype.InterpolantFactoryMethodSmooth=void 0;class tc extends Xl{}tc.prototype.ValueTypeName="vector";class ec{constructor(t,e=-1,n,i=2500){this.name=t,this.tracks=n,this.duration=e,this.blendMode=i,this.uuid=ht(),this.duration<0&&this.resetDuration()}static parse(t){const e=[],n=t.tracks,i=1/(t.fps||1);for(let t=0,r=n.length;t!==r;++t)e.push(nc(n[t]).scale(i));const r=new this(t.name,t.duration,e,t.blendMode);return r.uuid=t.uuid,r}static toJSON(t){const e=[],n=t.tracks,i={name:t.name,duration:t.duration,tracks:e,uuid:t.uuid,blendMode:t.blendMode};for(let t=0,i=n.length;t!==i;++t)e.push(Xl.toJSON(n[t]));return i}static CreateFromMorphTargetSequence(t,e,n,i){const r=e.length,s=[];for(let t=0;t<r;t++){let a=[],o=[];a.push((t+r-1)%r,t,(t+1)%r),o.push(0,1,0);const l=kl.getKeyframeOrder(a);a=kl.sortedArray(a,1,l),o=kl.sortedArray(o,1,l),i||0!==a[0]||(a.push(r),o.push(o[0])),s.push(new Zl(".morphTargetInfluences["+e[t].name+"]",a,o).scale(1/n))}return new this(t,-1,s)}static findByName(t,e){let n=t;if(!Array.isArray(t)){const e=t;n=e.geometry&&e.geometry.animations||e.animations}for(let t=0;t<n.length;t++)if(n[t].name===e)return n[t];return null}static CreateClipsFromMorphTargetSequences(t,e,n){const i={},r=/^([\w-]*?)([\d]+)$/;for(let e=0,n=t.length;e<n;e++){const n=t[e],s=n.name.match(r);if(s&&s.length>1){const t=s[1];let e=i[t];e||(i[t]=e=[]),e.push(n)}}const s=[];for(const t in i)s.push(this.CreateFromMorphTargetSequence(t,i[t],e,n));return s}static parseAnimation(t,e){if(!t)return console.error("THREE.AnimationClip: No animation in JSONLoader data."),null;const n=function(t,e,n,i,r){if(0!==n.length){const s=[],a=[];kl.flattenJSON(n,s,a,i),0!==s.length&&r.push(new t(e,s,a))}},i=[],r=t.name||"default",s=t.fps||30,a=t.blendMode;let o=t.length||-1;const l=t.hierarchy||[];for(let t=0;t<l.length;t++){const r=l[t].keys;if(r&&0!==r.length)if(r[0].morphTargets){const t={};let e;for(e=0;e<r.length;e++)if(r[e].morphTargets)for(let n=0;n<r[e].morphTargets.length;n++)t[r[e].morphTargets[n]]=-1;for(const n in t){const t=[],s=[];for(let i=0;i!==r[e].morphTargets.length;++i){const i=r[e];t.push(i.time),s.push(i.morphTarget===n?1:0)}i.push(new Zl(".morphTargetInfluence["+n+"]",t,s))}o=t.length*(s||1)}else{const s=".bones["+e[t].name+"]";n(tc,s+".position",r,"pos",i),n(Kl,s+".quaternion",r,"rot",i),n(tc,s+".scale",r,"scl",i)}}if(0===i.length)return null;return new this(r,o,i,a)}resetDuration(){let t=0;for(let e=0,n=this.tracks.length;e!==n;++e){const n=this.tracks[e];t=Math.max(t,n.times[n.times.length-1])}return this.duration=t,this}trim(){for(let t=0;t<this.tracks.length;t++)this.tracks[t].trim(0,this.duration);return this}validate(){let t=!0;for(let e=0;e<this.tracks.length;e++)t=t&&this.tracks[e].validate();return t}optimize(){for(let t=0;t<this.tracks.length;t++)this.tracks[t].optimize();return this}clone(){const t=[];for(let e=0;e<this.tracks.length;e++)t.push(this.tracks[e].clone());return new this.constructor(this.name,this.duration,t,this.blendMode)}toJSON(){return this.constructor.toJSON(this)}}function nc(t){if(void 0===t.type)throw new Error("THREE.KeyframeTrack: track type undefined, can not parse");const e=function(t){switch(t.toLowerCase()){case"scalar":case"double":case"float":case"number":case"integer":return Zl;case"vector":case"vector2":case"vector3":case"vector4":return tc;case"color":return Jl;case"quaternion":return Kl;case"bool":case"boolean":return Yl;case"string":return $l}throw new Error("THREE.KeyframeTrack: Unsupported typeName: "+t)}(t.type);if(void 0===t.times){const e=[],n=[];kl.flattenJSON(t.keys,e,n,"value"),t.times=e,t.values=n}return void 0!==e.parse?e.parse(t):new e(t.name,t.times,t.values,t.interpolation)}const ic={enabled:!1,files:{},add:function(t,e){!1!==this.enabled&&(this.files[t]=e)},get:function(t){if(!1!==this.enabled)return this.files[t]},remove:function(t){delete this.files[t]},clear:function(){this.files={}}};class rc{constructor(t,e,n){const i=this;let r,s=!1,a=0,o=0;const l=[];this.onStart=void 0,this.onLoad=t,this.onProgress=e,this.onError=n,this.itemStart=function(t){o++,!1===s&&void 0!==i.onStart&&i.onStart(t,a,o),s=!0},this.itemEnd=function(t){a++,void 0!==i.onProgress&&i.onProgress(t,a,o),a===o&&(s=!1,void 0!==i.onLoad&&i.onLoad())},this.itemError=function(t){void 0!==i.onError&&i.onError(t)},this.resolveURL=function(t){return r?r(t):t},this.setURLModifier=function(t){return r=t,this},this.addHandler=function(t,e){return l.push(t,e),this},this.removeHandler=function(t){const e=l.indexOf(t);return-1!==e&&l.splice(e,2),this},this.getHandler=function(t){for(let e=0,n=l.length;e<n;e+=2){const n=l[e],i=l[e+1];if(n.global&&(n.lastIndex=0),n.test(t))return i}return null}}}const sc=new rc;class ac{constructor(t){this.manager=void 0!==t?t:sc,this.crossOrigin="anonymous",this.withCredentials=!1,this.path="",this.resourcePath="",this.requestHeader={}}load(){}loadAsync(t,e){const n=this;return new Promise((function(i,r){n.load(t,i,e,r)}))}parse(){}setCrossOrigin(t){return this.crossOrigin=t,this}setWithCredentials(t){return this.withCredentials=t,this}setPath(t){return this.path=t,this}setResourcePath(t){return this.resourcePath=t,this}setRequestHeader(t){return this.requestHeader=t,this}}const oc={};class lc extends ac{constructor(t){super(t)}load(t,e,n,i){void 0===t&&(t=""),void 0!==this.path&&(t=this.path+t),t=this.manager.resolveURL(t);const r=ic.get(t);if(void 0!==r)return this.manager.itemStart(t),setTimeout((()=>{e&&e(r),this.manager.itemEnd(t)}),0),r;if(void 0!==oc[t])return void oc[t].push({onLoad:e,onProgress:n,onError:i});oc[t]=[],oc[t].push({onLoad:e,onProgress:n,onError:i});const s=new Request(t,{headers:new Headers(this.requestHeader),credentials:this.withCredentials?"include":"same-origin"});fetch(s).then((e=>{if(200===e.status||0===e.status){0===e.status&&console.warn("THREE.FileLoader: HTTP Status 0 received.");const n=oc[t],i=e.body.getReader(),r=e.headers.get("Content-Length"),s=r?parseInt(r):0,a=0!==s;let o=0;return new ReadableStream({start(t){!function e(){i.read().then((({done:i,value:r})=>{if(i)t.close();else{o+=r.byteLength;const i=new ProgressEvent("progress",{lengthComputable:a,loaded:o,total:s});for(let t=0,e=n.length;t<e;t++){const e=n[t];e.onProgress&&e.onProgress(i)}t.enqueue(r),e()}}))}()}})}throw Error(`fetch for "${e.url}" responded with ${e.status}: ${e.statusText}`)})).then((t=>{const e=new Response(t);switch(this.responseType){case"arraybuffer":return e.arrayBuffer();case"blob":return e.blob();case"document":return e.text().then((t=>(new DOMParser).parseFromString(t,this.mimeType)));case"json":return e.json();default:return e.text()}})).then((e=>{ic.add(t,e);const n=oc[t];delete oc[t];for(let t=0,i=n.length;t<i;t++){const i=n[t];i.onLoad&&i.onLoad(e)}this.manager.itemEnd(t)})).catch((e=>{const n=oc[t];delete oc[t];for(let t=0,i=n.length;t<i;t++){const i=n[t];i.onError&&i.onError(e)}this.manager.itemError(t),this.manager.itemEnd(t)})),this.manager.itemStart(t)}setResponseType(t){return this.responseType=t,this}setMimeType(t){return this.mimeType=t,this}}class cc extends ac{constructor(t){super(t)}load(t,e,n,i){void 0!==this.path&&(t=this.path+t),t=this.manager.resolveURL(t);const r=this,s=ic.get(t);if(void 0!==s)return r.manager.itemStart(t),setTimeout((function(){e&&e(s),r.manager.itemEnd(t)}),0),s;const a=wt("img");function o(){c(),ic.add(t,this),e&&e(this),r.manager.itemEnd(t)}function l(e){c(),i&&i(e),r.manager.itemError(t),r.manager.itemEnd(t)}function c(){a.removeEventListener("load",o,!1),a.removeEventListener("error",l,!1)}return a.addEventListener("load",o,!1),a.addEventListener("error",l,!1),"data:"!==t.substr(0,5)&&void 0!==this.crossOrigin&&(a.crossOrigin=this.crossOrigin),r.manager.itemStart(t),a.src=t,a}}class hc extends ac{constructor(t){super(t)}load(t,e,n,i){const r=new ei,s=new cc(this.manager);s.setCrossOrigin(this.crossOrigin),s.setPath(this.path);let a=0;function o(n){s.load(t[n],(function(t){r.images[n]=t,a++,6===a&&(r.needsUpdate=!0,e&&e(r))}),void 0,i)}for(let e=0;e<t.length;++e)o(e);return r}}class uc extends ac{constructor(t){super(t)}load(t,e,n,i){const r=this,s=new Pa,a=new lc(this.manager);return a.setResponseType("arraybuffer"),a.setRequestHeader(this.requestHeader),a.setPath(this.path),a.setWithCredentials(r.withCredentials),a.load(t,(function(t){const n=r.parse(t);n&&(void 0!==n.image?s.image=n.image:void 0!==n.data&&(s.image.width=n.width,s.image.height=n.height,s.image.data=n.data),s.wrapS=void 0!==n.wrapS?n.wrapS:u,s.wrapT=void 0!==n.wrapT?n.wrapT:u,s.magFilter=void 0!==n.magFilter?n.magFilter:g,s.minFilter=void 0!==n.minFilter?n.minFilter:g,s.anisotropy=void 0!==n.anisotropy?n.anisotropy:1,void 0!==n.encoding&&(s.encoding=n.encoding),void 0!==n.flipY&&(s.flipY=n.flipY),void 0!==n.format&&(s.format=n.format),void 0!==n.type&&(s.type=n.type),void 0!==n.mipmaps&&(s.mipmaps=n.mipmaps,s.minFilter=y),1===n.mipmapCount&&(s.minFilter=g),void 0!==n.generateMipmaps&&(s.generateMipmaps=n.generateMipmaps),s.needsUpdate=!0,e&&e(s,n))}),n,i),s}}class dc extends ac{constructor(t){super(t)}load(t,e,n,i){const r=new Lt,s=new cc(this.manager);return s.setCrossOrigin(this.crossOrigin),s.setPath(this.path),s.load(t,(function(t){r.image=t,r.needsUpdate=!0,void 0!==e&&e(r)}),n,i),r}}class pc extends Fe{constructor(t,e=1){super(),this.type="Light",this.color=new rn(t),this.intensity=e}dispose(){}copy(t){return super.copy(t),this.color.copy(t.color),this.intensity=t.intensity,this}toJSON(t){const e=super.toJSON(t);return e.object.color=this.color.getHex(),e.object.intensity=this.intensity,void 0!==this.groundColor&&(e.object.groundColor=this.groundColor.getHex()),void 0!==this.distance&&(e.object.distance=this.distance),void 0!==this.angle&&(e.object.angle=this.angle),void 0!==this.decay&&(e.object.decay=this.decay),void 0!==this.penumbra&&(e.object.penumbra=this.penumbra),void 0!==this.shadow&&(e.object.shadow=this.shadow.toJSON()),e}}pc.prototype.isLight=!0;class mc extends pc{constructor(t,e,n){super(t,n),this.type="HemisphereLight",this.position.copy(Fe.DefaultUp),this.updateMatrix(),this.groundColor=new rn(e)}copy(t){return pc.prototype.copy.call(this,t),this.groundColor.copy(t.groundColor),this}}mc.prototype.isHemisphereLight=!0;const fc=new de,gc=new zt,vc=new zt;class yc{constructor(t){this.camera=t,this.bias=0,this.normalBias=0,this.radius=1,this.blurSamples=8,this.mapSize=new yt(512,512),this.map=null,this.mapPass=null,this.matrix=new de,this.autoUpdate=!0,this.needsUpdate=!1,this._frustum=new ci,this._frameExtents=new yt(1,1),this._viewportCount=1,this._viewports=[new Ct(0,0,1,1)]}getViewportCount(){return this._viewportCount}getFrustum(){return this._frustum}updateMatrices(t){const e=this.camera,n=this.matrix;gc.setFromMatrixPosition(t.matrixWorld),e.position.copy(gc),vc.setFromMatrixPosition(t.target.matrixWorld),e.lookAt(vc),e.updateMatrixWorld(),fc.multiplyMatrices(e.projectionMatrix,e.matrixWorldInverse),this._frustum.setFromProjectionMatrix(fc),n.set(.5,0,0,.5,0,.5,0,.5,0,0,.5,.5,0,0,0,1),n.multiply(e.projectionMatrix),n.multiply(e.matrixWorldInverse)}getViewport(t){return this._viewports[t]}getFrameExtents(){return this._frameExtents}dispose(){this.map&&this.map.dispose(),this.mapPass&&this.mapPass.dispose()}copy(t){return this.camera=t.camera.clone(),this.bias=t.bias,this.radius=t.radius,this.mapSize.copy(t.mapSize),this}clone(){return(new this.constructor).copy(this)}toJSON(){const t={};return 0!==this.bias&&(t.bias=this.bias),0!==this.normalBias&&(t.normalBias=this.normalBias),1!==this.radius&&(t.radius=this.radius),512===this.mapSize.x&&512===this.mapSize.y||(t.mapSize=this.mapSize.toArray()),t.camera=this.camera.toJSON(!1).object,delete t.camera.matrix,t}}class xc extends yc{constructor(){super(new Kn(50,1,.5,500)),this.focus=1}updateMatrices(t){const e=this.camera,n=2*ot*t.angle*this.focus,i=this.mapSize.width/this.mapSize.height,r=t.distance||e.far;n===e.fov&&i===e.aspect&&r===e.far||(e.fov=n,e.aspect=i,e.far=r,e.updateProjectionMatrix()),super.updateMatrices(t)}copy(t){return super.copy(t),this.focus=t.focus,this}}xc.prototype.isSpotLightShadow=!0;class _c extends pc{constructor(t,e,n=0,i=Math.PI/3,r=0,s=1){super(t,e),this.type="SpotLight",this.position.copy(Fe.DefaultUp),this.updateMatrix(),this.target=new Fe,this.distance=n,this.angle=i,this.penumbra=r,this.decay=s,this.shadow=new xc}get power(){return this.intensity*Math.PI}set power(t){this.intensity=t/Math.PI}dispose(){this.shadow.dispose()}copy(t){return super.copy(t),this.distance=t.distance,this.angle=t.angle,this.penumbra=t.penumbra,this.decay=t.decay,this.target=t.target.clone(),this.shadow=t.shadow.clone(),this}}_c.prototype.isSpotLight=!0;const Mc=new de,bc=new zt,wc=new zt;class Sc extends yc{constructor(){super(new Kn(90,1,.5,500)),this._frameExtents=new yt(4,2),this._viewportCount=6,this._viewports=[new Ct(2,1,1,1),new Ct(0,1,1,1),new Ct(3,1,1,1),new Ct(1,1,1,1),new Ct(3,0,1,1),new Ct(1,0,1,1)],this._cubeDirections=[new zt(1,0,0),new zt(-1,0,0),new zt(0,0,1),new zt(0,0,-1),new zt(0,1,0),new zt(0,-1,0)],this._cubeUps=[new zt(0,1,0),new zt(0,1,0),new zt(0,1,0),new zt(0,1,0),new zt(0,0,1),new zt(0,0,-1)]}updateMatrices(t,e=0){const n=this.camera,i=this.matrix,r=t.distance||n.far;r!==n.far&&(n.far=r,n.updateProjectionMatrix()),bc.setFromMatrixPosition(t.matrixWorld),n.position.copy(bc),wc.copy(n.position),wc.add(this._cubeDirections[e]),n.up.copy(this._cubeUps[e]),n.lookAt(wc),n.updateMatrixWorld(),i.makeTranslation(-bc.x,-bc.y,-bc.z),Mc.multiplyMatrices(n.projectionMatrix,n.matrixWorldInverse),this._frustum.setFromProjectionMatrix(Mc)}}Sc.prototype.isPointLightShadow=!0;class Tc extends pc{constructor(t,e,n=0,i=1){super(t,e),this.type="PointLight",this.distance=n,this.decay=i,this.shadow=new Sc}get power(){return 4*this.intensity*Math.PI}set power(t){this.intensity=t/(4*Math.PI)}dispose(){this.shadow.dispose()}copy(t){return super.copy(t),this.distance=t.distance,this.decay=t.decay,this.shadow=t.shadow.clone(),this}}Tc.prototype.isPointLight=!0;class Ec extends yc{constructor(){super(new bi(-5,5,5,-5,.5,500))}}Ec.prototype.isDirectionalLightShadow=!0;class Ac extends pc{constructor(t,e){super(t,e),this.type="DirectionalLight",this.position.copy(Fe.DefaultUp),this.updateMatrix(),this.target=new Fe,this.shadow=new Ec}dispose(){this.shadow.dispose()}copy(t){return super.copy(t),this.target=t.target.clone(),this.shadow=t.shadow.clone(),this}}Ac.prototype.isDirectionalLight=!0;class Lc extends pc{constructor(t,e){super(t,e),this.type="AmbientLight"}}Lc.prototype.isAmbientLight=!0;class Rc extends pc{constructor(t,e,n=10,i=10){super(t,e),this.type="RectAreaLight",this.width=n,this.height=i}get power(){return this.intensity*this.width*this.height*Math.PI}set power(t){this.intensity=t/(this.width*this.height*Math.PI)}copy(t){return super.copy(t),this.width=t.width,this.height=t.height,this}toJSON(t){const e=super.toJSON(t);return e.object.width=this.width,e.object.height=this.height,e}}Rc.prototype.isRectAreaLight=!0;class Cc{constructor(){this.coefficients=[];for(let t=0;t<9;t++)this.coefficients.push(new zt)}set(t){for(let e=0;e<9;e++)this.coefficients[e].copy(t[e]);return this}zero(){for(let t=0;t<9;t++)this.coefficients[t].set(0,0,0);return this}getAt(t,e){const n=t.x,i=t.y,r=t.z,s=this.coefficients;return e.copy(s[0]).multiplyScalar(.282095),e.addScaledVector(s[1],.488603*i),e.addScaledVector(s[2],.488603*r),e.addScaledVector(s[3],.488603*n),e.addScaledVector(s[4],n*i*1.092548),e.addScaledVector(s[5],i*r*1.092548),e.addScaledVector(s[6],.315392*(3*r*r-1)),e.addScaledVector(s[7],n*r*1.092548),e.addScaledVector(s[8],.546274*(n*n-i*i)),e}getIrradianceAt(t,e){const n=t.x,i=t.y,r=t.z,s=this.coefficients;return e.copy(s[0]).multiplyScalar(.886227),e.addScaledVector(s[1],1.023328*i),e.addScaledVector(s[2],1.023328*r),e.addScaledVector(s[3],1.023328*n),e.addScaledVector(s[4],.858086*n*i),e.addScaledVector(s[5],.858086*i*r),e.addScaledVector(s[6],.743125*r*r-.247708),e.addScaledVector(s[7],.858086*n*r),e.addScaledVector(s[8],.429043*(n*n-i*i)),e}add(t){for(let e=0;e<9;e++)this.coefficients[e].add(t.coefficients[e]);return this}addScaledSH(t,e){for(let n=0;n<9;n++)this.coefficients[n].addScaledVector(t.coefficients[n],e);return this}scale(t){for(let e=0;e<9;e++)this.coefficients[e].multiplyScalar(t);return this}lerp(t,e){for(let n=0;n<9;n++)this.coefficients[n].lerp(t.coefficients[n],e);return this}equals(t){for(let e=0;e<9;e++)if(!this.coefficients[e].equals(t.coefficients[e]))return!1;return!0}copy(t){return this.set(t.coefficients)}clone(){return(new this.constructor).copy(this)}fromArray(t,e=0){const n=this.coefficients;for(let i=0;i<9;i++)n[i].fromArray(t,e+3*i);return this}toArray(t=[],e=0){const n=this.coefficients;for(let i=0;i<9;i++)n[i].toArray(t,e+3*i);return t}static getBasisAt(t,e){const n=t.x,i=t.y,r=t.z;e[0]=.282095,e[1]=.488603*i,e[2]=.488603*r,e[3]=.488603*n,e[4]=1.092548*n*i,e[5]=1.092548*i*r,e[6]=.315392*(3*r*r-1),e[7]=1.092548*n*r,e[8]=.546274*(n*n-i*i)}}Cc.prototype.isSphericalHarmonics3=!0;class Pc extends pc{constructor(t=new Cc,e=1){super(void 0,e),this.sh=t}copy(t){return super.copy(t),this.sh.copy(t.sh),this}fromJSON(t){return this.intensity=t.intensity,this.sh.fromArray(t.sh),this}toJSON(t){const e=super.toJSON(t);return e.object.sh=this.sh.toArray(),e}}Pc.prototype.isLightProbe=!0;class Ic extends ac{constructor(t){super(t),this.textures={}}load(t,e,n,i){const r=this,s=new lc(r.manager);s.setPath(r.path),s.setRequestHeader(r.requestHeader),s.setWithCredentials(r.withCredentials),s.load(t,(function(n){try{e(r.parse(JSON.parse(n)))}catch(e){i?i(e):console.error(e),r.manager.itemError(t)}}),n,i)}parse(t){const e=this.textures;function n(t){return void 0===e[t]&&console.warn("THREE.MaterialLoader: Undefined texture",t),e[t]}const i=new Gl[t.type];if(void 0!==t.uuid&&(i.uuid=t.uuid),void 0!==t.name&&(i.name=t.name),void 0!==t.color&&void 0!==i.color&&i.color.setHex(t.color),void 0!==t.roughness&&(i.roughness=t.roughness),void 0!==t.metalness&&(i.metalness=t.metalness),void 0!==t.sheen&&(i.sheen=t.sheen),void 0!==t.sheenColor&&(i.sheenColor=(new rn).setHex(t.sheenColor)),void 0!==t.sheenRoughness&&(i.sheenRoughness=t.sheenRoughness),void 0!==t.emissive&&void 0!==i.emissive&&i.emissive.setHex(t.emissive),void 0!==t.specular&&void 0!==i.specular&&i.specular.setHex(t.specular),void 0!==t.specularIntensity&&(i.specularIntensity=t.specularIntensity),void 0!==t.specularColor&&void 0!==i.specularColor&&i.specularColor.setHex(t.specularColor),void 0!==t.shininess&&(i.shininess=t.shininess),void 0!==t.clearcoat&&(i.clearcoat=t.clearcoat),void 0!==t.clearcoatRoughness&&(i.clearcoatRoughness=t.clearcoatRoughness),void 0!==t.transmission&&(i.transmission=t.transmission),void 0!==t.thickness&&(i.thickness=t.thickness),void 0!==t.attenuationDistance&&(i.attenuationDistance=t.attenuationDistance),void 0!==t.attenuationColor&&void 0!==i.attenuationColor&&i.attenuationColor.setHex(t.attenuationColor),void 0!==t.fog&&(i.fog=t.fog),void 0!==t.flatShading&&(i.flatShading=t.flatShading),void 0!==t.blending&&(i.blending=t.blending),void 0!==t.combine&&(i.combine=t.combine),void 0!==t.side&&(i.side=t.side),void 0!==t.shadowSide&&(i.shadowSide=t.shadowSide),void 0!==t.opacity&&(i.opacity=t.opacity),void 0!==t.format&&(i.format=t.format),void 0!==t.transparent&&(i.transparent=t.transparent),void 0!==t.alphaTest&&(i.alphaTest=t.alphaTest),void 0!==t.depthTest&&(i.depthTest=t.depthTest),void 0!==t.depthWrite&&(i.depthWrite=t.depthWrite),void 0!==t.colorWrite&&(i.colorWrite=t.colorWrite),void 0!==t.stencilWrite&&(i.stencilWrite=t.stencilWrite),void 0!==t.stencilWriteMask&&(i.stencilWriteMask=t.stencilWriteMask),void 0!==t.stencilFunc&&(i.stencilFunc=t.stencilFunc),void 0!==t.stencilRef&&(i.stencilRef=t.stencilRef),void 0!==t.stencilFuncMask&&(i.stencilFuncMask=t.stencilFuncMask),void 0!==t.stencilFail&&(i.stencilFail=t.stencilFail),void 0!==t.stencilZFail&&(i.stencilZFail=t.stencilZFail),void 0!==t.stencilZPass&&(i.stencilZPass=t.stencilZPass),void 0!==t.wireframe&&(i.wireframe=t.wireframe),void 0!==t.wireframeLinewidth&&(i.wireframeLinewidth=t.wireframeLinewidth),void 0!==t.wireframeLinecap&&(i.wireframeLinecap=t.wireframeLinecap),void 0!==t.wireframeLinejoin&&(i.wireframeLinejoin=t.wireframeLinejoin),void 0!==t.rotation&&(i.rotation=t.rotation),1!==t.linewidth&&(i.linewidth=t.linewidth),void 0!==t.dashSize&&(i.dashSize=t.dashSize),void 0!==t.gapSize&&(i.gapSize=t.gapSize),void 0!==t.scale&&(i.scale=t.scale),void 0!==t.polygonOffset&&(i.polygonOffset=t.polygonOffset),void 0!==t.polygonOffsetFactor&&(i.polygonOffsetFactor=t.polygonOffsetFactor),void 0!==t.polygonOffsetUnits&&(i.polygonOffsetUnits=t.polygonOffsetUnits),void 0!==t.dithering&&(i.dithering=t.dithering),void 0!==t.alphaToCoverage&&(i.alphaToCoverage=t.alphaToCoverage),void 0!==t.premultipliedAlpha&&(i.premultipliedAlpha=t.premultipliedAlpha),void 0!==t.visible&&(i.visible=t.visible),void 0!==t.toneMapped&&(i.toneMapped=t.toneMapped),void 0!==t.userData&&(i.userData=t.userData),void 0!==t.vertexColors&&("number"==typeof t.vertexColors?i.vertexColors=t.vertexColors>0:i.vertexColors=t.vertexColors),void 0!==t.uniforms)for(const e in t.uniforms){const r=t.uniforms[e];switch(i.uniforms[e]={},r.type){case"t":i.uniforms[e].value=n(r.value);break;case"c":i.uniforms[e].value=(new rn).setHex(r.value);break;case"v2":i.uniforms[e].value=(new yt).fromArray(r.value);break;case"v3":i.uniforms[e].value=(new zt).fromArray(r.value);break;case"v4":i.uniforms[e].value=(new Ct).fromArray(r.value);break;case"m3":i.uniforms[e].value=(new xt).fromArray(r.value);break;case"m4":i.uniforms[e].value=(new de).fromArray(r.value);break;default:i.uniforms[e].value=r.value}}if(void 0!==t.defines&&(i.defines=t.defines),void 0!==t.vertexShader&&(i.vertexShader=t.vertexShader),void 0!==t.fragmentShader&&(i.fragmentShader=t.fragmentShader),void 0!==t.extensions)for(const e in t.extensions)i.extensions[e]=t.extensions[e];if(void 0!==t.shading&&(i.flatShading=1===t.shading),void 0!==t.size&&(i.size=t.size),void 0!==t.sizeAttenuation&&(i.sizeAttenuation=t.sizeAttenuation),void 0!==t.map&&(i.map=n(t.map)),void 0!==t.matcap&&(i.matcap=n(t.matcap)),void 0!==t.alphaMap&&(i.alphaMap=n(t.alphaMap)),void 0!==t.bumpMap&&(i.bumpMap=n(t.bumpMap)),void 0!==t.bumpScale&&(i.bumpScale=t.bumpScale),void 0!==t.normalMap&&(i.normalMap=n(t.normalMap)),void 0!==t.normalMapType&&(i.normalMapType=t.normalMapType),void 0!==t.normalScale){let e=t.normalScale;!1===Array.isArray(e)&&(e=[e,e]),i.normalScale=(new yt).fromArray(e)}return void 0!==t.displacementMap&&(i.displacementMap=n(t.displacementMap)),void 0!==t.displacementScale&&(i.displacementScale=t.displacementScale),void 0!==t.displacementBias&&(i.displacementBias=t.displacementBias),void 0!==t.roughnessMap&&(i.roughnessMap=n(t.roughnessMap)),void 0!==t.metalnessMap&&(i.metalnessMap=n(t.metalnessMap)),void 0!==t.emissiveMap&&(i.emissiveMap=n(t.emissiveMap)),void 0!==t.emissiveIntensity&&(i.emissiveIntensity=t.emissiveIntensity),void 0!==t.specularMap&&(i.specularMap=n(t.specularMap)),void 0!==t.specularIntensityMap&&(i.specularIntensityMap=n(t.specularIntensityMap)),void 0!==t.specularColorMap&&(i.specularColorMap=n(t.specularColorMap)),void 0!==t.envMap&&(i.envMap=n(t.envMap)),void 0!==t.envMapIntensity&&(i.envMapIntensity=t.envMapIntensity),void 0!==t.reflectivity&&(i.reflectivity=t.reflectivity),void 0!==t.refractionRatio&&(i.refractionRatio=t.refractionRatio),void 0!==t.lightMap&&(i.lightMap=n(t.lightMap)),void 0!==t.lightMapIntensity&&(i.lightMapIntensity=t.lightMapIntensity),void 0!==t.aoMap&&(i.aoMap=n(t.aoMap)),void 0!==t.aoMapIntensity&&(i.aoMapIntensity=t.aoMapIntensity),void 0!==t.gradientMap&&(i.gradientMap=n(t.gradientMap)),void 0!==t.clearcoatMap&&(i.clearcoatMap=n(t.clearcoatMap)),void 0!==t.clearcoatRoughnessMap&&(i.clearcoatRoughnessMap=n(t.clearcoatRoughnessMap)),void 0!==t.clearcoatNormalMap&&(i.clearcoatNormalMap=n(t.clearcoatNormalMap)),void 0!==t.clearcoatNormalScale&&(i.clearcoatNormalScale=(new yt).fromArray(t.clearcoatNormalScale)),void 0!==t.transmissionMap&&(i.transmissionMap=n(t.transmissionMap)),void 0!==t.thicknessMap&&(i.thicknessMap=n(t.thicknessMap)),void 0!==t.sheenColorMap&&(i.sheenColorMap=n(t.sheenColorMap)),void 0!==t.sheenRoughnessMap&&(i.sheenRoughnessMap=n(t.sheenRoughnessMap)),i}setTextures(t){return this.textures=t,this}}class Dc{static decodeText(t){if("undefined"!=typeof TextDecoder)return(new TextDecoder).decode(t);let e="";for(let n=0,i=t.length;n<i;n++)e+=String.fromCharCode(t[n]);try{return decodeURIComponent(escape(e))}catch(t){return e}}static extractUrlBase(t){const e=t.lastIndexOf("/");return-1===e?"./":t.substr(0,e+1)}static resolveURL(t,e){return"string"!=typeof t||""===t?"":(/^https?:\/\//i.test(e)&&/^\//.test(t)&&(e=e.replace(/(^https?:\/\/[^\/]+).*/i,"$1")),/^(https?:)?\/\//i.test(t)||/^data:.*,.*$/i.test(t)||/^blob:.*$/i.test(t)?t:e+t)}}class Nc extends En{constructor(){super(),this.type="InstancedBufferGeometry",this.instanceCount=1/0}copy(t){return super.copy(t),this.instanceCount=t.instanceCount,this}clone(){return(new this.constructor).copy(this)}toJSON(){const t=super.toJSON(this);return t.instanceCount=this.instanceCount,t.isInstancedBufferGeometry=!0,t}}Nc.prototype.isInstancedBufferGeometry=!0;class zc extends ac{constructor(t){super(t)}load(t,e,n,i){const r=this,s=new lc(r.manager);s.setPath(r.path),s.setRequestHeader(r.requestHeader),s.setWithCredentials(r.withCredentials),s.load(t,(function(n){try{e(r.parse(JSON.parse(n)))}catch(e){i?i(e):console.error(e),r.manager.itemError(t)}}),n,i)}parse(t){const e={},n={};function i(t,i){if(void 0!==e[i])return e[i];const r=t.interleavedBuffers[i],s=function(t,e){if(void 0!==n[e])return n[e];const i=t.arrayBuffers[e],r=new Uint32Array(i).buffer;return n[e]=r,r}(t,r.buffer),a=bt(r.type,s),o=new na(a,r.stride);return o.uuid=r.uuid,e[i]=o,o}const r=t.isInstancedBufferGeometry?new Nc:new En,s=t.data.index;if(void 0!==s){const t=bt(s.type,s.array);r.setIndex(new ln(t,1))}const a=t.data.attributes;for(const e in a){const n=a[e];let s;if(n.isInterleavedBufferAttribute){const e=i(t.data,n.data);s=new ra(e,n.itemSize,n.offset,n.normalized)}else{const t=bt(n.type,n.array);s=new(n.isInstancedBufferAttribute?za:ln)(t,n.itemSize,n.normalized)}void 0!==n.name&&(s.name=n.name),void 0!==n.usage&&s.setUsage(n.usage),void 0!==n.updateRange&&(s.updateRange.offset=n.updateRange.offset,s.updateRange.count=n.updateRange.count),r.setAttribute(e,s)}const o=t.data.morphAttributes;if(o)for(const e in o){const n=o[e],s=[];for(let e=0,r=n.length;e<r;e++){const r=n[e];let a;if(r.isInterleavedBufferAttribute){const e=i(t.data,r.data);a=new ra(e,r.itemSize,r.offset,r.normalized)}else{const t=bt(r.type,r.array);a=new ln(t,r.itemSize,r.normalized)}void 0!==r.name&&(a.name=r.name),s.push(a)}r.morphAttributes[e]=s}t.data.morphTargetsRelative&&(r.morphTargetsRelative=!0);const l=t.data.groups||t.data.drawcalls||t.data.offsets;if(void 0!==l)for(let t=0,e=l.length;t!==e;++t){const e=l[t];r.addGroup(e.start,e.count,e.materialIndex)}const c=t.data.boundingSphere;if(void 0!==c){const t=new zt;void 0!==c.center&&t.fromArray(c.center),r.boundingSphere=new ie(t,c.radius)}return t.name&&(r.name=t.name),t.userData&&(r.userData=t.userData),r}}const Bc={UVMapping:i,CubeReflectionMapping:r,CubeRefractionMapping:s,EquirectangularReflectionMapping:a,EquirectangularRefractionMapping:o,CubeUVReflectionMapping:l,CubeUVRefractionMapping:c},Fc={RepeatWrapping:h,ClampToEdgeWrapping:u,MirroredRepeatWrapping:d},Oc={NearestFilter:p,NearestMipmapNearestFilter:m,NearestMipmapLinearFilter:f,LinearFilter:g,LinearMipmapNearestFilter:v,LinearMipmapLinearFilter:y};class Uc extends ac{constructor(t){super(t),"undefined"==typeof createImageBitmap&&console.warn("THREE.ImageBitmapLoader: createImageBitmap() not supported."),"undefined"==typeof fetch&&console.warn("THREE.ImageBitmapLoader: fetch() not supported."),this.options={premultiplyAlpha:"none"}}setOptions(t){return this.options=t,this}load(t,e,n,i){void 0===t&&(t=""),void 0!==this.path&&(t=this.path+t),t=this.manager.resolveURL(t);const r=this,s=ic.get(t);if(void 0!==s)return r.manager.itemStart(t),setTimeout((function(){e&&e(s),r.manager.itemEnd(t)}),0),s;const a={};a.credentials="anonymous"===this.crossOrigin?"same-origin":"include",a.headers=this.requestHeader,fetch(t,a).then((function(t){return t.blob()})).then((function(t){return createImageBitmap(t,Object.assign(r.options,{colorSpaceConversion:"none"}))})).then((function(n){ic.add(t,n),e&&e(n),r.manager.itemEnd(t)})).catch((function(e){i&&i(e),r.manager.itemError(t),r.manager.itemEnd(t)})),r.manager.itemStart(t)}}let Hc;Uc.prototype.isImageBitmapLoader=!0;const Gc={getContext:function(){return void 0===Hc&&(Hc=new(window.AudioContext||window.webkitAudioContext)),Hc},setContext:function(t){Hc=t}};class kc extends ac{constructor(t){super(t)}load(t,e,n,i){const r=this,s=new lc(this.manager);s.setResponseType("arraybuffer"),s.setPath(this.path),s.setRequestHeader(this.requestHeader),s.setWithCredentials(this.withCredentials),s.load(t,(function(n){try{const t=n.slice(0);Gc.getContext().decodeAudioData(t,(function(t){e(t)}))}catch(e){i?i(e):console.error(e),r.manager.itemError(t)}}),n,i)}}class Vc extends Pc{constructor(t,e,n=1){super(void 0,n);const i=(new rn).set(t),r=(new rn).set(e),s=new zt(i.r,i.g,i.b),a=new zt(r.r,r.g,r.b),o=Math.sqrt(Math.PI),l=o*Math.sqrt(.75);this.sh.coefficients[0].copy(s).add(a).multiplyScalar(o),this.sh.coefficients[1].copy(s).sub(a).multiplyScalar(l)}}Vc.prototype.isHemisphereLightProbe=!0;class Wc extends Pc{constructor(t,e=1){super(void 0,e);const n=(new rn).set(t);this.sh.coefficients[0].set(n.r,n.g,n.b).multiplyScalar(2*Math.sqrt(Math.PI))}}Wc.prototype.isAmbientLightProbe=!0;const jc=new de,qc=new de;class Xc{constructor(t=!0){this.autoStart=t,this.startTime=0,this.oldTime=0,this.elapsedTime=0,this.running=!1}start(){this.startTime=Yc(),this.oldTime=this.startTime,this.elapsedTime=0,this.running=!0}stop(){this.getElapsedTime(),this.running=!1,this.autoStart=!1}getElapsedTime(){return this.getDelta(),this.elapsedTime}getDelta(){let t=0;if(this.autoStart&&!this.running)return this.start(),0;if(this.running){const e=Yc();t=(e-this.oldTime)/1e3,this.oldTime=e,this.elapsedTime+=t}return t}}function Yc(){return("undefined"==typeof performance?Date:performance).now()}const Jc=new zt,Zc=new Nt,Qc=new zt,Kc=new zt;class $c extends Fe{constructor(t){super(),this.type="Audio",this.listener=t,this.context=t.context,this.gain=this.context.createGain(),this.gain.connect(t.getInput()),this.autoplay=!1,this.buffer=null,this.detune=0,this.loop=!1,this.loopStart=0,this.loopEnd=0,this.offset=0,this.duration=void 0,this.playbackRate=1,this.isPlaying=!1,this.hasPlaybackControl=!0,this.source=null,this.sourceType="empty",this._startedAt=0,this._progress=0,this._connected=!1,this.filters=[]}getOutput(){return this.gain}setNodeSource(t){return this.hasPlaybackControl=!1,this.sourceType="audioNode",this.source=t,this.connect(),this}setMediaElementSource(t){return this.hasPlaybackControl=!1,this.sourceType="mediaNode",this.source=this.context.createMediaElementSource(t),this.connect(),this}setMediaStreamSource(t){return this.hasPlaybackControl=!1,this.sourceType="mediaStreamNode",this.source=this.context.createMediaStreamSource(t),this.connect(),this}setBuffer(t){return this.buffer=t,this.sourceType="buffer",this.autoplay&&this.play(),this}play(t=0){if(!0===this.isPlaying)return void console.warn("THREE.Audio: Audio is already playing.");if(!1===this.hasPlaybackControl)return void console.warn("THREE.Audio: this Audio has no playback control.");this._startedAt=this.context.currentTime+t;const e=this.context.createBufferSource();return e.buffer=this.buffer,e.loop=this.loop,e.loopStart=this.loopStart,e.loopEnd=this.loopEnd,e.onended=this.onEnded.bind(this),e.start(this._startedAt,this._progress+this.offset,this.duration),this.isPlaying=!0,this.source=e,this.setDetune(this.detune),this.setPlaybackRate(this.playbackRate),this.connect()}pause(){if(!1!==this.hasPlaybackControl)return!0===this.isPlaying&&(this._progress+=Math.max(this.context.currentTime-this._startedAt,0)*this.playbackRate,!0===this.loop&&(this._progress=this._progress%(this.duration||this.buffer.duration)),this.source.stop(),this.source.onended=null,this.isPlaying=!1),this;console.warn("THREE.Audio: this Audio has no playback control.")}stop(){if(!1!==this.hasPlaybackControl)return this._progress=0,this.source.stop(),this.source.onended=null,this.isPlaying=!1,this;console.warn("THREE.Audio: this Audio has no playback control.")}connect(){if(this.filters.length>0){this.source.connect(this.filters[0]);for(let t=1,e=this.filters.length;t<e;t++)this.filters[t-1].connect(this.filters[t]);this.filters[this.filters.length-1].connect(this.getOutput())}else this.source.connect(this.getOutput());return this._connected=!0,this}disconnect(){if(this.filters.length>0){this.source.disconnect(this.filters[0]);for(let t=1,e=this.filters.length;t<e;t++)this.filters[t-1].disconnect(this.filters[t]);this.filters[this.filters.length-1].disconnect(this.getOutput())}else this.source.disconnect(this.getOutput());return this._connected=!1,this}getFilters(){return this.filters}setFilters(t){return t||(t=[]),!0===this._connected?(this.disconnect(),this.filters=t.slice(),this.connect()):this.filters=t.slice(),this}setDetune(t){if(this.detune=t,void 0!==this.source.detune)return!0===this.isPlaying&&this.source.detune.setTargetAtTime(this.detune,this.context.currentTime,.01),this}getDetune(){return this.detune}getFilter(){return this.getFilters()[0]}setFilter(t){return this.setFilters(t?[t]:[])}setPlaybackRate(t){if(!1!==this.hasPlaybackControl)return this.playbackRate=t,!0===this.isPlaying&&this.source.playbackRate.setTargetAtTime(this.playbackRate,this.context.currentTime,.01),this;console.warn("THREE.Audio: this Audio has no playback control.")}getPlaybackRate(){return this.playbackRate}onEnded(){this.isPlaying=!1}getLoop(){return!1===this.hasPlaybackControl?(console.warn("THREE.Audio: this Audio has no playback control."),!1):this.loop}setLoop(t){if(!1!==this.hasPlaybackControl)return this.loop=t,!0===this.isPlaying&&(this.source.loop=this.loop),this;console.warn("THREE.Audio: this Audio has no playback control.")}setLoopStart(t){return this.loopStart=t,this}setLoopEnd(t){return this.loopEnd=t,this}getVolume(){return this.gain.gain.value}setVolume(t){return this.gain.gain.setTargetAtTime(t,this.context.currentTime,.01),this}}const th=new zt,eh=new Nt,nh=new zt,ih=new zt;class rh{constructor(t,e=2048){this.analyser=t.context.createAnalyser(),this.analyser.fftSize=e,this.data=new Uint8Array(this.analyser.frequencyBinCount),t.getOutput().connect(this.analyser)}getFrequencyData(){return this.analyser.getByteFrequencyData(this.data),this.data}getAverageFrequency(){let t=0;const e=this.getFrequencyData();for(let n=0;n<e.length;n++)t+=e[n];return t/e.length}}class sh{constructor(t,e,n){let i,r,s;switch(this.binding=t,this.valueSize=n,e){case"quaternion":i=this._slerp,r=this._slerpAdditive,s=this._setAdditiveIdentityQuaternion,this.buffer=new Float64Array(6*n),this._workIndex=5;break;case"string":case"bool":i=this._select,r=this._select,s=this._setAdditiveIdentityOther,this.buffer=new Array(5*n);break;default:i=this._lerp,r=this._lerpAdditive,s=this._setAdditiveIdentityNumeric,this.buffer=new Float64Array(5*n)}this._mixBufferRegion=i,this._mixBufferRegionAdditive=r,this._setIdentity=s,this._origIndex=3,this._addIndex=4,this.cumulativeWeight=0,this.cumulativeWeightAdditive=0,this.useCount=0,this.referenceCount=0}accumulate(t,e){const n=this.buffer,i=this.valueSize,r=t*i+i;let s=this.cumulativeWeight;if(0===s){for(let t=0;t!==i;++t)n[r+t]=n[t];s=e}else{s+=e;const t=e/s;this._mixBufferRegion(n,r,0,t,i)}this.cumulativeWeight=s}accumulateAdditive(t){const e=this.buffer,n=this.valueSize,i=n*this._addIndex;0===this.cumulativeWeightAdditive&&this._setIdentity(),this._mixBufferRegionAdditive(e,i,0,t,n),this.cumulativeWeightAdditive+=t}apply(t){const e=this.valueSize,n=this.buffer,i=t*e+e,r=this.cumulativeWeight,s=this.cumulativeWeightAdditive,a=this.binding;if(this.cumulativeWeight=0,this.cumulativeWeightAdditive=0,r<1){const t=e*this._origIndex;this._mixBufferRegion(n,i,t,1-r,e)}s>0&&this._mixBufferRegionAdditive(n,i,this._addIndex*e,1,e);for(let t=e,r=e+e;t!==r;++t)if(n[t]!==n[t+e]){a.setValue(n,i);break}}saveOriginalState(){const t=this.binding,e=this.buffer,n=this.valueSize,i=n*this._origIndex;t.getValue(e,i);for(let t=n,r=i;t!==r;++t)e[t]=e[i+t%n];this._setIdentity(),this.cumulativeWeight=0,this.cumulativeWeightAdditive=0}restoreOriginalState(){const t=3*this.valueSize;this.binding.setValue(this.buffer,t)}_setAdditiveIdentityNumeric(){const t=this._addIndex*this.valueSize,e=t+this.valueSize;for(let n=t;n<e;n++)this.buffer[n]=0}_setAdditiveIdentityQuaternion(){this._setAdditiveIdentityNumeric(),this.buffer[this._addIndex*this.valueSize+3]=1}_setAdditiveIdentityOther(){const t=this._origIndex*this.valueSize,e=this._addIndex*this.valueSize;for(let n=0;n<this.valueSize;n++)this.buffer[e+n]=this.buffer[t+n]}_select(t,e,n,i,r){if(i>=.5)for(let i=0;i!==r;++i)t[e+i]=t[n+i]}_slerp(t,e,n,i){Nt.slerpFlat(t,e,t,e,t,n,i)}_slerpAdditive(t,e,n,i,r){const s=this._workIndex*r;Nt.multiplyQuaternionsFlat(t,s,t,e,t,n),Nt.slerpFlat(t,e,t,e,t,s,i)}_lerp(t,e,n,i,r){const s=1-i;for(let a=0;a!==r;++a){const r=e+a;t[r]=t[r]*s+t[n+a]*i}}_lerpAdditive(t,e,n,i,r){for(let s=0;s!==r;++s){const r=e+s;t[r]=t[r]+t[n+s]*i}}}const ah="\\[\\]\\.:\\/",oh=new RegExp("[\\[\\]\\.:\\/]","g"),lh="[^\\[\\]\\.:\\/]",ch="[^"+ah.replace("\\.","")+"]",hh=/((?:WC+[\/:])*)/.source.replace("WC",lh),uh=/(WCOD+)?/.source.replace("WCOD",ch),dh=/(?:\.(WC+)(?:\[(.+)\])?)?/.source.replace("WC",lh),ph=/\.(WC+)(?:\[(.+)\])?/.source.replace("WC",lh),mh=new RegExp("^"+hh+uh+dh+ph+"$"),fh=["material","materials","bones"];class gh{constructor(t,e,n){this.path=e,this.parsedPath=n||gh.parseTrackName(e),this.node=gh.findNode(t,this.parsedPath.nodeName)||t,this.rootNode=t,this.getValue=this._getValue_unbound,this.setValue=this._setValue_unbound}static create(t,e,n){return t&&t.isAnimationObjectGroup?new gh.Composite(t,e,n):new gh(t,e,n)}static sanitizeNodeName(t){return t.replace(/\s/g,"_").replace(oh,"")}static parseTrackName(t){const e=mh.exec(t);if(!e)throw new Error("PropertyBinding: Cannot parse trackName: "+t);const n={nodeName:e[2],objectName:e[3],objectIndex:e[4],propertyName:e[5],propertyIndex:e[6]},i=n.nodeName&&n.nodeName.lastIndexOf(".");if(void 0!==i&&-1!==i){const t=n.nodeName.substring(i+1);-1!==fh.indexOf(t)&&(n.nodeName=n.nodeName.substring(0,i),n.objectName=t)}if(null===n.propertyName||0===n.propertyName.length)throw new Error("PropertyBinding: can not parse propertyName from trackName: "+t);return n}static findNode(t,e){if(!e||""===e||"."===e||-1===e||e===t.name||e===t.uuid)return t;if(t.skeleton){const n=t.skeleton.getBoneByName(e);if(void 0!==n)return n}if(t.children){const n=function(t){for(let i=0;i<t.length;i++){const r=t[i];if(r.name===e||r.uuid===e)return r;const s=n(r.children);if(s)return s}return null},i=n(t.children);if(i)return i}return null}_getValue_unavailable(){}_setValue_unavailable(){}_getValue_direct(t,e){t[e]=this.targetObject[this.propertyName]}_getValue_array(t,e){const n=this.resolvedProperty;for(let i=0,r=n.length;i!==r;++i)t[e++]=n[i]}_getValue_arrayElement(t,e){t[e]=this.resolvedProperty[this.propertyIndex]}_getValue_toArray(t,e){this.resolvedProperty.toArray(t,e)}_setValue_direct(t,e){this.targetObject[this.propertyName]=t[e]}_setValue_direct_setNeedsUpdate(t,e){this.targetObject[this.propertyName]=t[e],this.targetObject.needsUpdate=!0}_setValue_direct_setMatrixWorldNeedsUpdate(t,e){this.targetObject[this.propertyName]=t[e],this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_array(t,e){const n=this.resolvedProperty;for(let i=0,r=n.length;i!==r;++i)n[i]=t[e++]}_setValue_array_setNeedsUpdate(t,e){const n=this.resolvedProperty;for(let i=0,r=n.length;i!==r;++i)n[i]=t[e++];this.targetObject.needsUpdate=!0}_setValue_array_setMatrixWorldNeedsUpdate(t,e){const n=this.resolvedProperty;for(let i=0,r=n.length;i!==r;++i)n[i]=t[e++];this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_arrayElement(t,e){this.resolvedProperty[this.propertyIndex]=t[e]}_setValue_arrayElement_setNeedsUpdate(t,e){this.resolvedProperty[this.propertyIndex]=t[e],this.targetObject.needsUpdate=!0}_setValue_arrayElement_setMatrixWorldNeedsUpdate(t,e){this.resolvedProperty[this.propertyIndex]=t[e],this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_fromArray(t,e){this.resolvedProperty.fromArray(t,e)}_setValue_fromArray_setNeedsUpdate(t,e){this.resolvedProperty.fromArray(t,e),this.targetObject.needsUpdate=!0}_setValue_fromArray_setMatrixWorldNeedsUpdate(t,e){this.resolvedProperty.fromArray(t,e),this.targetObject.matrixWorldNeedsUpdate=!0}_getValue_unbound(t,e){this.bind(),this.getValue(t,e)}_setValue_unbound(t,e){this.bind(),this.setValue(t,e)}bind(){let t=this.node;const e=this.parsedPath,n=e.objectName,i=e.propertyName;let r=e.propertyIndex;if(t||(t=gh.findNode(this.rootNode,e.nodeName)||this.rootNode,this.node=t),this.getValue=this._getValue_unavailable,this.setValue=this._setValue_unavailable,!t)return void console.error("THREE.PropertyBinding: Trying to update node for track: "+this.path+" but it wasn't found.");if(n){let i=e.objectIndex;switch(n){case"materials":if(!t.material)return void console.error("THREE.PropertyBinding: Can not bind to material as node does not have a material.",this);if(!t.material.materials)return void console.error("THREE.PropertyBinding: Can not bind to material.materials as node.material does not have a materials array.",this);t=t.material.materials;break;case"bones":if(!t.skeleton)return void console.error("THREE.PropertyBinding: Can not bind to bones as node does not have a skeleton.",this);t=t.skeleton.bones;for(let e=0;e<t.length;e++)if(t[e].name===i){i=e;break}break;default:if(void 0===t[n])return void console.error("THREE.PropertyBinding: Can not bind to objectName of node undefined.",this);t=t[n]}if(void 0!==i){if(void 0===t[i])return void console.error("THREE.PropertyBinding: Trying to bind to objectIndex of objectName, but is undefined.",this,t);t=t[i]}}const s=t[i];if(void 0===s){const n=e.nodeName;return void console.error("THREE.PropertyBinding: Trying to update property for track: "+n+"."+i+" but it wasn't found.",t)}let a=this.Versioning.None;this.targetObject=t,void 0!==t.needsUpdate?a=this.Versioning.NeedsUpdate:void 0!==t.matrixWorldNeedsUpdate&&(a=this.Versioning.MatrixWorldNeedsUpdate);let o=this.BindingType.Direct;if(void 0!==r){if("morphTargetInfluences"===i){if(!t.geometry)return void console.error("THREE.PropertyBinding: Can not bind to morphTargetInfluences because node does not have a geometry.",this);if(!t.geometry.isBufferGeometry)return void console.error("THREE.PropertyBinding: Can not bind to morphTargetInfluences on THREE.Geometry. Use THREE.BufferGeometry instead.",this);if(!t.geometry.morphAttributes)return void console.error("THREE.PropertyBinding: Can not bind to morphTargetInfluences because node does not have a geometry.morphAttributes.",this);void 0!==t.morphTargetDictionary[r]&&(r=t.morphTargetDictionary[r])}o=this.BindingType.ArrayElement,this.resolvedProperty=s,this.propertyIndex=r}else void 0!==s.fromArray&&void 0!==s.toArray?(o=this.BindingType.HasFromToArray,this.resolvedProperty=s):Array.isArray(s)?(o=this.BindingType.EntireArray,this.resolvedProperty=s):this.propertyName=i;this.getValue=this.GetterByBindingType[o],this.setValue=this.SetterByBindingTypeAndVersioning[o][a]}unbind(){this.node=null,this.getValue=this._getValue_unbound,this.setValue=this._setValue_unbound}}gh.Composite=class{constructor(t,e,n){const i=n||gh.parseTrackName(e);this._targetGroup=t,this._bindings=t.subscribe_(e,i)}getValue(t,e){this.bind();const n=this._targetGroup.nCachedObjects_,i=this._bindings[n];void 0!==i&&i.getValue(t,e)}setValue(t,e){const n=this._bindings;for(let i=this._targetGroup.nCachedObjects_,r=n.length;i!==r;++i)n[i].setValue(t,e)}bind(){const t=this._bindings;for(let e=this._targetGroup.nCachedObjects_,n=t.length;e!==n;++e)t[e].bind()}unbind(){const t=this._bindings;for(let e=this._targetGroup.nCachedObjects_,n=t.length;e!==n;++e)t[e].unbind()}},gh.prototype.BindingType={Direct:0,EntireArray:1,ArrayElement:2,HasFromToArray:3},gh.prototype.Versioning={None:0,NeedsUpdate:1,MatrixWorldNeedsUpdate:2},gh.prototype.GetterByBindingType=[gh.prototype._getValue_direct,gh.prototype._getValue_array,gh.prototype._getValue_arrayElement,gh.prototype._getValue_toArray],gh.prototype.SetterByBindingTypeAndVersioning=[[gh.prototype._setValue_direct,gh.prototype._setValue_direct_setNeedsUpdate,gh.prototype._setValue_direct_setMatrixWorldNeedsUpdate],[gh.prototype._setValue_array,gh.prototype._setValue_array_setNeedsUpdate,gh.prototype._setValue_array_setMatrixWorldNeedsUpdate],[gh.prototype._setValue_arrayElement,gh.prototype._setValue_arrayElement_setNeedsUpdate,gh.prototype._setValue_arrayElement_setMatrixWorldNeedsUpdate],[gh.prototype._setValue_fromArray,gh.prototype._setValue_fromArray_setNeedsUpdate,gh.prototype._setValue_fromArray_setMatrixWorldNeedsUpdate]];class vh{constructor(){this.uuid=ht(),this._objects=Array.prototype.slice.call(arguments),this.nCachedObjects_=0;const t={};this._indicesByUUID=t;for(let e=0,n=arguments.length;e!==n;++e)t[arguments[e].uuid]=e;this._paths=[],this._parsedPaths=[],this._bindings=[],this._bindingsIndicesByPath={};const e=this;this.stats={objects:{get total(){return e._objects.length},get inUse(){return this.total-e.nCachedObjects_}},get bindingsPerObject(){return e._bindings.length}}}add(){const t=this._objects,e=this._indicesByUUID,n=this._paths,i=this._parsedPaths,r=this._bindings,s=r.length;let a,o=t.length,l=this.nCachedObjects_;for(let c=0,h=arguments.length;c!==h;++c){const h=arguments[c],u=h.uuid;let d=e[u];if(void 0===d){d=o++,e[u]=d,t.push(h);for(let t=0,e=s;t!==e;++t)r[t].push(new gh(h,n[t],i[t]))}else if(d<l){a=t[d];const o=--l,c=t[o];e[c.uuid]=d,t[d]=c,e[u]=o,t[o]=h;for(let t=0,e=s;t!==e;++t){const e=r[t],s=e[o];let a=e[d];e[d]=s,void 0===a&&(a=new gh(h,n[t],i[t])),e[o]=a}}else t[d]!==a&&console.error("THREE.AnimationObjectGroup: Different objects with the same UUID detected. Clean the caches or recreate your infrastructure when reloading scenes.")}this.nCachedObjects_=l}remove(){const t=this._objects,e=this._indicesByUUID,n=this._bindings,i=n.length;let r=this.nCachedObjects_;for(let s=0,a=arguments.length;s!==a;++s){const a=arguments[s],o=a.uuid,l=e[o];if(void 0!==l&&l>=r){const s=r++,c=t[s];e[c.uuid]=l,t[l]=c,e[o]=s,t[s]=a;for(let t=0,e=i;t!==e;++t){const e=n[t],i=e[s],r=e[l];e[l]=i,e[s]=r}}}this.nCachedObjects_=r}uncache(){const t=this._objects,e=this._indicesByUUID,n=this._bindings,i=n.length;let r=this.nCachedObjects_,s=t.length;for(let a=0,o=arguments.length;a!==o;++a){const o=arguments[a].uuid,l=e[o];if(void 0!==l)if(delete e[o],l<r){const a=--r,o=t[a],c=--s,h=t[c];e[o.uuid]=l,t[l]=o,e[h.uuid]=a,t[a]=h,t.pop();for(let t=0,e=i;t!==e;++t){const e=n[t],i=e[a],r=e[c];e[l]=i,e[a]=r,e.pop()}}else{const r=--s,a=t[r];r>0&&(e[a.uuid]=l),t[l]=a,t.pop();for(let t=0,e=i;t!==e;++t){const e=n[t];e[l]=e[r],e.pop()}}}this.nCachedObjects_=r}subscribe_(t,e){const n=this._bindingsIndicesByPath;let i=n[t];const r=this._bindings;if(void 0!==i)return r[i];const s=this._paths,a=this._parsedPaths,o=this._objects,l=o.length,c=this.nCachedObjects_,h=new Array(l);i=r.length,n[t]=i,s.push(t),a.push(e),r.push(h);for(let n=c,i=o.length;n!==i;++n){const i=o[n];h[n]=new gh(i,t,e)}return h}unsubscribe_(t){const e=this._bindingsIndicesByPath,n=e[t];if(void 0!==n){const i=this._paths,r=this._parsedPaths,s=this._bindings,a=s.length-1,o=s[a];e[t[a]]=n,s[n]=o,s.pop(),r[n]=r[a],r.pop(),i[n]=i[a],i.pop()}}}vh.prototype.isAnimationObjectGroup=!0;class yh{constructor(t,e,n=null,i=e.blendMode){this._mixer=t,this._clip=e,this._localRoot=n,this.blendMode=i;const r=e.tracks,s=r.length,a=new Array(s),o={endingStart:k,endingEnd:k};for(let t=0;t!==s;++t){const e=r[t].createInterpolant(null);a[t]=e,e.settings=o}this._interpolantSettings=o,this._interpolants=a,this._propertyBindings=new Array(s),this._cacheIndex=null,this._byClipCacheIndex=null,this._timeScaleInterpolant=null,this._weightInterpolant=null,this.loop=2201,this._loopCount=-1,this._startTime=null,this.time=0,this.timeScale=1,this._effectiveTimeScale=1,this.weight=1,this._effectiveWeight=1,this.repetitions=1/0,this.paused=!1,this.enabled=!0,this.clampWhenFinished=!1,this.zeroSlopeAtStart=!0,this.zeroSlopeAtEnd=!0}play(){return this._mixer._activateAction(this),this}stop(){return this._mixer._deactivateAction(this),this.reset()}reset(){return this.paused=!1,this.enabled=!0,this.time=0,this._loopCount=-1,this._startTime=null,this.stopFading().stopWarping()}isRunning(){return this.enabled&&!this.paused&&0!==this.timeScale&&null===this._startTime&&this._mixer._isActiveAction(this)}isScheduled(){return this._mixer._isActiveAction(this)}startAt(t){return this._startTime=t,this}setLoop(t,e){return this.loop=t,this.repetitions=e,this}setEffectiveWeight(t){return this.weight=t,this._effectiveWeight=this.enabled?t:0,this.stopFading()}getEffectiveWeight(){return this._effectiveWeight}fadeIn(t){return this._scheduleFading(t,0,1)}fadeOut(t){return this._scheduleFading(t,1,0)}crossFadeFrom(t,e,n){if(t.fadeOut(e),this.fadeIn(e),n){const n=this._clip.duration,i=t._clip.duration,r=i/n,s=n/i;t.warp(1,r,e),this.warp(s,1,e)}return this}crossFadeTo(t,e,n){return t.crossFadeFrom(this,e,n)}stopFading(){const t=this._weightInterpolant;return null!==t&&(this._weightInterpolant=null,this._mixer._takeBackControlInterpolant(t)),this}setEffectiveTimeScale(t){return this.timeScale=t,this._effectiveTimeScale=this.paused?0:t,this.stopWarping()}getEffectiveTimeScale(){return this._effectiveTimeScale}setDuration(t){return this.timeScale=this._clip.duration/t,this.stopWarping()}syncWith(t){return this.time=t.time,this.timeScale=t.timeScale,this.stopWarping()}halt(t){return this.warp(this._effectiveTimeScale,0,t)}warp(t,e,n){const i=this._mixer,r=i.time,s=this.timeScale;let a=this._timeScaleInterpolant;null===a&&(a=i._lendControlInterpolant(),this._timeScaleInterpolant=a);const o=a.parameterPositions,l=a.sampleValues;return o[0]=r,o[1]=r+n,l[0]=t/s,l[1]=e/s,this}stopWarping(){const t=this._timeScaleInterpolant;return null!==t&&(this._timeScaleInterpolant=null,this._mixer._takeBackControlInterpolant(t)),this}getMixer(){return this._mixer}getClip(){return this._clip}getRoot(){return this._localRoot||this._mixer._root}_update(t,e,n,i){if(!this.enabled)return void this._updateWeight(t);const r=this._startTime;if(null!==r){const i=(t-r)*n;if(i<0||0===n)return;this._startTime=null,e=n*i}e*=this._updateTimeScale(t);const s=this._updateTime(e),a=this._updateWeight(t);if(a>0){const t=this._interpolants,e=this._propertyBindings;if(this.blendMode===q)for(let n=0,i=t.length;n!==i;++n)t[n].evaluate(s),e[n].accumulateAdditive(a);else for(let n=0,r=t.length;n!==r;++n)t[n].evaluate(s),e[n].accumulate(i,a)}}_updateWeight(t){let e=0;if(this.enabled){e=this.weight;const n=this._weightInterpolant;if(null!==n){const i=n.evaluate(t)[0];e*=i,t>n.parameterPositions[1]&&(this.stopFading(),0===i&&(this.enabled=!1))}}return this._effectiveWeight=e,e}_updateTimeScale(t){let e=0;if(!this.paused){e=this.timeScale;const n=this._timeScaleInterpolant;if(null!==n){e*=n.evaluate(t)[0],t>n.parameterPositions[1]&&(this.stopWarping(),0===e?this.paused=!0:this.timeScale=e)}}return this._effectiveTimeScale=e,e}_updateTime(t){const e=this._clip.duration,n=this.loop;let i=this.time+t,r=this._loopCount;const s=2202===n;if(0===t)return-1===r?i:s&&1==(1&r)?e-i:i;if(2200===n){-1===r&&(this._loopCount=0,this._setEndings(!0,!0,!1));t:{if(i>=e)i=e;else{if(!(i<0)){this.time=i;break t}i=0}this.clampWhenFinished?this.paused=!0:this.enabled=!1,this.time=i,this._mixer.dispatchEvent({type:"finished",action:this,direction:t<0?-1:1})}}else{if(-1===r&&(t>=0?(r=0,this._setEndings(!0,0===this.repetitions,s)):this._setEndings(0===this.repetitions,!0,s)),i>=e||i<0){const n=Math.floor(i/e);i-=e*n,r+=Math.abs(n);const a=this.repetitions-r;if(a<=0)this.clampWhenFinished?this.paused=!0:this.enabled=!1,i=t>0?e:0,this.time=i,this._mixer.dispatchEvent({type:"finished",action:this,direction:t>0?1:-1});else{if(1===a){const e=t<0;this._setEndings(e,!e,s)}else this._setEndings(!1,!1,s);this._loopCount=r,this.time=i,this._mixer.dispatchEvent({type:"loop",action:this,loopDelta:n})}}else this.time=i;if(s&&1==(1&r))return e-i}return i}_setEndings(t,e,n){const i=this._interpolantSettings;n?(i.endingStart=V,i.endingEnd=V):(i.endingStart=t?this.zeroSlopeAtStart?V:k:W,i.endingEnd=e?this.zeroSlopeAtEnd?V:k:W)}_scheduleFading(t,e,n){const i=this._mixer,r=i.time;let s=this._weightInterpolant;null===s&&(s=i._lendControlInterpolant(),this._weightInterpolant=s);const a=s.parameterPositions,o=s.sampleValues;return a[0]=r,o[0]=e,a[1]=r+t,o[1]=n,this}}class xh extends rt{constructor(t){super(),this._root=t,this._initMemoryManager(),this._accuIndex=0,this.time=0,this.timeScale=1}_bindAction(t,e){const n=t._localRoot||this._root,i=t._clip.tracks,r=i.length,s=t._propertyBindings,a=t._interpolants,o=n.uuid,l=this._bindingsByRootAndName;let c=l[o];void 0===c&&(c={},l[o]=c);for(let t=0;t!==r;++t){const r=i[t],l=r.name;let h=c[l];if(void 0!==h)s[t]=h;else{if(h=s[t],void 0!==h){null===h._cacheIndex&&(++h.referenceCount,this._addInactiveBinding(h,o,l));continue}const i=e&&e._propertyBindings[t].binding.parsedPath;h=new sh(gh.create(n,l,i),r.ValueTypeName,r.getValueSize()),++h.referenceCount,this._addInactiveBinding(h,o,l),s[t]=h}a[t].resultBuffer=h.buffer}}_activateAction(t){if(!this._isActiveAction(t)){if(null===t._cacheIndex){const e=(t._localRoot||this._root).uuid,n=t._clip.uuid,i=this._actionsByClip[n];this._bindAction(t,i&&i.knownActions[0]),this._addInactiveAction(t,n,e)}const e=t._propertyBindings;for(let t=0,n=e.length;t!==n;++t){const n=e[t];0==n.useCount++&&(this._lendBinding(n),n.saveOriginalState())}this._lendAction(t)}}_deactivateAction(t){if(this._isActiveAction(t)){const e=t._propertyBindings;for(let t=0,n=e.length;t!==n;++t){const n=e[t];0==--n.useCount&&(n.restoreOriginalState(),this._takeBackBinding(n))}this._takeBackAction(t)}}_initMemoryManager(){this._actions=[],this._nActiveActions=0,this._actionsByClip={},this._bindings=[],this._nActiveBindings=0,this._bindingsByRootAndName={},this._controlInterpolants=[],this._nActiveControlInterpolants=0;const t=this;this.stats={actions:{get total(){return t._actions.length},get inUse(){return t._nActiveActions}},bindings:{get total(){return t._bindings.length},get inUse(){return t._nActiveBindings}},controlInterpolants:{get total(){return t._controlInterpolants.length},get inUse(){return t._nActiveControlInterpolants}}}}_isActiveAction(t){const e=t._cacheIndex;return null!==e&&e<this._nActiveActions}_addInactiveAction(t,e,n){const i=this._actions,r=this._actionsByClip;let s=r[e];if(void 0===s)s={knownActions:[t],actionByRoot:{}},t._byClipCacheIndex=0,r[e]=s;else{const e=s.knownActions;t._byClipCacheIndex=e.length,e.push(t)}t._cacheIndex=i.length,i.push(t),s.actionByRoot[n]=t}_removeInactiveAction(t){const e=this._actions,n=e[e.length-1],i=t._cacheIndex;n._cacheIndex=i,e[i]=n,e.pop(),t._cacheIndex=null;const r=t._clip.uuid,s=this._actionsByClip,a=s[r],o=a.knownActions,l=o[o.length-1],c=t._byClipCacheIndex;l._byClipCacheIndex=c,o[c]=l,o.pop(),t._byClipCacheIndex=null;delete a.actionByRoot[(t._localRoot||this._root).uuid],0===o.length&&delete s[r],this._removeInactiveBindingsForAction(t)}_removeInactiveBindingsForAction(t){const e=t._propertyBindings;for(let t=0,n=e.length;t!==n;++t){const n=e[t];0==--n.referenceCount&&this._removeInactiveBinding(n)}}_lendAction(t){const e=this._actions,n=t._cacheIndex,i=this._nActiveActions++,r=e[i];t._cacheIndex=i,e[i]=t,r._cacheIndex=n,e[n]=r}_takeBackAction(t){const e=this._actions,n=t._cacheIndex,i=--this._nActiveActions,r=e[i];t._cacheIndex=i,e[i]=t,r._cacheIndex=n,e[n]=r}_addInactiveBinding(t,e,n){const i=this._bindingsByRootAndName,r=this._bindings;let s=i[e];void 0===s&&(s={},i[e]=s),s[n]=t,t._cacheIndex=r.length,r.push(t)}_removeInactiveBinding(t){const e=this._bindings,n=t.binding,i=n.rootNode.uuid,r=n.path,s=this._bindingsByRootAndName,a=s[i],o=e[e.length-1],l=t._cacheIndex;o._cacheIndex=l,e[l]=o,e.pop(),delete a[r],0===Object.keys(a).length&&delete s[i]}_lendBinding(t){const e=this._bindings,n=t._cacheIndex,i=this._nActiveBindings++,r=e[i];t._cacheIndex=i,e[i]=t,r._cacheIndex=n,e[n]=r}_takeBackBinding(t){const e=this._bindings,n=t._cacheIndex,i=--this._nActiveBindings,r=e[i];t._cacheIndex=i,e[i]=t,r._cacheIndex=n,e[n]=r}_lendControlInterpolant(){const t=this._controlInterpolants,e=this._nActiveControlInterpolants++;let n=t[e];return void 0===n&&(n=new jl(new Float32Array(2),new Float32Array(2),1,this._controlInterpolantsResultBuffer),n.__cacheIndex=e,t[e]=n),n}_takeBackControlInterpolant(t){const e=this._controlInterpolants,n=t.__cacheIndex,i=--this._nActiveControlInterpolants,r=e[i];t.__cacheIndex=i,e[i]=t,r.__cacheIndex=n,e[n]=r}clipAction(t,e,n){const i=e||this._root,r=i.uuid;let s="string"==typeof t?ec.findByName(i,t):t;const a=null!==s?s.uuid:t,o=this._actionsByClip[a];let l=null;if(void 0===n&&(n=null!==s?s.blendMode:j),void 0!==o){const t=o.actionByRoot[r];if(void 0!==t&&t.blendMode===n)return t;l=o.knownActions[0],null===s&&(s=l._clip)}if(null===s)return null;const c=new yh(this,s,e,n);return this._bindAction(c,l),this._addInactiveAction(c,a,r),c}existingAction(t,e){const n=e||this._root,i=n.uuid,r="string"==typeof t?ec.findByName(n,t):t,s=r?r.uuid:t,a=this._actionsByClip[s];return void 0!==a&&a.actionByRoot[i]||null}stopAllAction(){const t=this._actions;for(let e=this._nActiveActions-1;e>=0;--e)t[e].stop();return this}update(t){t*=this.timeScale;const e=this._actions,n=this._nActiveActions,i=this.time+=t,r=Math.sign(t),s=this._accuIndex^=1;for(let a=0;a!==n;++a){e[a]._update(i,t,r,s)}const a=this._bindings,o=this._nActiveBindings;for(let t=0;t!==o;++t)a[t].apply(s);return this}setTime(t){this.time=0;for(let t=0;t<this._actions.length;t++)this._actions[t].time=0;return this.update(t)}getRoot(){return this._root}uncacheClip(t){const e=this._actions,n=t.uuid,i=this._actionsByClip,r=i[n];if(void 0!==r){const t=r.knownActions;for(let n=0,i=t.length;n!==i;++n){const i=t[n];this._deactivateAction(i);const r=i._cacheIndex,s=e[e.length-1];i._cacheIndex=null,i._byClipCacheIndex=null,s._cacheIndex=r,e[r]=s,e.pop(),this._removeInactiveBindingsForAction(i)}delete i[n]}}uncacheRoot(t){const e=t.uuid,n=this._actionsByClip;for(const t in n){const i=n[t].actionByRoot[e];void 0!==i&&(this._deactivateAction(i),this._removeInactiveAction(i))}const i=this._bindingsByRootAndName[e];if(void 0!==i)for(const t in i){const e=i[t];e.restoreOriginalState(),this._removeInactiveBinding(e)}}uncacheAction(t,e){const n=this.existingAction(t,e);null!==n&&(this._deactivateAction(n),this._removeInactiveAction(n))}}xh.prototype._controlInterpolantsResultBuffer=new Float32Array(1);class _h{constructor(t){"string"==typeof t&&(console.warn("THREE.Uniform: Type parameter is no longer needed."),t=arguments[1]),this.value=t}clone(){return new _h(void 0===this.value.clone?this.value:this.value.clone())}}class Mh extends na{constructor(t,e,n=1){super(t,e),this.meshPerAttribute=n}copy(t){return super.copy(t),this.meshPerAttribute=t.meshPerAttribute,this}clone(t){const e=super.clone(t);return e.meshPerAttribute=this.meshPerAttribute,e}toJSON(t){const e=super.toJSON(t);return e.isInstancedInterleavedBuffer=!0,e.meshPerAttribute=this.meshPerAttribute,e}}Mh.prototype.isInstancedInterleavedBuffer=!0;class bh{constructor(t,e,n,i,r){this.buffer=t,this.type=e,this.itemSize=n,this.elementSize=i,this.count=r,this.version=0}set needsUpdate(t){!0===t&&this.version++}setBuffer(t){return this.buffer=t,this}setType(t,e){return this.type=t,this.elementSize=e,this}setItemSize(t){return this.itemSize=t,this}setCount(t){return this.count=t,this}}bh.prototype.isGLBufferAttribute=!0;function wh(t,e){return t.distance-e.distance}function Sh(t,e,n,i){if(t.layers.test(e.layers)&&t.raycast(e,n),!0===i){const i=t.children;for(let t=0,r=i.length;t<r;t++)Sh(i[t],e,n,!0)}}const Th=new yt;class Eh{constructor(t=new yt(1/0,1/0),e=new yt(-1/0,-1/0)){this.min=t,this.max=e}set(t,e){return this.min.copy(t),this.max.copy(e),this}setFromPoints(t){this.makeEmpty();for(let e=0,n=t.length;e<n;e++)this.expandByPoint(t[e]);return this}setFromCenterAndSize(t,e){const n=Th.copy(e).multiplyScalar(.5);return this.min.copy(t).sub(n),this.max.copy(t).add(n),this}clone(){return(new this.constructor).copy(this)}copy(t){return this.min.copy(t.min),this.max.copy(t.max),this}makeEmpty(){return this.min.x=this.min.y=1/0,this.max.x=this.max.y=-1/0,this}isEmpty(){return this.max.x<this.min.x||this.max.y<this.min.y}getCenter(t){return this.isEmpty()?t.set(0,0):t.addVectors(this.min,this.max).multiplyScalar(.5)}getSize(t){return this.isEmpty()?t.set(0,0):t.subVectors(this.max,this.min)}expandByPoint(t){return this.min.min(t),this.max.max(t),this}expandByVector(t){return this.min.sub(t),this.max.add(t),this}expandByScalar(t){return this.min.addScalar(-t),this.max.addScalar(t),this}containsPoint(t){return!(t.x<this.min.x||t.x>this.max.x||t.y<this.min.y||t.y>this.max.y)}containsBox(t){return this.min.x<=t.min.x&&t.max.x<=this.max.x&&this.min.y<=t.min.y&&t.max.y<=this.max.y}getParameter(t,e){return e.set((t.x-this.min.x)/(this.max.x-this.min.x),(t.y-this.min.y)/(this.max.y-this.min.y))}intersectsBox(t){return!(t.max.x<this.min.x||t.min.x>this.max.x||t.max.y<this.min.y||t.min.y>this.max.y)}clampPoint(t,e){return e.copy(t).clamp(this.min,this.max)}distanceToPoint(t){return Th.copy(t).clamp(this.min,this.max).sub(t).length()}intersect(t){return this.min.max(t.min),this.max.min(t.max),this}union(t){return this.min.min(t.min),this.max.max(t.max),this}translate(t){return this.min.add(t),this.max.add(t),this}equals(t){return t.min.equals(this.min)&&t.max.equals(this.max)}}Eh.prototype.isBox2=!0;const Ah=new zt,Lh=new zt;class Rh{constructor(t=new zt,e=new zt){this.start=t,this.end=e}set(t,e){return this.start.copy(t),this.end.copy(e),this}copy(t){return this.start.copy(t.start),this.end.copy(t.end),this}getCenter(t){return t.addVectors(this.start,this.end).multiplyScalar(.5)}delta(t){return t.subVectors(this.end,this.start)}distanceSq(){return this.start.distanceToSquared(this.end)}distance(){return this.start.distanceTo(this.end)}at(t,e){return this.delta(e).multiplyScalar(t).add(this.start)}closestPointToPointParameter(t,e){Ah.subVectors(t,this.start),Lh.subVectors(this.end,this.start);const n=Lh.dot(Lh);let i=Lh.dot(Ah)/n;return e&&(i=ut(i,0,1)),i}closestPointToPoint(t,e,n){const i=this.closestPointToPointParameter(t,e);return this.delta(n).multiplyScalar(i).add(this.start)}applyMatrix4(t){return this.start.applyMatrix4(t),this.end.applyMatrix4(t),this}equals(t){return t.start.equals(this.start)&&t.end.equals(this.end)}clone(){return(new this.constructor).copy(this)}}const Ch=new zt;const Ph=new zt,Ih=new de,Dh=new de;class Nh extends Za{constructor(t){const e=zh(t),n=new En,i=[],r=[],s=new rn(0,0,1),a=new rn(0,1,0);for(let t=0;t<e.length;t++){const n=e[t];n.parent&&n.parent.isBone&&(i.push(0,0,0),i.push(0,0,0),r.push(s.r,s.g,s.b),r.push(a.r,a.g,a.b))}n.setAttribute("position",new vn(i,3)),n.setAttribute("color",new vn(r,3));super(n,new Ga({vertexColors:!0,depthTest:!1,depthWrite:!1,toneMapped:!1,transparent:!0})),this.type="SkeletonHelper",this.isSkeletonHelper=!0,this.root=t,this.bones=e,this.matrix=t.matrixWorld,this.matrixAutoUpdate=!1}updateMatrixWorld(t){const e=this.bones,n=this.geometry,i=n.getAttribute("position");Dh.copy(this.root.matrixWorld).invert();for(let t=0,n=0;t<e.length;t++){const r=e[t];r.parent&&r.parent.isBone&&(Ih.multiplyMatrices(Dh,r.matrixWorld),Ph.setFromMatrixPosition(Ih),i.setXYZ(n,Ph.x,Ph.y,Ph.z),Ih.multiplyMatrices(Dh,r.parent.matrixWorld),Ph.setFromMatrixPosition(Ih),i.setXYZ(n+1,Ph.x,Ph.y,Ph.z),n+=2)}n.getAttribute("position").needsUpdate=!0,super.updateMatrixWorld(t)}}function zh(t){const e=[];t&&t.isBone&&e.push(t);for(let n=0;n<t.children.length;n++)e.push.apply(e,zh(t.children[n]));return e}const Bh=new zt,Fh=new rn,Oh=new rn;class Uh extends Za{constructor(t=10,e=10,n=4473924,i=8947848){n=new rn(n),i=new rn(i);const r=e/2,s=t/e,a=t/2,o=[],l=[];for(let t=0,c=0,h=-a;t<=e;t++,h+=s){o.push(-a,0,h,a,0,h),o.push(h,0,-a,h,0,a);const e=t===r?n:i;e.toArray(l,c),c+=3,e.toArray(l,c),c+=3,e.toArray(l,c),c+=3,e.toArray(l,c),c+=3}const c=new En;c.setAttribute("position",new vn(o,3)),c.setAttribute("color",new vn(l,3));super(c,new Ga({vertexColors:!0,toneMapped:!1})),this.type="GridHelper"}}const Hh=new zt,Gh=new zt,kh=new zt;const Vh=new zt,Wh=new Qn;function jh(t,e,n,i,r,s,a){Vh.set(r,s,a).unproject(i);const o=e[t];if(void 0!==o){const t=n.getAttribute("position");for(let e=0,n=o.length;e<n;e++)t.setXYZ(o[e],Vh.x,Vh.y,Vh.z)}}const qh=new Ot;class Xh extends Za{constructor(t,e=16776960){const n=new Uint16Array([0,1,1,2,2,3,3,0,4,5,5,6,6,7,7,4,0,4,1,5,2,6,3,7]),i=new Float32Array(24),r=new En;r.setIndex(new ln(n,1)),r.setAttribute("position",new ln(i,3)),super(r,new Ga({color:e,toneMapped:!1})),this.object=t,this.type="BoxHelper",this.matrixAutoUpdate=!1,this.update()}update(t){if(void 0!==t&&console.warn("THREE.BoxHelper: .update() has no longer arguments."),void 0!==this.object&&qh.setFromObject(this.object),qh.isEmpty())return;const e=qh.min,n=qh.max,i=this.geometry.attributes.position,r=i.array;r[0]=n.x,r[1]=n.y,r[2]=n.z,r[3]=e.x,r[4]=n.y,r[5]=n.z,r[6]=e.x,r[7]=e.y,r[8]=n.z,r[9]=n.x,r[10]=e.y,r[11]=n.z,r[12]=n.x,r[13]=n.y,r[14]=e.z,r[15]=e.x,r[16]=n.y,r[17]=e.z,r[18]=e.x,r[19]=e.y,r[20]=e.z,r[21]=n.x,r[22]=e.y,r[23]=e.z,i.needsUpdate=!0,this.geometry.computeBoundingSphere()}setFromObject(t){return this.object=t,this.update(),this}copy(t){return Za.prototype.copy.call(this,t),this.object=t.object,this}}const Yh=new zt;let Jh,Zh;class Qh extends Za{constructor(t=1){const e=[0,0,0,t,0,0,0,0,0,0,t,0,0,0,0,0,0,t],n=new En;n.setAttribute("position",new vn(e,3)),n.setAttribute("color",new vn([1,0,0,1,.6,0,0,1,0,.6,1,0,0,0,1,0,.6,1],3));super(n,new Ga({vertexColors:!0,toneMapped:!1})),this.type="AxesHelper"}setColors(t,e,n){const i=new rn,r=this.geometry.attributes.color.array;return i.set(t),i.toArray(r,0),i.toArray(r,3),i.set(e),i.toArray(r,6),i.toArray(r,9),i.set(n),i.toArray(r,12),i.toArray(r,15),this.geometry.attributes.color.needsUpdate=!0,this}dispose(){this.geometry.dispose(),this.material.dispose()}}const Kh=new Float32Array(1),$h=new Int32Array(Kh.buffer);_o.create=function(t,e){return console.log("THREE.Curve.create() has been deprecated"),t.prototype=Object.create(_o.prototype),t.prototype.constructor=t,t.prototype.getPoint=e,t},Go.prototype.fromPoints=function(t){return console.warn("THREE.Path: .fromPoints() has been renamed to .setFromPoints()."),this.setFromPoints(t)},Uh.prototype.setColors=function(){console.error("THREE.GridHelper: setColors() has been deprecated, pass them in the constructor instead.")},Nh.prototype.update=function(){console.error("THREE.SkeletonHelper: update() no longer needs to be called.")},ac.prototype.extractUrlBase=function(t){return console.warn("THREE.Loader: .extractUrlBase() has been deprecated. Use THREE.LoaderUtils.extractUrlBase() instead."),Dc.extractUrlBase(t)},ac.Handlers={add:function(){console.error("THREE.Loader: Handlers.add() has been removed. Use LoadingManager.addHandler() instead.")},get:function(){console.error("THREE.Loader: Handlers.get() has been removed. Use LoadingManager.getHandler() instead.")}},Eh.prototype.center=function(t){return console.warn("THREE.Box2: .center() has been renamed to .getCenter()."),this.getCenter(t)},Eh.prototype.empty=function(){return console.warn("THREE.Box2: .empty() has been renamed to .isEmpty()."),this.isEmpty()},Eh.prototype.isIntersectionBox=function(t){return console.warn("THREE.Box2: .isIntersectionBox() has been renamed to .intersectsBox()."),this.intersectsBox(t)},Eh.prototype.size=function(t){return console.warn("THREE.Box2: .size() has been renamed to .getSize()."),this.getSize(t)},Ot.prototype.center=function(t){return console.warn("THREE.Box3: .center() has been renamed to .getCenter()."),this.getCenter(t)},Ot.prototype.empty=function(){return console.warn("THREE.Box3: .empty() has been renamed to .isEmpty()."),this.isEmpty()},Ot.prototype.isIntersectionBox=function(t){return console.warn("THREE.Box3: .isIntersectionBox() has been renamed to .intersectsBox()."),this.intersectsBox(t)},Ot.prototype.isIntersectionSphere=function(t){return console.warn("THREE.Box3: .isIntersectionSphere() has been renamed to .intersectsSphere()."),this.intersectsSphere(t)},Ot.prototype.size=function(t){return console.warn("THREE.Box3: .size() has been renamed to .getSize()."),this.getSize(t)},ie.prototype.empty=function(){return console.warn("THREE.Sphere: .empty() has been renamed to .isEmpty()."),this.isEmpty()},ci.prototype.setFromMatrix=function(t){return console.warn("THREE.Frustum: .setFromMatrix() has been renamed to .setFromProjectionMatrix()."),this.setFromProjectionMatrix(t)},Rh.prototype.center=function(t){return console.warn("THREE.Line3: .center() has been renamed to .getCenter()."),this.getCenter(t)},xt.prototype.flattenToArrayOffset=function(t,e){return console.warn("THREE.Matrix3: .flattenToArrayOffset() has been deprecated. Use .toArray() instead."),this.toArray(t,e)},xt.prototype.multiplyVector3=function(t){return console.warn("THREE.Matrix3: .multiplyVector3() has been removed. Use vector.applyMatrix3( matrix ) instead."),t.applyMatrix3(this)},xt.prototype.multiplyVector3Array=function(){console.error("THREE.Matrix3: .multiplyVector3Array() has been removed.")},xt.prototype.applyToBufferAttribute=function(t){return console.warn("THREE.Matrix3: .applyToBufferAttribute() has been removed. Use attribute.applyMatrix3( matrix ) instead."),t.applyMatrix3(this)},xt.prototype.applyToVector3Array=function(){console.error("THREE.Matrix3: .applyToVector3Array() has been removed.")},xt.prototype.getInverse=function(t){return console.warn("THREE.Matrix3: .getInverse() has been removed. Use matrixInv.copy( matrix ).invert(); instead."),this.copy(t).invert()},de.prototype.extractPosition=function(t){return console.warn("THREE.Matrix4: .extractPosition() has been renamed to .copyPosition()."),this.copyPosition(t)},de.prototype.flattenToArrayOffset=function(t,e){return console.warn("THREE.Matrix4: .flattenToArrayOffset() has been deprecated. Use .toArray() instead."),this.toArray(t,e)},de.prototype.getPosition=function(){return console.warn("THREE.Matrix4: .getPosition() has been removed. Use Vector3.setFromMatrixPosition( matrix ) instead."),(new zt).setFromMatrixColumn(this,3)},de.prototype.setRotationFromQuaternion=function(t){return console.warn("THREE.Matrix4: .setRotationFromQuaternion() has been renamed to .makeRotationFromQuaternion()."),this.makeRotationFromQuaternion(t)},de.prototype.multiplyToArray=function(){console.warn("THREE.Matrix4: .multiplyToArray() has been removed.")},de.prototype.multiplyVector3=function(t){return console.warn("THREE.Matrix4: .multiplyVector3() has been removed. Use vector.applyMatrix4( matrix ) instead."),t.applyMatrix4(this)},de.prototype.multiplyVector4=function(t){return console.warn("THREE.Matrix4: .multiplyVector4() has been removed. Use vector.applyMatrix4( matrix ) instead."),t.applyMatrix4(this)},de.prototype.multiplyVector3Array=function(){console.error("THREE.Matrix4: .multiplyVector3Array() has been removed.")},de.prototype.rotateAxis=function(t){console.warn("THREE.Matrix4: .rotateAxis() has been removed. Use Vector3.transformDirection( matrix ) instead."),t.transformDirection(this)},de.prototype.crossVector=function(t){return console.warn("THREE.Matrix4: .crossVector() has been removed. Use vector.applyMatrix4( matrix ) instead."),t.applyMatrix4(this)},de.prototype.translate=function(){console.error("THREE.Matrix4: .translate() has been removed.")},de.prototype.rotateX=function(){console.error("THREE.Matrix4: .rotateX() has been removed.")},de.prototype.rotateY=function(){console.error("THREE.Matrix4: .rotateY() has been removed.")},de.prototype.rotateZ=function(){console.error("THREE.Matrix4: .rotateZ() has been removed.")},de.prototype.rotateByAxis=function(){console.error("THREE.Matrix4: .rotateByAxis() has been removed.")},de.prototype.applyToBufferAttribute=function(t){return console.warn("THREE.Matrix4: .applyToBufferAttribute() has been removed. Use attribute.applyMatrix4( matrix ) instead."),t.applyMatrix4(this)},de.prototype.applyToVector3Array=function(){console.error("THREE.Matrix4: .applyToVector3Array() has been removed.")},de.prototype.makeFrustum=function(t,e,n,i,r,s){return console.warn("THREE.Matrix4: .makeFrustum() has been removed. Use .makePerspective( left, right, top, bottom, near, far ) instead."),this.makePerspective(t,e,i,n,r,s)},de.prototype.getInverse=function(t){return console.warn("THREE.Matrix4: .getInverse() has been removed. Use matrixInv.copy( matrix ).invert(); instead."),this.copy(t).invert()},ai.prototype.isIntersectionLine=function(t){return console.warn("THREE.Plane: .isIntersectionLine() has been renamed to .intersectsLine()."),this.intersectsLine(t)},Nt.prototype.multiplyVector3=function(t){return console.warn("THREE.Quaternion: .multiplyVector3() has been removed. Use is now vector.applyQuaternion( quaternion ) instead."),t.applyQuaternion(this)},Nt.prototype.inverse=function(){return console.warn("THREE.Quaternion: .inverse() has been renamed to invert()."),this.invert()},ue.prototype.isIntersectionBox=function(t){return console.warn("THREE.Ray: .isIntersectionBox() has been renamed to .intersectsBox()."),this.intersectsBox(t)},ue.prototype.isIntersectionPlane=function(t){return console.warn("THREE.Ray: .isIntersectionPlane() has been renamed to .intersectsPlane()."),this.intersectsPlane(t)},ue.prototype.isIntersectionSphere=function(t){return console.warn("THREE.Ray: .isIntersectionSphere() has been renamed to .intersectsSphere()."),this.intersectsSphere(t)},Ye.prototype.area=function(){return console.warn("THREE.Triangle: .area() has been renamed to .getArea()."),this.getArea()},Ye.prototype.barycoordFromPoint=function(t,e){return console.warn("THREE.Triangle: .barycoordFromPoint() has been renamed to .getBarycoord()."),this.getBarycoord(t,e)},Ye.prototype.midpoint=function(t){return console.warn("THREE.Triangle: .midpoint() has been renamed to .getMidpoint()."),this.getMidpoint(t)},Ye.prototypenormal=function(t){return console.warn("THREE.Triangle: .normal() has been renamed to .getNormal()."),this.getNormal(t)},Ye.prototype.plane=function(t){return console.warn("THREE.Triangle: .plane() has been renamed to .getPlane()."),this.getPlane(t)},Ye.barycoordFromPoint=function(t,e,n,i,r){return console.warn("THREE.Triangle: .barycoordFromPoint() has been renamed to .getBarycoord()."),Ye.getBarycoord(t,e,n,i,r)},Ye.normal=function(t,e,n,i){return console.warn("THREE.Triangle: .normal() has been renamed to .getNormal()."),Ye.getNormal(t,e,n,i)},ko.prototype.extractAllPoints=function(t){return console.warn("THREE.Shape: .extractAllPoints() has been removed. Use .extractPoints() instead."),this.extractPoints(t)},ko.prototype.extrude=function(t){return console.warn("THREE.Shape: .extrude() has been removed. Use ExtrudeGeometry() instead."),new vl(this,t)},ko.prototype.makeGeometry=function(t){return console.warn("THREE.Shape: .makeGeometry() has been removed. Use ShapeGeometry() instead."),new wl(this,t)},yt.prototype.fromAttribute=function(t,e,n){return console.warn("THREE.Vector2: .fromAttribute() has been renamed to .fromBufferAttribute()."),this.fromBufferAttribute(t,e,n)},yt.prototype.distanceToManhattan=function(t){return console.warn("THREE.Vector2: .distanceToManhattan() has been renamed to .manhattanDistanceTo()."),this.manhattanDistanceTo(t)},yt.prototype.lengthManhattan=function(){return console.warn("THREE.Vector2: .lengthManhattan() has been renamed to .manhattanLength()."),this.manhattanLength()},zt.prototype.setEulerFromRotationMatrix=function(){console.error("THREE.Vector3: .setEulerFromRotationMatrix() has been removed. Use Euler.setFromRotationMatrix() instead.")},zt.prototype.setEulerFromQuaternion=function(){console.error("THREE.Vector3: .setEulerFromQuaternion() has been removed. Use Euler.setFromQuaternion() instead.")},zt.prototype.getPositionFromMatrix=function(t){return console.warn("THREE.Vector3: .getPositionFromMatrix() has been renamed to .setFromMatrixPosition()."),this.setFromMatrixPosition(t)},zt.prototype.getScaleFromMatrix=function(t){return console.warn("THREE.Vector3: .getScaleFromMatrix() has been renamed to .setFromMatrixScale()."),this.setFromMatrixScale(t)},zt.prototype.getColumnFromMatrix=function(t,e){return console.warn("THREE.Vector3: .getColumnFromMatrix() has been renamed to .setFromMatrixColumn()."),this.setFromMatrixColumn(e,t)},zt.prototype.applyProjection=function(t){return console.warn("THREE.Vector3: .applyProjection() has been removed. Use .applyMatrix4( m ) instead."),this.applyMatrix4(t)},zt.prototype.fromAttribute=function(t,e,n){return console.warn("THREE.Vector3: .fromAttribute() has been renamed to .fromBufferAttribute()."),this.fromBufferAttribute(t,e,n)},zt.prototype.distanceToManhattan=function(t){return console.warn("THREE.Vector3: .distanceToManhattan() has been renamed to .manhattanDistanceTo()."),this.manhattanDistanceTo(t)},zt.prototype.lengthManhattan=function(){return console.warn("THREE.Vector3: .lengthManhattan() has been renamed to .manhattanLength()."),this.manhattanLength()},Ct.prototype.fromAttribute=function(t,e,n){return console.warn("THREE.Vector4: .fromAttribute() has been renamed to .fromBufferAttribute()."),this.fromBufferAttribute(t,e,n)},Ct.prototype.lengthManhattan=function(){return console.warn("THREE.Vector4: .lengthManhattan() has been renamed to .manhattanLength()."),this.manhattanLength()},Fe.prototype.getChildByName=function(t){return console.warn("THREE.Object3D: .getChildByName() has been renamed to .getObjectByName()."),this.getObjectByName(t)},Fe.prototype.renderDepth=function(){console.warn("THREE.Object3D: .renderDepth has been removed. Use .renderOrder, instead.")},Fe.prototype.translate=function(t,e){return console.warn("THREE.Object3D: .translate() has been removed. Use .translateOnAxis( axis, distance ) instead."),this.translateOnAxis(e,t)},Fe.prototype.getWorldRotation=function(){console.error("THREE.Object3D: .getWorldRotation() has been removed. Use THREE.Object3D.getWorldQuaternion( target ) instead.")},Fe.prototype.applyMatrix=function(t){return console.warn("THREE.Object3D: .applyMatrix() has been renamed to .applyMatrix4()."),this.applyMatrix4(t)},Object.defineProperties(Fe.prototype,{eulerOrder:{get:function(){return console.warn("THREE.Object3D: .eulerOrder is now .rotation.order."),this.rotation.order},set:function(t){console.warn("THREE.Object3D: .eulerOrder is now .rotation.order."),this.rotation.order=t}},useQuaternion:{get:function(){console.warn("THREE.Object3D: .useQuaternion has been removed. The library now uses quaternions by default.")},set:function(){console.warn("THREE.Object3D: .useQuaternion has been removed. The library now uses quaternions by default.")}}}),Wn.prototype.setDrawMode=function(){console.error("THREE.Mesh: .setDrawMode() has been removed. The renderer now always assumes THREE.TrianglesDrawMode. Transform your geometry via BufferGeometryUtils.toTrianglesDrawMode() if necessary.")},Object.defineProperties(Wn.prototype,{drawMode:{get:function(){return console.error("THREE.Mesh: .drawMode has been removed. The renderer now always assumes THREE.TrianglesDrawMode."),0},set:function(){console.error("THREE.Mesh: .drawMode has been removed. The renderer now always assumes THREE.TrianglesDrawMode. Transform your geometry via BufferGeometryUtils.toTrianglesDrawMode() if necessary.")}}}),Ra.prototype.initBones=function(){console.error("THREE.SkinnedMesh: initBones() has been removed.")},Kn.prototype.setLens=function(t,e){console.warn("THREE.PerspectiveCamera.setLens is deprecated. Use .setFocalLength and .filmGauge for a photographic setup."),void 0!==e&&(this.filmGauge=e),this.setFocalLength(t)},Object.defineProperties(pc.prototype,{onlyShadow:{set:function(){console.warn("THREE.Light: .onlyShadow has been removed.")}},shadowCameraFov:{set:function(t){console.warn("THREE.Light: .shadowCameraFov is now .shadow.camera.fov."),this.shadow.camera.fov=t}},shadowCameraLeft:{set:function(t){console.warn("THREE.Light: .shadowCameraLeft is now .shadow.camera.left."),this.shadow.camera.left=t}},shadowCameraRight:{set:function(t){console.warn("THREE.Light: .shadowCameraRight is now .shadow.camera.right."),this.shadow.camera.right=t}},shadowCameraTop:{set:function(t){console.warn("THREE.Light: .shadowCameraTop is now .shadow.camera.top."),this.shadow.camera.top=t}},shadowCameraBottom:{set:function(t){console.warn("THREE.Light: .shadowCameraBottom is now .shadow.camera.bottom."),this.shadow.camera.bottom=t}},shadowCameraNear:{set:function(t){console.warn("THREE.Light: .shadowCameraNear is now .shadow.camera.near."),this.shadow.camera.near=t}},shadowCameraFar:{set:function(t){console.warn("THREE.Light: .shadowCameraFar is now .shadow.camera.far."),this.shadow.camera.far=t}},shadowCameraVisible:{set:function(){console.warn("THREE.Light: .shadowCameraVisible has been removed. Use new THREE.CameraHelper( light.shadow.camera ) instead.")}},shadowBias:{set:function(t){console.warn("THREE.Light: .shadowBias is now .shadow.bias."),this.shadow.bias=t}},shadowDarkness:{set:function(){console.warn("THREE.Light: .shadowDarkness has been removed.")}},shadowMapWidth:{set:function(t){console.warn("THREE.Light: .shadowMapWidth is now .shadow.mapSize.width."),this.shadow.mapSize.width=t}},shadowMapHeight:{set:function(t){console.warn("THREE.Light: .shadowMapHeight is now .shadow.mapSize.height."),this.shadow.mapSize.height=t}}}),Object.defineProperties(ln.prototype,{length:{get:function(){return console.warn("THREE.BufferAttribute: .length has been deprecated. Use .count instead."),this.array.length}},dynamic:{get:function(){return console.warn("THREE.BufferAttribute: .dynamic has been deprecated. Use .usage instead."),this.usage===nt},set:function(){console.warn("THREE.BufferAttribute: .dynamic has been deprecated. Use .usage instead."),this.setUsage(nt)}}}),ln.prototype.setDynamic=function(t){return console.warn("THREE.BufferAttribute: .setDynamic() has been deprecated. Use .setUsage() instead."),this.setUsage(!0===t?nt:et),this},ln.prototype.copyIndicesArray=function(){console.error("THREE.BufferAttribute: .copyIndicesArray() has been removed.")},ln.prototype.setArray=function(){console.error("THREE.BufferAttribute: .setArray has been removed. Use BufferGeometry .setAttribute to replace/resize attribute buffers")},En.prototype.addIndex=function(t){console.warn("THREE.BufferGeometry: .addIndex() has been renamed to .setIndex()."),this.setIndex(t)},En.prototype.addAttribute=function(t,e){return console.warn("THREE.BufferGeometry: .addAttribute() has been renamed to .setAttribute()."),e&&e.isBufferAttribute||e&&e.isInterleavedBufferAttribute?"index"===t?(console.warn("THREE.BufferGeometry.addAttribute: Use .setIndex() for index attribute."),this.setIndex(e),this):this.setAttribute(t,e):(console.warn("THREE.BufferGeometry: .addAttribute() now expects ( name, attribute )."),this.setAttribute(t,new ln(arguments[1],arguments[2])))},En.prototype.addDrawCall=function(t,e,n){void 0!==n&&console.warn("THREE.BufferGeometry: .addDrawCall() no longer supports indexOffset."),console.warn("THREE.BufferGeometry: .addDrawCall() is now .addGroup()."),this.addGroup(t,e)},En.prototype.clearDrawCalls=function(){console.warn("THREE.BufferGeometry: .clearDrawCalls() is now .clearGroups()."),this.clearGroups()},En.prototype.computeOffsets=function(){console.warn("THREE.BufferGeometry: .computeOffsets() has been removed.")},En.prototype.removeAttribute=function(t){return console.warn("THREE.BufferGeometry: .removeAttribute() has been renamed to .deleteAttribute()."),this.deleteAttribute(t)},En.prototype.applyMatrix=function(t){return console.warn("THREE.BufferGeometry: .applyMatrix() has been renamed to .applyMatrix4()."),this.applyMatrix4(t)},Object.defineProperties(En.prototype,{drawcalls:{get:function(){return console.error("THREE.BufferGeometry: .drawcalls has been renamed to .groups."),this.groups}},offsets:{get:function(){return console.warn("THREE.BufferGeometry: .offsets has been renamed to .groups."),this.groups}}}),na.prototype.setDynamic=function(t){return console.warn("THREE.InterleavedBuffer: .setDynamic() has been deprecated. Use .setUsage() instead."),this.setUsage(!0===t?nt:et),this},na.prototype.setArray=function(){console.error("THREE.InterleavedBuffer: .setArray has been removed. Use BufferGeometry .setAttribute to replace/resize attribute buffers")},vl.prototype.getArrays=function(){console.error("THREE.ExtrudeGeometry: .getArrays() has been removed.")},vl.prototype.addShapeList=function(){console.error("THREE.ExtrudeGeometry: .addShapeList() has been removed.")},vl.prototype.addShape=function(){console.error("THREE.ExtrudeGeometry: .addShape() has been removed.")},ea.prototype.dispose=function(){console.error("THREE.Scene: .dispose() has been removed.")},_h.prototype.onUpdate=function(){return console.warn("THREE.Uniform: .onUpdate() has been removed. Use object.onBeforeRender() instead."),this},Object.defineProperties(Ze.prototype,{wrapAround:{get:function(){console.warn("THREE.Material: .wrapAround has been removed.")},set:function(){console.warn("THREE.Material: .wrapAround has been removed.")}},overdraw:{get:function(){console.warn("THREE.Material: .overdraw has been removed.")},set:function(){console.warn("THREE.Material: .overdraw has been removed.")}},wrapRGB:{get:function(){return console.warn("THREE.Material: .wrapRGB has been removed."),new rn}},shading:{get:function(){console.error("THREE."+this.type+": .shading has been removed. Use the boolean .flatShading instead.")},set:function(t){console.warn("THREE."+this.type+": .shading has been removed. Use the boolean .flatShading instead."),this.flatShading=1===t}},stencilMask:{get:function(){return console.warn("THREE."+this.type+": .stencilMask has been removed. Use .stencilFuncMask instead."),this.stencilFuncMask},set:function(t){console.warn("THREE."+this.type+": .stencilMask has been removed. Use .stencilFuncMask instead."),this.stencilFuncMask=t}},vertexTangents:{get:function(){console.warn("THREE."+this.type+": .vertexTangents has been removed.")},set:function(){console.warn("THREE."+this.type+": .vertexTangents has been removed.")}}}),Object.defineProperties(Zn.prototype,{derivatives:{get:function(){return console.warn("THREE.ShaderMaterial: .derivatives has been moved to .extensions.derivatives."),this.extensions.derivatives},set:function(t){console.warn("THREE. ShaderMaterial: .derivatives has been moved to .extensions.derivatives."),this.extensions.derivatives=t}}}),Qs.prototype.clearTarget=function(t,e,n,i){console.warn("THREE.WebGLRenderer: .clearTarget() has been deprecated. Use .setRenderTarget() and .clear() instead."),this.setRenderTarget(t),this.clear(e,n,i)},Qs.prototype.animate=function(t){console.warn("THREE.WebGLRenderer: .animate() is now .setAnimationLoop()."),this.setAnimationLoop(t)},Qs.prototype.getCurrentRenderTarget=function(){return console.warn("THREE.WebGLRenderer: .getCurrentRenderTarget() is now .getRenderTarget()."),this.getRenderTarget()},Qs.prototype.getMaxAnisotropy=function(){return console.warn("THREE.WebGLRenderer: .getMaxAnisotropy() is now .capabilities.getMaxAnisotropy()."),this.capabilities.getMaxAnisotropy()},Qs.prototype.getPrecision=function(){return console.warn("THREE.WebGLRenderer: .getPrecision() is now .capabilities.precision."),this.capabilities.precision},Qs.prototype.resetGLState=function(){return console.warn("THREE.WebGLRenderer: .resetGLState() is now .state.reset()."),this.state.reset()},Qs.prototype.supportsFloatTextures=function(){return console.warn("THREE.WebGLRenderer: .supportsFloatTextures() is now .extensions.get( 'OES_texture_float' )."),this.extensions.get("OES_texture_float")},Qs.prototype.supportsHalfFloatTextures=function(){return console.warn("THREE.WebGLRenderer: .supportsHalfFloatTextures() is now .extensions.get( 'OES_texture_half_float' )."),this.extensions.get("OES_texture_half_float")},Qs.prototype.supportsStandardDerivatives=function(){return console.warn("THREE.WebGLRenderer: .supportsStandardDerivatives() is now .extensions.get( 'OES_standard_derivatives' )."),this.extensions.get("OES_standard_derivatives")},Qs.prototype.supportsCompressedTextureS3TC=function(){return console.warn("THREE.WebGLRenderer: .supportsCompressedTextureS3TC() is now .extensions.get( 'WEBGL_compressed_texture_s3tc' )."),this.extensions.get("WEBGL_compressed_texture_s3tc")},Qs.prototype.supportsCompressedTexturePVRTC=function(){return console.warn("THREE.WebGLRenderer: .supportsCompressedTexturePVRTC() is now .extensions.get( 'WEBGL_compressed_texture_pvrtc' )."),this.extensions.get("WEBGL_compressed_texture_pvrtc")},Qs.prototype.supportsBlendMinMax=function(){return console.warn("THREE.WebGLRenderer: .supportsBlendMinMax() is now .extensions.get( 'EXT_blend_minmax' )."),this.extensions.get("EXT_blend_minmax")},Qs.prototype.supportsVertexTextures=function(){return console.warn("THREE.WebGLRenderer: .supportsVertexTextures() is now .capabilities.vertexTextures."),this.capabilities.vertexTextures},Qs.prototype.supportsInstancedArrays=function(){return console.warn("THREE.WebGLRenderer: .supportsInstancedArrays() is now .extensions.get( 'ANGLE_instanced_arrays' )."),this.extensions.get("ANGLE_instanced_arrays")},Qs.prototype.enableScissorTest=function(t){console.warn("THREE.WebGLRenderer: .enableScissorTest() is now .setScissorTest()."),this.setScissorTest(t)},Qs.prototype.initMaterial=function(){console.warn("THREE.WebGLRenderer: .initMaterial() has been removed.")},Qs.prototype.addPrePlugin=function(){console.warn("THREE.WebGLRenderer: .addPrePlugin() has been removed.")},Qs.prototype.addPostPlugin=function(){console.warn("THREE.WebGLRenderer: .addPostPlugin() has been removed.")},Qs.prototype.updateShadowMap=function(){console.warn("THREE.WebGLRenderer: .updateShadowMap() has been removed.")},Qs.prototype.setFaceCulling=function(){console.warn("THREE.WebGLRenderer: .setFaceCulling() has been removed.")},Qs.prototype.allocTextureUnit=function(){console.warn("THREE.WebGLRenderer: .allocTextureUnit() has been removed.")},Qs.prototype.setTexture=function(){console.warn("THREE.WebGLRenderer: .setTexture() has been removed.")},Qs.prototype.setTexture2D=function(){console.warn("THREE.WebGLRenderer: .setTexture2D() has been removed.")},Qs.prototype.setTextureCube=function(){console.warn("THREE.WebGLRenderer: .setTextureCube() has been removed.")},Qs.prototype.getActiveMipMapLevel=function(){return console.warn("THREE.WebGLRenderer: .getActiveMipMapLevel() is now .getActiveMipmapLevel()."),this.getActiveMipmapLevel()},Object.defineProperties(Qs.prototype,{shadowMapEnabled:{get:function(){return this.shadowMap.enabled},set:function(t){console.warn("THREE.WebGLRenderer: .shadowMapEnabled is now .shadowMap.enabled."),this.shadowMap.enabled=t}},shadowMapType:{get:function(){return this.shadowMap.type},set:function(t){console.warn("THREE.WebGLRenderer: .shadowMapType is now .shadowMap.type."),this.shadowMap.type=t}},shadowMapCullFace:{get:function(){console.warn("THREE.WebGLRenderer: .shadowMapCullFace has been removed. Set Material.shadowSide instead.")},set:function(){console.warn("THREE.WebGLRenderer: .shadowMapCullFace has been removed. Set Material.shadowSide instead.")}},context:{get:function(){return console.warn("THREE.WebGLRenderer: .context has been removed. Use .getContext() instead."),this.getContext()}},vr:{get:function(){return console.warn("THREE.WebGLRenderer: .vr has been renamed to .xr"),this.xr}},gammaInput:{get:function(){return console.warn("THREE.WebGLRenderer: .gammaInput has been removed. Set the encoding for textures via Texture.encoding instead."),!1},set:function(){console.warn("THREE.WebGLRenderer: .gammaInput has been removed. Set the encoding for textures via Texture.encoding instead.")}},gammaOutput:{get:function(){return console.warn("THREE.WebGLRenderer: .gammaOutput has been removed. Set WebGLRenderer.outputEncoding instead."),!1},set:function(t){console.warn("THREE.WebGLRenderer: .gammaOutput has been removed. Set WebGLRenderer.outputEncoding instead."),this.outputEncoding=!0===t?Y:X}},toneMappingWhitePoint:{get:function(){return console.warn("THREE.WebGLRenderer: .toneMappingWhitePoint has been removed."),1},set:function(){console.warn("THREE.WebGLRenderer: .toneMappingWhitePoint has been removed.")}}}),Object.defineProperties(Gs.prototype,{cullFace:{get:function(){console.warn("THREE.WebGLRenderer: .shadowMap.cullFace has been removed. Set Material.shadowSide instead.")},set:function(){console.warn("THREE.WebGLRenderer: .shadowMap.cullFace has been removed. Set Material.shadowSide instead.")}},renderReverseSided:{get:function(){console.warn("THREE.WebGLRenderer: .shadowMap.renderReverseSided has been removed. Set Material.shadowSide instead.")},set:function(){console.warn("THREE.WebGLRenderer: .shadowMap.renderReverseSided has been removed. Set Material.shadowSide instead.")}},renderSingleSided:{get:function(){console.warn("THREE.WebGLRenderer: .shadowMap.renderSingleSided has been removed. Set Material.shadowSide instead.")},set:function(){console.warn("THREE.WebGLRenderer: .shadowMap.renderSingleSided has been removed. Set Material.shadowSide instead.")}}}),Object.defineProperties(Pt.prototype,{wrapS:{get:function(){return console.warn("THREE.WebGLRenderTarget: .wrapS is now .texture.wrapS."),this.texture.wrapS},set:function(t){console.warn("THREE.WebGLRenderTarget: .wrapS is now .texture.wrapS."),this.texture.wrapS=t}},wrapT:{get:function(){return console.warn("THREE.WebGLRenderTarget: .wrapT is now .texture.wrapT."),this.texture.wrapT},set:function(t){console.warn("THREE.WebGLRenderTarget: .wrapT is now .texture.wrapT."),this.texture.wrapT=t}},magFilter:{get:function(){return console.warn("THREE.WebGLRenderTarget: .magFilter is now .texture.magFilter."),this.texture.magFilter},set:function(t){console.warn("THREE.WebGLRenderTarget: .magFilter is now .texture.magFilter."),this.texture.magFilter=t}},minFilter:{get:function(){return console.warn("THREE.WebGLRenderTarget: .minFilter is now .texture.minFilter."),this.texture.minFilter},set:function(t){console.warn("THREE.WebGLRenderTarget: .minFilter is now .texture.minFilter."),this.texture.minFilter=t}},anisotropy:{get:function(){return console.warn("THREE.WebGLRenderTarget: .anisotropy is now .texture.anisotropy."),this.texture.anisotropy},set:function(t){console.warn("THREE.WebGLRenderTarget: .anisotropy is now .texture.anisotropy."),this.texture.anisotropy=t}},offset:{get:function(){return console.warn("THREE.WebGLRenderTarget: .offset is now .texture.offset."),this.texture.offset},set:function(t){console.warn("THREE.WebGLRenderTarget: .offset is now .texture.offset."),this.texture.offset=t}},repeat:{get:function(){return console.warn("THREE.WebGLRenderTarget: .repeat is now .texture.repeat."),this.texture.repeat},set:function(t){console.warn("THREE.WebGLRenderTarget: .repeat is now .texture.repeat."),this.texture.repeat=t}},format:{get:function(){return console.warn("THREE.WebGLRenderTarget: .format is now .texture.format."),this.texture.format},set:function(t){console.warn("THREE.WebGLRenderTarget: .format is now .texture.format."),this.texture.format=t}},type:{get:function(){return console.warn("THREE.WebGLRenderTarget: .type is now .texture.type."),this.texture.type},set:function(t){console.warn("THREE.WebGLRenderTarget: .type is now .texture.type."),this.texture.type=t}},generateMipmaps:{get:function(){return console.warn("THREE.WebGLRenderTarget: .generateMipmaps is now .texture.generateMipmaps."),this.texture.generateMipmaps},set:function(t){console.warn("THREE.WebGLRenderTarget: .generateMipmaps is now .texture.generateMipmaps."),this.texture.generateMipmaps=t}}}),$c.prototype.load=function(t){console.warn("THREE.Audio: .load has been deprecated. Use THREE.AudioLoader instead.");const e=this;return(new kc).load(t,(function(t){e.setBuffer(t)})),this},rh.prototype.getData=function(){return console.warn("THREE.AudioAnalyser: .getData() is now .getFrequencyData()."),this.getFrequencyData()},ti.prototype.updateCubeMap=function(t,e){return console.warn("THREE.CubeCamera: .updateCubeMap() is now .update()."),this.update(t,e)},ti.prototype.clear=function(t,e,n,i){return console.warn("THREE.CubeCamera: .clear() is now .renderTarget.clear()."),this.renderTarget.clear(t,e,n,i)},Et.crossOrigin=void 0,Et.loadTexture=function(t,e,n,i){console.warn("THREE.ImageUtils.loadTexture has been deprecated. Use THREE.TextureLoader() instead.");const r=new dc;r.setCrossOrigin(this.crossOrigin);const s=r.load(t,n,void 0,i);return e&&(s.mapping=e),s},Et.loadTextureCube=function(t,e,n,i){console.warn("THREE.ImageUtils.loadTextureCube has been deprecated. Use THREE.CubeTextureLoader() instead.");const r=new hc;r.setCrossOrigin(this.crossOrigin);const s=r.load(t,n,void 0,i);return e&&(s.mapping=e),s},Et.loadCompressedTexture=function(){console.error("THREE.ImageUtils.loadCompressedTexture has been removed. Use THREE.DDSLoader instead.")},Et.loadCompressedTextureCube=function(){console.error("THREE.ImageUtils.loadCompressedTextureCube has been removed. Use THREE.DDSLoader instead.")};const tu={createMultiMaterialObject:function(){console.error("THREE.SceneUtils has been moved to /examples/jsm/utils/SceneUtils.js")},detach:function(){console.error("THREE.SceneUtils has been moved to /examples/jsm/utils/SceneUtils.js")},attach:function(){console.error("THREE.SceneUtils has been moved to /examples/jsm/utils/SceneUtils.js")}};"undefined"!=typeof __THREE_DEVTOOLS__&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("register",{detail:{revision:e}})),"undefined"!=typeof window&&(window.__THREE__?console.warn("WARNING: Multiple instances of Three.js being imported."):window.__THREE__=e),t.ACESFilmicToneMapping=4,t.AddEquation=n,t.AddOperation=2,t.AdditiveAnimationBlendMode=q,t.AdditiveBlending=2,t.AlphaFormat=1021,t.AlwaysDepth=1,t.AlwaysStencilFunc=519,t.AmbientLight=Lc,t.AmbientLightProbe=Wc,t.AnimationClip=ec,t.AnimationLoader=class extends ac{constructor(t){super(t)}load(t,e,n,i){const r=this,s=new lc(this.manager);s.setPath(this.path),s.setRequestHeader(this.requestHeader),s.setWithCredentials(this.withCredentials),s.load(t,(function(n){try{e(r.parse(JSON.parse(n)))}catch(e){i?i(e):console.error(e),r.manager.itemError(t)}}),n,i)}parse(t){const e=[];for(let n=0;n<t.length;n++){const i=ec.parse(t[n]);e.push(i)}return e}},t.AnimationMixer=xh,t.AnimationObjectGroup=vh,t.AnimationUtils=kl,t.ArcCurve=bo,t.ArrayCamera=js,t.ArrowHelper=class extends Fe{constructor(t=new zt(0,0,1),e=new zt(0,0,0),n=1,i=16776960,r=.2*n,s=.2*r){super(),this.type="ArrowHelper",void 0===Jh&&(Jh=new En,Jh.setAttribute("position",new vn([0,0,0,0,1,0],3)),Zh=new ho(0,.5,1,5,1),Zh.translate(0,-.5,0)),this.position.copy(e),this.line=new Xa(Jh,new Ga({color:i,toneMapped:!1})),this.line.matrixAutoUpdate=!1,this.add(this.line),this.cone=new Wn(Zh,new sn({color:i,toneMapped:!1})),this.cone.matrixAutoUpdate=!1,this.add(this.cone),this.setDirection(t),this.setLength(n,r,s)}setDirection(t){if(t.y>.99999)this.quaternion.set(0,0,0,1);else if(t.y<-.99999)this.quaternion.set(1,0,0,0);else{Yh.set(t.z,0,-t.x).normalize();const e=Math.acos(t.y);this.quaternion.setFromAxisAngle(Yh,e)}}setLength(t,e=.2*t,n=.2*e){this.line.scale.set(1,Math.max(1e-4,t-e),1),this.line.updateMatrix(),this.cone.scale.set(n,e,n),this.cone.position.y=t,this.cone.updateMatrix()}setColor(t){this.line.material.color.set(t),this.cone.material.color.set(t)}copy(t){return super.copy(t,!1),this.line.copy(t.line),this.cone.copy(t.cone),this}},t.Audio=$c,t.AudioAnalyser=rh,t.AudioContext=Gc,t.AudioListener=class extends Fe{constructor(){super(),this.type="AudioListener",this.context=Gc.getContext(),this.gain=this.context.createGain(),this.gain.connect(this.context.destination),this.filter=null,this.timeDelta=0,this._clock=new Xc}getInput(){return this.gain}removeFilter(){return null!==this.filter&&(this.gain.disconnect(this.filter),this.filter.disconnect(this.context.destination),this.gain.connect(this.context.destination),this.filter=null),this}getFilter(){return this.filter}setFilter(t){return null!==this.filter?(this.gain.disconnect(this.filter),this.filter.disconnect(this.context.destination)):this.gain.disconnect(this.context.destination),this.filter=t,this.gain.connect(this.filter),this.filter.connect(this.context.destination),this}getMasterVolume(){return this.gain.gain.value}setMasterVolume(t){return this.gain.gain.setTargetAtTime(t,this.context.currentTime,.01),this}updateMatrixWorld(t){super.updateMatrixWorld(t);const e=this.context.listener,n=this.up;if(this.timeDelta=this._clock.getDelta(),this.matrixWorld.decompose(Jc,Zc,Qc),Kc.set(0,0,-1).applyQuaternion(Zc),e.positionX){const t=this.context.currentTime+this.timeDelta;e.positionX.linearRampToValueAtTime(Jc.x,t),e.positionY.linearRampToValueAtTime(Jc.y,t),e.positionZ.linearRampToValueAtTime(Jc.z,t),e.forwardX.linearRampToValueAtTime(Kc.x,t),e.forwardY.linearRampToValueAtTime(Kc.y,t),e.forwardZ.linearRampToValueAtTime(Kc.z,t),e.upX.linearRampToValueAtTime(n.x,t),e.upY.linearRampToValueAtTime(n.y,t),e.upZ.linearRampToValueAtTime(n.z,t)}else e.setPosition(Jc.x,Jc.y,Jc.z),e.setOrientation(Kc.x,Kc.y,Kc.z,n.x,n.y,n.z)}},t.AudioLoader=kc,t.AxesHelper=Qh,t.AxisHelper=function(t){return console.warn("THREE.AxisHelper has been renamed to THREE.AxesHelper."),new Qh(t)},t.BackSide=1,t.BasicDepthPacking=3200,t.BasicShadowMap=0,t.BinaryTextureLoader=function(t){return console.warn("THREE.BinaryTextureLoader has been renamed to THREE.DataTextureLoader."),new uc(t)},t.Bone=Ca,t.BooleanKeyframeTrack=Yl,t.BoundingBoxHelper=function(t,e){return console.warn("THREE.BoundingBoxHelper has been deprecated. Creating a THREE.BoxHelper instead."),new Xh(t,e)},t.Box2=Eh,t.Box3=Ot,t.Box3Helper=class extends Za{constructor(t,e=16776960){const n=new Uint16Array([0,1,1,2,2,3,3,0,4,5,5,6,6,7,7,4,0,4,1,5,2,6,3,7]),i=new En;i.setIndex(new ln(n,1)),i.setAttribute("position",new vn([1,1,1,-1,1,1,-1,-1,1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1],3)),super(i,new Ga({color:e,toneMapped:!1})),this.box=t,this.type="Box3Helper",this.geometry.computeBoundingSphere()}updateMatrixWorld(t){const e=this.box;e.isEmpty()||(e.getCenter(this.position),e.getSize(this.scale),this.scale.multiplyScalar(.5),super.updateMatrixWorld(t))}},t.BoxBufferGeometry=qn,t.BoxGeometry=qn,t.BoxHelper=Xh,t.BufferAttribute=ln,t.BufferGeometry=En,t.BufferGeometryLoader=zc,t.ByteType=1010,t.Cache=ic,t.Camera=Qn,t.CameraHelper=class extends Za{constructor(t){const e=new En,n=new Ga({color:16777215,vertexColors:!0,toneMapped:!1}),i=[],r=[],s={},a=new rn(16755200),o=new rn(16711680),l=new rn(43775),c=new rn(16777215),h=new rn(3355443);function u(t,e,n){d(t,n),d(e,n)}function d(t,e){i.push(0,0,0),r.push(e.r,e.g,e.b),void 0===s[t]&&(s[t]=[]),s[t].push(i.length/3-1)}u("n1","n2",a),u("n2","n4",a),u("n4","n3",a),u("n3","n1",a),u("f1","f2",a),u("f2","f4",a),u("f4","f3",a),u("f3","f1",a),u("n1","f1",a),u("n2","f2",a),u("n3","f3",a),u("n4","f4",a),u("p","n1",o),u("p","n2",o),u("p","n3",o),u("p","n4",o),u("u1","u2",l),u("u2","u3",l),u("u3","u1",l),u("c","t",c),u("p","c",h),u("cn1","cn2",h),u("cn3","cn4",h),u("cf1","cf2",h),u("cf3","cf4",h),e.setAttribute("position",new vn(i,3)),e.setAttribute("color",new vn(r,3)),super(e,n),this.type="CameraHelper",this.camera=t,this.camera.updateProjectionMatrix&&this.camera.updateProjectionMatrix(),this.matrix=t.matrixWorld,this.matrixAutoUpdate=!1,this.pointMap=s,this.update()}update(){const t=this.geometry,e=this.pointMap;Wh.projectionMatrixInverse.copy(this.camera.projectionMatrixInverse),jh("c",e,t,Wh,0,0,-1),jh("t",e,t,Wh,0,0,1),jh("n1",e,t,Wh,-1,-1,-1),jh("n2",e,t,Wh,1,-1,-1),jh("n3",e,t,Wh,-1,1,-1),jh("n4",e,t,Wh,1,1,-1),jh("f1",e,t,Wh,-1,-1,1),jh("f2",e,t,Wh,1,-1,1),jh("f3",e,t,Wh,-1,1,1),jh("f4",e,t,Wh,1,1,1),jh("u1",e,t,Wh,.7,1.1,-1),jh("u2",e,t,Wh,-.7,1.1,-1),jh("u3",e,t,Wh,0,2,-1),jh("cf1",e,t,Wh,-1,0,1),jh("cf2",e,t,Wh,1,0,1),jh("cf3",e,t,Wh,0,-1,1),jh("cf4",e,t,Wh,0,1,1),jh("cn1",e,t,Wh,-1,0,-1),jh("cn2",e,t,Wh,1,0,-1),jh("cn3",e,t,Wh,0,-1,-1),jh("cn4",e,t,Wh,0,1,-1),t.getAttribute("position").needsUpdate=!0}dispose(){this.geometry.dispose(),this.material.dispose()}},t.CanvasRenderer=function(){console.error("THREE.CanvasRenderer has been removed")},t.CanvasTexture=oo,t.CatmullRomCurve3=Lo,t.CineonToneMapping=3,t.CircleBufferGeometry=co,t.CircleGeometry=co,t.ClampToEdgeWrapping=u,t.Clock=Xc,t.Color=rn,t.ColorKeyframeTrack=Jl,t.CompressedTexture=ao,t.CompressedTextureLoader=class extends ac{constructor(t){super(t)}load(t,e,n,i){const r=this,s=[],a=new ao,o=new lc(this.manager);o.setPath(this.path),o.setResponseType("arraybuffer"),o.setRequestHeader(this.requestHeader),o.setWithCredentials(r.withCredentials);let l=0;function c(c){o.load(t[c],(function(t){const n=r.parse(t,!0);s[c]={width:n.width,height:n.height,format:n.format,mipmaps:n.mipmaps},l+=1,6===l&&(1===n.mipmapCount&&(a.minFilter=g),a.image=s,a.format=n.format,a.needsUpdate=!0,e&&e(a))}),n,i)}if(Array.isArray(t))for(let e=0,n=t.length;e<n;++e)c(e);else o.load(t,(function(t){const n=r.parse(t,!0);if(n.isCubemap){const t=n.mipmaps.length/n.mipmapCount;for(let e=0;e<t;e++){s[e]={mipmaps:[]};for(let t=0;t<n.mipmapCount;t++)s[e].mipmaps.push(n.mipmaps[e*n.mipmapCount+t]),s[e].format=n.format,s[e].width=n.width,s[e].height=n.height}a.image=s}else a.image.width=n.width,a.image.height=n.height,a.mipmaps=n.mipmaps;1===n.mipmapCount&&(a.minFilter=g),a.format=n.format,a.needsUpdate=!0,e&&e(a)}),n,i);return a}},t.ConeBufferGeometry=uo,t.ConeGeometry=uo,t.CubeCamera=ti,t.CubeReflectionMapping=r,t.CubeRefractionMapping=s,t.CubeTexture=ei,t.CubeTextureLoader=hc,t.CubeUVReflectionMapping=l,t.CubeUVRefractionMapping=c,t.CubicBezierCurve=Io,t.CubicBezierCurve3=Do,t.CubicInterpolant=Wl,t.CullFaceBack=1,t.CullFaceFront=2,t.CullFaceFrontBack=3,t.CullFaceNone=0,t.Curve=_o,t.CurvePath=Ho,t.CustomBlending=5,t.CustomToneMapping=5,t.CylinderBufferGeometry=ho,t.CylinderGeometry=ho,t.Cylindrical=class{constructor(t=1,e=0,n=0){return this.radius=t,this.theta=e,this.y=n,this}set(t,e,n){return this.radius=t,this.theta=e,this.y=n,this}copy(t){return this.radius=t.radius,this.theta=t.theta,this.y=t.y,this}setFromVector3(t){return this.setFromCartesianCoords(t.x,t.y,t.z)}setFromCartesianCoords(t,e,n){return this.radius=Math.sqrt(t*t+n*n),this.theta=Math.atan2(t,n),this.y=e,this}clone(){return(new this.constructor).copy(this)}},t.DataTexture=Pa,t.DataTexture2DArray=Ki,t.DataTexture3D=rr,t.DataTextureLoader=uc,t.DataUtils=class{static toHalfFloat(t){t>65504&&(console.warn("THREE.DataUtils.toHalfFloat(): value exceeds 65504."),t=65504),Kh[0]=t;const e=$h[0];let n=e>>16&32768,i=e>>12&2047;const r=e>>23&255;return r<103?n:r>142?(n|=31744,n|=(255==r?0:1)&&8388607&e,n):r<113?(i|=2048,n|=(i>>114-r)+(i>>113-r&1),n):(n|=r-112<<10|i>>1,n+=1&i,n)}},t.DecrementStencilOp=7683,t.DecrementWrapStencilOp=34056,t.DefaultLoadingManager=sc,t.DepthFormat=A,t.DepthStencilFormat=L,t.DepthTexture=lo,t.DirectionalLight=Ac,t.DirectionalLightHelper=class extends Fe{constructor(t,e,n){super(),this.light=t,this.light.updateMatrixWorld(),this.matrix=t.matrixWorld,this.matrixAutoUpdate=!1,this.color=n,void 0===e&&(e=1);let i=new En;i.setAttribute("position",new vn([-e,e,0,e,e,0,e,-e,0,-e,-e,0,-e,e,0],3));const r=new Ga({fog:!1,toneMapped:!1});this.lightPlane=new Xa(i,r),this.add(this.lightPlane),i=new En,i.setAttribute("position",new vn([0,0,0,0,0,1],3)),this.targetLine=new Xa(i,r),this.add(this.targetLine),this.update()}dispose(){this.lightPlane.geometry.dispose(),this.lightPlane.material.dispose(),this.targetLine.geometry.dispose(),this.targetLine.material.dispose()}update(){Hh.setFromMatrixPosition(this.light.matrixWorld),Gh.setFromMatrixPosition(this.light.target.matrixWorld),kh.subVectors(Gh,Hh),this.lightPlane.lookAt(Gh),void 0!==this.color?(this.lightPlane.material.color.set(this.color),this.targetLine.material.color.set(this.color)):(this.lightPlane.material.color.copy(this.light.color),this.targetLine.material.color.copy(this.light.color)),this.targetLine.lookAt(Gh),this.targetLine.scale.z=kh.length()}},t.DiscreteInterpolant=ql,t.DodecahedronBufferGeometry=mo,t.DodecahedronGeometry=mo,t.DoubleSide=2,t.DstAlphaFactor=206,t.DstColorFactor=208,t.DynamicBufferAttribute=function(t,e){return console.warn("THREE.DynamicBufferAttribute has been removed. Use new THREE.BufferAttribute().setUsage( THREE.DynamicDrawUsage ) instead."),new ln(t,e).setUsage(nt)},t.DynamicCopyUsage=35050,t.DynamicDrawUsage=nt,t.DynamicReadUsage=35049,t.EdgesGeometry=xo,t.EdgesHelper=function(t,e){return console.warn("THREE.EdgesHelper has been removed. Use THREE.EdgesGeometry instead."),new Za(new xo(t.geometry),new Ga({color:void 0!==e?e:16777215}))},t.EllipseCurve=Mo,t.EqualDepth=4,t.EqualStencilFunc=514,t.EquirectangularReflectionMapping=a,t.EquirectangularRefractionMapping=o,t.Euler=be,t.EventDispatcher=rt,t.ExtrudeBufferGeometry=vl,t.ExtrudeGeometry=vl,t.FaceColors=1,t.FileLoader=lc,t.FlatShading=1,t.Float16BufferAttribute=gn,t.Float32Attribute=function(t,e){return console.warn("THREE.Float32Attribute has been removed. Use new THREE.Float32BufferAttribute() instead."),new vn(t,e)},t.Float32BufferAttribute=vn,t.Float64Attribute=function(t,e){return console.warn("THREE.Float64Attribute has been removed. Use new THREE.Float64BufferAttribute() instead."),new yn(t,e)},t.Float64BufferAttribute=yn,t.FloatType=b,t.Fog=ta,t.FogExp2=$s,t.Font=function(){console.error("THREE.Font has been moved to /examples/jsm/loaders/FontLoader.js")},t.FontLoader=function(){console.error("THREE.FontLoader has been moved to /examples/jsm/loaders/FontLoader.js")},t.FrontSide=0,t.Frustum=ci,t.GLBufferAttribute=bh,t.GLSL1="100",t.GLSL3=it,t.GammaEncoding=J,t.GreaterDepth=6,t.GreaterEqualDepth=5,t.GreaterEqualStencilFunc=518,t.GreaterStencilFunc=516,t.GridHelper=Uh,t.Group=qs,t.HalfFloatType=w,t.HemisphereLight=mc,t.HemisphereLightHelper=class extends Fe{constructor(t,e,n){super(),this.light=t,this.light.updateMatrixWorld(),this.matrix=t.matrixWorld,this.matrixAutoUpdate=!1,this.color=n;const i=new Ml(e);i.rotateY(.5*Math.PI),this.material=new sn({wireframe:!0,fog:!1,toneMapped:!1}),void 0===this.color&&(this.material.vertexColors=!0);const r=i.getAttribute("position"),s=new Float32Array(3*r.count);i.setAttribute("color",new ln(s,3)),this.add(new Wn(i,this.material)),this.update()}dispose(){this.children[0].geometry.dispose(),this.children[0].material.dispose()}update(){const t=this.children[0];if(void 0!==this.color)this.material.color.set(this.color);else{const e=t.geometry.getAttribute("color");Fh.copy(this.light.color),Oh.copy(this.light.groundColor);for(let t=0,n=e.count;t<n;t++){const i=t<n/2?Fh:Oh;e.setXYZ(t,i.r,i.g,i.b)}e.needsUpdate=!0}t.lookAt(Bh.setFromMatrixPosition(this.light.matrixWorld).negate())}},t.HemisphereLightProbe=Vc,t.IcosahedronBufferGeometry=xl,t.IcosahedronGeometry=xl,t.ImageBitmapLoader=Uc,t.ImageLoader=cc,t.ImageUtils=Et,t.ImmediateRenderObject=function(){console.error("THREE.ImmediateRenderObject has been removed.")},t.IncrementStencilOp=7682,t.IncrementWrapStencilOp=34055,t.InstancedBufferAttribute=za,t.InstancedBufferGeometry=Nc,t.InstancedInterleavedBuffer=Mh,t.InstancedMesh=Ha,t.Int16Attribute=function(t,e){return console.warn("THREE.Int16Attribute has been removed. Use new THREE.Int16BufferAttribute() instead."),new dn(t,e)},t.Int16BufferAttribute=dn,t.Int32Attribute=function(t,e){return console.warn("THREE.Int32Attribute has been removed. Use new THREE.Int32BufferAttribute() instead."),new mn(t,e)},t.Int32BufferAttribute=mn,t.Int8Attribute=function(t,e){return console.warn("THREE.Int8Attribute has been removed. Use new THREE.Int8BufferAttribute() instead."),new cn(t,e)},t.Int8BufferAttribute=cn,t.IntType=1013,t.InterleavedBuffer=na,t.InterleavedBufferAttribute=ra,t.Interpolant=Vl,t.InterpolateDiscrete=U,t.InterpolateLinear=H,t.InterpolateSmooth=G,t.InvertStencilOp=5386,t.JSONLoader=function(){console.error("THREE.JSONLoader has been removed.")},t.KeepStencilOp=tt,t.KeyframeTrack=Xl,t.LOD=wa,t.LatheBufferGeometry=_l,t.LatheGeometry=_l,t.Layers=we,t.LensFlare=function(){console.error("THREE.LensFlare has been moved to /examples/jsm/objects/Lensflare.js")},t.LessDepth=2,t.LessEqualDepth=3,t.LessEqualStencilFunc=515,t.LessStencilFunc=513,t.Light=pc,t.LightProbe=Pc,t.Line=Xa,t.Line3=Rh,t.LineBasicMaterial=Ga,t.LineCurve=No,t.LineCurve3=zo,t.LineDashedMaterial=Hl,t.LineLoop=Qa,t.LinePieces=1,t.LineSegments=Za,t.LineStrip=0,t.LinearEncoding=X,t.LinearFilter=g,t.LinearInterpolant=jl,t.LinearMipMapLinearFilter=1008,t.LinearMipMapNearestFilter=1007,t.LinearMipmapLinearFilter=y,t.LinearMipmapNearestFilter=v,t.LinearToneMapping=1,t.Loader=ac,t.LoaderUtils=Dc,t.LoadingManager=rc,t.LogLuvEncoding=3003,t.LoopOnce=2200,t.LoopPingPong=2202,t.LoopRepeat=2201,t.LuminanceAlphaFormat=1025,t.LuminanceFormat=1024,t.MOUSE={LEFT:0,MIDDLE:1,RIGHT:2,ROTATE:0,DOLLY:1,PAN:2},t.Material=Ze,t.MaterialLoader=Ic,t.Math=vt,t.MathUtils=vt,t.Matrix3=xt,t.Matrix4=de,t.MaxEquation=104,t.Mesh=Wn,t.MeshBasicMaterial=sn,t.MeshDepthMaterial=Us,t.MeshDistanceMaterial=Hs,t.MeshFaceMaterial=function(t){return console.warn("THREE.MeshFaceMaterial has been removed. Use an Array instead."),t},t.MeshLambertMaterial=Ol,t.MeshMatcapMaterial=Ul,t.MeshNormalMaterial=Fl,t.MeshPhongMaterial=zl,t.MeshPhysicalMaterial=Nl,t.MeshStandardMaterial=Dl,t.MeshToonMaterial=Bl,t.MinEquation=103,t.MirroredRepeatWrapping=d,t.MixOperation=1,t.MultiMaterial=function(t=[]){return console.warn("THREE.MultiMaterial has been removed. Use an Array instead."),t.isMultiMaterial=!0,t.materials=t,t.clone=function(){return t.slice()},t},t.MultiplyBlending=4,t.MultiplyOperation=0,t.NearestFilter=p,t.NearestMipMapLinearFilter=1005,t.NearestMipMapNearestFilter=1004,t.NearestMipmapLinearFilter=f,t.NearestMipmapNearestFilter=m,t.NeverDepth=0,t.NeverStencilFunc=512,t.NoBlending=0,t.NoColors=0,t.NoToneMapping=0,t.NormalAnimationBlendMode=j,t.NormalBlending=1,t.NotEqualDepth=7,t.NotEqualStencilFunc=517,t.NumberKeyframeTrack=Zl,t.Object3D=Fe,t.ObjectLoader=class extends ac{constructor(t){super(t)}load(t,e,n,i){const r=this,s=""===this.path?Dc.extractUrlBase(t):this.path;this.resourcePath=this.resourcePath||s;const a=new lc(this.manager);a.setPath(this.path),a.setRequestHeader(this.requestHeader),a.setWithCredentials(this.withCredentials),a.load(t,(function(n){let s=null;try{s=JSON.parse(n)}catch(e){return void 0!==i&&i(e),void console.error("THREE:ObjectLoader: Can't parse "+t+".",e.message)}const a=s.metadata;void 0!==a&&void 0!==a.type&&"geometry"!==a.type.toLowerCase()?r.parse(s,e):console.error("THREE.ObjectLoader: Can't load "+t)}),n,i)}async loadAsync(t,e){const n=""===this.path?Dc.extractUrlBase(t):this.path;this.resourcePath=this.resourcePath||n;const i=new lc(this.manager);i.setPath(this.path),i.setRequestHeader(this.requestHeader),i.setWithCredentials(this.withCredentials);const r=await i.loadAsync(t,e),s=JSON.parse(r),a=s.metadata;if(void 0===a||void 0===a.type||"geometry"===a.type.toLowerCase())throw new Error("THREE.ObjectLoader: Can't load "+t);return await this.parseAsync(s)}parse(t,e){const n=this.parseAnimations(t.animations),i=this.parseShapes(t.shapes),r=this.parseGeometries(t.geometries,i),s=this.parseImages(t.images,(function(){void 0!==e&&e(l)})),a=this.parseTextures(t.textures,s),o=this.parseMaterials(t.materials,a),l=this.parseObject(t.object,r,o,a,n),c=this.parseSkeletons(t.skeletons,l);if(this.bindSkeletons(l,c),void 0!==e){let t=!1;for(const e in s)if(s[e]instanceof HTMLImageElement){t=!0;break}!1===t&&e(l)}return l}async parseAsync(t){const e=this.parseAnimations(t.animations),n=this.parseShapes(t.shapes),i=this.parseGeometries(t.geometries,n),r=await this.parseImagesAsync(t.images),s=this.parseTextures(t.textures,r),a=this.parseMaterials(t.materials,s),o=this.parseObject(t.object,i,a,s,e),l=this.parseSkeletons(t.skeletons,o);return this.bindSkeletons(o,l),o}parseShapes(t){const e={};if(void 0!==t)for(let n=0,i=t.length;n<i;n++){const i=(new ko).fromJSON(t[n]);e[i.uuid]=i}return e}parseSkeletons(t,e){const n={},i={};if(e.traverse((function(t){t.isBone&&(i[t.uuid]=t)})),void 0!==t)for(let e=0,r=t.length;e<r;e++){const r=(new Na).fromJSON(t[e],i);n[r.uuid]=r}return n}parseGeometries(t,e){const n={};if(void 0!==t){const i=new zc;for(let r=0,s=t.length;r<s;r++){let s;const a=t[r];switch(a.type){case"BufferGeometry":case"InstancedBufferGeometry":s=i.parse(a);break;case"Geometry":console.error("THREE.ObjectLoader: The legacy Geometry type is no longer supported.");break;default:a.type in Pl?s=Pl[a.type].fromJSON(a,e):console.warn(`THREE.ObjectLoader: Unsupported geometry type "${a.type}"`)}s.uuid=a.uuid,void 0!==a.name&&(s.name=a.name),!0===s.isBufferGeometry&&void 0!==a.userData&&(s.userData=a.userData),n[a.uuid]=s}}return n}parseMaterials(t,e){const n={},i={};if(void 0!==t){const r=new Ic;r.setTextures(e);for(let e=0,s=t.length;e<s;e++){const s=t[e];if("MultiMaterial"===s.type){const t=[];for(let e=0;e<s.materials.length;e++){const i=s.materials[e];void 0===n[i.uuid]&&(n[i.uuid]=r.parse(i)),t.push(n[i.uuid])}i[s.uuid]=t}else void 0===n[s.uuid]&&(n[s.uuid]=r.parse(s)),i[s.uuid]=n[s.uuid]}}return i}parseAnimations(t){const e={};if(void 0!==t)for(let n=0;n<t.length;n++){const i=t[n],r=ec.parse(i);e[r.uuid]=r}return e}parseImages(t,e){const n=this,i={};let r;function s(t){if("string"==typeof t){const e=t;return function(t){return n.manager.itemStart(t),r.load(t,(function(){n.manager.itemEnd(t)}),void 0,(function(){n.manager.itemError(t),n.manager.itemEnd(t)}))}(/^(\/\/)|([a-z]+:(\/\/)?)/i.test(e)?e:n.resourcePath+e)}return t.data?{data:bt(t.type,t.data),width:t.width,height:t.height}:null}if(void 0!==t&&t.length>0){const n=new rc(e);r=new cc(n),r.setCrossOrigin(this.crossOrigin);for(let e=0,n=t.length;e<n;e++){const n=t[e],r=n.url;if(Array.isArray(r)){i[n.uuid]=[];for(let t=0,e=r.length;t<e;t++){const e=s(r[t]);null!==e&&(e instanceof HTMLImageElement?i[n.uuid].push(e):i[n.uuid].push(new Pa(e.data,e.width,e.height)))}}else{const t=s(n.url);null!==t&&(i[n.uuid]=t)}}}return i}async parseImagesAsync(t){const e=this,n={};let i;async function r(t){if("string"==typeof t){const n=t,r=/^(\/\/)|([a-z]+:(\/\/)?)/i.test(n)?n:e.resourcePath+n;return await i.loadAsync(r)}return t.data?{data:bt(t.type,t.data),width:t.width,height:t.height}:null}if(void 0!==t&&t.length>0){i=new cc(this.manager),i.setCrossOrigin(this.crossOrigin);for(let e=0,i=t.length;e<i;e++){const i=t[e],s=i.url;if(Array.isArray(s)){n[i.uuid]=[];for(let t=0,e=s.length;t<e;t++){const e=s[t],a=await r(e);null!==a&&(a instanceof HTMLImageElement?n[i.uuid].push(a):n[i.uuid].push(new Pa(a.data,a.width,a.height)))}}else{const t=await r(i.url);null!==t&&(n[i.uuid]=t)}}}return n}parseTextures(t,e){function n(t,e){return"number"==typeof t?t:(console.warn("THREE.ObjectLoader.parseTexture: Constant should be in numeric form.",t),e[t])}const i={};if(void 0!==t)for(let r=0,s=t.length;r<s;r++){const s=t[r];let a;void 0===s.image&&console.warn('THREE.ObjectLoader: No "image" specified for',s.uuid),void 0===e[s.image]&&console.warn("THREE.ObjectLoader: Undefined image",s.image);const o=e[s.image];Array.isArray(o)?(a=new ei(o),6===o.length&&(a.needsUpdate=!0)):(a=o&&o.data?new Pa(o.data,o.width,o.height):new Lt(o),o&&(a.needsUpdate=!0)),a.uuid=s.uuid,void 0!==s.name&&(a.name=s.name),void 0!==s.mapping&&(a.mapping=n(s.mapping,Bc)),void 0!==s.offset&&a.offset.fromArray(s.offset),void 0!==s.repeat&&a.repeat.fromArray(s.repeat),void 0!==s.center&&a.center.fromArray(s.center),void 0!==s.rotation&&(a.rotation=s.rotation),void 0!==s.wrap&&(a.wrapS=n(s.wrap[0],Fc),a.wrapT=n(s.wrap[1],Fc)),void 0!==s.format&&(a.format=s.format),void 0!==s.type&&(a.type=s.type),void 0!==s.encoding&&(a.encoding=s.encoding),void 0!==s.minFilter&&(a.minFilter=n(s.minFilter,Oc)),void 0!==s.magFilter&&(a.magFilter=n(s.magFilter,Oc)),void 0!==s.anisotropy&&(a.anisotropy=s.anisotropy),void 0!==s.flipY&&(a.flipY=s.flipY),void 0!==s.premultiplyAlpha&&(a.premultiplyAlpha=s.premultiplyAlpha),void 0!==s.unpackAlignment&&(a.unpackAlignment=s.unpackAlignment),void 0!==s.userData&&(a.userData=s.userData),i[s.uuid]=a}return i}parseObject(t,e,n,i,r){let s,a,o;function l(t){return void 0===e[t]&&console.warn("THREE.ObjectLoader: Undefined geometry",t),e[t]}function c(t){if(void 0!==t){if(Array.isArray(t)){const e=[];for(let i=0,r=t.length;i<r;i++){const r=t[i];void 0===n[r]&&console.warn("THREE.ObjectLoader: Undefined material",r),e.push(n[r])}return e}return void 0===n[t]&&console.warn("THREE.ObjectLoader: Undefined material",t),n[t]}}function h(t){return void 0===i[t]&&console.warn("THREE.ObjectLoader: Undefined texture",t),i[t]}switch(t.type){case"Scene":s=new ea,void 0!==t.background&&(Number.isInteger(t.background)?s.background=new rn(t.background):s.background=h(t.background)),void 0!==t.environment&&(s.environment=h(t.environment)),void 0!==t.fog&&("Fog"===t.fog.type?s.fog=new ta(t.fog.color,t.fog.near,t.fog.far):"FogExp2"===t.fog.type&&(s.fog=new $s(t.fog.color,t.fog.density)));break;case"PerspectiveCamera":s=new Kn(t.fov,t.aspect,t.near,t.far),void 0!==t.focus&&(s.focus=t.focus),void 0!==t.zoom&&(s.zoom=t.zoom),void 0!==t.filmGauge&&(s.filmGauge=t.filmGauge),void 0!==t.filmOffset&&(s.filmOffset=t.filmOffset),void 0!==t.view&&(s.view=Object.assign({},t.view));break;case"OrthographicCamera":s=new bi(t.left,t.right,t.top,t.bottom,t.near,t.far),void 0!==t.zoom&&(s.zoom=t.zoom),void 0!==t.view&&(s.view=Object.assign({},t.view));break;case"AmbientLight":s=new Lc(t.color,t.intensity);break;case"DirectionalLight":s=new Ac(t.color,t.intensity);break;case"PointLight":s=new Tc(t.color,t.intensity,t.distance,t.decay);break;case"RectAreaLight":s=new Rc(t.color,t.intensity,t.width,t.height);break;case"SpotLight":s=new _c(t.color,t.intensity,t.distance,t.angle,t.penumbra,t.decay);break;case"HemisphereLight":s=new mc(t.color,t.groundColor,t.intensity);break;case"LightProbe":s=(new Pc).fromJSON(t);break;case"SkinnedMesh":a=l(t.geometry),o=c(t.material),s=new Ra(a,o),void 0!==t.bindMode&&(s.bindMode=t.bindMode),void 0!==t.bindMatrix&&s.bindMatrix.fromArray(t.bindMatrix),void 0!==t.skeleton&&(s.skeleton=t.skeleton);break;case"Mesh":a=l(t.geometry),o=c(t.material),s=new Wn(a,o);break;case"InstancedMesh":a=l(t.geometry),o=c(t.material);const e=t.count,n=t.instanceMatrix,i=t.instanceColor;s=new Ha(a,o,e),s.instanceMatrix=new za(new Float32Array(n.array),16),void 0!==i&&(s.instanceColor=new za(new Float32Array(i.array),i.itemSize));break;case"LOD":s=new wa;break;case"Line":s=new Xa(l(t.geometry),c(t.material));break;case"LineLoop":s=new Qa(l(t.geometry),c(t.material));break;case"LineSegments":s=new Za(l(t.geometry),c(t.material));break;case"PointCloud":case"Points":s=new io(l(t.geometry),c(t.material));break;case"Sprite":s=new xa(c(t.material));break;case"Group":s=new qs;break;case"Bone":s=new Ca;break;default:s=new Fe}if(s.uuid=t.uuid,void 0!==t.name&&(s.name=t.name),void 0!==t.matrix?(s.matrix.fromArray(t.matrix),void 0!==t.matrixAutoUpdate&&(s.matrixAutoUpdate=t.matrixAutoUpdate),s.matrixAutoUpdate&&s.matrix.decompose(s.position,s.quaternion,s.scale)):(void 0!==t.position&&s.position.fromArray(t.position),void 0!==t.rotation&&s.rotation.fromArray(t.rotation),void 0!==t.quaternion&&s.quaternion.fromArray(t.quaternion),void 0!==t.scale&&s.scale.fromArray(t.scale)),void 0!==t.castShadow&&(s.castShadow=t.castShadow),void 0!==t.receiveShadow&&(s.receiveShadow=t.receiveShadow),t.shadow&&(void 0!==t.shadow.bias&&(s.shadow.bias=t.shadow.bias),void 0!==t.shadow.normalBias&&(s.shadow.normalBias=t.shadow.normalBias),void 0!==t.shadow.radius&&(s.shadow.radius=t.shadow.radius),void 0!==t.shadow.mapSize&&s.shadow.mapSize.fromArray(t.shadow.mapSize),void 0!==t.shadow.camera&&(s.shadow.camera=this.parseObject(t.shadow.camera))),void 0!==t.visible&&(s.visible=t.visible),void 0!==t.frustumCulled&&(s.frustumCulled=t.frustumCulled),void 0!==t.renderOrder&&(s.renderOrder=t.renderOrder),void 0!==t.userData&&(s.userData=t.userData),void 0!==t.layers&&(s.layers.mask=t.layers),void 0!==t.children){const a=t.children;for(let t=0;t<a.length;t++)s.add(this.parseObject(a[t],e,n,i,r))}if(void 0!==t.animations){const e=t.animations;for(let t=0;t<e.length;t++){const n=e[t];s.animations.push(r[n])}}if("LOD"===t.type){void 0!==t.autoUpdate&&(s.autoUpdate=t.autoUpdate);const e=t.levels;for(let t=0;t<e.length;t++){const n=e[t],i=s.getObjectByProperty("uuid",n.object);void 0!==i&&s.addLevel(i,n.distance)}}return s}bindSkeletons(t,e){0!==Object.keys(e).length&&t.traverse((function(t){if(!0===t.isSkinnedMesh&&void 0!==t.skeleton){const n=e[t.skeleton];void 0===n?console.warn("THREE.ObjectLoader: No skeleton found with UUID:",t.skeleton):t.bind(n,t.bindMatrix)}}))}setTexturePath(t){return console.warn("THREE.ObjectLoader: .setTexturePath() has been renamed to .setResourcePath()."),this.setResourcePath(t)}},t.ObjectSpaceNormalMap=1,t.OctahedronBufferGeometry=Ml,t.OctahedronGeometry=Ml,t.OneFactor=201,t.OneMinusDstAlphaFactor=207,t.OneMinusDstColorFactor=209,t.OneMinusSrcAlphaFactor=205,t.OneMinusSrcColorFactor=203,t.OrthographicCamera=bi,t.PCFShadowMap=1,t.PCFSoftShadowMap=2,t.PMREMGenerator=Oi,t.ParametricGeometry=function(){return console.error("THREE.ParametricGeometry has been moved to /examples/jsm/geometries/ParametricGeometry.js"),new En},t.Particle=function(t){return console.warn("THREE.Particle has been renamed to THREE.Sprite."),new xa(t)},t.ParticleBasicMaterial=function(t){return console.warn("THREE.ParticleBasicMaterial has been renamed to THREE.PointsMaterial."),new Ka(t)},t.ParticleSystem=function(t,e){return console.warn("THREE.ParticleSystem has been renamed to THREE.Points."),new io(t,e)},t.ParticleSystemMaterial=function(t){return console.warn("THREE.ParticleSystemMaterial has been renamed to THREE.PointsMaterial."),new Ka(t)},t.Path=Go,t.PerspectiveCamera=Kn,t.Plane=ai,t.PlaneBufferGeometry=di,t.PlaneGeometry=di,t.PlaneHelper=class extends Xa{constructor(t,e=1,n=16776960){const i=n,r=new En;r.setAttribute("position",new vn([1,-1,1,-1,1,1,-1,-1,1,1,1,1,-1,1,1,-1,-1,1,1,-1,1,1,1,1,0,0,1,0,0,0],3)),r.computeBoundingSphere(),super(r,new Ga({color:i,toneMapped:!1})),this.type="PlaneHelper",this.plane=t,this.size=e;const s=new En;s.setAttribute("position",new vn([1,1,1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,1],3)),s.computeBoundingSphere(),this.add(new Wn(s,new sn({color:i,opacity:.2,transparent:!0,depthWrite:!1,toneMapped:!1})))}updateMatrixWorld(t){let e=-this.plane.constant;Math.abs(e)<1e-8&&(e=1e-8),this.scale.set(.5*this.size,.5*this.size,e),this.children[0].material.side=e<0?1:0,this.lookAt(this.plane.normal),super.updateMatrixWorld(t)}},t.PointCloud=function(t,e){return console.warn("THREE.PointCloud has been renamed to THREE.Points."),new io(t,e)},t.PointCloudMaterial=function(t){return console.warn("THREE.PointCloudMaterial has been renamed to THREE.PointsMaterial."),new Ka(t)},t.PointLight=Tc,t.PointLightHelper=class extends Wn{constructor(t,e,n){super(new Sl(e,4,2),new sn({wireframe:!0,fog:!1,toneMapped:!1})),this.light=t,this.light.updateMatrixWorld(),this.color=n,this.type="PointLightHelper",this.matrix=this.light.matrixWorld,this.matrixAutoUpdate=!1,this.update()}dispose(){this.geometry.dispose(),this.material.dispose()}update(){void 0!==this.color?this.material.color.set(this.color):this.material.color.copy(this.light.color)}},t.Points=io,t.PointsMaterial=Ka,t.PolarGridHelper=class extends Za{constructor(t=10,e=16,n=8,i=64,r=4473924,s=8947848){r=new rn(r),s=new rn(s);const a=[],o=[];for(let n=0;n<=e;n++){const i=n/e*(2*Math.PI),l=Math.sin(i)*t,c=Math.cos(i)*t;a.push(0,0,0),a.push(l,0,c);const h=1&n?r:s;o.push(h.r,h.g,h.b),o.push(h.r,h.g,h.b)}for(let e=0;e<=n;e++){const l=1&e?r:s,c=t-t/n*e;for(let t=0;t<i;t++){let e=t/i*(2*Math.PI),n=Math.sin(e)*c,r=Math.cos(e)*c;a.push(n,0,r),o.push(l.r,l.g,l.b),e=(t+1)/i*(2*Math.PI),n=Math.sin(e)*c,r=Math.cos(e)*c,a.push(n,0,r),o.push(l.r,l.g,l.b)}}const l=new En;l.setAttribute("position",new vn(a,3)),l.setAttribute("color",new vn(o,3));super(l,new Ga({vertexColors:!0,toneMapped:!1})),this.type="PolarGridHelper"}},t.PolyhedronBufferGeometry=po,t.PolyhedronGeometry=po,t.PositionalAudio=class extends $c{constructor(t){super(t),this.panner=this.context.createPanner(),this.panner.panningModel="HRTF",this.panner.connect(this.gain)}getOutput(){return this.panner}getRefDistance(){return this.panner.refDistance}setRefDistance(t){return this.panner.refDistance=t,this}getRolloffFactor(){return this.panner.rolloffFactor}setRolloffFactor(t){return this.panner.rolloffFactor=t,this}getDistanceModel(){return this.panner.distanceModel}setDistanceModel(t){return this.panner.distanceModel=t,this}getMaxDistance(){return this.panner.maxDistance}setMaxDistance(t){return this.panner.maxDistance=t,this}setDirectionalCone(t,e,n){return this.panner.coneInnerAngle=t,this.panner.coneOuterAngle=e,this.panner.coneOuterGain=n,this}updateMatrixWorld(t){if(super.updateMatrixWorld(t),!0===this.hasPlaybackControl&&!1===this.isPlaying)return;this.matrixWorld.decompose(th,eh,nh),ih.set(0,0,1).applyQuaternion(eh);const e=this.panner;if(e.positionX){const t=this.context.currentTime+this.listener.timeDelta;e.positionX.linearRampToValueAtTime(th.x,t),e.positionY.linearRampToValueAtTime(th.y,t),e.positionZ.linearRampToValueAtTime(th.z,t),e.orientationX.linearRampToValueAtTime(ih.x,t),e.orientationY.linearRampToValueAtTime(ih.y,t),e.orientationZ.linearRampToValueAtTime(ih.z,t)}else e.setPosition(th.x,th.y,th.z),e.setOrientation(ih.x,ih.y,ih.z)}},t.PropertyBinding=gh,t.PropertyMixer=sh,t.QuadraticBezierCurve=Bo,t.QuadraticBezierCurve3=Fo,t.Quaternion=Nt,t.QuaternionKeyframeTrack=Kl,t.QuaternionLinearInterpolant=Ql,t.REVISION=e,t.RGBADepthPacking=3201,t.RGBAFormat=E,t.RGBAIntegerFormat=1033,t.RGBA_ASTC_10x10_Format=37819,t.RGBA_ASTC_10x5_Format=37816,t.RGBA_ASTC_10x6_Format=37817,t.RGBA_ASTC_10x8_Format=37818,t.RGBA_ASTC_12x10_Format=37820,t.RGBA_ASTC_12x12_Format=37821,t.RGBA_ASTC_4x4_Format=37808,t.RGBA_ASTC_5x4_Format=37809,t.RGBA_ASTC_5x5_Format=37810,t.RGBA_ASTC_6x5_Format=37811,t.RGBA_ASTC_6x6_Format=37812,t.RGBA_ASTC_8x5_Format=37813,t.RGBA_ASTC_8x6_Format=37814,t.RGBA_ASTC_8x8_Format=37815,t.RGBA_BPTC_Format=36492,t.RGBA_ETC2_EAC_Format=O,t.RGBA_PVRTC_2BPPV1_Format=B,t.RGBA_PVRTC_4BPPV1_Format=z,t.RGBA_S3TC_DXT1_Format=C,t.RGBA_S3TC_DXT3_Format=P,t.RGBA_S3TC_DXT5_Format=I,t.RGBDEncoding=$,t.RGBEEncoding=Z,t.RGBEFormat=1023,t.RGBFormat=T,t.RGBIntegerFormat=1032,t.RGBM16Encoding=K,t.RGBM7Encoding=Q,t.RGB_ETC1_Format=36196,t.RGB_ETC2_Format=F,t.RGB_PVRTC_2BPPV1_Format=N,t.RGB_PVRTC_4BPPV1_Format=D,t.RGB_S3TC_DXT1_Format=R,t.RGFormat=1030,t.RGIntegerFormat=1031,t.RawShaderMaterial=wi,t.Ray=ue,t.Raycaster=class{constructor(t,e,n=0,i=1/0){this.ray=new ue(t,e),this.near=n,this.far=i,this.camera=null,this.layers=new we,this.params={Mesh:{},Line:{threshold:1},LOD:{},Points:{threshold:1},Sprite:{}}}set(t,e){this.ray.set(t,e)}setFromCamera(t,e){e&&e.isPerspectiveCamera?(this.ray.origin.setFromMatrixPosition(e.matrixWorld),this.ray.direction.set(t.x,t.y,.5).unproject(e).sub(this.ray.origin).normalize(),this.camera=e):e&&e.isOrthographicCamera?(this.ray.origin.set(t.x,t.y,(e.near+e.far)/(e.near-e.far)).unproject(e),this.ray.direction.set(0,0,-1).transformDirection(e.matrixWorld),this.camera=e):console.error("THREE.Raycaster: Unsupported camera type: "+e.type)}intersectObject(t,e=!0,n=[]){return Sh(t,this,n,e),n.sort(wh),n}intersectObjects(t,e=!0,n=[]){for(let i=0,r=t.length;i<r;i++)Sh(t[i],this,n,e);return n.sort(wh),n}},t.RectAreaLight=Rc,t.RedFormat=1028,t.RedIntegerFormat=1029,t.ReinhardToneMapping=2,t.RepeatWrapping=h,t.ReplaceStencilOp=7681,t.ReverseSubtractEquation=102,t.RingBufferGeometry=bl,t.RingGeometry=bl,t.SRGB8_ALPHA8_ASTC_10x10_Format=37851,t.SRGB8_ALPHA8_ASTC_10x5_Format=37848,t.SRGB8_ALPHA8_ASTC_10x6_Format=37849,t.SRGB8_ALPHA8_ASTC_10x8_Format=37850,t.SRGB8_ALPHA8_ASTC_12x10_Format=37852,t.SRGB8_ALPHA8_ASTC_12x12_Format=37853,t.SRGB8_ALPHA8_ASTC_4x4_Format=37840,t.SRGB8_ALPHA8_ASTC_5x4_Format=37841,t.SRGB8_ALPHA8_ASTC_5x5_Format=37842,t.SRGB8_ALPHA8_ASTC_6x5_Format=37843,t.SRGB8_ALPHA8_ASTC_6x6_Format=37844,t.SRGB8_ALPHA8_ASTC_8x5_Format=37845,t.SRGB8_ALPHA8_ASTC_8x6_Format=37846,t.SRGB8_ALPHA8_ASTC_8x8_Format=37847,t.Scene=ea,t.SceneUtils=tu,t.ShaderChunk=pi,t.ShaderLib=fi,t.ShaderMaterial=Zn,t.ShadowMaterial=Il,t.Shape=ko,t.ShapeBufferGeometry=wl,t.ShapeGeometry=wl,t.ShapePath=class{constructor(){this.type="ShapePath",this.color=new rn,this.subPaths=[],this.currentPath=null}moveTo(t,e){return this.currentPath=new Go,this.subPaths.push(this.currentPath),this.currentPath.moveTo(t,e),this}lineTo(t,e){return this.currentPath.lineTo(t,e),this}quadraticCurveTo(t,e,n,i){return this.currentPath.quadraticCurveTo(t,e,n,i),this}bezierCurveTo(t,e,n,i,r,s){return this.currentPath.bezierCurveTo(t,e,n,i,r,s),this}splineThru(t){return this.currentPath.splineThru(t),this}toShapes(t,e){function n(t){const e=[];for(let n=0,i=t.length;n<i;n++){const i=t[n],r=new ko;r.curves=i.curves,e.push(r)}return e}function i(t,e){const n=e.length;let i=!1;for(let r=n-1,s=0;s<n;r=s++){let n=e[r],a=e[s],o=a.x-n.x,l=a.y-n.y;if(Math.abs(l)>Number.EPSILON){if(l<0&&(n=e[s],o=-o,a=e[r],l=-l),t.y<n.y||t.y>a.y)continue;if(t.y===n.y){if(t.x===n.x)return!0}else{const e=l*(t.x-n.x)-o*(t.y-n.y);if(0===e)return!0;if(e<0)continue;i=!i}}else{if(t.y!==n.y)continue;if(a.x<=t.x&&t.x<=n.x||n.x<=t.x&&t.x<=a.x)return!0}}return i}const r=ml.isClockWise,s=this.subPaths;if(0===s.length)return[];if(!0===e)return n(s);let a,o,l;const c=[];if(1===s.length)return o=s[0],l=new ko,l.curves=o.curves,c.push(l),c;let h=!r(s[0].getPoints());h=t?!h:h;const u=[],d=[];let p,m,f=[],g=0;d[g]=void 0,f[g]=[];for(let e=0,n=s.length;e<n;e++)o=s[e],p=o.getPoints(),a=r(p),a=t?!a:a,a?(!h&&d[g]&&g++,d[g]={s:new ko,p:p},d[g].s.curves=o.curves,h&&g++,f[g]=[]):f[g].push({h:o,p:p[0]});if(!d[0])return n(s);if(d.length>1){let t=!1;const e=[];for(let t=0,e=d.length;t<e;t++)u[t]=[];for(let n=0,r=d.length;n<r;n++){const r=f[n];for(let s=0;s<r.length;s++){const a=r[s];let o=!0;for(let r=0;r<d.length;r++)i(a.p,d[r].p)&&(n!==r&&e.push({froms:n,tos:r,hole:s}),o?(o=!1,u[r].push(a)):t=!0);o&&u[n].push(a)}}e.length>0&&(t||(f=u))}for(let t=0,e=d.length;t<e;t++){l=d[t].s,c.push(l),m=f[t];for(let t=0,e=m.length;t<e;t++)l.holes.push(m[t].h)}return c}},t.ShapeUtils=ml,t.ShortType=1011,t.Skeleton=Na,t.SkeletonHelper=Nh,t.SkinnedMesh=Ra,t.SmoothShading=2,t.Sphere=ie,t.SphereBufferGeometry=Sl,t.SphereGeometry=Sl,t.Spherical=class{constructor(t=1,e=0,n=0){return this.radius=t,this.phi=e,this.theta=n,this}set(t,e,n){return this.radius=t,this.phi=e,this.theta=n,this}copy(t){return this.radius=t.radius,this.phi=t.phi,this.theta=t.theta,this}makeSafe(){const t=1e-6;return this.phi=Math.max(t,Math.min(Math.PI-t,this.phi)),this}setFromVector3(t){return this.setFromCartesianCoords(t.x,t.y,t.z)}setFromCartesianCoords(t,e,n){return this.radius=Math.sqrt(t*t+e*e+n*n),0===this.radius?(this.theta=0,this.phi=0):(this.theta=Math.atan2(t,n),this.phi=Math.acos(ut(e/this.radius,-1,1))),this}clone(){return(new this.constructor).copy(this)}},t.SphericalHarmonics3=Cc,t.SplineCurve=Oo,t.SpotLight=_c,t.SpotLightHelper=class extends Fe{constructor(t,e){super(),this.light=t,this.light.updateMatrixWorld(),this.matrix=t.matrixWorld,this.matrixAutoUpdate=!1,this.color=e;const n=new En,i=[0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,-1,0,1,0,0,0,0,1,1,0,0,0,0,-1,1];for(let t=0,e=1,n=32;t<n;t++,e++){const r=t/n*Math.PI*2,s=e/n*Math.PI*2;i.push(Math.cos(r),Math.sin(r),1,Math.cos(s),Math.sin(s),1)}n.setAttribute("position",new vn(i,3));const r=new Ga({fog:!1,toneMapped:!1});this.cone=new Za(n,r),this.add(this.cone),this.update()}dispose(){this.cone.geometry.dispose(),this.cone.material.dispose()}update(){this.light.updateMatrixWorld();const t=this.light.distance?this.light.distance:1e3,e=t*Math.tan(this.light.angle);this.cone.scale.set(e,e,t),Ch.setFromMatrixPosition(this.light.target.matrixWorld),this.cone.lookAt(Ch),void 0!==this.color?this.cone.material.color.set(this.color):this.cone.material.color.copy(this.light.color)}},t.Sprite=xa,t.SpriteMaterial=sa,t.SrcAlphaFactor=204,t.SrcAlphaSaturateFactor=210,t.SrcColorFactor=202,t.StaticCopyUsage=35046,t.StaticDrawUsage=et,t.StaticReadUsage=35045,t.StereoCamera=class{constructor(){this.type="StereoCamera",this.aspect=1,this.eyeSep=.064,this.cameraL=new Kn,this.cameraL.layers.enable(1),this.cameraL.matrixAutoUpdate=!1,this.cameraR=new Kn,this.cameraR.layers.enable(2),this.cameraR.matrixAutoUpdate=!1,this._cache={focus:null,fov:null,aspect:null,near:null,far:null,zoom:null,eyeSep:null}}update(t){const e=this._cache;if(e.focus!==t.focus||e.fov!==t.fov||e.aspect!==t.aspect*this.aspect||e.near!==t.near||e.far!==t.far||e.zoom!==t.zoom||e.eyeSep!==this.eyeSep){e.focus=t.focus,e.fov=t.fov,e.aspect=t.aspect*this.aspect,e.near=t.near,e.far=t.far,e.zoom=t.zoom,e.eyeSep=this.eyeSep;const n=t.projectionMatrix.clone(),i=e.eyeSep/2,r=i*e.near/e.focus,s=e.near*Math.tan(at*e.fov*.5)/e.zoom;let a,o;qc.elements[12]=-i,jc.elements[12]=i,a=-s*e.aspect+r,o=s*e.aspect+r,n.elements[0]=2*e.near/(o-a),n.elements[8]=(o+a)/(o-a),this.cameraL.projectionMatrix.copy(n),a=-s*e.aspect-r,o=s*e.aspect-r,n.elements[0]=2*e.near/(o-a),n.elements[8]=(o+a)/(o-a),this.cameraR.projectionMatrix.copy(n)}this.cameraL.matrixWorld.copy(t.matrixWorld).multiply(qc),this.cameraR.matrixWorld.copy(t.matrixWorld).multiply(jc)}},t.StreamCopyUsage=35042,t.StreamDrawUsage=35040,t.StreamReadUsage=35041,t.StringKeyframeTrack=$l,t.SubtractEquation=101,t.SubtractiveBlending=3,t.TOUCH={ROTATE:0,PAN:1,DOLLY_PAN:2,DOLLY_ROTATE:3},t.TangentSpaceNormalMap=0,t.TetrahedronBufferGeometry=Tl,t.TetrahedronGeometry=Tl,t.TextGeometry=function(){return console.error("THREE.TextGeometry has been moved to /examples/jsm/geometries/TextGeometry.js"),new En},t.Texture=Lt,t.TextureLoader=dc,t.TorusBufferGeometry=El,t.TorusGeometry=El,t.TorusKnotBufferGeometry=Al,t.TorusKnotGeometry=Al,t.Triangle=Ye,t.TriangleFanDrawMode=2,t.TriangleStripDrawMode=1,t.TrianglesDrawMode=0,t.TubeBufferGeometry=Ll,t.TubeGeometry=Ll,t.UVMapping=i,t.Uint16Attribute=function(t,e){return console.warn("THREE.Uint16Attribute has been removed. Use new THREE.Uint16BufferAttribute() instead."),new pn(t,e)},t.Uint16BufferAttribute=pn,t.Uint32Attribute=function(t,e){return console.warn("THREE.Uint32Attribute has been removed. Use new THREE.Uint32BufferAttribute() instead."),new fn(t,e)},t.Uint32BufferAttribute=fn,t.Uint8Attribute=function(t,e){return console.warn("THREE.Uint8Attribute has been removed. Use new THREE.Uint8BufferAttribute() instead."),new hn(t,e)},t.Uint8BufferAttribute=hn,t.Uint8ClampedAttribute=function(t,e){return console.warn("THREE.Uint8ClampedAttribute has been removed. Use new THREE.Uint8ClampedBufferAttribute() instead."),new un(t,e)},t.Uint8ClampedBufferAttribute=un,t.Uniform=_h,t.UniformsLib=mi,t.UniformsUtils=Jn,t.UnsignedByteType=x,t.UnsignedInt248Type=S,t.UnsignedIntType=M,t.UnsignedShort4444Type=1017,t.UnsignedShort5551Type=1018,t.UnsignedShort565Type=1019,t.UnsignedShortType=_,t.VSMShadowMap=3,t.Vector2=yt,t.Vector3=zt,t.Vector4=Ct,t.VectorKeyframeTrack=tc,t.Vertex=function(t,e,n){return console.warn("THREE.Vertex has been removed. Use THREE.Vector3 instead."),new zt(t,e,n)},t.VertexColors=2,t.VideoTexture=so,t.WebGL1Renderer=Ks,t.WebGLCubeRenderTarget=ni,t.WebGLMultipleRenderTargets=It,t.WebGLMultisampleRenderTarget=Dt,t.WebGLRenderTarget=Pt,t.WebGLRenderTargetCube=function(t,e,n){return console.warn("THREE.WebGLRenderTargetCube( width, height, options ) is now WebGLCubeRenderTarget( size, options )."),new ni(t,n)},t.WebGLRenderer=Qs,t.WebGLUtils=Ws,t.WireframeGeometry=Rl,t.WireframeHelper=function(t,e){return console.warn("THREE.WireframeHelper has been removed. Use THREE.WireframeGeometry instead."),new Za(new Rl(t.geometry),new Ga({color:void 0!==e?e:16777215}))},t.WrapAroundEnding=W,t.XHRLoader=function(t){return console.warn("THREE.XHRLoader has been renamed to THREE.FileLoader."),new lc(t)},t.ZeroCurvatureEnding=k,t.ZeroFactor=200,t.ZeroSlopeEnding=V,t.ZeroStencilOp=0,t.sRGBEncoding=Y,Object.defineProperty(t,"__esModule",{value:!0})}));
  
  var d3=d3||{};
  (function(){function di(a){return function(){return this.matches(a)}}function Mb(a,b){return a<b?-1:a>b?1:a>=b?0:NaN}function Bf(a){1===a.length&&(a=Bn(a));return{left:function(b,c,d,e){null==d&&(d=0);null==e&&(e=b.length);for(;d<e;){var g=d+e>>>1;0>a(b[g],c)?d=g+1:e=g}return d},right:function(b,c,d,e){null==d&&(d=0);null==e&&(e=b.length);for(;d<e;){var g=d+e>>>1;0<a(b[g],c)?e=g:d=g+1}return d}}}function Bn(a){return function(b,c){return Mb(a(b),c)}}function ei(a,b){return[a,b]}function Ab(a){return null===a?
  NaN:+a}function fi(a,b){var c=a.length,d=0,e=-1,g=0,k,m=0;if(null==b)for(;++e<c;){if(!isNaN(k=Ab(a[e]))){var p=k-g;g+=p/++d;m+=p*(k-g)}}else for(;++e<c;)isNaN(k=Ab(b(a[e],e,a)))||(p=k-g,g+=p/++d,m+=p*(k-g));if(1<d)return m/(d-1)}function gi(a,b){return(a=fi(a,b))?Math.sqrt(a):a}function Cf(a,b){var c=a.length,d=-1,e,g,k;if(null==b)for(;++d<c;){if(null!=(e=a[d])&&e>=e)for(g=k=e;++d<c;)null!=(e=a[d])&&(g>e&&(g=e),k<e&&(k=e))}else for(;++d<c;)if(null!=(e=b(a[d],d,a))&&e>=e)for(g=k=e;++d<c;)null!=(e=
  b(a[d],d,a))&&(g>e&&(g=e),k<e&&(k=e));return[g,k]}function Od(a){return function(){return a}}function Cn(a){return a}function Ta(a,b,c){a=+a;b=+b;c=2>(e=arguments.length)?(b=a,a=0,1):3>e?1:+c;for(var d=-1,e=Math.max(0,Math.ceil((b-a)/c))|0,g=Array(e);++d<e;)g[d]=a+d*c;return g}function Df(a,b,c){var d,e=-1,g;b=+b;a=+a;c=+c;if(a===b&&0<c)return[a];if(d=b<a){var k=a;a=b;b=k}if(0===(g=Nc(a,b,c))||!isFinite(g))return[];if(0<g)for(a=Math.ceil(a/g),b=Math.floor(b/g),b=Array(k=Math.ceil(b-a+1));++e<k;)b[e]=
  (a+e)*g;else for(a=Math.floor(a*g),b=Math.ceil(b*g),b=Array(k=Math.ceil(a-b+1));++e<k;)b[e]=(a-e)/g;d&&b.reverse();return b}function Nc(a,b,c){b=(b-a)/Math.max(0,c);a=Math.floor(Math.log(b)/Math.LN10);b/=Math.pow(10,a);return 0<=a?(b>=Ef?10:b>=Ff?5:b>=Gf?2:1)*Math.pow(10,a):-Math.pow(10,-a)/(b>=Ef?10:b>=Ff?5:b>=Gf?2:1)}function Nb(a,b,c){var d=Math.abs(b-a)/Math.max(0,c);c=Math.pow(10,Math.floor(Math.log(d)/Math.LN10));d/=c;d>=Ef?c*=10:d>=Ff?c*=5:d>=Gf&&(c*=2);return b<a?-c:c}function Hf(a){return Math.ceil(Math.log(a.length)/
  Math.LN2)+1}function Oc(a,b,c){null==c&&(c=Ab);if(d=a.length){if(0>=(b=+b)||2>d)return+c(a[0],0,a);if(1<=b)return+c(a[d-1],d-1,a);var d;b*=d-1;d=Math.floor(b);var e=+c(a[d],d,a);a=+c(a[d+1],d+1,a);return e+(a-e)*(b-d)}}function hi(a,b){var c=a.length,d=-1,e,g;if(null==b)for(;++d<c;){if(null!=(e=a[d])&&e>=e)for(g=e;++d<c;)null!=(e=a[d])&&e>g&&(g=e)}else for(;++d<c;)if(null!=(e=b(a[d],d,a))&&e>=e)for(g=e;++d<c;)null!=(e=b(a[d],d,a))&&e>g&&(g=e);return g}function If(a){var b=a.length;var c=-1;for(var d=
  0,e,g;++c<b;)d+=a[c].length;for(e=Array(d);0<=--b;)for(g=a[b],c=g.length;0<=--c;)e[--d]=g[c];return e}function ii(a,b){var c=a.length,d=-1,e,g;if(null==b)for(;++d<c;){if(null!=(e=a[d])&&e>=e)for(g=e;++d<c;)null!=(e=a[d])&&g>e&&(g=e)}else for(;++d<c;)if(null!=(e=b(a[d],d,a))&&e>=e)for(g=e;++d<c;)null!=(e=b(a[d],d,a))&&g>e&&(g=e);return g}function ji(a){if(!(g=a.length))return[];for(var b=-1,c=ii(a,Dn),d=Array(c);++b<c;)for(var e=-1,g,k=d[b]=Array(g);++e<g;)k[e]=a[e][b];return d}function Dn(a){return a.length}
  function En(a){return a}function Fn(a){return"translate("+(a+.5)+",0)"}function Gn(a){return"translate(0,"+(a+.5)+")"}function Hn(a){return function(b){return+a(b)}}function In(a){var b=Math.max(0,a.bandwidth()-1)/2;a.round()&&(b=Math.round(b));return function(c){return+a(c)+b}}function Jn(){return!this.__axis}function Pd(a,b){function c(q){var w=null==e?b.ticks?b.ticks.apply(b,d):b.domain():e,B=null==g?b.tickFormat?b.tickFormat.apply(b,d):En:g,F=Math.max(k,0)+p,J=b.range(),P=+J[0]+.5;J=+J[J.length-
  1]+.5;var x=(b.bandwidth?In:Hn)(b.copy()),y=q.selection?q.selection():q,I=y.selectAll(".domain").data([null]);w=y.selectAll(".tick").data(w,b).order();var Q=w.exit(),V=w.enter().append("g").attr("class","tick"),N=w.select("line"),T=w.select("text");I=I.merge(I.enter().insert("path",".tick").attr("class","domain").attr("stroke","currentColor"));w=w.merge(V);N=N.merge(V.append("line").attr("stroke","currentColor").attr(h+"2",v*k));T=T.merge(V.append("text").attr("fill","currentColor").attr(h,v*F).attr("dy",
  1===a?"0em":3===a?"0.71em":"0.32em"));q!==y&&(I=I.transition(q),w=w.transition(q),N=N.transition(q),T=T.transition(q),Q=Q.transition(q).attr("opacity",1E-6).attr("transform",function(f){return isFinite(f=x(f))?l(f):this.getAttribute("transform")}),V.attr("opacity",1E-6).attr("transform",function(f){var n=this.parentNode.__axis;return l(n&&isFinite(n=n(f))?n:x(f))}));Q.remove();I.attr("d",4===a||2==a?m?"M"+v*m+","+P+"H0.5V"+J+"H"+v*m:"M0.5,"+P+"V"+J:m?"M"+P+","+v*m+"V0.5H"+J+"V"+v*m:"M"+P+",0.5H"+
  J);w.attr("opacity",1).attr("transform",function(f){return l(x(f))});N.attr(h+"2",v*k);T.attr(h,v*F).text(B);y.filter(Jn).attr("fill","none").attr("font-size",10).attr("font-family","sans-serif").attr("text-anchor",2===a?"start":4===a?"end":"middle");y.each(function(){this.__axis=x})}var d=[],e=null,g=null,k=6,m=6,p=3,v=1===a||4===a?-1:1,h=4===a||2===a?"x":"y",l=1===a||3===a?Fn:Gn;c.scale=function(q){return arguments.length?(b=q,c):b};c.ticks=function(){return d=Jf.call(arguments),c};c.tickArguments=
  function(q){return arguments.length?(d=null==q?[]:Jf.call(q),c):d.slice()};c.tickValues=function(q){return arguments.length?(e=null==q?null:Jf.call(q),c):e&&e.slice()};c.tickFormat=function(q){return arguments.length?(g=q,c):g};c.tickSize=function(q){return arguments.length?(k=m=+q,c):k};c.tickSizeInner=function(q){return arguments.length?(k=+q,c):k};c.tickSizeOuter=function(q){return arguments.length?(m=+q,c):m};c.tickPadding=function(q){return arguments.length?(p=+q,c):p};return c}function Ob(){for(var a=
  0,b=arguments.length,c={},d;a<b;++a){if(!(d=arguments[a]+"")||d in c)throw Error("illegal type: "+d);c[d]=[]}return new Qd(c)}function Qd(a){this._=a}function Kn(a,b){return a.trim().split(/^|\s+/).map(function(c){var d="",e=c.indexOf(".");0<=e&&(d=c.slice(e+1),c=c.slice(0,e));if(c&&!b.hasOwnProperty(c))throw Error("unknown type: "+c);return{type:c,name:d}})}function ki(a,b,c){for(var d=0,e=a.length;d<e;++d)if(a[d].name===b){a[d]=Ln;a=a.slice(0,d).concat(a.slice(d+1));break}null!=c&&a.push({name:b,
  value:c});return a}function Pc(a){var b=a+="",c=b.indexOf(":");0<=c&&"xmlns"!==(b=a.slice(0,c))&&(a=a.slice(c+1));return Ua.hasOwnProperty(b)?{space:Ua[b],local:a}:a}function Mn(a){return function(){var b=this.ownerDocument,c=this.namespaceURI;return"http://www.w3.org/1999/xhtml"===c&&"http://www.w3.org/1999/xhtml"===b.documentElement.namespaceURI?b.createElement(a):b.createElementNS(c,a)}}function Nn(a){return function(){return this.ownerDocument.createElementNS(a.space,a.local)}}function Rd(a){a=
  Pc(a);return(a.local?Nn:Mn)(a)}function On(){}function Sd(a){return null==a?On:function(){return this.querySelector(a)}}function Pn(){return[]}function Kf(a){return null==a?Pn:function(){return this.querySelectorAll(a)}}function li(a){return Array(a.length)}function Td(a,b){this.ownerDocument=a.ownerDocument;this.namespaceURI=a.namespaceURI;this._next=null;this._parent=a;this.__data__=b}function Qn(a){return function(){return a}}function Rn(a,b,c,d,e,g){for(var k=0,m,p=b.length,v=g.length;k<v;++k)(m=
  b[k])?(m.__data__=g[k],d[k]=m):c[k]=new Td(a,g[k]);for(;k<p;++k)if(m=b[k])e[k]=m}function Sn(a,b,c,d,e,g,k){var m,p,v={},h=b.length,l=g.length,q=Array(h),w;for(m=0;m<h;++m)if(p=b[m])q[m]=w="$"+k.call(p,p.__data__,m,b),w in v?e[m]=p:v[w]=p;for(m=0;m<l;++m)w="$"+k.call(a,g[m],m,g),(p=v[w])?(d[m]=p,p.__data__=g[m],v[w]=null):c[m]=new Td(a,g[m]);for(m=0;m<h;++m)(p=b[m])&&v[q[m]]===p&&(e[m]=p)}function Tn(a,b){return a<b?-1:a>b?1:a>=b?0:NaN}function Un(a){return function(){this.removeAttribute(a)}}function Vn(a){return function(){this.removeAttributeNS(a.space,
  a.local)}}function Wn(a,b){return function(){this.setAttribute(a,b)}}function Xn(a,b){return function(){this.setAttributeNS(a.space,a.local,b)}}function Yn(a,b){return function(){var c=b.apply(this,arguments);null==c?this.removeAttribute(a):this.setAttribute(a,c)}}function Zn(a,b){return function(){var c=b.apply(this,arguments);null==c?this.removeAttributeNS(a.space,a.local):this.setAttributeNS(a.space,a.local,c)}}function Lf(a){return a.ownerDocument&&a.ownerDocument.defaultView||a.document&&a||
  a.defaultView}function $n(a){return function(){this.style.removeProperty(a)}}function ao(a,b,c){return function(){this.style.setProperty(a,b,c)}}function bo(a,b,c){return function(){var d=b.apply(this,arguments);null==d?this.style.removeProperty(a):this.style.setProperty(a,d,c)}}function Pb(a,b){return a.style.getPropertyValue(b)||Lf(a).getComputedStyle(a,null).getPropertyValue(b)}function co(a){return function(){delete this[a]}}function eo(a,b){return function(){this[a]=b}}function fo(a,b){return function(){var c=
  b.apply(this,arguments);null==c?delete this[a]:this[a]=c}}function Mf(a){return a.classList||new mi(a)}function mi(a){this._node=a;this._names=(a.getAttribute("class")||"").trim().split(/^|\s+/)}function ni(a,b){a=Mf(a);for(var c=-1,d=b.length;++c<d;)a.add(b[c])}function oi(a,b){a=Mf(a);for(var c=-1,d=b.length;++c<d;)a.remove(b[c])}function go(a){return function(){ni(this,a)}}function ho(a){return function(){oi(this,a)}}function io(a,b){return function(){(b.apply(this,arguments)?ni:oi)(this,a)}}function jo(){this.textContent=
  ""}function ko(a){return function(){this.textContent=a}}function lo(a){return function(){var b=a.apply(this,arguments);this.textContent=null==b?"":b}}function mo(){this.innerHTML=""}function no(a){return function(){this.innerHTML=a}}function oo(a){return function(){var b=a.apply(this,arguments);this.innerHTML=null==b?"":b}}function po(){this.nextSibling&&this.parentNode.appendChild(this)}function qo(){this.previousSibling&&this.parentNode.insertBefore(this,this.parentNode.firstChild)}function ro(){return null}
  function so(){var a=this.parentNode;a&&a.removeChild(this)}function to(){return this.parentNode.insertBefore(this.cloneNode(!1),this.nextSibling)}function uo(){return this.parentNode.insertBefore(this.cloneNode(!0),this.nextSibling)}function vo(a,b,c){a=pi(a,b,c);return function(d){var e=d.relatedTarget;e&&(e===this||e.compareDocumentPosition(this)&8)||a.call(this,d)}}function pi(a,b,c){return function(d){var e=d3.event;d3.event=d;try{a.call(this,this.__data__,b,c)}finally{d3.event=e}}}function wo(a){return a.trim().split(/^|\s+/).map(function(b){var c=
  "",d=b.indexOf(".");0<=d&&(c=b.slice(d+1),b=b.slice(0,d));return{type:b,name:c}})}function xo(a){return function(){var b=this.__on;if(b){for(var c=0,d=-1,e=b.length,g;c<e;++c)(g=b[c],a.type&&g.type!==a.type||g.name!==a.name)?b[++d]=g:this.removeEventListener(g.type,g.listener,g.capture);++d?b.length=d:delete this.__on}}}function yo(a,b,c){var d=qi.hasOwnProperty(a.type)?vo:pi;return function(e,g,k){e=this.__on;var m;g=d(b,g,k);if(e){k=0;for(var p=e.length;k<p;++k)if((m=e[k]).type===a.type&&m.name===
  a.name){this.removeEventListener(m.type,m.listener,m.capture);this.addEventListener(m.type,m.listener=g,m.capture=c);m.value=b;return}}this.addEventListener(a.type,g,c);m={type:a.type,name:a.name,value:b,listener:g,capture:c};e?e.push(m):this.__on=[m]}}function Qc(a,b,c,d){var e=d3.event;a.sourceEvent=d3.event;d3.event=a;try{return b.apply(c,d)}finally{d3.event=e}}function ri(a,b,c){var d=Lf(a),e=d.CustomEvent;"function"===typeof e?e=new e(b,c):(e=d.document.createEvent("Event"),c?(e.initEvent(b,
  c.bubbles,c.cancelable),e.detail=c.detail):e.initEvent(b,!1,!1));a.dispatchEvent(e)}function zo(a,b){return function(){return ri(this,a,b)}}function Ao(a,b){return function(){return ri(this,a,b.apply(this,arguments))}}function Ja(a,b){this._groups=a;this._parents=b}function Qb(){return new Ja([[document.documentElement]],Nf)}function Ra(a){return"string"===typeof a?new Ja([[document.querySelector(a)]],[document.documentElement]):new Ja([[a]],Nf)}function si(){return new Of}function Of(){this._="@"+
  (++Bo).toString(36)}function Pf(){for(var a=d3.event,b;b=a.sourceEvent;)a=b;return a}function Ud(a,b){var c=a.ownerSVGElement||a;if(c.createSVGPoint)return c=c.createSVGPoint(),c.x=b.clientX,c.y=b.clientY,c=c.matrixTransform(a.getScreenCTM().inverse()),[c.x,c.y];c=a.getBoundingClientRect();return[b.clientX-c.left-a.clientLeft,b.clientY-c.top-a.clientTop]}function Bb(a){var b=Pf();b.changedTouches&&(b=b.changedTouches[0]);return Ud(a,b)}function Vd(a,b,c){3>arguments.length&&(c=b,b=Pf().changedTouches);
  for(var d=0,e=b?b.length:0,g;d<e;++d)if((g=b[d]).identifier===c)return Ud(a,g);return null}function fc(){d3.event.preventDefault();d3.event.stopImmediatePropagation()}function Wd(a){var b=a.document.documentElement;a=Ra(a).on("dragstart.drag",fc,!0);if("onselectstart"in b)a.on("selectstart.drag",fc,!0);else b.__noselect=b.style.MozUserSelect,b.style.MozUserSelect="none"}function Xd(a,b){var c=a.document.documentElement,d=Ra(a).on("dragstart.drag",null);b&&(d.on("click.drag",fc,!0),setTimeout(function(){d.on("click.drag",
  null)},0));if("onselectstart"in c)d.on("selectstart.drag",null);else c.style.MozUserSelect=c.__noselect,delete c.__noselect}function Yd(a){return function(){return a}}function Qf(a,b,c,d,e,g,k,m,p,v){this.target=a;this.type=b;this.subject=c;this.identifier=d;this.active=e;this.x=g;this.y=k;this.dx=m;this.dy=p;this._=v}function Co(){return!d3.event.button}function Do(){return this.parentNode}function Eo(a){return null==a?{x:d3.event.x,y:d3.event.y}:a}function Fo(){return"ontouchstart"in this}function gc(a,
  b,c){a.prototype=b.prototype=c;c.constructor=a}function Rc(a,b){a=Object.create(a.prototype);for(var c in b)a[c]=b[c];return a}function Cb(){}function Db(a){var b;a=(a+"").trim().toLowerCase();return(b=Go.exec(a))?(b=parseInt(b[1],16),new Fa(b>>8&15|b>>4&240,b>>4&15|b&240,(b&15)<<4|b&15,1)):(b=Ho.exec(a))?ti(parseInt(b[1],16)):(b=Io.exec(a))?new Fa(b[1],b[2],b[3],1):(b=Jo.exec(a))?new Fa(255*b[1]/100,255*b[2]/100,255*b[3]/100,1):(b=Ko.exec(a))?ui(b[1],b[2],b[3],b[4]):(b=Lo.exec(a))?ui(255*b[1]/100,
  255*b[2]/100,255*b[3]/100,b[4]):(b=Mo.exec(a))?vi(b[1],b[2]/100,b[3]/100,1):(b=No.exec(a))?vi(b[1],b[2]/100,b[3]/100,b[4]):wi.hasOwnProperty(a)?ti(wi[a]):"transparent"===a?new Fa(NaN,NaN,NaN,0):null}function ti(a){return new Fa(a>>16&255,a>>8&255,a&255,1)}function ui(a,b,c,d){0>=d&&(a=b=c=NaN);return new Fa(a,b,c,d)}function Rf(a){a instanceof Cb||(a=Db(a));if(!a)return new Fa;a=a.rgb();return new Fa(a.r,a.g,a.b,a.opacity)}function hc(a,b,c,d){return 1===arguments.length?Rf(a):new Fa(a,b,c,null==
  d?1:d)}function Fa(a,b,c,d){this.r=+a;this.g=+b;this.b=+c;this.opacity=+d}function Sf(a){a=Math.max(0,Math.min(255,Math.round(a)||0));return(16>a?"0":"")+a.toString(16)}function vi(a,b,c,d){0>=d?a=b=c=NaN:0>=c||1<=c?a=b=NaN:0>=b&&(a=NaN);return new ib(a,b,c,d)}function Oo(a){if(a instanceof ib)return new ib(a.h,a.s,a.l,a.opacity);a instanceof Cb||(a=Db(a));if(!a)return new ib;if(a instanceof ib)return a;a=a.rgb();var b=a.r/255,c=a.g/255,d=a.b/255,e=Math.min(b,c,d),g=Math.max(b,c,d),k=NaN,m=g-e,p=
  (g+e)/2;m?(k=b===g?(c-d)/m+6*(c<d):c===g?(d-b)/m+2:(b-c)/m+4,m/=.5>p?g+e:2-g-e,k*=60):m=0<p&&1>p?0:k;return new ib(k,m,p,a.opacity)}function Zd(a,b,c,d){return 1===arguments.length?Oo(a):new ib(a,b,c,null==d?1:d)}function ib(a,b,c,d){this.h=+a;this.s=+b;this.l=+c;this.opacity=+d}function Tf(a,b,c){return 255*(60>a?b+(c-b)*a/60:180>a?c:240>a?b+(c-b)*(240-a)/60:b)}function Uf(a){if(a instanceof cb)return new cb(a.l,a.a,a.b,a.opacity);if(a instanceof jb){if(isNaN(a.h))return new cb(a.l,0,0,a.opacity);
  var b=a.h*xi;return new cb(a.l,Math.cos(b)*a.c,Math.sin(b)*a.c,a.opacity)}a instanceof Fa||(a=Rf(a));var c=Vf(a.r),d=Vf(a.g),e=Vf(a.b);b=Wf(.2225045*c+.7168786*d+.0606169*e);if(c===d&&d===e)var g=c=b;else g=Wf((.4360747*c+.3850649*d+.1430804*e)/.96422),c=Wf((.0139322*c+.0971045*d+.7141733*e)/.82521);return new cb(116*b-16,500*(g-b),200*(b-c),a.opacity)}function $d(a,b,c,d){return 1===arguments.length?Uf(a):new cb(a,b,c,null==d?1:d)}function cb(a,b,c,d){this.l=+a;this.a=+b;this.b=+c;this.opacity=+d}
  function Wf(a){return a>Po?Math.pow(a,1/3):a/yi+zi}function Xf(a){return a>ic?a*a*a:yi*(a-zi)}function Yf(a){return 255*(.0031308>=a?12.92*a:1.055*Math.pow(a,1/2.4)-.055)}function Vf(a){return.04045>=(a/=255)?a/12.92:Math.pow((a+.055)/1.055,2.4)}function Ai(a){if(a instanceof jb)return new jb(a.h,a.c,a.l,a.opacity);a instanceof cb||(a=Uf(a));if(0===a.a&&0===a.b)return new jb(NaN,0,a.l,a.opacity);var b=Math.atan2(a.b,a.a)*Bi;return new jb(0>b?b+360:b,Math.sqrt(a.a*a.a+a.b*a.b),a.l,a.opacity)}function ae(a,
  b,c,d){return 1===arguments.length?Ai(a):new jb(a,b,c,null==d?1:d)}function jb(a,b,c,d){this.h=+a;this.c=+b;this.l=+c;this.opacity=+d}function db(a,b,c,d){if(1===arguments.length){var e=a;if(e instanceof Rb)e=new Rb(e.h,e.s,e.l,e.opacity);else{e instanceof Fa||(e=Rf(e));var g=e.g/255,k=e.b/255,m=(Ci*k+e.r/255*-1.7884503806-3.5172982438*g)/(Ci+-1.7884503806-3.5172982438);k-=m;var p=(1.97294*(g-m)- -.29227*k)/-.90649;k=(g=Math.sqrt(p*p+k*k)/(1.97294*m*(1-m)))?Math.atan2(p,k)*Bi-120:NaN;e=new Rb(0>k?
  k+360:k,g,m,e.opacity)}}else e=new Rb(a,b,c,null==d?1:d);return e}function Rb(a,b,c,d){this.h=+a;this.s=+b;this.l=+c;this.opacity=+d}function Di(a,b,c,d,e){var g=a*a,k=g*a;return((1-3*a+3*g-k)*b+(4-6*g+3*k)*c+(1+3*a+3*g-3*k)*d+k*e)/6}function Ei(a){var b=a.length-1;return function(c){var d=0>=c?c=0:1<=c?(c=1,b-1):Math.floor(c*b),e=a[d],g=a[d+1];return Di((c-d/b)*b,0<d?a[d-1]:2*e-g,e,g,d<b-1?a[d+2]:2*g-e)}}function Fi(a){var b=a.length;return function(c){var d=Math.floor((0>(c%=1)?++c:c)*b);return Di((c-
  d/b)*b,a[(d+b-1)%b],a[d%b],a[(d+1)%b],a[(d+2)%b])}}function be(a){return function(){return a}}function Gi(a,b){return function(c){return a+c*b}}function Qo(a,b,c){return a=Math.pow(a,c),b=Math.pow(b,c)-a,c=1/c,function(d){return Math.pow(a+d*b,c)}}function ce(a,b){var c=b-a;return c?Gi(a,180<c||-180>c?c-360*Math.round(c/360):c):be(isNaN(a)?b:a)}function Ro(a){return 1===(a=+a)?Ea:function(b,c){return c-b?Qo(b,c,a):be(isNaN(b)?c:b)}}function Ea(a,b){var c=b-a;return c?Gi(a,c):be(isNaN(a)?b:a)}function Hi(a){return function(b){var c=
  b.length,d=Array(c),e=Array(c),g=Array(c),k;for(k=0;k<c;++k){var m=hc(b[k]);d[k]=m.r||0;e[k]=m.g||0;g[k]=m.b||0}d=a(d);e=a(e);g=a(g);m.opacity=1;return function(p){m.r=d(p);m.g=e(p);m.b=g(p);return m+""}}}function Ii(a,b){var c=b?b.length:0,d=a?Math.min(c,a.length):0,e=Array(d),g=Array(c),k;for(k=0;k<d;++k)e[k]=Sc(a[k],b[k]);for(;k<c;++k)g[k]=b[k];return function(m){for(k=0;k<d;++k)g[k]=e[k](m);return g}}function Ji(a,b){var c=new Date;return a=+a,b-=a,function(d){return c.setTime(a+b*d),c}}function Va(a,
  b){return a=+a,b-=a,function(c){return a+b*c}}function Ki(a,b){var c={},d={},e;if(null===a||"object"!==typeof a)a={};if(null===b||"object"!==typeof b)b={};for(e in b)e in a?c[e]=Sc(a[e],b[e]):d[e]=b[e];return function(g){for(e in c)d[e]=c[e](g);return d}}function So(a){return function(){return a}}function To(a){return function(b){return a(b)+""}}function Zf(a,b){var c=$f.lastIndex=ag.lastIndex=0,d,e,g,k=-1,m=[],p=[];a+="";for(b+="";(d=$f.exec(a))&&(e=ag.exec(b));)(g=e.index)>c&&(g=b.slice(c,g),m[k]?
  m[k]+=g:m[++k]=g),(d=d[0])===(e=e[0])?m[k]?m[k]+=e:m[++k]=e:(m[++k]=null,p.push({i:k,x:Va(d,e)})),c=ag.lastIndex;c<b.length&&(g=b.slice(c),m[k]?m[k]+=g:m[++k]=g);return 2>m.length?p[0]?To(p[0].x):So(b):(b=p.length,function(v){for(var h=0,l;h<b;++h)m[(l=p[h]).i]=l.x(v);return m.join("")})}function Sc(a,b){var c=typeof b,d;return null==b||"boolean"===c?be(b):("number"===c?Va:"string"===c?(d=Db(b))?(b=d,Tc):Zf:b instanceof Db?Tc:b instanceof Date?Ji:Array.isArray(b)?Ii:"function"!==typeof b.valueOf&&
  "function"!==typeof b.toString||isNaN(b)?Ki:Va)(a,b)}function Li(a,b){return a=+a,b-=a,function(c){return Math.round(a+b*c)}}function Mi(a,b,c,d,e,g){var k,m,p;if(k=Math.sqrt(a*a+b*b))a/=k,b/=k;if(p=a*c+b*d)c-=a*p,d-=b*p;if(m=Math.sqrt(c*c+d*d))c/=m,d/=m,p/=m;a*d<b*c&&(a=-a,b=-b,p=-p,k=-k);return{translateX:e,translateY:g,rotate:Math.atan2(b,a)*Ni,skewX:Math.atan(p)*Ni,scaleX:k,scaleY:m}}function Oi(a,b,c,d){function e(v){return v.length?v.pop()+" ":""}function g(v,h,l,q,w,B){v!==l||h!==q?(w=w.push("translate(",
  null,b,null,c),B.push({i:w-4,x:Va(v,l)},{i:w-2,x:Va(h,q)})):(l||q)&&w.push("translate("+l+b+q+c)}function k(v,h,l,q){v!==h?(180<v-h?h+=360:180<h-v&&(v+=360),q.push({i:l.push(e(l)+"rotate(",null,d)-2,x:Va(v,h)})):h&&l.push(e(l)+"rotate("+h+d)}function m(v,h,l,q){v!==h?q.push({i:l.push(e(l)+"skewX(",null,d)-2,x:Va(v,h)}):h&&l.push(e(l)+"skewX("+h+d)}function p(v,h,l,q,w,B){v!==l||h!==q?(w=w.push(e(w)+"scale(",null,",",null,")"),B.push({i:w-4,x:Va(v,l)},{i:w-2,x:Va(h,q)})):1===l&&1===q||w.push(e(w)+
  "scale("+l+","+q+")")}return function(v,h){var l=[],q=[];v=a(v);h=a(h);g(v.translateX,v.translateY,h.translateX,h.translateY,l,q);k(v.rotate,h.rotate,l,q);m(v.skewX,h.skewX,l,q);p(v.scaleX,v.scaleY,h.scaleX,h.scaleY,l,q);v=h=null;return function(w){for(var B=-1,F=q.length,J;++B<F;)l[(J=q[B]).i]=J.x(w);return l.join("")}}}function Pi(a){return((a=Math.exp(a))+1/a)/2}function Qi(a,b){var c=a[0],d=a[1],e=a[2];a=b[2];var g=b[0]-c,k=b[1]-d,m=g*g+k*k;if(1E-12>m){var p=Math.log(a/e)/Uc;a=function(l){return[c+
  l*g,d+l*k,e*Math.exp(Uc*l*p)]}}else{var v=Math.sqrt(m);b=(a*a-e*e+4*m)/(4*e*v);a=(a*a-e*e-4*m)/(4*a*v);var h=Math.log(Math.sqrt(b*b+1)-b);p=(Math.log(Math.sqrt(a*a+1)-a)-h)/Uc;a=function(l){l*=p;var q=Pi(h),w=Uc*l+h;var B=((w=Math.exp(2*w))-1)/(w+1);var F=h;w=((F=Math.exp(F))-1/F)/2;B=e/(2*v)*(q*B-w);return[c+B*g,d+B*k,e*q/Pi(Uc*l+h)]}}a.duration=1E3*p;return a}function Ri(a){return function(b,c){var d=a((b=Zd(b)).h,(c=Zd(c)).h),e=Ea(b.s,c.s),g=Ea(b.l,c.l),k=Ea(b.opacity,c.opacity);return function(m){b.h=
  d(m);b.s=e(m);b.l=g(m);b.opacity=k(m);return b+""}}}function Si(a){return function(b,c){var d=a((b=ae(b)).h,(c=ae(c)).h),e=Ea(b.c,c.c),g=Ea(b.l,c.l),k=Ea(b.opacity,c.opacity);return function(m){b.h=d(m);b.c=e(m);b.l=g(m);b.opacity=k(m);return b+""}}}function Ti(a){return function d(c){function e(g,k){var m=a((g=db(g)).h,(k=db(k)).h),p=Ea(g.s,k.s),v=Ea(g.l,k.l),h=Ea(g.opacity,k.opacity);return function(l){g.h=m(l);g.s=p(l);g.l=v(Math.pow(l,c));g.opacity=h(l);return g+""}}c=+c;e.gamma=d;return e}(1)}
  function jc(){return Sb||(Ui(Uo),Sb=Vc.now()+de)}function Uo(){Sb=0}function Wc(){this._call=this._time=this._next=null}function ee(a,b,c){var d=new Wc;d.restart(a,b,c);return d}function Vi(){jc();++kc;for(var a=fe,b;a;)0<=(b=Sb-a._time)&&a._call.call(null,b),a=a._next;--kc}function Wi(){Sb=(ge=Vc.now())+de;kc=Xc=0;try{Vi()}finally{kc=0;for(var a,b=fe,c,d=Infinity;b;)b._call?(d>b._time&&(d=b._time),a=b,b=b._next):(c=b._next,b._next=null,b=a?a._next=c:fe=c);Yc=a;bg(d);Sb=0}}function Vo(){var a=Vc.now(),
  b=a-ge;1E3<b&&(de-=b,ge=a)}function bg(a){kc||(Xc&&(Xc=clearTimeout(Xc)),24<a-Sb?(Infinity>a&&(Xc=setTimeout(Wi,a-Vc.now()-de)),Zc&&(Zc=clearInterval(Zc))):(Zc||(ge=Vc.now(),Zc=setInterval(Vo,1E3)),kc=1,Ui(Wi)))}function cg(a,b,c){var d=new Wc;b=null==b?0:+b;d.restart(function(e){d.stop();a(e+b)},b,c);return d}function he(a,b,c,d,e,g){var k=a.__transition;if(!k)a.__transition={};else if(c in k)return;Wo(a,c,{name:b,index:d,group:e,on:Xo,tween:Yo,time:g.time,delay:g.delay,duration:g.duration,ease:g.ease,
  timer:null,state:0})}function dg(a,b){a=eb(a,b);if(0<a.state)throw Error("too late; already scheduled");return a}function Tb(a,b){a=eb(a,b);if(2<a.state)throw Error("too late; already started");return a}function eb(a,b){a=a.__transition;if(!a||!(a=a[b]))throw Error("transition not found");return a}function Wo(a,b,c){function d(p){var v,h;if(1!==c.state)return g();for(q in k){var l=k[q];if(l.name===c.name){if(3===l.state)return cg(d);4===l.state?(l.state=6,l.timer.stop(),l.on.call("interrupt",a,a.__data__,
  l.index,l.group),delete k[q]):+q<b&&(l.state=6,l.timer.stop(),delete k[q])}}cg(function(){3===c.state&&(c.state=4,c.timer.restart(e,c.delay,c.time),e(p))});c.state=2;c.on.call("start",a,a.__data__,c.index,c.group);if(2===c.state){c.state=3;m=Array(h=c.tween.length);var q=0;for(v=-1;q<h;++q)if(l=c.tween[q].value.call(a,a.__data__,c.index,c.group))m[++v]=l;m.length=v+1}}function e(p){p=p<c.duration?c.ease.call(null,p/c.duration):(c.timer.restart(g),c.state=5,1);for(var v=-1,h=m.length;++v<h;)m[v].call(null,
  p);5===c.state&&(c.on.call("end",a,a.__data__,c.index,c.group),g())}function g(){c.state=6;c.timer.stop();delete k[b];for(var p in k)return;delete a.__transition}var k=a.__transition,m;k[b]=c;c.timer=ee(function(p){c.state=1;c.timer.restart(d,c.delay,c.time);c.delay<=p&&d(p-c.delay)},0,c.time)}function Ub(a,b){var c=a.__transition,d,e=!0,g;if(c){b=null==b?null:b+"";for(g in c)if((d=c[g]).name!==b)e=!1;else{var k=2<d.state&&5>d.state;d.state=6;d.timer.stop();k&&d.on.call("interrupt",a,a.__data__,d.index,
  d.group);delete c[g]}e&&delete a.__transition}}function Zo(a,b){var c,d;return function(){var e=Tb(this,a),g=e.tween;if(g!==c){d=c=g;g=0;for(var k=d.length;g<k;++g)if(d[g].name===b){d=d.slice();d.splice(g,1);break}}e.tween=d}}function $o(a,b,c){var d,e;if("function"!==typeof c)throw Error();return function(){var g=Tb(this,a),k=g.tween;if(k!==d){e=(d=k).slice();k={name:b,value:c};for(var m=0,p=e.length;m<p;++m)if(e[m].name===b){e[m]=k;break}m===p&&e.push(k)}g.tween=e}}function eg(a,b,c){var d=a._id;
  a.each(function(){var e=Tb(this,d);(e.value||(e.value={}))[b]=c.apply(this,arguments)});return function(e){return eb(e,d).value[b]}}function Xi(a,b){var c;return("number"===typeof b?Va:b instanceof Db?Tc:(c=Db(b))?(b=c,Tc):Zf)(a,b)}function ap(a){return function(){this.removeAttribute(a)}}function bp(a){return function(){this.removeAttributeNS(a.space,a.local)}}function cp(a,b,c){var d,e;return function(){var g=this.getAttribute(a);return g===c?null:g===d?e:e=b(d=g,c)}}function dp(a,b,c){var d,e;
  return function(){var g=this.getAttributeNS(a.space,a.local);return g===c?null:g===d?e:e=b(d=g,c)}}function ep(a,b,c){var d,e,g;return function(){var k=c(this);if(null==k)return void this.removeAttribute(a);var m=this.getAttribute(a);return m===k?null:m===d&&k===e?g:g=b(d=m,e=k)}}function fp(a,b,c){var d,e,g;return function(){var k=c(this);if(null==k)return void this.removeAttributeNS(a.space,a.local);var m=this.getAttributeNS(a.space,a.local);return m===k?null:m===d&&k===e?g:g=b(d=m,e=k)}}function gp(a,
  b){function c(){var d=this,e=b.apply(d,arguments);return e&&function(g){d.setAttributeNS(a.space,a.local,e(g))}}c._value=b;return c}function hp(a,b){function c(){var d=this,e=b.apply(d,arguments);return e&&function(g){d.setAttribute(a,e(g))}}c._value=b;return c}function ip(a,b){return function(){dg(this,a).delay=+b.apply(this,arguments)}}function jp(a,b){return b=+b,function(){dg(this,a).delay=b}}function kp(a,b){return function(){Tb(this,a).duration=+b.apply(this,arguments)}}function lp(a,b){return b=
  +b,function(){Tb(this,a).duration=b}}function mp(a,b){if("function"!==typeof b)throw Error();return function(){Tb(this,a).ease=b}}function np(a){return(a+"").trim().split(/^|\s+/).every(function(b){var c=b.indexOf(".");0<=c&&(b=b.slice(0,c));return!b||"start"===b})}function op(a,b,c){var d,e,g=np(b)?dg:Tb;return function(){var k=g(this,a),m=k.on;if(m!==d)(e=(d=m).copy()).on(b,c);k.on=e}}function pp(a){return function(){var b=this.parentNode,c;for(c in this.__transition)if(+c!==a)return;b&&b.removeChild(this)}}
  function qp(a,b){var c,d,e;return function(){var g=Pb(this,a),k=(this.style.removeProperty(a),Pb(this,a));return g===k?null:g===c&&k===d?e:e=b(c=g,d=k)}}function rp(a){return function(){this.style.removeProperty(a)}}function sp(a,b,c){var d,e;return function(){var g=Pb(this,a);return g===c?null:g===d?e:e=b(d=g,c)}}function tp(a,b,c){var d,e,g;return function(){var k=Pb(this,a),m=c(this);null==m&&(m=(this.style.removeProperty(a),Pb(this,a)));return k===m?null:k===d&&m===e?g:g=b(d=k,e=m)}}function up(a,
  b,c){function d(){var e=this,g=b.apply(e,arguments);return g&&function(k){e.style.setProperty(a,g(k),c)}}d._value=b;return d}function vp(a){return function(){this.textContent=a}}function wp(a){return function(){var b=a(this);this.textContent=null==b?"":b}}function kb(a,b,c,d){this._groups=a;this._parents=b;this._name=c;this._id=d}function Yi(a){return Qb().transition(a)}function Zi(a){return(1>=(a*=2)?a*a:--a*(2-a)+1)/2}function fg(a){return(1>=(a*=2)?a*a*a:(a-=2)*a*a+2)/2}function $i(a){return(1-
  Math.cos(aj*a))/2}function bj(a){return(1>=(a*=2)?Math.pow(2,10*a-10):2-Math.pow(2,10-10*a))/2}function cj(a){return(1>=(a*=2)?1-Math.sqrt(1-a*a):Math.sqrt(1-(a-=2)*a)+1)/2}function $c(a){return(a=+a)<gg?ie*a*a:a<xp?ie*(a-=yp)*a+.75:a<zp?ie*(a-=Ap)*a+.9375:ie*(a-=Bp)*a+.984375}function dj(a){return function(){return a}}function Cp(a,b,c){this.target=a;this.type=b;this.selection=c}function je(){d3.event.preventDefault();d3.event.stopImmediatePropagation()}function ad(a){return{type:a}}function Dp(){return!d3.event.button}
  function Ep(){var a=this.ownerSVGElement||this;return[[0,0],[a.width.baseVal.value,a.height.baseVal.value]]}function hg(a){for(;!a.__brush;)if(!(a=a.parentNode))return;return a.__brush}function ig(a){return a[0][0]===a[1][0]||a[0][1]===a[1][1]}function jg(a){function b(q){var w=q.property("__brush",k).selectAll(".overlay").data([ad("overlay")]);w.enter().append("rect").attr("class","overlay").attr("pointer-events","all").attr("cursor",qb.overlay).merge(w).each(function(){var B=hg(this).extent;Ra(this).attr("x",
  B[0][0]).attr("y",B[0][1]).attr("width",B[1][0]-B[0][0]).attr("height",B[1][1]-B[0][1])});q.selectAll(".selection").data([ad("selection")]).enter().append("rect").attr("class","selection").attr("cursor",qb.selection).attr("fill","#777").attr("fill-opacity",.3).attr("stroke","#fff").attr("shape-rendering","crispEdges");w=q.selectAll(".handle").data(a.handles,function(B){return B.type});w.exit().remove();w.enter().append("rect").attr("class",function(B){return"handle handle--"+B.type}).attr("cursor",
  function(B){return qb[B.type]});q.each(c).attr("fill","none").attr("pointer-events","all").style("-webkit-tap-highlight-color","rgba(0,0,0,0)").on("mousedown.brush touchstart.brush",g)}function c(){var q=Ra(this),w=hg(this).selection;w?(q.selectAll(".selection").style("display",null).attr("x",w[0][0]).attr("y",w[0][1]).attr("width",w[1][0]-w[0][0]).attr("height",w[1][1]-w[0][1]),q.selectAll(".handle").style("display",null).attr("x",function(B){return"e"===B.type[B.type.length-1]?w[1][0]-h/2:w[0][0]-
  h/2}).attr("y",function(B){return"s"===B.type[0]?w[1][1]-h/2:w[0][1]-h/2}).attr("width",function(B){return"n"===B.type||"s"===B.type?w[1][0]-w[0][0]+h:h}).attr("height",function(B){return"e"===B.type||"w"===B.type?w[1][1]-w[0][1]+h:h})):q.selectAll(".selection,.handle").style("display","none").attr("x",null).attr("y",null).attr("width",null).attr("height",null)}function d(q,w){return q.__brush.emitter||new e(q,w)}function e(q,w){this.that=q;this.args=w;this.state=q.__brush;this.active=0}function g(){function q(){var ea=
  Bb(P);!S||E||K||(Math.abs(ea[0]-L[0])>Math.abs(ea[1]-L[1])?K=!0:E=!0);L=ea;O=!0;je();w()}function w(){C=L[0]-H[0];G=L[1]-H[1];switch(y){case kg:case ej:I&&(C=Math.max(f-n,Math.min(t-z,C)),M=n+C,Y=z+C);Q&&(G=Math.max(u-r,Math.min(D-A,G)),X=r+G,W=A+G);break;case lc:0>I?(C=Math.max(f-n,Math.min(t-n,C)),M=n+C,Y=z):0<I&&(C=Math.max(f-z,Math.min(t-z,C)),M=n,Y=z+C);0>Q?(G=Math.max(u-r,Math.min(D-r,G)),X=r+G,W=A):0<Q&&(G=Math.max(u-A,Math.min(D-A,G)),X=r,W=A+G);break;case mc:I&&(M=Math.max(f,Math.min(t,n-
  C*I)),Y=Math.max(f,Math.min(t,z+C*I))),Q&&(X=Math.max(u,Math.min(D,r-G*Q)),W=Math.max(u,Math.min(D,A+G*Q)))}if(Y<M){I*=-1;var ea=n;n=z;z=ea;ea=M;M=Y;Y=ea;x in fj&&aa.attr("cursor",qb[x=fj[x]])}W<X&&(Q*=-1,ea=r,r=A,A=ea,ea=X,X=W,W=ea,x in gj&&aa.attr("cursor",qb[x=gj[x]]));V.selection&&(T=V.selection);E&&(M=T[0][0],Y=T[1][0]);K&&(X=T[0][1],W=T[1][1]);if(T[0][0]!==M||T[0][1]!==X||T[1][0]!==Y||T[1][1]!==W)V.selection=[[M,X],[Y,W]],c.call(P),U.brush()}function B(){d3.event.stopImmediatePropagation();
  if(d3.event.touches){if(d3.event.touches.length)return;l&&clearTimeout(l);l=setTimeout(function(){l=null},500);ba.on("touchmove.brush touchend.brush touchcancel.brush",null)}else Xd(d3.event.view,O),ha.on("keydown.brush keyup.brush mousemove.brush mouseup.brush",null);ba.attr("pointer-events","all");aa.attr("cursor",qb.overlay);V.selection&&(T=V.selection);ig(T)&&(V.selection=null,c.call(P));U.end()}function F(){switch(d3.event.keyCode){case 16:S=I&&Q;break;case 18:y===lc&&(I&&(z=Y-C*I,n=M+C*I),Q&&
  (A=W-G*Q,r=X+G*Q),y=mc,w());break;case 32:if(y===lc||y===mc)0>I?z=Y-C:0<I&&(n=M-C),0>Q?A=W-G:0<Q&&(r=X-G),y=kg,aa.attr("cursor",qb.selection),w();break;default:return}je()}function J(){switch(d3.event.keyCode){case 16:S&&(E=K=S=!1,w());break;case 18:y===mc&&(0>I?z=Y:0<I&&(n=M),0>Q?A=W:0<Q&&(r=X),y=lc,w());break;case 32:y===kg&&(d3.event.altKey?(I&&(z=Y-C*I,n=M+C*I),Q&&(A=W-G*Q,r=X+G*Q),y=mc):(0>I?z=Y:0<I&&(n=M),0>Q?A=W:0<Q&&(r=X),y=lc),aa.attr("cursor",qb[x]),w());break;default:return}je()}if(d3.event.touches){if(d3.event.changedTouches.length<
  d3.event.touches.length)return je()}else if(l)return;if(p.apply(this,arguments)){var P=this,x=d3.event.target.__data__.type,y="selection"===(d3.event.metaKey?x="overlay":x)?ej:d3.event.altKey?mc:lc,I=a===ke?null:Fp[x],Q=a===le?null:Gp[x],V=hg(P),N=V.extent,T=V.selection,f=N[0][0],n,u=N[0][1],r,t=N[1][0],z,D=N[1][1],A,C,G,O,S=I&&Q&&d3.event.shiftKey,E,K,H=Bb(P),L=H,U=d(P,arguments).beforestart();"overlay"===x?V.selection=T=[[n=a===ke?f:H[0],r=a===le?u:H[1]],[z=a===ke?t:n,A=a===le?D:r]]:(n=T[0][0],
  r=T[0][1],z=T[1][0],A=T[1][1]);var M=n;var X=r;var Y=z;var W=A;var ba=Ra(P).attr("pointer-events","none"),aa=ba.selectAll(".overlay").attr("cursor",qb[x]);if(d3.event.touches)ba.on("touchmove.brush",q,!0).on("touchend.brush touchcancel.brush",B,!0);else{var ha=Ra(d3.event.view).on("keydown.brush",F,!0).on("keyup.brush",J,!0).on("mousemove.brush",q,!0).on("mouseup.brush",B,!0);Wd(d3.event.view)}d3.event.stopImmediatePropagation();Ub(P);c.call(P);U.start()}}function k(){var q=this.__brush||{selection:null};
  q.extent=m.apply(this,arguments);q.dim=a;return q}var m=Ep,p=Dp,v=Ob(b,"start","brush","end"),h=6,l;b.move=function(q,w){q.selection?q.on("start.brush",function(){d(this,arguments).beforestart().start()}).on("interrupt.brush end.brush",function(){d(this,arguments).end()}).tween("brush",function(){function B(Q){J.selection=1===Q&&ig(y)?null:I(Q);c.call(F);P.brush()}var F=this,J=F.__brush,P=d(F,arguments),x=J.selection,y=a.input("function"===typeof w?w.apply(this,arguments):w,J.extent),I=Sc(x,y);return x&&
  y?B:B(1)}):q.each(function(){var B=arguments,F=this.__brush,J=a.input("function"===typeof w?w.apply(this,B):w,F.extent);B=d(this,B).beforestart();Ub(this);F.selection=null==J||ig(J)?null:J;c.call(this);B.start().brush().end()})};e.prototype={beforestart:function(){1===++this.active&&(this.state.emitter=this,this.starting=!0);return this},start:function(){this.starting&&(this.starting=!1,this.emit("start"));return this},brush:function(){this.emit("brush");return this},end:function(){0===--this.active&&
  (delete this.state.emitter,this.emit("end"));return this},emit:function(q){Qc(new Cp(b,q,a.output(this.state.selection)),v.apply,v,[q,this.that,this.args])}};b.extent=function(q){return arguments.length?(m="function"===typeof q?q:dj([[+q[0][0],+q[0][1]],[+q[1][0],+q[1][1]]]),b):m};b.filter=function(q){return arguments.length?(p="function"===typeof q?q:dj(!!q),b):p};b.handleSize=function(q){return arguments.length?(h=+q,b):h};b.on=function(){var q=v.on.apply(v,arguments);return q===v?b:q};return b}
  function Hp(a){return function(b,c){return a(b.source.value+b.target.value,c.source.value+c.target.value)}}function lg(a){return function(){return a}}function mg(){this._x0=this._y0=this._x1=this._y1=null;this._=""}function Eb(){return new mg}function Ip(a){return a.source}function Jp(a){return a.target}function Kp(a){return a.radius}function Lp(a){return a.startAngle}function Mp(a){return a.endAngle}function me(){}function rb(a,b){var c=new me;if(a instanceof me)a.each(function(k,m){c.set(m,k)});
  else if(Array.isArray(a)){var d=-1,e=a.length,g;if(null==b)for(;++d<e;)c.set(d,a[d]);else for(;++d<e;)c.set(b(g=a[d],d,a),g)}else if(a)for(d in a)c.set(d,a[d]);return c}function Np(){return{}}function Op(a,b,c){a[b]=c}function hj(){return rb()}function ij(a,b,c){a.set(b,c)}function ne(){}function jj(a,b){var c=new ne;if(a instanceof ne)a.each(function(g){c.add(g)});else if(a){var d=-1,e=a.length;if(null==b)for(;++d<e;)c.add(a[d]);else for(;++d<e;)c.add(b(a[d],d,a))}return c}function Pp(a,b){return a-
  b}function Fb(a){return function(){return a}}function Qp(){}function kj(){function a(p){var v=k(p);if(Array.isArray(v))v=v.slice().sort(Pp);else{var h=Cf(p),l=h[0];h=h[1];v=Nb(l,h,v);v=Ta(Math.floor(l/v)*v,Math.floor(h/v)*v,v)}return v.map(function(q){return b(p,q)})}function b(p,v){var h=[],l=[];c(p,v,function(q){m(q,p,v);for(var w=0,B=q.length,F=q[B-1][1]*q[0][0]-q[B-1][0]*q[0][1];++w<B;)F+=q[w-1][1]*q[w][0]-q[w-1][0]*q[w][1];0<F?h.push([q]):l.push(q)});l.forEach(function(q){for(var w=0,B=h.length,
  F;w<B;++w){a:{var J=(F=h[w])[0];for(var P=q,x=-1,y=P.length;++x<y;){b:{var I=J;for(var Q=P[x],V=Q[0],N=Q[1],T=-1,f=0,n=I.length,u=n-1;f<n;u=f++){var r=I[f],t=r[0],z=r[1],D=I[u];u=D[0];var A=D[1],C,G=r;r=D;D=Q;if(C=(r[0]-G[0])*(D[1]-G[1])===(D[0]-G[0])*(r[1]-G[1]))G=G[C=+(G[0]===r[0])],D=D[C],r=r[C],C=G<=D&&D<=r||r<=D&&D<=G;if(C){I=0;break b}z>N!==A>N&&V<(u-t)*(N-z)/(A-z)+t&&(T=-T)}I=T}if(I){J=I;break a}}J=0}if(-1!==J){F.push(q);break}}});return{type:"MultiPolygon",value:v,coordinates:h}}function c(p,
  v,h){function l(I){var Q=[I[0][0]+F,I[0][1]+B];I=[I[1][0]+F,I[1][1]+B];var V=2*Q[0]+Q[1]*(e+1)*4,N=2*I[0]+I[1]*(e+1)*4,T,f;(T=w[V])?(f=q[N])?(delete w[T.end],delete q[f.start],T===f?(T.ring.push(I),h(T.ring)):q[T.start]=w[f.end]={start:T.start,end:f.end,ring:T.ring.concat(f.ring)}):(delete w[T.end],T.ring.push(I),w[T.end=N]=T):(T=q[N])?(f=w[V])?(delete q[T.start],delete w[f.end],T===f?(T.ring.push(I),h(T.ring)):q[f.start]=w[T.end]={start:f.start,end:T.end,ring:f.ring.concat(T.ring)}):(delete q[T.start],
  T.ring.unshift(Q),q[T.start=V]=T):q[V]=w[N]={start:V,end:N,ring:[Q,I]}}var q=[],w=[],B;var F=B=-1;var J=p[0]>=v;for(sb[J<<1].forEach(l);++F<e-1;){var P=J;J=p[F+1]>=v;sb[P|J<<1].forEach(l)}for(sb[J<<0].forEach(l);++B<g-1;){F=-1;J=p[B*e+e]>=v;var x=p[B*e]>=v;for(sb[J<<1|x<<2].forEach(l);++F<e-1;){P=J;J=p[B*e+e+F+1]>=v;var y=x;x=p[B*e+F+1]>=v;sb[P|J<<1|x<<2|y<<3].forEach(l)}sb[J|x<<3].forEach(l)}F=-1;x=p[B*e]>=v;for(sb[x<<2].forEach(l);++F<e-1;)y=x,x=p[B*e+F+1]>=v,sb[x<<2|y<<3].forEach(l);sb[x<<3].forEach(l)}
  function d(p,v,h){p.forEach(function(l){var q=l[0],w=l[1],B=q|0,F=w|0,J=v[F*e+B];if(0<q&&q<e&&B===q){var P=v[F*e+B-1];l[0]=q+(h-P)/(J-P)-.5}0<w&&w<g&&F===w&&(P=v[(F-1)*e+B],l[1]=w+(h-P)/(J-P)-.5)})}var e=1,g=1,k=Hf,m=d;a.contour=b;a.size=function(p){if(!arguments.length)return[e,g];var v=Math.ceil(p[0]),h=Math.ceil(p[1]);if(!(0<v&&0<h))throw Error("invalid size");return e=v,g=h,a};a.thresholds=function(p){return arguments.length?(k="function"===typeof p?p:Array.isArray(p)?Fb(lj.call(p)):Fb(p),a):
  k};a.smooth=function(p){return arguments.length?(m=p?d:Qp,a):m===d};return a}function ng(a,b,c){for(var d=a.width,e=a.height,g=(c<<1)+1,k=0;k<e;++k)for(var m=0,p=0;m<d+c;++m)m<d&&(p+=a.data[m+k*d]),m>=c&&(m>=g&&(p-=a.data[m-g+k*d]),b.data[m-c+k*d]=p/Math.min(m+1,d-1+g-m,g))}function og(a,b,c){for(var d=a.width,e=a.height,g=(c<<1)+1,k=0;k<d;++k)for(var m=0,p=0;m<e+c;++m)m<e&&(p+=a.data[k+m*d]),m>=c&&(m>=g&&(p-=a.data[k+(m-g)*d]),b.data[k+(m-c)*d]=p/Math.min(m+1,e-1+g-m,g))}function Rp(a){return a[0]}
  function Sp(a){return a[1]}function Tp(){return 1}function mj(a){return function(b){for(var c={},d=0;d<a.length;d++)c[a[d]]=b[d]||"";return c}}function Up(a,b){var c=mj(a);return function(d,e){return b(c(d),e,a)}}function Vp(a){var b=Object.create(null),c=[];a.forEach(function(d){for(var e in d)e in b||c.push(b[e]=e)});return c}function oe(a){function b(k,m){function p(){if(B)return pg;if(F)return F=!1,nj;var P,x=l,y;if(34===k.charCodeAt(x)){for(;l++<h&&34!==k.charCodeAt(l)||34===k.charCodeAt(++l););
  (P=l)>=h?B=!0:10===(y=k.charCodeAt(l++))?F=!0:13===y&&(F=!0,10===k.charCodeAt(l)&&++l);return k.slice(x+1,P-1).replace(/""/g,'"')}for(;l<h;){if(10===(y=k.charCodeAt(P=l++)))F=!0;else if(13===y)F=!0,10===k.charCodeAt(l)&&++l;else if(y!==g)continue;return k.slice(x,P)}return B=!0,k.slice(x,h)}var v=[],h=k.length,l=0,q=0,w,B=0>=h,F=!1;10===k.charCodeAt(h-1)&&--h;for(13===k.charCodeAt(h-1)&&--h;(w=p())!==pg;){for(var J=[];w!==nj&&w!==pg;)J.push(w),w=p();m&&null==(J=m(J,q++))||v.push(J)}return v}function c(k){return k.map(d).join(a)}
  function d(k){return null==k?"":e.test(k+="")?'"'+k.replace(/"/g,'""')+'"':k}var e=new RegExp('["'+a+"\n\r]"),g=a.charCodeAt(0);return{parse:function(k,m){var p,v;k=b(k,function(h,l){if(p)return p(h,l-1);v=h;p=m?Up(h,m):mj(h)});k.columns=v||[];return k},parseRows:b,format:function(k,m){null==m&&(m=Vp(k));return[m.map(d).join(a)].concat(k.map(function(p){return m.map(function(v){return d(p[v])}).join(a)})).join("\n")},formatRows:function(k){return k.map(c).join("\n")}}}function Wp(a){if(!a.ok)throw Error(a.status+
  " "+a.statusText);return a.blob()}function Xp(a){if(!a.ok)throw Error(a.status+" "+a.statusText);return a.arrayBuffer()}function Yp(a){if(!a.ok)throw Error(a.status+" "+a.statusText);return a.text()}function qg(a,b){return fetch(a,b).then(Yp)}function oj(a){return function(b,c,d){2===arguments.length&&"function"===typeof c&&(d=c,c=void 0);return qg(b,c).then(function(e){return a(e,d)})}}function Zp(a){if(!a.ok)throw Error(a.status+" "+a.statusText);return a.json()}function Ca(a){return function(){return a}}
  function Gb(){return 1E-6*(Math.random()-.5)}function pj(a,b,c,d){if(isNaN(b)||isNaN(c))return a;var e,g=a._root;d={data:d};var k=a._x0,m=a._y0,p=a._x1,v=a._y1,h,l,q,w,B;if(!g)return a._root=d,a;for(;g.length;)if((q=b>=(h=(k+p)/2))?k=h:p=h,(w=c>=(l=(m+v)/2))?m=l:v=l,e=g,!(g=g[B=w<<1|q]))return e[B]=d,a;var F=+a._x.call(null,g.data);var J=+a._y.call(null,g.data);if(b===F&&c===J)return d.next=g,e?e[B]=d:a._root=d,a;do e=e?e[B]=Array(4):a._root=Array(4),(q=b>=(h=(k+p)/2))?k=h:p=h,(w=c>=(l=(m+v)/2))?
  m=l:v=l;while((B=w<<1|q)===(q=(J>=l)<<1|F>=h));return e[q]=g,e[B]=d,a}function Ka(a,b,c,d,e){this.node=a;this.x0=b;this.y0=c;this.x1=d;this.y1=e}function $p(a){return a[0]}function aq(a){return a[1]}function pe(a,b,c){b=new rg(null==b?$p:b,null==c?aq:c,NaN,NaN,NaN,NaN);return null==a?b:b.addAll(a)}function rg(a,b,c,d,e,g){this._x=a;this._y=b;this._x0=c;this._y0=d;this._x1=e;this._y1=g;this._root=void 0}function qj(a){for(var b={data:a.data},c=b;a=a.next;)c=c.next={data:a.data};return b}function bq(a){return a.x+
  a.vx}function cq(a){return a.y+a.vy}function dq(a){return a.index}function rj(a,b){a=a.get(b);if(!a)throw Error("missing: "+b);return a}function eq(a){return a.x}function fq(a){return a.y}function qe(a,b){if(0>(b=(a=b?a.toExponential(b-1):a.toExponential()).indexOf("e")))return null;var c=a.slice(0,b);return[1<c.length?c[0]+c.slice(2):c,+a.slice(b+1)]}function nc(a){return a=qe(Math.abs(a)),a?a[1]:NaN}function gq(a,b){return function(c,d){for(var e=c.length,g=[],k=0,m=a[0],p=0;0<e&&0<m;){p+m+1>d&&
  (m=Math.max(1,d-p));g.push(c.substring(e-=m,e+m));if((p+=m+1)>d)break;m=a[k=(k+1)%a.length]}return g.reverse().join(b)}}function hq(a){return function(b){return b.replace(/[0-9]/g,function(c){return a[+c]})}}function bd(a){return new sg(a)}function sg(a){if(!(b=iq.exec(a)))throw Error("invalid format: "+a);var b;this.fill=b[1]||" ";this.align=b[2]||">";this.sign=b[3]||"-";this.symbol=b[4]||"";this.zero=!!b[5];this.width=b[6]&&+b[6];this.comma=!!b[7];this.precision=b[8]&&+b[8].slice(1);this.trim=!!b[9];
  this.type=b[10]||""}function sj(a,b){b=qe(a,b);if(!b)return a+"";a=b[0];b=b[1];return 0>b?"0."+Array(-b).join("0")+a:a.length>b+1?a.slice(0,b+1)+"."+a.slice(b+1):a+Array(b-a.length+2).join("0")}function tj(a){return a}function uj(a){function b(m){function p(N){var T=y,f=I,n,u;if("c"===x)f=Q(N)+f,N="";else{N=+N;var r=0>N;N=Q(Math.abs(N),J);if(P){var t=N.length,z=1,D=-1;a:for(;z<t;++z)switch(N[z]){case ".":D=n=z;break;case "0":0===D&&(D=z);n=z;break;default:if(!+N[z])break a;0<D&&(D=0)}N=0<D?N.slice(0,
  D)+N.slice(n+1):N}r&&0===+N&&(r=!1);T=(r?"("===l?l:"-":"-"===l||"("===l?"":l)+T;f=("s"===x?vj[8+wj/3]:"")+f+(r&&"("===l?")":"");if(V)for(r=-1,n=N.length;++r<n;)if(u=N.charCodeAt(r),48>u||57<u){f=(46===u?e+N.slice(r+1):N.slice(r))+f;N=N.slice(0,r);break}}F&&!w&&(N=c(N,Infinity));u=T.length+N.length+f.length;r=u<B?Array(B-u+1).join(v):"";F&&w&&(N=c(r+N,r.length?B-f.length:Infinity),r="");switch(h){case "<":N=T+N+f+r;break;case "=":N=T+r+N+f;break;case "^":N=r.slice(0,u=r.length>>1)+T+N+f+r.slice(u);
  break;default:N=r+T+N+f}return g(N)}m=bd(m);var v=m.fill,h=m.align,l=m.sign,q=m.symbol,w=m.zero,B=m.width,F=m.comma,J=m.precision,P=m.trim,x=m.type;"n"===x?(F=!0,x="g"):xj[x]||(null==J&&(J=12),P=!0,x="g");if(w||"0"===v&&"="===h)w=!0,v="0",h="=";var y="$"===q?d[0]:"#"===q&&/[boxX]/.test(x)?"0"+x.toLowerCase():"",I="$"===q?d[1]:/[%p]/.test(x)?k:"",Q=xj[x],V=/[defgprs%]/.test(x);J=null==J?6:/[gprs]/.test(x)?Math.max(1,Math.min(21,J)):Math.max(0,Math.min(20,J));p.toString=function(){return m+""};return p}
  var c=a.grouping&&a.thousands?gq(a.grouping,a.thousands):tj,d=a.currency,e=a.decimal,g=a.numerals?hq(a.numerals):tj,k=a.percent||"%";return{format:b,formatPrefix:function(m,p){var v=b((m=bd(m),m.type="f",m));m=3*Math.max(-8,Math.min(8,Math.floor(nc(p)/3)));var h=Math.pow(10,-m),l=vj[8+m/3];return function(q){return v(h*q)+l}}}}function yj(a){re=uj(a);d3.format=re.format;d3.formatPrefix=re.formatPrefix;return re}function zj(a){return Math.max(0,-nc(Math.abs(a)))}function Aj(a,b){return Math.max(0,
  3*Math.max(-8,Math.min(8,Math.floor(nc(b)/3)))-nc(Math.abs(a)))}function Bj(a,b){a=Math.abs(a);b=Math.abs(b)-a;return Math.max(0,nc(b)-nc(a))+1}function fb(){this.reset()}function Cj(a,b,c){var d=a.s=b+c,e=d-b;a.t=b-(d-e)+(c-e)}function Dj(a){return 1<a?0:-1>a?oa:Math.acos(a)}function La(a){return 1<a?wa:-1>a?-wa:Math.asin(a)}function Ej(a){return(a=ca(a/2))*a}function xa(){}function se(a,b){if(a&&Fj.hasOwnProperty(a.type))Fj[a.type](a,b)}function tg(a,b,c){var d=-1;c=a.length-c;for(b.lineStart();++d<
  c;){var e=a[d];b.point(e[0],e[1],e[2])}b.lineEnd()}function Gj(a,b){var c=-1,d=a.length;for(b.polygonStart();++c<d;)tg(a[c],b,1);b.polygonEnd()}function gb(a,b){if(a&&Hj.hasOwnProperty(a.type))Hj[a.type](a,b);else se(a,b)}function jq(){lb.point=kq}function lq(){Ij(Jj,Kj)}function kq(a,b){lb.point=Ij;Jj=a;Kj=b;a*=ia;b*=ia;ug=a;vg=da(b=b/2+te);wg=ca(b)}function Ij(a,b){a*=ia;b*=ia;b=b/2+te;var c=a-ug,d=0<=c?1:-1,e=d*c;c=da(b);b=ca(b);var g=wg*b,k=vg*c+g*da(e);d=g*d*ca(e);ue.add(Ma(d,k));ug=a;vg=c;wg=
  b}function ve(a){return[Ma(a[1],a[0]),La(a[2])]}function Vb(a){var b=a[0];a=a[1];var c=da(a);return[c*da(b),c*ca(b),ca(a)]}function we(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]}function oc(a,b){return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]}function xg(a,b){a[0]+=b[0];a[1]+=b[1];a[2]+=b[2]}function xe(a,b){return[a[0]*b,a[1]*b,a[2]*b]}function ye(a){var b=Ba(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);a[0]/=b;a[1]/=b;a[2]/=b}function yg(a,b){Hb.push(tb=[za=a,Aa=a]);b<Wa&&(Wa=b);b>Za&&(Za=
  b)}function Lj(a,b){var c=Vb([a*ia,b*ia]);if(pc){var d=oc(pc,c);d=oc([d[1],-d[0],0],d);ye(d);d=ve(d);var e=a-Wb,g=0<e?1:-1,k=d[0]*va*g;e=180<ra(e);e^(g*Wb<k&&k<g*a)?(d=d[1]*va,d>Za&&(Za=d)):(k=(k+360)%360-180,e^(g*Wb<k&&k<g*a))?(d=-d[1]*va,d<Wa&&(Wa=d)):(b<Wa&&(Wa=b),b>Za&&(Za=b));e?a<Wb?Xa(za,a)>Xa(za,Aa)&&(Aa=a):Xa(a,Aa)>Xa(za,Aa)&&(za=a):Aa>=za?(a<za&&(za=a),a>Aa&&(Aa=a)):a>Wb?Xa(za,a)>Xa(za,Aa)&&(Aa=a):Xa(a,Aa)>Xa(za,Aa)&&(za=a)}else Hb.push(tb=[za=a,Aa=a]);b<Wa&&(Wa=b);b>Za&&(Za=b);pc=c;Wb=a}
  function Mj(){ub.point=Lj}function Nj(){tb[0]=za;tb[1]=Aa;ub.point=yg;pc=null}function Oj(a,b){if(pc){var c=a-Wb;cd.add(180<ra(c)?c+(0<c?360:-360):c)}else Pj=a,Qj=b;lb.point(a,b);Lj(a,b)}function mq(){lb.lineStart()}function nq(){Oj(Pj,Qj);lb.lineEnd();1E-6<ra(cd)&&(za=-(Aa=180));tb[0]=za;tb[1]=Aa;pc=null}function Xa(a,b){return 0>(b-=a)?b+360:b}function oq(a,b){return a[0]-b[0]}function Rj(a,b){return a[0]<=a[1]?a[0]<=b&&b<=a[1]:b<a[0]||a[1]<b}function zg(a,b){a*=ia;b*=ia;var c=da(b);dd(c*da(a),
  c*ca(a),ca(b))}function dd(a,b,c){++ed;ze+=(a-ze)/ed;Ae+=(b-Ae)/ed;Be+=(c-Be)/ed}function Sj(){hb.point=pq}function pq(a,b){a*=ia;b*=ia;var c=da(b);Na=c*da(a);Oa=c*ca(a);Pa=ca(b);hb.point=qq;dd(Na,Oa,Pa)}function qq(a,b){a*=ia;b*=ia;var c=da(b),d=c*da(a);a=c*ca(a);b=ca(b);var e=Ma(Ba((e=Oa*b-Pa*a)*e+(e=Pa*d-Na*b)*e+(e=Na*a-Oa*d)*e),Na*d+Oa*a+Pa*b);Ce+=e;De+=e*(Na+(Na=d));Ee+=e*(Oa+(Oa=a));Fe+=e*(Pa+(Pa=b));dd(Na,Oa,Pa)}function Tj(){hb.point=zg}function rq(){hb.point=sq}function tq(){Uj(Vj,Wj);hb.point=
  zg}function sq(a,b){Vj=a;Wj=b;a*=ia;b*=ia;hb.point=Uj;var c=da(b);Na=c*da(a);Oa=c*ca(a);Pa=ca(b);dd(Na,Oa,Pa)}function Uj(a,b){a*=ia;b*=ia;var c=da(b),d=c*da(a);a=c*ca(a);b=ca(b);c=Oa*b-Pa*a;var e=Pa*d-Na*b,g=Na*a-Oa*d,k=Ba(c*c+e*e+g*g),m=La(k);k=k&&-m/k;Ag+=k*c;Bg+=k*e;Cg+=k*g;Ce+=m;De+=m*(Na+(Na=d));Ee+=m*(Oa+(Oa=a));Fe+=m*(Pa+(Pa=b));dd(Na,Oa,Pa)}function qc(a){return function(){return a}}function Dg(a,b){function c(d,e){return d=a(d,e),b(d[0],d[1])}a.invert&&b.invert&&(c.invert=function(d,e){return d=
  b.invert(d,e),d&&a.invert(d[0],d[1])});return c}function Eg(a,b){return[a>oa?a-Sa:a<-oa?a+Sa:a,b]}function Fg(a,b,c){return(a%=Sa)?b||c?Dg(Xj(a),Yj(b,c)):Xj(a):b||c?Yj(b,c):Eg}function Zj(a){return function(b,c){return b+=a,[b>oa?b-Sa:b<-oa?b+Sa:b,c]}}function Xj(a){var b=Zj(a);b.invert=Zj(-a);return b}function Yj(a,b){function c(m,p){var v=da(p),h=da(m)*v;m=ca(m)*v;p=ca(p);v=p*d+h*e;return[Ma(m*g-v*k,h*d-p*e),La(v*g+m*k)]}var d=da(a),e=ca(a),g=da(b),k=ca(b);c.invert=function(m,p){var v=da(p),h=da(m)*
  v;m=ca(m)*v;p=ca(p);v=p*g-m*k;return[Ma(m*g+p*k,h*d+v*e),La(v*d-h*e)]};return c}function ak(a){function b(c){c=a(c[0]*ia,c[1]*ia);return c[0]*=va,c[1]*=va,c}a=Fg(a[0]*ia,a[1]*ia,2<a.length?a[2]*ia:0);b.invert=function(c){c=a.invert(c[0]*ia,c[1]*ia);return c[0]*=va,c[1]*=va,c};return b}function bk(a,b,c,d,e,g){if(c){var k=da(b),m=ca(b);c*=d;if(null==e)e=b+d*Sa,g=b-c/2;else if(e=ck(k,e),g=ck(k,g),0<d?e<g:e>g)e+=d*Sa;for(;0<d?e>g:e<g;e-=c)b=ve([k,-m*da(e),-m*ca(e)]),a.point(b[0],b[1])}}function ck(a,
  b){b=Vb(b);b[0]-=a;ye(b);a=Dj(-b[1]);return((0>-b[2]?-a:a)+Sa-1E-6)%Sa}function dk(){var a=[],b;return{point:function(c,d){b.push([c,d])},lineStart:function(){a.push(b=[])},lineEnd:xa,rejoin:function(){1<a.length&&a.push(a.pop().concat(a.shift()))},result:function(){var c=a;a=[];b=null;return c}}}function Ge(a,b){return 1E-6>ra(a[0]-b[0])&&1E-6>ra(a[1]-b[1])}function He(a,b,c,d){this.x=a;this.z=b;this.o=c;this.e=d;this.v=!1;this.n=this.p=null}function ek(a,b,c,d,e){var g=[],k=[];a.forEach(function(l){if(!(0>=
  (q=l.length-1))){var q,w=l[0],B=l[q];if(Ge(w,B)){e.lineStart();for(m=0;m<q;++m)e.point((w=l[m])[0],w[1]);e.lineEnd()}else g.push(q=new He(w,l,null,!0)),k.push(q.o=new He(w,null,q,!1)),g.push(q=new He(B,l,null,!1)),k.push(q.o=new He(B,null,q,!0))}});if(g.length){k.sort(b);fk(g);fk(k);var m=0;for(a=k.length;m<a;++m)k[m].e=c=!c;c=g[0];for(var p;;){for(var v=c,h=!0;v.v;)if((v=v.n)===c)return;b=v.z;e.lineStart();do{v.v=v.o.v=!0;if(v.e){if(h)for(m=0,a=b.length;m<a;++m)e.point((p=b[m])[0],p[1]);else d(v.x,
  v.n.x,1,e);v=v.n}else{if(h)for(b=v.p.z,m=b.length-1;0<=m;--m)e.point((p=b[m])[0],p[1]);else d(v.x,v.p.x,-1,e);v=v.p}v=v.o;b=v.z;h=!h}while(!v.v);e.lineEnd()}}}function fk(a){if(b=a.length){for(var b,c=0,d=a[0],e;++c<b;)d.n=e=a[c],e.p=d,d=e;d.n=e=a[0];e.p=d}}function gk(a,b){var c=b[0];b=b[1];var d=ca(b),e=[ca(c),-da(c),0],g=0,k=0;Gg.reset();1===d?b=wa+1E-6:-1===d&&(b=-wa-1E-6);d=0;for(var m=a.length;d<m;++d)if(v=(p=a[d]).length){var p,v,h=p[v-1],l=h[0],q=h[1]/2+te,w=ca(q),B=da(q);for(q=0;q<v;++q,
  l=J,w=P,B=x,h=F){var F=p[q],J=F[0];x=F[1]/2+te;var P=ca(x),x=da(x),y=J-l,I=0<=y?1:-1,Q=I*y,V=Q>oa;w*=P;Gg.add(Ma(w*I*ca(Q),B*x+w*da(Q)));g+=V?y+I*Sa:y;V^l>=c^J>=c&&(h=oc(Vb(h),Vb(F)),ye(h),l=oc(e,h),ye(l),l=(V^0<=y?-1:1)*La(l[2]),b>l||b===l&&(h[0]||h[1]))&&(k+=V^0<=y?1:-1)}}return(-1E-6>g||1E-6>g&&-1E-6>Gg)^k&1}function hk(a,b,c,d){return function(e){function g(I,Q){a(I,Q)&&e.point(I,Q)}function k(I,Q){q.point(I,Q)}function m(){y.point=k;q.lineStart()}function p(){y.point=g;q.lineEnd()}function v(I,
  Q){x.push([I,Q]);B.point(I,Q)}function h(){B.lineStart();x=[]}function l(){v(x[0][0],x[0][1]);B.lineEnd();var I=B.clean(),Q=w.result(),V=Q.length,N;x.pop();J.push(x);x=null;if(V)if(I&1){if(V=Q[0],0<(Q=V.length-1)){F||(e.polygonStart(),F=!0);e.lineStart();for(I=0;I<Q;++I)e.point((N=V[I])[0],N[1]);e.lineEnd()}}else 1<V&&I&2&&Q.push(Q.pop().concat(Q.shift())),P.push(Q.filter(uq))}var q=b(e),w=dk(),B=b(w),F=!1,J,P,x,y={point:g,lineStart:m,lineEnd:p,polygonStart:function(){y.point=v;y.lineStart=h;y.lineEnd=
  l;P=[];J=[]},polygonEnd:function(){y.point=g;y.lineStart=m;y.lineEnd=p;P=If(P);var I=gk(J,d);P.length?(F||(e.polygonStart(),F=!0),ek(P,vq,I,c,e)):I&&(F||(e.polygonStart(),F=!0),e.lineStart(),c(null,null,1,e),e.lineEnd());F&&(e.polygonEnd(),F=!1);P=J=null},sphere:function(){e.polygonStart();e.lineStart();c(null,null,1,e);e.lineEnd();e.polygonEnd()}};return y}}function uq(a){return 1<a.length}function vq(a,b){return(0>(a=a.x)[0]?a[1]-wa-1E-6:wa-a[1])-(0>(b=b.x)[0]?b[1]-wa-1E-6:wa-b[1])}function ik(a){function b(p,
  v){return da(p)*da(v)>e}function c(p,v,h){var l=Vb(p),q=Vb(v),w=[1,0,0];q=oc(l,q);var B=we(q,q);l=q[0];var F=B-l*l;if(!F)return!h&&p;B=e*B/F;F=-e*l/F;l=oc(w,q);w=xe(w,B);q=xe(q,F);xg(w,q);q=we(w,l);B=we(l,l);F=q*q-B*(we(w,w)-1);if(!(0>F)){var J=Ba(F);F=xe(l,(-q-J)/B);xg(F,w);F=ve(F);if(!h)return F;h=p[0];var P=v[0];p=p[1];v=v[1];if(P<h){var x=h;h=P;P=x}var y=P-h,I=1E-6>ra(y-oa);!I&&v<p&&(x=p,p=v,v=x);if(I||1E-6>y?I?0<p+v^F[1]<(1E-6>ra(F[0]-h)?p:v):p<=F[1]&&F[1]<=v:y>oa^(h<=F[0]&&F[0]<=P))return v=
  xe(l,(-q+J)/B),xg(v,w),[F,ve(v)]}}function d(p,v){var h=k?a:oa-a,l=0;p<-h?l|=1:p>h&&(l|=2);v<-h?l|=4:v>h&&(l|=8);return l}var e=da(a),g=6*ia,k=0<e,m=1E-6<ra(e);return hk(b,function(p){var v,h,l,q,w;return{lineStart:function(){q=l=!1;w=1},point:function(B,F){var J=[B,F],P=b(B,F);F=k?P?0:d(B,F):P?d(B+(0>B?oa:-oa),F):0;!v&&(q=l=P)&&p.lineStart();P!==l&&(B=c(v,J),!B||Ge(v,B)||Ge(J,B))&&(J[0]+=1E-6,J[1]+=1E-6,P=b(J[0],J[1]));if(P!==l)w=0,P?(p.lineStart(),B=c(J,v),p.point(B[0],B[1])):(B=c(v,J),p.point(B[0],
  B[1]),p.lineEnd()),v=B;else if(m&&v&&k^P){var x;F&h||!(x=c(J,v,!0))||(w=0,k?(p.lineStart(),p.point(x[0][0],x[0][1]),p.point(x[1][0],x[1][1]),p.lineEnd()):(p.point(x[1][0],x[1][1]),p.lineEnd(),p.lineStart(),p.point(x[0][0],x[0][1])))}!P||v&&Ge(v,J)||p.point(J[0],J[1]);v=J;l=P;h=F},lineEnd:function(){l&&p.lineEnd();v=null},clean:function(){return w|(q&&l)<<1}}},function(p,v,h,l){bk(l,a,g,h,p,v)},k?[0,-a]:[-oa,a-oa])}function wq(a,b,c,d,e,g){var k=a[0],m=a[1],p=0,v=1,h=b[0]-k,l=b[1]-m;c-=k;if(h||!(0<
  c)){c/=h;if(0>h){if(c<p)return;c<v&&(v=c)}else if(0<h){if(c>v)return;c>p&&(p=c)}c=e-k;if(h||!(0>c)){c/=h;if(0>h){if(c>v)return;c>p&&(p=c)}else if(0<h){if(c<p)return;c<v&&(v=c)}c=d-m;if(l||!(0<c)){c/=l;if(0>l){if(c<p)return;c<v&&(v=c)}else if(0<l){if(c>v)return;c>p&&(p=c)}c=g-m;if(l||!(0>c)){c/=l;if(0>l){if(c>v)return;c>p&&(p=c)}else if(0<l){if(c<p)return;c<v&&(v=c)}0<p&&(a[0]=k+p*h,a[1]=m+p*l);1>v&&(b[0]=k+v*h,b[1]=m+v*l);return!0}}}}}function Ie(a,b,c,d){function e(p,v,h,l){var q=0,w=0;if(null==
  p||(q=g(p,h))!==(w=g(v,h))||0>m(p,v)^0<h){do l.point(0===q||3===q?a:c,1<q?d:b);while((q=(q+h+4)%4)!==w)}else l.point(v[0],v[1])}function g(p,v){return 1E-6>ra(p[0]-a)?0<v?0:3:1E-6>ra(p[0]-c)?0<v?2:1:1E-6>ra(p[1]-b)?0<v?1:0:0<v?3:2}function k(p,v){return m(p.x,v.x)}function m(p,v){var h=g(p,1),l=g(v,1);return h!==l?h-l:0===h?v[1]-p[1]:1===h?p[0]-v[0]:2===h?p[1]-v[1]:v[0]-p[0]}return function(p){function v(f,n){a<=f&&f<=c&&b<=n&&n<=d&&l.point(f,n)}function h(f,n){var u=a<=f&&f<=c&&b<=n&&n<=d;B&&F.push([f,
  n]);if(V)J=f,P=n,x=u,V=!1,u&&(l.lineStart(),l.point(f,n));else if(u&&Q)l.point(f,n);else{var r=[y=Math.max(-1E9,Math.min(1E9,y)),I=Math.max(-1E9,Math.min(1E9,I))],t=[f=Math.max(-1E9,Math.min(1E9,f)),n=Math.max(-1E9,Math.min(1E9,n))];wq(r,t,a,b,c,d)?(Q||(l.lineStart(),l.point(r[0],r[1])),l.point(t[0],t[1]),u||l.lineEnd(),N=!1):u&&(l.lineStart(),l.point(f,n),N=!1)}y=f;I=n;Q=u}var l=p,q=dk(),w,B,F,J,P,x,y,I,Q,V,N,T={point:v,lineStart:function(){T.point=h;B&&B.push(F=[]);V=!0;Q=!1;y=I=NaN},lineEnd:function(){w&&
  (h(J,P),x&&Q&&q.rejoin(),w.push(q.result()));T.point=v;Q&&l.lineEnd()},polygonStart:function(){l=q;w=[];B=[];N=!0},polygonEnd:function(){for(var f,n=f=0,u=B.length;n<u;++n){var r=B[n],t=1,z=r.length,D=r[0],A=D[0];for(D=D[1];t<z;++t){var C=A;var G=D;D=r[t];A=D[0];D=D[1];G<=d?D>d&&(A-C)*(d-G)>(D-G)*(a-C)&&++f:D<=d&&(A-C)*(d-G)<(D-G)*(a-C)&&--f}}n=N&&f;u=(w=If(w)).length;if(n||u)p.polygonStart(),n&&(p.lineStart(),e(null,null,1,p),p.lineEnd()),u&&ek(w,k,f,e,p),p.polygonEnd();l=p;w=B=F=null}};return T}}
  function xq(){rc.point=rc.lineEnd=xa}function yq(a,b){a*=ia;b*=ia;Hg=a;Je=ca(b);Ke=da(b);rc.point=zq}function zq(a,b){a*=ia;b*=ia;var c=ca(b);b=da(b);var d=ra(a-Hg),e=da(d);d=ca(d);d*=b;var g=Ke*c-Je*b*e;e=Je*c+Ke*b*e;Ig.add(Ma(Ba(d*d+g*g),e));Hg=a;Je=c;Ke=b}function jk(a){Ig.reset();gb(a,rc);return+Ig}function sc(a,b){Jg[0]=a;Jg[1]=b;return jk(Aq)}function Le(a,b){return a&&kk.hasOwnProperty(a.type)?kk[a.type](a,b):!1}function lk(a,b){var c=sc(a[0],a[1]),d=sc(a[0],b);a=sc(b,a[1]);return d+a<=c+1E-6}
  function mk(a,b){return!!gk(a.map(Bq),nk(b))}function Bq(a){return a=a.map(nk),a.pop(),a}function nk(a){return[a[0]*ia,a[1]*ia]}function ok(a,b,c){var d=Ta(a,b-1E-6,c).concat(b);return function(e){return d.map(function(g){return[e,g]})}}function pk(a,b,c){var d=Ta(a,b-1E-6,c).concat(b);return function(e){return d.map(function(g){return[g,e]})}}function qk(){function a(){return{type:"MultiLineString",coordinates:b()}}function b(){return Ta(Me(g/q)*q,e,q).map(J).concat(Ta(Me(v/w)*w,p,w).map(P)).concat(Ta(Me(d/
  h)*h,c,h).filter(function(y){return 1E-6<ra(y%q)}).map(B)).concat(Ta(Me(m/l)*l,k,l).filter(function(y){return 1E-6<ra(y%w)}).map(F))}var c,d,e,g,k,m,p,v,h=10,l=h,q=90,w=360,B,F,J,P,x=2.5;a.lines=function(){return b().map(function(y){return{type:"LineString",coordinates:y}})};a.outline=function(){return{type:"Polygon",coordinates:[J(g).concat(P(p).slice(1),J(e).reverse().slice(1),P(v).reverse().slice(1))]}};a.extent=function(y){return arguments.length?a.extentMajor(y).extentMinor(y):a.extentMinor()};
  a.extentMajor=function(y){if(!arguments.length)return[[g,v],[e,p]];g=+y[0][0];e=+y[1][0];v=+y[0][1];p=+y[1][1];g>e&&(y=g,g=e,e=y);v>p&&(y=v,v=p,p=y);return a.precision(x)};a.extentMinor=function(y){if(!arguments.length)return[[d,m],[c,k]];d=+y[0][0];c=+y[1][0];m=+y[0][1];k=+y[1][1];d>c&&(y=d,d=c,c=y);m>k&&(y=m,m=k,k=y);return a.precision(x)};a.step=function(y){return arguments.length?a.stepMajor(y).stepMinor(y):a.stepMinor()};a.stepMajor=function(y){if(!arguments.length)return[q,w];q=+y[0];w=+y[1];
  return a};a.stepMinor=function(y){if(!arguments.length)return[h,l];h=+y[0];l=+y[1];return a};a.precision=function(y){if(!arguments.length)return x;x=+y;B=ok(m,k,90);F=pk(d,c,x);J=ok(v,p,90);P=pk(g,e,x);return a};return a.extentMajor([[-180,-89.999999],[180,89.999999]]).extentMinor([[-180,-80.000001],[180,80.000001]])}function Xb(a){return a}function Cq(){vb.point=Dq}function Dq(a,b){vb.point=rk;sk=Kg=a;tk=Lg=b}function rk(a,b){Mg.add(Lg*a-Kg*b);Kg=a;Lg=b}function Eq(){rk(sk,tk)}function Yb(a,b){Ng+=
  a;Og+=b;++fd}function uk(){$a.point=Fq}function Fq(a,b){$a.point=Gq;Yb(mb=a,nb=b)}function Gq(a,b){var c=a-mb,d=b-nb;c=Ba(c*c+d*d);Ne+=c*(mb+a)/2;Oe+=c*(nb+b)/2;tc+=c;Yb(mb=a,nb=b)}function vk(){$a.point=Yb}function Hq(){$a.point=Iq}function Jq(){wk(xk,yk)}function Iq(a,b){$a.point=wk;Yb(xk=mb=a,yk=nb=b)}function wk(a,b){var c=a-mb,d=b-nb;c=Ba(c*c+d*d);Ne+=c*(mb+a)/2;Oe+=c*(nb+b)/2;tc+=c;c=nb*a-mb*b;Pg+=c*(mb+a);Qg+=c*(nb+b);gd+=3*c;Yb(mb=a,nb=b)}function zk(a){this._context=a}function Kq(a,b){hd.point=
  Ak;Bk=id=a;Ck=jd=b}function Ak(a,b){id-=a;jd-=b;Rg.add(Ba(id*id+jd*jd));id=a;jd=b}function Dk(){this._string=[]}function Ek(a){return"m0,"+a+"a"+a+","+a+" 0 1,1 0,"+-2*a+"a"+a+","+a+" 0 1,1 0,"+2*a+"z"}function kd(a){return function(b){var c=new Sg,d;for(d in a)c[d]=a[d];c.stream=b;return c}}function Sg(){}function Tg(a,b,c){var d=a.clipExtent&&a.clipExtent();a.scale(150).translate([0,0]);null!=d&&a.clipExtent(null);gb(c,a.stream(Pe));b(Pe.result());null!=d&&a.clipExtent(d);return a}function uc(a,
  b,c){return Tg(a,function(d){var e=b[1][0]-b[0][0],g=b[1][1]-b[0][1],k=Math.min(e/(d[1][0]-d[0][0]),g/(d[1][1]-d[0][1]));e=+b[0][0]+(e-k*(d[1][0]+d[0][0]))/2;d=+b[0][1]+(g-k*(d[1][1]+d[0][1]))/2;a.scale(150*k).translate([e,d])},c)}function Ug(a,b,c){return Tg(a,function(d){var e=+b,g=e/(d[1][0]-d[0][0]);e=(e-g*(d[1][0]+d[0][0]))/2;d=-g*d[0][1];a.scale(150*g).translate([e,d])},c)}function Vg(a,b,c){return Tg(a,function(d){var e=+b,g=e/(d[1][1]-d[0][1]),k=-g*d[0][0];d=(e-g*(d[1][1]+d[0][1]))/2;a.scale(150*
  g).translate([k,d])},c)}function Fk(a){return kd({point:function(b,c){b=a(b,c);this.stream.point(b[0],b[1])}})}function Gk(a,b){function c(d,e,g,k,m,p,v,h,l,q,w,B,F,J){var P=v-d,x=h-e,y=P*P+x*x;if(y>4*b&&F--){var I=k+q,Q=m+w,V=p+B,N=Ba(I*I+Q*Q+V*V),T=La(V/=N),f=1E-6>ra(ra(V)-1)||1E-6>ra(g-l)?(g+l)/2:Ma(Q,I),n=a(f,T);T=n[0];n=n[1];var u=T-d,r=n-e,t=x*u-P*r;if(t*t/y>b||.3<ra((P*u+x*r)/y-.5)||k*q+m*w+p*B<Lq)c(d,e,g,k,m,p,T,n,f,I/=N,Q/=N,V,F,J),J.point(T,n),c(T,n,f,I,Q,V,v,h,l,q,w,B,F,J)}}return function(d){function e(T,
  f){T=a(T,f);d.point(T[0],T[1])}function g(){x=NaN;N.point=k;d.lineStart()}function k(T,f){var n=Vb([T,f]);f=a(T,f);c(x,y,P,I,Q,V,x=f[0],y=f[1],P=T,I=n[0],Q=n[1],V=n[2],16,d);d.point(x,y)}function m(){N.point=e;d.lineEnd()}function p(){g();N.point=v;N.lineEnd=h}function v(T,f){k(l=T,f);q=x;w=y;B=I;F=Q;J=V;N.point=k}function h(){c(x,y,P,I,Q,V,q,w,l,B,F,J,16,d);N.lineEnd=m;m()}var l,q,w,B,F,J,P,x,y,I,Q,V,N={point:e,lineStart:g,lineEnd:m,polygonStart:function(){d.polygonStart();N.lineStart=p},polygonEnd:function(){d.polygonEnd();
  N.lineStart=g}};return N}}function Mq(a){return kd({point:function(b,c){b=a(b,c);return this.stream.point(b[0],b[1])}})}function Nq(a,b,c){function d(e,g){return[b+a*e,c-a*g]}d.invert=function(e,g){return[(e-b)/a,(c-g)/a]};return d}function Hk(a,b,c,d){function e(q,w){return[k*q-m*w+b,c-m*q-k*w]}var g=da(d);d=ca(d);var k=g*a,m=d*a,p=g/a,v=d/a,h=(d*c-g*b)/a,l=(d*b+g*c)/a;e.invert=function(q,w){return[p*q-v*w+h,l-v*q-p*w]};return e}function ob(a){return Wg(function(){return a})()}function Wg(a){function b(t){return n(t[0]*
  ia,t[1]*ia)}function c(t){return(t=n.invert(t[0],t[1]))&&[t[0]*va,t[1]*va]}function d(){var t=Hk(k,0,0,F).apply(null,g(v,h));t=(F?Hk:Nq)(k,m-t[0],p-t[1],F);B=Fg(l,q,w);f=Dg(g,t);n=Dg(B,f);t=f;T=+N?Gk(t,N):Fk(t);return e()}function e(){u=r=null;return b}var g,k=150,m=480,p=250,v=0,h=0,l=0,q=0,w=0,B,F=0,J=null,P=Xg,x=null,y,I,Q,V=Xb,N=.5,T,f,n,u,r;b.stream=function(t){return u&&r===t?u:u=Oq(Mq(B)(P(T(V(r=t)))))};b.preclip=function(t){return arguments.length?(P=t,J=void 0,e()):P};b.postclip=function(t){return arguments.length?
  (V=t,x=y=I=Q=null,e()):V};b.clipAngle=function(t){return arguments.length?(P=+t?ik(J=t*ia):(J=null,Xg),e()):J*va};b.clipExtent=function(t){return arguments.length?(V=null==t?(x=y=I=Q=null,Xb):Ie(x=+t[0][0],y=+t[0][1],I=+t[1][0],Q=+t[1][1]),e()):null==x?null:[[x,y],[I,Q]]};b.scale=function(t){return arguments.length?(k=+t,d()):k};b.translate=function(t){return arguments.length?(m=+t[0],p=+t[1],d()):[m,p]};b.center=function(t){return arguments.length?(v=t[0]%360*ia,h=t[1]%360*ia,d()):[v*va,h*va]};b.rotate=
  function(t){return arguments.length?(l=t[0]%360*ia,q=t[1]%360*ia,w=2<t.length?t[2]%360*ia:0,d()):[l*va,q*va,w*va]};b.angle=function(t){return arguments.length?(F=t%360*ia,d()):F*va};b.precision=function(t){if(arguments.length){var z=f;var D=N=t*t;z=(T=+D?Gk(z,D):Fk(z),e())}else z=Ba(N);return z};b.fitExtent=function(t,z){return uc(b,t,z)};b.fitSize=function(t,z){return uc(b,[[0,0],t],z)};b.fitWidth=function(t,z){return Ug(b,t,z)};b.fitHeight=function(t,z){return Vg(b,t,z)};return function(){g=a.apply(this,
  arguments);b.invert=g.invert&&c;return d()}}function Yg(a){var b=0,c=oa/3,d=Wg(a);a=d(b,c);a.parallels=function(e){return arguments.length?d(b=e[0]*ia,c=e[1]*ia):[b*va,c*va]};return a}function Pq(a){function b(d,e){return[d*c,ca(e)/c]}var c=da(a);b.invert=function(d,e){return[d/c,La(e*c)]};return b}function Ik(a,b){function c(m,p){p=Ba(g-2*e*ca(p))/e;return[p*ca(m*=e),k-p*da(m)]}var d=ca(a),e=(d+ca(b))/2;if(1E-6>ra(e))return Pq(a);var g=1+d*(2*e-d),k=Ba(g)/e;c.invert=function(m,p){p=k-p;return[Ma(m,
  ra(p))/e*ld(p),La((g-(m*m+p*p)*e*e)/(2*e))]};return c}function Qe(){return Yg(Ik).scale(155.424).center([0,33.6442])}function Jk(){return Qe().parallels([29.5,45.5]).scale(1070).translate([480,250]).rotate([96,0]).center([-.6,38.7])}function Qq(a){var b=a.length;return{point:function(c,d){for(var e=-1;++e<b;)a[e].point(c,d)},sphere:function(){for(var c=-1;++c<b;)a[c].sphere()},lineStart:function(){for(var c=-1;++c<b;)a[c].lineStart()},lineEnd:function(){for(var c=-1;++c<b;)a[c].lineEnd()},polygonStart:function(){for(var c=
  -1;++c<b;)a[c].polygonStart()},polygonEnd:function(){for(var c=-1;++c<b;)a[c].polygonEnd()}}}function Kk(a){return function(b,c){var d=da(b),e=da(c);d=a(d*e);return[d*e*ca(b),d*ca(c)]}}function md(a){return function(b,c){var d=Ba(b*b+c*c),e=a(d),g=ca(e);e=da(e);return[Ma(b*g,d*e),La(d&&c*g/d)]}}function nd(a,b){return[a,Re(vc((wa+b)/2))]}function Lk(a){function b(){var l=oa*e(),q=c(ak(c.rotate()).invert([0,0]));return k(null==m?[[q[0]-l,q[1]-l],[q[0]+l,q[1]+l]]:a===nd?[[Math.max(q[0]-l,m),p],[Math.min(q[0]+
  l,v),h]]:[[m,Math.max(q[1]-l,p)],[v,Math.min(q[1]+l,h)]])}var c=ob(a),d=c.center,e=c.scale,g=c.translate,k=c.clipExtent,m=null,p,v,h;c.scale=function(l){return arguments.length?(e(l),b()):e()};c.translate=function(l){return arguments.length?(g(l),b()):g()};c.center=function(l){return arguments.length?(d(l),b()):d()};c.clipExtent=function(l){return arguments.length?(null==l?m=p=v=h=null:(m=+l[0][0],p=+l[0][1],v=+l[1][0],h=+l[1][1]),b()):null==m?null:[[m,p],[v,h]]};return b()}function Mk(a,b){function c(k,
  m){0<g?m<-wa+1E-6&&(m=-wa+1E-6):m>wa-1E-6&&(m=wa-1E-6);m=g/Zg(vc((wa+m)/2),e);return[m*ca(e*k),g-m*da(e*k)]}var d=da(a),e=a===b?ca(a):Re(d/da(b))/Re(vc((wa+b)/2)/vc((wa+a)/2)),g=d*Zg(vc((wa+a)/2),e)/e;if(!e)return nd;c.invert=function(k,m){m=g-m;var p=ld(e)*Ba(k*k+m*m);return[Ma(k,ra(m))/e*ld(m),2*wc(Zg(g/p,1/e))-wa]};return c}function od(a,b){return[a,b]}function Nk(a,b){function c(k,m){m=g-m;k*=e;return[m*ca(k),g-m*da(k)]}var d=da(a),e=a===b?ca(a):(d-da(b))/(b-a),g=d/e+a;if(1E-6>ra(e))return od;
  c.invert=function(k,m){m=g-m;return[Ma(k,ra(m))/e*ld(m),g-ld(e)*Ba(k*k+m*m)]};return c}function $g(a,b){b=La(Se*ca(b));var c=b*b,d=c*c*c;return[a*da(b)/(Se*(1.340264+3*-.081106*c+d*(7*8.93E-4+.034164*c))),b*(1.340264+-.081106*c+d*(8.93E-4+.003796*c))]}function ah(a,b){var c=da(b),d=da(a)*c;return[c*ca(a)/d,ca(b)/d]}function Te(a,b,c,d){return 1===a&&1===b&&0===c&&0===d?Xb:kd({point:function(e,g){this.stream.point(e*a+c,g*b+d)}})}function bh(a,b){var c=b*b,d=c*c;return[a*(.8707-.131979*c+d*(-.013791+
  d*(.003971*c-.001529*d))),b*(1.007226+c*(.015085+d*(-.044475+.028874*c-.005916*d)))]}function ch(a,b){return[da(b)*ca(a),ca(b)]}function dh(a,b){var c=da(b),d=1+da(a)*c;return[c*ca(a)/d,ca(b)/d]}function eh(a,b){return[Re(vc((wa+b)/2)),-a]}function Rq(a,b){return a.parent===b.parent?1:2}function Sq(a,b){return a+b.x}function Tq(a,b){return Math.max(a,b.y)}function Uq(a){for(var b;b=a.children;)a=b[0];return a}function Vq(a){for(var b;b=a.children;)a=b[b.length-1];return a}function Wq(a){var b=0,c=
  a.children,d=c&&c.length;if(d)for(;0<=--d;)b+=c[d].value;else b=1;a.value=b}function fh(a,b){var c=new xc(a);a=+a.value&&(c.value=a.value);var d,e=[c],g,k,m,p;for(null==b&&(b=Xq);d=e.pop();)if(a&&(d.value=+d.data.value),(k=b(d.data))&&(p=k.length))for(d.children=Array(p),m=p-1;0<=m;--m)e.push(g=d.children[m]=new xc(k[m])),g.parent=d,g.depth=d.depth+1;return c.eachBefore(Ok)}function Xq(a){return a.children}function Yq(a){a.data=a.data.data}function Ok(a){var b=0;do a.height=b;while((a=a.parent)&&
  a.height<++b)}function xc(a){this.data=a;this.depth=this.height=0;this.parent=null}function Pk(a){var b=0;a=Zq.call(a);for(var c=a.length,d,e;c;)e=Math.random()*c--|0,d=a[c],a[c]=a[e],a[e]=d;c=a.length;d=[];for(var g;b<c;)if(e=a[b],g&&Qk(g,e))++b;else{a:if(b=d,gh(e,b))b=[e];else{for(d=0;d<b.length;++d)if(Ue(e,b[d])&&gh(pd(b[d],e),b)){b=[b[d],e];break a}for(d=0;d<b.length-1;++d)for(g=d+1;g<b.length;++g)if(Ue(pd(b[d],b[g]),e)&&Ue(pd(b[d],e),b[g])&&Ue(pd(b[g],e),b[d])&&gh(Rk(b[d],b[g],e),b)){b=[b[d],
  b[g],e];break a}throw Error();}a:{b=d=b;switch(b.length){case 1:b=b[0];b={x:b.x,y:b.y,r:b.r};break a;case 2:b=pd(b[0],b[1]);break a;case 3:b=Rk(b[0],b[1],b[2]);break a}b=void 0}g=b;b=0}return g}function Ue(a,b){var c=a.r-b.r,d=b.x-a.x;a=b.y-a.y;return 0>c||c*c<d*d+a*a}function Qk(a,b){var c=a.r-b.r+1E-6,d=b.x-a.x;a=b.y-a.y;return 0<c&&c*c>d*d+a*a}function gh(a,b){for(var c=0;c<b.length;++c)if(!Qk(a,b[c]))return!1;return!0}function pd(a,b){var c=a.x,d=a.y;a=a.r;var e=b.x,g=b.y;b=b.r;var k=e-c,m=g-
  d,p=b-a,v=Math.sqrt(k*k+m*m);return{x:(c+e+k/v*p)/2,y:(d+g+m/v*p)/2,r:(v+a+b)/2}}function Rk(a,b,c){var d=a.x,e=a.y;a=a.r;var g=b.x,k=b.y,m=b.r,p=c.x,v=c.y,h=c.r;c=d-g;b=d-p;var l=e-k,q=e-v,w=m-a,B=h-a,F=d*d+e*e-a*a;k=F-g*g-k*k+m*m;v=F-p*p-v*v+h*h;p=b*l-c*q;g=(l*v-q*k)/(2*p)-d;l=(q*w-l*B)/p;q=(b*k-c*v)/(2*p)-e;c=(c*B-b*w)/p;b=l*l+c*c-1;w=2*(a+g*l+q*c);a=g*g+q*q-a*a;a=-(b?(w+Math.sqrt(w*w-4*b*a))/(2*b):a/w);return{x:d+g+l*a,y:e+q+c*a,r:a}}function Sk(a,b,c){var d=a.x-b.x,e=a.y-b.y,g=d*d+e*e;if(g){var k=
  b.r+c.r;k*=k;var m=a.r+c.r;m*=m;if(k>m){var p=(g+m-k)/(2*g);k=Math.sqrt(Math.max(0,m/g-p*p));c.x=a.x-p*d-k*e;c.y=a.y-p*e+k*d}else p=(g+k-m)/(2*g),k=Math.sqrt(Math.max(0,k/g-p*p)),c.x=b.x+p*d-k*e,c.y=b.y+p*e+k*d}else c.x=b.x+c.r,c.y=b.y}function Tk(a,b){var c=a.r+b.r-1E-6,d=b.x-a.x;a=b.y-a.y;return 0<c&&c*c>d*d+a*a}function Uk(a){var b=a._,c=a.next._,d=b.r+c.r;a=(b.x*c.r+c.x*b.r)/d;b=(b.y*c.r+c.y*b.r)/d;return a*a+b*b}function Ve(a){this._=a;this.previous=this.next=null}function Vk(a){if(!(c=a.length))return 0;
  var b,c;var d=a[0];d.x=0;d.y=0;if(!(1<c))return d.r;var e=a[1];d.x=-e.r;e.x=d.r;e.y=0;if(!(2<c))return d.r+e.r;Sk(e,d,b=a[2]);d=new Ve(d);e=new Ve(e);b=new Ve(b);d.next=b.previous=e;e.next=d.previous=b;b.next=e.previous=d;var g=3;a:for(;g<c;++g){Sk(d._,e._,b=a[g]);b=new Ve(b);var k=e.next;var m=d.previous;var p=e._.r;var v=d._.r;do if(p<=v){if(Tk(k._,b._)){e=k;d.next=e;e.previous=d;--g;continue a}p+=k._.r;k=k.next}else{if(Tk(m._,b._)){d=m;d.next=e;e.previous=d;--g;continue a}v+=m._.r;m=m.previous}while(k!==
  m.next);b.previous=d;b.next=e;d.next=e.previous=e=b;for(k=Uk(d);(b=b.next)!==e;)(m=Uk(b))<k&&(d=b,k=m);e=d.next}d=[e._];for(b=e;(b=b.next)!==e;)d.push(b._);b=Pk(d);for(g=0;g<c;++g)d=a[g],d.x-=b.x,d.y-=b.y;return b.r}function We(a){if("function"!==typeof a)throw Error();return a}function Zb(){return 0}function yc(a){return function(){return a}}function $q(a){return Math.sqrt(a.value)}function Wk(a){return function(b){b.children||(b.r=Math.max(0,+a(b)||0))}}function hh(a,b){return function(c){if(d=
  c.children){var d,e,g=d.length,k=a(c)*b||0;if(k)for(e=0;e<g;++e)d[e].r+=k;var m=Vk(d);if(k)for(e=0;e<g;++e)d[e].r-=k;c.r=m+k}}}function Xk(a){return function(b){var c=b.parent;b.r*=a;c&&(b.x=c.x+a*b.x,b.y=c.y+a*b.y)}}function Yk(a){a.x0=Math.round(a.x0);a.y0=Math.round(a.y0);a.x1=Math.round(a.x1);a.y1=Math.round(a.y1)}function qd(a,b,c,d,e){var g=a.children,k=-1,m=g.length;for(d=a.value&&(d-b)/a.value;++k<m;)a=g[k],a.y0=c,a.y1=e,a.x0=b,a.x1=b+=a.value*d}function ar(a){return a.id}function br(a){return a.parentId}
  function cr(a,b){return a.parent===b.parent?1:2}function ih(a){var b=a.children;return b?b[0]:a.t}function jh(a){var b=a.children;return b?b[b.length-1]:a.t}function Xe(a,b){this._=a;this.A=this.children=this.parent=null;this.a=this;this.s=this.c=this.m=this.z=0;this.t=null;this.i=b}function dr(a){a=new Xe(a,0);for(var b,c=[a],d,e,g;b=c.pop();)if(e=b._.children)for(b.children=Array(d=e.length),g=d-1;0<=g;--g)c.push(d=b.children[g]=new Xe(e[g],g)),d.parent=b;(a.parent=new Xe(null,0)).children=[a];
  return a}function Ye(a,b,c,d,e){var g=a.children,k=-1,m=g.length;for(e=a.value&&(e-c)/a.value;++k<m;)a=g[k],a.x0=b,a.x1=d,a.y0=c,a.y1=c+=a.value*e}function Zk(a,b,c,d,e,g){for(var k=[],m=b.children,p,v,h=p=0,l=m.length,q,w=b.value,B,F,J,P,x,y;p<l;){b=e-c;q=g-d;do B=m[h++].value;while(!B&&h<l);F=J=B;y=Math.max(q/b,b/q)/(w*a);P=B*B*y;for(x=Math.max(J/P,P/F);h<l;++h){B+=v=m[h].value;v<F&&(F=v);v>J&&(J=v);P=B*B*y;P=Math.max(J/P,P/F);if(P>x){B-=v;break}x=P}k.push(p={value:B,dice:b<q,children:m.slice(p,
  h)});p.dice?qd(p,c,d,e,w?d+=q*B/w:g):Ye(p,c,d,w?c+=b*B/w:e,g);w-=B;p=h}return k}function er(a,b,c){return(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])}function fr(a,b){return a[0]-b[0]||a[1]-b[1]}function $k(a){for(var b=a.length,c=[0,1],d=2,e=2;e<b;++e){for(;1<d&&0>=er(a[c[d-2]],a[c[d-1]],a[e]);)--d;c[d++]=e}return c.slice(0,d)}function zc(){return Math.random()}function kh(a){function b(g){var k=g+"",m=c.get(k);if(!m){if(e!==lh)return e;c.set(k,m=d.push(g))}return a[(m-1)%a.length]}var c=rb(),
  d=[],e=lh;a=null==a?[]:Ib.call(a);b.domain=function(g){if(!arguments.length)return d.slice();d=[];c=rb();for(var k=-1,m=g.length,p,v;++k<m;)c.has(v=(p=g[k])+"")||c.set(v,d.push(p));return b};b.range=function(g){return arguments.length?(a=Ib.call(g),b):a.slice()};b.unknown=function(g){return arguments.length?(e=g,b):e};b.copy=function(){return kh().domain(d).range(a).unknown(e)};return b}function mh(){function a(){var l=c().length,q=e[1]<e[0],w=e[q-0],B=e[1-q];g=(B-w)/Math.max(1,l-p+2*v);m&&(g=Math.floor(g));
  w+=(B-w-g*(l-p))*h;k=g*(1-p);m&&(w=Math.round(w),k=Math.round(k));l=Ta(l).map(function(F){return w+g*F});return d(q?l.reverse():l)}var b=kh().unknown(void 0),c=b.domain,d=b.range,e=[0,1],g,k,m=!1,p=0,v=0,h=.5;delete b.unknown;b.domain=function(l){return arguments.length?(c(l),a()):c()};b.range=function(l){return arguments.length?(e=[+l[0],+l[1]],a()):e.slice()};b.rangeRound=function(l){return e=[+l[0],+l[1]],m=!0,a()};b.bandwidth=function(){return k};b.step=function(){return g};b.round=function(l){return arguments.length?
  (m=!!l,a()):m};b.padding=function(l){return arguments.length?(p=v=Math.max(0,Math.min(1,l)),a()):p};b.paddingInner=function(l){return arguments.length?(p=Math.max(0,Math.min(1,l)),a()):p};b.paddingOuter=function(l){return arguments.length?(v=Math.max(0,Math.min(1,l)),a()):v};b.align=function(l){return arguments.length?(h=Math.max(0,Math.min(1,l)),a()):h};b.copy=function(){return mh().domain(c()).range(e).round(m).paddingInner(p).paddingOuter(v).align(h)};return a()}function al(a){var b=a.copy;a.padding=
  a.paddingOuter;delete a.paddingInner;delete a.paddingOuter;a.copy=function(){return al(b())};return a}function nh(a){return function(){return a}}function bl(a){return+a}function oh(a,b){return(b-=a=+a)?function(c){return(c-a)/b}:nh(b)}function gr(a){return function(b,c){var d=a(b=+b,c=+c);return function(e){return e<=b?0:e>=c?1:d(e)}}}function hr(a){return function(b,c){var d=a(b=+b,c=+c);return function(e){return 0>=e?b:1<=e?c:d(e)}}}function ir(a,b,c,d){var e=a[0];a=a[1];var g=b[0];b=b[1];a<e?(e=
  c(a,e),g=d(b,g)):(e=c(e,a),g=d(g,b));return function(k){return g(e(k))}}function jr(a,b,c,d){var e=Math.min(a.length,b.length)-1,g=Array(e),k=Array(e),m=-1;a[e]<a[0]&&(a=a.slice().reverse(),b=b.slice().reverse());for(;++m<e;)g[m]=c(a[m],a[m+1]),k[m]=d(b[m],b[m+1]);return function(p){var v=$b(a,p,1,e)-1;return k[v](g[v](p))}}function Ze(a,b){return b.domain(a.domain()).range(a.range()).interpolate(a.interpolate()).clamp(a.clamp())}function $e(a,b){function c(){p=2<Math.min(e.length,g.length)?jr:ir;
  v=h=null;return d}function d(l){return(v||(v=p(e,g,m?gr(a):a,k)))(+l)}var e=cl,g=cl,k=Sc,m=!1,p,v,h;d.invert=function(l){return(h||(h=p(g,e,oh,m?hr(b):b)))(+l)};d.domain=function(l){return arguments.length?(e=ph.call(l,bl),c()):e.slice()};d.range=function(l){return arguments.length?(g=Ib.call(l),c()):g.slice()};d.rangeRound=function(l){return g=Ib.call(l),k=Li,c()};d.clamp=function(l){return arguments.length?(m=!!l,c()):m};d.interpolate=function(l){return arguments.length?(k=l,c()):k};return c()}
  function Ac(a){var b=a.domain;a.ticks=function(c){var d=b();return Df(d[0],d[d.length-1],null==c?10:c)};a.tickFormat=function(c,d){var e;a:{var g=b(),k=g[0];g=g[g.length-1];c=Nb(k,g,null==c?10:c);d=bd(null==d?",f":d);switch(d.type){case "s":k=Math.max(Math.abs(k),Math.abs(g));null!=d.precision||isNaN(e=Aj(c,k))||(d.precision=e);e=d3.formatPrefix(d,k);break a;case "":case "e":case "g":case "p":case "r":null!=d.precision||isNaN(e=Bj(c,Math.max(Math.abs(k),Math.abs(g))))||(d.precision=e-("e"===d.type));
  break;case "f":case "%":null!=d.precision||isNaN(e=zj(c))||(d.precision=e-2*("%"===d.type))}e=d3.format(d)}return e};a.nice=function(c){null==c&&(c=10);var d=b(),e=0,g=d.length-1,k=d[e],m=d[g];if(m<k){var p=k;k=m;m=p;p=e;e=g;g=p}p=Nc(k,m,c);0<p?(k=Math.floor(k/p)*p,m=Math.ceil(m/p)*p,p=Nc(k,m,c)):0>p&&(k=Math.ceil(k*p)/p,m=Math.floor(m*p)/p,p=Nc(k,m,c));0<p?(d[e]=Math.floor(k/p)*p,d[g]=Math.ceil(m/p)*p,b(d)):0>p&&(d[e]=Math.ceil(k*p)/p,d[g]=Math.floor(m*p)/p,b(d));return a};return a}function dl(){var a=
  $e(oh,Va);a.copy=function(){return Ze(a,dl())};return Ac(a)}function el(){function a(c){return+c}var b=[0,1];a.invert=a;a.domain=a.range=function(c){return arguments.length?(b=ph.call(c,bl),a):b.slice()};a.copy=function(){return el().domain(b)};return Ac(a)}function fl(a,b){a=a.slice();var c=0,d=a.length-1,e=a[c],g=a[d];if(g<e){var k=c;c=d;d=k;k=e;e=g;g=k}a[c]=b.floor(e);a[d]=b.ceil(g);return a}function kr(a,b){return(b=Math.log(b/a))?function(c){return Math.log(c/a)/b}:nh(b)}function lr(a,b){return 0>
  a?function(c){return-Math.pow(-b,c)*Math.pow(-a,1-c)}:function(c){return Math.pow(b,c)*Math.pow(a,1-c)}}function mr(a){return isFinite(a)?+("1e"+a):0>a?0:a}function gl(a){return 10===a?mr:a===Math.E?Math.exp:function(b){return Math.pow(a,b)}}function hl(a){return a===Math.E?Math.log:10===a&&Math.log10||2===a&&Math.log2||(a=Math.log(a),function(b){return Math.log(b)/a})}function il(a){return function(b){return-a(-b)}}function jl(){function a(){e=hl(d);g=gl(d);0>c()[0]&&(e=il(e),g=il(g));return b}var b=
  $e(kr,lr).domain([1,10]),c=b.domain,d=10,e=hl(10),g=gl(10);b.base=function(k){return arguments.length?(d=+k,a()):d};b.domain=function(k){return arguments.length?(c(k),a()):c()};b.ticks=function(k){var m=c(),p=m[0];m=m[m.length-1];var v;if(v=m<p)h=p,p=m,m=h;var h=e(p),l=e(m);var q=null==k?10:+k;k=[];if(!(d%1)&&l-h<q)if(h=Math.round(h)-1,l=Math.round(l)+1,0<p)for(;h<l;++h){var w=1;for(q=g(h);w<d;++w){var B=q*w;if(!(B<p)){if(B>m)break;k.push(B)}}}else for(;h<l;++h)for(w=d-1,q=g(h);1<=w;--w){if(B=q*w,
  !(B<p)){if(B>m)break;k.push(B)}}else k=Df(h,l,Math.min(l-h,q)).map(g);return v?k.reverse():k};b.tickFormat=function(k,m){null==m&&(m=10===d?".0e":",");"function"!==typeof m&&(m=d3.format(m));if(Infinity===k)return m;null==k&&(k=10);var p=Math.max(1,d*k/b.ticks().length);return function(v){var h=v/g(Math.round(e(v)));h*d<d-.5&&(h*=d);return h<=p?m(v):""}};b.nice=function(){return c(fl(c(),{floor:function(k){return g(Math.floor(e(k)))},ceil:function(k){return g(Math.ceil(e(k)))}}))};b.copy=function(){return Ze(b,
  jl().base(d))};return b}function Bc(a,b){return 0>a?-Math.pow(-a,b):Math.pow(a,b)}function qh(){var a=1,b=$e(function(d,e){return(e=Bc(e,a)-(d=Bc(d,a)))?function(g){return(Bc(g,a)-d)/e}:nh(e)},function(d,e){e=Bc(e,a)-(d=Bc(d,a));return function(g){return Bc(d+e*g,1/a)}}),c=b.domain;b.exponent=function(d){return arguments.length?(a=+d,c(c())):a};b.copy=function(){return Ze(b,qh().exponent(a))};return Ac(b)}function kl(){function a(){var g=0,k=Math.max(1,d.length);for(e=Array(k-1);++g<k;)e[g-1]=Oc(c,
  g/k);return b}function b(g){if(!isNaN(g=+g))return d[$b(e,g)]}var c=[],d=[],e=[];b.invertExtent=function(g){g=d.indexOf(g);return 0>g?[NaN,NaN]:[0<g?e[g-1]:c[0],g<e.length?e[g]:c[c.length-1]]};b.domain=function(g){if(!arguments.length)return c.slice();c=[];for(var k=0,m=g.length,p;k<m;++k)(p=g[k],null==p||isNaN(p=+p))||c.push(p);c.sort(Mb);return a()};b.range=function(g){return arguments.length?(d=Ib.call(g),a()):d.slice()};b.quantiles=function(){return e.slice()};b.copy=function(){return kl().domain(c).range(d)};
  return b}function ll(){function a(m){if(m<=m)return k[$b(g,m,0,e)]}function b(){var m=-1;for(g=Array(e);++m<e;)g[m]=((m+1)*d-(m-e)*c)/(e+1);return a}var c=0,d=1,e=1,g=[.5],k=[0,1];a.domain=function(m){return arguments.length?(c=+m[0],d=+m[1],b()):[c,d]};a.range=function(m){return arguments.length?(e=(k=Ib.call(m)).length-1,b()):k.slice()};a.invertExtent=function(m){m=k.indexOf(m);return 0>m?[NaN,NaN]:1>m?[c,g[0]]:m>=e?[g[e-1],d]:[g[m-1],g[m]]};a.copy=function(){return ll().domain([c,d]).range(k)};
  return Ac(a)}function ml(){function a(e){if(e<=e)return c[$b(b,e,0,d)]}var b=[.5],c=[0,1],d=1;a.domain=function(e){return arguments.length?(b=Ib.call(e),d=Math.min(b.length,c.length-1),a):b.slice()};a.range=function(e){return arguments.length?(c=Ib.call(e),d=Math.min(b.length,c.length-1),a):c.slice()};a.invertExtent=function(e){e=c.indexOf(e);return[b[e-1],b[e]]};a.copy=function(){return ml().domain(b).range(c)};return a}function Da(a,b,c,d){function e(g){return a(g=new Date(+g)),g}e.floor=e;e.ceil=
  function(g){return a(g=new Date(g-1)),b(g,1),a(g),g};e.round=function(g){var k=e(g),m=e.ceil(g);return g-k<m-g?k:m};e.offset=function(g,k){return b(g=new Date(+g),null==k?1:Math.floor(k)),g};e.range=function(g,k,m){var p=[],v;g=e.ceil(g);m=null==m?1:Math.floor(m);if(!(g<k&&0<m))return p;do p.push(v=new Date(+g)),b(g,m),a(g);while(v<g&&g<k);return p};e.filter=function(g){return Da(function(k){if(k>=k)for(;a(k),!g(k);)k.setTime(k-1)},function(k,m){if(k>=k)if(0>m)for(;0>=++m;)for(;b(k,-1),!g(k););else for(;0<=
  --m;)for(;b(k,1),!g(k););})};c&&(e.count=function(g,k){rh.setTime(+g);sh.setTime(+k);a(rh);a(sh);return Math.floor(c(rh,sh))},e.every=function(g){g=Math.floor(g);return isFinite(g)&&0<g?1<g?e.filter(d?function(k){return 0===d(k)%g}:function(k){return 0===e.count(0,k)%g}):e:null});return e}function ac(a){return Da(function(b){b.setDate(b.getDate()-(b.getDay()+7-a)%7);b.setHours(0,0,0,0)},function(b,c){b.setDate(b.getDate()+7*c)},function(b,c){return(c-b-6E4*(c.getTimezoneOffset()-b.getTimezoneOffset()))/
  6048E5})}function bc(a){return Da(function(b){b.setUTCDate(b.getUTCDate()-(b.getUTCDay()+7-a)%7);b.setUTCHours(0,0,0,0)},function(b,c){b.setUTCDate(b.getUTCDate()+7*c)},function(b,c){return(c-b)/6048E5})}function nr(a){if(0<=a.y&&100>a.y){var b=new Date(-1,a.m,a.d,a.H,a.M,a.S,a.L);b.setFullYear(a.y);return b}return new Date(a.y,a.m,a.d,a.H,a.M,a.S,a.L)}function af(a){if(0<=a.y&&100>a.y){var b=new Date(Date.UTC(-1,a.m,a.d,a.H,a.M,a.S,a.L));b.setUTCFullYear(a.y);return b}return new Date(Date.UTC(a.y,
  a.m,a.d,a.H,a.M,a.S,a.L))}function rd(a){return{y:a,m:0,d:1,H:0,M:0,S:0,L:0}}function nl(a){function b(f,n){return function(u){var r=[],t=-1,z=0,D=f.length,A,C;for(u instanceof Date||(u=new Date(+u));++t<D;)if(37===f.charCodeAt(t)){r.push(f.slice(z,t));null!=(z=ol[A=f.charAt(++t)])?A=f.charAt(++t):z="e"===A?" ":"0";if(C=n[A])A=C(u,z);r.push(A);z=t+1}r.push(f.slice(z,t));return r.join("")}}function c(f,n){return function(u){var r=rd(1900);if(d(r,f,u+="",0)!=u.length)return null;if("Q"in r)return new Date(r.Q);
  "p"in r&&(r.H=r.H%12+12*r.p);if("V"in r){if(1>r.V||53<r.V)return null;"w"in r||(r.w=1);if("Z"in r){u=af(rd(r.y));var t=u.getUTCDay();u=4<t||0===t?sd.ceil(u):sd(u);u=td.offset(u,7*(r.V-1));r.y=u.getUTCFullYear();r.m=u.getUTCMonth();r.d=u.getUTCDate()+(r.w+6)%7}else u=n(rd(r.y)),t=u.getDay(),u=4<t||0===t?ud.ceil(u):ud(u),u=vd.offset(u,7*(r.V-1)),r.y=u.getFullYear(),r.m=u.getMonth(),r.d=u.getDate()+(r.w+6)%7}else if("W"in r||"U"in r)"w"in r||(r.w="u"in r?r.u%7:"W"in r?1:0),t="Z"in r?af(rd(r.y)).getUTCDay():
  n(rd(r.y)).getDay(),r.m=0,r.d="W"in r?(r.w+6)%7+7*r.W-(t+5)%7:r.w+7*r.U-(t+6)%7;return"Z"in r?(r.H+=r.Z/100|0,r.M+=r.Z%100,af(r)):n(r)}}function d(f,n,u,r){for(var t=0,z=n.length,D=u.length,A;t<z;){if(r>=D)return-1;A=n.charCodeAt(t++);if(37===A){if(A=n.charAt(t++),A=T[A in ol?n.charAt(t++):A],!A||0>(r=A(f,u,r)))return-1}else if(A!=u.charCodeAt(r++))return-1}return r}var e=a.dateTime,g=a.date,k=a.time,m=a.periods,p=a.days,v=a.shortDays,h=a.months,l=a.shortMonths,q=wd(m),w=xd(m),B=wd(p),F=xd(p),J=wd(v),
  P=xd(v),x=wd(h),y=xd(h),I=wd(l),Q=xd(l),V={a:function(f){return v[f.getDay()]},A:function(f){return p[f.getDay()]},b:function(f){return l[f.getMonth()]},B:function(f){return h[f.getMonth()]},c:null,d:pl,e:pl,f:or,H:pr,I:qr,j:rr,L:ql,m:sr,M:tr,p:function(f){return m[+(12<=f.getHours())]},Q:rl,s:sl,S:ur,u:vr,U:wr,V:xr,w:yr,W:zr,x:null,X:null,y:Ar,Y:Br,Z:Cr,"%":tl},N={a:function(f){return v[f.getUTCDay()]},A:function(f){return p[f.getUTCDay()]},b:function(f){return l[f.getUTCMonth()]},B:function(f){return h[f.getUTCMonth()]},
  c:null,d:ul,e:ul,f:Dr,H:Er,I:Fr,j:Gr,L:vl,m:Hr,M:Ir,p:function(f){return m[+(12<=f.getUTCHours())]},Q:rl,s:sl,S:Jr,u:Kr,U:Lr,V:Mr,w:Nr,W:Or,x:null,X:null,y:Pr,Y:Qr,Z:Rr,"%":tl},T={a:function(f,n,u){return(n=J.exec(n.slice(u)))?(f.w=P[n[0].toLowerCase()],u+n[0].length):-1},A:function(f,n,u){return(n=B.exec(n.slice(u)))?(f.w=F[n[0].toLowerCase()],u+n[0].length):-1},b:function(f,n,u){return(n=I.exec(n.slice(u)))?(f.m=Q[n[0].toLowerCase()],u+n[0].length):-1},B:function(f,n,u){return(n=x.exec(n.slice(u)))?
  (f.m=y[n[0].toLowerCase()],u+n[0].length):-1},c:function(f,n,u){return d(f,e,n,u)},d:wl,e:wl,f:Sr,H:xl,I:xl,j:Tr,L:Ur,m:Vr,M:Wr,p:function(f,n,u){return(n=q.exec(n.slice(u)))?(f.p=w[n[0].toLowerCase()],u+n[0].length):-1},Q:Xr,s:Yr,S:Zr,u:$r,U:as,V:bs,w:cs,W:ds,x:function(f,n,u){return d(f,g,n,u)},X:function(f,n,u){return d(f,k,n,u)},y:es,Y:fs,Z:gs,"%":hs};V.x=b(g,V);V.X=b(k,V);V.c=b(e,V);N.x=b(g,N);N.X=b(k,N);N.c=b(e,N);return{format:function(f){var n=b(f+="",V);n.toString=function(){return f};return n},
  parse:function(f){var n=c(f+="",nr);n.toString=function(){return f};return n},utcFormat:function(f){var n=b(f+="",N);n.toString=function(){return f};return n},utcParse:function(f){var n=c(f,af);n.toString=function(){return f};return n}}}function sa(a,b,c){var d=0>a?"-":"";a=(d?-a:a)+"";var e=a.length;return d+(e<c?Array(c-e+1).join(b)+a:a)}function is(a){return a.replace(js,"\\$&")}function wd(a){return new RegExp("^(?:"+a.map(is).join("|")+")","i")}function xd(a){for(var b={},c=-1,d=a.length;++c<
  d;)b[a[c].toLowerCase()]=c;return b}function cs(a,b,c){return(b=Ga.exec(b.slice(c,c+1)))?(a.w=+b[0],c+b[0].length):-1}function $r(a,b,c){return(b=Ga.exec(b.slice(c,c+1)))?(a.u=+b[0],c+b[0].length):-1}function as(a,b,c){return(b=Ga.exec(b.slice(c,c+2)))?(a.U=+b[0],c+b[0].length):-1}function bs(a,b,c){return(b=Ga.exec(b.slice(c,c+2)))?(a.V=+b[0],c+b[0].length):-1}function ds(a,b,c){return(b=Ga.exec(b.slice(c,c+2)))?(a.W=+b[0],c+b[0].length):-1}function fs(a,b,c){return(b=Ga.exec(b.slice(c,c+4)))?(a.y=
  +b[0],c+b[0].length):-1}function es(a,b,c){return(b=Ga.exec(b.slice(c,c+2)))?(a.y=+b[0]+(68<+b[0]?1900:2E3),c+b[0].length):-1}function gs(a,b,c){return(b=/^(Z)|([+-]\d\d)(?::?(\d\d))?/.exec(b.slice(c,c+6)))?(a.Z=b[1]?0:-(b[2]+(b[3]||"00")),c+b[0].length):-1}function Vr(a,b,c){return(b=Ga.exec(b.slice(c,c+2)))?(a.m=b[0]-1,c+b[0].length):-1}function wl(a,b,c){return(b=Ga.exec(b.slice(c,c+2)))?(a.d=+b[0],c+b[0].length):-1}function Tr(a,b,c){return(b=Ga.exec(b.slice(c,c+3)))?(a.m=0,a.d=+b[0],c+b[0].length):
  -1}function xl(a,b,c){return(b=Ga.exec(b.slice(c,c+2)))?(a.H=+b[0],c+b[0].length):-1}function Wr(a,b,c){return(b=Ga.exec(b.slice(c,c+2)))?(a.M=+b[0],c+b[0].length):-1}function Zr(a,b,c){return(b=Ga.exec(b.slice(c,c+2)))?(a.S=+b[0],c+b[0].length):-1}function Ur(a,b,c){return(b=Ga.exec(b.slice(c,c+3)))?(a.L=+b[0],c+b[0].length):-1}function Sr(a,b,c){return(b=Ga.exec(b.slice(c,c+6)))?(a.L=Math.floor(b[0]/1E3),c+b[0].length):-1}function hs(a,b,c){return(a=ks.exec(b.slice(c,c+1)))?c+a[0].length:-1}function Xr(a,
  b,c){return(b=Ga.exec(b.slice(c)))?(a.Q=+b[0],c+b[0].length):-1}function Yr(a,b,c){return(b=Ga.exec(b.slice(c)))?(a.Q=1E3*+b[0],c+b[0].length):-1}function pl(a,b){return sa(a.getDate(),b,2)}function pr(a,b){return sa(a.getHours(),b,2)}function qr(a,b){return sa(a.getHours()%12||12,b,2)}function rr(a,b){return sa(1+vd.count(wb(a),a),b,3)}function ql(a,b){return sa(a.getMilliseconds(),b,3)}function or(a,b){return ql(a,b)+"000"}function sr(a,b){return sa(a.getMonth()+1,b,2)}function tr(a,b){return sa(a.getMinutes(),
  b,2)}function ur(a,b){return sa(a.getSeconds(),b,2)}function vr(a){a=a.getDay();return 0===a?7:a}function wr(a,b){return sa(yd.count(wb(a),a),b,2)}function xr(a,b){var c=a.getDay();a=4<=c||0===c?zd(a):zd.ceil(a);return sa(zd.count(wb(a),a)+(4===wb(a).getDay()),b,2)}function yr(a){return a.getDay()}function zr(a,b){return sa(ud.count(wb(a),a),b,2)}function Ar(a,b){return sa(a.getFullYear()%100,b,2)}function Br(a,b){return sa(a.getFullYear()%1E4,b,4)}function Cr(a){a=a.getTimezoneOffset();return(0<
  a?"-":(a*=-1,"+"))+sa(a/60|0,"0",2)+sa(a%60,"0",2)}function ul(a,b){return sa(a.getUTCDate(),b,2)}function Er(a,b){return sa(a.getUTCHours(),b,2)}function Fr(a,b){return sa(a.getUTCHours()%12||12,b,2)}function Gr(a,b){return sa(1+td.count(xb(a),a),b,3)}function vl(a,b){return sa(a.getUTCMilliseconds(),b,3)}function Dr(a,b){return vl(a,b)+"000"}function Hr(a,b){return sa(a.getUTCMonth()+1,b,2)}function Ir(a,b){return sa(a.getUTCMinutes(),b,2)}function Jr(a,b){return sa(a.getUTCSeconds(),b,2)}function Kr(a){a=
  a.getUTCDay();return 0===a?7:a}function Lr(a,b){return sa(Ad.count(xb(a),a),b,2)}function Mr(a,b){var c=a.getUTCDay();a=4<=c||0===c?Bd(a):Bd.ceil(a);return sa(Bd.count(xb(a),a)+(4===xb(a).getUTCDay()),b,2)}function Nr(a){return a.getUTCDay()}function Or(a,b){return sa(sd.count(xb(a),a),b,2)}function Pr(a,b){return sa(a.getUTCFullYear()%100,b,2)}function Qr(a,b){return sa(a.getUTCFullYear()%1E4,b,4)}function Rr(){return"+0000"}function tl(){return"%"}function rl(a){return+a}function sl(a){return Math.floor(+a/
  1E3)}function yl(a){Cc=nl(a);d3.timeFormat=Cc.format;d3.timeParse=Cc.parse;d3.utcFormat=Cc.utcFormat;d3.utcParse=Cc.utcParse;return Cc}function ls(a){return a.toISOString()}function ms(a){a=new Date(a);return isNaN(a)?null:a}function ns(a){return new Date(a)}function os(a){return a instanceof Date?+a:+new Date(+a)}function th(a,b,c,d,e,g,k,m,p){function v(N){return(k(N)<N?B:g(N)<N?F:e(N)<N?J:d(N)<N?P:b(N)<N?c(N)<N?x:y:a(N)<N?I:Q)(N)}function h(N,T,f,n){null==N&&(N=10);if("number"===typeof N){n=Math.abs(f-
  T)/N;var u=Bf(function(r){return r[2]}).right(V,n);u===V.length?(n=Nb(T/31536E6,f/31536E6,N),N=a):u?(u=V[n/V[u-1][2]<V[u][2]/n?u-1:u],n=u[1],N=u[0]):(n=Math.max(Nb(T,f,N),1),N=m)}return null==n?N:N.every(n)}var l=$e(oh,Va),q=l.invert,w=l.domain,B=p(".%L"),F=p(":%S"),J=p("%I:%M"),P=p("%I %p"),x=p("%a %d"),y=p("%b %d"),I=p("%B"),Q=p("%Y"),V=[[k,1,1E3],[k,5,5E3],[k,15,15E3],[k,30,3E4],[g,1,6E4],[g,5,3E5],[g,15,9E5],[g,30,18E5],[e,1,36E5],[e,3,108E5],[e,6,216E5],[e,12,432E5],[d,1,864E5],[d,2,1728E5],
  [c,1,6048E5],[b,1,2592E6],[b,3,7776E6],[a,1,31536E6]];l.invert=function(N){return new Date(q(N))};l.domain=function(N){return arguments.length?w(ph.call(N,os)):w().map(ns)};l.ticks=function(N,T){var f=w(),n=f[0];f=f[f.length-1];var u=f<n;if(u){var r=n;n=f;f=r}r=(r=h(N,n,f,T))?r.range(n,f+1):[];return u?r.reverse():r};l.tickFormat=function(N,T){return null==T?v:p(T)};l.nice=function(N,T){var f=w();return(N=h(N,f[0],f[f.length-1],T))?w(fl(f,N)):l};l.copy=function(){return Ze(l,th(a,b,c,d,e,g,k,m,p))};
  return l}function zl(a){function b(k){k=(k-c)*e;return a(g?Math.max(0,Math.min(1,k)):k)}var c=0,d=1,e=1,g=!1;b.domain=function(k){return arguments.length?(c=+k[0],d=+k[1],e=c===d?0:1/(d-c),b):[c,d]};b.clamp=function(k){return arguments.length?(g=!!k,b):g};b.interpolator=function(k){return arguments.length?(a=k,b):a};b.copy=function(){return zl(a).domain([c,d]).clamp(g)};return Ac(b)}function Al(a){function b(p){var v=.5+((p=+p)-d)*(p<d?g:k);return a(m?Math.max(0,Math.min(1,v)):v)}var c=0,d=.5,e=1,
  g=1,k=1,m=!1;b.domain=function(p){return arguments.length?(c=+p[0],d=+p[1],e=+p[2],g=c===d?0:.5/(d-c),k=d===e?0:.5/(e-d),b):[c,d,e]};b.clamp=function(p){return arguments.length?(m=!!p,b):m};b.interpolator=function(p){return arguments.length?(a=p,b):a};b.copy=function(){return Al(a).domain([c,d,e]).clamp(m)};return Ac(b)}function ka(a){for(var b=a.length/6|0,c=Array(b),d=0;d<b;)c[d]="#"+a.slice(6*d,6*++d);return c}function ua(a){return Bl(a[a.length-1])}function bf(a){var b=a.length;return function(c){return a[Math.max(0,
  Math.min(b-1,Math.floor(c*b)))]}}function na(a){return function(){return a}}function Cl(a){return 1<=a?cf:-1>=a?-cf:Math.asin(a)}function ps(a){return a.innerRadius}function qs(a){return a.outerRadius}function rs(a){return a.startAngle}function ss(a){return a.endAngle}function ts(a){return a&&a.padAngle}function df(a,b,c,d,e,g,k){var m=a-c,p=b-d;k=(k?g:-g)/Dc(m*m+p*p);p*=k;m*=-k;var v=a+p,h=b+m,l=c+p,q=d+m;c=(v+l)/2;d=(h+q)/2;b=l-v;a=q-h;k=b*b+a*a;g=e-g;q=v*q-l*h;var w=(0>a?-1:1)*Dc(us(0,g*g*k-q*
  q));v=(q*a-b*w)/k;h=(-q*b-a*w)/k;l=(q*a+b*w)/k;b=(-q*b+a*w)/k;a=v-c;k=h-d;c=l-c;d=b-d;a*a+k*k>c*c+d*d&&(v=l,h=b);return{cx:v,cy:h,x01:-p,y01:-m,x11:v*(e/g-1),y11:h*(e/g-1)}}function Dl(a){this._context=a}function ef(a){return new Dl(a)}function uh(a){return a[0]}function vh(a){return a[1]}function wh(){function a(m){var p,v=m.length,h,l=!1,q;null==e&&(k=g(q=Eb()));for(p=0;p<=v;++p)!(p<v&&d(h=m[p],p,m))===l&&((l=!l)?k.lineStart():k.lineEnd()),l&&k.point(+b(h,p,m),+c(h,p,m));if(q)return k=null,q+""||
  null}var b=uh,c=vh,d=na(!0),e=null,g=ef,k=null;a.x=function(m){return arguments.length?(b="function"===typeof m?m:na(+m),a):b};a.y=function(m){return arguments.length?(c="function"===typeof m?m:na(+m),a):c};a.defined=function(m){return arguments.length?(d="function"===typeof m?m:na(!!m),a):d};a.curve=function(m){return arguments.length?(g=m,null!=e&&(k=g(e)),a):g};a.context=function(m){return arguments.length?(null==m?e=k=null:k=g(e=m),a):e};return a}function El(){function a(h){var l,q,w=h.length,
  B,F=!1,J,P=Array(w),x=Array(w);null==m&&(v=p(J=Eb()));for(l=0;l<=w;++l){if(!(l<w&&k(B=h[l],l,h))===F)if(F=!F){var y=l;v.areaStart();v.lineStart()}else{v.lineEnd();v.lineStart();for(q=l-1;q>=y;--q)v.point(P[q],x[q]);v.lineEnd();v.areaEnd()}F&&(P[l]=+c(B,l,h),x[l]=+e(B,l,h),v.point(d?+d(B,l,h):P[l],g?+g(B,l,h):x[l]))}if(J)return v=null,J+""||null}function b(){return wh().defined(k).curve(p).context(m)}var c=uh,d=null,e=na(0),g=vh,k=na(!0),m=null,p=ef,v=null;a.x=function(h){return arguments.length?(c=
  "function"===typeof h?h:na(+h),d=null,a):c};a.x0=function(h){return arguments.length?(c="function"===typeof h?h:na(+h),a):c};a.x1=function(h){return arguments.length?(d=null==h?null:"function"===typeof h?h:na(+h),a):d};a.y=function(h){return arguments.length?(e="function"===typeof h?h:na(+h),g=null,a):e};a.y0=function(h){return arguments.length?(e="function"===typeof h?h:na(+h),a):e};a.y1=function(h){return arguments.length?(g=null==h?null:"function"===typeof h?h:na(+h),a):g};a.lineX0=a.lineY0=function(){return b().x(c).y(e)};
  a.lineY1=function(){return b().x(c).y(g)};a.lineX1=function(){return b().x(d).y(e)};a.defined=function(h){return arguments.length?(k="function"===typeof h?h:na(!!h),a):k};a.curve=function(h){return arguments.length?(p=h,null!=m&&(v=p(m)),a):p};a.context=function(h){return arguments.length?(null==h?m=v=null:v=p(m=h),a):m};return a}function vs(a,b){return b<a?-1:b>a?1:b>=a?0:NaN}function ws(a){return a}function Fl(a){this._curve=a}function xh(a){function b(c){return new Fl(a(c))}b._curve=a;return b}
  function Cd(a){var b=a.curve;a.angle=a.x;delete a.x;a.radius=a.y;delete a.y;a.curve=function(c){return arguments.length?b(xh(c)):b()._curve};return a}function Gl(){return Cd(wh().curve(Hl))}function Il(){var a=El().curve(Hl),b=a.curve,c=a.lineX0,d=a.lineX1,e=a.lineY0,g=a.lineY1;a.angle=a.x;delete a.x;a.startAngle=a.x0;delete a.x0;a.endAngle=a.x1;delete a.x1;a.radius=a.y;delete a.y;a.innerRadius=a.y0;delete a.y0;a.outerRadius=a.y1;delete a.y1;a.lineStartAngle=function(){return Cd(c())};delete a.lineX0;
  a.lineEndAngle=function(){return Cd(d())};delete a.lineX1;a.lineInnerRadius=function(){return Cd(e())};delete a.lineY0;a.lineOuterRadius=function(){return Cd(g())};delete a.lineY1;a.curve=function(k){return arguments.length?b(xh(k)):b()._curve};return a}function Dd(a,b){return[(b=+b)*Math.cos(a-=Math.PI/2),b*Math.sin(a)]}function xs(a){return a.source}function ys(a){return a.target}function yh(a){function b(){var m,p=zh.call(arguments),v=c.apply(this,p),h=d.apply(this,p);k||(k=m=Eb());a(k,+e.apply(this,
  (p[0]=v,p)),+g.apply(this,p),+e.apply(this,(p[0]=h,p)),+g.apply(this,p));if(m)return k=null,m+""||null}var c=xs,d=ys,e=uh,g=vh,k=null;b.source=function(m){return arguments.length?(c=m,b):c};b.target=function(m){return arguments.length?(d=m,b):d};b.x=function(m){return arguments.length?(e="function"===typeof m?m:na(+m),b):e};b.y=function(m){return arguments.length?(g="function"===typeof m?m:na(+m),b):g};b.context=function(m){return arguments.length?(k=null==m?null:m,b):k};return b}function zs(a,b,
  c,d,e){a.moveTo(b,c);a.bezierCurveTo(b=(b+d)/2,c,b,e,d,e)}function As(a,b,c,d,e){a.moveTo(b,c);a.bezierCurveTo(b,c=(c+e)/2,d,c,d,e)}function Bs(a,b,c,d,e){var g=Dd(b,c);b=Dd(b,c=(c+e)/2);c=Dd(d,c);d=Dd(d,e);a.moveTo(g[0],g[1]);a.bezierCurveTo(b[0],b[1],c[0],c[1],d[0],d[1])}function Jb(){}function ff(a,b,c){a._context.bezierCurveTo((2*a._x0+a._x1)/3,(2*a._y0+a._y1)/3,(a._x0+2*a._x1)/3,(a._y0+2*a._y1)/3,(a._x0+4*a._x1+b)/6,(a._y0+4*a._y1+c)/6)}function gf(a){this._context=a}function Jl(a){this._context=
  a}function Kl(a){this._context=a}function Ll(a,b){this._basis=new gf(a);this._beta=b}function hf(a,b,c){a._context.bezierCurveTo(a._x1+a._k*(a._x2-a._x0),a._y1+a._k*(a._y2-a._y0),a._x2+a._k*(a._x1-b),a._y2+a._k*(a._y1-c),a._x2,a._y2)}function Ah(a,b){this._context=a;this._k=(1-b)/6}function Bh(a,b){this._context=a;this._k=(1-b)/6}function Ch(a,b){this._context=a;this._k=(1-b)/6}function Dh(a,b,c){var d=a._x1,e=a._y1,g=a._x2,k=a._y2;if(1E-12<a._l01_a){var m=2*a._l01_2a+3*a._l01_a*a._l12_a+a._l12_2a,
  p=3*a._l01_a*(a._l01_a+a._l12_a);d=(d*m-a._x0*a._l12_2a+a._x2*a._l01_2a)/p;e=(e*m-a._y0*a._l12_2a+a._y2*a._l01_2a)/p}1E-12<a._l23_a&&(m=2*a._l23_2a+3*a._l23_a*a._l12_a+a._l12_2a,p=3*a._l23_a*(a._l23_a+a._l12_a),g=(g*m+a._x1*a._l23_2a-b*a._l12_2a)/p,k=(k*m+a._y1*a._l23_2a-c*a._l12_2a)/p);a._context.bezierCurveTo(d,e,g,k,a._x2,a._y2)}function Ml(a,b){this._context=a;this._alpha=b}function Nl(a,b){this._context=a;this._alpha=b}function Ol(a,b){this._context=a;this._alpha=b}function Pl(a){this._context=
  a}function Ql(a,b,c){var d=a._x1-a._x0;b-=a._x1;var e=(a._y1-a._y0)/(d||0>b&&-0);a=(c-a._y1)/(b||0>d&&-0);return((0>e?-1:1)+(0>a?-1:1))*Math.min(Math.abs(e),Math.abs(a),.5*Math.abs((e*b+a*d)/(d+b)))||0}function Rl(a,b){var c=a._x1-a._x0;return c?(3*(a._y1-a._y0)/c-b)/2:b}function Eh(a,b,c){var d=a._x0,e=a._x1,g=a._y1,k=(e-d)/3;a._context.bezierCurveTo(d+k,a._y0+k*b,e-k,g-k*c,e,g)}function jf(a){this._context=a}function Sl(a){this._context=new Tl(a)}function Tl(a){this._context=a}function Ul(a){this._context=
  a}function Vl(a){var b,c=a.length-1,d=Array(c),e=Array(c),g=Array(c);d[0]=0;e[0]=2;g[0]=a[0]+2*a[1];for(b=1;b<c-1;++b)d[b]=1,e[b]=4,g[b]=4*a[b]+2*a[b+1];d[c-1]=2;e[c-1]=7;g[c-1]=8*a[c-1]+a[c];for(b=1;b<c;++b){var k=d[b]/e[b-1];e[b]-=k;g[b]-=k*g[b-1]}d[c-1]=g[c-1]/e[c-1];for(b=c-2;0<=b;--b)d[b]=(g[b]-d[b+1])/e[b];e[c-1]=(a[c]+d[c-1])/2;for(b=0;b<c-1;++b)e[b]=2*a[b+1]-d[b+1];return[d,e]}function kf(a,b){this._context=a;this._t=b}function Ec(a,b){if(1<(k=a.length))for(var c=1,d,e,g=a[b[0]],k,m=g.length;c<
  k;++c)for(e=g,g=a[b[c]],d=0;d<m;++d)g[d][1]+=g[d][0]=isNaN(e[d][1])?e[d][0]:e[d][1]}function Fc(a){a=a.length;for(var b=Array(a);0<=--a;)b[a]=a;return b}function Cs(a,b){return a[b]}function Wl(a){var b=a.map(Xl);return Fc(a).sort(function(c,d){return b[c]-b[d]})}function Xl(a){for(var b=0,c=-1,d=a.length,e;++c<d;)if(e=+a[c][1])b+=e;return b}function Yl(a){return function(){return a}}function Ds(a){return a[0]}function Es(a){return a[1]}function lf(){this._=null}function mf(a){a.U=a.C=a.L=a.R=a.P=
  a.N=null}function Ed(a,b){var c=b.R,d=b.U;d?d.L===b?d.L=c:d.R=c:a._=c;c.U=d;b.U=c;b.R=c.L;b.R&&(b.R.U=b);c.L=b}function Fd(a,b){var c=b.L,d=b.U;d?d.L===b?d.L=c:d.R=c:a._=c;c.U=d;b.U=c;b.L=c.R;b.L&&(b.L.U=b);c.R=b}function Zl(a){for(;a.L;)a=a.L;return a}function Gd(a,b,c,d){var e=[null,null],g=Ha.push(e)-1;e.left=a;e.right=b;c&&nf(e,a,b,c);d&&nf(e,b,a,d);Ya[a.index].halfedges.push(g);Ya[b.index].halfedges.push(g);return e}function Hd(a,b,c){b=[b,c];b.left=a;return b}function nf(a,b,c,d){a[0]||a[1]?
  a.left===c?a[1]=d:a[0]=d:(a[0]=d,a.left=b,a.right=c)}function Fs(a,b,c,d,e){var g=a[0],k=a[1],m=g[0];g=g[1];var p=0,v=1,h=k[0]-m;k=k[1]-g;b-=m;if(h||!(0<b)){b/=h;if(0>h){if(b<p)return;b<v&&(v=b)}else if(0<h){if(b>v)return;b>p&&(p=b)}b=d-m;if(h||!(0>b)){b/=h;if(0>h){if(b>v)return;b>p&&(p=b)}else if(0<h){if(b<p)return;b<v&&(v=b)}b=c-g;if(k||!(0<b)){b/=k;if(0>k){if(b<p)return;b<v&&(v=b)}else if(0<k){if(b>v)return;b>p&&(p=b)}b=e-g;if(k||!(0>b)){b/=k;if(0>k){if(b>v)return;b>p&&(p=b)}else if(0<k){if(b<
  p)return;b<v&&(v=b)}if(!(0<p||1>v))return!0;0<p&&(a[0]=[m+p*h,g+p*k]);1>v&&(a[1]=[m+v*h,g+v*k]);return!0}}}}}function Gs(a,b,c,d,e){var g=a[1];if(g)return!0;var k=a[0],m=a.left,p=a.right;g=m[0];m=m[1];var v=p[0];p=p[1];var h=(g+v)/2;if(p===m){if(h<b||h>=d)return;if(g>v){if(!k)k=[h,c];else if(k[1]>=e)return;g=[h,e]}else{if(!k)k=[h,e];else if(k[1]<c)return;g=[h,c]}}else{var l=(g-v)/(p-m);h=(m+p)/2-l*h;if(-1>l||1<l)if(g>v){if(!k)k=[(c-h)/l,c];else if(k[1]>=e)return;g=[(e-h)/l,e]}else{if(!k)k=[(e-h)/
  l,e];else if(k[1]<c)return;g=[(c-h)/l,c]}else if(m<p){if(!k)k=[b,l*b+h];else if(k[0]>=d)return;g=[d,l*d+h]}else{if(!k)k=[d,l*d+h];else if(k[0]<b)return;g=[b,l*b+h]}}a[0]=k;a[1]=g;return!0}function Hs(a,b){a=a.site;var c=b.left,d=b.right;a===d&&(d=c,c=a);if(d)return Math.atan2(d[1]-c[1],d[0]-c[0]);a===c?(c=b[1],d=b[0]):(c=b[0],d=b[1]);return Math.atan2(c[0]-d[0],d[1]-c[1])}function $l(a,b){return b[+(b.left!==a.site)]}function Is(){for(var a=0,b=Ya.length,c,d,e,g;a<b;++a)if((c=Ya[a])&&(g=(d=c.halfedges).length)){var k=
  Array(g),m=Array(g);for(e=0;e<g;++e)k[e]=e,m[e]=Hs(c,Ha[d[e]]);k.sort(function(p,v){return m[v]-m[p]});for(e=0;e<g;++e)m[e]=d[k[e]];for(e=0;e<g;++e)d[e]=m[e]}}function Js(){mf(this);this.x=this.y=this.arc=this.site=this.cy=null}function Gc(a){var b=a.P,c=a.N;if(b&&c){var d=b.site;b=a.site;var e=c.site;if(d!==e){c=b[0];var g=b[1],k=d[0]-c,m=d[1]-g;d=e[0]-c;var p=e[1]-g;e=2*(k*p-m*d);if(!(e>=-Ks)){var v=k*k+m*m,h=d*d+p*p;m=(p*v-m*h)/e;d=(k*h-d*v)/e;k=am.pop()||new Js;k.arc=a;k.site=b;k.x=m+c;k.y=(k.cy=
  d+g)+Math.sqrt(m*m+d*d);a.circle=k;a=null;for(b=Id._;b;)if(k.y<b.y||k.y===b.y&&k.x<=b.x)if(b.L)b=b.L;else{a=b.P;break}else if(b.R)b=b.R;else{a=b;break}Id.insert(a,k);a||(Fh=k)}}}}function Hc(a){var b=a.circle;b&&(b.P||(Fh=b.N),Id.remove(b),am.push(b),mf(b),a.circle=null)}function Ls(){mf(this);this.edge=this.site=this.circle=null}function bm(a){var b=cm.pop()||new Ls;b.site=a;return b}function Gh(a){Hc(a);Ic.remove(a);cm.push(a);mf(a)}function dm(a,b){var c=a.site,d=c[0],e=c[1],g=e-b;if(!g)return d;
  a=a.P;if(!a)return-Infinity;c=a.site;a=c[0];c=c[1];b=c-b;if(!b)return a;var k=a-d,m=1/g-1/b,p=k/b;return m?(-p+Math.sqrt(p*p-2*m*(k*k/(-2*b)-c+b/2+e-g/2)))/m+d:(d+a)/2}function Ms(a,b){return b[1]-a[1]||b[0]-a[0]}function Hh(a,b){var c=a.sort(Ms).pop(),d;Ha=[];Ya=Array(a.length);Ic=new lf;for(Id=new lf;;){var e=Fh;if(c&&(!e||c[1]<e.y||c[1]===e.y&&c[0]<e.x)){if(c[0]!==d||c[1]!==g){var g=d=void 0;e=c;for(var k=e[0],m=e[1],p=Ic._;p;){var v=dm(p,m)-k;if(v>ta)p=p.L;else{var h=p;var l=m;var q=h.N;q?l=dm(q,
  l):(h=h.site,l=h[1]===l?h[0]:Infinity);l=k-l;if(l>ta){if(!p.R){g=p;break}p=p.R}else{v>-ta?(g=p.P,d=p):l>-ta?(g=p,d=p.N):g=d=p;break}}}Ya[e.index]={site:e,halfedges:[]};v=bm(e);Ic.insert(g,v);if(g||d)if(g===d)Hc(g),d=bm(g.site),Ic.insert(v,d),v.edge=d.edge=Gd(g.site,v.site),Gc(g),Gc(d);else if(d){Hc(g);Hc(d);k=g.site;p=k[0];l=k[1];h=e[0]-p;q=e[1]-l;m=d.site;var w=m[0]-p,B=m[1]-l,F=2*(h*B-q*w),J=h*h+q*q,P=w*w+B*B;p=[(B*J-q*P)/F+p,(h*P-w*J)/F+l];nf(d.edge,k,m,p);v.edge=Gd(k,e,null,p);d.edge=Gd(e,m,null,
  p);Gc(g);Gc(d)}else v.edge=Gd(g.site,v.site);d=c[0];g=c[1]}c=a.pop()}else if(e){m=e.arc;e=m.circle;k=e.x;p=e.cy;e=[k,p];h=m.P;l=m.N;v=[m];Gh(m);for(m=h;m.circle&&Math.abs(k-m.circle.x)<ta&&Math.abs(p-m.circle.cy)<ta;)h=m.P,v.unshift(m),Gh(m),m=h;v.unshift(m);Hc(m);for(h=l;h.circle&&Math.abs(k-h.circle.x)<ta&&Math.abs(p-h.circle.cy)<ta;)l=h.N,v.push(h),Gh(h),h=l;v.push(h);Hc(h);p=v.length;for(k=1;k<p;++k)h=v[k],m=v[k-1],nf(h.edge,m.site,h.site,e);m=v[0];h=v[p-1];h.edge=Gd(m.site,h.site,null,e);Gc(m);
  Gc(h)}else break}Is();if(b){d=+b[0][0];a=+b[0][1];c=+b[1][0];b=+b[1][1];g=Ha.length;for(var x;g--;)Gs(x=Ha[g],d,a,c,b)&&Fs(x,d,a,c,b)&&(Math.abs(x[0][0]-x[1][0])>ta||Math.abs(x[0][1]-x[1][1])>ta)||delete Ha[g];x=Ya.length;g=!0;for(e=0;e<x;++e)if(v=Ya[e]){var y=v.site;m=v.halfedges;for(k=m.length;k--;)Ha[m[k]]||m.splice(k,1);k=0;for(p=m.length;k<p;)if(l=Ha[m[k]],h=l[+(l.left===v.site)],q=h[0],w=h[1],B=$l(v,Ha[m[++k%p]]),l=B[0],B=B[1],Math.abs(q-l)>ta||Math.abs(w-B)>ta)m.splice(k,0,Ha.push(Hd(y,h,Math.abs(q-
  d)<ta&&b-w>ta?[d,Math.abs(l-d)<ta?B:b]:Math.abs(w-b)<ta&&c-q>ta?[Math.abs(B-b)<ta?l:c,b]:Math.abs(q-c)<ta&&w-a>ta?[c,Math.abs(l-c)<ta?B:a]:Math.abs(w-a)<ta&&q-d>ta?[Math.abs(B-a)<ta?l:d,a]:null))-1),++p;p&&(g=!1)}if(g){k=Infinity;e=0;for(g=null;e<x;++e)if(v=Ya[e])y=v.site,m=y[0]-d,p=y[1]-a,m=m*m+p*p,m<k&&(k=m,g=v);g&&(e=[d,a],d=[d,b],b=[c,b],a=[c,a],g.halfedges.push(Ha.push(Hd(y=g.site,e,d))-1,Ha.push(Hd(y,d,b))-1,Ha.push(Hd(y,b,a))-1,Ha.push(Hd(y,a,e))-1))}for(e=0;e<x;++e)if(v=Ya[e])v.halfedges.length||
  delete Ya[e]}this.edges=Ha;this.cells=Ya;Ic=Id=Ha=Ya=null}function of(a){return function(){return a}}function Ns(a,b,c){this.target=a;this.type=b;this.transform=c}function yb(a,b,c){this.k=a;this.x=b;this.y=c}function em(a){return a.__zoom||pf}function Jd(){d3.event.preventDefault();d3.event.stopImmediatePropagation()}function Os(){return!d3.event.button}function Ps(){var a=this;if(a instanceof SVGElement){a=a.ownerSVGElement||a;var b=a.width.baseVal.value;a=a.height.baseVal.value}else b=a.clientWidth,
  a=a.clientHeight;return[[0,0],[b,a]]}function fm(){return this.__zoom||pf}function Qs(){return-d3.event.deltaY*(d3.event.deltaMode?120:1)/500}function Rs(){return"ontouchstart"in this}function Ss(a,b,c){var d=a.invertX(b[0][0])-c[0][0],e=a.invertX(b[1][0])-c[1][0],g=a.invertY(b[0][1])-c[0][1];b=a.invertY(b[1][1])-c[1][1];return a.translate(e>d?(d+e)/2:Math.min(0,d)||Math.max(0,e),b>g?(g+b)/2:Math.min(0,g)||Math.max(0,b))}var gm=Bf(Mb),$b=gm.right,Ts=gm.left,hm=Array.prototype,Us=hm.slice,Vs=hm.map,
  Ef=Math.sqrt(50),Ff=Math.sqrt(10),Gf=Math.sqrt(2),Jf=Array.prototype.slice,Ln={value:function(){}};Qd.prototype=Ob.prototype={constructor:Qd,on:function(a,b){var c=this._,d=Kn(a+"",c),e,g=-1,k=d.length;if(2>arguments.length)for(;++g<k;){var m;if(m=e=(a=d[g]).type){a:{m=c[e];for(var p=0,v=m.length;p<v;++p)if((e=m[p]).name===a.name){e=e.value;break a}e=void 0}m=e}if(m)return e}else{if(null!=b&&"function"!==typeof b)throw Error("invalid callback: "+b);for(;++g<k;)if(e=(a=d[g]).type)c[e]=ki(c[e],a.name,
  b);else if(null==b)for(e in c)c[e]=ki(c[e],a.name,null);return this}},copy:function(){var a={},b=this._,c;for(c in b)a[c]=b[c].slice();return new Qd(a)},call:function(a,b){if(0<(e=arguments.length-2))for(var c=Array(e),d=0,e,g;d<e;++d)c[d]=arguments[d+2];if(!this._.hasOwnProperty(a))throw Error("unknown type: "+a);g=this._[a];d=0;for(e=g.length;d<e;++d)g[d].value.apply(b,c)},apply:function(a,b,c){if(!this._.hasOwnProperty(a))throw Error("unknown type: "+a);a=this._[a];for(var d=0,e=a.length;d<e;++d)a[d].value.apply(b,
  c)}};var Ua={svg:"http://www.w3.org/2000/svg",xhtml:"http://www.w3.org/1999/xhtml",xlink:"http://www.w3.org/1999/xlink",xml:"http://www.w3.org/XML/1998/namespace",xmlns:"http://www.w3.org/2000/xmlns/"};if("undefined"!==typeof document){var Kd=document.documentElement;if(!Kd.matches){var Ws=Kd.webkitMatchesSelector||Kd.msMatchesSelector||Kd.mozMatchesSelector||Kd.oMatchesSelector;di=function(a){return function(){return Ws.call(this,a)}}}}var Ih=di;Td.prototype={constructor:Td,appendChild:function(a){return this._parent.insertBefore(a,
  this._next)},insertBefore:function(a,b){return this._parent.insertBefore(a,b)},querySelector:function(a){return this._parent.querySelector(a)},querySelectorAll:function(a){return this._parent.querySelectorAll(a)}};mi.prototype={add:function(a){0>this._names.indexOf(a)&&(this._names.push(a),this._node.setAttribute("class",this._names.join(" ")))},remove:function(a){a=this._names.indexOf(a);0<=a&&(this._names.splice(a,1),this._node.setAttribute("class",this._names.join(" ")))},contains:function(a){return 0<=
  this._names.indexOf(a)}};var qi={};d3.event=null;"undefined"!==typeof document&&("onmouseenter"in document.documentElement||(qi={mouseenter:"mouseover",mouseleave:"mouseout"}));var Nf=[null];Ja.prototype=Qb.prototype={constructor:Ja,select:function(a){"function"!==typeof a&&(a=Sd(a));for(var b=this._groups,c=b.length,d=Array(c),e=0;e<c;++e)for(var g=b[e],k=g.length,m=d[e]=Array(k),p,v,h=0;h<k;++h)(p=g[h])&&(v=a.call(p,p.__data__,h,g))&&("__data__"in p&&(v.__data__=p.__data__),m[h]=v);return new Ja(d,
  this._parents)},selectAll:function(a){"function"!==typeof a&&(a=Kf(a));for(var b=this._groups,c=b.length,d=[],e=[],g=0;g<c;++g)for(var k=b[g],m=k.length,p,v=0;v<m;++v)if(p=k[v])d.push(a.call(p,p.__data__,v,k)),e.push(p);return new Ja(d,e)},filter:function(a){"function"!==typeof a&&(a=Ih(a));for(var b=this._groups,c=b.length,d=Array(c),e=0;e<c;++e)for(var g=b[e],k=g.length,m=d[e]=[],p,v=0;v<k;++v)(p=g[v])&&a.call(p,p.__data__,v,g)&&m.push(p);return new Ja(d,this._parents)},data:function(a,b){if(!a)return w=
  Array(this.size()),v=-1,this.each(function(x){w[++v]=x}),w;var c=b?Sn:Rn,d=this._parents,e=this._groups;"function"!==typeof a&&(a=Qn(a));for(var g=e.length,k=Array(g),m=Array(g),p=Array(g),v=0;v<g;++v){var h=d[v],l=e[v],q=l.length,w=a.call(h,h&&h.__data__,v,d),B=w.length,F=m[v]=Array(B),J=k[v]=Array(B);q=p[v]=Array(q);c(h,l,F,J,q,w,b);l=h=0;for(var P;h<B;++h)if(q=F[h]){for(h>=l&&(l=h+1);!(P=J[l])&&++l<B;);q._next=P||null}}k=new Ja(k,d);k._enter=m;k._exit=p;return k},enter:function(){return new Ja(this._enter||
  this._groups.map(li),this._parents)},exit:function(){return new Ja(this._exit||this._groups.map(li),this._parents)},merge:function(a){var b=this._groups;a=a._groups;for(var c=b.length,d=Math.min(c,a.length),e=Array(c),g=0;g<d;++g)for(var k=b[g],m=a[g],p=k.length,v=e[g]=Array(p),h,l=0;l<p;++l)if(h=k[l]||m[l])v[l]=h;for(;g<c;++g)e[g]=b[g];return new Ja(e,this._parents)},order:function(){for(var a=this._groups,b=-1,c=a.length;++b<c;)for(var d=a[b],e=d.length-1,g=d[e],k;0<=--e;)if(k=d[e])g&&g!==k.nextSibling&&
  g.parentNode.insertBefore(k,g),g=k;return this},sort:function(a){function b(l,q){return l&&q?a(l.__data__,q.__data__):!l-!q}a||(a=Tn);for(var c=this._groups,d=c.length,e=Array(d),g=0;g<d;++g){for(var k=c[g],m=k.length,p=e[g]=Array(m),v,h=0;h<m;++h)if(v=k[h])p[h]=v;p.sort(b)}return(new Ja(e,this._parents)).order()},call:function(){var a=arguments[0];arguments[0]=this;a.apply(null,arguments);return this},nodes:function(){var a=Array(this.size()),b=-1;this.each(function(){a[++b]=this});return a},node:function(){for(var a=
  this._groups,b=0,c=a.length;b<c;++b)for(var d=a[b],e=0,g=d.length;e<g;++e){var k=d[e];if(k)return k}return null},size:function(){var a=0;this.each(function(){++a});return a},empty:function(){return!this.node()},each:function(a){for(var b=this._groups,c=0,d=b.length;c<d;++c)for(var e=b[c],g=0,k=e.length,m;g<k;++g)(m=e[g])&&a.call(m,m.__data__,g,e);return this},attr:function(a,b){var c=Pc(a);if(2>arguments.length){var d=this.node();return c.local?d.getAttributeNS(c.space,c.local):d.getAttribute(c)}return this.each((null==
  b?c.local?Vn:Un:"function"===typeof b?c.local?Zn:Yn:c.local?Xn:Wn)(c,b))},style:function(a,b,c){return 1<arguments.length?this.each((null==b?$n:"function"===typeof b?bo:ao)(a,b,null==c?"":c)):Pb(this.node(),a)},property:function(a,b){return 1<arguments.length?this.each((null==b?co:"function"===typeof b?fo:eo)(a,b)):this.node()[a]},classed:function(a,b){var c=(a+"").trim().split(/^|\s+/);if(2>arguments.length){for(var d=Mf(this.node()),e=-1,g=c.length;++e<g;)if(!d.contains(c[e]))return!1;return!0}return this.each(("function"===
  typeof b?io:b?go:ho)(c,b))},text:function(a){return arguments.length?this.each(null==a?jo:("function"===typeof a?lo:ko)(a)):this.node().textContent},html:function(a){return arguments.length?this.each(null==a?mo:("function"===typeof a?oo:no)(a)):this.node().innerHTML},raise:function(){return this.each(po)},lower:function(){return this.each(qo)},append:function(a){var b="function"===typeof a?a:Rd(a);return this.select(function(){return this.appendChild(b.apply(this,arguments))})},insert:function(a,
  b){var c="function"===typeof a?a:Rd(a),d=null==b?ro:"function"===typeof b?b:Sd(b);return this.select(function(){return this.insertBefore(c.apply(this,arguments),d.apply(this,arguments)||null)})},remove:function(){return this.each(so)},clone:function(a){return this.select(a?uo:to)},datum:function(a){return arguments.length?this.property("__data__",a):this.node().__data__},on:function(a,b,c){var d=wo(a+""),e=d.length,g;if(2>arguments.length){var k=this.node().__on;if(k)for(var m=0,p=k.length,v;m<p;++m){var h=
  0;for(v=k[m];h<e;++h)if((g=d[h]).type===v.type&&g.name===v.name)return v.value}}else{k=b?yo:xo;null==c&&(c=!1);for(h=0;h<e;++h)this.each(k(d[h],b,c));return this}},dispatch:function(a,b){return this.each(("function"===typeof b?Ao:zo)(a,b))}};var Bo=0;Of.prototype=si.prototype={constructor:Of,get:function(a){for(var b=this._;!(b in a);)if(!(a=a.parentNode))return;return a[b]},set:function(a,b){return a[this._]=b},remove:function(a){return this._ in a&&delete a[this._]},toString:function(){return this._}};
  Qf.prototype.on=function(){var a=this._.on.apply(this._,arguments);return a===this._?this:a};var Jc=1/.7,Go=/^#([0-9a-f]{3})$/,Ho=/^#([0-9a-f]{6})$/,Io=RegExp("^rgb\\(\\s*([+-]?\\d+)\\s*,\\s*([+-]?\\d+)\\s*,\\s*([+-]?\\d+)\\s*\\)$"),Jo=RegExp("^rgb\\(\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)%\\s*,\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)%\\s*,\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)%\\s*\\)$"),Ko=RegExp("^rgba\\(\\s*([+-]?\\d+)\\s*,\\s*([+-]?\\d+)\\s*,\\s*([+-]?\\d+)\\s*,\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)\\s*\\)$"),
  Lo=RegExp("^rgba\\(\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)%\\s*,\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)%\\s*,\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)%\\s*,\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)\\s*\\)$"),Mo=RegExp("^hsl\\(\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)\\s*,\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)%\\s*,\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)%\\s*\\)$"),No=RegExp("^hsla\\(\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)\\s*,\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)%\\s*,\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)%\\s*,\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)\\s*\\)$"),
  wi={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,
  darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,
  hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,
  linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,
  palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,
  turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074};gc(Cb,Db,{displayable:function(){return this.rgb().displayable()},hex:function(){return this.rgb().hex()},toString:function(){return this.rgb()+""}});gc(Fa,hc,Rc(Cb,{brighter:function(a){a=null==a?Jc:Math.pow(Jc,a);return new Fa(this.r*a,this.g*a,this.b*a,this.opacity)},darker:function(a){a=null==a?.7:Math.pow(.7,a);return new Fa(this.r*a,this.g*a,this.b*a,this.opacity)},rgb:function(){return this},
  displayable:function(){return 0<=this.r&&255>=this.r&&0<=this.g&&255>=this.g&&0<=this.b&&255>=this.b&&0<=this.opacity&&1>=this.opacity},hex:function(){return"#"+Sf(this.r)+Sf(this.g)+Sf(this.b)},toString:function(){var a=this.opacity;a=isNaN(a)?1:Math.max(0,Math.min(1,a));return(1===a?"rgb(":"rgba(")+Math.max(0,Math.min(255,Math.round(this.r)||0))+", "+Math.max(0,Math.min(255,Math.round(this.g)||0))+", "+Math.max(0,Math.min(255,Math.round(this.b)||0))+(1===a?")":", "+a+")")}}));gc(ib,Zd,Rc(Cb,{brighter:function(a){a=
  null==a?Jc:Math.pow(Jc,a);return new ib(this.h,this.s,this.l*a,this.opacity)},darker:function(a){a=null==a?.7:Math.pow(.7,a);return new ib(this.h,this.s,this.l*a,this.opacity)},rgb:function(){var a=this.h%360+360*(0>this.h),b=isNaN(a)||isNaN(this.s)?0:this.s,c=this.l;b=c+(.5>c?c:1-c)*b;c=2*c-b;return new Fa(Tf(240<=a?a-240:a+120,c,b),Tf(a,c,b),Tf(120>a?a+240:a-120,c,b),this.opacity)},displayable:function(){return(0<=this.s&&1>=this.s||isNaN(this.s))&&0<=this.l&&1>=this.l&&0<=this.opacity&&1>=this.opacity}}));
  var xi=Math.PI/180,Bi=180/Math.PI,zi=4/29,ic=6/29,yi=3*ic*ic,Po=ic*ic*ic;gc(cb,$d,Rc(Cb,{brighter:function(a){return new cb(this.l+18*(null==a?1:a),this.a,this.b,this.opacity)},darker:function(a){return new cb(this.l-18*(null==a?1:a),this.a,this.b,this.opacity)},rgb:function(){var a=(this.l+16)/116,b=isNaN(this.a)?a:a+this.a/500,c=isNaN(this.b)?a:a-this.b/200;b=.96422*Xf(b);a=1*Xf(a);c=.82521*Xf(c);return new Fa(Yf(3.1338561*b-1.6168667*a-.4906146*c),Yf(-.9787684*b+1.9161415*a+.033454*c),Yf(.0719453*
  b-.2289914*a+1.4052427*c),this.opacity)}}));gc(jb,ae,Rc(Cb,{brighter:function(a){return new jb(this.h,this.c,this.l+18*(null==a?1:a),this.opacity)},darker:function(a){return new jb(this.h,this.c,this.l-18*(null==a?1:a),this.opacity)},rgb:function(){return Uf(this).rgb()}}));var Ci=1.78277*-.29227-.1347134789;gc(Rb,db,Rc(Cb,{brighter:function(a){a=null==a?Jc:Math.pow(Jc,a);return new Rb(this.h,this.s,this.l*a,this.opacity)},darker:function(a){a=null==a?.7:Math.pow(.7,a);return new Rb(this.h,this.s,
  this.l*a,this.opacity)},rgb:function(){var a=isNaN(this.h)?0:(this.h+120)*xi,b=+this.l,c=isNaN(this.s)?0:this.s*b*(1-b),d=Math.cos(a);a=Math.sin(a);return new Fa(255*(b+c*(-.14861*d+1.78277*a)),255*(b+c*(-.29227*d+-.90649*a)),255*(b+1.97294*c*d),this.opacity)}}));var Tc=function c(b){function d(g,k){var m=e((g=hc(g)).r,(k=hc(k)).r),p=e(g.g,k.g),v=e(g.b,k.b),h=Ea(g.opacity,k.opacity);return function(l){g.r=m(l);g.g=p(l);g.b=v(l);g.opacity=h(l);return g+""}}var e=Ro(b);d.gamma=c;return d}(1),Bl=Hi(Ei),
  Xs=Hi(Fi),$f=/[-+]?(?:\d+\.?\d*|\.?\d+)(?:[eE][-+]?\d+)?/g,ag=new RegExp($f.source,"g"),Ni=180/Math.PI,Jh={translateX:0,translateY:0,rotate:0,skewX:0,scaleX:1,scaleY:1},Ld,Kh,im,qf,jm=Oi(function(b){if("none"===b)return Jh;Ld||(Ld=document.createElement("DIV"),Kh=document.documentElement,im=document.defaultView);Ld.style.transform=b;b=im.getComputedStyle(Kh.appendChild(Ld),null).getPropertyValue("transform");Kh.removeChild(Ld);b=b.slice(7,-1).split(",");return Mi(+b[0],+b[1],+b[2],+b[3],+b[4],+b[5])},
  "px, ","px)","deg)"),km=Oi(function(b){if(null==b)return Jh;qf||(qf=document.createElementNS("http://www.w3.org/2000/svg","g"));qf.setAttribute("transform",b);if(!(b=qf.transform.baseVal.consolidate()))return Jh;b=b.matrix;return Mi(b.a,b.b,b.c,b.d,b.e,b.f)},", ",")",")"),Uc=Math.SQRT2,Ys=Ri(ce),Zs=Ri(Ea),$s=Si(ce),at=Si(Ea),bt=Ti(ce),rf=Ti(Ea),kc=0,Xc=0,Zc=0,fe,Yc,ge=0,Sb=0,de=0,Vc="object"===typeof performance&&performance.now?performance:Date,Ui="object"===typeof window&&window.requestAnimationFrame?
  window.requestAnimationFrame.bind(window):function(b){setTimeout(b,17)};Wc.prototype=ee.prototype={constructor:Wc,restart:function(b,c,d){if("function"!==typeof b)throw new TypeError("callback is not a function");d=(null==d?jc():+d)+(null==c?0:+c);this._next||Yc===this||(Yc?Yc._next=this:fe=this,Yc=this);this._call=b;this._time=d;bg()},stop:function(){this._call&&(this._call=null,this._time=Infinity,bg())}};var Xo=Ob("start","end","interrupt"),Yo=[],ct=Qb.prototype.constructor,lm=0,Kc=Qb.prototype;
  kb.prototype=Yi.prototype={constructor:kb,select:function(b){var c=this._name,d=this._id;"function"!==typeof b&&(b=Sd(b));for(var e=this._groups,g=e.length,k=Array(g),m=0;m<g;++m)for(var p=e[m],v=p.length,h=k[m]=Array(v),l,q,w=0;w<v;++w)(l=p[w])&&(q=b.call(l,l.__data__,w,p))&&("__data__"in l&&(q.__data__=l.__data__),h[w]=q,he(h[w],c,d,w,h,eb(l,d)));return new kb(k,this._parents,c,d)},selectAll:function(b){var c=this._name,d=this._id;"function"!==typeof b&&(b=Kf(b));for(var e=this._groups,g=e.length,
  k=[],m=[],p=0;p<g;++p)for(var v=e[p],h=v.length,l,q=0;q<h;++q)if(l=v[q]){for(var w=b.call(l,l.__data__,q,v),B,F=eb(l,d),J=0,P=w.length;J<P;++J)(B=w[J])&&he(B,c,d,J,w,F);k.push(w);m.push(l)}return new kb(k,m,c,d)},filter:function(b){"function"!==typeof b&&(b=Ih(b));for(var c=this._groups,d=c.length,e=Array(d),g=0;g<d;++g)for(var k=c[g],m=k.length,p=e[g]=[],v,h=0;h<m;++h)(v=k[h])&&b.call(v,v.__data__,h,k)&&p.push(v);return new kb(e,this._parents,this._name,this._id)},merge:function(b){if(b._id!==this._id)throw Error();
  var c=this._groups;b=b._groups;for(var d=c.length,e=Math.min(d,b.length),g=Array(d),k=0;k<e;++k)for(var m=c[k],p=b[k],v=m.length,h=g[k]=Array(v),l,q=0;q<v;++q)if(l=m[q]||p[q])h[q]=l;for(;k<d;++k)g[k]=c[k];return new kb(g,this._parents,this._name,this._id)},selection:function(){return new ct(this._groups,this._parents)},transition:function(){for(var b=this._name,c=this._id,d=++lm,e=this._groups,g=e.length,k=0;k<g;++k)for(var m=e[k],p=m.length,v,h=0;h<p;++h)if(v=m[h]){var l=eb(v,c);he(v,b,d,h,m,{time:l.time+
  l.delay+l.duration,delay:0,duration:l.duration,ease:l.ease})}return new kb(e,this._parents,b,d)},call:Kc.call,nodes:Kc.nodes,node:Kc.node,size:Kc.size,empty:Kc.empty,each:Kc.each,on:function(b,c){var d=this._id;return 2>arguments.length?eb(this.node(),d).on.on(b):this.each(op(d,b,c))},attr:function(b,c){var d=Pc(b),e="transform"===d?km:Xi;return this.attrTween(b,"function"===typeof c?(d.local?fp:ep)(d,e,eg(this,"attr."+b,c)):null==c?(d.local?bp:ap)(d):(d.local?dp:cp)(d,e,c+""))},attrTween:function(b,
  c){var d="attr."+b;if(2>arguments.length)return(d=this.tween(d))&&d._value;if(null==c)return this.tween(d,null);if("function"!==typeof c)throw Error();var e=Pc(b);return this.tween(d,(e.local?gp:hp)(e,c))},style:function(b,c,d){var e="transform"===(b+="")?jm:Xi;return null==c?this.styleTween(b,qp(b,e)).on("end.style."+b,rp(b)):this.styleTween(b,"function"===typeof c?tp(b,e,eg(this,"style."+b,c)):sp(b,e,c+""),d)},styleTween:function(b,c,d){var e="style."+(b+="");if(2>arguments.length)return(e=this.tween(e))&&
  e._value;if(null==c)return this.tween(e,null);if("function"!==typeof c)throw Error();return this.tween(e,up(b,c,null==d?"":d))},text:function(b){return this.tween("text","function"===typeof b?wp(eg(this,"text",b)):vp(null==b?"":b+""))},remove:function(){return this.on("end.remove",pp(this._id))},tween:function(b,c){var d=this._id;b+="";if(2>arguments.length){d=eb(this.node(),d).tween;for(var e=0,g=d.length,k;e<g;++e)if((k=d[e]).name===b)return k.value;return null}return this.each((null==c?Zo:$o)(d,
  b,c))},delay:function(b){var c=this._id;return arguments.length?this.each(("function"===typeof b?ip:jp)(c,b)):eb(this.node(),c).delay},duration:function(b){var c=this._id;return arguments.length?this.each(("function"===typeof b?kp:lp)(c,b)):eb(this.node(),c).duration},ease:function(b){var c=this._id;return arguments.length?this.each(mp(c,b)):eb(this.node(),c).ease}};var dt=function d(c){function e(g){return Math.pow(g,c)}c=+c;e.exponent=d;return e}(3),et=function e(d){function g(k){return 1-Math.pow(1-
  k,d)}d=+d;g.exponent=e;return g}(3),mm=function g(e){function k(m){return(1>=(m*=2)?Math.pow(m,e):2-Math.pow(2-m,e))/2}e=+e;k.exponent=g;return k}(3),aj=Math.PI,nm=aj/2,gg=4/11,yp=6/11,xp=8/11,Ap=9/11,zp=10/11,Bp=21/22,ie=1/gg/gg,ft=function k(g){function m(p){return p*p*((g+1)*p-g)}g=+g;m.overshoot=k;return m}(1.70158),gt=function m(k){function p(v){return--v*v*((k+1)*v+k)+1}k=+k;p.overshoot=m;return p}(1.70158),om=function p(m){function v(h){return(1>(h*=2)?h*h*((m+1)*h-m):(h-=2)*h*((m+1)*h+m)+
  2)/2}m=+m;v.overshoot=p;return v}(1.70158),Lc=2*Math.PI,ht=function h(p,v){function l(w){return p*Math.pow(2,10*--w)*Math.sin((q-w)/v)}var q=Math.asin(1/(p=Math.max(1,p)))*(v/=Lc);l.amplitude=function(w){return h(w,v*Lc)};l.period=function(w){return h(p,w)};return l}(1,.3),pm=function l(v,h){function q(B){return 1-v*Math.pow(2,-10*(B=+B))*Math.sin((B+w)/h)}var w=Math.asin(1/(v=Math.max(1,v)))*(h/=Lc);q.amplitude=function(B){return l(B,h*Lc)};q.period=function(B){return l(v,B)};return q}(1,.3),it=
  function q(h,l){function w(F){return(0>(F=2*F-1)?h*Math.pow(2,10*F)*Math.sin((B-F)/l):2-h*Math.pow(2,-10*F)*Math.sin((B+F)/l))/2}var B=Math.asin(1/(h=Math.max(1,h)))*(l/=Lc);w.amplitude=function(F){return q(F,l*Lc)};w.period=function(F){return q(h,F)};return w}(1,.3),Lh={time:null,delay:0,duration:250,ease:fg};Qb.prototype.interrupt=function(h){return this.each(function(){Ub(this,h)})};Qb.prototype.transition=function(h){var l;if(h instanceof kb){var q=h._id;h=h._name}else q=++lm,(l=Lh).time=jc(),
  h=null==h?null:h+"";for(var w=this._groups,B=w.length,F=0;F<B;++F)for(var J=w[F],P=J.length,x,y=0;y<P;++y)if(x=J[y]){var I=x,Q=h,V=q,N=y,T=J,f;if(!(f=l))a:{f=void 0;for(var n=q;!(f=x.__transition)||!(f=f[n]);)if(!(x=x.parentNode)){f=(Lh.time=jc(),Lh);break a}}he(I,Q,V,N,T,f)}return new kb(w,this._parents,h,q)};var jt=[null],ej={name:"drag"},kg={name:"space"},lc={name:"handle"},mc={name:"center"},le={name:"x",handles:["e","w"].map(ad),input:function(h,l){return h&&[[h[0],l[0][1]],[h[1],l[1][1]]]},
  output:function(h){return h&&[h[0][0],h[1][0]]}},ke={name:"y",handles:["n","s"].map(ad),input:function(h,l){return h&&[[l[0][0],h[0]],[l[1][0],h[1]]]},output:function(h){return h&&[h[0][1],h[1][1]]}},kt={name:"xy",handles:"n e s w nw ne se sw".split(" ").map(ad),input:function(h){return h},output:function(h){return h}},qb={overlay:"crosshair",selection:"move",n:"ns-resize",e:"ew-resize",s:"ns-resize",w:"ew-resize",nw:"nwse-resize",ne:"nesw-resize",se:"nwse-resize",sw:"nesw-resize"},fj={e:"w",w:"e",
  nw:"ne",ne:"nw",se:"sw",sw:"se"},gj={n:"s",s:"n",nw:"sw",ne:"se",se:"ne",sw:"nw"},Fp={overlay:1,selection:1,n:null,e:1,s:null,w:-1,nw:-1,ne:1,se:1,sw:-1},Gp={overlay:1,selection:1,n:-1,e:null,s:1,w:null,nw:-1,ne:-1,se:1,sw:1},qm=Math.cos,rm=Math.sin,sm=Math.PI,sf=sm/2,tm=2*sm,um=Math.max,lt=Array.prototype.slice,Mh=Math.PI,Nh=2*Mh,mt=Nh-1E-6;mg.prototype=Eb.prototype={constructor:mg,moveTo:function(h,l){this._+="M"+(this._x0=this._x1=+h)+","+(this._y0=this._y1=+l)},closePath:function(){null!==this._x1&&
  (this._x1=this._x0,this._y1=this._y0,this._+="Z")},lineTo:function(h,l){this._+="L"+(this._x1=+h)+","+(this._y1=+l)},quadraticCurveTo:function(h,l,q,w){this._+="Q"+ +h+","+ +l+","+(this._x1=+q)+","+(this._y1=+w)},bezierCurveTo:function(h,l,q,w,B,F){this._+="C"+ +h+","+ +l+","+ +q+","+ +w+","+(this._x1=+B)+","+(this._y1=+F)},arcTo:function(h,l,q,w,B){h=+h;l=+l;q=+q;w=+w;B=+B;var F=this._x1,J=this._y1,P=q-h,x=w-l,y=F-h,I=J-l,Q=y*y+I*I;if(0>B)throw Error("negative radius: "+B);if(null===this._x1)this._+=
  "M"+(this._x1=h)+","+(this._y1=l);else if(1E-6<Q)if(1E-6<Math.abs(I*P-x*y)&&B){q-=F;w-=J;var V=P*P+x*x;J=Math.sqrt(V);F=Math.sqrt(Q);Q=B*Math.tan((Mh-Math.acos((V+Q-(q*q+w*w))/(2*J*F)))/2);F=Q/F;Q/=J;1E-6<Math.abs(F-1)&&(this._+="L"+(h+F*y)+","+(l+F*I));this._+="A"+B+","+B+",0,0,"+ +(I*q>y*w)+","+(this._x1=h+Q*P)+","+(this._y1=l+Q*x)}else this._+="L"+(this._x1=h)+","+(this._y1=l)},arc:function(h,l,q,w,B,F){h=+h;l=+l;q=+q;var J=q*Math.cos(w),P=q*Math.sin(w),x=h+J,y=l+P,I=1^F;w=F?w-B:B-w;if(0>q)throw Error("negative radius: "+
  q);if(null===this._x1)this._+="M"+x+","+y;else if(1E-6<Math.abs(this._x1-x)||1E-6<Math.abs(this._y1-y))this._+="L"+x+","+y;q&&(0>w&&(w=w%Nh+Nh),w>mt?this._+="A"+q+","+q+",0,1,"+I+","+(h-J)+","+(l-P)+"A"+q+","+q+",0,1,"+I+","+(this._x1=x)+","+(this._y1=y):1E-6<w&&(this._+="A"+q+","+q+",0,"+ +(w>=Mh)+","+I+","+(this._x1=h+q*Math.cos(B))+","+(this._y1=l+q*Math.sin(B))))},rect:function(h,l,q,w){this._+="M"+(this._x0=this._x1=+h)+","+(this._y0=this._y1=+l)+"h"+ +q+"v"+ +w+"h"+-q+"Z"},toString:function(){return this._}};
  me.prototype=rb.prototype={constructor:me,has:function(h){return" "+h in this},get:function(h){return this[" "+h]},set:function(h,l){this[" "+h]=l;return this},remove:function(h){h=" "+h;return h in this&&delete this[h]},clear:function(){for(var h in this)" "===h[0]&&delete this[h]},keys:function(){var h=[],l;for(l in this)" "===l[0]&&h.push(l.slice(1));return h},values:function(){var h=[],l;for(l in this)" "===l[0]&&h.push(this[l]);return h},entries:function(){var h=[],l;for(l in this)" "===l[0]&&
  h.push({key:l.slice(1),value:this[l]});return h},size:function(){var h=0,l;for(l in this)" "===l[0]&&++h;return h},empty:function(){for(var h in this)if(" "===h[0])return!1;return!0},each:function(h){for(var l in this)" "===l[0]&&h(this[l],l.slice(1),this)}};var cc=rb.prototype;ne.prototype=jj.prototype={constructor:ne,has:cc.has,add:function(h){h+="";this[" "+h]=h;return this},remove:cc.remove,clear:cc.clear,values:cc.keys,size:cc.size,empty:cc.empty,each:cc.each};var lj=Array.prototype.slice,sb=
  [[],[[[1,1.5],[.5,1]]],[[[1.5,1],[1,1.5]]],[[[1.5,1],[.5,1]]],[[[1,.5],[1.5,1]]],[[[1,1.5],[.5,1]],[[1,.5],[1.5,1]]],[[[1,.5],[1,1.5]]],[[[1,.5],[.5,1]]],[[[.5,1],[1,.5]]],[[[1,1.5],[1,.5]]],[[[.5,1],[1,.5]],[[1.5,1],[1,1.5]]],[[[1.5,1],[1,.5]]],[[[.5,1],[1.5,1]]],[[[1,1.5],[1.5,1]]],[[[.5,1],[1,1.5]]],[]],nj={},pg={},tf=oe(","),vm=tf.parse,nt=tf.parseRows,ot=tf.format,pt=tf.formatRows,uf=oe("\t"),wm=uf.parse,qt=uf.parseRows,rt=uf.format,st=uf.formatRows,tt=oj(vm),ut=oj(wm),Qa=pe.prototype=rg.prototype;
  Qa.copy=function(){var h=new rg(this._x,this._y,this._x0,this._y0,this._x1,this._y1),l=this._root,q,w;if(!l)return h;if(!l.length)return h._root=qj(l),h;for(q=[{source:l,target:h._root=Array(4)}];l=q.pop();)for(var B=0;4>B;++B)if(w=l.source[B])w.length?q.push({source:w,target:l.target[B]=Array(4)}):l.target[B]=qj(w);return h};Qa.add=function(h){var l=+this._x.call(null,h),q=+this._y.call(null,h);return pj(this.cover(l,q),l,q,h)};Qa.addAll=function(h){var l,q,w=h.length,B,F,J=Array(w),P=Array(w),x=
  Infinity,y=Infinity,I=-Infinity,Q=-Infinity;for(q=0;q<w;++q)isNaN(B=+this._x.call(null,l=h[q]))||isNaN(F=+this._y.call(null,l))||(J[q]=B,P[q]=F,B<x&&(x=B),B>I&&(I=B),F<y&&(y=F),F>Q&&(Q=F));I<x&&(x=this._x0,I=this._x1);Q<y&&(y=this._y0,Q=this._y1);this.cover(x,y).cover(I,Q);for(q=0;q<w;++q)pj(this,J[q],P[q],h[q]);return this};Qa.cover=function(h,l){if(isNaN(h=+h)||isNaN(l=+l))return this;var q=this._x0,w=this._y0,B=this._x1,F=this._y1;if(isNaN(q))B=(q=Math.floor(h))+1,F=(w=Math.floor(l))+1;else if(q>
  h||h>B||w>l||l>F){var J=B-q,P=this._root,x;switch(x=(l<(w+F)/2)<<1|h<(q+B)/2){case 0:do{var y=Array(4);y[x]=P;P=y}while(J*=2,B=q+J,F=w+J,h>B||l>F);break;case 1:do y=Array(4),y[x]=P,P=y;while(J*=2,q=B-J,F=w+J,q>h||l>F);break;case 2:do y=Array(4),y[x]=P,P=y;while(J*=2,B=q+J,w=F-J,h>B||w>l);break;case 3:do y=Array(4),y[x]=P,P=y;while(J*=2,q=B-J,w=F-J,q>h||w>l)}this._root&&this._root.length&&(this._root=P)}else return this;this._x0=q;this._y0=w;this._x1=B;this._y1=F;return this};Qa.data=function(){var h=
  [];this.visit(function(l){if(!l.length){do h.push(l.data);while(l=l.next)}});return h};Qa.extent=function(h){return arguments.length?this.cover(+h[0][0],+h[0][1]).cover(+h[1][0],+h[1][1]):isNaN(this._x0)?void 0:[[this._x0,this._y0],[this._x1,this._y1]]};Qa.find=function(h,l,q){var w=this._x0,B=this._y0,F,J,P,x,y=this._x1,I=this._y1,Q=[],V=this._root,N;V&&Q.push(new Ka(V,w,B,y,I));null==q?q=Infinity:(w=h-q,B=l-q,y=h+q,I=l+q,q*=q);for(;N=Q.pop();)if(!(!(V=N.node)||(F=N.x0)>y||(J=N.y0)>I||(P=N.x1)<w||
  (x=N.y1)<B))if(V.length){N=(F+P)/2;var T=(J+x)/2;Q.push(new Ka(V[3],N,T,P,x),new Ka(V[2],F,T,N,x),new Ka(V[1],N,J,P,T),new Ka(V[0],F,J,N,T));if(V=(l>=T)<<1|h>=N)N=Q[Q.length-1],Q[Q.length-1]=Q[Q.length-1-V],Q[Q.length-1-V]=N}else if(N=h-+this._x.call(null,V.data),T=l-+this._y.call(null,V.data),N=N*N+T*T,N<q){var f=Math.sqrt(q=N);w=h-f;B=l-f;y=h+f;I=l+f;f=V.data}return f};Qa.remove=function(h){if(isNaN(x=+this._x.call(null,h))||isNaN(y=+this._y.call(null,h)))return this;var l,q=this._root,w,B=this._x0,
  F=this._y0,J=this._x1,P=this._y1,x,y,I,Q,V,N,T;if(!q)return this;if(q.length)for(;;){(V=x>=(I=(B+J)/2))?B=I:J=I;(N=y>=(Q=(F+P)/2))?F=Q:P=Q;if(!(l=q,q=q[T=N<<1|V]))return this;if(!q.length)break;if(l[T+1&3]||l[T+2&3]||l[T+3&3]){var f=l;var n=T}}for(;q.data!==h;)if(!(w=q,q=q.next))return this;(h=q.next)&&delete q.next;if(w)return h?w.next=h:delete w.next,this;if(!l)return this._root=h,this;h?l[T]=h:delete l[T];(q=l[0]||l[1]||l[2]||l[3])&&q===(l[3]||l[2]||l[1]||l[0])&&!q.length&&(f?f[n]=q:this._root=
  q);return this};Qa.removeAll=function(h){for(var l=0,q=h.length;l<q;++l)this.remove(h[l]);return this};Qa.root=function(){return this._root};Qa.size=function(){var h=0;this.visit(function(l){if(!l.length){do++h;while(l=l.next)}});return h};Qa.visit=function(h){var l=[],q,w=this._root,B,F,J,P,x;for(w&&l.push(new Ka(w,this._x0,this._y0,this._x1,this._y1));q=l.pop();)if(!h(w=q.node,F=q.x0,J=q.y0,P=q.x1,x=q.y1)&&w.length){q=(F+P)/2;var y=(J+x)/2;(B=w[3])&&l.push(new Ka(B,q,y,P,x));(B=w[2])&&l.push(new Ka(B,
  F,y,q,x));(B=w[1])&&l.push(new Ka(B,q,J,P,y));(B=w[0])&&l.push(new Ka(B,F,J,q,y))}return this};Qa.visitAfter=function(h){var l=[],q=[],w;for(this._root&&l.push(new Ka(this._root,this._x0,this._y0,this._x1,this._y1));w=l.pop();){var B=w.node;if(B.length){var F,J=w.x0,P=w.y0,x=w.x1,y=w.y1,I=(J+x)/2,Q=(P+y)/2;(F=B[0])&&l.push(new Ka(F,J,P,I,Q));(F=B[1])&&l.push(new Ka(F,I,P,x,Q));(F=B[2])&&l.push(new Ka(F,J,Q,I,y));(F=B[3])&&l.push(new Ka(F,I,Q,x,y))}q.push(w)}for(;w=q.pop();)h(w.node,w.x0,w.y0,w.x1,
  w.y1);return this};Qa.x=function(h){return arguments.length?(this._x=h,this):this._x};Qa.y=function(h){return arguments.length?(this._y=h,this):this._y};var vt=Math.PI*(3-Math.sqrt(5)),iq=/^(?:(.)?([<>=^]))?([+\-( ])?([$#])?(0)?(\d+)?(,)?(\.\d+)?(~)?([a-z%])?$/i;bd.prototype=sg.prototype;sg.prototype.toString=function(){return this.fill+this.align+this.sign+this.symbol+(this.zero?"0":"")+(null==this.width?"":Math.max(1,this.width|0))+(this.comma?",":"")+(null==this.precision?"":"."+Math.max(0,this.precision|
  0))+(this.trim?"~":"")+this.type};var wj,xj={"%":function(h,l){return(100*h).toFixed(l)},b:function(h){return Math.round(h).toString(2)},c:function(h){return h+""},d:function(h){return Math.round(h).toString(10)},e:function(h,l){return h.toExponential(l)},f:function(h,l){return h.toFixed(l)},g:function(h,l){return h.toPrecision(l)},o:function(h){return Math.round(h).toString(8)},p:function(h,l){return sj(100*h,l)},r:sj,s:function(h,l){var q=qe(h,l);if(!q)return h+"";var w=q[0];q=q[1];q=q-(wj=3*Math.max(-8,
  Math.min(8,Math.floor(q/3))))+1;var B=w.length;return q===B?w:q>B?w+Array(q-B+1).join("0"):0<q?w.slice(0,q)+"."+w.slice(q):"0."+Array(1-q).join("0")+qe(h,Math.max(0,l+q-1))[0]},X:function(h){return Math.round(h).toString(16).toUpperCase()},x:function(h){return Math.round(h).toString(16)}},vj="y z a f p n \u00b5 m  k M G T P E Z Y".split(" "),re;yj({decimal:".",thousands:",",grouping:[3],currency:["$",""]});fb.prototype={constructor:fb,reset:function(){this.s=this.t=0},add:function(h){Cj(vf,h,this.t);
  Cj(this,vf.s,this.s);this.s?this.t+=vf.t:this.s=vf.t},valueOf:function(){return this.s}};var vf=new fb,oa=Math.PI,wa=oa/2,te=oa/4,Sa=2*oa,va=180/oa,ia=oa/180,ra=Math.abs,wc=Math.atan,Ma=Math.atan2,da=Math.cos,Me=Math.ceil,xm=Math.exp,Re=Math.log,Zg=Math.pow,ca=Math.sin,ld=Math.sign||function(h){return 0<h?1:0>h?-1:0},Ba=Math.sqrt,vc=Math.tan,Hj={Feature:function(h,l){se(h.geometry,l)},FeatureCollection:function(h,l){h=h.features;for(var q=-1,w=h.length;++q<w;)se(h[q].geometry,l)}},Fj={Sphere:function(h,
  l){l.sphere()},Point:function(h,l){h=h.coordinates;l.point(h[0],h[1],h[2])},MultiPoint:function(h,l){for(var q=h.coordinates,w=-1,B=q.length;++w<B;)h=q[w],l.point(h[0],h[1],h[2])},LineString:function(h,l){tg(h.coordinates,l,0)},MultiLineString:function(h,l){h=h.coordinates;for(var q=-1,w=h.length;++q<w;)tg(h[q],l,0)},Polygon:function(h,l){Gj(h.coordinates,l)},MultiPolygon:function(h,l){h=h.coordinates;for(var q=-1,w=h.length;++q<w;)Gj(h[q],l)},GeometryCollection:function(h,l){h=h.geometries;for(var q=
  -1,w=h.length;++q<w;)se(h[q],l)}},ue=new fb,wf=new fb,Jj,Kj,ug,vg,wg,lb={point:xa,lineStart:xa,lineEnd:xa,polygonStart:function(){ue.reset();lb.lineStart=jq;lb.lineEnd=lq},polygonEnd:function(){var h=+ue;wf.add(0>h?Sa+h:h);this.lineStart=this.lineEnd=this.point=xa},sphere:function(){wf.add(Sa)}},za,Wa,Aa,Za,Wb,Pj,Qj,pc,cd=new fb,Hb,tb,ub={point:yg,lineStart:Mj,lineEnd:Nj,polygonStart:function(){ub.point=Oj;ub.lineStart=mq;ub.lineEnd=nq;cd.reset();lb.polygonStart()},polygonEnd:function(){lb.polygonEnd();
  ub.point=yg;ub.lineStart=Mj;ub.lineEnd=Nj;0>ue?(za=-(Aa=180),Wa=-(Za=90)):1E-6<cd?Za=90:-1E-6>cd&&(Wa=-90);tb[0]=za;tb[1]=Aa}},ed,Ce,ze,Ae,Be,De,Ee,Fe,Ag,Bg,Cg,Vj,Wj,Na,Oa,Pa,hb={sphere:xa,point:zg,lineStart:Sj,lineEnd:Tj,polygonStart:function(){hb.lineStart=rq;hb.lineEnd=tq},polygonEnd:function(){hb.lineStart=Sj;hb.lineEnd=Tj}};Eg.invert=Eg;var Gg=new fb,Xg=hk(function(){return!0},function(h){var l=NaN,q=NaN,w=NaN,B;return{lineStart:function(){h.lineStart();B=1},point:function(F,J){var P=0<F?oa:
  -oa,x=ra(F-l);if(1E-6>ra(x-oa))h.point(l,q=0<(q+J)/2?wa:-wa),h.point(w,q),h.lineEnd(),h.lineStart(),h.point(P,q),h.point(F,q),B=0;else if(w!==P&&x>=oa){1E-6>ra(l-w)&&(l-=1E-6*w);1E-6>ra(F-P)&&(F-=1E-6*P);x=l;var y=q,I=F,Q,V,N=ca(x-I);q=1E-6<ra(N)?wc((ca(y)*(V=da(J))*ca(I)-ca(J)*(Q=da(y))*ca(x))/(Q*V*N)):(y+J)/2;h.point(w,q);h.lineEnd();h.lineStart();h.point(P,q);B=0}h.point(l=F,q=J);w=P},lineEnd:function(){h.lineEnd();l=q=NaN},clean:function(){return 2-B}}},function(h,l,q,w){null==h?(q*=wa,w.point(-oa,
  q),w.point(0,q),w.point(oa,q),w.point(oa,0),w.point(oa,-q),w.point(0,-q),w.point(-oa,-q),w.point(-oa,0),w.point(-oa,q)):1E-6<ra(h[0]-l[0])?(h=h[0]<l[0]?oa:-oa,q=q*h/2,w.point(-h,q),w.point(0,q),w.point(h,q)):w.point(l[0],l[1])},[-oa,-wa]),Ig=new fb,Hg,Je,Ke,rc={sphere:xa,point:xa,lineStart:function(){rc.point=yq;rc.lineEnd=xq},lineEnd:xa,polygonStart:xa,polygonEnd:xa},Jg=[null,null],Aq={type:"LineString",coordinates:Jg},ym={Feature:function(h,l){return Le(h.geometry,l)},FeatureCollection:function(h,
  l){h=h.features;for(var q=-1,w=h.length;++q<w;)if(Le(h[q].geometry,l))return!0;return!1}},kk={Sphere:function(){return!0},Point:function(h,l){return 0===sc(h.coordinates,l)},MultiPoint:function(h,l){h=h.coordinates;for(var q=-1,w=h.length;++q<w;)if(0===sc(h[q],l))return!0;return!1},LineString:function(h,l){return lk(h.coordinates,l)},MultiLineString:function(h,l){h=h.coordinates;for(var q=-1,w=h.length;++q<w;)if(lk(h[q],l))return!0;return!1},Polygon:function(h,l){return mk(h.coordinates,l)},MultiPolygon:function(h,
  l){h=h.coordinates;for(var q=-1,w=h.length;++q<w;)if(mk(h[q],l))return!0;return!1},GeometryCollection:function(h,l){h=h.geometries;for(var q=-1,w=h.length;++q<w;)if(Le(h[q],l))return!0;return!1}},Oh=new fb,Mg=new fb,sk,tk,Kg,Lg,vb={point:xa,lineStart:xa,lineEnd:xa,polygonStart:function(){vb.lineStart=Cq;vb.lineEnd=Eq},polygonEnd:function(){vb.lineStart=vb.lineEnd=vb.point=xa;Oh.add(ra(Mg));Mg.reset()},result:function(){var h=Oh/2;Oh.reset();return h}},Mc=Infinity,xf=Mc,Md=-Mc,yf=Md,Pe={point:function(h,
  l){h<Mc&&(Mc=h);h>Md&&(Md=h);l<xf&&(xf=l);l>yf&&(yf=l)},lineStart:xa,lineEnd:xa,polygonStart:xa,polygonEnd:xa,result:function(){var h=[[Mc,xf],[Md,yf]];Md=yf=-(xf=Mc=Infinity);return h}},Ng=0,Og=0,fd=0,Ne=0,Oe=0,tc=0,Pg=0,Qg=0,gd=0,xk,yk,mb,nb,$a={point:Yb,lineStart:uk,lineEnd:vk,polygonStart:function(){$a.lineStart=Hq;$a.lineEnd=Jq},polygonEnd:function(){$a.point=Yb;$a.lineStart=uk;$a.lineEnd=vk},result:function(){var h=gd?[Pg/gd,Qg/gd]:tc?[Ne/tc,Oe/tc]:fd?[Ng/fd,Og/fd]:[NaN,NaN];Ng=Og=fd=Ne=Oe=
  tc=Pg=Qg=gd=0;return h}};zk.prototype={_radius:4.5,pointRadius:function(h){return this._radius=h,this},polygonStart:function(){this._line=0},polygonEnd:function(){this._line=NaN},lineStart:function(){this._point=0},lineEnd:function(){0===this._line&&this._context.closePath();this._point=NaN},point:function(h,l){switch(this._point){case 0:this._context.moveTo(h,l);this._point=1;break;case 1:this._context.lineTo(h,l);break;default:this._context.moveTo(h+this._radius,l),this._context.arc(h,l,this._radius,
  0,Sa)}},result:xa};var Rg=new fb,Ph,Bk,Ck,id,jd,hd={point:xa,lineStart:function(){hd.point=Kq},lineEnd:function(){Ph&&Ak(Bk,Ck);hd.point=xa},polygonStart:function(){Ph=!0},polygonEnd:function(){Ph=null},result:function(){var h=+Rg;Rg.reset();return h}};Dk.prototype={_radius:4.5,_circle:Ek(4.5),pointRadius:function(h){(h=+h)!==this._radius&&(this._radius=h,this._circle=null);return this},polygonStart:function(){this._line=0},polygonEnd:function(){this._line=NaN},lineStart:function(){this._point=0},
  lineEnd:function(){0===this._line&&this._string.push("Z");this._point=NaN},point:function(h,l){switch(this._point){case 0:this._string.push("M",h,",",l);this._point=1;break;case 1:this._string.push("L",h,",",l);break;default:null==this._circle&&(this._circle=Ek(this._radius)),this._string.push("M",h,",",l,this._circle)}},result:function(){if(this._string.length){var h=this._string.join("");this._string=[];return h}return null}};Sg.prototype={constructor:Sg,point:function(h,l){this.stream.point(h,
  l)},sphere:function(){this.stream.sphere()},lineStart:function(){this.stream.lineStart()},lineEnd:function(){this.stream.lineEnd()},polygonStart:function(){this.stream.polygonStart()},polygonEnd:function(){this.stream.polygonEnd()}};var Lq=da(30*ia),Oq=kd({point:function(h,l){this.stream.point(h*ia,l*ia)}}),Qh=Kk(function(h){return Ba(2/(1+h))});Qh.invert=md(function(h){return 2*La(h/2)});var Rh=Kk(function(h){return(h=Dj(h))&&h/ca(h)});Rh.invert=md(function(h){return h});nd.invert=function(h,l){return[h,
  2*wc(xm(l))-wa]};od.invert=od;var Se=Ba(3)/2;$g.invert=function(h,l){for(var q=l,w=q*q,B=w*w*w,F=0,J;12>F&&!(J=q*(1.340264+-.081106*w+B*(8.93E-4+.003796*w))-l,w=1.340264+3*-.081106*w+B*(7*8.93E-4+.034164*w),q-=J/=w,w=q*q,B=w*w*w,1E-12>ra(J));++F);return[Se*h*(1.340264+3*-.081106*w+B*(7*8.93E-4+.034164*w))/da(q),La(ca(q)/Se)]};ah.invert=md(wc);bh.invert=function(h,l){var q=l,w=25;do{var B=q*q;var F=B*B;q-=F=(q*(1.007226+B*(.015085+F*(-.044475+.028874*B-.005916*F)))-l)/(1.007226+B*(.045255+F*(-.311325+
  .259866*B-.005916*11*F)))}while(1E-6<ra(F)&&0<--w);return[h/(.8707+(B=q*q)*(-.131979+B*(-.013791+B*B*B*(.003971-.001529*B)))),q]};ch.invert=md(La);dh.invert=md(function(h){return 2*wc(h)});eh.invert=function(h,l){return[-l,2*wc(xm(h))-wa]};xc.prototype=fh.prototype={constructor:xc,count:function(){return this.eachAfter(Wq)},each:function(h){var l,q=[this],w;do{var B=q.reverse();for(q=[];l=B.pop();)if(h(l),l=l.children){var F=0;for(w=l.length;F<w;++F)q.push(l[F])}}while(q.length);return this},eachAfter:function(h){for(var l,
  q=[this],w=[],B,F;l=q.pop();)if(w.push(l),l=l.children)for(B=0,F=l.length;B<F;++B)q.push(l[B]);for(;l=w.pop();)h(l);return this},eachBefore:function(h){for(var l,q=[this],w;l=q.pop();)if(h(l),l=l.children)for(w=l.length-1;0<=w;--w)q.push(l[w]);return this},sum:function(h){return this.eachAfter(function(l){for(var q=+h(l.data)||0,w=l.children,B=w&&w.length;0<=--B;)q+=w[B].value;l.value=q})},sort:function(h){return this.eachBefore(function(l){l.children&&l.children.sort(h)})},path:function(h){var l=
  this;var q=l;var w=h;if(q!==w){var B=q.ancestors(),F=w.ancestors(),J=null;q=B.pop();for(w=F.pop();q===w;)J=q,q=B.pop(),w=F.pop();q=J}for(w=[l];l!==q;)l=l.parent,w.push(l);for(l=w.length;h!==q;)w.splice(l,0,h),h=h.parent;return w},ancestors:function(){for(var h=this,l=[h];h=h.parent;)l.push(h);return l},descendants:function(){var h=[];this.each(function(l){h.push(l)});return h},leaves:function(){var h=[];this.eachBefore(function(l){l.children||h.push(l)});return h},links:function(){var h=this,l=[];
  h.each(function(q){q!==h&&l.push({source:q.parent,target:q})});return l},copy:function(){return fh(this).eachBefore(Yq)}};var Zq=Array.prototype.slice,wt={depth:-1},zm={};Xe.prototype=Object.create(xc.prototype);var Am=(1+Math.sqrt(5))/2,Bm=function q(l){function w(B,F,J,P,x){Zk(l,B,F,J,P,x)}w.ratio=function(B){return q(1<(B=+B)?B:1)};return w}(Am),xt=function w(q){function B(F,J,P,x,y){if((I=F._squarify)&&I.ratio===q)for(var I,Q,V,N=-1,T,f=I.length,n=F.value;++N<f;){F=I[N];Q=F.children;V=F.value=
  0;for(T=Q.length;V<T;++V)F.value+=Q[V].value;F.dice?qd(F,J,P,x,P+=(y-P)*F.value/n):Ye(F,J,P,J+=(x-J)*F.value/n,y);n-=F.value}else F._squarify=I=Zk(q,F,J,P,x,y),I.ratio=q}B.ratio=function(F){return w(1<(F=+F)?F:1)};return B}(Am),yt=function B(w){function F(J,P){J=null==J?0:+J;P=null==P?1:+P;1===arguments.length?(P=J,J=0):P-=J;return function(){return w()*P+J}}F.source=B;return F}(zc),Cm=function F(B){function J(P,x){var y,I;P=null==P?0:+P;x=null==x?1:+x;return function(){if(null!=y){var Q=y;y=null}else{do y=
  2*B()-1,Q=2*B()-1,I=y*y+Q*Q;while(!I||1<I)}return P+x*Q*Math.sqrt(-2*Math.log(I)/I)}}J.source=F;return J}(zc),zt=function J(F){function P(){var x=Cm.source(F).apply(this,arguments);return function(){return Math.exp(x())}}P.source=J;return P}(zc),Dm=function P(J){function x(y){return function(){for(var I=0,Q=0;Q<y;++Q)I+=J();return I}}x.source=P;return x}(zc),At=function x(P){function y(I){var Q=Dm.source(P)(I);return function(){return Q()/I}}y.source=x;return y}(zc),Bt=function y(x){function I(Q){return function(){return-Math.log(1-
  x())/Q}}I.source=y;return I}(zc),Em=Array.prototype,ph=Em.map,Ib=Em.slice,lh={name:"implicit"},cl=[0,1],rh=new Date,sh=new Date,dc=Da(function(){},function(x,y){x.setTime(+x+y)},function(x,y){return y-x});dc.every=function(x){x=Math.floor(x);return isFinite(x)&&0<x?1<x?Da(function(y){y.setTime(Math.floor(y/x)*x)},function(y,I){y.setTime(+y+I*x)},function(y,I){return(I-y)/x}):dc:null};var Fm=dc.range,Nd=Da(function(x){x.setTime(1E3*Math.floor(x/1E3))},function(x,y){x.setTime(+x+1E3*y)},function(x,
  y){return(y-x)/1E3},function(x){return x.getUTCSeconds()}),Gm=Nd.range,Sh=Da(function(x){x.setTime(6E4*Math.floor(x/6E4))},function(x,y){x.setTime(+x+6E4*y)},function(x,y){return(y-x)/6E4},function(x){return x.getMinutes()}),Ct=Sh.range,Th=Da(function(x){var y=6E4*x.getTimezoneOffset()%36E5;0>y&&(y+=36E5);x.setTime(36E5*Math.floor((+x-y)/36E5)+y)},function(x,y){x.setTime(+x+36E5*y)},function(x,y){return(y-x)/36E5},function(x){return x.getHours()}),Dt=Th.range,vd=Da(function(x){x.setHours(0,0,0,0)},
  function(x,y){x.setDate(x.getDate()+y)},function(x,y){return(y-x-6E4*(y.getTimezoneOffset()-x.getTimezoneOffset()))/864E5},function(x){return x.getDate()-1}),Et=vd.range,yd=ac(0),ud=ac(1),Hm=ac(2),Im=ac(3),zd=ac(4),Jm=ac(5),Km=ac(6),Lm=yd.range,Ft=ud.range,Gt=Hm.range,Ht=Im.range,It=zd.range,Jt=Jm.range,Kt=Km.range,Uh=Da(function(x){x.setDate(1);x.setHours(0,0,0,0)},function(x,y){x.setMonth(x.getMonth()+y)},function(x,y){return y.getMonth()-x.getMonth()+12*(y.getFullYear()-x.getFullYear())},function(x){return x.getMonth()}),
  Lt=Uh.range,wb=Da(function(x){x.setMonth(0,1);x.setHours(0,0,0,0)},function(x,y){x.setFullYear(x.getFullYear()+y)},function(x,y){return y.getFullYear()-x.getFullYear()},function(x){return x.getFullYear()});wb.every=function(x){return isFinite(x=Math.floor(x))&&0<x?Da(function(y){y.setFullYear(Math.floor(y.getFullYear()/x)*x);y.setMonth(0,1);y.setHours(0,0,0,0)},function(y,I){y.setFullYear(y.getFullYear()+I*x)}):null};var Mt=wb.range,Vh=Da(function(x){x.setUTCSeconds(0,0)},function(x,y){x.setTime(+x+
  6E4*y)},function(x,y){return(y-x)/6E4},function(x){return x.getUTCMinutes()}),Nt=Vh.range,Wh=Da(function(x){x.setUTCMinutes(0,0,0)},function(x,y){x.setTime(+x+36E5*y)},function(x,y){return(y-x)/36E5},function(x){return x.getUTCHours()}),Ot=Wh.range,td=Da(function(x){x.setUTCHours(0,0,0,0)},function(x,y){x.setUTCDate(x.getUTCDate()+y)},function(x,y){return(y-x)/864E5},function(x){return x.getUTCDate()-1}),Pt=td.range,Ad=bc(0),sd=bc(1),Mm=bc(2),Nm=bc(3),Bd=bc(4),Om=bc(5),Pm=bc(6),Qm=Ad.range,Qt=sd.range,
  Rt=Mm.range,St=Nm.range,Tt=Bd.range,Ut=Om.range,Vt=Pm.range,Xh=Da(function(x){x.setUTCDate(1);x.setUTCHours(0,0,0,0)},function(x,y){x.setUTCMonth(x.getUTCMonth()+y)},function(x,y){return y.getUTCMonth()-x.getUTCMonth()+12*(y.getUTCFullYear()-x.getUTCFullYear())},function(x){return x.getUTCMonth()}),Wt=Xh.range,xb=Da(function(x){x.setUTCMonth(0,1);x.setUTCHours(0,0,0,0)},function(x,y){x.setUTCFullYear(x.getUTCFullYear()+y)},function(x,y){return y.getUTCFullYear()-x.getUTCFullYear()},function(x){return x.getUTCFullYear()});
  xb.every=function(x){return isFinite(x=Math.floor(x))&&0<x?Da(function(y){y.setUTCFullYear(Math.floor(y.getUTCFullYear()/x)*x);y.setUTCMonth(0,1);y.setUTCHours(0,0,0,0)},function(y,I){y.setUTCFullYear(y.getUTCFullYear()+I*x)}):null};var Xt=xb.range,ol={"-":"",_:" ",0:"0"},Ga=/^\s*\d+/,ks=/^%/,js=/[\\^$*+?|[\]().{}]/g,Cc;yl({dateTime:"%x, %X",date:"%-m/%-d/%Y",time:"%-I:%M:%S %p",periods:["AM","PM"],days:"Sunday Monday Tuesday Wednesday Thursday Friday Saturday".split(" "),shortDays:"Sun Mon Tue Wed Thu Fri Sat".split(" "),
  months:"January February March April May June July August September October November December".split(" "),shortMonths:"Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(" ")});var Yt=Date.prototype.toISOString?ls:d3.utcFormat("%Y-%m-%dT%H:%M:%S.%LZ"),Zt=+new Date("2000-01-01T00:00:00.000Z")?ms:d3.utcParse("%Y-%m-%dT%H:%M:%S.%LZ"),$t=ka("1f77b4ff7f0e2ca02cd627289467bd8c564be377c27f7f7fbcbd2217becf"),au=ka("393b795254a36b6ecf9c9ede6379398ca252b5cf6bcedb9c8c6d31bd9e39e7ba52e7cb94843c39ad494ad6616be7969c7b4173a55194ce6dbdde9ed6"),
  bu=ka("3182bd6baed69ecae1c6dbefe6550dfd8d3cfdae6bfdd0a231a35474c476a1d99bc7e9c0756bb19e9ac8bcbddcdadaeb636363969696bdbdbdd9d9d9"),cu=ka("1f77b4aec7e8ff7f0effbb782ca02c98df8ad62728ff98969467bdc5b0d58c564bc49c94e377c2f7b6d27f7f7fc7c7c7bcbd22dbdb8d17becf9edae5"),du=ka("7fc97fbeaed4fdc086ffff99386cb0f0027fbf5b17666666"),eu=ka("1b9e77d95f027570b3e7298a66a61ee6ab02a6761d666666"),fu=ka("a6cee31f78b4b2df8a33a02cfb9a99e31a1cfdbf6fff7f00cab2d66a3d9affff99b15928"),gu=ka("fbb4aeb3cde3ccebc5decbe4fed9a6ffffcce5d8bdfddaecf2f2f2"),
  hu=ka("b3e2cdfdcdaccbd5e8f4cae4e6f5c9fff2aef1e2cccccccc"),iu=ka("e41a1c377eb84daf4a984ea3ff7f00ffff33a65628f781bf999999"),ju=ka("66c2a5fc8d628da0cbe78ac3a6d854ffd92fe5c494b3b3b3"),ku=ka("8dd3c7ffffb3bebadafb807280b1d3fdb462b3de69fccde5d9d9d9bc80bdccebc5ffed6f"),Rm=Array(3).concat("d8b365f5f5f55ab4ac","a6611adfc27d80cdc1018571","a6611adfc27df5f5f580cdc1018571","8c510ad8b365f6e8c3c7eae55ab4ac01665e","8c510ad8b365f6e8c3f5f5f5c7eae55ab4ac01665e","8c510abf812ddfc27df6e8c3c7eae580cdc135978f01665e","8c510abf812ddfc27df6e8c3f5f5f5c7eae580cdc135978f01665e",
  "5430058c510abf812ddfc27df6e8c3c7eae580cdc135978f01665e003c30","5430058c510abf812ddfc27df6e8c3f5f5f5c7eae580cdc135978f01665e003c30").map(ka),lu=ua(Rm),Sm=Array(3).concat("af8dc3f7f7f77fbf7b","7b3294c2a5cfa6dba0008837","7b3294c2a5cff7f7f7a6dba0008837","762a83af8dc3e7d4e8d9f0d37fbf7b1b7837","762a83af8dc3e7d4e8f7f7f7d9f0d37fbf7b1b7837","762a839970abc2a5cfe7d4e8d9f0d3a6dba05aae611b7837","762a839970abc2a5cfe7d4e8f7f7f7d9f0d3a6dba05aae611b7837","40004b762a839970abc2a5cfe7d4e8d9f0d3a6dba05aae611b783700441b",
  "40004b762a839970abc2a5cfe7d4e8f7f7f7d9f0d3a6dba05aae611b783700441b").map(ka),mu=ua(Sm),Tm=Array(3).concat("e9a3c9f7f7f7a1d76a","d01c8bf1b6dab8e1864dac26","d01c8bf1b6daf7f7f7b8e1864dac26","c51b7de9a3c9fde0efe6f5d0a1d76a4d9221","c51b7de9a3c9fde0eff7f7f7e6f5d0a1d76a4d9221","c51b7dde77aef1b6dafde0efe6f5d0b8e1867fbc414d9221","c51b7dde77aef1b6dafde0eff7f7f7e6f5d0b8e1867fbc414d9221","8e0152c51b7dde77aef1b6dafde0efe6f5d0b8e1867fbc414d9221276419","8e0152c51b7dde77aef1b6dafde0eff7f7f7e6f5d0b8e1867fbc414d9221276419").map(ka),
  nu=ua(Tm),Um=Array(3).concat("998ec3f7f7f7f1a340","5e3c99b2abd2fdb863e66101","5e3c99b2abd2f7f7f7fdb863e66101","542788998ec3d8daebfee0b6f1a340b35806","542788998ec3d8daebf7f7f7fee0b6f1a340b35806","5427888073acb2abd2d8daebfee0b6fdb863e08214b35806","5427888073acb2abd2d8daebf7f7f7fee0b6fdb863e08214b35806","2d004b5427888073acb2abd2d8daebfee0b6fdb863e08214b358067f3b08","2d004b5427888073acb2abd2d8daebf7f7f7fee0b6fdb863e08214b358067f3b08").map(ka),ou=ua(Um),Vm=Array(3).concat("ef8a62f7f7f767a9cf","ca0020f4a58292c5de0571b0",
  "ca0020f4a582f7f7f792c5de0571b0","b2182bef8a62fddbc7d1e5f067a9cf2166ac","b2182bef8a62fddbc7f7f7f7d1e5f067a9cf2166ac","b2182bd6604df4a582fddbc7d1e5f092c5de4393c32166ac","b2182bd6604df4a582fddbc7f7f7f7d1e5f092c5de4393c32166ac","67001fb2182bd6604df4a582fddbc7d1e5f092c5de4393c32166ac053061","67001fb2182bd6604df4a582fddbc7f7f7f7d1e5f092c5de4393c32166ac053061").map(ka),pu=ua(Vm),Wm=Array(3).concat("ef8a62ffffff999999","ca0020f4a582bababa404040","ca0020f4a582ffffffbababa404040","b2182bef8a62fddbc7e0e0e09999994d4d4d",
  "b2182bef8a62fddbc7ffffffe0e0e09999994d4d4d","b2182bd6604df4a582fddbc7e0e0e0bababa8787874d4d4d","b2182bd6604df4a582fddbc7ffffffe0e0e0bababa8787874d4d4d","67001fb2182bd6604df4a582fddbc7e0e0e0bababa8787874d4d4d1a1a1a","67001fb2182bd6604df4a582fddbc7ffffffe0e0e0bababa8787874d4d4d1a1a1a").map(ka),qu=ua(Wm),Xm=Array(3).concat("fc8d59ffffbf91bfdb","d7191cfdae61abd9e92c7bb6","d7191cfdae61ffffbfabd9e92c7bb6","d73027fc8d59fee090e0f3f891bfdb4575b4","d73027fc8d59fee090ffffbfe0f3f891bfdb4575b4","d73027f46d43fdae61fee090e0f3f8abd9e974add14575b4",
  "d73027f46d43fdae61fee090ffffbfe0f3f8abd9e974add14575b4","a50026d73027f46d43fdae61fee090e0f3f8abd9e974add14575b4313695","a50026d73027f46d43fdae61fee090ffffbfe0f3f8abd9e974add14575b4313695").map(ka),ru=ua(Xm),Ym=Array(3).concat("fc8d59ffffbf91cf60","d7191cfdae61a6d96a1a9641","d7191cfdae61ffffbfa6d96a1a9641","d73027fc8d59fee08bd9ef8b91cf601a9850","d73027fc8d59fee08bffffbfd9ef8b91cf601a9850","d73027f46d43fdae61fee08bd9ef8ba6d96a66bd631a9850","d73027f46d43fdae61fee08bffffbfd9ef8ba6d96a66bd631a9850","a50026d73027f46d43fdae61fee08bd9ef8ba6d96a66bd631a9850006837",
  "a50026d73027f46d43fdae61fee08bffffbfd9ef8ba6d96a66bd631a9850006837").map(ka),su=ua(Ym),Zm=Array(3).concat("fc8d59ffffbf99d594","d7191cfdae61abdda42b83ba","d7191cfdae61ffffbfabdda42b83ba","d53e4ffc8d59fee08be6f59899d5943288bd","d53e4ffc8d59fee08bffffbfe6f59899d5943288bd","d53e4ff46d43fdae61fee08be6f598abdda466c2a53288bd","d53e4ff46d43fdae61fee08bffffbfe6f598abdda466c2a53288bd","9e0142d53e4ff46d43fdae61fee08be6f598abdda466c2a53288bd5e4fa2","9e0142d53e4ff46d43fdae61fee08bffffbfe6f598abdda466c2a53288bd5e4fa2").map(ka),
  tu=ua(Zm),$m=Array(3).concat("e5f5f999d8c92ca25f","edf8fbb2e2e266c2a4238b45","edf8fbb2e2e266c2a42ca25f006d2c","edf8fbccece699d8c966c2a42ca25f006d2c","edf8fbccece699d8c966c2a441ae76238b45005824","f7fcfde5f5f9ccece699d8c966c2a441ae76238b45005824","f7fcfde5f5f9ccece699d8c966c2a441ae76238b45006d2c00441b").map(ka),uu=ua($m),an=Array(3).concat("e0ecf49ebcda8856a7","edf8fbb3cde38c96c688419d","edf8fbb3cde38c96c68856a7810f7c","edf8fbbfd3e69ebcda8c96c68856a7810f7c","edf8fbbfd3e69ebcda8c96c68c6bb188419d6e016b",
  "f7fcfde0ecf4bfd3e69ebcda8c96c68c6bb188419d6e016b","f7fcfde0ecf4bfd3e69ebcda8c96c68c6bb188419d810f7c4d004b").map(ka),vu=ua(an),bn=Array(3).concat("e0f3dba8ddb543a2ca","f0f9e8bae4bc7bccc42b8cbe","f0f9e8bae4bc7bccc443a2ca0868ac","f0f9e8ccebc5a8ddb57bccc443a2ca0868ac","f0f9e8ccebc5a8ddb57bccc44eb3d32b8cbe08589e","f7fcf0e0f3dbccebc5a8ddb57bccc44eb3d32b8cbe08589e","f7fcf0e0f3dbccebc5a8ddb57bccc44eb3d32b8cbe0868ac084081").map(ka),wu=ua(bn),cn=Array(3).concat("fee8c8fdbb84e34a33","fef0d9fdcc8afc8d59d7301f",
  "fef0d9fdcc8afc8d59e34a33b30000","fef0d9fdd49efdbb84fc8d59e34a33b30000","fef0d9fdd49efdbb84fc8d59ef6548d7301f990000","fff7ecfee8c8fdd49efdbb84fc8d59ef6548d7301f990000","fff7ecfee8c8fdd49efdbb84fc8d59ef6548d7301fb300007f0000").map(ka),xu=ua(cn),dn=Array(3).concat("ece2f0a6bddb1c9099","f6eff7bdc9e167a9cf02818a","f6eff7bdc9e167a9cf1c9099016c59","f6eff7d0d1e6a6bddb67a9cf1c9099016c59","f6eff7d0d1e6a6bddb67a9cf3690c002818a016450","fff7fbece2f0d0d1e6a6bddb67a9cf3690c002818a016450","fff7fbece2f0d0d1e6a6bddb67a9cf3690c002818a016c59014636").map(ka),
  yu=ua(dn),en=Array(3).concat("ece7f2a6bddb2b8cbe","f1eef6bdc9e174a9cf0570b0","f1eef6bdc9e174a9cf2b8cbe045a8d","f1eef6d0d1e6a6bddb74a9cf2b8cbe045a8d","f1eef6d0d1e6a6bddb74a9cf3690c00570b0034e7b","fff7fbece7f2d0d1e6a6bddb74a9cf3690c00570b0034e7b","fff7fbece7f2d0d1e6a6bddb74a9cf3690c00570b0045a8d023858").map(ka),zu=ua(en),fn=Array(3).concat("e7e1efc994c7dd1c77","f1eef6d7b5d8df65b0ce1256","f1eef6d7b5d8df65b0dd1c77980043","f1eef6d4b9dac994c7df65b0dd1c77980043","f1eef6d4b9dac994c7df65b0e7298ace125691003f",
  "f7f4f9e7e1efd4b9dac994c7df65b0e7298ace125691003f","f7f4f9e7e1efd4b9dac994c7df65b0e7298ace125698004367001f").map(ka),Au=ua(fn),gn=Array(3).concat("fde0ddfa9fb5c51b8a","feebe2fbb4b9f768a1ae017e","feebe2fbb4b9f768a1c51b8a7a0177","feebe2fcc5c0fa9fb5f768a1c51b8a7a0177","feebe2fcc5c0fa9fb5f768a1dd3497ae017e7a0177","fff7f3fde0ddfcc5c0fa9fb5f768a1dd3497ae017e7a0177","fff7f3fde0ddfcc5c0fa9fb5f768a1dd3497ae017e7a017749006a").map(ka),Bu=ua(gn),hn=Array(3).concat("edf8b17fcdbb2c7fb8","ffffcca1dab441b6c4225ea8",
  "ffffcca1dab441b6c42c7fb8253494","ffffccc7e9b47fcdbb41b6c42c7fb8253494","ffffccc7e9b47fcdbb41b6c41d91c0225ea80c2c84","ffffd9edf8b1c7e9b47fcdbb41b6c41d91c0225ea80c2c84","ffffd9edf8b1c7e9b47fcdbb41b6c41d91c0225ea8253494081d58").map(ka),Cu=ua(hn),jn=Array(3).concat("f7fcb9addd8e31a354","ffffccc2e69978c679238443","ffffccc2e69978c67931a354006837","ffffccd9f0a3addd8e78c67931a354006837","ffffccd9f0a3addd8e78c67941ab5d238443005a32","ffffe5f7fcb9d9f0a3addd8e78c67941ab5d238443005a32","ffffe5f7fcb9d9f0a3addd8e78c67941ab5d238443006837004529").map(ka),
  Du=ua(jn),kn=Array(3).concat("fff7bcfec44fd95f0e","ffffd4fed98efe9929cc4c02","ffffd4fed98efe9929d95f0e993404","ffffd4fee391fec44ffe9929d95f0e993404","ffffd4fee391fec44ffe9929ec7014cc4c028c2d04","ffffe5fff7bcfee391fec44ffe9929ec7014cc4c028c2d04","ffffe5fff7bcfee391fec44ffe9929ec7014cc4c02993404662506").map(ka),Eu=ua(kn),ln=Array(3).concat("ffeda0feb24cf03b20","ffffb2fecc5cfd8d3ce31a1c","ffffb2fecc5cfd8d3cf03b20bd0026","ffffb2fed976feb24cfd8d3cf03b20bd0026","ffffb2fed976feb24cfd8d3cfc4e2ae31a1cb10026",
  "ffffccffeda0fed976feb24cfd8d3cfc4e2ae31a1cb10026","ffffccffeda0fed976feb24cfd8d3cfc4e2ae31a1cbd0026800026").map(ka),Fu=ua(ln),mn=Array(3).concat("deebf79ecae13182bd","eff3ffbdd7e76baed62171b5","eff3ffbdd7e76baed63182bd08519c","eff3ffc6dbef9ecae16baed63182bd08519c","eff3ffc6dbef9ecae16baed64292c62171b5084594","f7fbffdeebf7c6dbef9ecae16baed64292c62171b5084594","f7fbffdeebf7c6dbef9ecae16baed64292c62171b508519c08306b").map(ka),Gu=ua(mn),nn=Array(3).concat("e5f5e0a1d99b31a354","edf8e9bae4b374c476238b45",
  "edf8e9bae4b374c47631a354006d2c","edf8e9c7e9c0a1d99b74c47631a354006d2c","edf8e9c7e9c0a1d99b74c47641ab5d238b45005a32","f7fcf5e5f5e0c7e9c0a1d99b74c47641ab5d238b45005a32","f7fcf5e5f5e0c7e9c0a1d99b74c47641ab5d238b45006d2c00441b").map(ka),Hu=ua(nn),on=Array(3).concat("f0f0f0bdbdbd636363","f7f7f7cccccc969696525252","f7f7f7cccccc969696636363252525","f7f7f7d9d9d9bdbdbd969696636363252525","f7f7f7d9d9d9bdbdbd969696737373525252252525","fffffff0f0f0d9d9d9bdbdbd969696737373525252252525","fffffff0f0f0d9d9d9bdbdbd969696737373525252252525000000").map(ka),
  Iu=ua(on),pn=Array(3).concat("efedf5bcbddc756bb1","f2f0f7cbc9e29e9ac86a51a3","f2f0f7cbc9e29e9ac8756bb154278f","f2f0f7dadaebbcbddc9e9ac8756bb154278f","f2f0f7dadaebbcbddc9e9ac8807dba6a51a34a1486","fcfbfdefedf5dadaebbcbddc9e9ac8807dba6a51a34a1486","fcfbfdefedf5dadaebbcbddc9e9ac8807dba6a51a354278f3f007d").map(ka),Ju=ua(pn),qn=Array(3).concat("fee0d2fc9272de2d26","fee5d9fcae91fb6a4acb181d","fee5d9fcae91fb6a4ade2d26a50f15","fee5d9fcbba1fc9272fb6a4ade2d26a50f15","fee5d9fcbba1fc9272fb6a4aef3b2ccb181d99000d",
  "fff5f0fee0d2fcbba1fc9272fb6a4aef3b2ccb181d99000d","fff5f0fee0d2fcbba1fc9272fb6a4aef3b2ccb181da50f1567000d").map(ka),Ku=ua(qn),rn=Array(3).concat("fee6cefdae6be6550d","feeddefdbe85fd8d3cd94701","feeddefdbe85fd8d3ce6550da63603","feeddefdd0a2fdae6bfd8d3ce6550da63603","feeddefdd0a2fdae6bfd8d3cf16913d948018c2d04","fff5ebfee6cefdd0a2fdae6bfd8d3cf16913d948018c2d04","fff5ebfee6cefdd0a2fdae6bfd8d3cf16913d94801a636037f2704").map(ka),Lu=ua(rn),Mu=rf(db(300,.5,0),db(-240,.5,1)),Nu=rf(db(-100,.75,.35),db(80,
  1.5,.8)),Ou=rf(db(260,.75,.35),db(80,1.5,.8)),zf=db(),Af=hc(),Pu=Math.PI/3,Qu=2*Math.PI/3,Ru=bf(ka("44015444025645045745055946075a46085c460a5d460b5e470d60470e6147106347116447136548146748166848176948186a481a6c481b6d481c6e481d6f481f70482071482173482374482475482576482677482878482979472a7a472c7a472d7b472e7c472f7d46307e46327e46337f463480453581453781453882443983443a83443b84433d84433e85423f854240864241864142874144874045884046883f47883f48893e49893e4a893e4c8a3d4d8a3d4e8a3c4f8a3c508b3b518b3b528b3a538b3a548c39558c39568c38588c38598c375a8c375b8d365c8d365d8d355e8d355f8d34608d34618d33628d33638d32648e32658e31668e31678e31688e30698e306a8e2f6b8e2f6c8e2e6d8e2e6e8e2e6f8e2d708e2d718e2c718e2c728e2c738e2b748e2b758e2a768e2a778e2a788e29798e297a8e297b8e287c8e287d8e277e8e277f8e27808e26818e26828e26828e25838e25848e25858e24868e24878e23888e23898e238a8d228b8d228c8d228d8d218e8d218f8d21908d21918c20928c20928c20938c1f948c1f958b1f968b1f978b1f988b1f998a1f9a8a1e9b8a1e9c891e9d891f9e891f9f881fa0881fa1881fa1871fa28720a38620a48621a58521a68522a78522a88423a98324aa8325ab8225ac8226ad8127ad8128ae8029af7f2ab07f2cb17e2db27d2eb37c2fb47c31b57b32b67a34b67935b77937b87838b9773aba763bbb753dbc743fbc7340bd7242be7144bf7046c06f48c16e4ac16d4cc26c4ec36b50c46a52c56954c56856c66758c7655ac8645cc8635ec96260ca6063cb5f65cb5e67cc5c69cd5b6ccd5a6ece5870cf5773d05675d05477d1537ad1517cd2507fd34e81d34d84d44b86d54989d5488bd6468ed64590d74393d74195d84098d83e9bd93c9dd93ba0da39a2da37a5db36a8db34aadc32addc30b0dd2fb2dd2db5de2bb8de29bade28bddf26c0df25c2df23c5e021c8e020cae11fcde11dd0e11cd2e21bd5e21ad8e219dae319dde318dfe318e2e418e5e419e7e419eae51aece51befe51cf1e51df4e61ef6e620f8e621fbe723fde725")),
  Su=bf(ka("00000401000501010601010802010902020b02020d03030f03031204041405041606051806051a07061c08071e0907200a08220b09240c09260d0a290e0b2b100b2d110c2f120d31130d34140e36150e38160f3b180f3d19103f1a10421c10441d11471e114920114b21114e22115024125325125527125829115a2a115c2c115f2d11612f116331116533106734106936106b38106c390f6e3b0f703d0f713f0f72400f74420f75440f764510774710784910784a10794c117a4e117b4f127b51127c52137c54137d56147d57157e59157e5a167e5c167f5d177f5f187f601880621980641a80651a80671b80681c816a1c816b1d816d1d816e1e81701f81721f817320817521817621817822817922827b23827c23827e24828025828125818326818426818627818827818928818b29818c29818e2a81902a81912b81932b80942c80962c80982d80992d809b2e7f9c2e7f9e2f7fa02f7fa1307ea3307ea5317ea6317da8327daa337dab337cad347cae347bb0357bb2357bb3367ab5367ab73779b83779ba3878bc3978bd3977bf3a77c03a76c23b75c43c75c53c74c73d73c83e73ca3e72cc3f71cd4071cf4070d0416fd2426fd3436ed5446dd6456cd8456cd9466bdb476adc4869de4968df4a68e04c67e24d66e34e65e44f64e55064e75263e85362e95462ea5661eb5760ec5860ed5a5fee5b5eef5d5ef05f5ef1605df2625df2645cf3655cf4675cf4695cf56b5cf66c5cf66e5cf7705cf7725cf8745cf8765cf9785df9795df97b5dfa7d5efa7f5efa815ffb835ffb8560fb8761fc8961fc8a62fc8c63fc8e64fc9065fd9266fd9467fd9668fd9869fd9a6afd9b6bfe9d6cfe9f6dfea16efea36ffea571fea772fea973feaa74feac76feae77feb078feb27afeb47bfeb67cfeb77efeb97ffebb81febd82febf84fec185fec287fec488fec68afec88cfeca8dfecc8ffecd90fecf92fed194fed395fed597fed799fed89afdda9cfddc9efddea0fde0a1fde2a3fde3a5fde5a7fde7a9fde9aafdebacfcecaefceeb0fcf0b2fcf2b4fcf4b6fcf6b8fcf7b9fcf9bbfcfbbdfcfdbf")),
  Tu=bf(ka("00000401000501010601010802010a02020c02020e03021004031204031405041706041907051b08051d09061f0a07220b07240c08260d08290e092b10092d110a30120a32140b34150b37160b39180c3c190c3e1b0c411c0c431e0c451f0c48210c4a230c4c240c4f260c51280b53290b552b0b572d0b592f0a5b310a5c320a5e340a5f3609613809623909633b09643d09653e0966400a67420a68440a68450a69470b6a490b6a4a0c6b4c0c6b4d0d6c4f0d6c510e6c520e6d540f6d550f6d57106e59106e5a116e5c126e5d126e5f136e61136e62146e64156e65156e67166e69166e6a176e6c186e6d186e6f196e71196e721a6e741a6e751b6e771c6d781c6d7a1d6d7c1d6d7d1e6d7f1e6c801f6c82206c84206b85216b87216b88226a8a226a8c23698d23698f24699025689225689326679526679727669827669a28659b29649d29649f2a63a02a63a22b62a32c61a52c60a62d60a82e5fa92e5eab2f5ead305dae305cb0315bb1325ab3325ab43359b63458b73557b93556ba3655bc3754bd3853bf3952c03a51c13a50c33b4fc43c4ec63d4dc73e4cc83f4bca404acb4149cc4248ce4347cf4446d04545d24644d34743d44842d54a41d74b3fd84c3ed94d3dda4e3cdb503bdd513ade5238df5337e05536e15635e25734e35933e45a31e55c30e65d2fe75e2ee8602de9612bea632aeb6429eb6628ec6726ed6925ee6a24ef6c23ef6e21f06f20f1711ff1731df2741cf3761bf37819f47918f57b17f57d15f67e14f68013f78212f78410f8850ff8870ef8890cf98b0bf98c0af98e09fa9008fa9207fa9407fb9606fb9706fb9906fb9b06fb9d07fc9f07fca108fca309fca50afca60cfca80dfcaa0ffcac11fcae12fcb014fcb216fcb418fbb61afbb81dfbba1ffbbc21fbbe23fac026fac228fac42afac62df9c72ff9c932f9cb35f8cd37f8cf3af7d13df7d340f6d543f6d746f5d949f5db4cf4dd4ff4df53f4e156f3e35af3e55df2e661f2e865f2ea69f1ec6df1ed71f1ef75f1f179f2f27df2f482f3f586f3f68af4f88ef5f992f6fa96f8fb9af9fc9dfafda1fcffa4")),
  Uu=bf(ka("0d088710078813078916078a19068c1b068d1d068e20068f2206902406912605912805922a05932c05942e05952f059631059733059735049837049938049a3a049a3c049b3e049c3f049c41049d43039e44039e46039f48039f4903a04b03a14c02a14e02a25002a25102a35302a35502a45601a45801a45901a55b01a55c01a65e01a66001a66100a76300a76400a76600a76700a86900a86a00a86c00a86e00a86f00a87100a87201a87401a87501a87701a87801a87a02a87b02a87d03a87e03a88004a88104a78305a78405a78606a68707a68808a68a09a58b0aa58d0ba58e0ca48f0da4910ea3920fa39410a29511a19613a19814a099159f9a169f9c179e9d189d9e199da01a9ca11b9ba21d9aa31e9aa51f99a62098a72197a82296aa2395ab2494ac2694ad2793ae2892b02991b12a90b22b8fb32c8eb42e8db52f8cb6308bb7318ab83289ba3388bb3488bc3587bd3786be3885bf3984c03a83c13b82c23c81c33d80c43e7fc5407ec6417dc7427cc8437bc9447aca457acb4679cc4778cc4977cd4a76ce4b75cf4c74d04d73d14e72d24f71d35171d45270d5536fd5546ed6556dd7566cd8576bd9586ada5a6ada5b69db5c68dc5d67dd5e66de5f65de6164df6263e06363e16462e26561e26660e3685fe4695ee56a5de56b5de66c5ce76e5be76f5ae87059e97158e97257ea7457eb7556eb7655ec7754ed7953ed7a52ee7b51ef7c51ef7e50f07f4ff0804ef1814df1834cf2844bf3854bf3874af48849f48948f58b47f58c46f68d45f68f44f79044f79143f79342f89441f89540f9973ff9983ef99a3efa9b3dfa9c3cfa9e3bfb9f3afba139fba238fca338fca537fca636fca835fca934fdab33fdac33fdae32fdaf31fdb130fdb22ffdb42ffdb52efeb72dfeb82cfeba2cfebb2bfebd2afebe2afec029fdc229fdc328fdc527fdc627fdc827fdca26fdcb26fccd25fcce25fcd025fcd225fbd324fbd524fbd724fad824fada24f9dc24f9dd25f8df25f8e125f7e225f7e425f6e626f6e826f5e926f5eb27f4ed27f3ee27f3f027f2f227f1f426f1f525f0f724f0f921")),
  sn=Math.abs,Ia=Math.atan2,ec=Math.cos,us=Math.max,Yh=Math.min,pb=Math.sin,Dc=Math.sqrt,Kb=Math.PI,cf=Kb/2,Lb=2*Kb;Dl.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._point=0},lineEnd:function(){(this._line||0!==this._line&&1===this._point)&&this._context.closePath();this._line=1-this._line},point:function(x,y){x=+x;y=+y;switch(this._point){case 0:this._point=1;this._line?this._context.lineTo(x,y):this._context.moveTo(x,y);break;case 1:this._point=
  2;default:this._context.lineTo(x,y)}}};var Hl=xh(ef);Fl.prototype={areaStart:function(){this._curve.areaStart()},areaEnd:function(){this._curve.areaEnd()},lineStart:function(){this._curve.lineStart()},lineEnd:function(){this._curve.lineEnd()},point:function(x,y){this._curve.point(y*Math.sin(x),y*-Math.cos(x))}};var zh=Array.prototype.slice,Zh={draw:function(x,y){y=Math.sqrt(y/Kb);x.moveTo(y,0);x.arc(0,0,y,0,Lb)}},tn={draw:function(x,y){y=Math.sqrt(y/5)/2;x.moveTo(-3*y,-y);x.lineTo(-y,-y);x.lineTo(-y,
  -3*y);x.lineTo(y,-3*y);x.lineTo(y,-y);x.lineTo(3*y,-y);x.lineTo(3*y,y);x.lineTo(y,y);x.lineTo(y,3*y);x.lineTo(-y,3*y);x.lineTo(-y,y);x.lineTo(-3*y,y);x.closePath()}},un=Math.sqrt(1/3),Vu=2*un,vn={draw:function(x,y){y=Math.sqrt(y/Vu);var I=y*un;x.moveTo(0,-y);x.lineTo(I,0);x.lineTo(0,y);x.lineTo(-I,0);x.closePath()}},wn=Math.sin(Kb/10)/Math.sin(7*Kb/10),Wu=Math.sin(Lb/10)*wn,Xu=-Math.cos(Lb/10)*wn,xn={draw:function(x,y){y=Math.sqrt(.8908130915292852*y);var I=Wu*y,Q=Xu*y;x.moveTo(0,-y);x.lineTo(I,Q);
  for(var V=1;5>V;++V){var N=Lb*V/5,T=Math.cos(N);N=Math.sin(N);x.lineTo(N*y,-T*y);x.lineTo(T*I-N*Q,N*I+T*Q)}x.closePath()}},yn={draw:function(x,y){y=Math.sqrt(y);var I=-y/2;x.rect(I,I,y,y)}},$h=Math.sqrt(3),zn={draw:function(x,y){y=-Math.sqrt(y/(3*$h));x.moveTo(0,2*y);x.lineTo(-$h*y,-y);x.lineTo($h*y,-y);x.closePath()}},ab=Math.sqrt(3)/2,ai=1/Math.sqrt(12),Yu=3*(ai/2+1),An={draw:function(x,y){var I=Math.sqrt(y/Yu);y=I/2;var Q=I*ai;I=I*ai+I;var V=-y;x.moveTo(y,Q);x.lineTo(y,I);x.lineTo(V,I);x.lineTo(-.5*
  y-ab*Q,ab*y+-.5*Q);x.lineTo(-.5*y-ab*I,ab*y+-.5*I);x.lineTo(-.5*V-ab*I,ab*V+-.5*I);x.lineTo(-.5*y+ab*Q,-.5*Q-ab*y);x.lineTo(-.5*y+ab*I,-.5*I-ab*y);x.lineTo(-.5*V+ab*I,-.5*I-ab*V);x.closePath()}},Zu=[Zh,tn,vn,yn,xn,zn,An];gf.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x0=this._x1=this._y0=this._y1=NaN;this._point=0},lineEnd:function(){switch(this._point){case 3:ff(this,this._x1,this._y1);case 2:this._context.lineTo(this._x1,this._y1)}(this._line||
  0!==this._line&&1===this._point)&&this._context.closePath();this._line=1-this._line},point:function(x,y){x=+x;y=+y;switch(this._point){case 0:this._point=1;this._line?this._context.lineTo(x,y):this._context.moveTo(x,y);break;case 1:this._point=2;break;case 2:this._point=3,this._context.lineTo((5*this._x0+this._x1)/6,(5*this._y0+this._y1)/6);default:ff(this,x,y)}this._x0=this._x1;this._x1=x;this._y0=this._y1;this._y1=y}};Jl.prototype={areaStart:Jb,areaEnd:Jb,lineStart:function(){this._x0=this._x1=
  this._x2=this._x3=this._x4=this._y0=this._y1=this._y2=this._y3=this._y4=NaN;this._point=0},lineEnd:function(){switch(this._point){case 1:this._context.moveTo(this._x2,this._y2);this._context.closePath();break;case 2:this._context.moveTo((this._x2+2*this._x3)/3,(this._y2+2*this._y3)/3);this._context.lineTo((this._x3+2*this._x2)/3,(this._y3+2*this._y2)/3);this._context.closePath();break;case 3:this.point(this._x2,this._y2),this.point(this._x3,this._y3),this.point(this._x4,this._y4)}},point:function(x,
  y){x=+x;y=+y;switch(this._point){case 0:this._point=1;this._x2=x;this._y2=y;break;case 1:this._point=2;this._x3=x;this._y3=y;break;case 2:this._point=3;this._x4=x;this._y4=y;this._context.moveTo((this._x0+4*this._x1+x)/6,(this._y0+4*this._y1+y)/6);break;default:ff(this,x,y)}this._x0=this._x1;this._x1=x;this._y0=this._y1;this._y1=y}};Kl.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x0=this._x1=this._y0=this._y1=NaN;this._point=0},lineEnd:function(){(this._line||
  0!==this._line&&3===this._point)&&this._context.closePath();this._line=1-this._line},point:function(x,y){x=+x;y=+y;switch(this._point){case 0:this._point=1;break;case 1:this._point=2;break;case 2:this._point=3;var I=(this._x0+4*this._x1+x)/6,Q=(this._y0+4*this._y1+y)/6;this._line?this._context.lineTo(I,Q):this._context.moveTo(I,Q);break;case 3:this._point=4;default:ff(this,x,y)}this._x0=this._x1;this._x1=x;this._y0=this._y1;this._y1=y}};Ll.prototype={lineStart:function(){this._x=[];this._y=[];this._basis.lineStart()},
  lineEnd:function(){var x=this._x,y=this._y,I=x.length-1;if(0<I)for(var Q=x[0],V=y[0],N=x[I]-Q,T=y[I]-V,f=-1,n;++f<=I;)n=f/I,this._basis.point(this._beta*x[f]+(1-this._beta)*(Q+n*N),this._beta*y[f]+(1-this._beta)*(V+n*T));this._x=this._y=null;this._basis.lineEnd()},point:function(x,y){this._x.push(+x);this._y.push(+y)}};var $u=function I(y){function Q(V){return 1===y?new gf(V):new Ll(V,y)}Q.beta=function(V){return I(+V)};return Q}(.85);Ah.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=
  NaN},lineStart:function(){this._x0=this._x1=this._x2=this._y0=this._y1=this._y2=NaN;this._point=0},lineEnd:function(){switch(this._point){case 2:this._context.lineTo(this._x2,this._y2);break;case 3:hf(this,this._x1,this._y1)}(this._line||0!==this._line&&1===this._point)&&this._context.closePath();this._line=1-this._line},point:function(y,I){y=+y;I=+I;switch(this._point){case 0:this._point=1;this._line?this._context.lineTo(y,I):this._context.moveTo(y,I);break;case 1:this._point=2;this._x1=y;this._y1=
  I;break;case 2:this._point=3;default:hf(this,y,I)}this._x0=this._x1;this._x1=this._x2;this._x2=y;this._y0=this._y1;this._y1=this._y2;this._y2=I}};var av=function Q(I){function V(N){return new Ah(N,I)}V.tension=function(N){return Q(+N)};return V}(0);Bh.prototype={areaStart:Jb,areaEnd:Jb,lineStart:function(){this._x0=this._x1=this._x2=this._x3=this._x4=this._x5=this._y0=this._y1=this._y2=this._y3=this._y4=this._y5=NaN;this._point=0},lineEnd:function(){switch(this._point){case 1:this._context.moveTo(this._x3,
  this._y3);this._context.closePath();break;case 2:this._context.lineTo(this._x3,this._y3);this._context.closePath();break;case 3:this.point(this._x3,this._y3),this.point(this._x4,this._y4),this.point(this._x5,this._y5)}},point:function(I,Q){I=+I;Q=+Q;switch(this._point){case 0:this._point=1;this._x3=I;this._y3=Q;break;case 1:this._point=2;this._context.moveTo(this._x4=I,this._y4=Q);break;case 2:this._point=3;this._x5=I;this._y5=Q;break;default:hf(this,I,Q)}this._x0=this._x1;this._x1=this._x2;this._x2=
  I;this._y0=this._y1;this._y1=this._y2;this._y2=Q}};var bv=function V(Q){function N(T){return new Bh(T,Q)}N.tension=function(T){return V(+T)};return N}(0);Ch.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x0=this._x1=this._x2=this._y0=this._y1=this._y2=NaN;this._point=0},lineEnd:function(){(this._line||0!==this._line&&3===this._point)&&this._context.closePath();this._line=1-this._line},point:function(Q,V){Q=+Q;V=+V;switch(this._point){case 0:this._point=
  1;break;case 1:this._point=2;break;case 2:this._point=3;this._line?this._context.lineTo(this._x2,this._y2):this._context.moveTo(this._x2,this._y2);break;case 3:this._point=4;default:hf(this,Q,V)}this._x0=this._x1;this._x1=this._x2;this._x2=Q;this._y0=this._y1;this._y1=this._y2;this._y2=V}};var cv=function N(V){function T(f){return new Ch(f,V)}T.tension=function(f){return N(+f)};return T}(0);Ml.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x0=
  this._x1=this._x2=this._y0=this._y1=this._y2=NaN;this._l01_a=this._l12_a=this._l23_a=this._l01_2a=this._l12_2a=this._l23_2a=this._point=0},lineEnd:function(){switch(this._point){case 2:this._context.lineTo(this._x2,this._y2);break;case 3:this.point(this._x2,this._y2)}(this._line||0!==this._line&&1===this._point)&&this._context.closePath();this._line=1-this._line},point:function(V,N){V=+V;N=+N;if(this._point){var T=this._x2-V,f=this._y2-N;this._l23_a=Math.sqrt(this._l23_2a=Math.pow(T*T+f*f,this._alpha))}switch(this._point){case 0:this._point=
  1;this._line?this._context.lineTo(V,N):this._context.moveTo(V,N);break;case 1:this._point=2;break;case 2:this._point=3;default:Dh(this,V,N)}this._l01_a=this._l12_a;this._l12_a=this._l23_a;this._l01_2a=this._l12_2a;this._l12_2a=this._l23_2a;this._x0=this._x1;this._x1=this._x2;this._x2=V;this._y0=this._y1;this._y1=this._y2;this._y2=N}};var dv=function T(N){function f(n){return N?new Ml(n,N):new Ah(n,0)}f.alpha=function(n){return T(+n)};return f}(.5);Nl.prototype={areaStart:Jb,areaEnd:Jb,lineStart:function(){this._x0=
  this._x1=this._x2=this._x3=this._x4=this._x5=this._y0=this._y1=this._y2=this._y3=this._y4=this._y5=NaN;this._l01_a=this._l12_a=this._l23_a=this._l01_2a=this._l12_2a=this._l23_2a=this._point=0},lineEnd:function(){switch(this._point){case 1:this._context.moveTo(this._x3,this._y3);this._context.closePath();break;case 2:this._context.lineTo(this._x3,this._y3);this._context.closePath();break;case 3:this.point(this._x3,this._y3),this.point(this._x4,this._y4),this.point(this._x5,this._y5)}},point:function(N,
  T){N=+N;T=+T;if(this._point){var f=this._x2-N,n=this._y2-T;this._l23_a=Math.sqrt(this._l23_2a=Math.pow(f*f+n*n,this._alpha))}switch(this._point){case 0:this._point=1;this._x3=N;this._y3=T;break;case 1:this._point=2;this._context.moveTo(this._x4=N,this._y4=T);break;case 2:this._point=3;this._x5=N;this._y5=T;break;default:Dh(this,N,T)}this._l01_a=this._l12_a;this._l12_a=this._l23_a;this._l01_2a=this._l12_2a;this._l12_2a=this._l23_2a;this._x0=this._x1;this._x1=this._x2;this._x2=N;this._y0=this._y1;this._y1=
  this._y2;this._y2=T}};var ev=function f(T){function n(u){return T?new Nl(u,T):new Bh(u,0)}n.alpha=function(u){return f(+u)};return n}(.5);Ol.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x0=this._x1=this._x2=this._y0=this._y1=this._y2=NaN;this._l01_a=this._l12_a=this._l23_a=this._l01_2a=this._l12_2a=this._l23_2a=this._point=0},lineEnd:function(){(this._line||0!==this._line&&3===this._point)&&this._context.closePath();this._line=1-this._line},
  point:function(T,f){T=+T;f=+f;if(this._point){var n=this._x2-T,u=this._y2-f;this._l23_a=Math.sqrt(this._l23_2a=Math.pow(n*n+u*u,this._alpha))}switch(this._point){case 0:this._point=1;break;case 1:this._point=2;break;case 2:this._point=3;this._line?this._context.lineTo(this._x2,this._y2):this._context.moveTo(this._x2,this._y2);break;case 3:this._point=4;default:Dh(this,T,f)}this._l01_a=this._l12_a;this._l12_a=this._l23_a;this._l01_2a=this._l12_2a;this._l12_2a=this._l23_2a;this._x0=this._x1;this._x1=
  this._x2;this._x2=T;this._y0=this._y1;this._y1=this._y2;this._y2=f}};var fv=function n(f){function u(r){return f?new Ol(r,f):new Ch(r,0)}u.alpha=function(r){return n(+r)};return u}(.5);Pl.prototype={areaStart:Jb,areaEnd:Jb,lineStart:function(){this._point=0},lineEnd:function(){this._point&&this._context.closePath()},point:function(f,n){f=+f;n=+n;this._point?this._context.lineTo(f,n):(this._point=1,this._context.moveTo(f,n))}};jf.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=
  NaN},lineStart:function(){this._x0=this._x1=this._y0=this._y1=this._t0=NaN;this._point=0},lineEnd:function(){switch(this._point){case 2:this._context.lineTo(this._x1,this._y1);break;case 3:Eh(this,this._t0,Rl(this,this._t0))}(this._line||0!==this._line&&1===this._point)&&this._context.closePath();this._line=1-this._line},point:function(f,n){var u=NaN;f=+f;n=+n;if(f!==this._x1||n!==this._y1){switch(this._point){case 0:this._point=1;this._line?this._context.lineTo(f,n):this._context.moveTo(f,n);break;
  case 1:this._point=2;break;case 2:this._point=3;Eh(this,Rl(this,u=Ql(this,f,n)),u);break;default:Eh(this,this._t0,u=Ql(this,f,n))}this._x0=this._x1;this._x1=f;this._y0=this._y1;this._y1=n;this._t0=u}}};(Sl.prototype=Object.create(jf.prototype)).point=function(f,n){jf.prototype.point.call(this,n,f)};Tl.prototype={moveTo:function(f,n){this._context.moveTo(n,f)},closePath:function(){this._context.closePath()},lineTo:function(f,n){this._context.lineTo(n,f)},bezierCurveTo:function(f,n,u,r,t,z){this._context.bezierCurveTo(n,
  f,r,u,z,t)}};Ul.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x=[];this._y=[]},lineEnd:function(){var f=this._x,n=this._y,u=f.length;if(u)if(this._line?this._context.lineTo(f[0],n[0]):this._context.moveTo(f[0],n[0]),2===u)this._context.lineTo(f[1],n[1]);else for(var r=Vl(f),t=Vl(n),z=0,D=1;D<u;++z,++D)this._context.bezierCurveTo(r[0][z],t[0][z],r[1][z],t[1][z],f[D],n[D]);(this._line||0!==this._line&&1===u)&&this._context.closePath();this._line=
  1-this._line;this._x=this._y=null},point:function(f,n){this._x.push(+f);this._y.push(+n)}};kf.prototype={areaStart:function(){this._line=0},areaEnd:function(){this._line=NaN},lineStart:function(){this._x=this._y=NaN;this._point=0},lineEnd:function(){0<this._t&&1>this._t&&2===this._point&&this._context.lineTo(this._x,this._y);(this._line||0!==this._line&&1===this._point)&&this._context.closePath();0<=this._line&&(this._t=1-this._t,this._line=1-this._line)},point:function(f,n){f=+f;n=+n;switch(this._point){case 0:this._point=
  1;this._line?this._context.lineTo(f,n):this._context.moveTo(f,n);break;case 1:this._point=2;default:if(0>=this._t)this._context.lineTo(this._x,n),this._context.lineTo(f,n);else{var u=this._x*(1-this._t)+f*this._t;this._context.lineTo(u,this._y);this._context.lineTo(u,n)}}this._x=f;this._y=n}};lf.prototype={constructor:lf,insert:function(f,n){var u;if(f){n.P=f;if(n.N=f.N)f.N.P=n;f.N=n;if(f.R){for(f=f.R;f.L;)f=f.L;f.L=n}else f.R=n;var r=f}else this._?(f=Zl(this._),n.P=null,n.N=f,f.P=f.L=n,r=f):(n.P=
  n.N=null,this._=n,r=null);n.L=n.R=null;n.U=r;n.C=!0;for(f=n;r&&r.C;)n=r.U,r===n.L?(u=n.R)&&u.C?(r.C=u.C=!1,n.C=!0,f=n):(f===r.R&&(Ed(this,r),f=r,r=f.U),r.C=!1,n.C=!0,Fd(this,n)):(u=n.L)&&u.C?(r.C=u.C=!1,n.C=!0,f=n):(f===r.L&&(Fd(this,r),f=r,r=f.U),r.C=!1,n.C=!0,Ed(this,n)),r=f.U;this._.C=!1},remove:function(f){f.N&&(f.N.P=f.P);f.P&&(f.P.N=f.N);f.N=f.P=null;var n=f.U,u=f.L,r=f.R;var t=u?r?Zl(r):u:r;n?n.L===f?n.L=t:n.R=t:this._=t;if(u&&r){var z=t.C;t.C=f.C;t.L=u;u.U=t;t!==r?(n=t.U,t.U=f.U,f=t.R,n.L=
  f,t.R=r,r.U=t):(t.U=n,n=t,f=t.R)}else z=f.C,f=t;f&&(f.U=n);if(!z)if(f&&f.C)f.C=!1;else{do{if(f===this._)break;if(f===n.L){if(f=n.R,f.C&&(f.C=!1,n.C=!0,Ed(this,n),f=n.R),f.L&&f.L.C||f.R&&f.R.C){f.R&&f.R.C||(f.L.C=!1,f.C=!0,Fd(this,f),f=n.R);f.C=n.C;n.C=f.R.C=!1;Ed(this,n);f=this._;break}}else if(f=n.L,f.C&&(f.C=!1,n.C=!0,Fd(this,n),f=n.L),f.L&&f.L.C||f.R&&f.R.C){f.L&&f.L.C||(f.R.C=!1,f.C=!0,Ed(this,f),f=n.L);f.C=n.C;n.C=f.L.C=!1;Fd(this,n);f=this._;break}f.C=!0;f=n;n=n.U}while(!f.C);f&&(f.C=!1)}}};
  var am=[],Fh,cm=[],ta=1E-6,Ks=1E-12,Ic,Ya,Id,Ha;Hh.prototype={constructor:Hh,polygons:function(){var f=this.edges;return this.cells.map(function(n){var u=n.halfedges.map(function(r){return $l(n,f[r])});u.data=n.site.data;return u})},triangles:function(){var f=[],n=this.edges;this.cells.forEach(function(u,r){if(D=(t=u.halfedges).length){u=u.site;var t,z=-1,D,A=n[t[D-1]];for(A=A.left===u?A.right:A.left;++z<D;){var C=A;A=n[t[z]];A=A.left===u?A.right:A.left;C&&A&&r<C.index&&r<A.index&&0>(u[0]-A[0])*(C[1]-
  u[1])-(u[0]-C[0])*(A[1]-u[1])&&f.push([u.data,C.data,A.data])}}});return f},links:function(){return this.edges.filter(function(f){return f.right}).map(function(f){return{source:f.left.data,target:f.right.data}})},find:function(f,n,u){var r=this,t=r._found||0;var z=r.cells.length;for(var D;!(D=r.cells[t]);)if(++t>=z)return null;z=f-D.site[0];var A=n-D.site[1],C=z*z+A*A;do D=r.cells[z=t],t=null,D.halfedges.forEach(function(G){var O=r.edges[G];G=O.left;if(G!==D.site&&G||(G=O.right)){O=f-G[0];var S=n-
  G[1];O=O*O+S*S;O<C&&(C=O,t=G.index)}});while(null!==t);r._found=z;return null==u||C<=u*u?D.site:null}};yb.prototype={constructor:yb,scale:function(f){return 1===f?this:new yb(this.k*f,this.x,this.y)},translate:function(f,n){return 0===f&0===n?this:new yb(this.k,this.x+this.k*f,this.y+this.k*n)},apply:function(f){return[f[0]*this.k+this.x,f[1]*this.k+this.y]},applyX:function(f){return f*this.k+this.x},applyY:function(f){return f*this.k+this.y},invert:function(f){return[(f[0]-this.x)/this.k,(f[1]-this.y)/
  this.k]},invertX:function(f){return(f-this.x)/this.k},invertY:function(f){return(f-this.y)/this.k},rescaleX:function(f){return f.copy().domain(f.range().map(this.invertX,this).map(f.invert,f))},rescaleY:function(f){return f.copy().domain(f.range().map(this.invertY,this).map(f.invert,f))},toString:function(){return"translate("+this.x+","+this.y+") scale("+this.k+")"}};var pf=new yb(1,0,0);em.prototype=yb.prototype;d3.version="5.7.0";d3.bisect=$b;d3.bisectRight=$b;d3.bisectLeft=Ts;d3.ascending=Mb;d3.bisector=
  Bf;d3.cross=function(f,n,u){var r=f.length,t=n.length,z=Array(r*t),D,A,C;null==u&&(u=ei);for(D=C=0;D<r;++D){var G=f[D];for(A=0;A<t;++A,++C)z[C]=u(G,n[A])}return z};d3.descending=function(f,n){return n<f?-1:n>f?1:n>=f?0:NaN};d3.deviation=gi;d3.extent=Cf;d3.histogram=function(){function f(t){var z,D=t.length,A=Array(D);for(z=0;z<D;++z)A[z]=n(t[z],z,t);z=u(A);var C=z[0],G=z[1],O=r(A,C,G);Array.isArray(O)||(O=Nb(C,G,O),O=Ta(Math.ceil(C/O)*O,G,O));for(var S=O.length;O[0]<=C;)O.shift(),--S;for(;O[S-1]>
  G;)O.pop(),--S;var E=Array(S+1);for(z=0;z<=S;++z){var K=E[z]=[];K.x0=0<z?O[z-1]:C;K.x1=z<S?O[z]:G}for(z=0;z<D;++z)K=A[z],C<=K&&K<=G&&E[$b(O,K,0,S)].push(t[z]);return E}var n=Cn,u=Cf,r=Hf;f.value=function(t){return arguments.length?(n="function"===typeof t?t:Od(t),f):n};f.domain=function(t){return arguments.length?(u="function"===typeof t?t:Od([t[0],t[1]]),f):u};f.thresholds=function(t){return arguments.length?(r="function"===typeof t?t:Array.isArray(t)?Od(Us.call(t)):Od(t),f):r};return f};d3.thresholdFreedmanDiaconis=
  function(f,n,u){f=Vs.call(f,Ab).sort(Mb);return Math.ceil((u-n)/(2*(Oc(f,.75)-Oc(f,.25))*Math.pow(f.length,-1/3)))};d3.thresholdScott=function(f,n,u){return Math.ceil((u-n)/(3.5*gi(f)*Math.pow(f.length,-1/3)))};d3.thresholdSturges=Hf;d3.max=hi;d3.mean=function(f,n){var u=f.length,r=u,t=-1,z,D=0;if(null==n)for(;++t<u;)isNaN(z=Ab(f[t]))?--r:D+=z;else for(;++t<u;)isNaN(z=Ab(n(f[t],t,f)))?--r:D+=z;if(r)return D/r};d3.median=function(f,n){var u=f.length,r=-1,t,z=[];if(null==n)for(;++r<u;)isNaN(t=Ab(f[r]))||
  z.push(t);else for(;++r<u;)isNaN(t=Ab(n(f[r],r,f)))||z.push(t);return Oc(z.sort(Mb),.5)};d3.merge=If;d3.min=ii;d3.pairs=function(f,n){null==n&&(n=ei);for(var u=0,r=f.length-1,t=f[0],z=Array(0>r?0:r);u<r;)z[u]=n(t,t=f[++u]);return z};d3.permute=function(f,n){for(var u=n.length,r=Array(u);u--;)r[u]=f[n[u]];return r};d3.quantile=Oc;d3.range=Ta;d3.scan=function(f,n){if(u=f.length){var u,r=0,t=0,z,D=f[t];for(null==n&&(n=Mb);++r<u;)if(0>n(z=f[r],D)||0!==n(D,D))D=z,t=r;if(0===n(D,D))return t}};d3.shuffle=
  function(f,n,u){u=(null==u?f.length:u)-(n=null==n?0:+n);for(var r,t;u;)t=Math.random()*u--|0,r=f[u+n],f[u+n]=f[t+n],f[t+n]=r;return f};d3.sum=function(f,n){var u=f.length,r=-1,t,z=0;if(null==n)for(;++r<u;){if(t=+f[r])z+=t}else for(;++r<u;)if(t=+n(f[r],r,f))z+=t;return z};d3.ticks=Df;d3.tickIncrement=Nc;d3.tickStep=Nb;d3.transpose=ji;d3.variance=fi;d3.zip=function(){return ji(arguments)};d3.axisTop=function(f){return Pd(1,f)};d3.axisRight=function(f){return Pd(2,f)};d3.axisBottom=function(f){return Pd(3,
  f)};d3.axisLeft=function(f){return Pd(4,f)};d3.brush=function(){return jg(kt)};d3.brushX=function(){return jg(le)};d3.brushY=function(){return jg(ke)};d3.brushSelection=function(f){return(f=f.__brush)?f.dim.output(f.selection):null};d3.chord=function(){function f(z){var D=z.length,A=[],C=Ta(D),G=[],O=[],S=O.groups=Array(D),E=Array(D*D),K,H;var L=0;for(K=-1;++K<D;){var U=0;for(H=-1;++H<D;)U+=z[K][H];A.push(U);G.push(Ta(D));L+=U}u&&C.sort(function(ea,la){return u(A[ea],A[la])});r&&G.forEach(function(ea,
  la){ea.sort(function(pa,R){return r(z[la][pa],z[la][R])})});var M=(L=um(0,tm-n*D)/L)?n:tm/D;U=0;for(K=-1;++K<D;){var X=U;for(H=-1;++H<D;){var Y=C[K],W=G[Y][H],ba=z[Y][W],aa=U,ha=U+=ba*L;E[W*D+Y]={index:Y,subindex:W,startAngle:aa,endAngle:ha,value:ba}}S[Y]={index:Y,startAngle:X,endAngle:U,value:A[Y]};U+=M}for(K=-1;++K<D;)for(H=K-1;++H<D;)C=E[H*D+K],G=E[K*D+H],(C.value||G.value)&&O.push(C.value<G.value?{source:G,target:C}:{source:C,target:G});return t?O.sort(t):O}var n=0,u=null,r=null,t=null;f.padAngle=
  function(z){return arguments.length?(n=um(0,z),f):n};f.sortGroups=function(z){return arguments.length?(u=z,f):u};f.sortSubgroups=function(z){return arguments.length?(r=z,f):r};f.sortChords=function(z){return arguments.length?(null==z?t=null:(t=Hp(z))._=z,f):t&&t._};return f};d3.ribbon=function(){function f(){var A,C=lt.call(arguments),G=n.apply(this,C),O=u.apply(this,C);G=+r.apply(this,(C[0]=G,C));var S=t.apply(this,C)-sf,E=z.apply(this,C)-sf,K=G*qm(S),H=G*rm(S);O=+r.apply(this,(C[0]=O,C));var L=
  t.apply(this,C)-sf;C=z.apply(this,C)-sf;D||(D=A=Eb());D.moveTo(K,H);D.arc(0,0,G,S,E);if(S!==L||E!==C)D.quadraticCurveTo(0,0,O*qm(L),O*rm(L)),D.arc(0,0,O,L,C);D.quadraticCurveTo(0,0,K,H);D.closePath();if(A)return D=null,A+""||null}var n=Ip,u=Jp,r=Kp,t=Lp,z=Mp,D=null;f.radius=function(A){return arguments.length?(r="function"===typeof A?A:lg(+A),f):r};f.startAngle=function(A){return arguments.length?(t="function"===typeof A?A:lg(+A),f):t};f.endAngle=function(A){return arguments.length?(z="function"===
  typeof A?A:lg(+A),f):z};f.source=function(A){return arguments.length?(n=A,f):n};f.target=function(A){return arguments.length?(u=A,f):u};f.context=function(A){return arguments.length?(D=null==A?null:A,f):D};return f};d3.nest=function(){function f(A,C,G,O){if(C>=u.length)return null!=t&&A.sort(t),null!=z?z(A):A;for(var S=-1,E=A.length,K=u[C++],H,L,U=rb(),M,X=G();++S<E;)(M=U.get(H=K(L=A[S])+""))?M.push(L):U.set(H,[L]);U.each(function(Y,W){O(X,W,f(Y,C,G,O))});return X}function n(A,C){if(++C>u.length)return A;
  var G=r[C-1];if(null!=z&&C>=u.length)var O=A.entries();else O=[],A.each(function(S,E){O.push({key:E,values:n(S,C)})});return null!=G?O.sort(function(S,E){return G(S.key,E.key)}):O}var u=[],r=[],t,z,D;return D={object:function(A){return f(A,0,Np,Op)},map:function(A){return f(A,0,hj,ij)},entries:function(A){return n(f(A,0,hj,ij),0)},key:function(A){u.push(A);return D},sortKeys:function(A){r[u.length-1]=A;return D},sortValues:function(A){t=A;return D},rollup:function(A){z=A;return D}}};d3.set=jj;d3.map=
  rb;d3.keys=function(f){var n=[],u;for(u in f)n.push(u);return n};d3.values=function(f){var n=[],u;for(u in f)n.push(f[u]);return n};d3.entries=function(f){var n=[],u;for(u in f)n.push({key:u,value:f[u]});return n};d3.color=Db;d3.rgb=hc;d3.hsl=Zd;d3.lab=$d;d3.hcl=ae;d3.lch=function(f,n,u,r){return 1===arguments.length?Ai(f):new jb(u,n,f,null==r?1:r)};d3.gray=function(f,n){return new cb(f,0,0,null==n?1:n)};d3.cubehelix=db;d3.contours=kj;d3.contourDensity=function(){function f(M){var X=new Float32Array(H*
  L),Y=new Float32Array(H*L);M.forEach(function(W,ba,aa){var ha=+D(W,ba,aa)+K>>E,ea=+A(W,ba,aa)+K>>E;W=+C(W,ba,aa);0<=ha&&ha<H&&0<=ea&&ea<L&&(X[ha+ea*H]+=W)});ng({width:H,height:L,data:X},{width:H,height:L,data:Y},S>>E);og({width:H,height:L,data:Y},{width:H,height:L,data:X},S>>E);ng({width:H,height:L,data:X},{width:H,height:L,data:Y},S>>E);og({width:H,height:L,data:Y},{width:H,height:L,data:X},S>>E);ng({width:H,height:L,data:X},{width:H,height:L,data:Y},S>>E);og({width:H,height:L,data:Y},{width:H,height:L,
  data:X},S>>E);M=U(X);Array.isArray(M)||(Y=hi(X),M=Nb(0,Y,M),M=Ta(0,Math.floor(Y/M)*M,M),M.shift());return kj().thresholds(M).size([H,L])(X).map(n)}function n(M){M.value*=Math.pow(2,-2*E);M.coordinates.forEach(u);return M}function u(M){M.forEach(r)}function r(M){M.forEach(t)}function t(M){M[0]=M[0]*Math.pow(2,E)-K;M[1]=M[1]*Math.pow(2,E)-K}function z(){K=3*S;H=G+2*K>>E;L=O+2*K>>E;return f}var D=Rp,A=Sp,C=Tp,G=960,O=500,S=20,E=2,K=3*S,H=G+2*K>>E,L=O+2*K>>E,U=Fb(20);f.x=function(M){return arguments.length?
  (D="function"===typeof M?M:Fb(+M),f):D};f.y=function(M){return arguments.length?(A="function"===typeof M?M:Fb(+M),f):A};f.weight=function(M){return arguments.length?(C="function"===typeof M?M:Fb(+M),f):C};f.size=function(M){if(!arguments.length)return[G,O];var X=Math.ceil(M[0]),Y=Math.ceil(M[1]);if(!(0<=X||0<=X))throw Error("invalid size");return G=X,O=Y,z()};f.cellSize=function(M){if(!arguments.length)return 1<<E;if(!(1<=(M=+M)))throw Error("invalid cell size");return E=Math.floor(Math.log(M)/Math.LN2),
  z()};f.thresholds=function(M){return arguments.length?(U="function"===typeof M?M:Array.isArray(M)?Fb(lj.call(M)):Fb(M),f):U};f.bandwidth=function(M){if(!arguments.length)return Math.sqrt(S*(S+1));if(!(0<=(M=+M)))throw Error("invalid bandwidth");return S=Math.round((Math.sqrt(4*M*M+1)-1)/2),z()};return f};d3.dispatch=Ob;d3.drag=function(){function f(W){W.on("mousedown.drag",n).filter(S).on("touchstart.drag",t).on("touchmove.drag",z).on("touchend.drag touchcancel.drag",D).style("touch-action","none").style("-webkit-tap-highlight-color",
  "rgba(0,0,0,0)")}function n(){if(!X&&C.apply(this,arguments)){var W=A("mouse",G.apply(this,arguments),Bb,this,arguments);W&&(Ra(d3.event.view).on("mousemove.drag",u,!0).on("mouseup.drag",r,!0),Wd(d3.event.view),d3.event.stopImmediatePropagation(),M=!1,L=d3.event.clientX,U=d3.event.clientY,W("start"))}}function u(){fc();if(!M){var W=d3.event.clientX-L,ba=d3.event.clientY-U;M=W*W+ba*ba>Y}E.mouse("drag")}function r(){Ra(d3.event.view).on("mousemove.drag mouseup.drag",null);Xd(d3.event.view,M);fc();E.mouse("end")}
  function t(){if(C.apply(this,arguments)){var W=d3.event.changedTouches,ba=G.apply(this,arguments),aa=W.length,ha,ea;for(ha=0;ha<aa;++ha)if(ea=A(W[ha].identifier,ba,Vd,this,arguments))d3.event.stopImmediatePropagation(),ea("start")}}function z(){var W=d3.event.changedTouches,ba=W.length,aa,ha;for(aa=0;aa<ba;++aa)if(ha=E[W[aa].identifier])fc(),ha("drag")}function D(){var W=d3.event.changedTouches,ba=W.length,aa,ha;X&&clearTimeout(X);X=setTimeout(function(){X=null},500);for(aa=0;aa<ba;++aa)if(ha=E[W[aa].identifier])d3.event.stopImmediatePropagation(),
  ha("end")}function A(W,ba,aa,ha,ea){var la=aa(ba,W),pa,R,Z,fa=K.copy();if(Qc(new Qf(f,"beforestart",pa,W,H,la[0],la[1],0,0,fa),function(){if(null==(d3.event.subject=pa=O.apply(ha,ea)))return!1;R=pa.x-la[0]||0;Z=pa.y-la[1]||0;return!0}))return function ma(qa){var ya=la;switch(qa){case "start":E[W]=ma;var bb=H++;break;case "end":delete E[W],--H;case "drag":la=aa(ba,W),bb=H}Qc(new Qf(f,qa,pa,W,bb,la[0]+R,la[1]+Z,la[0]-ya[0],la[1]-ya[1],fa),fa.apply,fa,[qa,ha,ea])}}var C=Co,G=Do,O=Eo,S=Fo,E={},K=Ob("start",
  "drag","end"),H=0,L,U,M,X,Y=0;f.filter=function(W){return arguments.length?(C="function"===typeof W?W:Yd(!!W),f):C};f.container=function(W){return arguments.length?(G="function"===typeof W?W:Yd(W),f):G};f.subject=function(W){return arguments.length?(O="function"===typeof W?W:Yd(W),f):O};f.touchable=function(W){return arguments.length?(S="function"===typeof W?W:Yd(!!W),f):S};f.on=function(){var W=K.on.apply(K,arguments);return W===K?f:W};f.clickDistance=function(W){return arguments.length?(Y=(W=+W)*
  W,f):Math.sqrt(Y)};return f};d3.dragDisable=Wd;d3.dragEnable=Xd;d3.dsvFormat=oe;d3.csvParse=vm;d3.csvParseRows=nt;d3.csvFormat=ot;d3.csvFormatRows=pt;d3.tsvParse=wm;d3.tsvParseRows=qt;d3.tsvFormat=rt;d3.tsvFormatRows=st;d3.easeLinear=function(f){return+f};d3.easeQuad=Zi;d3.easeQuadIn=function(f){return f*f};d3.easeQuadOut=function(f){return f*(2-f)};d3.easeQuadInOut=Zi;d3.easeCubic=fg;d3.easeCubicIn=function(f){return f*f*f};d3.easeCubicOut=function(f){return--f*f*f+1};d3.easeCubicInOut=fg;d3.easePoly=
  mm;d3.easePolyIn=dt;d3.easePolyOut=et;d3.easePolyInOut=mm;d3.easeSin=$i;d3.easeSinIn=function(f){return 1-Math.cos(f*nm)};d3.easeSinOut=function(f){return Math.sin(f*nm)};d3.easeSinInOut=$i;d3.easeExp=bj;d3.easeExpIn=function(f){return Math.pow(2,10*f-10)};d3.easeExpOut=function(f){return 1-Math.pow(2,-10*f)};d3.easeExpInOut=bj;d3.easeCircle=cj;d3.easeCircleIn=function(f){return 1-Math.sqrt(1-f*f)};d3.easeCircleOut=function(f){return Math.sqrt(1- --f*f)};d3.easeCircleInOut=cj;d3.easeBounce=$c;d3.easeBounceIn=
  function(f){return 1-$c(1-f)};d3.easeBounceOut=$c;d3.easeBounceInOut=function(f){return(1>=(f*=2)?1-$c(1-f):$c(f-1)+1)/2};d3.easeBack=om;d3.easeBackIn=ft;d3.easeBackOut=gt;d3.easeBackInOut=om;d3.easeElastic=pm;d3.easeElasticIn=ht;d3.easeElasticOut=pm;d3.easeElasticInOut=it;d3.blob=function(f,n){return fetch(f,n).then(Wp)};d3.buffer=function(f,n){return fetch(f,n).then(Xp)};d3.dsv=function(f,n,u,r){3===arguments.length&&"function"===typeof u&&(r=u,u=void 0);var t=oe(f);return qg(n,u).then(function(z){return t.parse(z,
  r)})};d3.csv=tt;d3.tsv=ut;d3.json=function(f,n){return fetch(f,n).then(Zp)};d3.text=qg;d3.forceCenter=function(f,n){function u(){var t,z=r.length,D=0,A=0;for(t=0;t<z;++t){var C=r[t];D+=C.x;A+=C.y}D=D/z-f;A=A/z-n;for(t=0;t<z;++t)C=r[t],C.x-=D,C.y-=A}var r;null==f&&(f=0);null==n&&(n=0);u.initialize=function(t){r=t};u.x=function(t){return arguments.length?(f=+t,u):f};u.y=function(t){return arguments.length?(n=+t,u):n};return u};d3.forceCollide=function(f){function n(){function C(X,Y,W,ba,aa){var ha=
  X.data;X=X.r;var ea=L+X;if(ha)ha.index>E.index&&(Y=K-ha.x-ha.vx,W=H-ha.y-ha.vy,ba=Y*Y+W*W,ba<ea*ea&&(0===Y&&(Y=Gb(),ba+=Y*Y),0===W&&(W=Gb(),ba+=W*W),ba=(ea-(ba=Math.sqrt(ba)))/ba*D,E.vx+=(Y*=ba)*(ea=(X*=X)/(U+X)),E.vy+=(W*=ba)*ea,ha.vx-=Y*(ea=1-ea),ha.vy-=W*ea));else return Y>K+ea||ba<K-ea||W>H+ea||aa<H-ea}for(var G,O=t.length,S,E,K,H,L,U,M=0;M<A;++M)for(S=pe(t,bq,cq).visitAfter(u),G=0;G<O;++G)E=t[G],L=z[E.index],U=L*L,K=E.x+E.vx,H=E.y+E.vy,S.visit(C)}function u(C){if(C.data)return C.r=z[C.data.index];
  for(var G=C.r=0;4>G;++G)C[G]&&C[G].r>C.r&&(C.r=C[G].r)}function r(){if(t){var C,G=t.length;z=Array(G);for(C=0;C<G;++C){var O=t[C];z[O.index]=+f(O,C,t)}}}var t,z,D=1,A=1;"function"!==typeof f&&(f=Ca(null==f?1:+f));n.initialize=function(C){t=C;r()};n.iterations=function(C){return arguments.length?(A=+C,n):A};n.strength=function(C){return arguments.length?(D=+C,n):D};n.radius=function(C){return arguments.length?(f="function"===typeof C?C:Ca(+C),r(),n):f};return n};d3.forceLink=function(f){function n(H){return 1/
  Math.min(S[H.source.index],S[H.target.index])}function u(H){for(var L=0,U=f.length;L<K;++L)for(var M=0,X,Y,W,ba,aa;M<U;++M)X=f[M],Y=X.source,X=X.target,W=X.x+X.vx-Y.x-Y.vx||Gb(),ba=X.y+X.vy-Y.y-Y.vy||Gb(),aa=Math.sqrt(W*W+ba*ba),aa=(aa-G[M])/aa*H*A[M],W*=aa,ba*=aa,X.vx-=W*(aa=E[M]),X.vy-=ba*aa,Y.vx+=W*(aa=1-aa),Y.vy+=ba*aa}function r(){if(O){var H=O.length,L=f.length,U=rb(O,D);var M=0;for(S=Array(H);M<L;++M)H=f[M],H.index=M,"object"!==typeof H.source&&(H.source=rj(U,H.source)),"object"!==typeof H.target&&
  (H.target=rj(U,H.target)),S[H.source.index]=(S[H.source.index]||0)+1,S[H.target.index]=(S[H.target.index]||0)+1;M=0;for(E=Array(L);M<L;++M)H=f[M],E[M]=S[H.source.index]/(S[H.source.index]+S[H.target.index]);A=Array(L);t();G=Array(L);z()}}function t(){if(O)for(var H=0,L=f.length;H<L;++H)A[H]=+n(f[H],H,f)}function z(){if(O)for(var H=0,L=f.length;H<L;++H)G[H]=+C(f[H],H,f)}var D=dq,A,C=Ca(30),G,O,S,E,K=1;null==f&&(f=[]);u.initialize=function(H){O=H;r()};u.links=function(H){return arguments.length?(f=
  H,r(),u):f};u.id=function(H){return arguments.length?(D=H,u):D};u.iterations=function(H){return arguments.length?(K=+H,u):K};u.strength=function(H){return arguments.length?(n="function"===typeof H?H:Ca(+H),t(),u):n};u.distance=function(H){return arguments.length?(C="function"===typeof H?H:Ca(+H),z(),u):C};return u};d3.forceManyBody=function(){function f(E){var K,H=t.length,L=pe(t,eq,fq).visitAfter(u);D=E;for(K=0;K<H;++K)z=t[K],L.visit(r)}function n(){if(t){var E,K=t.length;C=Array(K);for(E=0;E<K;++E){var H=
  t[E];C[H.index]=+A(H,E,t)}}}function u(E){var K=0,H,L,U=0,M,X,Y;if(E.length){for(M=X=Y=0;4>Y;++Y)(H=E[Y])&&(L=Math.abs(H.value))&&(K+=H.value,U+=L,M+=L*H.x,X+=L*H.y);E.x=M/U;E.y=X/U}else{H=E;H.x=H.data.x;H.y=H.data.y;do K+=C[H.data.index];while(H=H.next)}E.value=K}function r(E,K,H,L){if(!E.value)return!0;var U=E.x-z.x,M=E.y-z.y;K=L-K;L=U*U+M*M;if(K*K/S<L)return L<O&&(0===U&&(U=Gb(),L+=U*U),0===M&&(M=Gb(),L+=M*M),L<G&&(L=Math.sqrt(G*L)),z.vx+=U*E.value*D/L,z.vy+=M*E.value*D/L),!0;if(!(E.length||L>=
  O)){if(E.data!==z||E.next)0===U&&(U=Gb(),L+=U*U),0===M&&(M=Gb(),L+=M*M),L<G&&(L=Math.sqrt(G*L));do E.data!==z&&(K=C[E.data.index]*D/L,z.vx+=U*K,z.vy+=M*K);while(E=E.next)}}var t,z,D,A=Ca(-30),C,G=1,O=Infinity,S=.81;f.initialize=function(E){t=E;n()};f.strength=function(E){return arguments.length?(A="function"===typeof E?E:Ca(+E),n(),f):A};f.distanceMin=function(E){return arguments.length?(G=E*E,f):Math.sqrt(G)};f.distanceMax=function(E){return arguments.length?(O=E*E,f):Math.sqrt(O)};f.theta=function(E){return arguments.length?
  (S=E*E,f):Math.sqrt(S)};return f};d3.forceRadial=function(f,n,u){function r(G){for(var O=0,S=z.length;O<S;++O){var E=z[O],K=E.x-n||1E-6,H=E.y-u||1E-6,L=Math.sqrt(K*K+H*H);L=(C[O]-L)*A[O]*G/L;E.vx+=K*L;E.vy+=H*L}}function t(){if(z){var G,O=z.length;A=Array(O);C=Array(O);for(G=0;G<O;++G)C[G]=+f(z[G],G,z),A[G]=isNaN(C[G])?0:+D(z[G],G,z)}}var z,D=Ca(.1),A,C;"function"!==typeof f&&(f=Ca(+f));null==n&&(n=0);null==u&&(u=0);r.initialize=function(G){z=G;t()};r.strength=function(G){return arguments.length?
  (D="function"===typeof G?G:Ca(+G),t(),r):D};r.radius=function(G){return arguments.length?(f="function"===typeof G?G:Ca(+G),t(),r):f};r.x=function(G){return arguments.length?(n=+G,r):n};r.y=function(G){return arguments.length?(u=+G,r):u};return r};d3.forceSimulation=function(f){function n(){u();K.call("tick",z);D<A&&(E.stop(),K.call("end",z))}function u(){var H,L=f.length;D+=(G-D)*C;S.each(function(M){M(D)});for(H=0;H<L;++H){var U=f[H];null==U.fx?U.x+=U.vx*=O:(U.x=U.fx,U.vx=0);null==U.fy?U.y+=U.vy*=
  O:(U.y=U.fy,U.vy=0)}}function r(){for(var H=0,L=f.length,U;H<L;++H){U=f[H];U.index=H;if(isNaN(U.x)||isNaN(U.y)){var M=10*Math.sqrt(H),X=H*vt;U.x=M*Math.cos(X);U.y=M*Math.sin(X)}if(isNaN(U.vx)||isNaN(U.vy))U.vx=U.vy=0}}function t(H){H.initialize&&H.initialize(f);return H}var z,D=1,A=.001,C=1-Math.pow(A,1/300),G=0,O=.6,S=rb(),E=ee(n),K=Ob("tick","end");null==f&&(f=[]);r();return z={tick:u,restart:function(){return E.restart(n),z},stop:function(){return E.stop(),z},nodes:function(H){return arguments.length?
  (f=H,r(),S.each(t),z):f},alpha:function(H){return arguments.length?(D=+H,z):D},alphaMin:function(H){return arguments.length?(A=+H,z):A},alphaDecay:function(H){return arguments.length?(C=+H,z):+C},alphaTarget:function(H){return arguments.length?(G=+H,z):G},velocityDecay:function(H){return arguments.length?(O=1-H,z):1-O},force:function(H,L){return 1<arguments.length?(null==L?S.remove(H):S.set(H,t(L)),z):S.get(H)},find:function(H,L,U){var M,X=f.length;U=null==U?Infinity:U*U;for(M=0;M<X;++M){var Y=f[M];
  var W=H-Y.x;var ba=L-Y.y;W=W*W+ba*ba;if(W<U){var aa=Y;U=W}}return aa},on:function(H,L){return 1<arguments.length?(K.on(H,L),z):K.on(H)}}};d3.forceX=function(f){function n(A){for(var C=0,G=t.length,O;C<G;++C)O=t[C],O.vx+=(D[C]-O.x)*z[C]*A}function u(){if(t){var A,C=t.length;z=Array(C);D=Array(C);for(A=0;A<C;++A)z[A]=isNaN(D[A]=+f(t[A],A,t))?0:+r(t[A],A,t)}}var r=Ca(.1),t,z,D;"function"!==typeof f&&(f=Ca(null==f?0:+f));n.initialize=function(A){t=A;u()};n.strength=function(A){return arguments.length?
  (r="function"===typeof A?A:Ca(+A),u(),n):r};n.x=function(A){return arguments.length?(f="function"===typeof A?A:Ca(+A),u(),n):f};return n};d3.forceY=function(f){function n(A){for(var C=0,G=t.length,O;C<G;++C)O=t[C],O.vy+=(D[C]-O.y)*z[C]*A}function u(){if(t){var A,C=t.length;z=Array(C);D=Array(C);for(A=0;A<C;++A)z[A]=isNaN(D[A]=+f(t[A],A,t))?0:+r(t[A],A,t)}}var r=Ca(.1),t,z,D;"function"!==typeof f&&(f=Ca(null==f?0:+f));n.initialize=function(A){t=A;u()};n.strength=function(A){return arguments.length?
  (r="function"===typeof A?A:Ca(+A),u(),n):r};n.y=function(A){return arguments.length?(f="function"===typeof A?A:Ca(+A),u(),n):f};return n};d3.formatDefaultLocale=yj;d3.formatLocale=uj;d3.formatSpecifier=bd;d3.precisionFixed=zj;d3.precisionPrefix=Aj;d3.precisionRound=Bj;d3.geoArea=function(f){wf.reset();gb(f,lb);return 2*wf};d3.geoBounds=function(f){var n,u,r;Za=Aa=-(za=Wa=Infinity);Hb=[];gb(f,ub);if(n=Hb.length){Hb.sort(oq);f=1;var t=Hb[0];for(u=[t];f<n;++f){var z=Hb[f];Rj(t,z[0])||Rj(t,z[1])?(Xa(t[0],
  z[1])>Xa(t[0],t[1])&&(t[1]=z[1]),Xa(z[0],t[1])>Xa(t[0],t[1])&&(t[0]=z[0])):u.push(t=z)}var D=-Infinity;n=u.length-1;f=0;for(t=u[n];f<=n;t=z,++f)z=u[f],(r=Xa(t[1],z[0]))>D&&(D=r,za=z[0],Aa=t[1])}Hb=tb=null;return Infinity===za||Infinity===Wa?[[NaN,NaN],[NaN,NaN]]:[[za,Wa],[Aa,Za]]};d3.geoCentroid=function(f){ed=Ce=ze=Ae=Be=De=Ee=Fe=Ag=Bg=Cg=0;gb(f,hb);f=Ag;var n=Bg,u=Cg,r=f*f+n*n+u*u;return 1E-12>r&&(f=De,n=Ee,u=Fe,1E-6>Ce&&(f=ze,n=Ae,u=Be),r=f*f+n*n+u*u,1E-12>r)?[NaN,NaN]:[Ma(n,f)*va,La(u/Ba(r))*
  va]};d3.geoCircle=function(){function f(){var A=n.apply(this,arguments),C=u.apply(this,arguments)*ia,G=r.apply(this,arguments)*ia;t=[];z=Fg(-A[0]*ia,-A[1]*ia,0).invert;bk(D,C,G,1);A={type:"Polygon",coordinates:[t]};t=z=null;return A}var n=qc([0,0]),u=qc(90),r=qc(6),t,z,D={point:function(A,C){t.push(A=z(A,C));A[0]*=va;A[1]*=va}};f.center=function(A){return arguments.length?(n="function"===typeof A?A:qc([+A[0],+A[1]]),f):n};f.radius=function(A){return arguments.length?(u="function"===typeof A?A:qc(+A),
  f):u};f.precision=function(A){return arguments.length?(r="function"===typeof A?A:qc(+A),f):r};return f};d3.geoClipAntimeridian=Xg;d3.geoClipCircle=ik;d3.geoClipExtent=function(){var f=0,n=0,u=960,r=500,t,z,D;return D={stream:function(A){return t&&z===A?t:t=Ie(f,n,u,r)(z=A)},extent:function(A){return arguments.length?(f=+A[0][0],n=+A[0][1],u=+A[1][0],r=+A[1][1],t=z=null,D):[[f,n],[u,r]]}}};d3.geoClipRectangle=Ie;d3.geoContains=function(f,n){return(f&&ym.hasOwnProperty(f.type)?ym[f.type]:Le)(f,n)};
  d3.geoDistance=sc;d3.geoGraticule=qk;d3.geoGraticule10=function(){return qk()()};d3.geoInterpolate=function(f,n){var u=f[0]*ia,r=f[1]*ia;f=n[0]*ia;n=n[1]*ia;var t=da(r),z=ca(r),D=da(n),A=ca(n),C=t*da(u),G=t*ca(u),O=D*da(f),S=D*ca(f),E=2*La(Ba(Ej(n-r)+t*D*Ej(f-u))),K=ca(E);f=E?function(H){var L=ca(H*=E)/K,U=ca(E-H)/K;H=U*C+L*O;var M=U*G+L*S;L=U*z+L*A;return[Ma(M,H)*va,Ma(L,Ba(H*H+M*M))*va]}:function(){return[u*va,r*va]};f.distance=E;return f};d3.geoLength=jk;d3.geoPath=function(f,n){function u(D){D&&
  ("function"===typeof r&&z.pointRadius(+r.apply(this,arguments)),gb(D,t(z)));return z.result()}var r=4.5,t,z;u.area=function(D){gb(D,t(vb));return vb.result()};u.measure=function(D){gb(D,t(hd));return hd.result()};u.bounds=function(D){gb(D,t(Pe));return Pe.result()};u.centroid=function(D){gb(D,t($a));return $a.result()};u.projection=function(D){return arguments.length?(t=null==D?(f=null,Xb):(f=D).stream,u):f};u.context=function(D){if(!arguments.length)return n;z=null==D?(n=null,new Dk):new zk(n=D);
  "function"!==typeof r&&z.pointRadius(r);return u};u.pointRadius=function(D){if(!arguments.length)return r;r="function"===typeof D?D:(z.pointRadius(+D),+D);return u};return u.projection(f).context(n)};d3.geoAlbers=Jk;d3.geoAlbersUsa=function(){function f(E){var K=E[0];E=E[1];return O=null,(z.point(K,E),O)||(A.point(K,E),O)||(G.point(K,E),O)}function n(){u=r=null;return f}var u,r,t=Jk(),z,D=Qe().rotate([154,0]).center([-2,58.5]).parallels([55,65]),A,C=Qe().rotate([157,0]).center([-3,19.9]).parallels([8,
  18]),G,O,S={point:function(E,K){O=[E,K]}};f.invert=function(E){var K=t.scale(),H=t.translate(),L=(E[0]-H[0])/K;K=(E[1]-H[1])/K;return(.12<=K&&.234>K&&-.425<=L&&-.214>L?D:.166<=K&&.234>K&&-.214<=L&&-.115>L?C:t).invert(E)};f.stream=function(E){return u&&r===E?u:u=Qq([t.stream(r=E),D.stream(E),C.stream(E)])};f.precision=function(E){if(!arguments.length)return t.precision();t.precision(E);D.precision(E);C.precision(E);return n()};f.scale=function(E){if(!arguments.length)return t.scale();t.scale(E);D.scale(.35*
  E);C.scale(E);return f.translate(t.translate())};f.translate=function(E){if(!arguments.length)return t.translate();var K=t.scale(),H=+E[0],L=+E[1];z=t.translate(E).clipExtent([[H-.455*K,L-.238*K],[H+.455*K,L+.238*K]]).stream(S);A=D.translate([H-.307*K,L+.201*K]).clipExtent([[H-.425*K+1E-6,L+.12*K+1E-6],[H-.214*K-1E-6,L+.234*K-1E-6]]).stream(S);G=C.translate([H-.205*K,L+.212*K]).clipExtent([[H-.214*K+1E-6,L+.166*K+1E-6],[H-.115*K-1E-6,L+.234*K-1E-6]]).stream(S);return n()};f.fitExtent=function(E,K){return uc(f,
  E,K)};f.fitSize=function(E,K){return uc(f,[[0,0],E],K)};f.fitWidth=function(E,K){return Ug(f,E,K)};f.fitHeight=function(E,K){return Vg(f,E,K)};return f.scale(1070)};d3.geoAzimuthalEqualArea=function(){return ob(Qh).scale(124.75).clipAngle(179.999)};d3.geoAzimuthalEqualAreaRaw=Qh;d3.geoAzimuthalEquidistant=function(){return ob(Rh).scale(79.4188).clipAngle(179.999)};d3.geoAzimuthalEquidistantRaw=Rh;d3.geoConicConformal=function(){return Yg(Mk).scale(109.5).parallels([30,30])};d3.geoConicConformalRaw=
  Mk;d3.geoConicEqualArea=Qe;d3.geoConicEqualAreaRaw=Ik;d3.geoConicEquidistant=function(){return Yg(Nk).scale(131.154).center([0,13.9389])};d3.geoConicEquidistantRaw=Nk;d3.geoEqualEarth=function(){return ob($g).scale(177.158)};d3.geoEqualEarthRaw=$g;d3.geoEquirectangular=function(){return ob(od).scale(152.63)};d3.geoEquirectangularRaw=od;d3.geoGnomonic=function(){return ob(ah).scale(144.049).clipAngle(60)};d3.geoGnomonicRaw=ah;d3.geoIdentity=function(){function f(){E=K=null;return H}var n=1,u=0,r=0,
  t=1,z=1,D=Xb,A=null,C,G,O,S=Xb,E,K,H;return H={stream:function(L){return E&&K===L?E:E=D(S(K=L))},postclip:function(L){return arguments.length?(S=L,A=C=G=O=null,f()):S},clipExtent:function(L){return arguments.length?(S=null==L?(A=C=G=O=null,Xb):Ie(A=+L[0][0],C=+L[0][1],G=+L[1][0],O=+L[1][1]),f()):null==A?null:[[A,C],[G,O]]},scale:function(L){return arguments.length?(D=Te((n=+L)*t,n*z,u,r),f()):n},translate:function(L){return arguments.length?(D=Te(n*t,n*z,u=+L[0],r=+L[1]),f()):[u,r]},reflectX:function(L){return arguments.length?
  (D=Te(n*(t=L?-1:1),n*z,u,r),f()):0>t},reflectY:function(L){return arguments.length?(D=Te(n*t,n*(z=L?-1:1),u,r),f()):0>z},fitExtent:function(L,U){return uc(H,L,U)},fitSize:function(L,U){return uc(H,[[0,0],L],U)},fitWidth:function(L,U){return Ug(H,L,U)},fitHeight:function(L,U){return Vg(H,L,U)}}};d3.geoProjection=ob;d3.geoProjectionMutator=Wg;d3.geoMercator=function(){return Lk(nd).scale(961/Sa)};d3.geoMercatorRaw=nd;d3.geoNaturalEarth1=function(){return ob(bh).scale(175.295)};d3.geoNaturalEarth1Raw=
  bh;d3.geoOrthographic=function(){return ob(ch).scale(249.5).clipAngle(90.000001)};d3.geoOrthographicRaw=ch;d3.geoStereographic=function(){return ob(dh).scale(250).clipAngle(142)};d3.geoStereographicRaw=dh;d3.geoTransverseMercator=function(){var f=Lk(eh),n=f.center,u=f.rotate;f.center=function(r){return arguments.length?n([-r[1],r[0]]):(r=n(),[r[1],-r[0]])};f.rotate=function(r){return arguments.length?u([r[0],r[1],2<r.length?r[2]+90:90]):(r=u(),[r[0],r[1],r[2]-90])};return u([0,0,90]).scale(159.155)};
  d3.geoTransverseMercatorRaw=eh;d3.geoRotation=ak;d3.geoStream=gb;d3.geoTransform=function(f){return{stream:kd(f)}};d3.cluster=function(){function f(z){var D,A=0;z.eachAfter(function(E){var K=E.children;if(K){var H=K.reduce(Sq,0)/K.length;E.x=H;E.y=1+K.reduce(Tq,0)}else E.x=D?A+=n(E,D):0,E.y=0,D=E});var C=Uq(z),G=Vq(z),O=C.x-n(C,G)/2,S=G.x+n(G,C)/2;return z.eachAfter(t?function(E){E.x=(E.x-z.x)*u;E.y=(z.y-E.y)*r}:function(E){E.x=(E.x-O)/(S-O)*u;E.y=(1-(z.y?E.y/z.y:1))*r})}var n=Rq,u=1,r=1,t=!1;f.separation=
  function(z){return arguments.length?(n=z,f):n};f.size=function(z){return arguments.length?(t=!1,u=+z[0],r=+z[1],f):t?null:[u,r]};f.nodeSize=function(z){return arguments.length?(t=!0,u=+z[0],r=+z[1],f):t?[u,r]:null};return f};d3.hierarchy=fh;d3.pack=function(){function f(z){z.x=u/2;z.y=r/2;n?z.eachBefore(Wk(n)).eachAfter(hh(t,.5)).eachBefore(Xk(1)):z.eachBefore(Wk($q)).eachAfter(hh(Zb,1)).eachAfter(hh(t,z.r/Math.min(u,r))).eachBefore(Xk(Math.min(u,r)/(2*z.r)));return z}var n=null,u=1,r=1,t=Zb;f.radius=
  function(z){return arguments.length?(n=null==z?null:We(z),f):n};f.size=function(z){return arguments.length?(u=+z[0],r=+z[1],f):[u,r]};f.padding=function(z){return arguments.length?(t="function"===typeof z?z:yc(+z),f):t};return f};d3.packSiblings=function(f){Vk(f);return f};d3.packEnclose=Pk;d3.partition=function(){function f(D){var A=D.height+1;D.x0=D.y0=t;D.x1=u;D.y1=r/A;D.eachBefore(n(r,A));z&&D.eachBefore(Yk);return D}function n(D,A){return function(C){C.children&&qd(C,C.x0,D*(C.depth+1)/A,C.x1,
  D*(C.depth+2)/A);var G=C.x0,O=C.y0,S=C.x1-t,E=C.y1-t;S<G&&(G=S=(G+S)/2);E<O&&(O=E=(O+E)/2);C.x0=G;C.y0=O;C.x1=S;C.y1=E}}var u=1,r=1,t=0,z=!1;f.round=function(D){return arguments.length?(z=!!D,f):z};f.size=function(D){return arguments.length?(u=+D[0],r=+D[1],f):[u,r]};f.padding=function(D){return arguments.length?(t=+D,f):t};return f};d3.stratify=function(){function f(r){var t,z=r.length,D=Array(z),A,C={};for(t=0;t<z;++t){var G=r[t];var O=D[t]=new xc(G);null!=(A=n(G,t,r))&&(A+="")&&(G="$"+(O.id=A),
  C[G]=G in C?zm:O)}for(t=0;t<z;++t)if(O=D[t],A=u(r[t],t,r),null!=A&&(A+="")){G=C["$"+A];if(!G)throw Error("missing: "+A);if(G===zm)throw Error("ambiguous: "+A);G.children?G.children.push(O):G.children=[O];O.parent=G}else{if(S)throw Error("multiple roots");var S=O}if(!S)throw Error("no root");S.parent=wt;S.eachBefore(function(E){E.depth=E.parent.depth+1;--z}).eachBefore(Ok);S.parent=null;if(0<z)throw Error("cycle");return S}var n=ar,u=br;f.id=function(r){return arguments.length?(n=We(r),f):n};f.parentId=
  function(r){return arguments.length?(u=We(r),f):u};return f};d3.tree=function(){function f(C){var G=dr(C);G.eachAfter(n);G.parent.m=-G.z;G.eachBefore(u);if(A)C.eachBefore(r);else{var O=C,S=C,E=C;C.eachBefore(function(U){U.x<O.x&&(O=U);U.x>S.x&&(S=U);U.depth>E.depth&&(E=U)});G=O===S?1:t(O,S)/2;var K=G-O.x,H=z/(S.x+G+K),L=D/(E.depth||1);C.eachBefore(function(U){U.x=(U.x+K)*H;U.y=U.depth*L})}return C}function n(C){var G=C.children,O=C.parent.children,S=C.i?O[C.i-1]:null;if(G){for(var E=0,K=0,H=C.children,
  L=H.length,U;0<=--L;)U=H[L],U.z+=E,U.m+=E,E+=U.s+(K+=U.c);G=(G[0].z+G[G.length-1].z)/2;S?(C.z=S.z+t(C._,S._),C.m=C.z-G):C.z=G}else S&&(C.z=S.z+t(C._,S._));G=C.parent;O=C.parent.A||O[0];if(S){K=E=C;H=E.parent.children[0];L=E.m;U=K.m;for(var M=S.m,X=H.m,Y;S=jh(S),E=ih(E),S&&E;){H=ih(H);K=jh(K);K.a=C;Y=S.z+M-E.z-L+t(S._,E._);if(0<Y){var W=S.a.parent===C.parent?S.a:O,ba=C,aa=Y,ha=aa/(ba.i-W.i);ba.c-=ha;ba.s+=aa;W.c+=ha;ba.z+=aa;ba.m+=aa;L+=Y;U+=Y}M+=S.m;L+=E.m;X+=H.m;U+=K.m}S&&!jh(K)&&(K.t=S,K.m+=M-U);
  E&&!ih(H)&&(H.t=E,H.m+=L-X,O=C)}G.A=O}function u(C){C._.x=C.z+C.parent.m;C.m+=C.parent.m}function r(C){C.x*=z;C.y=C.depth*D}var t=cr,z=1,D=1,A=null;f.separation=function(C){return arguments.length?(t=C,f):t};f.size=function(C){return arguments.length?(A=!1,z=+C[0],D=+C[1],f):A?null:[z,D]};f.nodeSize=function(C){return arguments.length?(A=!0,z=+C[0],D=+C[1],f):A?[z,D]:null};return f};d3.treemap=function(){function f(E){E.x0=E.y0=0;E.x1=t;E.y1=z;E.eachBefore(n);D=[0];r&&E.eachBefore(Yk);return E}function n(E){var K=
  D[E.depth],H=E.x0+K,L=E.y0+K,U=E.x1-K,M=E.y1-K;U<H&&(H=U=(H+U)/2);M<L&&(L=M=(L+M)/2);E.x0=H;E.y0=L;E.x1=U;E.y1=M;E.children&&(K=D[E.depth+1]=A(E)/2,H+=S(E)-K,L+=C(E)-K,U-=G(E)-K,M-=O(E)-K,U<H&&(H=U=(H+U)/2),M<L&&(L=M=(L+M)/2),u(E,H,L,U,M))}var u=Bm,r=!1,t=1,z=1,D=[0],A=Zb,C=Zb,G=Zb,O=Zb,S=Zb;f.round=function(E){return arguments.length?(r=!!E,f):r};f.size=function(E){return arguments.length?(t=+E[0],z=+E[1],f):[t,z]};f.tile=function(E){return arguments.length?(u=We(E),f):u};f.padding=function(E){return arguments.length?
  f.paddingInner(E).paddingOuter(E):f.paddingInner()};f.paddingInner=function(E){return arguments.length?(A="function"===typeof E?E:yc(+E),f):A};f.paddingOuter=function(E){return arguments.length?f.paddingTop(E).paddingRight(E).paddingBottom(E).paddingLeft(E):f.paddingTop()};f.paddingTop=function(E){return arguments.length?(C="function"===typeof E?E:yc(+E),f):C};f.paddingRight=function(E){return arguments.length?(G="function"===typeof E?E:yc(+E),f):G};f.paddingBottom=function(E){return arguments.length?
  (O="function"===typeof E?E:yc(+E),f):O};f.paddingLeft=function(E){return arguments.length?(S="function"===typeof E?E:yc(+E),f):S};return f};d3.treemapBinary=function(f,n,u,r,t){function z(S,E,K,H,L,U,M){if(S>=E-1)S=D[S],S.x0=H,S.y0=L,S.x1=U,S.y1=M;else{for(var X=O[S],Y=K/2+X,W=S+1,ba=E-1;W<ba;){var aa=W+ba>>>1;O[aa]<Y?W=aa+1:ba=aa}Y-O[W-1]<O[W]-Y&&S+1<W&&--W;X=O[W]-X;Y=K-X;U-H>M-L?(K=(H*Y+U*X)/K,z(S,W,X,H,L,K,M),z(W,E,Y,K,L,U,M)):(K=(L*Y+M*X)/K,z(S,W,X,H,L,U,K),z(W,E,Y,H,K,U,M))}}var D=f.children,
  A,C=D.length,G,O=Array(C+1);for(O[0]=G=A=0;A<C;++A)O[A+1]=G+=D[A].value;z(0,C,f.value,n,u,r,t)};d3.treemapDice=qd;d3.treemapSlice=Ye;d3.treemapSliceDice=function(f,n,u,r,t){(f.depth&1?Ye:qd)(f,n,u,r,t)};d3.treemapSquarify=Bm;d3.treemapResquarify=xt;d3.interpolate=Sc;d3.interpolateArray=Ii;d3.interpolateBasis=Ei;d3.interpolateBasisClosed=Fi;d3.interpolateDate=Ji;d3.interpolateDiscrete=function(f){var n=f.length;return function(u){return f[Math.max(0,Math.min(n-1,Math.floor(u*n)))]}};d3.interpolateHue=
  function(f,n){var u=ce(+f,+n);return function(r){r=u(r);return r-360*Math.floor(r/360)}};d3.interpolateNumber=Va;d3.interpolateObject=Ki;d3.interpolateRound=Li;d3.interpolateString=Zf;d3.interpolateTransformCss=jm;d3.interpolateTransformSvg=km;d3.interpolateZoom=Qi;d3.interpolateRgb=Tc;d3.interpolateRgbBasis=Bl;d3.interpolateRgbBasisClosed=Xs;d3.interpolateHsl=Ys;d3.interpolateHslLong=Zs;d3.interpolateLab=function(f,n){var u=Ea((f=$d(f)).l,(n=$d(n)).l),r=Ea(f.a,n.a),t=Ea(f.b,n.b),z=Ea(f.opacity,n.opacity);
  return function(D){f.l=u(D);f.a=r(D);f.b=t(D);f.opacity=z(D);return f+""}};d3.interpolateHcl=$s;d3.interpolateHclLong=at;d3.interpolateCubehelix=bt;d3.interpolateCubehelixLong=rf;d3.piecewise=function(f,n){for(var u=0,r=n.length-1,t=n[0],z=Array(0>r?0:r);u<r;)z[u]=f(t,t=n[++u]);return function(D){var A=Math.max(0,Math.min(r-1,Math.floor(D*=r)));return z[A](D-A)}};d3.quantize=function(f,n){for(var u=Array(n),r=0;r<n;++r)u[r]=f(r/(n-1));return u};d3.path=Eb;d3.polygonArea=function(f){for(var n=-1,u=
  f.length,r,t=f[u-1],z=0;++n<u;)r=t,t=f[n],z+=r[1]*t[0]-r[0]*t[1];return z/2};d3.polygonCentroid=function(f){for(var n=-1,u=f.length,r=0,t=0,z,D=f[u-1],A,C=0;++n<u;)z=D,D=f[n],C+=A=z[0]*D[1]-D[0]*z[1],r+=(z[0]+D[0])*A,t+=(z[1]+D[1])*A;return C*=3,[r/C,t/C]};d3.polygonHull=function(f){if(3>(u=f.length))return null;var n,u,r=Array(u),t=Array(u);for(n=0;n<u;++n)r[n]=[+f[n][0],+f[n][1],n];r.sort(fr);for(n=0;n<u;++n)t[n]=[r[n][0],-r[n][1]];u=$k(r);t=$k(t);var z=t[0]===u[0],D=t[t.length-1]===u[u.length-
  1],A=[];for(n=u.length-1;0<=n;--n)A.push(f[r[u[n]][2]]);for(n=+z;n<t.length-D;++n)A.push(f[r[t[n]][2]]);return A};d3.polygonContains=function(f,n){var u=f.length,r=f[u-1],t=n[0];n=n[1];for(var z=r[0],D=r[1],A,C=!1,G=0;G<u;++G)r=f[G],A=r[0],r=r[1],r>n!==D>n&&t<(z-A)*(n-r)/(D-r)+A&&(C=!C),z=A,D=r;return C};d3.polygonLength=function(f){var n=-1,u=f.length,r=f[u-1],t=r[0];r=r[1];for(var z=0;++n<u;){var D=t;var A=r;r=f[n];t=r[0];r=r[1];D-=t;A-=r;z+=Math.sqrt(D*D+A*A)}return z};d3.quadtree=pe;d3.randomUniform=
  yt;d3.randomNormal=Cm;d3.randomLogNormal=zt;d3.randomBates=At;d3.randomIrwinHall=Dm;d3.randomExponential=Bt;d3.scaleBand=mh;d3.scalePoint=function(){return al(mh().paddingInner(1))};d3.scaleIdentity=el;d3.scaleLinear=dl;d3.scaleLog=jl;d3.scaleOrdinal=kh;d3.scaleImplicit=lh;d3.scalePow=qh;d3.scaleSqrt=function(){return qh().exponent(.5)};d3.scaleQuantile=kl;d3.scaleQuantize=ll;d3.scaleThreshold=ml;d3.scaleTime=function(){return th(wb,Uh,yd,vd,Th,Sh,Nd,dc,d3.timeFormat).domain([new Date(2E3,0,1),new Date(2E3,
  0,2)])};d3.scaleUtc=function(){return th(xb,Xh,Ad,td,Wh,Vh,Nd,dc,d3.utcFormat).domain([Date.UTC(2E3,0,1),Date.UTC(2E3,0,2)])};d3.scaleSequential=zl;d3.scaleDiverging=Al;d3.schemeCategory10=$t;d3.schemeCategory20b=au;d3.schemeCategory20c=bu;d3.schemeCategory20=cu;d3.schemeAccent=du;d3.schemeDark2=eu;d3.schemePaired=fu;d3.schemePastel1=gu;d3.schemePastel2=hu;d3.schemeSet1=iu;d3.schemeSet2=ju;d3.schemeSet3=ku;d3.interpolateBrBG=lu;d3.schemeBrBG=Rm;d3.interpolatePRGn=mu;d3.schemePRGn=Sm;d3.interpolatePiYG=
  nu;d3.schemePiYG=Tm;d3.interpolatePuOr=ou;d3.schemePuOr=Um;d3.interpolateRdBu=pu;d3.schemeRdBu=Vm;d3.interpolateRdGy=qu;d3.schemeRdGy=Wm;d3.interpolateRdYlBu=ru;d3.schemeRdYlBu=Xm;d3.interpolateRdYlGn=su;d3.schemeRdYlGn=Ym;d3.interpolateSpectral=tu;d3.schemeSpectral=Zm;d3.interpolateBuGn=uu;d3.schemeBuGn=$m;d3.interpolateBuPu=vu;d3.schemeBuPu=an;d3.interpolateGnBu=wu;d3.schemeGnBu=bn;d3.interpolateOrRd=xu;d3.schemeOrRd=cn;d3.interpolatePuBuGn=yu;d3.schemePuBuGn=dn;d3.interpolatePuBu=zu;d3.schemePuBu=
  en;d3.interpolatePuRd=Au;d3.schemePuRd=fn;d3.interpolateRdPu=Bu;d3.schemeRdPu=gn;d3.interpolateYlGnBu=Cu;d3.schemeYlGnBu=hn;d3.interpolateYlGn=Du;d3.schemeYlGn=jn;d3.interpolateYlOrBr=Eu;d3.schemeYlOrBr=kn;d3.interpolateYlOrRd=Fu;d3.schemeYlOrRd=ln;d3.interpolateBlues=Gu;d3.schemeBlues=mn;d3.interpolateGreens=Hu;d3.schemeGreens=nn;d3.interpolateGreys=Iu;d3.schemeGreys=on;d3.interpolatePurples=Ju;d3.schemePurples=pn;d3.interpolateReds=Ku;d3.schemeReds=qn;d3.interpolateOranges=Lu;d3.schemeOranges=rn;
  d3.interpolateCubehelixDefault=Mu;d3.interpolateRainbow=function(f){if(0>f||1<f)f-=Math.floor(f);var n=Math.abs(f-.5);zf.h=360*f-100;zf.s=1.5-1.5*n;zf.l=.8-.9*n;return zf+""};d3.interpolateWarm=Nu;d3.interpolateCool=Ou;d3.interpolateSinebow=function(f){var n;f=(.5-f)*Math.PI;Af.r=255*(n=Math.sin(f))*n;Af.g=255*(n=Math.sin(f+Pu))*n;Af.b=255*(n=Math.sin(f+Qu))*n;return Af+""};d3.interpolateViridis=Ru;d3.interpolateMagma=Su;d3.interpolateInferno=Tu;d3.interpolatePlasma=Uu;d3.create=function(f){return Ra(Rd(f).call(document.documentElement))};
  d3.creator=Rd;d3.local=si;d3.matcher=Ih;d3.mouse=Bb;d3.namespace=Pc;d3.namespaces=Ua;d3.clientPoint=Ud;d3.select=Ra;d3.selectAll=function(f){return"string"===typeof f?new Ja([document.querySelectorAll(f)],[document.documentElement]):new Ja([null==f?[]:f],Nf)};d3.selection=Qb;d3.selector=Sd;d3.selectorAll=Kf;d3.style=Pb;d3.touch=Vd;d3.touches=function(f,n){null==n&&(n=Pf().touches);for(var u=0,r=n?n.length:0,t=Array(r);u<r;++u)t[u]=Ud(f,n[u]);return t};d3.window=Lf;d3.customEvent=Qc;d3.arc=function(){function f(){var G,
  O=+n.apply(this,arguments),S=+u.apply(this,arguments),E=z.apply(this,arguments)-cf,K=D.apply(this,arguments)-cf,H=sn(K-E),L=K>E;C||(C=G=Eb());if(S<O){var U=S;S=O;O=U}if(1E-12<S)if(H>Lb-1E-12)C.moveTo(S*ec(E),S*pb(E)),C.arc(0,0,S,E,K,!L),1E-12<O&&(C.moveTo(O*ec(K),O*pb(K)),C.arc(0,0,O,K,E,L));else{var M=E,X=K;U=E;var Y=K,W=H,ba=H,aa=A.apply(this,arguments)/2,ha=1E-12<aa&&(t?+t.apply(this,arguments):Dc(O*O+S*S)),ea=Yh(sn(S-O)/2,+r.apply(this,arguments)),la=ea,pa=ea;if(1E-12<ha){var R=Cl(ha/O*pb(aa));
  aa=Cl(ha/S*pb(aa));1E-12<(W-=2*R)?(R*=L?1:-1,U+=R,Y-=R):(W=0,U=Y=(E+K)/2);1E-12<(ba-=2*aa)?(aa*=L?1:-1,M+=aa,X-=aa):(ba=0,M=X=(E+K)/2)}E=S*ec(M);K=S*pb(M);R=O*ec(Y);aa=O*pb(Y);if(1E-12<ea){var Z=S*ec(X),fa=S*pb(X),ja=O*ec(U),qa=O*pb(U);if(H<Kb){1E-12<W?(la=ja-E,pa=qa-K,H=R-Z,ha=aa-fa,H=(H*(K-fa)-ha*(E-Z))/(ha*la-H*pa),la=[E+H*la,K+H*pa]):la=[R,aa];pa=E-la[0];H=K-la[1];ha=Z-la[0];var ma=fa-la[1];pa=(pa*ha+H*ma)/(Dc(pa*pa+H*H)*Dc(ha*ha+ma*ma));pa=1/pb((1<pa?0:-1>pa?Kb:Math.acos(pa))/2);H=Dc(la[0]*la[0]+
  la[1]*la[1]);la=Yh(ea,(O-H)/(pa-1));pa=Yh(ea,(S-H)/(pa+1))}}1E-12<ba?1E-12<pa?(M=df(ja,qa,E,K,S,pa,L),X=df(Z,fa,R,aa,S,pa,L),C.moveTo(M.cx+M.x01,M.cy+M.y01),pa<ea?C.arc(M.cx,M.cy,pa,Ia(M.y01,M.x01),Ia(X.y01,X.x01),!L):(C.arc(M.cx,M.cy,pa,Ia(M.y01,M.x01),Ia(M.y11,M.x11),!L),C.arc(0,0,S,Ia(M.cy+M.y11,M.cx+M.x11),Ia(X.cy+X.y11,X.cx+X.x11),!L),C.arc(X.cx,X.cy,pa,Ia(X.y11,X.x11),Ia(X.y01,X.x01),!L))):(C.moveTo(E,K),C.arc(0,0,S,M,X,!L)):C.moveTo(E,K);1E-12<O&&1E-12<W?1E-12<la?(M=df(R,aa,Z,fa,O,-la,L),X=
  df(E,K,ja,qa,O,-la,L),C.lineTo(M.cx+M.x01,M.cy+M.y01),la<ea?C.arc(M.cx,M.cy,la,Ia(M.y01,M.x01),Ia(X.y01,X.x01),!L):(C.arc(M.cx,M.cy,la,Ia(M.y01,M.x01),Ia(M.y11,M.x11),!L),C.arc(0,0,O,Ia(M.cy+M.y11,M.cx+M.x11),Ia(X.cy+X.y11,X.cx+X.x11),L),C.arc(X.cx,X.cy,la,Ia(X.y11,X.x11),Ia(X.y01,X.x01),!L))):C.arc(0,0,O,Y,U,L):C.lineTo(R,aa)}else C.moveTo(0,0);C.closePath();if(G)return C=null,G+""||null}var n=ps,u=qs,r=na(0),t=null,z=rs,D=ss,A=ts,C=null;f.centroid=function(){var G=(+n.apply(this,arguments)+ +u.apply(this,
  arguments))/2,O=(+z.apply(this,arguments)+ +D.apply(this,arguments))/2-Kb/2;return[ec(O)*G,pb(O)*G]};f.innerRadius=function(G){return arguments.length?(n="function"===typeof G?G:na(+G),f):n};f.outerRadius=function(G){return arguments.length?(u="function"===typeof G?G:na(+G),f):u};f.cornerRadius=function(G){return arguments.length?(r="function"===typeof G?G:na(+G),f):r};f.padRadius=function(G){return arguments.length?(t=null==G?null:"function"===typeof G?G:na(+G),f):t};f.startAngle=function(G){return arguments.length?
  (z="function"===typeof G?G:na(+G),f):z};f.endAngle=function(G){return arguments.length?(D="function"===typeof G?G:na(+G),f):D};f.padAngle=function(G){return arguments.length?(A="function"===typeof G?G:na(+G),f):A};f.context=function(G){return arguments.length?(C=null==G?null:G,f):C};return f};d3.area=El;d3.line=wh;d3.pie=function(){function f(A){var C,G=A.length;var O=0;var S=Array(G),E=Array(G),K=+t.apply(this,arguments);var H=Math.min(Lb,Math.max(-Lb,z.apply(this,arguments)-K));var L=Math.min(Math.abs(H)/
  G,D.apply(this,arguments)),U=L*(0>H?-1:1),M;for(C=0;C<G;++C)0<(M=E[S[C]=C]=+n(A[C],C,A))&&(O+=M);null!=u?S.sort(function(Y,W){return u(E[Y],E[W])}):null!=r&&S.sort(function(Y,W){return r(A[Y],A[W])});C=0;for(H=O?(H-G*U)/O:0;C<G;++C,K=X){O=S[C];M=E[O];var X=K+(0<M?M*H:0)+U;E[O]={data:A[O],index:C,value:M,startAngle:K,endAngle:X,padAngle:L}}return E}var n=ws,u=vs,r=null,t=na(0),z=na(Lb),D=na(0);f.value=function(A){return arguments.length?(n="function"===typeof A?A:na(+A),f):n};f.sortValues=function(A){return arguments.length?
  (u=A,r=null,f):u};f.sort=function(A){return arguments.length?(r=A,u=null,f):r};f.startAngle=function(A){return arguments.length?(t="function"===typeof A?A:na(+A),f):t};f.endAngle=function(A){return arguments.length?(z="function"===typeof A?A:na(+A),f):z};f.padAngle=function(A){return arguments.length?(D="function"===typeof A?A:na(+A),f):D};return f};d3.areaRadial=Il;d3.radialArea=Il;d3.lineRadial=Gl;d3.radialLine=Gl;d3.pointRadial=Dd;d3.linkHorizontal=function(){return yh(zs)};d3.linkVertical=function(){return yh(As)};
  d3.linkRadial=function(){var f=yh(Bs);f.angle=f.x;delete f.x;f.radius=f.y;delete f.y;return f};d3.symbol=function(){function f(){var t;r||(r=t=Eb());n.apply(this,arguments).draw(r,+u.apply(this,arguments));if(t)return r=null,t+""||null}var n=na(Zh),u=na(64),r=null;f.type=function(t){return arguments.length?(n="function"===typeof t?t:na(t),f):n};f.size=function(t){return arguments.length?(u="function"===typeof t?t:na(+t),f):u};f.context=function(t){return arguments.length?(r=null==t?null:t,f):r};return f};
  d3.symbols=Zu;d3.symbolCircle=Zh;d3.symbolCross=tn;d3.symbolDiamond=vn;d3.symbolSquare=yn;d3.symbolStar=xn;d3.symbolTriangle=zn;d3.symbolWye=An;d3.curveBasisClosed=function(f){return new Jl(f)};d3.curveBasisOpen=function(f){return new Kl(f)};d3.curveBasis=function(f){return new gf(f)};d3.curveBundle=$u;d3.curveCardinalClosed=bv;d3.curveCardinalOpen=cv;d3.curveCardinal=av;d3.curveCatmullRomClosed=ev;d3.curveCatmullRomOpen=fv;d3.curveCatmullRom=dv;d3.curveLinearClosed=function(f){return new Pl(f)};
  d3.curveLinear=ef;d3.curveMonotoneX=function(f){return new jf(f)};d3.curveMonotoneY=function(f){return new Sl(f)};d3.curveNatural=function(f){return new Ul(f)};d3.curveStep=function(f){return new kf(f,.5)};d3.curveStepAfter=function(f){return new kf(f,1)};d3.curveStepBefore=function(f){return new kf(f,0)};d3.stack=function(){function f(z){var D=n.apply(this,arguments),A,C=z.length,G=D.length,O=Array(G);for(A=0;A<G;++A){for(var S=D[A],E=O[A]=Array(C),K=0,H;K<C;++K)E[K]=H=[0,+t(z[K],S,K,z)],H.data=
  z[K];E.key=S}A=0;for(D=u(O);A<G;++A)O[D[A]].index=A;r(O,D);return O}var n=na([]),u=Fc,r=Ec,t=Cs;f.keys=function(z){return arguments.length?(n="function"===typeof z?z:na(zh.call(z)),f):n};f.value=function(z){return arguments.length?(t="function"===typeof z?z:na(+z),f):t};f.order=function(z){return arguments.length?(u=null==z?Fc:"function"===typeof z?z:na(zh.call(z)),f):u};f.offset=function(z){return arguments.length?(r=null==z?Ec:z,f):r};return f};d3.stackOffsetExpand=function(f,n){if(0<(r=f.length)){for(var u,
  r,t=0,z=f[0].length,D;t<z;++t){for(D=u=0;u<r;++u)D+=f[u][t][1]||0;if(D)for(u=0;u<r;++u)f[u][t][1]/=D}Ec(f,n)}};d3.stackOffsetDiverging=function(f,n){if(1<(C=f.length))for(var u,r=0,t,z,D,A,C,G=f[n[0]].length;r<G;++r)for(u=D=A=0;u<C;++u)0<=(z=(t=f[n[u]][r])[1]-t[0])?(t[0]=D,t[1]=D+=z):0>z?(t[1]=A,t[0]=A+=z):t[0]=D};d3.stackOffsetNone=Ec;d3.stackOffsetSilhouette=function(f,n){if(0<(t=f.length)){for(var u=0,r=f[n[0]],t,z=r.length;u<z;++u){for(var D=0,A=0;D<t;++D)A+=f[D][u][1]||0;r[u][1]+=r[u][0]=-A/
  2}Ec(f,n)}};d3.stackOffsetWiggle=function(f,n){if(0<(D=f.length)&&0<(z=(t=f[n[0]]).length)){for(var u=0,r=1,t,z,D;r<z;++r){for(var A=0,C=0,G=0;A<D;++A){var O=f[n[A]],S=O[r][1]||0;O=(S-(O[r-1][1]||0))/2;for(var E=0;E<A;++E){var K=f[n[E]];O+=(K[r][1]||0)-(K[r-1][1]||0)}C+=S;G+=O*S}t[r-1][1]+=t[r-1][0]=u;C&&(u-=G/C)}t[r-1][1]+=t[r-1][0]=u;Ec(f,n)}};d3.stackOrderAscending=Wl;d3.stackOrderDescending=function(f){return Wl(f).reverse()};d3.stackOrderInsideOut=function(f){var n=f.length,u=f.map(Xl),r=Fc(f).sort(function(G,
  O){return u[O]-u[G]}),t=0,z=0,D=[],A=[];for(f=0;f<n;++f){var C=r[f];t<z?(t+=u[C],D.push(C)):(z+=u[C],A.push(C))}return A.reverse().concat(D)};d3.stackOrderNone=Fc;d3.stackOrderReverse=function(f){return Fc(f).reverse()};d3.timeInterval=Da;d3.timeMillisecond=dc;d3.timeMilliseconds=Fm;d3.utcMillisecond=dc;d3.utcMilliseconds=Fm;d3.timeSecond=Nd;d3.timeSeconds=Gm;d3.utcSecond=Nd;d3.utcSeconds=Gm;d3.timeMinute=Sh;d3.timeMinutes=Ct;d3.timeHour=Th;d3.timeHours=Dt;d3.timeDay=vd;d3.timeDays=Et;d3.timeWeek=
  yd;d3.timeWeeks=Lm;d3.timeSunday=yd;d3.timeSundays=Lm;d3.timeMonday=ud;d3.timeMondays=Ft;d3.timeTuesday=Hm;d3.timeTuesdays=Gt;d3.timeWednesday=Im;d3.timeWednesdays=Ht;d3.timeThursday=zd;d3.timeThursdays=It;d3.timeFriday=Jm;d3.timeFridays=Jt;d3.timeSaturday=Km;d3.timeSaturdays=Kt;d3.timeMonth=Uh;d3.timeMonths=Lt;d3.timeYear=wb;d3.timeYears=Mt;d3.utcMinute=Vh;d3.utcMinutes=Nt;d3.utcHour=Wh;d3.utcHours=Ot;d3.utcDay=td;d3.utcDays=Pt;d3.utcWeek=Ad;d3.utcWeeks=Qm;d3.utcSunday=Ad;d3.utcSundays=Qm;d3.utcMonday=
  sd;d3.utcMondays=Qt;d3.utcTuesday=Mm;d3.utcTuesdays=Rt;d3.utcWednesday=Nm;d3.utcWednesdays=St;d3.utcThursday=Bd;d3.utcThursdays=Tt;d3.utcFriday=Om;d3.utcFridays=Ut;d3.utcSaturday=Pm;d3.utcSaturdays=Vt;d3.utcMonth=Xh;d3.utcMonths=Wt;d3.utcYear=xb;d3.utcYears=Xt;d3.timeFormatDefaultLocale=yl;d3.timeFormatLocale=nl;d3.isoFormat=Yt;d3.isoParse=Zt;d3.now=jc;d3.timer=ee;d3.timerFlush=Vi;d3.timeout=cg;d3.interval=function(f,n,u){var r=new Wc,t=n;if(null==n)return r.restart(f,n,u),r;n=+n;u=null==u?jc():+u;
  r.restart(function A(D){D+=t;r.restart(A,t+=n,u);f(D)},n,u);return r};d3.transition=Yi;d3.active=function(f,n){var u=f.__transition,r,t;if(u)for(t in n=null==n?null:n+"",u)if(1<(r=u[t]).state&&r.name===n)return new kb([[f]],jt,n,+t);return null};d3.interrupt=Ub;d3.voronoi=function(){function f(t){return new Hh(t.map(function(z,D){var A=[Math.round(n(z,D,t)/ta)*ta,Math.round(u(z,D,t)/ta)*ta];A.index=D;A.data=z;return A}),r)}var n=Ds,u=Es,r=null;f.polygons=function(t){return f(t).polygons()};f.links=
  function(t){return f(t).links()};f.triangles=function(t){return f(t).triangles()};f.x=function(t){return arguments.length?(n="function"===typeof t?t:Yl(+t),f):n};f.y=function(t){return arguments.length?(u="function"===typeof t?t:Yl(+t),f):u};f.extent=function(t){return arguments.length?(r=null==t?null:[[+t[0][0],+t[0][1]],[+t[1][0],+t[1][1]]],f):r&&[[r[0][0],r[0][1]],[r[1][0],r[1][1]]]};f.size=function(t){return arguments.length?(r=null==t?null:[[0,0],[+t[0],+t[1]]],f):r&&[r[1][0]-r[0][0],r[1][1]-
  r[0][1]]};return f};d3.zoom=function(){function f(R){R.property("__zoom",fm).on("wheel.zoom",A).on("mousedown.zoom",C).on("dblclick.zoom",G).filter(M).on("touchstart.zoom",O).on("touchmove.zoom",S).on("touchend.zoom touchcancel.zoom",E).style("touch-action","none").style("-webkit-tap-highlight-color","rgba(0,0,0,0)")}function n(R,Z){Z=Math.max(X[0],Math.min(X[1],Z));return Z===R.k?R:new yb(Z,R.x,R.y)}function u(R,Z,fa){var ja=Z[0]-fa[0]*R.k;Z=Z[1]-fa[1]*R.k;return ja===R.x&&Z===R.y?R:new yb(R.k,ja,
  Z)}function r(R){return[(+R[0][0]+ +R[1][0])/2,(+R[0][1]+ +R[1][1])/2]}function t(R,Z,fa){R.on("start.zoom",function(){z(this,arguments).start()}).on("interrupt.zoom end.zoom",function(){z(this,arguments).end()}).tween("zoom",function(){var ja=arguments,qa=z(this,ja),ma=H.apply(this,ja),ya=fa||r(ma),bb=Math.max(ma[1][0]-ma[0][0],ma[1][1]-ma[0][1]);ma=this.__zoom;var bi="function"===typeof Z?Z.apply(this,ja):Z,gv=ba(ma.invert(ya).concat(bb/ma.k),bi.invert(ya).concat(bb/bi.k));return function(zb){if(1===
  zb)zb=bi;else{zb=gv(zb);var ci=bb/zb[2];zb=new yb(ci,ya[0]-zb[0]*ci,ya[1]-zb[1]*ci)}qa.zoom(null,zb)}})}function z(R,Z){for(var fa=0,ja=aa.length,qa;fa<ja;++fa)if((qa=aa[fa]).that===R)return qa;return new D(R,Z)}function D(R,Z){this.that=R;this.args=Z;this.index=-1;this.active=0;this.extent=H.apply(R,Z)}function A(){if(K.apply(this,arguments)){var R=z(this,arguments),Z=this.__zoom,fa=Math.max(X[0],Math.min(X[1],Z.k*Math.pow(2,U.apply(this,arguments)))),ja=Bb(this);if(R.wheel){if(R.mouse[0][0]!==ja[0]||
  R.mouse[0][1]!==ja[1])R.mouse[1]=Z.invert(R.mouse[0]=ja);clearTimeout(R.wheel)}else{if(Z.k===fa)return;R.mouse=[ja,Z.invert(ja)];Ub(this);R.start()}Jd();R.wheel=setTimeout(function(){R.wheel=null;R.end()},150);R.zoom("mouse",L(u(n(Z,fa),R.mouse[0],R.mouse[1]),R.extent,Y))}}function C(){if(!la&&K.apply(this,arguments)){var R=z(this,arguments),Z=Ra(d3.event.view).on("mousemove.zoom",function(){Jd();if(!R.moved){var ma=d3.event.clientX-ja,ya=d3.event.clientY-qa;R.moved=ma*ma+ya*ya>pa}R.zoom("mouse",
  L(u(R.that.__zoom,R.mouse[0]=Bb(R.that),R.mouse[1]),R.extent,Y))},!0).on("mouseup.zoom",function(){Z.on("mousemove.zoom mouseup.zoom",null);Xd(d3.event.view,R.moved);Jd();R.end()},!0),fa=Bb(this),ja=d3.event.clientX,qa=d3.event.clientY;Wd(d3.event.view);d3.event.stopImmediatePropagation();R.mouse=[fa,this.__zoom.invert(fa)];Ub(this);R.start()}}function G(){if(K.apply(this,arguments)){var R=this.__zoom,Z=Bb(this),fa=R.invert(Z);R=L(u(n(R,R.k*(d3.event.shiftKey?.5:2)),Z,fa),H.apply(this,arguments),
  Y);Jd();0<W?Ra(this).transition().duration(W).call(t,R,Z):Ra(this).call(f.transform,R)}}function O(){if(K.apply(this,arguments)){var R=z(this,arguments),Z=d3.event.changedTouches,fa=Z.length,ja;d3.event.stopImmediatePropagation();for(ja=0;ja<fa;++ja){var qa=Z[ja];var ma=Vd(this,Z,qa.identifier);ma=[ma,this.__zoom.invert(ma),qa.identifier];if(R.touch0)R.touch1||(R.touch1=ma);else{R.touch0=ma;var ya=!0}}if(ea&&(ea=clearTimeout(ea),!R.touch1)){R.end();(ma=Ra(this).on("dblclick.zoom"))&&ma.apply(this,
  arguments);return}ya&&(ea=setTimeout(function(){ea=null},500),Ub(this),R.start())}}function S(){var R=z(this,arguments),Z=d3.event.changedTouches,fa=Z.length,ja;Jd();ea&&(ea=clearTimeout(ea));for(ja=0;ja<fa;++ja){var qa=Z[ja];var ma=Vd(this,Z,qa.identifier);R.touch0&&R.touch0[2]===qa.identifier?R.touch0[0]=ma:R.touch1&&R.touch1[2]===qa.identifier&&(R.touch1[0]=ma)}qa=R.that.__zoom;if(R.touch1){ma=R.touch0[0];Z=R.touch0[1];ja=R.touch1[0];fa=R.touch1[1];var ya=(ya=ja[0]-ma[0])*ya+(ya=ja[1]-ma[1])*ya;
  var bb=(bb=fa[0]-Z[0])*bb+(bb=fa[1]-Z[1])*bb;qa=n(qa,Math.sqrt(ya/bb));ma=[(ma[0]+ja[0])/2,(ma[1]+ja[1])/2];ya=[(Z[0]+fa[0])/2,(Z[1]+fa[1])/2]}else if(R.touch0)ma=R.touch0[0],ya=R.touch0[1];else return;R.zoom("touch",L(u(qa,ma,ya),R.extent,Y))}function E(){var R=z(this,arguments),Z=d3.event.changedTouches,fa=Z.length,ja;d3.event.stopImmediatePropagation();la&&clearTimeout(la);la=setTimeout(function(){la=null},500);for(ja=0;ja<fa;++ja){var qa=Z[ja];R.touch0&&R.touch0[2]===qa.identifier?delete R.touch0:
  R.touch1&&R.touch1[2]===qa.identifier&&delete R.touch1}R.touch1&&!R.touch0&&(R.touch0=R.touch1,delete R.touch1);R.touch0?R.touch0[1]=this.__zoom.invert(R.touch0[0]):R.end()}var K=Os,H=Ps,L=Ss,U=Qs,M=Rs,X=[0,Infinity],Y=[[-Infinity,-Infinity],[Infinity,Infinity]],W=250,ba=Qi,aa=[],ha=Ob("start","zoom","end"),ea,la,pa=0;f.transform=function(R,Z){var fa=R.selection?R.selection():R;fa.property("__zoom",fm);R!==fa?t(R,Z):fa.interrupt().each(function(){z(this,arguments).start().zoom(null,"function"===typeof Z?
  Z.apply(this,arguments):Z).end()})};f.scaleBy=function(R,Z){f.scaleTo(R,function(){var fa=this.__zoom.k,ja="function"===typeof Z?Z.apply(this,arguments):Z;return fa*ja})};f.scaleTo=function(R,Z){f.transform(R,function(){var fa=H.apply(this,arguments),ja=this.__zoom,qa=r(fa),ma=ja.invert(qa),ya="function"===typeof Z?Z.apply(this,arguments):Z;return L(u(n(ja,ya),qa,ma),fa,Y)})};f.translateBy=function(R,Z,fa){f.transform(R,function(){return L(this.__zoom.translate("function"===typeof Z?Z.apply(this,
  arguments):Z,"function"===typeof fa?fa.apply(this,arguments):fa),H.apply(this,arguments),Y)})};f.translateTo=function(R,Z,fa){f.transform(R,function(){var ja=H.apply(this,arguments),qa=this.__zoom,ma=r(ja);return L(pf.translate(ma[0],ma[1]).scale(qa.k).translate("function"===typeof Z?-Z.apply(this,arguments):-Z,"function"===typeof fa?-fa.apply(this,arguments):-fa),ja,Y)})};D.prototype={start:function(){1===++this.active&&(this.index=aa.push(this)-1,this.emit("start"));return this},zoom:function(R,
  Z){this.mouse&&"mouse"!==R&&(this.mouse[1]=Z.invert(this.mouse[0]));this.touch0&&"touch"!==R&&(this.touch0[1]=Z.invert(this.touch0[0]));this.touch1&&"touch"!==R&&(this.touch1[1]=Z.invert(this.touch1[0]));this.that.__zoom=Z;this.emit("zoom");return this},end:function(){0===--this.active&&(aa.splice(this.index,1),this.index=-1,this.emit("end"));return this},emit:function(R){Qc(new Ns(f,R,this.that.__zoom),ha.apply,ha,[R,this.that,this.args])}};f.wheelDelta=function(R){return arguments.length?(U="function"===
  typeof R?R:of(+R),f):U};f.filter=function(R){return arguments.length?(K="function"===typeof R?R:of(!!R),f):K};f.touchable=function(R){return arguments.length?(M="function"===typeof R?R:of(!!R),f):M};f.extent=function(R){return arguments.length?(H="function"===typeof R?R:of([[+R[0][0],+R[0][1]],[+R[1][0],+R[1][1]]]),f):H};f.scaleExtent=function(R){return arguments.length?(X[0]=+R[0],X[1]=+R[1],f):[X[0],X[1]]};f.translateExtent=function(R){return arguments.length?(Y[0][0]=+R[0][0],Y[1][0]=+R[1][0],
  Y[0][1]=+R[0][1],Y[1][1]=+R[1][1],f):[[Y[0][0],Y[0][1]],[Y[1][0],Y[1][1]]]};f.constrain=function(R){return arguments.length?(L=R,f):L};f.duration=function(R){return arguments.length?(W=+R,f):W};f.interpolate=function(R){return arguments.length?(ba=R,f):ba};f.on=function(){var R=ha.on.apply(ha,arguments);return R===ha?f:R};f.clickDistance=function(R){return arguments.length?(pa=(R=+R)*R,f):Math.sqrt(pa)};return f};d3.zoomTransform=em;d3.zoomIdentity=pf;Ua.svg=Ua.svg;Ua.xhtml=Ua.xhtml;Ua.xlink=Ua.xlink;
  Ua.xml=Ua.xml;Ua.xmlns=Ua.xmlns})();