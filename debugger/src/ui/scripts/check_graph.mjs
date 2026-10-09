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

import {build} from 'esbuild';
import {mkdtemp, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';

// Exercise the production architecture topology over every layer of the bundled
// example.
const semantic = fileURLToPath(
    new URL('../../../examples/gemma4-e2b/semantic.json', import.meta.url));
const root = fileURLToPath(new URL('..', import.meta.url));
const dir = await mkdtemp(join(tmpdir(), 'graph-test-'));
try {
  await build({
    stdin: {
      contents: `
    import assert from 'node:assert/strict';
    import fs from 'node:fs';
    import {buildArchitectureTopology, architectureMetric} from './src/features/graph/architecture_view/architecture_renderer';
    const model = JSON.parse(fs.readFileSync(${
          JSON.stringify(semantic)}, 'utf8'));
    let anchors = 0;
    for (let layer = 0; layer < model.layers.length; layer++) {
      const topology = buildArchitectureTopology(model, layer);
      assert(topology, 'layer ' + layer + ' has a topology');
      const ids = new Set(topology.nodes.map((node) => node.id));
      assert.equal(ids.size, topology.nodes.length, 'node ids are unique');
      assert(topology.nodes.length >= 10, 'layer ' + layer + ' renders its semantic nodes');
      assert(topology.nodes.every((node) => Number.isFinite(node.width) && Number.isFinite(node.height) && node.width > 0 && node.height > 0));
      assert(topology.edges.every((edge) => ids.has(edge.source) && ids.has(edge.target)), 'edges reference known nodes');
      assert(topology.edges.every((edge) => edge.source !== edge.target), 'no self edges');
      anchors += topology.nodes.filter((node) => node.kind === 'tensor').length;
    }
    assert(anchors > 0, 'anchored tensors appear as tensor nodes');
    assert.equal(buildArchitectureTopology(model, model.layers.length), null, 'out-of-range layer has no topology');
    assert.equal(architectureMetric(undefined, 'CosSim'), null);
    console.log('PASS: ' + model.layers.length + ' layer topologies from the live renderer: unique nodes, finite sizes, resolved edges, tensor anchors, out-of-range layer');
  `,
      resolveDir: root,
      loader: 'ts',
    },
    bundle: true,
    platform: 'node',
    format: 'esm',
    outfile: join(dir, 'test.mjs'),
  });
  await import(pathToFileURL(join(dir, 'test.mjs')).href);
} finally {
  await rm(dir, {recursive: true, force: true});
}
