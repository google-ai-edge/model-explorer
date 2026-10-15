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

import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('../public/graph_arch_preview/',import.meta.url));
const html=fs.readFileSync(root+'index.html','utf8');
const context={};vm.createContext(context);
for(const [start,end] of [['function mappingTensor(', 'const automaticPairScopes'],['function scopeLayerMapping(', 'function currentMappingPairs(']])vm.runInContext(html.slice(html.indexOf(start),html.indexOf(end)),context);
const sandbox={window:{}};vm.runInNewContext(fs.readFileSync(root+'semantic_graph.js','utf8'),sandbox);
const semantic=sandbox.window.SEMANTIC_MODEL;
let files=0,anchors=0,pairs=0;
function checkEdges(graph){const nodes=new Map(graph.nodes.map(n=>[n.id,n]));assert.equal(nodes.size,graph.nodes.length);for(const n of graph.nodes)for(const e of n.incomingEdges||[])assert.ok(nodes.get(e.sourceNodeId)?.outputsMetadata?.some(p=>p.id===e.sourceNodeOutputId),`${graph.id}: missing source ${e.sourceNodeId}:${e.sourceNodeOutputId}`);}
for(const file of fs.readdirSync(root+'mapping_data').filter(f=>/decoder_layer_\d+\.json$/.test(f))){
 const graph=JSON.parse(fs.readFileSync(root+'mapping_data/'+file));checkEdges(graph);
 const layer=Number(file.match(/layer_(\d+)/)[1]),definition=semantic.semantic_graph[semantic.layers[layer].def];
 const bindings=graph.mappingPresentation.anchors;
 const partial=graph.stage==='prefill'&&layer===14;
 if(!partial)assert.equal(Object.keys(bindings).length,definition.nodes.length,file);
 else assert.ok(!bindings.output,'Partial export must not invent a layer output');
 const automatic=context.identicalGraphPairs(graph,graph,{anchor:'output'});
 assert.equal(new Set(automatic.map(p=>p.reference.id)).size,automatic.length);
 for(const [anchor,binding] of Object.entries(bindings)){
  assert.ok(graph.nodes.find(n=>n.id===binding.outputNode)?.outputsMetadata.some(p=>p.id===binding.outputPort));
  assert.equal(automatic.filter(p=>p.anchors.includes(anchor)).length,1,file+' '+anchor);
  anchors++;
 }
 for(const scope of ['attention','ffn','ple']){
  const scoped=context.scopeLayerMapping(graph,{key:'decoder_layer',configurationScope:scope});checkEdges(scoped);
  assert.ok(scoped.nodes.length<graph.nodes.length,`${file}: ${scope} must not fall back to the whole layer`);
  const expected=definition.nodes.filter(n=>n.namespace===scope).map(n=>n.attrs.find(a=>a.key==='semantic_output').value);
  const complete=expected.every(a=>bindings[a]);
  if(complete)assert.deepEqual(Object.keys(scoped.mappingPresentation.anchors).sort(),[...expected].sort(),`${file}: ${scope} hover and Mapping must cover the same outputs`);
  else assert.equal(scoped.nodes.length,0,`${file}: unavailable range must be explicit`);
 }

 assert.equal(context.identicalGraphPairs(graph,{...graph,id:'other-runtime'},{anchor:'output'}).length,0,'Different graph identity must not be auto-paired');
 assert.equal(context.identicalGraphPairs(graph,{...graph,nodes:graph.nodes.slice(1)},{anchor:'output'}).length,0,'Different topology must not be auto-paired');
 files++;pairs+=automatic.length;
}
assert.equal(files,65);
console.log(`PASS: ${files} runtime regions, ${anchors} verified output anchors, ${pairs} identity pairs; boundaries, scope filtering, partial coverage and non-identical graphs checked.`);
