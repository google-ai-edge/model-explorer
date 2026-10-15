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
const base=new URL('../public/graph_arch_preview/',import.meta.url);
// PLE entry scopes follow the two confirmed execution boundaries.
const html=fs.readFileSync(new URL('index.html',base),'utf8');
const catalog={window:{}};
vm.runInNewContext(fs.readFileSync(new URL('mapping_catalog.js',base),'utf8'),catalog);
const contexts={window:catalog.window,isModuleArch:()=>true,overviewState:{page:'ple_preparation',index:0},executionStage:'prefill',overviewTitle:()=> 'Per-Layer Embedding',overviewData:()=>({blocks:[],nodes:[{id:'lookup',mappingRegion:'lookup'},{id:'project',mappingRegion:'projection'},{id:'combine',mappingRegion:'projection'}]})};
vm.createContext(contexts);
vm.runInContext(html.slice(html.indexOf('function mappingContext('),html.indexOf('function mappingTensor(')),contexts);
for(const stage of ['prefill','decode']){
 contexts.executionStage=stage;
 const [lookup,project,combine]=['lookup','project','combine'].map(id=>contexts.mappingContext(id));
 assert.notEqual(lookup.store,project.store,'lookup has its own signature and pair scope');
 assert.equal(project.store,combine.store,'projection and combine share one execution range');
 assert.equal(contexts.runtimeOptions(lookup)[0].key,'c1-main-ple_preparation');
 for(const c of [project,combine])assert(contexts.runtimeOptions(c).every(o=>o.mappingRegion==='projection'&&o.stage===stage));
}
console.log('PASS: independent lookup signature; shared Projection/Combine scope in the selected Prefill/Decode stage.');

const ui={window:{ARCH_LAYOUTS:{}}};
vm.runInNewContext(fs.readFileSync(new URL('semantic_graph.js',base),'utf8'),ui);
vm.runInNewContext(fs.readFileSync(new URL('ple_arch.js',base),'utf8'),ui);
const page=ui.window.SEMANTIC_MODEL.arch_graphs.ple_preparation;
const layout=ui.window.ARCH_LAYOUTS.ple_preparation.collapsed;
assert.equal(page.blocks.length,2);
assert.equal(layout.nodes.filter(n=>n.kind==='boundary'&&n.id==='input').length,1);
const inputEdges=layout.edges.filter(e=>e.from==='input');
assert.equal(inputEdges.length,2);
assert.equal(new Set(inputEdges.map(e=>e.port)).size,2,'input collection retains distinct token and embedding ports');
assert.equal(new Set(inputEdges.map(e=>e.to)).size,2,'inputs feed different execution blocks');
assert.deepEqual(Array.from(layout.tensors,t=>t.source),['input_ids','main_embeddings','lookup','combine']);
const ys=layout.nodes.map(n=>layout.positions[n.id].y);
assert(ys.every((y,i)=>i===0||y>ys[i-1]),'modules follow a vertical flow');
assert.equal(page.blocks[0].mappingAnchor,'lookup');
assert.equal(page.blocks[1].mappingAnchor,'combine');
console.log('PASS: vertical two-block PLE Arch, one Inputs node with two ports, lookup and final outputs.');
