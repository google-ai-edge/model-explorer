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
const html=fs.readFileSync(new URL('index.html',base),'utf8');
const sandbox={window:{},setTimeout,clearTimeout};vm.createContext(sandbox);
for(const file of ['ple_mapping.js','mapping_metrics.js'])vm.runInContext(fs.readFileSync(new URL(file,base),'utf8'),sandbox);
vm.runInContext(html.slice(html.indexOf('function mappingTensor('),html.indexOf('const automaticPairScopes=')),sandbox);
vm.runInContext(html.slice(html.indexOf('const loadedPleGraphs='),html.indexOf('function archTensorSources(')),sandbox);
const samples=Array.from({length:101},(_,i)=>({value:i/100,text:(i/100).toFixed(6),bgColor:`bg${i}`,textColor:`fg${i}`}));
sandbox.analysisContext={summary:'max'};
sandbox.mappingMetricPreview=sandbox.window.createMappingMetricPreview();
sandbox.mappingMetricPreview.setContext('prefill:0:cosine',samples);
sandbox.mappingState={pairs:new Map()};
let currentGraph;
sandbox.mappingContext=()=>({store:currentGraph.id});
for(const [name,region,anchor,inputs,count,internal] of [
 ['c1_main','lookup','lookup',1,43,41],
 ['c8_prefill_1024','projection','combine',2,10,7],
 ['c8_decode','projection','combine',2,10,7],
]){
 const graph=sandbox.window.pleMapping.prepare(JSON.parse(fs.readFileSync(new URL(`mapping_data/${name}_ple_preparation.json`,base),'utf8')),region);
 assert.equal(graph.mappingPresentation.nativeStyle,true);
 assert.equal(graph.mappingPresentation.expandedGroups,undefined,'native expansion is not overridden');
 for(const binding of Object.values(graph.mappingPresentation.anchors))assert.equal(binding.expandedGroups,undefined);
 for(const node of graph.nodes){const source=node.attrs?.find(a=>a.key==='source_namespace')?.value;if(source!=null)assert.equal(node.namespace,source,'restore exact runtime namespace');}
 currentGraph=graph;sandbox.graph=graph;
 vm.runInContext('loadedPleGraphs.set(graph.id,graph)',sandbox);
 const pairs=sandbox.identicalGraphPairs(graph,graph,{anchor});sandbox.mappingState.pairs.set(graph.id,pairs);
 assert.equal(pairs.length,count,'every runtime output and input defaults to paired');
 assert.equal(pairs.filter(p=>p.reference.node==='boundary:inputs').length,inputs);
 const bound=sandbox.pleArchRecords(anchor)[0].pair;
 assert(bound,'Arch output resolves to a runtime pair');
 const metric=sandbox.pleArchMetric(anchor,false),runtime=sandbox.mappingMetricPreview.read(bound);
 for(const field of ['value','text','bg','fg'])assert.equal(metric[field],runtime[field],`${name}: shared ${field}`);
 const records=sandbox.pleArchRecords(anchor,true);
 assert.equal(records.length,internal);
 assert(records.every(r=>r.reference.node!=='boundary:inputs'&&r.reference.id!==bound.reference.id));
 const values=records.map(r=>sandbox.mappingMetricPreview.read(r.pair)).filter(m=>m.state==='ready').map(m=>m.value);
 assert.equal(sandbox.pleArchMetric(anchor,true).value,Math.max(...values));
 sandbox.analysisContext.summary='min';assert.equal(sandbox.pleArchMetric(anchor,true).value,Math.min(...values));sandbox.analysisContext.summary='max';
 const before=pairs.indexOf(bound);pairs.splice(before,1);
 assert.equal(sandbox.pleArchMetric(anchor,false).value,null,'unpaired output never falls back to fabricated Arch metric');
 const replacement={...bound,target:{...bound.target,id:bound.target.id+':edited'},anchors:[]};pairs.push(replacement);
 assert.equal(sandbox.pleArchRecords(anchor)[0].pair,replacement,'binding follows the reference port after editing, independent of anchor copy');
 assert.equal(sandbox.pleArchMetric(anchor,false).value,sandbox.mappingMetricPreview.read(replacement).value);
 if(region==='lookup')assert.equal(sandbox.pleArchMetric('input_ids',false).value,null,'integer token IDs are paired without fabricated similarity');
 else assert.deepEqual(Array.from(sandbox.pleArchRecords('main_embeddings')[0].reference.shape),name.includes('decode')?['1','1','1536']:['1','1024','1536']);
}
const pending=sandbox.window.pleMapping.aggregate([{state:'ready',value:.5},{state:'pending',value:null}],'max');assert.equal(pending.state,'pending');assert.equal(pending.value,null);
console.log('PASS: PLE shared runtime bindings, all default inputs/pairs, output value/color identity, complete internal Min/Max, missing and edited pairs, pending aggregation, Prefill/Decode shapes.');
