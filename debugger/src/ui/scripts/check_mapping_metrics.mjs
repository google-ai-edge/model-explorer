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
const tasks=[],cleared=new Set();let changes=0;
const sandbox={window:{},setTimeout,clearTimeout};
vm.runInNewContext(fs.readFileSync(new URL('../public/graph_arch_preview/mapping_metrics.js',import.meta.url),'utf8'),sandbox);
const engine=sandbox.window.createMappingMetricPreview({onChange:()=>changes++,schedule:fn=>(tasks.push(fn),tasks.length-1),cancel:id=>cleared.add(id)});
const pair=(r,t,state)=>({reference:{id:r},target:{id:t},previewMetricState:state});
const samples=[{value:0,text:'0.000',bgColor:'#abc',textColor:'#123'}];
engine.setContext('metric-a/token-0',samples);
const ready=pair('r0','t0'),missing=pair('r1','t1','missing'),pending=pair('r2','t2','pending');
assert.equal(engine.read(ready).value,0);assert.equal(engine.read(ready).label,'Paired · Preview metric');assert.equal(engine.read(missing).state,'missing');
assert.equal(engine.read(pending).state,'pending');tasks.at(-1)();assert.equal(engine.read(pending).state,'ready');
assert.equal(changes,1);
engine.compute(ready);const obsoletePair=tasks.at(-1);engine.remove(ready);obsoletePair();assert.equal(changes,1,'removed pair cannot publish');
engine.compute(ready);const obsoleteContext=tasks.at(-1);engine.setContext('metric-b/token-1',[{...samples[0],value:2,text:'2.000'}]);
assert.equal(engine.read(ready).state,'pending');obsoleteContext();assert.equal(changes,1,'old metric/token context cannot publish');
tasks.at(-1)();assert.equal(engine.read(ready).text,'2.000');
engine.compute(ready);const obsoleteRevision=tasks.at(-1);engine.compute(ready);obsoleteRevision();assert.equal(engine.read(ready).state,'pending','re-pairing the same IDs invalidates the prior calculation');tasks.at(-1)();assert.equal(engine.read(ready).state,'ready');
engine.setContext('metric-b/token-1',[{...samples[0],value:2,text:'2.000',bgColor:'#def'}]);assert.equal(engine.read(ready).bg,'#def','theme update preserves ready state and updates colors');
console.log('PASS: missing/pending/ready/zero, deferred completion, removed pairs, obsolete context, repeated pair revisions and theme refresh.');
