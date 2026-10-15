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
const html = fs.readFileSync(new URL('../public/graph_arch_preview/index.html', import.meta.url), 'utf8');
const source = html.slice(html.indexOf('function draftPairInfo(){'), html.indexOf('function publishMappingView(){'));
const tensor = id => ({id, displayLabel: id});
const initial = () => [
  {reference: tensor('r1'), target: tensor('t1'), origin: 'automatic', anchors: ['project']},
  {reference: tensor('r2'), target: tensor('t2'), origin: 'automatic', anchors: ['combine']},
  {reference: tensor('r3'), target: tensor('t3'), origin: 'automatic', anchors: []},
];
function setup(side = 'reference') {
  const list = initial();
  const state = {context: {store: 'scope'}, draftSide: side, selected: {reference: tensor('r1'), target: tensor('t2')}, selectionBeforeDraft: {reference: tensor('r3'), target: tensor('t3')}, pairs: new Map([['scope', list]])};
  const sandbox = {pairableMappingNode: (g,n)=>!n.id.startsWith('boundary:'),mappingMetricPreview: {remove() {}, compute() {}}, mappingState: state, currentMappingPairs: () => list, renderMappingDetails() {}, syncMapping() {}};
  vm.createContext(sandbox);vm.runInContext(source, sandbox);
  return {list, state, sandbox};
}
{
  const {list, state, sandbox} = setup();
  const before = JSON.stringify(list);
  sandbox.cancelDraftPair();
  assert.equal(JSON.stringify(list), before);
  assert.equal(state.selected.reference, null);
  assert.equal(state.selected.target, null);
  assert.equal(state.draftSide, null);
}
for (const side of ['reference', 'target']) {
  const {list, state, sandbox} = setup(side);
  assert.equal(sandbox.draftPairInfo().conflicts, 2);
  sandbox.confirmDraftPair();
  assert.equal(state.pairs.get('scope'), list, 'shared runtime pairs keep the same backing array');
  assert.equal(list.length, 2);
  assert.equal(list[0].reference.id, 'r3', 'unrelated pairs survive');
  assert.equal(list[1].reference.id, 'r1');
  assert.equal(list[1].target.id, 't2');
  assert.equal(list[1].anchors[0], side === 'reference' ? 'project' : 'combine');
  assert.equal(list[1].origin, 'manual');
  assert.equal(state.draftSide, null);
}
{
  const {list, state, sandbox} = setup();
  state.selected.target = tensor('t1');
  const before = JSON.stringify(list);
  sandbox.confirmDraftPair();
  assert.equal(list.length, 3, 'keeping an existing pair makes no duplicate');
  assert.equal(list.filter(p => p.reference.id === 'r1' && p.target.id === 't1').length, 1);
  assert.equal(JSON.stringify(list), before, 'selecting the existing counterpart preserves its metric');
}
{
  const {list, state, sandbox} = setup();
  state.selected.target = null;
  const before = JSON.stringify(list);
  sandbox.confirmDraftPair();
  assert.equal(JSON.stringify(list), before, 'a partial draft cannot modify pairs');
}
{
  const {state, sandbox} = setup();
  const graph = (id, nodes) => ({id, nodes:nodes.map(id=>({id,label:'op',outputsMetadata:[{id:'0'}]}))});
  state.context.scope='layer';state.context.configurationScope='attention';
  state.graphs={reference:graph('ref',['a','b']),target:graph('tar',['a','b'])};
  const pair=(r,t)=>({reference:{...tensor('ref::'+r+'::0'),node:r},target:{...tensor('tar::'+t+'::0'),node:t},anchors:[]});
  const a=pair('a','a'),b=pair('b','b'),outside=pair('c','c');
  a.previewMetricState='missing';
  const shared=[a,b,outside],removed=[];
  state.pairs.set('scope',shared);
  sandbox.mappingDefaultPairs=new Map();sandbox.structuredClone=structuredClone;
  sandbox.currentMappingPairs=()=>shared.filter(p=>['a','b'].includes(p.reference.node)&&['a','b'].includes(p.target.node));
  sandbox.mappingMetricPreview.remove=p=>removed.push(p);
  sandbox.rememberMappingDefaults();
  assert.equal(sandbox.canResetMappingDefaults(),false);
  shared.splice(0,2,pair('a','b'));
  assert.equal(sandbox.canResetMappingDefaults(),true);
  sandbox.resetMappingDefaults();
  assert.equal(state.pairs.get('scope'),shared,'reset preserves shared array');
  assert.equal(shared.find(p=>p.reference.node==='c'),outside,'outside scope unchanged');
  assert.equal(shared.find(p=>p.reference.node==='a').target.node,'a');
  assert.equal(shared.find(p=>p.reference.node==='a').previewMetricState,'missing');
  assert.equal(shared.find(p=>p.reference.node==='b').target.node,'b');
  assert.equal(removed.length,1,'replaced pair metric and timer invalidated');
  assert.equal(sandbox.canResetMappingDefaults(),false);
  assert.equal(state.draftSide,null);
  assert.equal(state.selected.reference,null);
}
console.log('PASS: draft cancellation, partial draft, existing pair, conflict replacement in both directions, semantic anchors and shared pair storage.');

// Native expand/collapse handles mousedown before our pointerup. Layout changes
// must not turn that control/node gesture into a blank-canvas cancellation.
{
  const pane=fs.readFileSync(new URL('../public/graph_arch_preview/mapping_pane.js',import.meta.url),'utf8');
  const handlers={},sent=[];
  let geo={rect:{left:0,top:0},worldTransform:{x:0,y:0,scale:1},nodes:[{x:20,y:20,width:60,height:30}]};
  const host={addEventListener(type,fn){handlers[type]=fn;}};
  const sandbox={$:()=>host,ready:true,e:{readPairGeometry:()=>geo},send:message=>sent.push(message)};
  vm.createContext(sandbox);
  vm.runInContext(pane.slice(pane.indexOf('// A blank click'),pane.indexOf('function resetPaneView')),sandbox);
  const event=(x,y,control=false)=>({button:0,pointerId:1,clientX:x,clientY:y,composedPath:()=>control?[{matches:()=>true}]:[]});
  const click=(down,up=down,change)=>{sent.length=0;handlers.pointerdown(down);change?.();handlers.pointerup(up);return sent.length;};
  assert.equal(click(event(30,30),event(30,30),()=>{geo={...geo,nodes:[]};}),0,'collapse moving a node away preserves tensor selection');
  assert.equal(click(event(10,10,true)),0,'native control outside node bounds preserves selection');
  assert.equal(click(event(150,150)),1,'a genuine blank click cancels');
  assert.equal(click(event(150,150),event(180,150)),0,'pan preserves selection');
  sent.length=0;handlers.pointerdown(event(150,150));handlers.pointercancel();handlers.pointerup(event(150,150));
  assert.equal(sent.length,0,'cancelled pointer never clears tensor selection');
  console.log('PASS: native controls, layout changes, blank click, pan and pointer cancellation.');
}
