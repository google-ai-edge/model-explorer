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

(async()=>{'use strict';
const side=new URLSearchParams(location.search).get('side'),channel='mapping-pane-v1',$=id=>document.getElementById(id),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let candidateSide=null,drafting=false,pickerState=null,pickerContent='';
let revision='',e=null,raw={nodes:[],edges:[],tensors:{}},ready=false,pairs=[],chosen={reference:null,target:null},palette={},theme='light',lastGeometry='',mountId=0,entryAnchor='';
const send=d=>parent.postMessage({channel,side,revision,...d},location.origin);
window.modelExplorer.assetFilesBaseUrl='../graph_execution/upstream/model_explorer/dist/static_files';window.modelExplorer.workerScriptPath='../graph_execution/upstream/model_explorer/dist/worker.js';
const canConfigure=t=>t&&!t.boundary;
function paint(){document.documentElement.dataset.theme=theme;for(const [k,v] of Object.entries(palette))document.documentElement.style.setProperty('--'+k,v);if(ready){e.setPanePalette(palette,0);e.setAttribute('data-theme',theme);if(e.shadowRoot&&!e.shadowRoot.querySelector('#mapping-theme')){const link=document.createElement('link');link.id='mapping-theme';link.rel='stylesheet';link.href='native_model_theme.css?v=86';e.shadowRoot.append(link);}for(const [k,v] of Object.entries(palette))e.style.setProperty('--'+k,v);}lastGeometry='';renderOverlay();}
function prepare(graph){
 const tensorMap=new Map(),tensors={},nodes=[],edges=[];let slot=0;
 for(const n of graph.nodes){const outputs=[];for(const m of n.outputsMetadata||[]){const attrs=Object.fromEntries((m.attrs||[]).map(a=>[a.key,a.value])),i=slot++,shape=String(attrs.tensor_shape||attrs.shape||''),match=shape.match(/^([^\[]*)\[(.*)\]$/),label=attrs.tensor_index!=null?'t'+attrs.tensor_index:n.id+':'+m.id;
  tensors[i]={id:graph.id+'::'+n.id+'::'+m.id,displayLabel:label,name:attrs.tensor_name||attrs.name||label,dtype:match?.[1]||'unknown',shape:match?match[2].split(','):shape?[shape]:[],sourceGraph:n.sourceGraph||graph.sourceGraph||graph.id,node:n.id,port:m.id,boundary:n.label==='pseudo_const'||((n.id.startsWith('boundary:')||n.label==='GraphInputs')&&!graph.mappingPresentation?.pairInputs)};tensorMap.set(n.id+'\0'+m.id,i);outputs.push(i);}
  nodes.push({id:n.id,outputs});
 }
 for(const n of graph.nodes)for(const edge of n.incomingEdges||[]){const tensor=tensorMap.get(edge.sourceNodeId+'\0'+edge.sourceNodeOutputId);if(tensor!=null)edges.push({from:edge.sourceNodeId,to:n.id,tensor});}
 raw={nodes,edges,tensors};return graph.nodes.map(n=>({...n,id:side+'-op-'+n.id,subgraphIds:[],incomingEdges:(n.incomingEdges||[]).map(e=>({...e,sourceNodeId:side+'-op-'+e.sourceNodeId}))}));
}
async function mount(graph){const generation=++mountId;ready=false;lastGeometry='';$('host').replaceChildren();$('tensor-overlay').replaceChildren();closePicker();
 if(!graph){$('loading').textContent='No execution graph selected';$('loading').hidden=false;return;}
 $('loading').textContent='Loading Model Explorer…';$('loading').hidden=false;await customElements.whenDefined('model-explorer-visualizer');if(generation!==mountId)return;
 const entry=graph.mappingPresentation?.anchors?.[entryAnchor];
 const element=document.createElement('model-explorer-visualizer');element.classList.toggle('native-ple',!!graph.mappingPresentation?.nativeStyle);e=element;const nodes=prepare(graph);element.graphCollections=[{label:graph.runtime||'Execution graph',graphs:[{id:graph.id,nodes}]}];element.initialUiState={paneStates:[{deepestExpandedGroupNodeIds:(graph.mappingPresentation?.expandedGroups||entry?.expandedGroups||[]).map(id=>id+'___group___'),selectedNodeId:'',selectedGraphId:graph.id,selectedCollectionLabel:graph.runtime||'Execution graph',widthFraction:1,selected:true,flattenLayers:false}]};element.config={hideTitleBar:true,hideToolBar:false,hideInfoPanel:true,hideLegends:true,...(graph.mappingPresentation?.nativeStyle?{}:{keepLayersWithASingleChild:!!graph.mappingPresentation,viewOnNodeConfig:{hideLayerNodeChildrenCount:true,hideLayerNodeDescendantsCount:true}})};
 element.addEventListener('modelGraphProcessed',()=>{if(generation!==mountId)return;ready=true;$('loading').hidden=true;paint();let frames=0;function fit(){if(generation!==mountId)return;if(++frames<24){requestAnimationFrame(fit);return;}paint();configureNativeToolbar(element);resetPaneView();}requestAnimationFrame(fit);send({type:'ready'});});element.addEventListener('pairGeometryRendered',renderOverlay);element.addEventListener('pointerdown',()=>{closePicker();});$('host').append(element);const css=document.createElement('style');css.textContent='.pane-title-container,.sync-navigation-container{display:none!important}.pane-container,.renderer-container{background:var(--bg)!important;color:var(--text)!important}.pane-container{border:0!important}';element.shadowRoot.append(css);const themeLink=document.createElement('link');themeLink.rel='stylesheet';themeLink.href='native_model_theme.css?v=86';element.shadowRoot.append(themeLink);const toolbarStyle=document.createElement('link');toolbarStyle.rel='stylesheet';toolbarStyle.href='mapping_toolbar.css?v=3';element.shadowRoot.append(toolbarStyle);
}
const glyph=paired=>paired?'<path class="cube-body" d="M6 .7L11 3.3v5.4L6 11.3 1 8.7V3.3z"/><path class="cube-seams" d="M1 3.3L6 6l5-2.7M6 6v5.3"/>':'<rect x="1" y="1" width="10" height="10" rx="1" stroke-dasharray="2 1.5"/>';
const pairActionIcon='<span class="pair-action-icon" aria-hidden="true"><svg viewBox="0 0 16 16"><path d="M6.4 5.2l1.9-1.9a3 3 0 014.3 4.3l-1.9 1.9M9.6 10.8l-1.9 1.9a3 3 0 01-4.3-4.3l1.9-1.9M5.8 10.2l4.4-4.4"/></svg></span>';
function candidateState(side,ids){const other=side==='reference'?'target':'reference',eligible=ids.filter(id=>canConfigure(raw.tensors[id])),candidate=candidateSide===side&&eligible.length>0;const current=id=>pairs.some(p=>p[other]===chosen[other]&&p[side]===raw.tensors[id].id);return {candidate,currentMatch:candidate&&eligible.some(current),pairAction:candidate&&eligible.some(id=>!current(id))};}
function tensorMetric(id){return pairs.find(p=>p[side]===raw.tensors[id]?.id)?.metric||{state:'unpaired',text:'',label:'Unpaired'};}
function metricMarkup(metric){
 if(metric.state==='unpaired')return '';
 if(metric.state==='pending')return '<span class="metric-spinner" aria-label="Calculating metric"></span>';
 return `<span class="mapping-metric-value" style="${metric.state==='ready'?`background:${esc(metric.bg)};color:${esc(metric.fg)}`:''}" title="${esc(metric.label)}">${esc(metric.text)}</span>`;
}
function tensorSourceLabel(t){if(t.name===t.displayLabel)return 'Tensor output';const parts=t.name.split('/').filter(Boolean);return parts.slice(-2).join(' / ')||'Tensor output';}
function closePicker(restoreFocus=false){const key=pickerState?.key;pickerState=null;pickerContent='';$('tensor-picker').hidden=true;for(const tag of $('tensor-overlay').querySelectorAll('[aria-expanded]'))tag.setAttribute('aria-expanded','false');if(restoreFocus&&key)$('tensor-overlay').querySelector(`button[data-tensors="${key}"]`)?.focus({preventScroll:true});}
function renderPicker(){
 if(!pickerState)return;
 const picker=$('tensor-picker'),{side,ids,key}=pickerState;
 const tag=$('tensor-overlay').querySelector(`button[data-tensors="${key}"]`);
 if(tag){pickerState.anchor=tag.getBoundingClientRect();tag.setAttribute('aria-expanded','true');}
 const content=ids.map(id=>{
  const t=raw.tensors[id],paired=pairs.some(p=>p[side]===t.id),state=candidateState(side,[id]),selected=chosen[side]===t.id,metric=tensorMetric(id);
  return `<button class="${state.candidate?'candidate':''}${state.currentMatch?' current-match':''}${selected?' chosen':''}" data-pick-tensor="${id}" data-side="${side}" aria-pressed="${selected}" ${canConfigure(t)?'':'disabled'} title="${esc(t.name)}"><svg viewBox="0 0 12 12" aria-hidden="true">${glyph(paired)}</svg><span class="tensor-source"><span>${esc(tensorSourceLabel(t))}</span><small>${esc(t.dtype)} [${esc(t.shape.join(', '))}]${canConfigure(t)?'':' · Outside this configuration range'}</small></span>${metricMarkup(metric)}${state.currentMatch?'<span class="pair-status">Current match</span>':state.pairAction?pairActionIcon:!paired?'<span class="pair-status">Unpaired</span>':metric.state==='ready'||metric.state==='pending'?'':'<span class="pair-status">Paired</span>'}</button>`;
 }).join('');
 if(content!==pickerContent){const focus=picker.contains(document.activeElement)?document.activeElement.dataset.pickTensor:null,scroll=picker.scrollTop;picker.innerHTML=content;pickerContent=content;picker.scrollTop=scroll;if(focus)picker.querySelector(`[data-pick-tensor="${focus}"]`)?.focus({preventScroll:true});}
 picker.hidden=false;
 const anchor=pickerState.anchor;picker.style.left=Math.max(8,Math.min(innerWidth-picker.offsetWidth-8,anchor.left))+'px';picker.style.top=Math.max(8,Math.min(innerHeight-picker.offsetHeight-8,anchor.bottom+4))+'px';
}
function pick(side,t){const tensor=raw.tensors[t];if(!canConfigure(tensor))return;chosen[side]=tensor.id;send({type:'tensor',tensor});lastGeometry='';renderOverlay();}
function openPick(side,tensors,anchor){
 if(tensors.length===1){closePicker();pick(side,tensors[0]);return;}
 const key=tensors.join(',');
 if(pickerState?.key===key){closePicker(true);return;}
 pickerState={side,ids:tensors,key,anchor};pickerContent='';renderPicker();$('tensor-picker').querySelector('button:not([disabled])')?.focus({preventScroll:true});
}
$('tensor-overlay').addEventListener('click',ev=>{const tag=ev.target.closest('[data-tensors]');if(!tag)return;ev.stopPropagation();openPick(tag.dataset.side,tag.dataset.tensors.split(',').map(Number),tag.getBoundingClientRect());});
$('tensor-picker').addEventListener('click',ev=>{const b=ev.target.closest('[data-pick-tensor]');if(b)pick(b.dataset.side,Number(b.dataset.pickTensor));});
$('tensor-picker').addEventListener('keydown',ev=>{if(ev.key==='Escape')closePicker(true);});
function positionOnPath(points,fraction){let length=0;const lengths=[];for(let i=1;i<points.length;i++){const n=Math.hypot(points[i].x-points[i-1].x,points[i].y-points[i-1].y);length+=n;lengths.push(n);}let at=length*fraction;for(let i=1;i<points.length;i++){if(at<=lengths[i-1]){const t=lengths[i-1]?at/lengths[i-1]:0;return {x:points[i-1].x+(points[i].x-points[i-1].x)*t,y:points[i-1].y+(points[i].y-points[i-1].y)*t};}at-=lengths[i-1];}return points.at(-1);}
function overlayPane(side,geo,origin){const nodes=new Map(geo.nodes.map(n=>[n.id,n])),bundles=new Map();
 for(const edge of raw.edges){const from=geo.visible[side+'-op-'+edge.from],to=geo.visible[side+'-op-'+edge.to];if(!from||!to||from===to)continue;const key=from+'|'+to;if(!bundles.has(key))bundles.set(key,{from,to,sources:[],tensors:new Set()});bundles.get(key).tensors.add(edge.tensor);bundles.get(key).sources.push(edge);}
 const paths=[],labels=[],seen=new Set(),placed=[];
 function add(points,allTensors,terminal=false){if(points.length<2)return;const selected=allTensors.some(t=>chosen[side]===raw.tensors[t].id),d=points.map((p,i)=>`${i?'L':'M'}${p.x.toFixed(2)} ${p.y.toFixed(2)}`).join(' ');
 paths.push(`${terminal?`<path class="tensor-terminal" d="${d}"/>`:''}${selected?`<path class="tensor-highlight" d="${d}"/>`:''}<path class="tensor-hit" ${allTensors.some(t=>canConfigure(raw.tensors[t]))?`data-side="${side}" data-tensors="${allTensors.join(',')}"`:'style="pointer-events:none"'} d="${d}"/>`);
 const tensors=allTensors.filter(t=>!seen.has(t));if(!tensors.length)return;tensors.forEach(t=>seen.add(t));
 const isPaired=t=>pairs.some(p=>p[side]===raw.tensors[t].id),pairedCount=tensors.filter(isPaired).length,configured=pairedCount===tensors.length,partial=pairedCount>0&&!configured,multi=tensors.length>1,ratio=pairedCount+'/'+tensors.length,metric=multi?{state:'bundle'}:tensorMetric(tensors[0]),metricWidth=metric.state==='ready'?metric.text.length*7+12:metric.state==='pending'||metric.state==='missing'?20:0,w=(multi?ratio.length*6.1+40:24)+metricWidth,h=24;
 let point=positionOnPath(points,.4),left=point.x+8,top=point.y-h/2;
 for(const fraction of [.4,.6,.8,.25]){const candidate=positionOnPath(points,fraction),x=candidate.x+8,y=candidate.y-h/2;const overlaps=[...geo.nodes.map(n=>n.group&&n.expanded?{...n,height:22}:n),...placed].some(n=>x<n.x+n.width&&x+w>n.x&&y<n.y+n.height&&y+h>n.y);point=candidate;left=x;top=y;if(!overlaps)break;}
 placed.push({x:left,y:top,width:w,height:h});const title=`${multi?'':metric.label+'\n'}${pairedCount} of ${tensors.length} tensors paired\n\n`+tensors.map(t=>`${raw.tensors[t].displayLabel} · ${raw.tensors[t].dtype} [${raw.tensors[t].shape.join(', ')}]\n${raw.tensors[t].name}`).join('\n\n');
 const icon=multi?`<rect class="stack-back${configured?' stack-filled':' stack-empty'}" x="5" y="1" width="10" height="10" rx="1"/><rect class="stack-front${pairedCount?' stack-filled':' stack-empty'}" x="1" y="5" width="10" height="10" rx="1"/>`:glyph(configured);
 const {candidate,currentMatch,pairAction}=candidateState(side,tensors);
 const content=multi?`<span class="tensor-count" aria-hidden="true">${ratio}</span>`:'';
 labels.push(`<button class="tensor-tag metric-${metric.state}${candidate?' candidate':''}${currentMatch?' current-match':''}${drafting&&candidateSide!==side&&!selected?' subdued':''}${multi?' tensor-bundle':''}${configured?' paired':partial?' partial':''}${selected?' chosen':''}" data-side="${side}" data-metric-state="${metric.state}" data-tensors="${tensors.join(',')}" data-paired-count="${pairedCount}" data-total-count="${tensors.length}" aria-label="${side==='reference'?'Reference':'Target'} ${esc(multi?'tensor collection · '+tensors.length+' tensors':'tensor · '+tensorSourceLabel(raw.tensors[tensors[0]]))}" aria-description="${candidate?(currentMatch?'Current match. ':'Choose to pair. '):''}${pairedCount} of ${tensors.length} tensors paired" ${multi?`aria-haspopup="dialog" aria-controls="tensor-picker" aria-expanded="${pickerState?.key===tensors.join(',')}"`:''} ${tensors.some(id=>canConfigure(raw.tensors[id]))?'':'disabled'} title="${esc(title)}" style="left:${left}px;top:${top}px;${metric.state==='ready'?`--metric-bg:${esc(metric.bg)};--metric-fg:${esc(metric.fg)}`:''}"><svg class="tensor-symbol" viewBox="0 0 ${multi?16:12} ${multi?16:12}" aria-hidden="true">${icon}</svg>${content}${multi?'':metricMarkup(metric)}${pairAction?pairActionIcon:''}${multi?'<svg class="tensor-chevron" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 4.5L6 7.5L9 4.5"/></svg>':''}</button>`);
 }
 const routed=new Map();for(const bundle of bundles.values()){let edge=geo.edges.find(e=>e.from===bundle.from&&e.to===bundle.to);if(!edge){const candidates=[];for(const source of bundle.sources){const a=geo.ancestors[side+'-op-'+source.from]||[],b=geo.ancestors[side+'-op-'+source.to]||[];for(const e of geo.edges){const ai=a.indexOf(e.from),bi=b.indexOf(e.to);if(ai>=0&&bi>=0)candidates.push({edge:e,cost:ai+bi});}}candidates.sort((a,b)=>a.cost-b.cost);edge=candidates[0]?.edge;}if(edge){if(!routed.has(edge.id))routed.set(edge.id,{edge,tensors:new Set()});for(const t of bundle.tensors)routed.get(edge.id).tensors.add(t);}}
 for(const route of routed.values())add(route.edge.points,[...route.tensors]);
 for(const op of raw.nodes){const visible=geo.visible[side+'-op-'+op.id],n=nodes.get(visible);if(!n)continue;for(const t of op.outputs){if(raw.edges.some(e=>e.tensor===t)||seen.has(t))continue;if(visible!==side+'-op-'+op.id&&!raw.tensors[t].boundary)continue;add([{x:n.x+n.width/2,y:n.y+n.height},{x:n.x+n.width/2,y:n.y+n.height+30}],[t],true);}}
 return `<div class="tensor-pane" data-side="${side}" data-native-node-count="${geo.nodes.length}" style="left:${geo.rect.left-origin.left}px;top:${geo.rect.top-origin.top}px;width:${geo.rect.width}px;height:${geo.rect.height}px"><div class="tensor-world" style="transform:translate(${geo.worldTransform.x}px,${geo.worldTransform.y}px) scale(${geo.worldTransform.scale})"><svg>${paths.join('')}</svg>${labels.join('')}</div></div>`;
}
function renderOverlay(){if(!ready||!e)return;const geo=e.readPairGeometry(0);if(!geo)return;const origin=$('tensor-overlay').getBoundingClientRect(),key=JSON.stringify([geo,pairs,chosen,candidateSide,drafting]);if(key===lastGeometry)return;lastGeometry=key;const focus=document.activeElement?.matches('.tensor-tag')?document.activeElement.dataset.tensors:null;$('tensor-overlay').innerHTML=overlayPane(side,geo,origin);if(focus)$('tensor-overlay').querySelector(`[data-tensors="${focus}"]`)?.focus({preventScroll:true});renderPicker();}
// A blank click cancels pairing; a pan gesture or a node click does not.
let backgroundPointer=null;
function isGraphBackground(ev){
 // Native controls can sit outside a node and cross the visualizer shadow root.
 if(ev.composedPath().some(el=>el.matches?.('button,[role="button"],[role="menuitem"],.group-node-icon,.io-picker,.subgraph-indicator,.toolbar,search-bar,input')))return false;
 if(!ready||!e)return false;
 const geo=e.readPairGeometry(0);if(!geo)return false;
 const x=(ev.clientX-geo.rect.left-geo.worldTransform.x)/geo.worldTransform.scale,y=(ev.clientY-geo.rect.top-geo.worldTransform.y)/geo.worldTransform.scale;
 return !geo.nodes.some(n=>x>=n.x&&x<=n.x+n.width&&y>=n.y&&y<=n.y+n.height);
}
// Classify before native mousedown can collapse or reposition the clicked node.
$('host').addEventListener('pointerdown',ev=>{backgroundPointer=ev.button===0&&isGraphBackground(ev)?{x:ev.clientX,y:ev.clientY,id:ev.pointerId}:null;},true);
$('host').addEventListener('pointerup',ev=>{
 const start=backgroundPointer;backgroundPointer=null;
 if(!start||start.id!==ev.pointerId||Math.hypot(ev.clientX-start.x,ev.clientY-start.y)>4)return;
 if(isGraphBackground(ev))send({type:'cancel-selection'});
});
$('host').addEventListener('pointercancel',()=>{backgroundPointer=null;});
function resetPaneView(){if(!ready||!e)return;e.fitPaneGraph(0,{duration:0});renderOverlay();}
function configureNativeToolbar(element){
 const toolbar=element.shadowRoot.querySelector('.toolbar');if(!toolbar)return;
 const title=side==='reference'?'Reference':'Target';toolbar.setAttribute('role','toolbar');toolbar.setAttribute('aria-label',title+' graph controls');
 for(const [selector,label] of [['.zoom-fit','Reset '+title+' graph view'],['.expand-all','Expand all '+title+' groups'],['.collapse-all','Collapse all '+title+' groups']]){
  const button=toolbar.querySelector('.mat-icon-container'+selector);if(!button)continue;
  button.setAttribute('role','button');button.setAttribute('aria-label',label);button.tabIndex=0;
  button.addEventListener('keydown',ev=>{if(ev.key==='Enter'||ev.key===' '){ev.preventDefault();button.click();}});
 }
 toolbar.querySelector('input')?.setAttribute('aria-label','Search '+title+' nodes and layers');
}
new ResizeObserver(renderOverlay).observe($('tensor-overlay'));
document.addEventListener('keydown',ev=>{if(ev.key==='Escape'&&(drafting||chosen.reference||chosen.target)){ev.preventDefault();send({type:'cancel-selection'});}});
window.addEventListener('message',ev=>{const d=ev.data;if(ev.source!==parent||ev.origin!==location.origin||d?.channel!==channel||d.side!==side)return;
 if(d.type==='load'||d.type==='state'){candidateSide=d.candidateSide||null;drafting=!!d.drafting;}
 if(d.type==='load'){revision=d.revision;entryAnchor=d.anchor||'';palette=d.palette;theme=d.theme;pairs=d.pairs||[];chosen=d.chosen||{reference:null,target:null};paint();mount(d.graph).catch(err=>{$('loading').textContent='Could not load graph: '+err.message;});}
 if(d.type==='reset-view'&&d.revision===revision)resetPaneView();
 if(d.type==='state'&&d.revision===revision){palette=d.palette;theme=d.theme;pairs=d.pairs||[];chosen=d.chosen;paint();}
});send({type:'mounted'});
})().catch(error=>{document.getElementById('loading').textContent=error.message;console.error(error);});
