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

// Only known metrics participate in Min/Max. Output never falls back to an internal tensor.
window.summarizeModelTreeMetrics=function(values,output,mode){
 const known=values.filter(value=>typeof value==='number'&&Number.isFinite(value));
 const value=mode==='min'?(known.length?Math.min(...known):null):mode==='max'?(known.length?Math.max(...known):null):(typeof output==='number'&&Number.isFinite(output)?output:null);
 return {value,available:known.length,total:values.length};
};
// Stage membership comes from the bundled execution export, never metric availability.
// This is preview coverage, not a claim about an active session's executed trace.
window.modelTreeStageProfile=function(stage,catalog=window.MAPPING_CATALOG){
 const scopes=catalog?.scopes??{},has=id=>(scopes[id]??[]).some(graph=>graph.stage==='any'||graph.stage===stage);
 const allLayers=Object.keys(scopes).filter(id=>/^decoder_layer_\d+$/.test(id)).map(id=>Number(id.slice('decoder_layer_'.length))).sort((a,b)=>a-b);
 return {stage,layerIds:allLayers.filter(id=>has('decoder_layer_'+id)),lastLayer:allLayers.at(-1),
  embedding:has('text_embedding'),ple:has('ple_preparation'),finalNorm:has('final_norm'),outputHead:has('output_head')};
};
window.createModelTree=function(api){
 const host=document.getElementById('model-tree'),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 let active='',folded=new Set();
 const profile=()=>window.modelTreeStageProfile(window.GEMMA_METRIC_DEMO?.stage??'decode');
 const metric=()=>window.GEMMA_METRIC_DEMO;
 function metricStyle(value){const color=value==null?null:metric()?.color(value);return color?`;--row-metric-bg:${esc(color.bgColor)};--row-metric-fg:${esc(color.textColor)}`:'';}
 function metricValue(value,title='Layer output metric · selected token position'){
  return `<span class="tree-metric" title="${esc(value==null?title+' · No data':title)}">${value==null?'—':esc(metric()?.format?.(value)??value)}</span>`;
 }
 function row(id,name,depth=0,value){const disabled=id==='embedding';return `<button class="model-tree-row ${active===id?'active':''}" data-tree-open="${id}" style="--depth:${depth}${metricStyle(value)}" aria-label="${esc(name)}" ${disabled?'disabled title="Arch is not available for Embedding"':''} ${active===id?'aria-current="page"':''}><span class="tree-label">${esc(name)}</span>${metricValue(value)}</button>`;}
 function group(id,name,depth,values,output){
  const expanded=!folded.has(id),mode=metric()?.summary==='min'?'min':'max';
  const summary=window.summarizeModelTreeMetrics(values,output,mode);
  const title=`${mode==='min'?'Min':'Max'} of internal outputs · ${summary.available}/${summary.total} available`;
  return `<button class="tree-group" data-tree-fold="${id}" aria-label="${esc(name)}" aria-expanded="${expanded}" style="--depth:${depth}${expanded?'':metricStyle(summary.value)}"><span class="material-icon" aria-hidden="true">${expanded?'keyboard_arrow_down':'chevron_right'}</span><span class="tree-label">${esc(name)}</span>${expanded?'':metricValue(summary.value,title)}</button>`;
 }
 function render(){
  const scroll=host.scrollTop,focus=document.activeElement,focusKey=host.contains(focus)?(focus.dataset.treeOpen||focus.dataset.treeFold):null;
  const data=metric(),values=data?.values??{},scope=profile(),layers=scope.layerIds.map(i=>data?.layers.decoder_stack[i]??null);
  const decoderValues=[...(scope.ple?[values.ple_preparation]:[]),...layers];
  const internalLayers=scope.layerIds.filter(i=>i!==scope.lastLayer).map(i=>data?.layers.decoder_stack[i]??null);
  const decoderOpen=!folded.has('decoder'),layersOpen=!folded.has('layers');
  host.innerHTML=`<nav aria-label="Model modules">${scope.embedding?row('embedding','Embedding',0,values.embedding):''}${group('decoder','Decoder',0,decoderValues,values.decoder_final_norm)}<div ${decoderOpen?'':'hidden'}>${scope.ple?row('ple_preparation','Per-Layer Embedding',1,values.ple_preparation):''}${group('layers','Transformer Layers ('+scope.layerIds.length+')',1,internalLayers,data?.layers.decoder_stack[scope.lastLayer])}<div ${layersOpen?'':'hidden'}>${layers.map((value,i)=>row('layer:'+scope.layerIds[i],'Layer '+scope.layerIds[i],2,value)).join('')}</div>${scope.finalNorm?row('final_norm','Final Norm',1,values.decoder_final_norm):''}</div>${scope.outputHead?row('output_head','Output Head',0,values.output_head):''}</nav>`;
  if(focusKey)[...host.querySelectorAll('button')].find(b=>(b.dataset.treeOpen||b.dataset.treeFold)===focusKey)?.focus({preventScroll:true});
  host.scrollTop=scroll;
 }
 function available(id){const scope=profile();return id.startsWith('layer:')?scope.layerIds.includes(Number(id.slice(6))):id==='ple_preparation'?scope.ple:id==='final_norm'?scope.finalNorm:id==='output_head'?scope.outputHead:false;}
 function open(id){if(!available(id))return;
  active=id;render();if(id.startsWith('layer:'))api.layer(Number(id.split(':')[1]));else api.arch(id);
  document.getElementById('workspace').classList.remove('show-model-tree');document.getElementById('modules-toggle').setAttribute('aria-expanded','false');
 }
 function start(){active='';folded=new Set();host.scrollTop=0;open('ple_preparation');}
 host.addEventListener('click',e=>{const fold=e.target.closest('[data-tree-fold]');if(fold){const id=fold.dataset.treeFold;folded.has(id)?folded.delete(id):folded.add(id);render();return;}const item=e.target.closest('[data-tree-open]');if(item&&!item.disabled)open(item.dataset.treeOpen);});
 host.addEventListener('keydown',e=>{if(!['ArrowDown','ArrowUp','Home','End'].includes(e.key))return;const rows=[...host.querySelectorAll('button:not(:disabled)')].filter(b=>b.getClientRects().length),i=rows.indexOf(document.activeElement);if(i<0)return;e.preventDefault();rows[e.key==='Home'?0:e.key==='End'?rows.length-1:Math.max(0,Math.min(rows.length-1,i+(e.key==='ArrowDown'?1:-1)))].focus();});
 window.createModelTreeScrollbar(host);
 return {start,open,refresh:()=>{if(active&&!available(active))open(profile().ple?'ple_preparation':'layer:'+profile().layerIds[0]);else render();}};
};
