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

// Preview-only async metrics. Pair identity and analysis context own every result.
// A removed/replaced pair or obsolete context can never publish a late result.
window.createMappingMetricPreview = function({onChange = () => {}, schedule = setTimeout, cancel = clearTimeout} = {}) {
  const entries = new Map();
  let contextKey = null, samples = [], deferResults = false;
  const key = pair => JSON.stringify([pair.reference.id, pair.target.id]);
  const result = pair => {
    if([pair.reference,pair.target].some(t=>/^(?:u?int|bool)/i.test(t.dtype||'')))return {state:'missing',value:null,text:'—',label:'Paired · No metric available'};
    let hash = 2166136261;
    for (const c of key(pair) + contextKey) hash = Math.imul(hash ^ c.charCodeAt(0), 16777619) >>> 0;
    const sample = samples[hash % samples.length];
    return sample ? {state:'ready', value:sample.value, text:sample.text, bg:sample.bgColor, fg:sample.textColor, label:'Paired · Preview metric'} : {state:'missing', value:null, text:'—', label:'Paired · No metric available'};
  };
  function remove(pair) { const id=key(pair), old=entries.get(id); if(old?.timer!=null)cancel(old.timer); entries.delete(id); }
  function compute(pair, delay=3000) {
    remove(pair);
    const id=key(pair), context=contextKey, entry={pair, metric:{state:'pending',value:null,text:'Calculating',label:'Paired · Calculating metric'},timer:null};
    entries.set(id,entry);
    entry.timer=schedule(()=>{if(contextKey!==context||entries.get(id)!==entry)return;entry.timer=null;entry.metric=result(pair);onChange();},delay);
    return entry.metric;
  }
  return {
    key, remove, compute,
    setContext(nextKey, nextSamples) {
      samples=nextSamples||[];
      if(contextKey===nextKey){for(const entry of entries.values())if(entry.metric.state==='ready')entry.metric=result(entry.pair);return;}
      deferResults=contextKey!==null;contextKey=nextKey;
      for(const entry of entries.values())if(entry.timer!=null)cancel(entry.timer);
      entries.clear();
    },
    read(pair) {
      const existing=entries.get(key(pair));if(existing)return existing.metric;
      if(pair.previewMetricState==='missing'){
        const metric={state:'missing',value:null,text:'—',label:'Paired · No metric available'};
        entries.set(key(pair),{pair,metric,timer:null});return metric;
      }
      if(deferResults||pair.previewMetricState==='pending')return compute(pair,pair.previewDelay||3000);
      const metric=result(pair);entries.set(key(pair),{pair,metric,timer:null});return metric;
    },
  };
};
