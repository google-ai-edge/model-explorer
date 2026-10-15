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

import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const sandbox={window:{}};
vm.runInNewContext(fs.readFileSync(new URL('../public/graph_arch_preview/arch_metrics.js',import.meta.url),'utf8'),sandbox);
const metric=sandbox.window.archMetricData;
const context={layers:[0.99],colors:{'0.99':{bgColor:'#eeeeff',textColor:'#112233'}},formatted:{'0.99':'0.990000'}};
assert.equal(metric(context,'decoder_layer_0','output').text,'0.990000');
assert.equal(metric(context,'decoder_layer_0','output').bg,'#eeeeff');
assert.equal(metric(context,'decoder_layer_0','output').fg,'#112233');
assert.equal(metric(context,'decoder_layer_0','q_proj_out').value,null);
assert.equal(metric(context,'decoder_layer_0','post_attention_value',['q_proj_out','post_attention_value']).value,null,'Block output must not borrow layer output');
const explicit={...context,tensorMetrics:{decoder_layer_0:{q_proj_out:0,post_attention_value:null,k_proj_out:0.3}}};
const members=['q_proj_out','k_proj_out','post_attention_value','q_proj_out'];
assert.equal(metric({...explicit,summary:'min'},'decoder_layer_0','post_attention_value',members).value,0);
const max=metric({...explicit,summary:'max'},'decoder_layer_0','post_attention_value',members);
assert.equal(max.value,0.3);assert.equal(max.available,2);assert.equal(max.total,2);
assert.equal(metric({...explicit,summary:'output'},'decoder_layer_0','post_attention_value',members).value,0.3, 'Legacy Output falls back to internal Max');
assert.equal(metric({...context,tensorMetrics:{decoder_layer_0:{output:null}}},'decoder_layer_0','output').value,null);
assert.equal(metric(context,'ple_preparation','lookup').text,'—');
assert.equal(metric(context,'decoder_layer_1','output').value,null);
console.log('Arch metric tags: format/color parity, missing values, zero, explicit bindings, deduplicated Min/Max and output ownership passed.');
const samples=Array.from({length:101},(_,i)=>({value:i/100,text:(i/100).toFixed(2),bgColor:`color-${i}`,textColor:'foreground'}));
const preview={...context,position:3,stage:'decode',simulatedMetrics:samples};
const fake=metric(preview,'decoder_layer_0','q_proj_out');
assert.equal(fake.simulated,true);
assert.deepEqual(fake,metric(preview,'decoder_layer_0','q_proj_out'));
assert.doesNotMatch(fake.title,/Simulated/);
assert.match(fake.title,/Preview sample/);
assert.equal(metric(preview,'decoder_layer_0','output').value,0.99);
assert.equal(metric({...preview,...explicit},'decoder_layer_0','post_attention_value').value,null);
assert.notEqual(metric({...preview,position:4},'decoder_layer_0','q_proj_out').value,fake.value);
const fakeMin=metric({...preview,summary:'min'},'decoder_layer_0','post_attention_value',members);
assert.equal(fakeMin.value,Math.min(...[...new Set(members)].filter(id=>id!=='post_attention_value').map(id=>metric(preview,'decoder_layer_0',id).value)));
const display=sandbox.window.archTensorDisplay;
assert.equal(display(50).labelWidth,18);
assert.equal(display(50).totalWidth,77);
assert.equal(display(50).tagOffset,27);
console.log('Deterministic simulated metrics, binding precedence, summary and display geometry passed.');

const excluded={summary:'max',tensorMetrics:{block:{a:0.2,b:0.5,output:999}}};
assert.equal(metric(excluded,'block','output',['a','b','output']).value,0.5);
assert.equal(metric({...excluded,summary:'min'},'block','output',['a','b','output']).value,0.2);
assert.equal(metric(excluded,'block','output').value,999);
assert.equal(metric(excluded,'block','output',['output']).value,null,'No internal data must not fall back to output');
assert.equal(metric({...excluded,summary:'min'},'block','output',['output']).mode,'min');
console.log('Internal summary excludes final output, preserves standalone output and supports empty Min/Max.');
