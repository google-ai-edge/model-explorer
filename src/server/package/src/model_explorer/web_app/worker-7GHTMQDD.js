var q=4,ce=4,S=9,pe=16,ue=200,j=12,ge=14,fe=6,he=6,Ne=9,Ee=1e3;var z=25;var me=.1;var C="Node data provider: ",T=10;var H="__value",Z="__tensor_tag",A=20;var Y=14;var Ce=new OffscreenCanvas(300,300),Ie={},Gt=typeof navigator<"u"&&/Macintosh/.test(navigator.userAgent);function y(n){return n?.nodeType===0}function E(n){return n?.nodeType===1}function J(n){let e=[];for(let[o,t]of Object.entries(n))switch(o){case"Op node id":t.selected&&e.push("id");break;default:break}return e}function ee(n){let e=[];for(let[o,t]of Object.entries(n))switch(o){case"Layer node children count":t.selected&&e.push("#children");break;case"Layer node descendants count":t.selected&&e.push("#descendants");break;default:break}return e}function te(n,e){if(y(n))switch(e.toLowerCase()){case"id":return n.id;case"namespace":return _e(n);default:break}else if(E(n))switch(e.toLowerCase()){case"namespace":return _e(n);case"#children":return String((n.nsChildrenIds||[]).length);case"#descendants":return String((n.descendantsNodeIds||[]).length);default:break}return""}function _e(n){return n.fullNamespace||n.namespace||"<root>"}function x(n,e,o,t=!1){let r=[];n==null?r=e.rootNodes.map(s=>s.id):r=n.nsChildrenIds||[];for(let s of r){let a=e.nodesById[s];if(a&&E(a)&&(t||!t&&a.expanded)){let d=a.nsChildrenIds||[];(t?d.filter(i=>E(e.nodesById[i])).length===0:d.filter(i=>E(e.nodesById[i])).every(i=>!e.nodesById[i].expanded))&&o.push(a.id),x(a,e,o,t)}}}function ye(n,e,o,t){let r=[];if(n.length===2)r=n;else if(n.length===3&&n[0].x===n[1].x&&n[1].x===n[2].x)r=n;else{let s=!0,a=0;for(let l=0;l<n.length-1;l++){let i=n[l],p=n[l+1]>i?1:-1;if(a!==0&&a!==p){s=!1;break}a=p}let d=t.Vector3;if(s){let i=e().x(g=>g.x).y(g=>g.y).curve(o)(n).split(/M|C/).filter(g=>g!=="").map(g=>g.split(",").map(f=>Number(f))),c=new d(i[0][0],i[0][1],0),p=new t.CurvePath;for(let g=1;g<i.length;g++){let f=i[g];if(f.length===6){let N=c,_=new d(f[0],f[1]),u=new d(f[2],f[3]),h=new d(f[4],f[5]);c=h;let m=new t.CubicBezierCurve3(N,_,u,h);p.add(m)}}r=p.getPoints(z)}else{let l=n.map(c=>new d(c.x,c.y,0));r=new t.CatmullRomCurve3(l,!1,"catmullrom",me).getPoints(z)}}return r}function D(n,e,o,t=!0){let r=`${n}___${e}___${o}`,s=Ie[r];if(s==null){let a=Ce.getContext("2d");a.font=`${e}px "Google Sans Text", Arial, Helvetica, sans-serif`,o&&(a.font=`bold ${a.font}`);let l=a.measureText(n).width;t&&(Ie[r]=l),s=l}return s}function we(n,e,o){let t=o[Z];return t?`Input${n}:${t} (${e.label})`:`Input${n} (${e.label})`}function Be(n,e,o){let t=`Output${n}`;if(o.label==="GraphInputs"){let r=e.tensor_name;r!=null&&(t=`${t} (${r})`)}else{let r=e[Z];r&&(t=`Output${n}:${r}`)}return t}function Re(n){let e=((n||{}).shape||"").replace(/ /g,"").replace(/×/g,"x");return e===""&&(e="?"),e}function oe(n,e=""){let o=n.attrs||{},t=[],r=new RegExp(e,"i");for(let s of Object.keys(o)){let a=s,d=o[s],l=[`${a}:${d}`,`${a}=${d}`];if(e.trim()===""||l.some(i=>r.test(i))){let i=d;a===H?i=d.replace(/\s/gm,""):i=d.replace(/(\r\n|\n|\r)/gm," "),t.push({key:a,value:i})}}return t}function re(n,e,o=""){let t=e.groupNodeAttributes?.[n.id.replace("___group___","")]||{},r=[],s=new RegExp(o,"i");for(let a of Object.keys(t)){let d=a,l=t[a],i=[`${d}:${l}`,`${d}=${l}`];if(o.trim()===""||i.some(c=>s.test(c))){let c=l.replace(/(\r\n|\n|\r)/gm," ");r.push({key:d,value:c})}}return r}function Te(n,e){let o=n.incomingEdges||[],t=[];for(let r=0;r<Math.min(T,o.length);r++){let s=o[r],a=s.sourceNodeId,d=e.nodesById[a],l=Re((d.outputsMetadata||{})[s.sourceNodeOutputId]),i=(n.inputsMetadata||{})[s.targetNodeInputId]||{};t.push({key:we(r,d,i),value:l})}if(o.length>T){let r=o.length-T;t.push({key:`(${r} more input${r===1?"":"s"} omitted)`,value:"..."})}return t}function Oe(n){let e=[],o=n.outputsMetadata||{},t=Object.values(o);for(let r=0;r<Math.min(T,t.length);r++){let s=t[r],a=Re(s);e.push({key:Be(r,s,n),value:a})}if(t.length>T){let r=t.length-T;e.push({key:`(${r} more output${r===1?"":"s"} omitted)`,value:"..."})}return e}function Pe(n,e,o,t,r){let s=[],a=Object.keys(o).filter(l=>o[l].selected).filter(l=>l.startsWith(C)).map(l=>l.replace(C,"")),d=Object.values(t).filter(l=>a.includes($(l,{id:e})));for(let l of d){let i=((l.results||{})?.[e]||{})[n.id];if(r?.hideEmptyNodeDataEntries&&!i)continue;let c=i?.strValue||"-";s.push({key:$(l,{id:e}),value:c})}return s}function Ae(n,e){let o=n.split("/"),t=e.split("/"),r="";for(let s=Math.min(o.length,t.length);s>0;s--){let a=o.slice(0,s).join("/"),d=t.slice(0,s).join("/");if(a===d){r=d;break}}return r}function ne(n,e){if(n===e)return"";let o=n.split("/").filter(r=>r!==""),t=e.split("/").filter(r=>r!=="");return t.length===0?"":t[o.length]}function B(n){return n.split(`
`).map(e=>e.trim()).filter(e=>e!=="")}function F(n){return(B(n).length-1)*Y}function $(n,e){return n.nodeDataProviderData?.[e?.id||""]?.name??n.runName}function se(n,e,o){let t={},r=n?.descendantsOpNodeIds||e.nodes.map(s=>s.id);for(let s of r){let a=e.nodesById[s],d=o[a.id]?.bgColor||"";d&&(t[d]?t[d].count++:t[d]={label:`${o[s]?.value||""}`,bgColor:d,count:1})}return Object.values(t).sort((s,a)=>s.bgColor.localeCompare(a.bgColor))}var ae=36,xe=16,De=26,Ve=50,Ue=24,ke=80,He=8,R=class{constructor(e,o,t,r,s,a=!1,d){this.modelGraph=e;this.dagre=o;this.showOnNodeItemTypes=t;this.nodeDataProviderRuns=r;this.selectedNodeDataProviderRunId=s;this.testMode=a;this.config=d;this.dagreGraph=new this.dagre.graphlib.Graph}dagreGraph;layout(e){let o,t=[];e==null?t=this.modelGraph.rootNodes:(o=this.modelGraph.nodesById[e],t=(o.nsChildrenIds||[]).map(u=>this.modelGraph.nodesById[u])),this.configLayout(this.dagreGraph);let r=ie(o?.id||"",t,this.modelGraph,this.showOnNodeItemTypes,this.nodeDataProviderRuns,this.selectedNodeDataProviderRunId,this.testMode,!1,this.config);for(let u of Object.keys(r.nodes)){let h=r.nodes[u];h.config?.pinToGroupTop||this.dagreGraph.setNode(u,h)}for(let u of Object.keys(r.outgoingEdges))for(let h of r.outgoingEdges[u])this.dagreGraph.setEdge(u,h);this.dagre.layout(this.dagreGraph);let s=Number.MAX_VALUE,a=Number.MAX_VALUE,d=Number.NEGATIVE_INFINITY,l=Number.NEGATIVE_INFINITY;for(let u of t){let h=r.nodes[u.id];if(!h){console.warn(`Node "${u.id}" is not in the dagre layout result`);continue}u.x=(h.x||0)-h.width/2,u.y=(h.y||0)-h.height/2,u.width=h.width,u.height=h.height,u.localOffsetX=0,u.localOffsetY=0,h.config?.pinToGroupTop||(s=Math.min(s,u.x),a=Math.min(a,u.y),d=Math.max(d,u.x+u.width),l=Math.max(l,u.y+u.height))}let i=Number.MAX_VALUE,c=Number.MAX_VALUE,p=Number.NEGATIVE_INFINITY,g=Number.NEGATIVE_INFINITY,f=this.dagreGraph.edges(),N=[];for(let u of f){let h=this.dagreGraph.edge(u).points,m=globalThis.d3,O=globalThis.THREE,M=typeof O>"u"?[]:ye(h,m.line,m.curveMonotoneY,O),I=this.modelGraph.nodesById[u.v],G=this.modelGraph.nodesById[u.w];if(I==null){console.warn(`Edge from node not found: "${u.v}"`);continue}if(G==null){console.warn(`Edge to node not found: "${u.w}"`);continue}let w=`${I.id}|${G.id}`;N.push({id:w,fromNodeId:I.id,toNodeId:G.id,points:h,curvePoints:M});for(let P of h)i=Math.min(i,P.x),c=Math.min(c,P.y),p=Math.max(p,P.x),g=Math.max(g,P.y)}if(this.modelGraph.edgesByGroupNodeIds[e||""]=N,i<s)for(let u of t)u.localOffsetX=Math.max(0,s-i);s=Math.min(i,s),d=Math.max(p,d);let _=d-s+A*2;if(o){let u=U(o,this.modelGraph,this.showOnNodeItemTypes,this.nodeDataProviderRuns,this.selectedNodeDataProviderRunId,this.testMode,this.config);if(o.pinToTopOpNode){let h=U(o.pinToTopOpNode,this.modelGraph,this.showOnNodeItemTypes,this.nodeDataProviderRuns,this.selectedNodeDataProviderRunId,this.testMode,this.config)+A*2;u=Math.max(u,h)}if(_<u){let h=(u-_)/2;for(let m of t)m.localOffsetX||(m.localOffsetX=0),m.localOffsetX+=h;_=u}}if(t.length===1&&y(t[0])&&t[0].config?.pinToGroupTop&&(s=0,a=0,l=-15),o&&E(o)){let u=be(o,this.modelGraph,this.showOnNodeItemTypes);if(u>0){let h=u*j+16;for(let m of t)m.localOffsetY=h;l+=h}}return{x:s,y:a,width:_-A*2,height:l-a}}configLayout(e){e.setGraph({nodesep:20,ranksep:50,edgesep:20,marginx:A,marginy:ae}),e.setDefaultEdgeLabel(()=>({}))}};function U(n,e,o,t,r,s=!1,a){if(s)return Ve;let d=n.label,l=B(d),i=0;for(let N of l)i=Math.max(D(N,11,E(n))+Ue,i);E(n)&&(i+=28);let c=0,p=0,g=0;if(y(n)){let N=J(o);for(let h of N){let m=D(`${h}:`,S,!0),O=te(n,h),M=D(O,S,!1);c=Math.max(c,m),p=Math.max(p,M)}if(o["Op node attributes"]?.selected){let h=oe(n,o["Op node attributes"]?.filterRegex||""),m=V(h);c=Math.max(c,m.maxAttrLabelWidth),p=Math.max(p,m.maxAttrValueWidth)}if(o["Op node inputs"]?.selected){let h=Te(n,e),m=V(h);c=Math.max(c,m.maxAttrLabelWidth),p=Math.max(p,m.maxAttrValueWidth)}if(o["Op node outputs"]?.selected){let h=Oe(n),m=V(h);c=Math.max(c,m.maxAttrLabelWidth),p=Math.max(p,m.maxAttrValueWidth)}let _=Pe(n,e.id,o,t,a),u=V(_);c=Math.max(c,u.maxAttrLabelWidth),p=Math.max(p,u.maxAttrValueWidth)}else if(E(n)){let N=ee(o);for(let _ of N){let u=D(`${_}:`,S,!0),h=te(n,_),m=D(h,S,!1);c=Math.max(c,u),p=Math.max(p,m)}if(o["Layer node attributes"]?.selected){let _=re(n,e,o["Layer node attributes"]?.filterRegex||""),u=V(_);c=Math.max(c,u.maxAttrLabelWidth),p=Math.max(p,u.maxAttrValueWidth)}if(E(n)&&!n.expanded&&r&&t[r]){let _=t[r];if((_.nodeDataProviderData??{})[e.id]?.showExpandedSummaryOnGroupNode??!1){let h=se(n,e,(_.results??{})[e.id]);for(let m of h){let O=D(`${m.label} 100% (${m.count})`,Ne,!1)+30;g=Math.max(g,O)}}}}p=Math.min(p,ue);let f=c+p+ce*2+q;return f!==q&&(f+=He*2),Math.max(Math.max(ke,Math.max(i,f)),g)}function de(n,e,o,t,r,s=!1,a=!1,d){if(s)return De;if(n.height!=null&&!a)return n.height;let l=F(n.label),i=0;y(n)?i=Ye(o,n,t,e,d):E(n)&&(i=be(n,e,o));let c=0;if(E(n)&&!n.expanded&&r&&t[r]){let p=t[r];((p.nodeDataProviderData??{})[e.id]?.showExpandedSummaryOnGroupNode??!1)&&(c=se(n,e,(p.results??{})[e.id]).length)}return De+l+i*j+(i>0?pe-4:0)+c*ge+(c>0?fe+he:0)}function ie(n,e,o,t,r,s,a=!1,d=!1,l){let i={nodes:{},incomingEdges:{},outgoingEdges:{}};for(let p of e){if(y(p)&&p.hideInLayout)continue;let g={id:p.id,width:p.width||(d?10:U(p,o,t,r,s,a,l)),height:d?10:de(p,o,t,r,s,a,!1,l),config:y(p)?p.config:void 0};i.nodes[p.id]=g}let c=o.layoutGraphEdges[n]||{};for(let[p,g]of Object.entries(c))for(let f of Object.keys(g)){let N=o.nodesById[p],_=o.nodesById[f];N&&y(N)&&N.config?.pinToGroupTop||_&&y(_)&&_.config?.pinToGroupTop||$e(i,p,f)}return i}function Ye(n,e,o,t,r){let s=J(n),a=n["Op node attributes"]?.selected?oe(e,n["Op node attributes"]?.filterRegex||"").length:0,d=n["Op node inputs"]?.selected?Object.keys(e.incomingEdges||[]).length:0;d>T&&(d=T+1);let l=n["Op node outputs"]?.selected?Object.keys(e.outputsMetadata||{}).length:0;l>T&&(l=T+1);let i=Object.keys(n).filter(c=>n[c].selected).filter(c=>c.startsWith(C)&&Object.values(o).some(p=>{let g=((p.results||{})?.[t.id]||{})[e.id];return r?.hideEmptyNodeDataEntries&&!g?!1:$(p,t)===c.replace(C,"")})).length;return s.length+a+d+l+i}function be(n,e,o){let t=ee(o),r=o["Layer node attributes"]?.selected?re(n,e,o["Layer node attributes"]?.filterRegex||"").length:0;return t.length+r}function $e(n,e,o){n.outgoingEdges[e]==null&&(n.outgoingEdges[e]=[]),n.outgoingEdges[e].push(o),n.incomingEdges[o]==null&&(n.incomingEdges[o]=[]),n.incomingEdges[o].push(e)}function V(n){let e=0,o=0;for(let{key:t,value:r}of n){let s=D(t,S,!0);e=Math.max(e,s);let a=D(r,S,!1);o=Math.max(o,a)}return{maxAttrLabelWidth:e,maxAttrValueWidth:o}}var L=class{constructor(e,o,t,r,s,a=!1,d){this.modelGraph=e;this.dagre=o;this.showOnNodeItemTypes=t;this.nodeDataProviderRuns=r;this.selectedNodeDataProviderRunId=s;this.testMode=a;this.config=d}dagreGraphs=[];expandGroupNode(e){let o=this.modelGraph.nodesById[e];if(o&&E(o)){if(o.expanded)return;o.expanded=!0}let t=e;for(;t!=null;){let s=this.modelGraph.nodesById[t];if(!s)break;s.expanded=!0;let a=new R(this.modelGraph,this.dagre,this.showOnNodeItemTypes,this.nodeDataProviderRuns,this.selectedNodeDataProviderRunId,this.testMode,this.config),d=a.layout(t);this.testMode&&this.dagreGraphs.push(a.dagreGraph);let l=d.width+A*2,i=this.getTargetGroupNodeHeight(d,s);s.width=l,s.height=i,t=s.nsParentId}let r=new R(this.modelGraph,this.dagre,this.showOnNodeItemTypes,this.nodeDataProviderRuns,this.selectedNodeDataProviderRunId,this.testMode,this.config);r.layout(),this.testMode&&this.dagreGraphs.push(r.dagreGraph);for(let s of this.modelGraph.rootNodes)E(s)&&this.updateNodeOffset(s)}expandFromDeepestGroupNodes(e){let o=new Set,t=[...e];for(;t.length>0;){let a=t.shift();if(o.has(a)||!this.modelGraph.nodesById[a])continue;o.add(a);let l=this.modelGraph.nodesById[a]?.nsParentId;l&&t.push(l)}let r=Array.from(o).sort((a,d)=>{let l=this.modelGraph.nodesById[a];return this.modelGraph.nodesById[d].level-l.level});for(let a of r){let d=this.modelGraph.nodesById[a];d.expanded=!0;let l=new R(this.modelGraph,this.dagre,this.showOnNodeItemTypes,this.nodeDataProviderRuns,this.selectedNodeDataProviderRunId,this.testMode,this.config),i=l.layout(a);this.testMode&&this.dagreGraphs.push(l.dagreGraph);let c=i.width+A*2,p=this.getTargetGroupNodeHeight(i,d);d.width=c,d.height=p}let s=new R(this.modelGraph,this.dagre,this.showOnNodeItemTypes,this.nodeDataProviderRuns,this.selectedNodeDataProviderRunId,this.testMode,this.config);s.layout(),this.testMode&&this.dagreGraphs.push(s.dagreGraph);for(let a of this.modelGraph.rootNodes)E(a)&&this.updateNodeOffset(a)}expandToRevealNode(e){let o=this.modelGraph.nodesById[e],t=[],r=o;for(;;){let a=this.modelGraph.nodesById[r.nsParentId||""];if(!a)break;t.unshift(a),r=a}for(let a of t)this.expandGroupNode(a.id);let s=[];return x(void 0,this.modelGraph,s),s}collapseGroupNode(e){let o=this.modelGraph.nodesById[e];if(!o)return[];o.expanded=!1,delete this.modelGraph.edgesByGroupNodeIds[e],o.width=U(o,this.modelGraph,this.showOnNodeItemTypes,this.nodeDataProviderRuns,this.selectedNodeDataProviderRunId),o.height=de(o,this.modelGraph,this.showOnNodeItemTypes,this.nodeDataProviderRuns,this.selectedNodeDataProviderRunId,this.testMode,!0,this.config);let t=o.nsParentId;for(;t!=null;){let a=this.modelGraph.nodesById[t];if(!a)break;let d=new R(this.modelGraph,this.dagre,this.showOnNodeItemTypes,this.nodeDataProviderRuns,this.selectedNodeDataProviderRunId,this.testMode,this.config),l=d.layout(t);this.testMode&&this.dagreGraphs.push(d.dagreGraph);let i=l.width+A*2,c=this.getTargetGroupNodeHeight(l,a);a.width=i,a.height=c,t=a.nsParentId}let r=new R(this.modelGraph,this.dagre,this.showOnNodeItemTypes,this.nodeDataProviderRuns,this.selectedNodeDataProviderRunId,this.testMode,this.config);r.layout(),this.testMode&&this.dagreGraphs.push(r.dagreGraph);for(let a of this.modelGraph.rootNodes)E(a)&&this.updateNodeOffset(a);let s=[];return x(void 0,this.modelGraph,s),s}reLayoutGraph(e,o){let t=e;if(t)o&&this.clearLayoutData(void 0,!0);else{let r=[];this.clearLayoutData(void 0),x(void 0,this.modelGraph,r),t=r}return t.length>0?this.expandFromDeepestGroupNodes(t):new R(this.modelGraph,this.dagre,this.showOnNodeItemTypes,this.nodeDataProviderRuns,this.selectedNodeDataProviderRunId,this.testMode,this.config).layout(),t}expandAllGroups(){this.clearLayoutData(void 0,!0);let e=this.modelGraph.nodes.filter(o=>E(o)&&(o.nsChildrenIds||[]).filter(t=>E(this.modelGraph.nodesById[t])).length===0).map(o=>o.id);return e.length>0&&this.expandFromDeepestGroupNodes(e),e}collapseAllGroup(){this.clearLayoutData(void 0,!0),new R(this.modelGraph,this.dagre,this.showOnNodeItemTypes,this.nodeDataProviderRuns,this.selectedNodeDataProviderRunId,this.testMode,this.config).layout();for(let o of this.modelGraph.rootNodes)E(o)&&this.updateNodeOffset(o);return[]}updateNodeOffset(e){for(let o of e.nsChildrenIds||[]){let t=this.modelGraph.nodesById[o];if(t.x!=null&&t.y!=null){t.globalX=(e.x||0)+(e.globalX||0)+(t.localOffsetX||0),t.globalY=(e.y||0)+(e.globalY||0)+(t.localOffsetY||0);let r=(B(e.label).length-1)*Y;r>0&&(t.globalY+=r),e.pinToTopOpNode&&t.id!==e.pinToTopOpNode.id&&(t.globalY+=this.getPinToTopNodeVerticalSpace(e.pinToTopOpNode)),e.pinToTopOpNode?.id===t.id&&(t.globalX=(e.x||0)+(e.globalX||0)+(e.width||0)/2,t.globalY=(e.y||0)+(e.globalY||0)+(t.localOffsetY||0)+this.getPinToTopNodeVerticalSpace(t)-(t.height||0)/2+10)}E(t)&&this.updateNodeOffset(t)}}clearLayoutData(e,o){let t=[];e==null?t=this.modelGraph.rootNodes.map(r=>r.id):t=e.nsChildrenIds||[],o&&e!=null&&(e.expanded=!1,delete this.modelGraph.edgesByGroupNodeIds[e.id]);for(let r of t){let s=this.modelGraph.nodesById[r];s&&(s.width=void 0,s.height=void 0,E(s)&&s.expanded&&this.clearLayoutData(s,o))}}getPinToTopNodeVerticalSpace(e){return(e.height||0)+20}getTargetGroupNodeHeight(e,o){let t=F(o.label),r=e.height+ae+xe+t;return o.pinToTopOpNode&&(r+=this.getPinToTopNodeVerticalSpace(o.pinToTopOpNode)),r}};function b(n,e,o){let t={eventType:8,paneId:n,label:e,error:o};postMessage(t)}var Xe=/dense<([^>]*)>/,W=class{constructor(e,o,t,r={},s={},a=Ee,d=!1,l=!1,i=!1){this.paneId=e;this.graph=o;this.config=t;this.showOnNodeItemTypes=r;this.nodeDataProviderRuns=s;this.groupNodeChildrenCountThreshold=a;this.testMode=d;this.flattenLayers=l;this.keepLayersWithASingleChild=i;this.nodeLabelsToHide=new Set((this.config?.nodeLabelsToHide||[]).map(c=>c.toLowerCase()))}nodeLabelsToHide;process(){let e=this.createEmptyModelGraph();return this.processNodes(e),this.processEdgeRelationships(e),b(this.paneId,"Processing nodes and edges"),this.processNamespaceRelationships(e),b(this.paneId,"Processing layer namespaces"),this.generateLayoutGraphConnections(e),b(this.paneId,"Processing layout data"),this.splitLargeGroupNodes(e),b(this.paneId,"Splitting large layers (if any)"),this.populateDescendantsAndCounts(e),e}processNodes(e){let o=new Set;for(let t of this.graph.nodes){let s=t.namespace.split(";").filter(d=>d!=="");s.length>1&&(t.namespace=s[s.length-1]);let a={nodeType:0,id:t.id,namespace:this.flattenLayers?"":t.namespace,savedNamespace:t.namespace,fullNamespace:t.namespace,label:t.label,level:this.getNonEmptyNamespaceComponents(t.namespace).length};if(t.subgraphIds&&t.subgraphIds.length>0&&(a.subgraphIds=t.subgraphIds),this.nodeLabelsToHide.has(t.label.toLowerCase())&&(a.hideInLayout=!0),t.attrs){let d={},l={},i={};for(let c of t.attrs)d[c.key]=this.processAttrValue(c.key,c.value),c.editable&&(l[c.key]=c.editable),c.display_type&&(i[c.key]=c.display_type);a.attrs=d,a.editableAttrs=l,a.attrDisplayTypes=i}if(t.inputsMetadata&&(a.inputsMetadata=this.processMetadataList(t.inputsMetadata)),t.outputsMetadata&&(a.outputsMetadata=this.processMetadataList(t.outputsMetadata)),t.style&&(a.style=t.style),t.config&&(a.config=t.config),e.nodes.push(a),e.nodesById[a.id]=a,!a.hideInLayout&&!this.flattenLayers){let d=this.getAncestorNamespaces(a.namespace);for(let l of d){if(o.has(l))continue;o.add(l);let i=l.split("/"),c=i.splice(-1)[0],p=i.join("/"),g={nodeType:1,id:this.getGroupNodeIdFromNamespace(l),namespace:p,label:c,level:i.length,expanded:!1};e.nodes.push(g),e.nodesById[g.id]=g}}}}processEdgeRelationships(e){for(let o of this.graph.nodes){let t=e.nodesById[o.id];if(t)for(let r of o.incomingEdges||[]){let s=r.sourceNodeId,a=e.nodesById[s];a&&(t.incomingEdges==null&&(t.incomingEdges=[]),t.incomingEdges.find(d=>d.sourceNodeId===s&&d.sourceNodeOutputId===r.sourceNodeOutputId&&d.targetNodeInputId===r.targetNodeInputId)==null&&t.incomingEdges.push({...r}),a.outgoingEdges==null&&(a.outgoingEdges=[]),a.outgoingEdges.find(d=>d.targetNodeId===t.id&&d.sourceNodeOutputId===r.sourceNodeOutputId&&d.targetNodeInputId===r.targetNodeInputId)==null&&a.outgoingEdges.push({targetNodeId:t.id,sourceNodeOutputId:r.sourceNodeOutputId,targetNodeInputId:r.targetNodeInputId}))}}}processNamespaceRelationships(e){for(let o of e.nodes){if(y(o)&&o.hideInLayout)continue;let t=o.namespace;if(t===""){e.rootNodes.push(o);continue}let r=this.getGroupNodeIdFromNamespace(t),s=e.nodesById[r];s?o.nsParentId=s.id:console.warn(`Failed to find the NS parent of node "${o.id}": "${r}"`),s&&(s.nsChildrenIds==null&&(s.nsChildrenIds=[]),s.nsChildrenIds.includes(o.id)||(s.nsChildrenIds.push(o.id),y(o)&&o.config?.pinToGroupTop&&(s.pinToTopOpNode=o)))}if(!this.keepLayersWithASingleChild)for(;;){let o=0;for(let t of e.nodes)if(E(t)&&t.nsChildrenIds!=null&&t.nsChildrenIds.length===1){let r=e.nodesById[t.nsChildrenIds[0]];if(y(r)){o++;let s=e.nodes.indexOf(t);s>=0&&e.nodes.splice(s,1),delete e.nodesById[t.id];let a=r.namespace,d=this.getNonEmptyNamespaceComponents(a);d.pop(),r.namespace=d.join("/"),r.savedNamespace=r.namespace,r.level=d.length,r.nsParentId=t.nsParentId;let l=e.rootNodes.indexOf(t);if(l>=0&&(e.rootNodes.splice(l,1),e.rootNodes.push(r)),t.nsParentId){let i=e.nodesById[t.nsParentId],c=i.nsChildrenIds.indexOf(t.id);i.nsChildrenIds.splice(c,1),i.nsChildrenIds.push(r.id)}}}if(o===0)break}}generateLayoutGraphConnections(e){e.layoutGraphEdges={};let o=[];for(let s of e.nodes){if(!y(s)||s.hideInLayout)continue;(s.incomingEdges||[]).filter(d=>!e.nodesById[d.sourceNodeId].hideInLayout).length===0&&o.push(s)}let t=[...o],r=new Set;for(;t.length>0;){let s=t.shift();if(s==null||s.hideInLayout||r.has(s.id))continue;r.add(s.id);let a=s.outgoingEdges||[];for(let d of a){let l=e.nodesById[d.targetNodeId];if(l.hideInLayout)continue;let i=Ae(s.namespace,l.namespace),c=ne(i,s.namespace),p=c===""?s.id:`${i}${i===""?"":"/"}${c}___group___`,g=ne(i,l.namespace),f=g===""?l.id:`${i}${i===""?"":"/"}${g}___group___`,N=i===""?"":`${i}___group___`;e.layoutGraphEdges[N]==null&&(e.layoutGraphEdges[N]={}),e.layoutGraphEdges[N][p]==null&&(e.layoutGraphEdges[N][p]={}),e.layoutGraphEdges[N][p][f]=!0}for(let d of a){let l=e.nodesById[d.targetNodeId];t.push(l)}}}splitLargeGroupNodes(e){let o=[void 0],t=!1;for(;o.length>0;){let r=o.shift(),s=r==null?e.rootNodes:(r.nsChildrenIds||[]).map(a=>e.nodesById[a]);if(s.length>this.groupNodeChildrenCountThreshold){t=!0;let a=ie(r?.id||"",s,e,this.showOnNodeItemTypes,this.nodeDataProviderRuns,void 0,this.testMode,!0,this.config),d=[];for(let f of Object.keys(a.nodes))a.incomingEdges[f]==null&&d.push(e.nodesById[f]);let l=[],i=[],c=new Set,p=f=>{if(c.has(f))return;c.add(f);let N=e.nodesById[f];i.push(N),i.length===this.groupNodeChildrenCountThreshold&&(l.push(i),i=[]);for(let _ of a.outgoingEdges[N.id]||[])p(_)};for(let f of d)p(f.id);i.length<this.groupNodeChildrenCountThreshold&&i.length>0&&l.push(i);let g=[];for(let f=0;f<l.length;f++){let N=l[f],_=r==null?"":`${r.namespace}/${r.label}`,u=`section_${f+1}_of_${l.length}`,h=r==null?`${u}___group___`:`${_}/${u}___group___`,m={nodeType:1,id:h,label:u,namespace:_,level:_.split("/").filter(I=>I!=="").length,nsParentId:r?.id,nsChildrenIds:N.map(I=>I.id),expanded:!1,sectionContainer:!0};g.push(m),e.nodes.push(m),e.nodesById[m.id]=m,e.artificialGroupNodeIds==null&&(e.artificialGroupNodeIds=[]),e.artificialGroupNodeIds.push(m.id);for(let I of N)I.nsParentId=m.id;let O=h.replace("___group___",""),M=I=>{if(I.namespace===""?I.namespace=O:r==null?I.namespace=`${O}/${I.namespace}`:I.namespace=(I.nsParentId||"").replace("___group___",""),I.level=I.namespace.split("/").filter(w=>w!=="").length,E(I)){let w=I.id;if(delete e.nodesById[I.id],I.id=`${I.namespace}/${I.label}___group___`,e.nodesById[I.id]=I,I.nsParentId){let P=e.nodesById[I.nsParentId],v=(P.nsChildrenIds||[]).indexOf(w);v>=0&&((P.nsChildrenIds||[])[v]=I.id)}for(let P of I.nsChildrenIds||[]){let v=e.nodesById[P];v!=null&&(v.nsParentId=I.id,M(v))}}};for(let I of N)M(I);if(r==null){for(let I of N){let G=e.rootNodes.indexOf(I);G>=0&&e.rootNodes.splice(G,1)}m.namespace===""&&e.rootNodes.push(m)}s=g}r!=null&&(r.nsChildrenIds=g.map(f=>f.id))}for(let a of s)E(a)&&o.push(a)}t&&this.generateLayoutGraphConnections(e)}populateDescendantsAndCounts(e){let o=Number.MAX_VALUE,t=Number.NEGATIVE_INFINITY;for(let r of e.nodes)if(E(r)){let s=[];this.gatherDescendants(e,r,s),r.descendantsNodeIds=s.map(d=>d.id),r.descendantsOpNodeIds=s.filter(d=>d.nodeType===0).map(d=>d.id);let a=(r.descendantsOpNodeIds||[]).length;o=Math.min(a,o),t=Math.max(a,t)}e.minDescendantOpNodeCount=o,e.maxDescendantOpNodeCount=t}createEmptyModelGraph(){let e={id:this.graph.id,collectionLabel:this.graph.collectionLabel||"",nodes:[],nodesById:{},rootNodes:[],edgesByGroupNodeIds:{},layoutGraphEdges:{},minDescendantOpNodeCount:-1,maxDescendantOpNodeCount:-1};return this.graph.groupNodeAttributes&&(e.groupNodeAttributes=this.graph.groupNodeAttributes),e}getAncestorNamespaces(e){let o=this.getNonEmptyNamespaceComponents(e),t=[];for(;o.length>0;)t.push(o.join("/")),o.pop();return t}getNonEmptyNamespaceComponents(e){return e.split("/").filter(o=>o!=="")}getGroupNodeIdFromNamespace(e){return`${e}___group___`}gatherDescendants(e,o,t){for(let r of o.nsChildrenIds||[]){let s=e.nodesById[r];(E(s)||y(s)&&!s.hideInLayout)&&t.push(s),E(s)&&this.gatherDescendants(e,s,t)}}processAttrValue(e,o){if(o.startsWith("dense<")){let t=o.match(Xe);if(t!=null&&t.length>1){let r=t[1];return Ge(r)}}else if(e===H)return Ge(o);return o.replaceAll('"',"")||"<empty>"}processMetadataList(e){let o={};for(let t of e){let r={};for(let s of t.attrs){let a=s.key,d=s.value;a==="tensor_shape"&&(a="shape",d=d.replace("tensor<","").replace(">","").replace("*","\u2217").split("x").join(" x ")),r[a]=d}o[t.id]=r}return o}};function Ge(n){try{return JSON.stringify(JSON.parse(n),null,2).replaceAll("\\n",`
`).trim()}catch{return n}}var X=10000019,K=class{constructor(e){this.modelGraph=e}markIdenticalGroups(){let e={};for(let t of this.modelGraph.nodes){if(!E(t))continue;let r=0,s=(t.descendantsOpNodeIds||[]).map(d=>this.modelGraph.nodesById[d]).filter(d=>!d.hideInLayout),a=new Set(s.map(d=>d.id));for(let d of s)r=(r+this.getNodeHash(d,a))%X;for(let d of s)for(let l of d.outgoingEdges||[]){let i=l.targetNodeId;if(!a.has(i))continue;let c=this.modelGraph.nodesById[i];r=(r+this.getEdgeHash(d,c))%X}e[r]||(e[r]=[]),e[r].push(t)}let o=0;for(let t of Object.values(e))if(!(t.length<=1)&&!(t.length===2&&(t[0].nsParentId===t[1].id||t[1].nsParentId===t[0].id))){for(let r of t)r.identicalGroupIndex=o;o++}}getNodeHash(e,o){let t=0;t=this.addToHash(t,e.label);let r=0;for(let a of e.incomingEdges||[]){let d=a.sourceNodeId;if(o.has(d)){let l=this.modelGraph.nodesById[d];t=this.addToHash(t,`in ${l.label}`),r++}}let s=0;for(let a of e.outgoingEdges||[]){let d=a.targetNodeInputId;if(o.has(d)){let l=this.modelGraph.nodesById[d];t=this.addToHash(t,`out ${l.label}`),s++}}return t=this.addToHash(t,`${r}`),t=this.addToHash(t,`${s}`),t}getEdgeHash(e,o){return this.genHash(e.label+o.label)%X}genHash(e){let o=5381;e=e||"";for(let t=0,r=e.length;t<r;t++)o+=(o<<5)+e.charCodeAt(t);return o&2147483647}addToHash(e,o){return(e+this.genHash(o))%X}};try{importScripts("/static_files/worker_deps.js")}catch(n){console.error(`Failed to import libs: ${n}`)}var le={};self.addEventListener("message",n=>{let e=n.data;switch(e.eventType){case 0:{let o=Ke(e.paneId,e.graph,e.showOnNodeItemTypes,e.nodeDataProviderRuns,e.config,e.groupNodeChildrenCountThreshold,e.flattenLayers,e.keepLayersWithASingleChild,e.initialLayout);k(o,e.paneId);let t={eventType:1,modelGraph:o,paneId:e.paneId};postMessage(t);break}case 9:{let o=Q(e.modelGraphId,e.paneId),t=JSON.parse(JSON.stringify(o));k(t,e.rendererId);let r={eventType:10,modelGraph:o,paneId:e.paneId,rendererId:e.rendererId,groupNodeId:e.groupNodeId,initialPosition:e.initialPosition};postMessage(r);break}case 2:{let o=Q(e.modelGraphId,e.rendererId),t=[];e.expand?t=Qe(o,e.groupNodeId,e.showOnNodeItemTypes,e.nodeDataProviderRuns,e.selectedNodeDataProviderRunId,e.all===!0,e.config):t=qe(o,e.groupNodeId,e.showOnNodeItemTypes,e.nodeDataProviderRuns,e.selectedNodeDataProviderRunId,e.all===!0,e.config),k(o,e.rendererId);let r={eventType:3,modelGraph:o,expanded:e.expand,groupNodeId:e.groupNodeId,rendererId:e.rendererId,deepestExpandedGroupNodeIds:t};postMessage(r);break}case 4:{let o=Q(e.modelGraphId,e.rendererId);je(o,e.showOnNodeItemTypes,e.nodeDataProviderRuns,e.selectedNodeDataProviderRunId,e.targetDeepestGroupNodeIdsToExpand,e.clearAllExpandStates,e.config),k(o,e.rendererId);let t={eventType:5,modelGraph:o,selectedNodeId:e.selectedNodeId,rendererId:e.rendererId,forRestoringUiState:e.forRestoringUiState,rectToZoomFit:e.rectToZoomFit,forRestoringSnapshotAfterTogglingFlattenLayers:e.forRestoringSnapshotAfterTogglingFlattenLayers,targetDeepestGroupNodeIdsToExpand:e.targetDeepestGroupNodeIdsToExpand,triggerNavigationSync:e.triggerNavigationSync};postMessage(t);break}case 6:{let o=Q(e.modelGraphId,e.rendererId),t=ze(o,e.showOnNodeItemTypes,e.nodeDataProviderRuns,e.selectedNodeDataProviderRunId,e.nodeId,e.config);k(o,e.rendererId);let r={eventType:7,modelGraph:o,nodeId:e.nodeId,rendererId:e.rendererId,deepestExpandedGroupNodeIds:t,noNodeShake:e.noNodeShake,select:e.select};postMessage(r);break}case 11:{le={};break}default:break}});function Ke(n,e,o,t,r,s,a,d,l){let i,p=new W(n,e,r,o,{},s,!1,a,d).process();if(p.nodesById[""]!=null&&(i="Some nodes have empty strings as ids which will cause layout failures. See console for details.",console.warn("Nodes with empty ids",p.nodesById[""])),!i&&l){let f=new R(p,dagre,o,t,void 0);try{f.layout()}catch(N){i=`Failed to layout graph: ${N}`}}return b(n,"Laying out root layer",i),new K(p).markIdenticalGroups(),b(n,"Finding identical layers"),p}function Qe(n,e,o,t,r,s,a){let d=new L(n,dagre,o,t,r,!1,a);if(e!=null){let l,i=n.nodesById[e];if(i&&E(i)){i.expanded=!0;let p=i;for(;;){let f=p.nsChildrenIds||[];if(f.length===1){let N=n.nodesById[f[0]];if(N&&E(N))N.expanded=!0,p=N;else break}else break}let g=[];x(p,n,g),l=g.length===0?[p.id]:g;for(let f of p.descendantsNodeIds||[]){let N=n.nodesById[f];N.width=void 0,N.height=void 0}}if(s){for(let p of i.descendantsNodeIds||[]){let g=n.nodesById[p];E(g)&&(g.expanded=!0)}l=void 0}d.reLayoutGraph(l);let c=[];return x(void 0,n,c),c}else return d.expandAllGroups()}function qe(n,e,o,t,r,s,a){let d=new L(n,dagre,o,t,r,!1,a);if(e!=null){if(s){let l=n.nodesById[e];for(let i of l.descendantsNodeIds||[]){let c=n.nodesById[i];E(c)&&(c.expanded=!1,c.width=void 0,c.height=void 0,delete n.edgesByGroupNodeIds[c.id])}}return d.collapseGroupNode(e)}else return d.collapseAllGroup()}function je(n,e,o,t,r,s,a){new L(n,dagre,e,o,t,!1,a).reLayoutGraph(r,s)}function ze(n,e,o,t,r,s){return new L(n,dagre,e,o,t,!1,s).expandToRevealNode(r)}function k(n,e){le[Se(n.id,e)]=n}function Q(n,e){let o=le[Se(n,e)];if(o==null)throw new Error(`ModelGraph with id "${n}" not found for rendererId "${e}"`);return o}function Se(n,e){return`${n}___${e}`}
