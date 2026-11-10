const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');

// DOM contract harness for frontend/model_windows.js (no browser, no layout). Fixtures below are
// small hand-written shapes that follow the /api/windows and /api/scoreboard contract; no collected data.
class Element{
  constructor(tag,ns=null){this.tagName=tag;this.namespaceURI=ns;this.children=[];this.attributes={};this.listeners={};this.dataset={};
    this._text='';this.className='';this.hidden=false;this.offsetTop=0;this.offsetHeight=0}
  get classList(){const self=this,list=()=>self.className.split(/\s+/).filter(Boolean);
    return {contains:c=>list().includes(c),add:c=>{if(!list().includes(c))self.className=[...list(),c].join(' ')},
      remove:c=>{self.className=list().filter(x=>x!==c).join(' ')},
      toggle:(c,force)=>{const on=force===undefined?!list().includes(c):force;if(on)this.classList.add(c);else this.classList.remove(c);return on}}}
  set textContent(v){this._text=String(v);this.children=[]}
  get textContent(){return this._text+this.children.map(c=>c.textContent).join('')}
  append(...nodes){nodes.forEach(n=>{if(typeof n==='string')n=new Text(n);n.parentNode=this;this.children.push(n)})}
  replaceChildren(...nodes){this.children=[];this._text='';this.append(...nodes)}
  setAttribute(k,v){this.attributes[k]=String(v);if(k==='class')this.className=String(v)}
  removeAttribute(k){delete this.attributes[k]}
  getAttribute(k){return k==='class'?this.className:this.attributes[k]}
  addEventListener(k,fn){(this.listeners[k]??=[]).push(fn)}
  fire(k,event={}){(this.listeners[k]||[]).forEach(fn=>fn({preventDefault(){},stopPropagation(){},key:'',...event}))}
}
class Text{constructor(t){this.textContent=String(t);this.children=[];this.attributes={}}}
function all(e){return [e,...(e.children||[]).flatMap(all)]}
function byClass(root,c){return all(root).filter(e=>e.classList&&e.className.split(/\s+/).includes(c))}

function harness(fetchImpl){
  const ids=new Map();
  for(const id of ['model-windows','windows-status','windows-live','leaderboard','leaderboard-truth','leaderboard-population','leaderboard-axis','leaderboard-notes','leaderboard-tooltip']){
    const e=new Element('div');e.id=id;ids.set(id,e)}
  const document={createElement:t=>new Element(t),createElementNS:(ns,t)=>new Element(t,ns),createTextNode:t=>new Text(t),
    getElementById:id=>ids.get(id)||null,querySelectorAll:()=>[]};
  const opened=[],timers=[];
  const context={document,console,AbortController,Date,
    setTimeout:(fn,ms)=>{timers.push({fn,ms,live:true});return timers.length},clearTimeout:id=>{if(timers[id-1])timers[id-1].live=false},
    fetch:(...args)=>fetchImpl(...args),loadExplanation:(...args)=>opened.push(args)};
  vm.createContext(context);
  vm.runInContext(fs.readFileSync('frontend/model_windows.js','utf8'),context);
  const fire=ms=>{const t=timers.filter(t=>t.live&&t.ms===ms);t.forEach(x=>{x.live=false;x.fn()});return t.length};
  return {context,ids,opened,timers,fire,state:()=>vm.runInContext('mwState',context),card:m=>vm.runInContext(`mwCards[${JSON.stringify(m)}]`,context)};
}
const ok=body=>Promise.resolve({ok:true,status:200,json:async()=>JSON.parse(JSON.stringify(body))});
const fail=(status,detail)=>Promise.resolve({ok:false,status,json:async()=>({detail})});
const hang=opts=>new Promise((_,reject)=>opts?.signal?.addEventListener('abort',()=>reject(Object.assign(Error('aborted'),{name:'AbortError'}))));
const settle=()=>new Promise(r=>setImmediate(r));

function features(flat=false){return Array.from({length:60},(_,i)=>{const mid=flat?.41:.40+.01*Math.sin(i/6);return [mid-.02,mid+.02,.04,mid,500,400,.1]})}
function linear(scale){
  const names=['bid_last','ask_last','spread_last','midpoint_last','imbalance_mean','spread_mean','bid_depth_last','ask_depth_last'];
  return {kind:'linear',intercept:[-16.9,-17.5],reconstructed_raw:[-16.4,-17.2],
    features:names.map((name,i)=>({name,raw:.4,standardized:.1*i,coefficient:[1,-1],contribution:[scale*(i%2?-1:1)*(1+i/4),scale*(i%3?.5:-.5)]}))};
}
const tree={kind:'tree',leaf:7,prediction:[-15.7,-15.6],steps:[
  {node:0,leaf:false,feature:'spread_last',observed:.04,threshold:.335,direction:'left',training_samples:400,value:[-17,-17.5]},
  {node:1,leaf:false,feature:'bid_depth_last',observed:669,threshold:107004.9,direction:'left',training_samples:300,value:[-16,-16.8]},
  {node:4,leaf:false,feature:'imbalance_last',observed:.1,threshold:.2,direction:'left',training_samples:100,value:[-15.5,-16]},
  {node:7,leaf:true,training_samples:40,value:[-15.7,-15.6]}]};
function calc(y,n){return {raw:[y-2,n-2.5],clipped_raw:[y-2,n-2.5],calibration_correction:[2,2.5],final:[y,n],signal:'WAIT',margin_cents:1,cost_buffer_cents:4}}
function windowsPayload(id='abc123',horizon=30,{flat=false,shift=0}={}){
  const path=k=>Array.from({length:horizon+1},(_,t)=>flat?.41:+(.41+.002*k*Math.sin(t/3)).toFixed(4));
  const details={mean:{kind:'mean',training_mean:[-16.97,-17.58]},ridge:linear(.4),sgd:linear(.2),decision_tree:tree,
    random_forest:{kind:'forest',selected_tree:0,path:tree,tree_outputs:Array.from({length:24},(_,i)=>[-18+i*.15,-18.2+i*.1])},
    gradient_boosting:{kind:'boosting',stages:Array.from({length:150},(_,i)=>[-17+i*.01,-17.5+i*.008])},
    gru:{kind:'gru',feature_names:['bid','ask','spread','midpoint','bid_depth','ask_depth','imbalance'],sensitivity:[.1,.2,.3,.2,0,0,.02],
      time_sensitivity:Array.from({length:60},(_,i)=>i/59*3)},
    monte_carlo:{kind:'monte_carlo',seed:1,path_count:2048,preview_indices:[0,64,128,192],preview_paths:[0,1,2,3].map(path),
      preview_returns:[[-8,-8]],scenario_percentiles:[[-9,-9],[-8,-8],[-7,-7]],convergence_yes:[-8,-8],histogram:{counts:[1],edges:[0,1]}}};
  const models={};Object.entries(details).forEach(([m,d],i)=>{models[m]={calculation:calc(-14.8-i*.05+shift,-14.9+i*.03+shift),details:d}});
  return {forecast_id:id,ticker:'T-'+id,horizon,input_end:1,model_run:'run',input:{features:features(flat),summary_names:[],summary_values:[]},models,errors:{}};
}
function scoreboard(){
  const rows=[['ridge',.068,[.027,.111],false],['sgd',.04,[-.01,.09],true],['monte_carlo',-.16,[-.21,-.11],false],['mean',-.082,[-.12,-.04],false],
    ['random_forest',.051,[.01,.09],false],['gru',.031,[-.02,.08],true],['decision_tree',.012,[-.05,.07],true],['gradient_boosting',.039,[-.01,.08],true]];
  const avg=-7.3;
  const models=rows.map(([model,rel,ci,noise])=>({model,label:model.toUpperCase(),closed_mean_cents:avg+rel,lower_mean_cents:-13,upper_mean_cents:1,
    known_fraction:.9,yes_fraction:.55,relative_cents:rel,relative_ci_cents:ci,within_noise:noise}));
  [...models].sort((a,b)=>b.relative_cents-a.relative_cents).forEach((r,i)=>r.rank=i+1);
  const h={n:1000,sessions:50,average_closed_mean_cents:avg,models};
  // 60s rankings differ from 30s so tests can tell which horizon a chip uses.
  const m60=models.map(r=>({...r,relative_cents:-r.relative_cents,rank:9-r.rank}));
  return {run:'run',population:'Exploratory validation split',policy:'Always take the side with the higher forecast',
    bootstrap:{resamples:1000,unit:'session_id',seed:1,interval:[2.5,97.5]},horizons:{'30':h,'60':{...h,average_closed_mean_cents:-7.9,models:m60}},notes:['Every model is negative.']};
}
function market(id='abc123',ticker='T-abc123',{yes30=-14.8,yes60=-30}={}){
  return {ticker,predictions:[30,60].flatMap(h=>['mean','ridge','sgd','decision_tree','random_forest','gradient_boosting','gru','monte_carlo']
    .map(model=>({model,label:model,horizon:h,yes:h===30?yes30:yes60,no:(h===30?yes30:yes60)-.1,signal:'WAIT',forecast_id:id})))};
}
function router(map){return (url,opts)=>{for(const [k,v] of Object.entries(map))if(url.includes(k))return typeof v==='function'?v(url,opts):ok(v);return fail(404,'not found')}}
function attributeProblems(root){
  const bad=[];all(root).forEach(e=>Object.entries(e.attributes||{}).forEach(([k,v])=>{if(/NaN|undefined|Infinity/.test(v))bad.push(`${e.tagName}.${k}=${v}`)}));return bad}
const svgOf=c=>all(c.viz).find(e=>e.tagName==='svg');

const MODELS=['mean','ridge','sgd','decision_tree','random_forest','gradient_boosting','gru','monte_carlo'];

test('all eight windows render their own SVG from actual details, with no NaN attributes',async()=>{
  const urls=[];const h=harness(url=>{urls.push(url);return router({'/api/scoreboard':scoreboard(),'/api/windows/abc123':windowsPayload()})(url)});
  await h.context.updateModelWindows(market(),30,true);await settle();
  for(const m of MODELS){
    const c=h.card(m),svg=svgOf(c);
    assert.ok(svg,`${m} has an svg`);assert.equal(svg.attributes['data-window'],m);
    assert.ok(svg.attributes['aria-label'].length>10,`${m} has an aria label`);
    assert.ok(all(svg).filter(e=>['rect','circle','line','polyline','path'].includes(e.tagName)).length>=3,`${m} draws marks`);
    assert.ok(c.caption.textContent.length>10,`${m} caption`);
    assert.match(c.forecast.textContent,/^Final YES −1\d\.\d\d¢ · NO −1\d\.\d\d¢$/);
    assert.match(c.chip.textContent,/vs avg/);
    assert.equal(c.open.tagName,'button');assert.match(c.open.attributes['aria-label'],/Show the full .* calculation/);
  }
  assert.deepEqual(attributeProblems(h.ids.get('model-windows')),[]);
  assert.deepEqual(attributeProblems(h.ids.get('leaderboard')),[]);
  assert.equal(urls.filter(u=>u.includes('/api/windows/')).length,1);
  assert.match(urls.find(u=>u.includes('/api/windows/')),/horizon=30/);
});

test('renderers show distinct, data-tied mechanisms',async()=>{
  const p=windowsPayload();const h=harness(router({'/api/scoreboard':scoreboard(),'/api/windows/abc123':p}));
  await h.context.updateModelWindows(market(),30,true);await settle();
  const svg=m=>svgOf(h.card(m));
  const meanLine=all(svg('mean')).find(e=>e.tagName==='line'&&/mw-accent-line/.test(e.className));assert.equal(meanLine.attributes.y1,meanLine.attributes.y2);
  assert.match(svg('mean').textContent,/never uses/);assert.match(svg('mean').textContent,/height schematic/);
  assert.equal(all(svg('ridge')).filter(e=>e.tagName==='g').length,8);
  assert.match(h.card('ridge').caption.textContent,/Starts at −16\.90¢.*all 8 sum to/);
  const d=p.models.sgd.details,net=d.intercept[0]-d.intercept[1]+d.features.reduce((a,f)=>a+f.contribution[0]-f.contribution[1],0);
  assert.match(svg('sgd').textContent,new RegExp(`net ${net<0?'−':'\\+'}${Math.abs(net).toFixed(2)}¢ toward ${net>=0?'YES':'NO'}`));
  assert.ok(all(svg('sgd')).some(e=>/mw-yes/.test(e.className)));
  assert.equal(all(svg('decision_tree')).filter(e=>e.tagName==='circle').length,tree.steps.length);
  assert.match(svg('decision_tree').textContent,/leaf 7/);assert.match(svg('decision_tree').textContent,/layout schematic/);
  // thresholds carry units: prices in cents, depth in contracts, imbalance as a ratio
  assert.match(h.card('decision_tree').caption.textContent,/First: spread_last 4\.0¢ ≤ 33\.5¢/);
  const titles=all(svg('decision_tree')).filter(e=>e.tagName==='title').map(t=>t.textContent).join('|');
  assert.match(titles,/bid_depth_last: observed 669 contracts ≤ threshold 107,004\.9 contracts/);
  assert.match(titles,/imbalance_last: observed 0\.100 \(imbalance ratio\)/);
  const outs=p.models.random_forest.details.tree_outputs.map(v=>v[0]),mean=outs.reduce((a,b)=>a+b,0)/outs.length;
  assert.equal(all(svg('random_forest')).filter(e=>e.tagName==='circle'&&/mw-fly/.test(e.className)).length,outs.length);
  assert.match(svg('random_forest').textContent,new RegExp(`mean of 24 trees · −${Math.abs(mean).toFixed(2)}¢`));
  assert.match(svg('random_forest').textContent,/above forest mean/);
  assert.ok(!all(svg('random_forest')).some(e=>/mw-pos|mw-neg/.test(e.className)),'forest does not reuse leaderboard colors');
  assert.match(svg('gradient_boosting').textContent,/stage 150 · −15\.51¢/);
  assert.match(h.card('gradient_boosting').caption.textContent,/1 in 3 drawn.*from stage 1 \(already baseline \+ first tree\)/);
  const bars=all(svg('gru')).filter(e=>e.tagName==='rect'&&/mw-pulse/.test(e.className));assert.equal(bars.length,60);
  assert.ok(Number(bars[59].attributes.height)>Number(bars[10].attributes.height));
  assert.ok(all(svg('gru')).some(e=>/mw-scan/.test(e.className)));
  assert.doesNotMatch(h.card('gru').caption.textContent,/strongest input/);assert.match(h.card('gru').caption.textContent,/peaks at observation 60\/60/);
  const lines=all(svg('monte_carlo')).filter(e=>e.tagName==='polyline');assert.equal(lines.length,4);
  assert.equal(new Set(lines.map(l=>l.attributes.points.split(' ')[0])).size,1);
});

test('calculation strip shows raw → calibration → final for both sides and the lean matches it',async()=>{
  const p=windowsPayload();p.models.random_forest.calculation={raw:[-16.56,-16.75],clipped_raw:[-16.56,-16.75],calibration_correction:[1.87,2.15],final:[-14.69,-14.60],signal:'WAIT'};
  p.models.monte_carlo.calculation={raw:[-130,-8],clipped_raw:[-104,-8],calibration_correction:[-6.9,-6.81],final:[-110.9,-14.81],signal:'WAIT'};
  const h=harness(router({'/api/scoreboard':scoreboard(),'/api/windows/':p}));
  await h.context.updateModelWindows(market(),30,true);await settle();
  const rf=h.card('random_forest');
  assert.deepEqual(rf.calc.children.map(s=>s.textContent),['YES −16.56¢ raw + 1.87¢ calib. = −14.69¢','NO −16.75¢ raw + 2.15¢ calib. = −14.60¢']);
  assert.equal(rf.forecast.textContent,'Final YES −14.69¢ · NO −14.60¢');
  assert.match(rf.lean.textContent,/^Leans NO by 0\.09¢ · loss either way$/);
  const mcLine=h.card('monte_carlo').calc.children[0];
  assert.equal(mcLine.textContent,'YES −104.00¢ bounded raw − 6.90¢ calib. = −110.90¢');
  assert.match(mcLine.attributes.title,/raw −130\.0000¢ bounded to −104\.0000¢/);
  // the final value lives in its own non-shrinking span so it can never be ellipsized away
  assert.equal(byClass(mcLine,'mw-calc-final')[0].textContent,'= −110.90¢');
});

test('late windows response for an old forecast is ignored',async()=>{
  let releaseOld;const h=harness(url=>{
    if(url.includes('/api/scoreboard'))return ok(scoreboard());
    if(url.includes('/api/windows/old'))return new Promise(r=>{releaseOld=r});
    return ok(windowsPayload('new'))});
  const pending=h.context.updateModelWindows(market('old','T-old'),30,true);
  await h.context.updateModelWindows(market('new','T-new'),30,true);
  releaseOld({ok:true,status:200,json:async()=>windowsPayload('old')});await pending;await settle();
  assert.match(h.ids.get('windows-status').textContent,/T-new/);
  assert.equal(h.state().payload.forecast_id,'new');
  assert.ok(svgOf(h.card('ridge')));
});

test('same-market refresh keeps drawing and caption and does not re-animate; market or horizon change does',async()=>{
  const pending={};const h=harness((url,opts)=>{
    if(url.includes('scoreboard'))return ok(scoreboard());
    const id=url.match(/windows\/([^?]+)/)[1],hz=url.includes('horizon=60')?60:30;
    if(id==='slow'&&hz===30)return new Promise(r=>{pending.slow=()=>r({ok:true,status:200,json:async()=>windowsPayload('slow',hz,{shift:.5})})});
    return ok(windowsPayload(id,hz))});
  await h.context.updateModelWindows(market('f1','T-A'),30,true);await settle();
  const first=svgOf(h.card('random_forest'));assert.match(first.className,/mw-anim/);
  const caption=h.card('random_forest').caption.textContent;
  const snap=m=>{const c=h.card(m);return [c.caption.textContent,c.calc.textContent,c.forecast.textContent,c.lean.textContent,c.card.className]};
  const before=Object.fromEntries(MODELS.map(m=>[m,snap(m)]));
  // next 5 s forecast for the same market: while loading, keep the old drawing and caption with no message
  const loading=h.context.updateModelWindows(market('slow','T-A'),30,true);
  assert.equal(svgOf(h.card('random_forest')),first);assert.equal(h.card('random_forest').caption.textContent,caption);
  assert.ok(!h.card('random_forest').viz.classList.contains('mw-refreshing'));
  // R2-1: caption, strip, foot, lean and card classes all stay on the previous forecast while the next one loads
  for(const m of MODELS)assert.deepEqual(snap(m),before[m],`${m} unchanged while loading`);
  assert.match(h.card('ridge').forecast.textContent,/^Final YES −14\.85¢/);
  assert.match(h.ids.get('windows-status').textContent,/updating/);
  pending.slow();await loading;await settle();
  const second=svgOf(h.card('random_forest'));
  assert.notEqual(second,first);assert.doesNotMatch(second.className,/mw-anim/);assert.equal(second.attributes['data-forecast'],'slow');
  // ...then everything swaps together to the new payload (shifted by +0.5¢)
  assert.match(h.card('ridge').forecast.textContent,/^Final YES −14\.35¢/);
  assert.match(h.card('ridge').calc.textContent,/= −14\.35¢/);
  // announcements happen per market/horizon, not per forecast
  assert.match(h.ids.get('windows-live').textContent,/loaded for T-A · 30s/);
  // horizon change and market change animate again
  await h.context.updateModelWindows(market('slow','T-A'),60,true);await settle();
  assert.match(svgOf(h.card('random_forest')).className,/mw-anim/);
  await h.context.updateModelWindows(market('f2','T-B'),60,true);await settle();
  assert.match(svgOf(h.card('random_forest')).className,/mw-anim/);
  // re-rendering chips (scoreboard arrival) never rebuilds an up-to-date drawing
  const now=svgOf(h.card('ridge'));vm.runInContext('mwRenderAll()',h.context);assert.equal(svgOf(h.card('ridge')),now);
});

test('after a horizon switch the foot never shows the previous horizon payload numbers',async()=>{
  let release60;const h=harness(url=>{
    if(url.includes('scoreboard'))return ok(scoreboard());
    if(url.includes('horizon=60'))return new Promise(r=>{release60=()=>r({ok:true,status:200,json:async()=>windowsPayload('abc123',60,{shift:-15})})});
    return ok(windowsPayload('abc123',30))});
  await h.context.updateModelWindows(market(),30,true);await settle();
  assert.match(h.card('mean').forecast.textContent,/YES −14\.80¢/);
  const pending=h.context.updateModelWindows(market(),60,true);
  for(const m of MODELS){const c=h.card(m);
    assert.equal(c.forecast.textContent,'Final YES −30.00¢ · NO −30.10¢',`${m} uses 60s state rows while loading`);
    assert.match(c.calc.textContent,/calculation steps.*updating/);
    assert.ok(c.viz.classList.contains('mw-refreshing'),'old-horizon drawing is dimmed');
    assert.equal(c.caption.textContent,'Loading T-abc123 · 60s…','old-horizon caption is replaced')}
  assert.match(h.card('ridge').rank.textContent,/at 60s/);
  release60();await pending;await settle();
  assert.match(h.card('mean').forecast.textContent,/YES −29\.80¢/);
  assert.ok(!h.card('mean').viz.classList.contains('mw-refreshing'));
});

test('hung windows and scoreboard requests time out after 15 s and the windows retry',async()=>{
  let windowsCalls=0,scoreCalls=0;
  const h=harness((url,opts)=>{if(url.includes('scoreboard')){scoreCalls++;return hang(opts)}windowsCalls++;return hang(opts)});
  const pending=h.context.updateModelWindows(market(),30,true);
  assert.equal(h.fire(15000),2,'one 15 s timer per request');await pending;await settle();
  assert.match(h.card('ridge').viz.textContent,/request timed out after 15 s/);
  assert.match(h.ids.get('leaderboard').textContent,/Leaderboard unavailable: request timed out/);
  assert.ok(h.timers.some(t=>t.live&&t.ms===30000),'scoreboard retry scheduled');
  // no retry before retryAt, then a retry on the next poll
  await h.context.updateModelWindows(market(),30,false);assert.equal(windowsCalls,1);
  vm.runInContext('mwState.retryAt=0',h.context);
  h.context.updateModelWindows(market(),30,false);assert.equal(windowsCalls,2);
  assert.equal(scoreCalls,1);
});

test('expired (410) and bad-request forecasts are not retried; unreadable JSON is reported as such',async()=>{
  let calls=0;const h=harness(url=>{if(url.includes('scoreboard'))return ok(scoreboard());calls++;return fail(410,'Forecast expired or unknown')});
  await h.context.updateModelWindows(market(),30,true);await settle();
  for(const m of MODELS)assert.match(h.card(m).viz.textContent,/Calculations unavailable: Forecast expired/);
  vm.runInContext('mwState.retryAt=0',h.context);
  await h.context.updateModelWindows(market(),30,true);await h.context.updateModelWindows(market(),30,true);
  assert.equal(calls,1);
  const bad=harness(url=>url.includes('scoreboard')?ok(scoreboard()):Promise.resolve({ok:true,status:200,json:async()=>{throw SyntaxError('bad')}}));
  await bad.context.updateModelWindows(market(),30,true);await settle();
  assert.match(bad.card('mean').viz.textContent,/unreadable response/);
});

test('chips follow the horizon even when there is no forecast',async()=>{
  const h=harness(router({'/api/scoreboard':scoreboard()}));await settle();
  await h.context.updateModelWindows({ticker:'X',predictions:[]},30,true);
  assert.equal(h.card('ridge').chip.textContent,'+0.07¢ vs avg');assert.equal(h.card('ridge').rank.textContent,'#1 at 30s');
  await h.context.updateModelWindows({ticker:'X',predictions:[]},60,true);
  assert.equal(h.card('ridge').chip.textContent,'−0.07¢ vs avg');assert.equal(h.card('ridge').rank.textContent,'#8 at 60s');
  assert.match(h.card('ridge').chip.attributes.title,/^60s: .*model average −7\.900¢/);
  assert.match(h.ids.get('windows-status').textContent,/60s/);
});

test('per-model failure shows an explicit error card and never draws substitute data',async()=>{
  const p=windowsPayload();delete p.models.gradient_boosting;delete p.models.gru;
  p.errors.gradient_boosting='AssertionError: \nNot equal to tolerance rtol=1e-05';
  const h=harness(router({'/api/scoreboard':scoreboard(),'/api/windows/':p}));
  await h.context.updateModelWindows(market(),30,true);await settle();
  const gb=h.card('gradient_boosting');
  assert.ok(!svgOf(gb));
  assert.match(gb.viz.textContent,/failed its consistency check.*AssertionError/);
  assert.ok(!all(h.ids.get('model-windows')).some(e=>e.attributes.role==='alert'),'no per-card alert spam');
  assert.match(h.card('gru').viz.textContent,/No calculation was returned/);
  assert.match(gb.forecast.textContent,/YES −14\.80¢/);assert.match(gb.calc.textContent,/unavailable for this forecast/);
  assert.ok(svgOf(h.card('ridge')));
});

test('malformed numbers become an error card instead of NaN geometry',async()=>{
  const p=windowsPayload();p.models.random_forest.details.tree_outputs[3]=[null,1];p.models.monte_carlo.details.preview_paths[1][4]=null;
  const h=harness(router({'/api/scoreboard':scoreboard(),'/api/windows/':p}));
  await h.context.updateModelWindows(market(),30,true);await settle();
  assert.match(h.card('random_forest').viz.textContent,/could not be drawn/);
  assert.match(h.card('monte_carlo').viz.textContent,/could not be drawn/);
  assert.deepEqual(attributeProblems(h.ids.get('model-windows')),[]);
});

test('mismatched payload is rejected rather than shown for the wrong forecast',async()=>{
  const h=harness(router({'/api/scoreboard':scoreboard(),'/api/windows/':windowsPayload('someone-else')}));
  await h.context.updateModelWindows(market(),30,true);await settle();
  assert.match(h.card('mean').viz.textContent,/did not match/);
});

test('loading and waiting states are explicit',async()=>{
  let release;const h=harness(url=>url.includes('scoreboard')?ok(scoreboard()):new Promise(r=>{release=r}));
  assert.match(h.card('sgd').viz.textContent,/Waiting for a forecast/);
  const pending=h.context.updateModelWindows(market(),30,true);
  assert.match(h.card('sgd').viz.textContent,/Loading this model/);
  release({ok:true,status:200,json:async()=>windowsPayload()});await pending;
  await h.context.updateModelWindows({ticker:'X',predictions:[]},30,true);
  assert.match(h.card('sgd').viz.textContent,/Waiting for a forecast/);
});

test('flat Monte Carlo preview says so and adds no jitter',async()=>{
  const h=harness(router({'/api/scoreboard':scoreboard(),'/api/windows/':windowsPayload('abc123',30,{flat:true})}));
  await h.context.updateModelWindows(market(),30,true);await settle();
  const svg=svgOf(h.card('monte_carlo'));
  assert.match(svg.textContent,/All 4 shown paths stay at 41\.0¢/);assert.match(svg.textContent,/no jitter added/);
  assert.equal(all(svg).filter(e=>e.tagName==='polyline').length,0);
  const line=all(svg).find(e=>e.tagName==='line');assert.equal(line.attributes.y1,line.attributes.y2);
  assert.match(h.card('monte_carlo').caption.textContent,/sampled one-second move was zero/);
});

test('clicking a window or its real button opens the existing explanation dialog for that forecast',async()=>{
  const h=harness(router({'/api/scoreboard':scoreboard(),'/api/windows/':windowsPayload()}));
  await h.context.updateModelWindows(market(),30,true);await settle();
  h.card('ridge').card.fire('click');h.card('gru').open.fire('click');
  assert.equal(h.card('gru').open.attributes['aria-disabled'],'false');
  assert.deepEqual(h.opened.map(a=>[...a]),[['abc123','ridge',30],['abc123','gru',30]]);
});

test('leaderboard is ordered, centered on the average, and labels noise',async()=>{
  const h=harness(router({'/api/scoreboard':scoreboard()}));await settle();
  const list=h.ids.get('leaderboard'),rows=list.children;
  assert.equal(rows.length,8);
  const order=rows.map(r=>r.className.match(/mw-(\w+)/)[1]);
  assert.deepEqual(order,['ridge','random_forest','sgd','gradient_boosting','gru','decision_tree','mean','monte_carlo']);
  const geometry=r=>{const s=byClass(r,'lb-bar')[0].attributes.style;return {left:+s.match(/left:([\d.]+)%/)[1],width:+s.match(/width:([\d.]+)%/)[1]}};
  const best=geometry(rows[0]),worst=geometry(rows[7]);
  assert.equal(best.left,50);assert.ok(Math.abs(worst.left+worst.width-50)<.02);
  assert.ok(worst.width>best.width,'larger magnitude gets a longer bar');
  assert.ok(byClass(rows[0],'lb-pos').length&&byClass(rows[7],'lb-neg').length);
  const sgd=rows[2];assert.match(sgd.textContent,/within noise/);assert.ok(byClass(sgd,'lb-noise').length);
  assert.match(rows[0].textContent,/outside noise/);assert.match(rows[0].textContent,/\+0\.07¢ vs avg/);
  const whisker=byClass(rows[0],'lb-whisker')[0].attributes.style;assert.ok(+whisker.match(/left:([\d.]+)%/)[1]>50);
  assert.equal(h.ids.get('leaderboard-truth').textContent,'On trades whose outcome was observed, every model averages about −7.3¢ after the 4¢ buffer — none shows a profit on known outcomes (unknown outcomes stay intervals; profitability is not established). This view shows who loses least.');
  assert.match(h.ids.get('leaderboard-population').textContent,/Exploratory validation split/);
  assert.match(h.ids.get('leaderboard-axis').textContent,/model average −7\.30¢/);
  assert.match(rows[0].attributes['aria-label'],/Rank 1\. RIDGE: \+0\.068¢ vs the model average \(95% session-bootstrap interval \+0\.027¢ to \+0\.111¢\)/);
  rows[0].fire('focus');const tip=h.ids.get('leaderboard-tooltip');assert.equal(tip.hidden,false);
  assert.match(tip.textContent,/Own average −7\.232¢ per hypothetical trade with a known outcome; counting unknown outcomes, −13\.000¢ to \+1\.000¢/);
  rows[0].fire('blur');assert.equal(tip.hidden,true);
  assert.deepEqual(attributeProblems(list),[]);
});

test('leaderboard follows the horizon toggle and reports unavailability',async()=>{
  const h=harness(router({'/api/scoreboard':scoreboard()}));await settle();
  h.context.updateLeaderboard(60);assert.match(h.ids.get('leaderboard-truth').textContent,/about −7\.9¢/);
  const s=scoreboard();delete s.horizons['60'];
  const missing=harness(router({'/api/scoreboard':s}));await settle();missing.context.updateLeaderboard(60);
  assert.match(missing.ids.get('leaderboard').textContent,/No validation averages are available for the 60s horizon/);
  const down=harness(()=>fail(503,'Scoreboard unavailable: artifacts missing'));await settle();
  assert.match(down.ids.get('leaderboard').textContent,/Leaderboard unavailable: Scoreboard unavailable/);
  assert.ok(down.timers.some(t=>t.live&&t.ms===30000));
  assert.equal(down.card('ridge').chip.textContent,'Ranking unavailable');
});

test('truth line never claims nobody shows a profit when a model averages above zero',async()=>{
  const s=scoreboard();s.horizons['30'].models[0].closed_mean_cents=.5;
  const h=harness(router({'/api/scoreboard':s}));await settle();
  assert.doesNotMatch(h.ids.get('leaderboard-truth').textContent,/none shows a profit/);
  assert.match(h.ids.get('leaderboard-truth').textContent,/1 of 8 models average above zero/);
});

const FIX=process.env.MODEL_WINDOW_FIXTURES;
test('real backend fixtures render without NaN (optional, local only)',{skip:!(FIX&&fs.existsSync(FIX))&&'MODEL_WINDOW_FIXTURES not set'},async()=>{
  const read=f=>JSON.parse(fs.readFileSync(`${FIX}/${f}`,'utf8'));
  for(const h of [30,60]){
    const payload=read(`windows_${h}.json`),state=read('state.json');
    const t=harness(router({'/api/scoreboard':read('scoreboard.json'),'/api/windows/':payload}));
    await t.context.updateModelWindows(state.markets[0],h,true);await settle();t.context.updateLeaderboard(h);
    for(const m of MODELS){const c=t.card(m);
      if(payload.errors?.[m])assert.match(c.viz.textContent,/consistency check/);
      else{assert.ok(svgOf(c),`${m}@${h}: ${c.viz.textContent}`);
        const calc=payload.models[m].calculation;
        assert.equal(c.forecast.textContent,`Final YES ${t.context.mwC(calc.final[0])} · NO ${t.context.mwC(calc.final[1])}`)}}
    assert.deepEqual(attributeProblems(t.ids.get('model-windows')),[]);
    assert.deepEqual(attributeProblems(t.ids.get('leaderboard')),[]);
    assert.equal(t.ids.get('leaderboard').children.length,8);
    assert.match(t.ids.get('leaderboard-truth').textContent,/^On trades whose outcome was observed/);
  }
});

test('"Show the math" is disabled with no forecast and enabled once one exists',async()=>{
  const h=harness(router({'/api/scoreboard':scoreboard(),'/api/windows/':windowsPayload()}));await settle();
  for(const m of MODELS){const o=h.card(m).open;assert.equal(o.attributes['aria-disabled'],'true');assert.ok(o.classList.contains('mw-open-disabled'));assert.equal(o.attributes.title,'No forecast yet')}
  h.card('ridge').open.fire('click');h.card('ridge').card.fire('click');assert.equal(h.opened.length,0);
  await h.context.updateModelWindows(market(),30,true);await settle();
  const o=h.card('ridge').open;assert.equal(o.attributes['aria-disabled'],'false');assert.ok(!o.classList.contains('mw-open-disabled'));
  o.fire('click');assert.deepEqual([...h.opened[0]],['abc123','ridge',30]);
  await h.context.updateModelWindows({ticker:'X',predictions:[]},30,true);
  assert.equal(h.card('ridge').open.attributes['aria-disabled'],'true');
});
