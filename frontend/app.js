let state=null,selected='',horizon=30;
const $=id=>document.getElementById(id), ns='http://www.w3.org/2000/svg';
const cents=v=>`${v>=0?'+':''}${v.toFixed(2)}¢`, clock=t=>new Date(t).toLocaleTimeString();
const SHORT_NAMES={gradient_boosting:'Boosting',random_forest:'Forest',decision_tree:'Tree',monte_carlo:'Monte Carlo',mean:'Mean',gru:'GRU',ridge:'Ridge',sgd:'SGD'};
function node(tag,attrs={},text=''){const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));e.textContent=text;return e}
function emptyChart(svg,message){svg.replaceChildren(node('text',{x:30,y:120,class:'empty'},message))}
function priceChart(m){
  const s=$('price-chart');if(!m)return emptyChart(s,'Waiting for a valid market history');s.replaceChildren();
  const d=m.history,lo=Math.min(...d.map(r=>r.bid))-.5,hi=Math.max(...d.map(r=>r.ask))+.5;
  const x=i=>48+i/(d.length-1)*635,y=v=>205-(v-lo)/(hi-lo)*175;
  for(let i=0;i<4;i++){const v=lo+(hi-lo)*i/3;s.append(node('line',{x1:48,x2:683,y1:y(v),y2:y(v),class:'grid-line'}),node('text',{x:0,y:y(v)+3},v.toFixed(1)+'¢'))}
  const points=key=>d.map((r,i)=>[x(i),y(r[key])]);
  const area=[...points('ask'),...points('bid').reverse()].map(p=>p.join(',')).join(' ');
  s.append(node('polygon',{points:area,class:'band-fill'}),node('polyline',{points:points('mid').map(p=>p.join(',')).join(' '),class:'mid-line'}));
  s.append(node('text',{x:48,y:235},clock(d[0].time)),node('text',{x:620,y:235},clock(d.at(-1).time)))}
function returnsChart(rows){
  const s=$('return-chart');if(!rows.length)return emptyChart(s,'Forecasts appear when fresh data arrives');s.replaceChildren();
  const lo=Math.min(-5,...rows.flatMap(r=>[r.yes,r.no])),hi=Math.max(5,...rows.flatMap(r=>[r.yes,r.no]));
  const x=v=>110+(v-lo)/(hi-lo)*465;
  [lo,0,hi].forEach(v=>s.append(node('line',{x1:x(v),x2:x(v),y1:12,y2:215,class:v===0?'zero-line':'grid-line'}),node('text',{x:x(v)-8,y:235},v.toFixed(0)+'¢')));
  rows.forEach((r,i)=>{const y=18+i*25;s.append(node('text',{x:0,y:y+8},SHORT_NAMES[r.model]||r.model));
    [[r.yes,'bar-yes',0],[r.no,'bar-no',8]].forEach(([v,c,dy])=>s.append(node('rect',{x:Math.min(x(0),x(v)),y:y+dy,width:Math.max(1,Math.abs(x(v)-x(0))),height:5,rx:2,class:c})))})}
function render(){
  if(!state)return;const markets=state.markets||[];
  if(!markets.some(m=>m.ticker===selected))selected=markets[0]?.ticker||'';
  const menu=$('market');menu.replaceChildren();
  (markets.length?markets:[{ticker:'Waiting for market history'}]).forEach(m=>{const o=document.createElement('option');o.value=m.ticker;o.textContent=m.ticker;o.selected=m.ticker===selected;menu.append(o)});
  const m=markets.find(m=>m.ticker===selected),fresh=m?.fresh&&Date.now()-m.time<=30000;
  const rows=(m?.predictions||[]).filter(r=>r.horizon===horizon);
  $('connection').textContent=fresh?'● Synced feed · live':state.status==='loading_models'?'◌ Loading models':'● Waiting / stale feed';
  $('connection').className=fresh?'live':'';$('notice').className='notice'+(fresh?' live':'');
  $('notice').textContent=state.status==='error'?'Inference service error: '+(state.errors?.[0]?.message||'See server log'):fresh?'Fresh synced observations · research forecasts only · no orders are submitted.':m?`Latest observation is ${Math.max(0,Math.round((Date.now()-m.time)/1000))}s old. Live forecasts are paused until fresh data arrives.`:'Waiting for a valid, continuous 60-observation history from synced files. Models load once on startup.';
  $('mid').textContent=m?m.midpoint.toFixed(1)+'¢':'—';
  $('quote').textContent=m?`Bid ${m.bid.toFixed(1)}¢ / Ask ${m.ask.toFixed(1)}¢`:'Bid — / Ask —';
  $('spread').textContent=m?m.spread.toFixed(1)+'¢':'—';
  $('imbalance').textContent=m?(m.imbalance>0?'+':'')+(m.imbalance*100).toFixed(0)+'%':'—';
  $('depth').textContent=m?`${m.bid_depth.toFixed(0)} bid / ${m.ask_depth.toFixed(0)} ask contracts`:'Bid / ask depth';
  $('signals').textContent=fresh&&rows.length?`${rows.filter(r=>r.yes>r.no).length} / ${rows.length}`:'Paused';
  $('observed').textContent=m?`Observed ${clock(m.time)} · ${horizon}s horizon`:'No observations yet';
  $('run').textContent=`${state.model_run} · 5s refresh · Syncthing feed`;
  priceChart(m);returnsChart(rows);
  if(typeof updateModelWindows==='function')updateModelWindows(m,horizon,fresh);
  if(typeof updateLeaderboard==='function')updateLeaderboard(horizon);
  const body=$('predictions');body.replaceChildren();
  if(!rows.length){const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=5;td.textContent='No live forecasts yet. Waiting for fresh synced data and loaded models.';tr.append(td);body.append(tr)}
  rows.forEach(r=>{const tr=document.createElement('tr');const name=document.createElement('td'),dot=document.createElement('span');
    dot.className='model-dot mw-'+r.model;name.append(dot,document.createTextNode(r.label));tr.append(name);
    [r.yes,r.no].forEach(v=>{const td=document.createElement('td');td.className=v>=0?'positive':'negative';td.textContent=cents(v);tr.append(td)});
    const signal=document.createElement('td'),badge=document.createElement('span');badge.className='badge '+(fresh?r.signal.toLowerCase():'');badge.textContent=fresh?r.signal:'STALE';signal.append(badge);
    const t=document.createElement('td');t.textContent=clock(m.time);tr.append(signal,t);body.append(tr)});
  const list=$('timeline');list.replaceChildren();
  const events=(state.timeline||[]).filter(e=>e.ticker===selected).slice(-12).reverse();
  if(!events.length)list.textContent='New forecasts appear here as synced observations arrive.';
  events.forEach(e=>{const r=e.predictions.filter(p=>p.horizon===horizon),best=r.reduce((a,b)=>Math.max(a,b.yes,b.no),-Infinity),el=document.createElement('div');el.className='timeline-entry';
    [clock(e.time),`${r.filter(p=>p.yes>p.no).length} of ${r.length} lean YES`,`Highest forecast ${cents(best)}`].forEach(text=>{const span=document.createElement('span');span.textContent=text;el.append(span)});list.append(el)})}
$('market').addEventListener('change',e=>{selected=e.target.value;render()});
document.querySelectorAll('[data-h]').forEach(b=>b.addEventListener('click',()=>{horizon=Number(b.dataset.h);
  document.querySelectorAll('[data-h]').forEach(x=>{x.classList.toggle('selected',x===b);x.setAttribute('aria-pressed',String(x===b))});
  render();if(!state&&typeof updateLeaderboard==='function')updateLeaderboard(horizon)}));
function feedStatus(){if(!state?.feed)return;const f=state.feed,m=state.markets.find(m=>m.ticker===selected),fresh=m?.fresh&&Date.now()-m.time<=30000;$('run').textContent=`${state.model_run} · 5s inference · Kalshi direct WebSocket`;if(fresh){$('connection').textContent='● Kalshi WebSocket · live';$('notice').textContent='Connected directly to Kalshi · one-second book samples · research forecasts only · no orders.'}else{$('connection').textContent=`● Kalshi · ${f.status}`;$('notice').textContent=f.error||`Kalshi ${f.status} · ${f.snapshots} snapshots received · building continuous history: ${f.warmup_samples}/60 samples.`}}
async function poll(){try{const response=await fetch('/api/state',{cache:'no-store',signal:AbortSignal.timeout(10000)});if(!response.ok)throw Error('HTTP '+response.status);state=await response.json();render();feedStatus()}catch(e){$('connection').textContent='● Disconnected';$('connection').className='';$('notice').className='notice';$('notice').textContent='Local service disconnected. Reconnecting automatically; displayed forecasts may be old.';$('signals').textContent='Paused';document.querySelectorAll('.badge').forEach(b=>{b.textContent='DISCONNECTED';b.className='badge'})}setTimeout(poll,3000)}poll();
