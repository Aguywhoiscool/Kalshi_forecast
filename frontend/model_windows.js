/* Model windows + relative leaderboard.
   Every mark is drawn from /api/windows (actual saved calculations for one forecast) or
   /api/scoreboard (exploratory validation averages). Nothing is synthesized, jittered or
   back-filled: missing data produces an explicit message instead of a drawing. */
const MW_NS='http://www.w3.org/2000/svg',MW_W=300,MW_H=150;
const WINDOW_MODELS=[
  {model:'mean',label:'Mean baseline',how:'Ignores the market entirely and always predicts the average return it saw in training.'},
  {model:'ridge',label:'Ridge regression',how:'Adds up weighted market features on top of a starting value; large weights are penalized.'},
  {model:'sgd',label:'SGD regression',how:'The same additive recipe, but its weights were learned step by step with stochastic gradient descent.'},
  {model:'decision_tree',label:'Decision tree',how:'Asks one threshold question about the market at a time until it lands in a leaf.'},
  {model:'random_forest',label:'Random forest',how:'Many randomized trees each make a guess; the forest reports their plain average.'},
  {model:'gradient_boosting',label:'Gradient boosting',how:'Starts from a baseline, then each small tree adds a correction to what came before.'},
  {model:'gru',label:'GRU neural network',how:'Reads the last 60 seconds in order, carrying a learned memory from one second to the next.'},
  {model:'monte_carlo',label:'Monte Carlo',how:'Replays recent one-second price moves at random thousands of times and averages the outcomes.'}];
const mwState={key:'',token:0,controller:null,payload:null,error:'',loading:false,retryAt:0,
  forecastId:'',horizon:30,ticker:'',rows:[],fresh:false,context:'',announcedContext:'',errorStatus:0};
const mwCards={};
let mwScoreboard=null,mwScoreboardError='',mwScoreboardLoading=false,mwLeaderboardHorizon=30;

function mwNum(v){if(!Number.isFinite(v))throw Error('non-finite value in the payload');return String(Math.round(v*100)/100)}
function mwEl(tag,attrs={},text){
  const e=document.createElementNS(MW_NS,tag);
  for(const [k,v] of Object.entries(attrs)){if(v===undefined||v===null)continue;e.setAttribute(k,typeof v==='number'?mwNum(v):String(v))}
  if(text!==undefined)e.textContent=text;return e}
function mwHtml(tag,className='',text){const e=document.createElement(tag);if(className)e.className=className;if(text!==undefined)e.textContent=text;return e}
function mwC(v,digits=2){if(!Number.isFinite(v))return '—';const r=Number(v.toFixed(digits));return `${r<0?'−':r>0?'+':''}${Math.abs(r).toFixed(digits)}¢`}
function mwFinite(list){return Array.isArray(list)&&list.length>0&&list.every(Number.isFinite)}
function mwPair(v){return Array.isArray(v)&&v.length===2&&v.every(Number.isFinite)}
function mwExtent(values,pad=.06){let lo=Math.min(...values),hi=Math.max(...values);if(!(hi-lo>1e-9)){lo-=1;hi+=1}const p=(hi-lo)*pad;return [lo-p,hi+p]}
function mwScale(d0,d1,r0,r1){return v=>r0+(v-d0)/(d1-d0)*(r1-r0)}
function mwClamp(v,lo,hi){return Math.max(lo,Math.min(hi,v))}
function mwShort(name,n=15){name=String(name);return name.length>n?name.slice(0,n-1)+'…':name}
function mwTitle(e,text){e.append(mwEl('title',{},text));return e}
function mwLabel(svg,text){svg.setAttribute('aria-label',text)}
function mwDelay(ms){return `--d:${Math.round(ms)}ms`}

/* ---------- renderers: each returns a one-line caption built from the same numbers it drew ---------- */
function mwMean(svg,{details:d,payload:p}){
  if(!mwPair(d.training_mean))throw Error('training mean missing');
  const mids=(p.input?.features||[]).map(f=>f[3]*100);if(mids.length<2||!mwFinite(mids))throw Error('input midpoint trace missing');
  const [lo,hi]=mwExtent(mids,.1),x=i=>8+i/(mids.length-1)*284,y=mwScale(lo,hi,62,20);
  svg.append(mwEl('text',{x:8,y:12},'Market midpoint, last 60s · an input it never uses'),
    mwEl('polyline',{points:mids.map((v,i)=>`${mwNum(x(i))},${mwNum(y(v))}`).join(' '),class:'mw-muted-line'}),
    mwEl('line',{x1:8,x2:292,y1:78,y2:78,class:'mw-grid'}),
    mwEl('text',{x:8,y:98},`Training-mean YES return · ${mwC(d.training_mean[0])} raw`),
    mwEl('line',{x1:8,x2:292,y1:114,y2:114,class:'mw-accent-line mw-draw',pathLength:1}),
    mwEl('circle',{cx:292,cy:114,r:4,class:'mw-dot mw-fade',style:mwDelay(900)}),
    mwEl('text',{x:8,y:140},'Same value every time · line height schematic'));
  mwLabel(svg,`Mean baseline: flat training mean ${mwC(d.training_mean[0])} drawn beside the unused midpoint trace`);
  return `Raw YES ${mwC(d.training_mean[0])} / NO ${mwC(d.training_mean[1])} before calibration. The order book never changes it.`}

function mwLinearParts(d,side){
  if(!mwPair(d.intercept)||!Array.isArray(d.features)||!d.features.length)throw Error('linear contributions missing');
  d.features.forEach(f=>{if(!mwPair(f.contribution))throw Error('contribution missing for '+f.name)});
  return d.features.map(f=>({name:f.name,value:side(f.contribution)}))}
function mwRidge(svg,{details:d}){
  const parts=mwLinearParts(d,c=>c[0]).sort((a,b)=>Math.abs(b.value)-Math.abs(a.value));
  const top=parts.slice(0,5),rest=parts.slice(5),rows=[];let sum=d.intercept[0];
  rows.push({name:'Start (intercept)',start:sum,end:sum,anchor:true});
  top.forEach(f=>{rows.push({name:f.name,start:sum,end:sum+f.value,value:f.value});sum+=f.value});
  if(rest.length){const v=rest.reduce((a,f)=>a+f.value,0);rows.push({name:`${rest.length} other features`,start:sum,end:sum+v,value:v});sum+=v}
  rows.push({name:'= Raw YES sum',start:d.intercept[0],end:sum,total:true});
  const [lo,hi]=mwExtent(rows.flatMap(r=>[r.start,r.end])),x=mwScale(lo,hi,104,292),rowH=Math.min(15,124/rows.length);
  rows.forEach((r,i)=>{
    const y=8+i*rowH,g=mwEl('g',{class:'mw-fade',style:mwDelay(i*90)});
    g.append(mwEl('text',{x:4,y:y+8},mwShort(r.name,17)));
    if(r.anchor)g.append(mwEl('rect',{x:x(r.start)-1,y:y,width:2,height:10,class:'mw-fill-accent'}));
    else g.append(mwEl('rect',{x:Math.min(x(r.start),x(r.end)),y:y+1,width:Math.max(1.5,Math.abs(x(r.end)-x(r.start))),height:8,rx:2,
      class:r.value<0?'mw-fill-accent mw-pale':'mw-fill-accent'}));
    mwTitle(g,r.anchor?`Intercept ${mwC(r.start,4)}`:r.total?`Raw YES sum ${mwC(r.end,4)}`:`${r.name}: ${mwC(r.value,4)} → ${mwC(r.end,4)}`);
    if(i<rows.length-1&&!r.total)g.append(mwEl('line',{x1:x(r.end),x2:x(r.end),y1:y+9,y2:y+rowH+1,class:'mw-grid'}));
    svg.append(g)});
  svg.append(mwEl('text',{x:104,y:146},mwC(lo,1)),mwEl('text',{x:292,y:146,'text-anchor':'end'},mwC(hi,1)));
  mwLabel(svg,`Ridge waterfall from intercept ${mwC(d.intercept[0])} to raw YES ${mwC(sum)}`);
  const big=top[0];
  return `Starts at ${mwC(d.intercept[0])}; biggest push ${big.name} ${mwC(big.value)}; all ${d.features.length} sum to ${mwC(sum)} raw; pale bars lower it.`}

function mwSgd(svg,{details:d}){
  const pulls=mwLinearParts(d,c=>c[0]-c[1]),offset=d.intercept[0]-d.intercept[1],net=offset+pulls.reduce((a,f)=>a+f.value,0);
  const yes=pulls.filter(f=>f.value>0).sort((a,b)=>b.value-a.value).slice(0,4),no=pulls.filter(f=>f.value<0).sort((a,b)=>a.value-b.value).slice(0,4);
  const D=Math.max(Math.abs(net),...pulls.map(f=>Math.abs(f.value)))*1.1||1,len=v=>Math.abs(v)/D*130;
  svg.append(mwEl('text',{x:4,y:11},'◀ toward NO'),mwEl('text',{x:296,y:11,'text-anchor':'end'},'toward YES ▶'),
    mwEl('line',{x1:150,x2:150,y1:16,y2:120,class:'mw-grid'}));
  const row=(f,i,dir)=>{const y=26+i*18,g=mwEl('g',{class:'mw-fade',style:mwDelay(i*110+(dir>0?0:55))});
    g.append(mwEl('text',{x:dir>0?154:146,y,'text-anchor':dir>0?'start':'end'},mwShort(f.name,13)),
      mwEl('rect',{x:dir>0?150:150-len(f.value),y:y+3,width:Math.max(1.5,len(f.value)),height:4,rx:2,class:dir>0?'mw-yes':'mw-no'}));
    mwTitle(g,`${f.name}: YES contribution − NO contribution = ${mwC(f.value,4)}`);svg.append(g)};
  yes.forEach((f,i)=>row(f,i,1));no.forEach((f,i)=>row(f,i,-1));
  const kx=150+mwClamp(net/D*130,-135,135),dx=150-kx;
  svg.append(mwEl('line',{x1:12,x2:288,y1:110,y2:110,class:'mw-rope'}),
    mwTitle(mwEl('circle',{cx:kx,cy:110,r:5,class:'mw-dot mw-knot',style:`--dx:${mwNum(dx)}px`}),`Net raw lean ${mwC(net,4)}`),
    mwEl('text',{x:mwClamp(kx,78,222),y:132,'text-anchor':'middle'},`net ${mwC(net)} toward ${net>=0?'YES':'NO'} (raw)`));
  mwLabel(svg,`SGD tug of war: features pull YES minus NO; net raw lean ${mwC(net)}`);
  return `Each bar = that feature's YES minus NO contribution. Starting offset ${mwC(offset)}; strongest pulls shown.`}

function mwFeatureValue(name,v){v=Number(v);if(!Number.isFinite(v))return '—';name=String(name);
  if(/depth/.test(name))return `${v.toLocaleString(undefined,{maximumFractionDigits:1})} contracts`;
  if(/imbalance/.test(name))return `${v.toFixed(3)} (imbalance ratio)`;
  if(/^(bid|ask|spread|midpoint)_/.test(name))return `${(v*100).toFixed(1)}¢`;
  return v.toPrecision(4)}
function mwTreeSteps(d){if(!Array.isArray(d?.steps)||!d.steps.length)throw Error('decision path missing');return d.steps}
function mwTree(svg,{details:d},label='Decision tree'){
  const steps=mwTreeSteps(d),n=steps.length,gap=Math.min(24,108/Math.max(1,n-1)),r=Math.min(4.5,gap/2.4),leaf=steps.at(-1);
  let x=150,off=74;const pts=[];
  steps.forEach((s,i)=>{pts.push({x,y:18+i*gap,s});if(i<n-1){const dir=s.direction==='left'?-1:1;x=mwClamp(x+dir*off,10,290);off=Math.max(8,off/2)}});
  pts.forEach((p,i)=>{
    if(i<n-1){const q=pts[i+1],dir=p.s.direction==='left'?-1:1;
      svg.append(mwEl('line',{x1:p.x,y1:p.y,x2:mwClamp(p.x-dir*(Math.abs(q.x-p.x)||8)*.6,6,294),y2:p.y+gap*.6,class:'mw-stub'}),
        mwEl('line',{x1:p.x,y1:p.y,x2:q.x,y2:q.y,class:'mw-accent-line mw-draw',pathLength:1,style:mwDelay(i*130)}))}});
  pts.forEach((p,i)=>{const s=p.s,c=mwEl('circle',{cx:p.x,cy:p.y,r:i===n-1?r+1.5:r,class:'mw-node mw-light',style:mwDelay(i*130)});
    mwTitle(c,s.leaf?`Leaf ${s.node}: YES ${mwC(s.value?.[0],4)} raw`:`${s.feature}: observed ${mwFeatureValue(s.feature,s.observed)} ${s.direction==='left'?'≤':'>'} threshold ${mwFeatureValue(s.feature,s.threshold)} → ${s.direction}`);
    svg.append(c)});
  svg.append(mwEl('text',{x:4,y:10},'layout schematic · path and values exact'));
  const end=pts.at(-1),right=end.x<150;
  const leafValue=mwPair(d.prediction)?d.prediction:leaf.value;
  if(!mwPair(leafValue))throw Error('leaf value missing');
  svg.append(mwEl('text',{x:right?end.x+10:end.x-10,y:Math.min(end.y+4,146),'text-anchor':right?'start':'end'},`leaf ${leaf.node} · ${mwC(leafValue[0])} raw`));
  mwLabel(svg,`${label}: ${n-1} comparisons down to leaf ${leaf.node}`);
  const first=steps[0];
  return first.leaf?`Single leaf: raw YES ${mwC(leafValue[0])}.`:
    `${n-1} questions. First: ${first.feature} ${mwFeatureValue(first.feature,first.observed)} ${first.direction==='left'?'≤':'>'} ${mwFeatureValue(first.feature,first.threshold)}. Leaf raw YES ${mwC(leafValue[0])}.`}

function mwForest(svg,{details:d}){
  const outs=(d.tree_outputs||[]).map(v=>v?.[0]);if(!mwFinite(outs))throw Error('tree outputs missing');
  const n=outs.length,mean=outs.reduce((a,b)=>a+b,0)/n,[lo,hi]=mwExtent([...outs,mean]),x=mwScale(lo,hi,16,284);
  const cols=new Map();const placed=outs.map((v,i)=>{const c=Math.round((x(v)-16)/7);const k=cols.get(c)||0;cols.set(c,k+1);return {v,i,c,k}});
  const maxStack=Math.max(...cols.values()),r=Math.min(3.5,84/(2*maxStack)),step=2*r+.6;
  const gridCols=Math.ceil(Math.sqrt(n*3));
  svg.append(mwEl('circle',{cx:8,cy:8,r:3.5,class:'mw-hi'}),mwEl('text',{x:15,y:11},'above forest mean'),
    mwEl('circle',{cx:122,cy:8,r:3.5,class:'mw-lo'}),mwEl('text',{x:129,y:11},'below forest mean'),
    mwEl('line',{x1:16,x2:284,y1:114,y2:114,class:'mw-grid'}));
  placed.forEach(p=>{const cx=16+p.c*7,cy=110-r-p.k*step,gx=40+(p.i%gridCols)*(220/gridCols),gy=26+Math.floor(p.i/gridCols)*7;
    svg.append(mwTitle(mwEl('circle',{cx,cy,r,class:`${p.v>=mean?'mw-hi':'mw-lo'} mw-fly`,
      style:`--dx:${mwNum(gx-cx)}px;--dy:${mwNum(gy-cy)}px;--d:${Math.round(p.i*12)}ms`}),`Tree ${p.i}: YES ${mwC(p.v,4)} raw`))});
  const mx=x(mean);
  svg.append(mwEl('line',{x1:mx,x2:mx,y1:18,y2:118,class:'mw-mean-line mw-draw',pathLength:1,style:mwDelay(n*12+500)}),
    mwEl('text',{x:mwClamp(mx,78,222),y:130,'text-anchor':'middle'},`mean of ${n} trees · ${mwC(mean)}`),
    mwEl('text',{x:16,y:146},mwC(lo,1)),mwEl('text',{x:284,y:146,'text-anchor':'end'},mwC(hi,1)));
  mwLabel(svg,`Random forest: ${n} tree YES outputs from ${mwC(Math.min(...outs))} to ${mwC(Math.max(...outs))}, mean ${mwC(mean)}`);
  return `Stacking and fly-in are schematic; x is exact. ${n} trees disagree by ${(Math.max(...outs)-Math.min(...outs)).toFixed(2)}¢; the forest averages them.`}

function mwBoosting(svg,{details:d}){
  const all=(d.stages||[]).map(s=>s?.[0]);if(!mwFinite(all))throw Error('boosting stages missing');
  const N=all.length,k=Math.max(1,Math.ceil(N/60)),idx=[];for(let i=0;i<N;i+=k)idx.push(i);if(idx.at(-1)!==N-1)idx.push(N-1);
  const vals=idx.map(i=>all[i]),[lo,hi]=mwExtent(vals),x=i=>20+(N>1?i/(N-1):0)*266,y=mwScale(lo,hi,126,26);
  let path=`M${mwNum(x(idx[0]))},${mwNum(y(vals[0]))}`;for(let j=1;j<idx.length;j++)path+=` H${mwNum(x(idx[j]))} V${mwNum(y(vals[j]))}`;
  svg.append(mwEl('text',{x:20,y:12},`stage 1 · ${mwC(all[0])}`),mwEl('text',{x:286,y:12,'text-anchor':'end'},`stage ${N} · ${mwC(all[N-1])}`),
    mwEl('line',{x1:20,x2:286,y1:y(all[0]),y2:y(all[0]),class:'mw-grid'}),
    mwEl('path',{d:path,class:'mw-accent-line mw-draw mw-slow',pathLength:1}),
    mwEl('circle',{cx:x(N-1),cy:y(all[N-1]),r:4,class:'mw-dot mw-fade',style:mwDelay(1500)}),
    mwEl('text',{x:20,y:146},'first stage'),mwEl('text',{x:286,y:146,'text-anchor':'end'},'last stage'));
  mwLabel(svg,`Gradient boosting: ${N} cumulative stages from ${mwC(all[0])} to ${mwC(all[N-1])}`);
  return `${N} stages${k>1?` (1 in ${k} drawn)`:''}. Raw YES moves ${mwC(all[N-1]-all[0])} from stage 1 (already baseline + first tree) to stage ${N}.`}

function mwGru(svg,{details:d,payload:p}){
  const ts=d.time_sensitivity,feats=p.input?.features||[];
  if(!mwFinite(ts))throw Error('time sensitivity missing');
  const mids=feats.map(f=>f[3]*100);if(mids.length!==ts.length||!mwFinite(mids))throw Error('input strip does not match sensitivity length');
  const n=ts.length,w=288/n,max=Math.max(...ts),[lo,hi]=mwExtent(mids,.1),y=mwScale(lo,hi,52,20);
  svg.append(mwEl('text',{x:6,y:11},`Input strip · ${n} one-second observations`),
    mwEl('polyline',{points:mids.map((v,i)=>`${mwNum(6+(i+.5)*w)},${mwNum(y(v))}`).join(' '),class:'mw-muted-line'}),
    mwEl('line',{x1:6,x2:294,y1:128,y2:128,class:'mw-grid'}));
  if(!(max>0)){svg.append(mwEl('text',{x:150,y:100,'text-anchor':'middle'},'No measurable local sensitivity for this input'));
    mwLabel(svg,'GRU input strip with zero local sensitivity');return 'The local derivative is zero at every step for this forecast.'}
  ts.forEach((v,i)=>{const h=Math.max(1,v/max*62);
    svg.append(mwTitle(mwEl('rect',{x:6+i*w+.5,y:128-h,width:Math.max(1,w-1),height:h,rx:1,class:'mw-fill-accent mw-pulse',style:mwDelay(i*35)}),
      `Observation ${i+1}: midpoint ${mids[i].toFixed(2)}¢ · sensitivity ${v.toPrecision(3)}`))});
  svg.append(mwEl('rect',{x:6,y:58,width:w,height:70,class:'mw-scan',style:`--scan:${mwNum(288-w)}px;--dur:${n*35}ms`}));
  const peak=ts.indexOf(max),px=6+(peak+.5)*w;
  svg.append(mwEl('text',{x:mwClamp(px,40,260),y:62,'text-anchor':'middle'},`peak · obs ${peak+1}`),
    mwEl('text',{x:6,y:146},'oldest'),mwEl('text',{x:294,y:146,'text-anchor':'end'},'newest'));
  mwLabel(svg,`GRU: local sensitivity across ${n} input observations, peak at observation ${peak+1}`);
  return `Local sensitivity peaks at observation ${peak+1}/${n}. A local derivative, not importance; per-input units differ.`}

function mwMonteCarlo(svg,{details:d,horizon}){
  const paths=d.preview_paths;
  if(!Array.isArray(paths)||!paths.length||!paths.every(q=>mwFinite(q)&&q.length===paths[0].length&&q.length>1))throw Error('preview paths missing');
  const cents=paths.map(q=>q.map(v=>v*100)),start=cents[0][0],T=cents[0].length-1,flat=cents.every(q=>q.every(v=>Math.abs(v-start)<1e-7));
  const [lo,hi]=flat?[start-1,start+1]:mwExtent(cents.flat()),x=t=>40+t/T*246,y=mwScale(lo,hi,124,22);
  svg.append(mwEl('text',{x:40,y:146},'now'),mwEl('text',{x:286,y:146,'text-anchor':'end'},`+${horizon}s`),
    mwEl('text',{x:4,y:y(start)+3},`${start.toFixed(1)}¢`));
  if(flat){
    svg.append(mwEl('line',{x1:x(0),x2:x(T),y1:y(start),y2:y(start),class:'mw-accent-line mw-draw',pathLength:1}),
      mwEl('text',{x:163,y:44,'text-anchor':'middle',class:'mw-strong'},`All ${paths.length} shown paths stay at ${start.toFixed(1)}¢`),
      mwEl('text',{x:163,y:58,'text-anchor':'middle'},'quiet history · no jitter added'));
    mwLabel(svg,`Monte Carlo: all ${paths.length} preview paths are flat at ${start.toFixed(1)} cents`);
  }else{
    cents.forEach((q,i)=>svg.append(mwEl('polyline',{points:q.map((v,t)=>`${mwNum(x(t))},${mwNum(y(v))}`).join(' '),
      class:'mw-path mw-draw',pathLength:1,style:mwDelay(i*45)})));
    mwLabel(svg,`Monte Carlo fan of ${paths.length} actual simulated midpoint paths from ${start.toFixed(1)} cents`);
  }
  svg.append(mwEl('circle',{cx:x(0),cy:y(start),r:4,class:'mw-dot'}));
  const ends=cents.map(q=>q.at(-1)),pct=d.scenario_percentiles;
  const median=Array.isArray(pct)&&mwPair(pct[1])?` · median scenario YES ${mwC(pct[1][0])} raw`:'';
  return flat?`Every sampled one-second move was zero, so all shown scenarios agree${median}.`:
    `${d.path_count||'All'} seeded paths; shown endpoints ${Math.min(...ends).toFixed(1)}–${Math.max(...ends).toFixed(1)}¢${median}.`}

const MW_RENDERERS={mean:mwMean,ridge:mwRidge,sgd:mwSgd,decision_tree:mwTree,random_forest:mwForest,
  gradient_boosting:mwBoosting,gru:mwGru,monte_carlo:mwMonteCarlo};

/* ---------- cards ---------- */
const MW_TIMEOUT_MS=15000,MW_NO_RETRY=[400,404,410];
function mwScoreRow(model,h=mwState.horizon){return mwScoreboard?.horizons?.[String(h)]?.models?.find(r=>r.model===model)||null}
function mwInit(){
  const grid=document.getElementById('model-windows');if(!grid)return;grid.replaceChildren();
  WINDOW_MODELS.forEach(spec=>{
    const card=mwHtml('article',`mw-card mw-${spec.model}`),head=mwHtml('div','mw-head'),title=mwHtml('h3','',spec.label),rank=mwHtml('span','mw-rank');
    const open=mwHtml('button','mw-open','Show the math');open.type='button';
    open.setAttribute('aria-label',`Show the full ${spec.label} calculation for this forecast`);
    head.append(mwHtml('span','mw-swatch'),title,rank,open);
    const viz=mwHtml('div','mw-viz'),caption=mwHtml('p','mw-caption'),calc=mwHtml('div','mw-calc'),
      foot=mwHtml('div','mw-foot'),forecast=mwHtml('span','mw-forecast'),lean=mwHtml('span','mw-lean'),chip=mwHtml('span','mw-chip');
    foot.append(forecast,lean,chip);
    card.append(head,mwHtml('p','mw-how',spec.how),viz,caption,calc,foot);
    open.addEventListener('click',e=>{e?.stopPropagation?.();mwOpen(spec.model)});
    card.addEventListener('click',()=>mwOpen(spec.model));
    grid.append(card);mwCards[spec.model]={card,viz,caption,calc,forecast,lean,chip,rank,open,hasViz:false,drawnKey:'',context:'',spec}});
  mwRenderAll()}
function mwOpen(model){
  if(!mwState.forecastId||typeof loadExplanation!=='function')return;
  if(typeof explanationTab!=='undefined')explanationTab='calculation';
  document.querySelectorAll('[data-tab]').forEach(b=>b.classList.toggle('selected',b.dataset.tab==='calculation'));
  loadExplanation(mwState.forecastId,model,mwState.horizon)}
/* The payload is used for numbers only when it belongs to the selected forecast and horizon. */
function mwCurrentPayload(){const p=mwState.payload;return p&&p.forecast_id===mwState.forecastId&&Number(p.horizon)===mwState.horizon?p:null}
/* While the next forecast of the SAME market and horizon loads, every part of a card keeps showing the previous
   payload, so drawing, caption, strip and foot always describe one forecast and swap together. */
function mwHeldPayload(){const p=mwState.payload;
  return !mwCurrentPayload()&&mwState.loading&&p&&mwState.payloadContext===mwState.context?p:null}
function mwMessage(c,text,kind){
  c.viz.replaceChildren(mwHtml('p',`mw-msg mw-msg-${kind}`,text));c.viz.classList.remove('mw-refreshing');c.caption.classList.remove('mw-dim');
  c.caption.textContent='';c.caption.removeAttribute?.('title');c.hasViz=false;c.drawnKey='';c.context=''}
function mwCalcParts(side,i,calc){
  const raw=calc.raw[i],bounded=calc.clipped_raw[i],corr=calc.calibration_correction[i],fin=calc.final[i],clipped=Math.abs(bounded-raw)>1e-9;
  // When the raw output was clipped, show the bounded value (what calibration used); the original raw is in the title.
  const steps=`${side} ${mwC(clipped?bounded:raw)} ${clipped?'bounded raw':'raw'} ${corr<0?'−':'+'} ${Math.abs(corr).toFixed(2)}¢ calib. `;
  const title=`${side}: raw ${mwC(raw,4)}${clipped?` bounded to ${mwC(bounded,4)}`:''}, calibration ${mwC(corr,4)}, final ${mwC(fin,4)}`;
  return {steps,final:`= ${mwC(fin)}`,title}}
function mwRenderFoot(c,model,calc,held){
  const row=mwState.rows.find(r=>r.model===model);
  const valid=calc&&['raw','clipped_raw','calibration_correction','final'].every(k=>mwPair(calc[k]));
  const final=valid?calc.final:!held&&row&&Number.isFinite(row.yes)&&Number.isFinite(row.no)?[row.yes,row.no]:null;
  c.calc.replaceChildren();
  if(valid)c.calc.append(...[0,1].map(i=>{const p=mwCalcParts(i?'NO':'YES',i,calc),line=mwHtml('span','mw-calc-line');
    line.setAttribute('title',p.title);line.append(mwHtml('span','mw-calc-steps',p.steps),mwHtml('span','mw-calc-final',p.final));return line}));
  else c.calc.append(mwHtml('span','mw-calc-line',final?'Final forecast shown; calculation steps':'No calculation yet'),
    mwHtml('span','mw-calc-line',final?(mwState.loading?'updating…':'unavailable for this forecast.'):''));
  c.forecast.textContent=final?`Final YES ${mwC(final[0])} · NO ${mwC(final[1])}`:'No forecast yet';
  if(final){const diff=final[0]-final[1],side=diff>0?'YES':diff<0?'NO':'',loss=final[0]<0&&final[1]<0;
    c.lean.className='mw-lean'+(side?' mw-lean-'+side.toLowerCase():'');
    c.lean.textContent=(side?`Leans ${side} by ${Math.abs(diff).toFixed(2)}¢`:'No lean')+(loss?' · loss either way':'');
    c.lean.setAttribute('title',loss?'Both final forecasts lose money after the 4¢ buffer; the lean is the less-negative final side.':'Side with the higher final forecast.')}
  else{c.lean.className='mw-lean';c.lean.textContent=''}}
function mwRenderCard(model){
  const c=mwCards[model];if(!c)return;const p=mwCurrentPayload(),held=mwHeldPayload(),entry=(p||held)?.models?.[model];
  mwRenderFoot(c,model,entry?.calculation,Boolean(held));mwRenderChip(model);
  const ready=Boolean(mwState.forecastId);
  c.open.setAttribute('aria-disabled',String(!ready));c.open.classList.toggle('mw-open-disabled',!ready);
  c.open.setAttribute('title',ready?'Open the step-by-step calculation':'No forecast yet');
  if(!ready)return mwMessage(c,'Waiting for a forecast for this market and horizon.','wait');
  if(!p){
    if(mwState.error)return mwMessage(c,`Calculations unavailable: ${mwState.error}`,'error');
    if(c.hasViz){
      // Same market and horizon: leave the card untouched. Other market/horizon: dim marks and replace the old caption.
      const other=c.context!==mwState.context;c.viz.classList.toggle('mw-refreshing',other);c.caption.classList.toggle('mw-dim',other);
      if(other){const t=`Loading ${mwState.ticker||'this market'} · ${mwState.horizon}s…`;c.caption.textContent=t;c.caption.setAttribute('title',t)}
      return}
    return mwMessage(c,'Loading this model’s actual calculation…','loading')}
  if(p.errors?.[model])return mwMessage(c,`This model’s explanation failed its consistency check for this forecast, so nothing is drawn. (${String(p.errors[model]).split('\n')[0].slice(0,120)})`,'error');
  if(!entry?.details)return mwMessage(c,'No calculation was returned for this model; nothing is drawn.','error');
  const drawnKey=`${p.forecast_id}|${p.horizon}`;if(c.drawnKey===drawnKey&&c.hasViz)return;
  // Animate only on first draw or after a market/horizon change, never on a same-market forecast refresh.
  const animate=c.context!==mwState.context;
  const svg=mwEl('svg',{viewBox:`0 0 ${MW_W} ${MW_H}`,class:animate?'mw-svg mw-anim':'mw-svg',role:'img','data-window':model,'data-forecast':p.forecast_id});
  try{const caption=MW_RENDERERS[model](svg,{details:entry.details,calc:entry.calculation,payload:p,horizon:mwState.horizon});
    c.viz.replaceChildren(svg);c.viz.classList.remove('mw-refreshing');c.caption.classList.remove('mw-dim');c.caption.textContent=caption;c.caption.setAttribute('title',caption);
    c.hasViz=true;c.drawnKey=drawnKey;c.context=mwState.context}
  catch(error){mwMessage(c,`Visualization could not be drawn from this payload: ${error.message}`,'error')}}
function mwRenderChip(model){
  const c=mwCards[model],r=mwScoreRow(model),avg=mwScoreboard?.horizons?.[String(mwState.horizon)]?.average_closed_mean_cents;
  if(!r){c.chip.className='mw-chip';c.chip.textContent=mwScoreboardError?'Ranking unavailable':mwScoreboard?`Not ranked at ${mwState.horizon}s`:'Loading ranking…';
    c.rank.textContent='';c.chip.removeAttribute?.('title');return}
  c.rank.textContent=`#${r.rank} at ${mwState.horizon}s`;
  c.chip.className='mw-chip '+(r.within_noise?'mw-chip-noise':r.relative_cents>=0?'mw-chip-pos':'mw-chip-neg');
  c.chip.textContent=`${mwC(r.relative_cents)} vs avg`+(r.within_noise?' · within noise':'');
  c.chip.setAttribute('title',`${mwState.horizon}s: closed-outcome average ${mwC(r.closed_mean_cents,3)} per hypothetical trade vs model average ${mwC(avg,3)}. Relative, not profit.`)}
function mwRenderAll(){WINDOW_MODELS.forEach(s=>mwRenderCard(s.model));mwStatus()}
function mwStatus(){
  const s=document.getElementById('windows-status');if(!s)return;
  s.textContent=!mwState.forecastId?`No forecast yet · ${mwState.horizon}s`:
    `${mwState.ticker} · ${mwState.horizon}s · forecast ${mwState.forecastId.slice(0,8)}${mwState.loading?' · updating…':mwState.error?' · calculations unavailable':''}${mwState.fresh?'':' · feed stale, showing latest'}`}
/* One polite announcement per market/horizon change or error, instead of per-card alerts every refresh. */
function mwAnnounce(text){const live=document.getElementById('windows-live');if(live&&live.textContent!==text)live.textContent=text}

/* ---------- fetching: one request in flight, 15 s timeout, late responses ignored ---------- */
async function mwFetch(id,h){
  const token=++mwState.token;mwState.controller?.abort();
  const controller=new AbortController();mwState.controller=controller;
  let timedOut=false;const timer=setTimeout(()=>{timedOut=true;controller.abort()},MW_TIMEOUT_MS);
  const context=mwState.context,contextChanged=context!==mwState.announcedContext;
  mwState.loading=true;mwState.error='';mwState.errorStatus=0;
  if(contextChanged){mwState.announcedContext=context;mwAnnounce(`Loading model windows for ${mwState.ticker||'this market'} · ${h}s.`)}
  mwRenderAll();
  let payload=null,error='',status=0;
  try{const response=await fetch(`/api/windows/${encodeURIComponent(id)}?horizon=${h}`,{signal:controller.signal,cache:'no-store'});
    let body=null;try{body=await response.json()}catch{body=null}
    if(token!==mwState.token)return;
    if(!response.ok){status=response.status;throw Error(body?.detail||`HTTP ${response.status}`)}
    if(!body||typeof body!=='object')throw Error('the server returned an unreadable response');
    if(body.forecast_id!==id||Number(body.horizon)!==h)throw Error('response did not match the selected forecast');
    payload=body}
  catch(e){if(token!==mwState.token||(e.name==='AbortError'&&!timedOut))return;error=timedOut?'request timed out after 15 s':(e.message||String(e))}
  finally{clearTimeout(timer)}
  if(token!==mwState.token)return;
  mwState.loading=false;mwState.controller=null;mwState.payload=payload;mwState.payloadContext=payload?context:'';mwState.error=error;mwState.errorStatus=status;
  mwState.retryAt=error?Date.now()+MW_TIMEOUT_MS:0;
  if(error)mwAnnounce(`Model windows unavailable: ${error}`);else if(contextChanged)mwAnnounce(`Model windows loaded for ${mwState.ticker||'this market'} · ${h}s.`);
  mwRenderAll()}
function updateModelWindows(market,h,fresh){
  const rows=(market?.predictions||[]).filter(r=>r.horizon===h),id=rows.find(r=>r.forecast_id)?.forecast_id||'';
  const horizonChanged=h!==mwState.horizon;
  mwState.rows=rows;mwState.fresh=Boolean(fresh);mwState.ticker=market?.ticker||'';mwState.horizon=h;mwState.context=`${mwState.ticker}|${h}`;
  const key=id?`${id}|${h}`:'';
  const grid=document.getElementById('model-windows');grid?.classList.toggle('mw-stale',!fresh);
  if(key!==mwState.key){mwState.key=key;mwState.forecastId=id;
    if(id)return mwFetch(id,h);
    mwState.token++;mwState.controller?.abort();mwState.controller=null;mwState.payload=null;mwState.error='';mwState.loading=false;return mwRenderAll()}
  if(id&&mwState.error&&!mwState.loading&&!MW_NO_RETRY.includes(mwState.errorStatus)&&Date.now()>=mwState.retryAt)return mwFetch(id,h);
  if(horizonChanged)return mwRenderAll();
  mwStatus()}

/* ---------- relative leaderboard ---------- */
async function loadScoreboard(){
  if(mwScoreboardLoading)return;mwScoreboardLoading=true;
  const controller=new AbortController();let timedOut=false;const timer=setTimeout(()=>{timedOut=true;controller.abort()},MW_TIMEOUT_MS);
  try{const response=await fetch('/api/scoreboard',{cache:'no-store',signal:controller.signal});let body=null;try{body=await response.json()}catch{body=null}
    if(!response.ok)throw Error(body?.detail||`HTTP ${response.status}`);
    if(!body||typeof body!=='object')throw Error('the server returned an unreadable response');
    mwScoreboard=body;mwScoreboardError=''}
  catch(e){mwScoreboardError=timedOut?'request timed out after 15 s':(e.message||String(e));setTimeout(loadScoreboard,30000)}
  finally{clearTimeout(timer);mwScoreboardLoading=false}
  renderLeaderboard(mwLeaderboardHorizon);WINDOW_MODELS.forEach(s=>mwCards[s.model]&&mwRenderChip(s.model))}
function mwLabelFor(model){return WINDOW_MODELS.find(s=>s.model===model)?.label||model}
function renderLeaderboard(h){
  mwLeaderboardHorizon=h;const list=document.getElementById('leaderboard');if(!list)return;
  const truth=document.getElementById('leaderboard-truth'),pop=document.getElementById('leaderboard-population'),axis=document.getElementById('leaderboard-axis'),notes=document.getElementById('leaderboard-notes');
  const data=mwScoreboard?.horizons?.[String(h)];
  if(!data){list.replaceChildren(mwHtml('p','mw-msg mw-msg-'+(mwScoreboardError?'error':mwScoreboard?'wait':'loading'),
      mwScoreboardError?`Leaderboard unavailable: ${mwScoreboardError}`:mwScoreboard?`No validation averages are available for the ${h}s horizon.`:'Loading validation averages…'));
    if(truth)truth.textContent='';if(axis)axis.replaceChildren();if(pop)pop.textContent='';return}
  const rows=[...data.models].filter(r=>Number.isFinite(r.relative_cents)).sort((a,b)=>b.relative_cents-a.relative_cents);
  const avg=data.average_closed_mean_cents,closed=rows.map(r=>r.closed_mean_cents).filter(Number.isFinite);
  const profitable=closed.filter(v=>v>0).length;
  if(truth)truth.textContent=profitable===0?
    `On trades whose outcome was observed, every model averages about ${mwC(avg,1)} after the 4¢ buffer — none shows a profit on known outcomes (unknown outcomes stay intervals; profitability is not established). This view shows who loses least.`:
    `On trades whose outcome was observed, ${profitable} of ${closed.length} models average above zero; the model average is ${mwC(avg,1)} after the 4¢ buffer. Unknown outcomes stay intervals, and this exploratory split is not a trading result.`;
  const D=Math.max(...rows.flatMap(r=>[Math.abs(r.relative_cents),...(mwPair(r.relative_ci_cents)?r.relative_ci_cents.map(Math.abs):[])]))*1.15||1;
  const pct=v=>mwClamp(50+v/D*50,0,100);
  if(axis){axis.replaceChildren(mwHtml('span','',mwC(-D)),mwHtml('span','lb-axis-mid',`model average ${mwC(avg)}`),mwHtml('span','',mwC(D)))}
  const tip=document.getElementById('leaderboard-tooltip');
  list.replaceChildren(...rows.map((r,i)=>{
    const row=mwHtml('div',`lb-row mw-${r.model}`),rank=mwHtml('span','lb-rank',String(r.rank??i+1)),name=mwHtml('span','lb-name');
    name.append(mwHtml('span','mw-swatch'),document.createTextNode(r.label||mwLabelFor(r.model)));
    const track=mwHtml('div','lb-track'),mid=mwHtml('span','lb-mid'),bar=mwHtml('span',`lb-bar ${r.relative_cents>=0?'lb-pos':'lb-neg'}${r.within_noise?' lb-noise':''}`);
    const a=pct(Math.min(0,r.relative_cents)),b=pct(Math.max(0,r.relative_cents));
    bar.setAttribute('style',`left:${mwNum(a)}%;width:${mwNum(Math.max(.4,b-a))}%;--d:${i*70}ms`);
    track.append(mid,bar);
    if(mwPair(r.relative_ci_cents)){const w=mwHtml('span','lb-whisker'),l=pct(r.relative_ci_cents[0]),u=pct(r.relative_ci_cents[1]);
      w.setAttribute('style',`left:${mwNum(l)}%;width:${mwNum(Math.max(.2,u-l))}%`);track.append(w)}
    const value=mwHtml('span','lb-value');value.append(mwHtml('strong','',`${mwC(r.relative_cents)} vs avg`));
    value.append(mwHtml('span',r.within_noise?'lb-tag':'lb-tag lb-tag-clear',r.within_noise?'within noise':'outside noise'));
    row.append(rank,name,track,value);
    const ci=mwPair(r.relative_ci_cents)?`${mwC(r.relative_ci_cents[0],3)} to ${mwC(r.relative_ci_cents[1],3)}`:'unavailable';
    const pctText=v=>Number.isFinite(v)?`${Math.round(v*100)}%`:'—';
    const detail=`${r.label||mwLabelFor(r.model)}: ${mwC(r.relative_cents,3)} vs the model average (95% session-bootstrap interval ${ci})${r.within_noise?', within noise':''}. `+
      `Own average ${mwC(r.closed_mean_cents,3)} per hypothetical trade with a known outcome; counting unknown outcomes, ${mwC(r.lower_mean_cents,3)} to ${mwC(r.upper_mean_cents,3)}. `+
      `${pctText(r.known_fraction)} of chosen trades had a known outcome; picked YES ${pctText(r.yes_fraction)} of the time.`;
    row.tabIndex=0;row.setAttribute('role','listitem');row.setAttribute('aria-label',`Rank ${r.rank??i+1}. ${detail}`);
    const show=()=>{if(!tip)return;tip.textContent=detail;tip.hidden=false;tip.setAttribute('style',`top:${mwNum((row.offsetTop||0)+(row.offsetHeight||0)+4)}px`)};
    const hide=()=>{if(tip)tip.hidden=true};
    row.addEventListener('mouseenter',show);row.addEventListener('focus',show);row.addEventListener('mouseleave',hide);row.addEventListener('blur',hide);
    return row}));
  if(pop)pop.textContent=`${mwScoreboard.population}. ${data.n?.toLocaleString?.()??data.n} forecasts from ${data.sessions} sessions · ${mwScoreboard.policy}. Whiskers: ${mwScoreboard.bootstrap?.interval?.join('–')??'2.5–97.5'} percentile interval from ${mwScoreboard.bootstrap?.resamples??'—'} resamples of whole ${mwScoreboard.bootstrap?.unit??'sessions'}.`;
  if(notes){notes.replaceChildren(...(mwScoreboard.notes||[]).map(t=>mwHtml('li','',t)))}}
function updateLeaderboard(h){if(h!==mwLeaderboardHorizon||!document.getElementById('leaderboard')?.children?.length)renderLeaderboard(h)}

mwInit();renderLeaderboard(mwLeaderboardHorizon);loadScoreboard();
