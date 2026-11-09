const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');

// DOM contract harness, not a browser layout/screenshot test.
function harness(fetch){
  const ids=new Map(),tabs=[];
  class Element{
    constructor(tag){this.tagName=tag;this.children=[];this.listeners={};this.dataset={};this.open=false;this._text=''}
    set id(v){this._id=v;ids.set(v,this)} get id(){return this._id}
    set textContent(v){this._text=String(v)}
    get textContent(){return this._text+this.children.map(x=>x.textContent||'').join(' ')}
    set innerHTML(v){for(const match of v.matchAll(/id="([^"]+)"/g)){const e=new Element('div');e.id=match[1]}
      for(const name of ['calculation','training','mechanism']){const e=new Element('button');e.dataset.tab=name;tabs.push(e)}}
    append(...nodes){nodes.forEach(n=>{this.children.push(n);if(typeof n==='object')n.parentNode=this})}
    replaceChildren(...nodes){this.children=[];this._text='';this.append(...nodes)}
    setAttribute(k,v){this[k]=v}
    addEventListener(k,fn){this.listeners[k]=fn}
    showModal(){this.open=true}
    close(){this.open=false;this.listeners.close?.()}
    get classList(){return {toggle(){}}}
  }
  const body=new Element('body'),table=new Element('tbody');table.id='predictions';
  const document={body,createElement:t=>new Element(t),getElementById:id=>ids.get(id),querySelectorAll:()=>tabs};
  const context={document,console,AbortController,setTimeout,clearTimeout,setInterval:()=>1,clearInterval(){},
    MutationObserver:class{observe(){}},fetch,state:{markets:[]},selected:'',horizon:30,
    node:(tag,attrs={},text='')=>{const e=new Element(tag);e.textContent=text;Object.assign(e,attrs);return e}};
  vm.createContext(context);
  vm.runInContext(fs.readFileSync('frontend/explanations.js','utf8'),context);
  vm.runInContext(fs.readFileSync('frontend/model_mechanisms.js','utf8'),context);
  return {context,ids};
}

function fixture(id='f'){
  return {forecast_id:id,ticker:'T-'+id,horizon:30,model:'mean',input_end:1000000000,model_run:'frozen-run',
    calculation:{raw:[1,-1],clipped_raw:[1,-1],calibration_correction:[0,0],final:[1,-1],signal:'WAIT',margin_cents:1},
    details:{kind:'mean',training_mean:[1,-1]},training:{available:false,message:'No recorded training trace'}};
}

test('exact forecast provenance survives later live state changes',async()=>{
  const {context,ids}=harness(async()=>({ok:true,json:async()=>fixture('old')}));
  await context.loadExplanation('old','mean',30);
  context.state={markets:[{ticker:'new-live-market'}]};
  assert.match(ids.get('explanation-provenance').textContent,/T-old/);
  assert.match(ids.get('explanation-content').textContent,/exact return calculation/);
});

test('late response cannot replace a newly selected forecast',async()=>{
  let release;
  const {context,ids}=harness(url=>url.includes('/old/')?new Promise(resolve=>{release=resolve}):Promise.resolve({ok:true,json:async()=>fixture('new')}));
  const pending=context.loadExplanation('old','mean',30);
  await context.loadExplanation('new','mean',30);
  release({ok:true,json:async()=>fixture('old')});await pending;
  assert.match(ids.get('explanation-provenance').textContent,/T-new/);
});

test('expired forecast shows explicit error and never substitutes live data',async()=>{
  const {context,ids}=harness(async()=>({ok:false,json:async()=>({detail:'Forecast expired'})}));
  await context.loadExplanation('gone','mean',30);
  assert.equal(ids.get('explanation-content').textContent,'Forecast expired');
});

test('failed new request cannot reopen an old forecast through tabs',async()=>{
  const {context,ids}=harness(async url=>url.includes('/old/')?
    {ok:true,json:async()=>fixture('old')}:{ok:false,json:async()=>({detail:'Forecast expired'})});
  await context.loadExplanation('old','mean',30);
  await context.loadExplanation('gone','ridge',60);
  vm.runInContext("explanationTab='calculation';renderExplanation()",context);
  assert.equal(ids.get('explanation-provenance').textContent,'');
  assert.doesNotMatch(ids.get('explanation-content').textContent,/training mean|exact return/);
});

function mechanismFixture(model){
  const d=fixture();d.model=model;d.input={features:Array.from({length:60},()=>[.4,.42,.02,.41,2,3,-.2])};
  const tree={kind:'tree',steps:[{node:0,leaf:false,feature:'midpoint_last',threshold:.5,observed:.41,direction:'left',training_samples:10},{node:1,leaf:true,value:[1,-1],training_samples:4}]};
  const linear={kind:'linear',intercept:[0,0],features:[{name:'midpoint_last',contribution:[1,-1],raw:.41,standardized:1,coefficient:[1,-1]}]};
  const details={mean:{kind:'mean',training_mean:[1,-1]},ridge:linear,sgd:linear,decision_tree:tree,
    random_forest:{kind:'forest',tree_outputs:[[1,-1],[3,-3]],selected_tree:0,path:tree},
    gradient_boosting:{kind:'boosting',stages:[[1,-1],[2,-2]]},
    gru:{kind:'gru',time_sensitivity:Array(60).fill(.1)},
    monte_carlo:{kind:'monte_carlo',path_count:2048,seed:'123',preview_indices:[0,20],
      preview_paths:[[.41,.42,.43],[.41,.4,.39]],preview_draw_indices:[[0,0],[1,1]],historical_increments:[.01,-.01],
      convergence_yes:[1,2],scenario_percentiles:[[0,0],[1,1],[2,2]]}};
  d.details=details[model];if(model==='monte_carlo')d.horizon=2;
  if(model==='sgd')d.training={optimizer:'SGD',steps:[{step:0,epoch:1,loss:2,learning_rate:.1,gradient_norm:1,before:[[0,0],[0,0]],after:[[.1,.2],[.1,.2]]}]};
  return d;
}
function descendants(e){return [e,...e.children.flatMap(c=>typeof c==='object'?descendants(c):[])]}
const routeTitles={mean:'Mean baseline: summarize',ridge:'Ridge: add signed',sgd:'SGD: recorded',decision_tree:'Decision tree: follow',random_forest:'Random forest: many',gradient_boosting:'Gradient boosting: add',gru:'GRU: carry',monte_carlo:'Monte Carlo: sample'};
for(const [model,title]of Object.entries(routeTitles))test(`${model} renders its own mechanism rather than a universal gradient demo`,()=>{
  const {context,ids}=harness(()=>{});context.testFixture=mechanismFixture(model);
  vm.runInContext("explanationData=testFixture;explanationTab='mechanism';renderExplanation()",context);
  const content=ids.get('explanation-content');assert.ok(content.textContent.includes(title));
  assert.doesNotMatch(content.textContent,/Teaching demo|two-weight bowl|Illustrative quadratic/);
  if(model!=='sgd')assert.ok(descendants(content).some(e=>e['data-mechanism']===model));
  if(model==='gru')assert.match(content.textContent,/not measured hidden states/);
  if(model==='decision_tree')assert.match(content.textContent,/not visited/);
});
test('Monte Carlo step control reveals actual selected random increment',()=>{
  const {context,ids}=harness(()=>{});context.testFixture=mechanismFixture('monte_carlo');
  vm.runInContext("explanationData=testFixture;explanationTab='mechanism';renderExplanation()",context);
  const content=ids.get('explanation-content'),range=descendants(content).find(e=>e.type==='range');
  assert.ok(descendants(content).some(e=>e['aria-label']==='Highlighted actual path endpoint'));
  range.value='1';range.listeners.input();
  assert.match(content.textContent,/draw index 0 selects 1.0000¢/);
  assert.match(content.textContent,/bounded midpoint 42.0000¢/);
});
test('boosting control adds actual stage correction to previous forecast',()=>{
  const {context,ids}=harness(()=>{});context.testFixture=mechanismFixture('gradient_boosting');
  vm.runInContext("explanationData=testFixture;explanationTab='mechanism';renderExplanation()",context);
  const content=ids.get('explanation-content'),range=descendants(content).find(e=>e.type==='range');
  assert.match(content.textContent,/First recorded stage/);
  assert.match(content.textContent,/no zero starting state is assumed/);
  assert.doesNotMatch(content.textContent,/YES 0.0000¢/);
  range.value='1';range.listeners.input();assert.match(content.textContent,/YES 1.0000 \+ 1.0000 = 2.0000¢/);
});
test('SGD projection includes original weights on fixed axes across replay',()=>{
  const {context}=harness(()=>{}),data=mechanismFixture('sgd');
  data.training.steps.push({step:1,epoch:1,loss:1,learning_rate:.1,gradient_norm:1,
    before:[[.1,.2],[.1,.2]],after:[[.4,.5],[.4,.5]]});
  context.testFixture=data;
  vm.runInContext('explanationData=testFixture',context);
  const one=context.sectionlessProjection(data.training.steps.slice(0,1)),two=context.sectionlessProjection(data.training.steps);
  const initial=svg=>descendants(svg).find(e=>e['data-role']==='initial-weights');
  assert.equal(initial(one).cx,initial(two).cx);assert.equal(initial(one).cy,initial(two).cy);
  assert.match(two.textContent,/actual current update/);
});
