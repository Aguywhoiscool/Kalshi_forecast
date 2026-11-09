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
      for(const name of ['calculation','training','teaching']){const e=new Element('button');e.dataset.tab=name;tabs.push(e)}}
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

test('teaching animation is explicitly different from actual training',()=>{
  const {context,ids}=harness(()=>{});
  vm.runInContext("explanationData={};explanationTab='teaching';renderExplanation()",context);
  const text=ids.get('explanation-content').textContent;
  assert.match(text,/not the Kalshi model/);
  assert.match(text,/Play teaching demo/);
  assert.doesNotMatch(text,/Play recorded replay/);
});
