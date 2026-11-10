/* Distinct views of recorded calculations; diagrams are labeled when schematic. */
function mechanismCanvas(parent,kind,height=320){
  const svg=node('svg',{viewBox:`0 0 800 ${height}`,role:'img','data-mechanism':kind,'aria-label':`${kind} mechanism`});
  parent.append(svg);return svg;
}
function mechanismBox(svg,x,y,w,h,title,subtitle='',color='#66d9ff'){
  svg.append(node('rect',{x,y,width:w,height:h,rx:8,fill:'#1b2230',stroke:color}),
    node('text',{x:x+w/2,y:y+h/2-3,'text-anchor':'middle',fill:color},title),
    node('text',{x:x+w/2,y:y+h/2+15,'text-anchor':'middle'},subtitle));
}
function mechanismArrow(svg,x1,y1,x2,y2,color='#5c6880'){
  svg.append(node('line',{x1,y1,x2,y2,stroke:color,'stroke-width':2}),
    node('circle',{cx:x2,cy:y2,r:3,fill:color}));
}
function mechanismReplay(parent,count,draw,label,options={}){
  const wrapper=element('div','','ex-controls'),button=element('button',`Play ${label}`),range=element('input'),caption=element('span');
  range.type='range';range.min='0';range.max=String(count-1);range.value=String(options.initial??0);range.setAttribute('aria-label',label);
  let restart=Boolean(options.restartOnPlay);
  function update(){caption.textContent=`${Number(range.value)+1} / ${count}`;draw(Number(range.value))}
  range.addEventListener('input',()=>{restart=Number(range.value)===count-1;update()});
  button.addEventListener('click',()=>{if(replayTimer){clearInterval(replayTimer);replayTimer=null;button.textContent=`Play ${label}`;return}if(restart){range.value='0';restart=false;update()}button.textContent='Pause';replayTimer=setInterval(()=>{const next=Number(range.value)+1;if(next>=count&&options.loop===false){clearInterval(replayTimer);replayTimer=null;restart=true;button.textContent=`Play ${label}`;return}range.value=next%count;update()},150)});
  wrapper.append(button,range,caption);parent.append(wrapper);update();
  return {redraw:update,range};
}
function mechanismTree(data){
  const d=data.details.kind==='forest'?data.details.path:data.details;
  const s=section('Decision tree: follow the comparisons','Actual visited nodes and chosen branches. Grey branches are unexpanded alternatives, not invented leaf predictions.');
  const rows=d.steps,height=Math.max(270,rows.length*90+40),svg=mechanismCanvas(s,'decision_tree',height);
  let x=400;
  rows.forEach((step,i)=>{
    const y=30+i*90;
    if(!step.leaf){
      const direction=step.direction==='left'?-1:1,next=Math.max(180,Math.min(620,x+direction*85));
      mechanismArrow(svg,x,y+43,next,y+90,'#66d9ff');
      mechanismArrow(svg,x,y+43,x-direction*85,y+75,'#36425a');
      svg.append(node('text',{x:x-direction*90,y:y+82,'text-anchor':'middle'},'not visited'));
      mechanismBox(svg,x-140,y,280,44,`${step.feature} ≤ ${number(step.threshold)}`,`observed ${number(step.observed)} → ${step.direction}`);
      x=next;
      // Fold very deep paths back into the viewport; topology remains a path.
      x=Math.max(180,Math.min(620,x));
    }else mechanismBox(svg,x-140,y,280,50,`Leaf ${step.node}`,`YES ${number(step.value[0])}¢ · NO ${number(step.value[1])}¢`);
  });
  tableInto(s,['Visited node','Training samples'],rows.map(r=>[r.node,r.training_samples]));
}
function mechanismForest(data){
  const d=data.details,s=section('Random forest: many trees, one average','Tree glyphs are schematic. Their numerical outputs are actual; every tree contributes equally to the raw ensemble mean.');
  const shown=d.tree_outputs.slice(0,12),svg=mechanismCanvas(s,'random_forest',360),mean=[0,1].map(side=>d.tree_outputs.reduce((v,r)=>v+r[side],0)/d.tree_outputs.length);
  shown.forEach((values,i)=>{const x=40+i*62;
    svg.append(node('line',{x1:x+20,y1:50,x2:x+6,y2:80,stroke:'#7b88a0'}),node('line',{x1:x+20,y1:50,x2:x+34,y2:80,stroke:'#7b88a0'}),node('circle',{cx:x+20,cy:50,r:7,fill:'#66d9ff'}),node('circle',{cx:x+6,cy:80,r:5,fill:'#6d7a90'}),node('circle',{cx:x+34,cy:80,r:5,fill:'#6d7a90'}),node('text',{x:x+20,y:110,'text-anchor':'middle'},`Tree ${i}`),node('text',{x:x+20,y:130,'text-anchor':'middle'},`${values[0].toFixed(1)}¢`));
    mechanismArrow(svg,x+20,145,400,235,'#33415e');
  });
  svg.append(node('text',{x:30,y:20},'Actual member YES-return estimates'));
  mechanismBox(svg,240,235,320,65,`Mean of all ${d.tree_outputs.length} trees`,`YES ${number(mean[0])}¢ · NO ${number(mean[1])}¢`);
  svg.append(node('text',{x:400,y:335,'text-anchor':'middle'},`Showing ${shown.length}/${d.tree_outputs.length} member glyphs; the average uses every member`));
  const picker=element('select');picker.setAttribute('aria-label','Actual forest tree path');
  d.tree_outputs.forEach((_,i)=>{const option=element('option',`Inspect tree ${i}`);option.value=i;option.selected=i===d.selected_tree;picker.append(option)});
  picker.addEventListener('change',()=>loadExplanation(data.forecast_id,data.model,data.horizon,Number(picker.value)));s.append(picker);
  mechanismTree(data);
}
function mechanismBoosting(data){
  const stages=data.details.stages,s=section('Gradient boosting: add successive refinements','Each stage adds a correction to the previous cumulative forecast. The first displayed value includes the initial prediction plus the first tree.');
  const holder=element('div'),caption=element('p','','ex-muted');s.append(holder,caption);
  mechanismReplay(s,stages.length,i=>{
    holder.replaceChildren();const svg=mechanismCanvas(holder,'gradient_boosting',240);
    if(i===0){
      mechanismBox(svg,185,75,430,80,'First recorded stage',`YES ${number(stages[0][0])}¢ · NO ${number(stages[0][1])}¢`);
      caption.textContent='First recorded cumulative prediction includes the learned baseline plus the first tree. The separate baseline value is not exposed; no zero starting state is assumed.';
      return;
    }
    const prior=stages[i-1],delta=stages[i].map((v,j)=>v-prior[j]);
    mechanismBox(svg,20,75,220,70,i?'Previous prediction':'Start',`YES ${number(prior[0])}¢`);
    mechanismBox(svg,295,75,210,70,i?`Stage ${i+1} correction`:'Initial + stage 1',`YES ${number(delta[0])}¢`,'#ffb08a');
    mechanismBox(svg,560,75,220,70,'Updated prediction',`YES ${number(stages[i][0])}¢`);
    mechanismArrow(svg,240,110,295,110);mechanismArrow(svg,505,110,560,110);
    caption.textContent=`Actual stage ${i+1}: YES ${number(prior[0])} + ${number(delta[0])} = ${number(stages[i][0])}¢; NO ${number(prior[1])} + ${number(delta[1])} = ${number(stages[i][1])}¢.`;
  },'boosting stages');
  linePlot(s,[{values:stages.map(v=>v[0]),color:'#66d9ff'},{values:stages.map(v=>v[1]),color:'#ffb08a'}],{labelX:'Actual boosting stage',labelY:'Cumulative raw return (¢)'});
}
function mechanismLinear(data){
  const d=data.details,s=section('Ridge: add signed feature contributions','Actual standardized inputs × learned coefficients. Ridge solves a regularized linear fit; this is not a fabricated gradient-descent replay.');
  const ranked=[...d.features].sort((a,b)=>Math.abs(b.contribution[0])-Math.abs(a.contribution[0]));
  const parts=[{name:'Intercept',value:d.intercept[0]},...ranked.slice(0,8).map(f=>({name:f.name,value:f.contribution[0]})),
    {name:'Remaining features',value:ranked.slice(8).reduce((sum,f)=>sum+f.contribution[0],0)}];
  let sum=0;const bars=parts.map(p=>{const start=sum;sum+=p.value;return {...p,start,end:sum}}),all=bars.flatMap(b=>[b.start,b.end]),lo=Math.min(0,...all),hi=Math.max(0,...all),span=hi-lo||1;
  const svg=mechanismCanvas(s,'ridge',340),y=v=>270-(v-lo)/span*220;
  svg.append(node('line',{x1:25,x2:780,y1:y(0),y2:y(0),stroke:'#4e5a70'}));
  bars.forEach((b,i)=>{const x=30+i*75;svg.append(node('rect',{x,y:y(Math.max(b.start,b.end)),width:52,height:Math.max(1,Math.abs(y(b.start)-y(b.end))),fill:b.value>=0?'#66d9ff':'#ffb08a',rx:3}),node('text',{x:x+26,y:300,'text-anchor':'middle'},i===0?'Bias':i===bars.length-1?'Rest':`Feature ${i}`),node('text',{x:x+26,y:Math.max(20,y(Math.max(b.start,b.end))-8),'text-anchor':'middle'},b.value.toFixed(2)));if(i<bars.length-1)svg.append(node('line',{x1:x+52,x2:x+75,y1:y(b.end),y2:y(b.end),stroke:'#4e5a70'}))});
  svg.append(node('text',{x:30,y:330},`YES raw sum ${number(sum)}¢ · clipped and calibrated separately in the exact calculation`));
  tableInto(s,['Component','YES contribution','NO contribution'],[['Intercept',...d.intercept.map(number)],...ranked.map(f=>[f.name,...f.contribution.map(number)])]);
}
function mechanismMC(data){
  const d=data.details,s=section('Monte Carlo: sample possible futures','Actual bootstrap paths for this forecast. Each step draws one historical increment, then bounds the price. Quiet histories can produce overlapping paths; no jitter is added. These are scenarios, not confidence intervals.');
  const holder=element('div'),caption=element('p','','ex-muted');s.append(holder,caption);
  const paths=d.preview_paths,lo=Math.min(...paths.flat()),hi=Math.max(...paths.flat()),span=hi-lo||.01;
  let selectedPath=0;
  const picker=element('select');picker.setAttribute('aria-label','Highlight actual simulated path');
  d.preview_indices.forEach((index,i)=>{const option=element('option',`Actual path ${index}`);option.value=i;picker.append(option)});
  picker.value='0';s.append(picker);
  const identical=paths.every(path=>path.every((v,i)=>Math.abs(v-paths[0][i])<1e-12));
  const flat=identical&&paths[0].every(v=>Math.abs(v-paths[0][0])<1e-12);
  if(flat)s.append(element('p','The displayed paths overlap and stay at one price for this observed history. This is a valid flat simulation, not a broken chart; no artificial movement or jitter is added.','notice'));
  else if(identical)s.append(element('p','The displayed sampled paths coincide. No artificial jitter is added to force them apart.','notice'));
  const replay=mechanismReplay(s,data.horizon+1,step=>{
    holder.replaceChildren();const svg=mechanismCanvas(holder,'monte_carlo',300),x=t=>50+t/data.horizon*700,y=value=>250-(value-lo)/span*220;
    paths.forEach((path,i)=>svg.append(node('polyline',{points:path.slice(0,step+1).map((v,t)=>`${x(t)},${y(v)}`).join(' '),fill:'none',stroke:i===selectedPath?'#ffb08a':'#66d9ff','stroke-width':i===selectedPath?3:1,opacity:i===selectedPath?1:.24,'data-path-index':d.preview_indices[i]})));
    svg.append(node('circle',{cx:x(step),cy:y(paths[selectedPath][step]),r:5,fill:'#ffd2bd','aria-label':'Highlighted actual path endpoint'}));
    const index=step?d.preview_draw_indices[selectedPath][step-1]:null;
    caption.textContent=step?`Actual path ${d.preview_indices[selectedPath]} · second ${step} · draw index ${index} selects ${(d.historical_increments[index]*100).toFixed(4)}¢ · bounded midpoint ${(paths[selectedPath][step]*100).toFixed(4)}¢.`:`${d.path_count} seeded paths begin at the same observed midpoint. Showing ${paths.length} actual members.`;
    svg.append(node('text',{x:50,y:285},'Observed midpoint'),node('text',{x:690,y:285},`${data.horizon}s horizon`));
  },'actual random steps',{initial:data.horizon,restartOnPlay:true,loop:false});
  picker.addEventListener('change',()=>{selectedPath=Number(picker.value);replay.redraw()});
  const convergence=section('How the scenario mean settles','Cumulative YES return over all actual simulated paths; convergence does not establish accuracy.');
  linePlot(convergence,[{values:d.convergence_yes,color:'#66d9ff'}],{labelX:'Number of simulated paths',labelY:'Mean net scenario return (¢)'});
  const statistics=section('Scenario dispersion','Percentiles of simulated returns, not validated confidence intervals.');
  tableInto(statistics,['Percentile','YES return','NO return'],['5th','50th','95th'].map((label,i)=>[label,...d.scenario_percentiles[i].map(number)]));
  const download=element('a','Download every path and actual draw index','ex-download');
  download.href=`/api/explanations/${data.forecast_id}/monte_carlo?horizon=${data.horizon}&download=true`;statistics.append(download);
}
function mechanismGRU(data){
  const d=data.details,s=section('GRU: carry a state through 60 observations','Architecture schematic, not measured hidden states or gate activations. Highlighted inputs and sensitivity values come from this actual prediction.');
  const holder=element('div'),caption=element('p','','ex-muted');s.append(holder,caption);
  mechanismReplay(s,60,step=>{
    holder.replaceChildren();const svg=mechanismCanvas(holder,'gru',330);
    for(let i=0;i<60;i++)svg.append(node('rect',{x:25+i*12.5,y:35,width:9,height:30,fill:i===step?'#ffb08a':i<step?'#7b88a0':'#2a3446',rx:2}));
    svg.append(node('text',{x:25,y:20},'Actual 60-second input sequence'));
    const boxes=[['GRU · 64 units','Sequence of recurrent states'],['GRU · 32 units','Final recurrent state'],['Dense · 32','Learned nonlinear mapping'],['YES / NO','Two raw return estimates']];
    boxes.forEach(([title,sub],i)=>{mechanismBox(svg,20+i*195,145,175,75,title,sub,i===3?'#ffb08a':'#66d9ff');if(i<3)mechanismArrow(svg,195+i*195,182,215+i*195,182)});
    mechanismArrow(svg,100,75,100,145,'#ffb08a');
    svg.append(node('path',{d:'M 35 145 C 0 100, 175 100, 175 145',fill:'none',stroke:'#5c6880','stroke-width':2}),node('text',{x:50,y:108},'recurrent state'),node('text',{x:25,y:280},'Recurrent loops and connections are schematic; no invented internal values shown.'));
    const input=data.input.features[step];caption.textContent=`Actual observation ${step+1}/60: bid ${(input[0]*100).toFixed(2)}¢, ask ${(input[1]*100).toFixed(2)}¢, midpoint ${(input[3]*100).toFixed(2)}¢; local sensitivity magnitude ${number(d.time_sensitivity[step])}.`;
  },'input sequence');
  linePlot(s,[{values:d.time_sensitivity,color:'#ffb08a'}],{labelX:'Actual input observation',labelY:'Local YES-return sensitivity'});
}
function mechanismMean(data){
  const s=section('Mean baseline: summarize training targets','Schematic aggregation, using the recorded training means. Individual training targets are not exposed here and are not fabricated.');
  const svg=mechanismCanvas(s,'mean',240);
  mechanismBox(svg,20,70,230,80,'Training target observations','Individual values not displayed');
  mechanismBox(svg,305,70,170,80,'Arithmetic mean','Sum targets / count');
  mechanismBox(svg,530,70,250,80,'Recorded mean returns',`YES ${number(data.details.training_mean[0])}¢ · NO ${number(data.details.training_mean[1])}¢`);
  mechanismArrow(svg,250,110,305,110);mechanismArrow(svg,475,110,530,110);
}
function renderModelMechanism(data){
  const renderers={mean:mechanismMean,ridge:mechanismLinear,decision_tree:mechanismTree,random_forest:mechanismForest,
    gradient_boosting:mechanismBoosting,monte_carlo:mechanismMC,gru:mechanismGRU,
    sgd:d=>{section('SGD: recorded gradient updates','Actual optimizer steps from this training run. The plotted two coefficients are a projection, not a fictitious full loss surface.');renderTraining(d)}};
  if(renderers[data.model])renderers[data.model](data);else section('Mechanism unavailable','No substitute visualization is shown.');
}
