const $=id=>document.getElementById(id);
let data=null,result=null,busy=false,rows=[],snapshot=null,stopRequested=false;
const controls=[['coverage_weight','전체 관측',1],['importance_weight','영역 중요도',1],['overlap_penalty','중복 관측 penalty',0],['sensor_count_penalty','센서 수 penalty',0],['installation_cost_weight','설치 비용',0]];
for(const [id,title,value] of controls){
 const label=document.createElement('label');label.htmlFor=id;label.textContent=title;
 const out=document.createElement('output');out.id=id+'v';out.textContent=value;label.append(out);
 const input=document.createElement('input');Object.assign(input,{id,type:'range',min:0,max:10,step:0.1,value});
 $('weights').append(label,input);
}
const methods=()=>[...document.querySelectorAll('input[name=method]:checked')].map(el=>el.value);
function status(text){$('status').textContent=text}
function budgets(){
 const start=Number($('budget').value),end=$('budgetMode').value==='range'?Number($('budgetEnd').value):start;
 if(!Number.isInteger(start)||!Number.isInteger(end)||start<0||end>100||end<start)throw Error('K 범위를 확인하세요 (0~100, 시작 ≤ 종료)');
 if(methods().includes('mappo')&&(start<1||end>32))throw Error('MAPPO의 K 범위는 1~32입니다');
 return Array.from({length:end-start+1},(_,i)=>start+i);
}
function invalidate(){
 result=null;rows=[];snapshot=null;$('download').disabled=true;
 $('metrics').textContent='';$('poses').textContent='';$('selectedTitle').textContent='배치 결과';
 renderTable();renderChart();
}
function change(){
 if(busy)return;
 invalidate();
 for(const [id] of controls)$(id+'v').textContent=$(id).value;
 for(const method of ['greedy','milp','mappo'])$(method+'Options').hidden=!methods().includes(method);
 $('rangeOptions').hidden=$('budgetMode').value!=='range';
 try{$('plan').textContent=methods().length+'개 알고리즘 × '+budgets().length+'개 K = '+methods().length*budgets().length+'개 작업'}catch(e){$('plan').textContent=e.message}
 if(data){draw();status('설정 변경 · 실행 대기')}
}
for(const el of document.querySelectorAll('aside input:not([type=file]),#budgetMode'))el.addEventListener('input',change);
for(const el of [$('mode'),$('scenario'),$('file')])el.addEventListener('change',()=>{
 invalidate();data=null;$('solve').disabled=true;Plotly.purge('plot');status('입력 변경 · 환경을 불러오세요');
});
async function api(path,payload){
 const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
 const text=await response.text();let body;
 try{body=JSON.parse(text)}catch(e){throw Error('서버 응답 오류: HTTP '+response.status)}
 if(!response.ok)throw Error(body.error||response.statusText);
 return body;
}
function lockInputs(value){
 busy=value;
 for(const el of document.querySelectorAll('aside input,aside select,aside button'))el.disabled=value;
 $('stop').disabled=false;$('solve').disabled=value||!data;
}
function metrics(record){
 const density=record.density,n=density.length;
 if(!n)throw Error('평가 target이 없습니다');
 const unseen=density.filter(d=>d===0).length,overlap=density.filter(d=>d>=2).length;
 return {...record,num_targets:n,uncovered_count:unseen,uncovered_pct:100*unseen/n,
         overlap_count:overlap,overlap_pct:100*overlap/n};
}
function renderTable(){
 renderFilters();
 $('comparisonRows').replaceChildren();
 rows.forEach((row,index)=>{
  const tr=document.createElement('tr'),first=document.createElement('td'),graphCell=document.createElement('td');
  if(row.result){
   const check=document.createElement('input');check.type='checkbox';check.name='graphResult';
   check.setAttribute('aria-label',row.method.toUpperCase()+' K='+row.k+' 그래프');
   check.checked=row.chartSelected!==false;
   check.onchange=()=>{row.chartSelected=check.checked;renderFilters();renderChart()};
   graphCell.append(check);
  }
  tr.append(graphCell);
  if(row.result){
   const radio=document.createElement('input');radio.type='radio';radio.name='viewResult';
   radio.setAttribute('aria-label',row.method.toUpperCase()+' K='+row.k+' 지도');
   radio.checked=result===row.result;radio.onchange=()=>selectResult(index);first.append(radio);
  }
  tr.append(first);
  const r=row.result;
  const cells=[row.method.toUpperCase(),row.k,r?r.poses.length:'—',r?r.coverage_pct.toFixed(2):'—',
   r?r.overlap_pct.toFixed(2):'—',r?r.uncovered_pct.toFixed(2):'—',r?r.uncovered_count:'—',
   r?r.objective_score.toFixed(2):'—',row.error||row.status];
  for(const value of cells){const td=document.createElement('td');td.textContent=value;tr.append(td)}
  $('comparisonRows').append(tr);
 });
}
function renderFilters(){
 for(const [container,key,label] of [['methodFilters','method',v=>v.toUpperCase()],['kFilters','k',v=>'K='+v]]){
  $(container).replaceChildren();
  for(const value of [...new Set(rows.map(row=>row[key]))]){
   const matching=rows.filter(row=>row[key]===value&&row.result);
   const selected=matching.filter(row=>row.chartSelected!==false).length;
   const wrapper=document.createElement('label'),check=document.createElement('input');
   check.type='checkbox';check.name=key==='method'?'resultMethod':'resultK';check.value=value;
   check.disabled=!matching.length;check.checked=selected>0;
   check.indeterminate=selected>0&&selected<matching.length;
   check.onchange=()=>{matching.forEach(row=>row.chartSelected=check.checked);renderTable();renderChart()};
   wrapper.append(check,document.createTextNode(' '+label(value)));$(container).append(wrapper);
  }
 }
 $('chartCount').textContent=rows.filter(row=>row.result&&row.chartSelected!==false).length+'개 결과 선택';
}
for(const [id,selected] of [['selectAllResults',true],['clearResults',false]]){
 $(id).onclick=()=>{rows.forEach(row=>row.chartSelected=selected);renderTable();renderChart()};
}
function renderChart(){
 const metric=$('chartMetric').value,allK=[...new Set(rows.map(row=>row.k))].sort((a,b)=>a-b);
 const colors={greedy:'#008978',milp:'#b24b20',mappo:'#804cac'};
 const traces=[...new Set(rows.map(row=>row.method))].map(method=>{
  const group=allK.map(k=>rows.find(row=>row.method===method&&row.k===k));
  return {type:'scatter',mode:'lines+markers',name:method.toUpperCase(),x:allK,
   y:group.map(row=>row?.chartSelected===false?null:row?.result?.[metric]??null),customdata:group.map(row=>rows.indexOf(row)),
   connectgaps:false,line:{color:colors[method]},marker:{size:8,color:colors[method]},type:$('chartType').value};
 });
 Plotly.react('comparison',traces,{margin:{l:55,r:15,t:20,b:45},
  xaxis:{title:'센서 예산 K',dtick:1},yaxis:{title:$('chartMetric').selectedOptions[0].textContent,...(metric.endsWith('_pct')?{range:[0,100]}:{})},
  barmode:'group',legend:{orientation:'h',itemclick:false,itemdoubleclick:false},uirevision:metric},{responsive:true});
}
$('chartMetric').onchange=renderChart;
$('chartType').onchange=renderChart;
function selectResult(index){
 result=rows[index].result;if(!result)return;
 const r=result;
 $('selectedTitle').textContent=r.method.toUpperCase()+' · 예산 K='+r.budget_k+' · 설치 '+r.poses.length+'대'+(r.method==='mappo'?' · 탐색 최고 목적값 배치':'');
 $('metrics').innerHTML='<div>Coverage<strong>'+r.coverage_pct.toFixed(2)+'%</strong></div><div>중복 target<strong>'+r.overlap_pct.toFixed(2)+'%</strong></div><div>미관측 target<strong>'+r.uncovered_pct.toFixed(2)+'%</strong></div>';
 if(r.method==='mappo')$('metrics').innerHTML+='<div>최종 정책 Coverage<strong>'+r.metadata.deterministic_policy.coverage_pct.toFixed(2)+'%</strong></div>';
 $('poses').innerHTML=r.poses.map((p,i)=>'<tr><td>'+(i+1)+'</td>'+p.map(x=>'<td>'+x.toFixed(2)+'</td>').join('')+'</tr>').join('');
 renderTable();draw(r.density);
}
function draw(density=[]){if(!data)return;const v=data.vertices,t=data.triangles;const xyz=(a,i)=>a.map(p=>p[i]);const traces=[{type:'mesh3d',x:xyz(v,0),y:xyz(v,1),z:xyz(v,2),i:xyz(t,0),j:xyz(t,1),k:xyz(t,2),color:'#87969f',opacity:.35,name:'환경',hoverinfo:'skip'}];
for(const [name,color,test] of [['미관측','#d4773b',d=>!d],['관측','#008978',d=>d===1],['중복 관측','#9359a3',d=>d>1]]){const pts=data.targets.filter((p,i)=>test(density[i]||0));traces.push({type:'scatter3d',mode:'markers',x:xyz(pts,0),y:xyz(pts,1),z:xyz(pts,2),marker:{size:3,color},name})}
const candidates=data.candidates;traces.push({type:'scatter3d',mode:'markers',x:xyz(candidates,0),y:xyz(candidates,1),z:xyz(candidates,2),marker:{size:2,color:'#718bb0'},name:'설치 후보',visible:'legendonly'});
if(result){const p=result.poses;traces.push({type:'scatter3d',mode:'markers+text',x:xyz(p,0),y:xyz(p,1),z:xyz(p,2),text:p.map((_,i)=>'L'+(i+1)),marker:{size:8,color:'#cf2861'},name:'LiDAR'});}
Plotly.react('plot',traces,{margin:{l:0,r:0,t:0,b:0},uirevision:'scene',scene:{aspectmode:'data',xaxis:{title:'X [m]'},yaxis:{title:'Y [m]'},zaxis:{title:'Z [m]'}},legend:{orientation:'h'}},{responsive:true});}

async function solve(){
 if(!data||busy)return;
 try{
  if(!methods().length)throw Error('알고리즘을 하나 이상 선택하세요');
  for(const el of document.querySelectorAll('aside input[type=number]'))if(el.offsetParent!==null&&!el.checkValidity()){el.reportValidity();return}
  const ks=budgets(),chosen=methods();invalidate();
  snapshot={weights:Object.fromEntries(controls.map(([id])=>[id,Number($(id).value)])),
   region:$('regionOn').checked?{bounds:['xmin','ymin','xmax','ymax'].map(id=>Number($(id).value)),factor:Number($('factor').value)}:null,
   ...Object.fromEntries(['greedy_steps','milp_seconds','milp_gap','mappo_steps','seed'].map(id=>[id,Number($(id).value)]))};
  rows=ks.flatMap(k=>chosen.map(method=>({k,method,status:'대기'})));
 }catch(e){status(e.message);return}
 lockInputs(true);stopRequested=false;$('stop').hidden=false;renderTable();renderChart();draw();
 let completed=0;
 for(let index=0;index<rows.length;index++){
  if(stopRequested)break;
  const row=rows[index];row.status='실행 중';renderTable();
  const label=(index+1)+' / '+rows.length+' · '+row.method.toUpperCase()+' · K='+row.k;
  status(label);let active=true;
  const poll=row.method==='mappo'?setInterval(async()=>{
   try{const p=await(await fetch('/api/progress')).json();if(active)status(label+' · 학습 '+p.step+'/'+p.total)}catch(e){}
  },1000):null;
  try{
   const record=await api('/api/solve',{...snapshot,method:row.method,budget_k:row.k});
   row.result=metrics(record);row.status='완료';selectResult(index);
  }catch(e){row.status='실패';row.error=e.message}
  finally{active=false;clearInterval(poll)}
  completed++;renderTable();renderChart();$('download').disabled=false;
 }
 for(const row of rows)if(row.status==='대기')row.status='미실행';
 renderTable();lockInputs(false);$('stop').hidden=true;
 const failed=rows.filter(row=>row.status==='실패').length;
 status((stopRequested?'남은 작업 중단':'비교 완료')+' · 처리 '+completed+'/'+rows.length+' · 실패 '+failed);
}
$('stop').onclick=()=>{stopRequested=true;$('stop').disabled=true;status('현재 작업 완료 후 남은 조합 중단')};
$('load').onclick=async()=>{
 if(busy)return;invalidate();data=null;Plotly.purge('plot');lockInputs(true);status('환경과 visibility 계산 중');
 try{
  const payload={mode:$('mode').value,scenario_id:Number($('scenario').value)};
  if(payload.mode==='custom'){if(!$('file').files.length)throw Error('JSON 파일을 선택하세요');payload.data=JSON.parse(await $('file').files[0].text())}
  data=await api('/api/load',payload);$('source').textContent=data.source;draw();status('환경 준비 완료 · 실행 대기');
 }catch(e){status(e.message)}finally{lockInputs(false)}
};
$('solve').onclick=solve;
$('download').onclick=()=>{
 const bundle={schema_version:'placement_ui_comparison_v1',source:data.source,settings:snapshot,
  metric_definitions:{overlap:'fraction of targets observed by >=2 sensors',uncovered:'fraction of targets observed by 0 sensors',budget:'at most K'},
  runs:rows};
 const url=URL.createObjectURL(new Blob([JSON.stringify(bundle,null,2)],{type:'application/json'}));
 const a=document.createElement('a');a.href=url;a.download='placement_comparison.json';a.click();URL.revokeObjectURL(url);
};
change();
$('comparison').on('plotly_click',event=>{const index=event.points[0].customdata;if(rows[index]?.result)selectResult(index)});
$('load').click();
