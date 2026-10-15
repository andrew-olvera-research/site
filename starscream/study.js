/* Figures and tables use the curated, episode-verified public data export. */
(async () => {
  const NS = 'http://www.w3.org/2000/svg';
  const labels = {mlp:'MLP',lstm:'LSTM',transformer:'Transformer'};
  const colors = {mlp:'#337b70',lstm:'#b46b2d',transformer:'#355d9a'};
  const metrics = [['clean_timely','Clean & Timely'],['clean','Clean'],['eventual','Eventual']];
  const el = (tag, attrs={}, text='') => { const n=document.createElement(tag); for(const [k,v] of Object.entries(attrs)) n.setAttribute(k,v); n.textContent=text; return n; };
  const svg = (tag,attrs={},text='') => {const n=document.createElementNS(NS,tag); for(const [k,v] of Object.entries(attrs)) n.setAttribute(k,v); n.textContent=text; return n;};
  try {
    const response=await fetch('assets/data/development-comparison.json'); if(!response.ok) throw Error('Comparison data unavailable');
    const D=await response.json();
    for(const phase of ['nominal','recovery']) {
      const panel=D.panels[phase], container=document.getElementById(`${phase}-charts`);
      for(const [key,label] of metrics) {
        const f=el('figure',{'class':'metric-chart'}), title=el('h3',{},label);
        const chart=svg('svg',{viewBox:'0 0 320 230',role:'img','aria-label':`${phase} ${label}: mean percentages and individual training seeds`});
        for(const [i,model] of Object.keys(labels).entries()) {
          const data=panel.models[model], stat=data.statistics[key], y=25+i*66;
          chart.append(svg('text',{x:10,y,fill:'#34404b','font-size':14},labels[model]));
          chart.append(svg('text',{x:310,y,'text-anchor':'end',fill:'#34404b','font-size':14},`${stat.mean.toFixed(1)} ± ${stat.sample_sd.toFixed(1)}%`));
          chart.append(svg('rect',{x:10,y:y+12,width:300,height:12,fill:'#edf0f4',rx:2}));
          chart.append(svg('rect',{x:10,y:y+12,width:3*stat.mean,height:12,fill:colors[model],rx:2}));
          for(const [j,seed] of data.seeds.entries()) {
            const rate=100*seed.counts[key]/seed.denominator;
            const dot=svg('circle',{cx:10+3*rate,cy:y+18+(j-1)*4,r:3.2,tabindex:'0','aria-label':`${labels[model]} seed ${seed.seed_index}: ${seed.counts[key]} of ${seed.denominator}, ${rate.toFixed(2)} percent`,fill:'white',stroke:'#26313f','stroke-width':1.2});
            dot.append(svg('title',{},`${labels[model]}, seed ${seed.seed_index}: ${seed.counts[key]}/${seed.denominator} (${rate.toFixed(2)}%)`)); chart.append(dot);
          }
        }
        chart.append(svg('text',{x:10,y:222,fill:'#58616f','font-size':12},'0%'));
        chart.append(svg('text',{x:310,y:222,'text-anchor':'end',fill:'#58616f','font-size':12},'100% of assigned starts'));
        f.append(title,chart);container.append(f);
      }
    }
    const G=await (await fetch('assets/data/course-geometry.json')).json(), g=G.sets.eval100;
    const summary=[[`${g.ranges.gates_per_course.min}–${g.ranges.gates_per_course.max}`,'gates per course'],[`${g.ranges.spacing_m.min.toFixed(1)}–${g.ranges.spacing_m.max.toFixed(1)} m`,'gate-center spacing'],[`${g.course_counts.turn_at_least_120deg} / 100`,'courses with a ≥120° chord turn'],[`${g.course_counts.height_step_at_least_2m} / 100`,'courses with a ≥2 m height step']];
    const box=document.getElementById('geometry-summary');
    for(const [value,label] of summary){const item=el('div');item.append(el('strong',{},value),el('span',{},label));box.append(item);}
  } catch(error) {
    document.getElementById('data-status').textContent='The figures could not load. Download the comparison data below to inspect the results.';
    console.error(error);
  }

  // A small curated selection from recorded episodes; source names stay in the data.
  const choices={clean:[['v622-hard-v2b-compound-reversal-021-03-clean','Compound reversal'],['v622-hard-v2b-vertical-chain-008-06-clean','Vertical chain'],['v622-hard-long-braking-023-005-arena1-mirror1-clean','Long braking']],recovery:[['v622-hard-go-around-017-040-mirror1-recovery','Go-around recovery'],['v622-hard-slalom-055-040-arena1-late-recovery','Late slalom recovery'],['a2rl-s2-2026-source-consistent-v2-arena1-failure','Public-track adaptation: failure']]};
  try {
    const index=await (await fetch('assets/rollouts/index.json')).json();
    for(const [group,items] of Object.entries(choices)) {
      const host=document.querySelector(`.explorer[data-set=${group}]`),view=el('div',{'class':'scene3d'}),buttons=el('div',{'class':'replay-choices','role':'group','aria-label':`${group} episodes`}),info=el('p',{'class':'replay-info','aria-live':'polite'});
      host.append(view,buttons,info);const player=new Scene3D(view);let selection=0;
      async function select(slug,label,button){
        const current=++selection;
        buttons.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
        const record=index.find(x=>x.slug===slug); if(!record) throw Error('Missing curated replay');
        const data=await Scene3D.fetch(`assets/rollouts/${slug}.json`);if(current!==selection)return;
        player.load({...data,name:label});
        const result=record.clean?'Clean completion':record.completed?(record.timely?'Timely recovery':'Late recovery'):'Failure';
        info.textContent=`${label} · ${record.time.toFixed(2)} s · ${record.passed}/${record.gates} gates · ${result}`;
      }
      for(const [slug,label] of items){const button=el('button',{type:'button','aria-pressed':'false'},label);button.onclick=()=>select(slug,label,button).catch(console.error);buttons.append(button);}
      await select(items[0][0],items[0][1],buttons.firstChild);
    }
  }catch(error){console.error(error);document.getElementById('replay-status').textContent='Interactive replays could not load; the video above remains available.';}
})();
