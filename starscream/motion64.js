/* Uses the existing Scene3D renderer; trajectories are recorded physical samples. */
(async()=>{
 const host=document.getElementById('flight-viewer'), status=document.getElementById('replay-status');
 try {
  const response=await fetch('assets/motion64/manifest.json'); if(!response.ok)throw Error('Flight manifest unavailable');
  const manifest=await response.json(), assets=manifest.assets.filter(a=>a.role==='training');
  const viewer=new Scene3D(host); let request=0;
  async function select(index){
   const current=++request, a=assets[index]; status.textContent='Loading recorded flight…';
   const data=await Scene3D.fetch('assets/motion64/'+a.trajectory_json); if(current!==request)return;
   viewer.load(data); document.getElementById('course-title').textContent=a.title;
   document.getElementById('course-info').textContent=`${a.gate_count} gates · ${a.duration_seconds.toFixed(3)} s · learned seed geometry`;
   document.querySelectorAll('[data-course]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.course)===index)));
   status.textContent='Recorded at 130 Hz · deterministic mean policy · same selected motion64 checkpoint';
  }
  document.querySelectorAll('[data-course]').forEach(b=>b.addEventListener('click',()=>select(Number(b.dataset.course)).catch(fail)));
  function fail(error){status.textContent='The 3D flight could not load. Use the flight video or GIF below.';console.error(error);}
  await select(0);
 } catch(error){status.textContent='The 3D flight could not load. Use the flight video or GIF below.';console.error(error);}
})();
