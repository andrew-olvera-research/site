/* Course picker for the current checkpoint's recorded flights, drawn with Scene3D. */
(async()=>{
 const host=document.getElementById('flight-viewer'), status=document.getElementById('replay-status');
 const fail=error=>{status.textContent='The 3D viewer could not load. Each flight is still available as a video or GIF below.';console.error(error);};
 try {
  const response=await fetch('assets/motion64/manifest.json'); if(!response.ok)throw Error('Flight manifest unavailable');
  const manifest=await response.json(), assets=manifest.assets.filter(a=>a.role==='training');
  const tiles=[...document.querySelectorAll('[data-course]')];
  const viewer=new Scene3D(host); let request=0;
  async function select(index){
   const current=++request, a=assets[index], name=tiles[index].querySelector('b').textContent;
   status.textContent='Loading recorded flight…';
   const data=await Scene3D.fetch('assets/motion64/'+a.trajectory_json); if(current!==request)return;
   data.name=name; viewer.load(data);
   document.getElementById('course-title').textContent=name;
   document.getElementById('course-info').textContent=`${a.gate_count} gates · ${a.duration_seconds.toFixed(2)} s · deterministic policy`;
   tiles.forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.course)===index)));
   status.textContent='';
  }
  tiles.forEach(b=>b.addEventListener('click',()=>select(Number(b.dataset.course)).catch(fail)));
  await select(0);
 } catch(error){fail(error);}
})();
