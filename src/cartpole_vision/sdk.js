/* Genuine SDK observations retain source identity and effective timestamps. */
(function(root){
  function series(packet){
    const result={vision:[],encoder:[],fused:[],now:0};
    const last={};
    for(const row of packet.samples||[]){
      result.now=Math.max(result.now,row.host_t);
      for(const sample of row.observation?.samples||[]){
        if(sample.quantity!=='angle'||!['magnetic','vision','fused'].includes(sample.source))continue;
        const name=sample.source==='magnetic'?'encoder':sample.source;
        const t=sample.effective_ns/1e9;
        if(!Number.isFinite(t)||sample.effective_ns==null||t<last[name])continue;
        if(t===last[name])result[name].pop();
        last[name]=t;
        result[name].push({seq:row.sequence,t,theta_rad:sample.validity==='valid'?sample.value:null,source:sample.source,gap:sample.validity!=='valid'});
      }
    }
    return result;
  }
  const api={series};
  if(typeof module!=='undefined'&&module.exports){module.exports=api;return;}
  const status=document.getElementById('sdkStatus'),canvas=document.getElementById('sdkPlot');
  let lastSession='',lastSequence=-1,history=[],lastIdentity='';
  async function poll(){
    try{
      const response=await fetch('/api/sdk-state?'+new URLSearchParams({compact:'1',after:lastSequence,session:lastSession}),{cache:'no-store',signal:AbortSignal.timeout(2000)});
      const data=await response.json(), p=data.packet, d=p?.diagnostics||{};
      status.textContent=!p?'No SDK fusion configured.':
        'SDK shadow: '+data.status+' · '+(d.reason||'')+' · session '+p.session_id+' · '+p.clock_id+' · not commissioned for control';
      const identity=JSON.stringify([p?.session_id,p?.clock_id,d.calibration_id,d.frame_id]);
      if(identity!==lastIdentity){history=[];lastSequence=-1;lastIdentity=identity;}
      for(const row of data.samples||[]){if(row.sequence>lastSequence){history.push(row);lastSequence=row.sequence;}}
      lastSession=p?.session_id||'';
      history=history.filter(row=>!p||p.host_t-row.host_t<=15).slice(-2048);
      const s=series({samples:history});
      AngleComparison.draw(canvas,s.vision,{encoder:s.encoder,fused:s.fused,now:s.now,visionDelay:0},{vision:true,encoder:true,fused:true});
      document.getElementById('sdkDetails').textContent=JSON.stringify(p?{...p,observation:undefined}:data,null,2);
    }catch(error){status.textContent='SDK shadow unavailable: '+error.message;}
    setTimeout(poll,500);
  }
  poll();root.SDKShadow=api;
})(globalThis);
