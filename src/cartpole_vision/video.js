/* Length-prefixed Annex-B access units, decoded continuously with WebCodecs. */
(() => {
  const canvas=document.getElementById('video'), context=canvas.getContext('2d',{alpha:false});
  const label=document.getElementById('previewLabel'), received=document.getElementById('videoRate');
  const painted=document.getElementById('paintRate');
  let controller, decoder, latest=null, decoded=0, draws=0, total=0, lastPaint=0, lastFrameAt=0;
  let lastTick=performance.now(), reconnectTimer;
  let clock=null, clockProbes=[], canvasAge=null;
  async function probeClock(){
    if(!document.hidden){
      try {
        const start=performance.now();
        const response=await fetch('/api/clock',{cache:'no-store',signal:AbortSignal.timeout(2000)});
        const data=await response.json(),end=performance.now();
        const sample=VisionTiming.clockSample(start,end,data.monotonic_s);
        clockProbes=clockProbes.filter(s=>end-s.at_ms<15000);
        if(sample)clockProbes.push(sample);
        clock=clockProbes.reduce((best,s)=>!best||s.uncertainty_ms<best.uncertainty_ms?s:best,null);
      }catch(error){/* Missing probes leave the latency estimate unavailable. */}
    }
    setTimeout(probeClock,2000);
  }
  probeClock();
  function paint() {
    if(latest){
      if(canvas.width!==latest.displayWidth || canvas.height!==latest.displayHeight){canvas.width=latest.displayWidth;canvas.height=latest.displayHeight;}
      context.drawImage(latest,0,0,canvas.width,canvas.height);
      canvasAge=VisionTiming.frameAge(lastState?.video?.source_origin_s,latest.timestamp,performance.now(),clock);
      latest.close();latest=null;draws++;lastPaint=performance.now();
    }
    requestAnimationFrame(paint);
  }
  requestAnimationFrame(paint);
  setInterval(()=>{
    const now=performance.now(), dt=(now-lastTick)/1000;
    received.textContent=(decoded/dt).toFixed(1)+' FPS';painted.textContent=(draws/dt).toFixed(1)+' FPS';
    canvas.dataset.decodedFrames=String(total);canvas.dataset.lastFrameAt=String(lastFrameAt);
    canvas.dataset.decodeFps=(decoded/dt).toFixed(2);canvas.dataset.paintFps=(draws/dt).toFixed(2);
    document.getElementById('canvasAge').textContent=canvasAge&&now-lastPaint<2000?
      '≈ '+canvasAge.estimate_ms.toFixed(0)+' ± '+canvasAge.uncertainty_ms.toFixed(0)+' ms':'—';
    decoded=draws=0;lastTick=now;
    if(now-lastFrameAt>2000){context.clearRect(0,0,canvas.width,canvas.height);if(!document.hidden)label.textContent='Video disconnected — reconnecting';}
  },1000);
  function disconnect(){
    clearTimeout(reconnectTimer);controller?.abort();
    if(decoder && decoder.state!=='closed')decoder.close();
    if(latest){latest.close();latest=null;}canvasAge=null;
  }
  async function connect(){
    disconnect();if(document.hidden)return;
    const active=new AbortController();controller=active;
    let localDecoder;
    try {
      if(!window.VideoDecoder)throw Error('Open this page in Chrome: WebCodecs is required');
      localDecoder=new VideoDecoder({output:frame=>{
        if(active.signal.aborted){frame.close();return;}
        if(latest)latest.close();latest=frame;decoded++;total++;lastFrameAt=performance.now();
        const mode=document.getElementById('videoMode').value;
        label.textContent=mode==='camera'?'Smooth camera · tracking measurements at right':mode==='mask'?'Green mask · processed frames':'Tracking overlay · processed frames';
      },error:error=>{label.textContent=error.message;active.abort();}});
      decoder=localDecoder;
      localDecoder.configure({codec:'avc1.42E028',optimizeForLatency:true});
      const response=await fetch('/video.h264',{signal:active.signal,cache:'no-store'});
      if(!response.ok)throw Error('Video encoder unavailable');
      const reader=response.body.getReader();let pending=new Uint8Array(0), waitKey=true;
      let newestTime=0, startTime=null, arrivalStart=0;
      const watchdog=setInterval(()=>{if(performance.now()-lastFrameAt>5000)active.abort();},1000);
      lastFrameAt=performance.now();
      try {
        while(!active.signal.aborted){
          const {value,done}=await reader.read();if(done)break;
          const joined=new Uint8Array(pending.length+value.length);joined.set(pending);joined.set(value,pending.length);pending=joined;
          let offset=0;
          while(pending.length-offset>=13){
            const header=new DataView(pending.buffer,pending.byteOffset+offset,13),size=header.getUint32(0);
            if(size<1 || size>8388608)throw Error('Invalid video packet');
            if(pending.length-offset<13+size)break;
            const timestamp=Number(header.getBigUint64(4)),key=header.getUint8(12)!==0;
            if(startTime===null){startTime=timestamp;arrivalStart=performance.now();}
            newestTime=timestamp;
            // A network chunk can contain several frames. Let asynchronous decoding
            // run before declaring overload; never discard valid queued deltas.
            await VisionTiming.waitDecodeCapacity(localDecoder,active.signal);
            if(!waitKey || key){
              waitKey=false;
              localDecoder.decode(new EncodedVideoChunk({type:key?'key':'delta',timestamp,data:pending.slice(offset+13,offset+13+size)}));
            }
            offset+=13+size;
          }
          pending=pending.slice(offset);
          // If transport accumulates over one second, reconnect at a fresh keyframe.
          if(startTime!==null && performance.now()-arrivalStart-(newestTime-startTime)/1000>1000)throw Error('Catching up to live video');
        }
      } finally {clearInterval(watchdog);await reader.cancel().catch(()=>{});}
    } catch(error) {
      if(!active.signal.aborted){label.textContent=error.message;canvas.dataset.lastVideoError=error.message;}
    } finally {
      if(localDecoder && localDecoder.state!=='closed')localDecoder.close();
      if(controller===active && !document.hidden)reconnectTimer=setTimeout(connect,750);
    }
  }
  document.getElementById('videoMode').onchange=async event=>{
    const next=event.target.value;
    try {const response=await fetch('/api/video-mode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:next})});
      if(!response.ok)throw Error('Could not switch video view');connect();
    }catch(error){label.textContent=error.message;}
  };
  document.addEventListener('visibilitychange',()=>document.hidden?disconnect():connect());
  window.addEventListener('pagehide',disconnect);
  connect();
})();
