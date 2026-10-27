/* Clock intervals, not a claim of synchronized sensor/display clocks. */
(function(root){
  function clockSample(start,end,serverSeconds){
    if(![start,end,serverSeconds].every(Number.isFinite)||end<start)return null;
    return {offset_ms:(start+end)/2-serverSeconds*1000,uncertainty_ms:(end-start)/2,at_ms:end};
  }
  function frameAge(originSeconds,timestampUs,now,sample){
    if(![originSeconds,timestampUs,now].every(Number.isFinite)||!sample||now-sample.at_ms>30000)return null;
    const age=now-(originSeconds*1000+timestampUs/1000+sample.offset_ms);
    return {estimate_ms:Math.max(0,age),uncertainty_ms:sample.uncertainty_ms,
            lower_ms:Math.max(0,age-sample.uncertainty_ms),upper_ms:Math.max(0,age+sample.uncertainty_ms)};
  }
  async function waitDecodeCapacity(decoder,signal,options={}){
    const limit=options.limit??6,timeout=options.timeoutMs??250;
    const now=options.now??(()=>performance.now());
    const yieldTurn=options.yieldTurn??(()=>new Promise(resolve=>setTimeout(resolve,4)));
    const start=now();
    while(true){
      if(signal.aborted||decoder.state==='closed')throw Error('Video aborted');
      if(decoder.decodeQueueSize<limit)return;
      if(now()-start>=timeout)throw Error('Video decoder stalled');
      await yieldTurn();
    }
  }
  const api={clockSample,frameAge,waitDecodeCapacity};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.VisionTiming=api;
})(globalThis);
