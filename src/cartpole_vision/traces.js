/* Full measurement batches and display-only connections between real samples. */
(function(root){
  const maxBridge=.25;
  const angleDelta=(a,b)=>Math.atan2(Math.sin(a-b),Math.cos(a-b));
  const joinsAngle=(dt,delta,limit=maxBridge)=>dt>0&&dt<=limit&&Math.abs(delta)<=Math.PI/2;
  class HistoryStore {
    constructor(){this.epoch=null;this.after=0;this.rows=[];this.angleAnchor=null;}
    merge(packet){
      if(!packet||!Array.isArray(packet.samples))return;
      if(packet.reset||packet.epoch!==this.epoch){this.rows=[];this.after=0;this.angleAnchor=null;}
      this.epoch=packet.epoch;
      for(const row of packet.samples){
        if(!Number.isFinite(row.t)||!Number.isInteger(row.seq)||row.seq<=this.after)continue;
        const sample={...row};
        if(Number.isFinite(row.theta_rad)){
          const prev=this.angleAnchor,delta=prev?angleDelta(row.theta_rad,prev.theta_rad):0;
          // Retain this display branch when older samples leave the rolling buffer.
          sample.plot_theta_rad=prev&&joinsAngle(row.t-prev.t,delta)?prev.plot_theta_rad+delta:row.theta_rad;
          this.angleAnchor=sample;
        }
        this.rows.push(sample);this.after=row.seq;
      }
      const end=this.rows.length?this.rows[this.rows.length-1].t:0;
      this.rows=this.rows.filter(row=>end-row.t<=15).slice(-4096);
    }
  }

  function buildSeries(rows,key,{angular=false,scale=1,maxBridge=.25}={}){
    const points=[],segments=[];
    let previous=null,missing=false;
    for(const row of rows){
      const raw=row[key];
      if(!Number.isFinite(raw)){missing=true;continue;}
      let value=raw,connected=false,bridged=false;
      if(previous){
        const dt=row.t-previous.t;
        const delta=angular?angleDelta(raw,previous.raw):raw-previous.raw;
        bridged=missing||row.seq!==previous.seq+1||dt>.15;
        // Long outages and ambiguous motion don't imply a known path or turn count.
        connected=angular?joinsAngle(dt,delta,maxBridge):dt>0&&dt<=maxBridge;
        if(angular&&connected)value=previous.value+delta;
      }
      if(angular&&key==='theta_rad'&&Number.isFinite(row.plot_theta_rad))value=row.plot_theta_rad;
      const point={seq:row.seq,t:row.t,v:value*scale};
      if(connected)segments.push({a:previous.point,b:point,bridged});
      points.push(point);
      previous={seq:row.seq,t:row.t,raw,value,point};missing=false;
    }
    return {points,segments};
  }

  function draw(canvas,rows,key,{label,color,angular=false,scale=1,emptyMessage='No valid measurements yet'}={}){
    const dpr=root.devicePixelRatio||1,w=canvas.clientWidth,h=180;
    if(w<=0)return;
    canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);
    const g=canvas.getContext('2d');g.scale(dpr,dpr);g.clearRect(0,0,w,h);
    g.font='12px Avenir, sans-serif';g.fillStyle='#536877';g.fillText(label,0,14);
    const end=rows.length?rows[rows.length-1].t:0;
    const series=buildSeries(rows,key,{angular,scale});
    const visible=series.points.filter(p=>end-p.t<=12);
    let low=angular?-180:-12,high=angular?180:12;
    if(visible.length){
      const values=visible.map(p=>p.v),lo=Math.min(...values),hi=Math.max(...values);
      const pad=Math.max(angular?8:2,(hi-lo)*.12),step=angular?45:5;
      low=Math.floor((lo-pad)/step)*step;high=Math.ceil((hi+pad)/step)*step;
    }
    const left=55,right=w-8,top=34,bottom=146;
    const x=t=>left+(1-(end-t)/12)*(right-left);
    const y=v=>bottom-(v-low)/(high-low)*(bottom-top);
    g.strokeStyle='#dbe3e9';
    for(const value of [high,(low+high)/2,low]){
      const py=y(value);g.beginPath();g.moveTo(left,py);g.lineTo(right,py);g.stroke();
      g.fillText((value>0?'+':'')+value.toFixed(0),0,py+4);
    }
    g.fillText('12 s ago',left,168);g.fillText('6 s',x(end-6)-8,168);g.fillText('now',right-25,168);
    if(!visible.length){g.fillText(emptyMessage,left+12,91);return;}
    g.save();g.beginPath();g.rect(left,top,right-left,bottom-top);g.clip();
    for(const bridged of [false,true]){
      g.strokeStyle=bridged?'#ad7419':color;g.lineWidth=bridged?1.4:1.8;
      g.setLineDash(bridged?[4,4]:[]);g.beginPath();
      for(const segment of series.segments){
        if(segment.bridged!==bridged||end-segment.b.t>12)continue;
        g.moveTo(x(segment.a.t),y(segment.a.v));g.lineTo(x(segment.b.t),y(segment.b.v));
      }
      g.stroke();
    }
    g.setLineDash([]);g.fillStyle=color;
    // A single valid measurement must remain visible between two missing frames.
    for(const p of visible){g.beginPath();g.arc(x(p.t),y(p.v),1.35,0,Math.PI*2);g.fill();}
    g.restore();
  }
  const api={HistoryStore,buildSeries,draw};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.VisionTraces=api;
})(globalThis);
