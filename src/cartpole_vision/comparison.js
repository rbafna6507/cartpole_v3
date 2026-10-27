/* Native sensor histories. Visibility changes rendering only, never estimation. */
(function(root){
 const wrap=x=>Math.atan2(Math.sin(x),Math.cos(x));
 class ComparisonStore{
  constructor(){this.epoch=null;this.after=0;this.encoder=[];this.fused=[];this.now=null;this.anchor={};}
  merge(p){
   if(!p)return;
   if(p.reset||p.epoch!==this.epoch){this.encoder=[];this.fused=[];this.anchor={};this.after=0;}
   this.epoch=p.epoch;this.now=p.now;this.visionDelay=(p.status?.vision_delay_ms||0)/1000;
   for(const name of ['encoder','fused']){
    for(const row of p[name]?.samples||[]){
     if(row.seq<=this.after||!Number.isFinite(row.t))continue;
     const r={...row},a=this.anchor[name];
     if(Number.isFinite(r.theta_rad)){
      r.plot_theta_rad=a&&r.t-a.t>=0&&r.t-a.t<=.3?a.plot_theta_rad+wrap(r.theta_rad-a.theta_rad):r.theta_rad;
      this.anchor[name]=r;
     }
     this[name].push(r);
    }
   }
   this.after=p.last_seq;
   const end=Number.isFinite(p.now)?p.now:Math.max(0,...this.encoder.map(r=>r.t),...this.fused.map(r=>r.t));
   this.trim(end);
  }
  trim(now){for(const name of ['encoder','fused'])this[name]=this[name].filter(r=>now-r.t<=15).slice(-4096);}
 }
 class Visibility{
  constructor(storage){
   this.storage=storage;this.enabled={vision:true,encoder:true,fused:true};
   try{const p=JSON.parse(storage?.getItem('vision3-angle-traces'));for(const k of Object.keys(this.enabled))if(typeof p?.[k]==='boolean')this.enabled[k]=p[k];}catch(e){}
  }
  set(key,value){
   if(!(key in this.enabled))return;
   this.enabled[key]=!!value;
   try{this.storage?.setItem('vision3-angle-traces',JSON.stringify(this.enabled));}catch(e){}
  }
 }
 function angleLayers(vision,store,enabled){
  const groups=[{name:'vision',color:'#225db0',rows:vision.map(r=>({...r,t:r.t-(store.visionDelay||0)}))},{name:'encoder',color:'#c56a12',rows:store.encoder},{name:'fused',color:'#8549b6',rows:store.fused}];
  const enc=store.encoder.filter(p=>Number.isFinite(p.theta_rad));
  let index=0;
  function reference(t){
   if(!enc.length)return null;
   while(index+1<enc.length&&enc[index+1].t<=t)index++;
   const a=enc[index],b=enc[index+1];
   if(a.t<=t&&b&&b.t-a.t>0&&b.t-a.t<=.1)
    return a.plot_theta_rad+(b.plot_theta_rad-a.plot_theta_rad)*(t-a.t)/(b.t-a.t);
   return Math.abs(a.t-t)<=.05?a.plot_theta_rad:null;
  }
  return groups.map(group=>{
   index=0;const points=[],segments=[];let prev=null,missing=false;
   for(const row of group.rows){
    if(!Number.isFinite(row.theta_rad)){missing=true;continue;}
    const dt=prev?row.t-prev.t:Infinity;
    const delta=prev?wrap(row.theta_rad-prev.raw):0;
    let v=Number.isFinite(row.plot_theta_rad)?row.plot_theta_rad:prev&&dt>0&&dt<=.3?prev.rad+delta:row.theta_rad;
    const ref=reference(row.t);
    // Choose the same 360-degree display branch; retain the measured offset.
    if(group.name!=='encoder'&&ref!==null)v=ref+wrap(v-ref);
    const point={t:row.t,v:v*180/Math.PI,source:row.source};
    if(prev&&dt>0&&dt<=.3&&Math.abs(v-prev.rad)<=Math.PI)
     segments.push({a:prev.point,b:point,dashed:missing||!!row.gap||(group.name==='vision'&&row.seq!==prev.seq+1)||dt>.15||row.source==='prediction'});
    points.push(point);prev={seq:row.seq,t:row.t,rad:v,raw:row.theta_rad,point};missing=false;
   }
   return {...group,visible:!!enabled[group.name],points,segments};
  });
 }
 function draw(canvas,vision,store,enabled){
  const width=canvas.clientWidth;if(width<=0)return;
  const dpr=root.devicePixelRatio||1,height=200;
  canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);
  const g=canvas.getContext('2d');g.scale(dpr,dpr);g.clearRect(0,0,width,height);
  const layers=angleLayers(vision,store,enabled);
  const end=Number.isFinite(store.now)?store.now:Math.max(0,...vision.map(r=>r.t),...store.encoder.map(r=>r.t));
  const points=layers.filter(l=>l.visible).flatMap(l=>l.points.filter(p=>end-p.t<=12));
  let low=-180,high=180;
  if(points.length){const values=points.map(p=>p.v),lo=Math.min(...values),hi=Math.max(...values),pad=Math.max(8,(hi-lo)*.12);low=Math.floor((lo-pad)/45)*45;high=Math.ceil((hi+pad)/45)*45;}
  const left=55,right=width-10,top=34,bottom=168;
  const x=t=>left+(1-(end-t)/12)*(right-left),y=v=>bottom-(v-low)/(high-low)*(bottom-top);
  g.font='12px Avenir, sans-serif';g.fillStyle='#536877';g.fillText('Angle θ · degrees from upright',0,14);g.strokeStyle='#dbe3e9';
  for(const value of [low,(low+high)/2,high]){g.beginPath();g.moveTo(left,y(value));g.lineTo(right,y(value));g.stroke();g.fillText((value>0?'+':'')+value.toFixed(0),0,y(value)+4);}
  g.fillText('12 s ago',left,190);g.fillText('6 s',x(end-6)-8,190);g.fillText('now',right-25,190);
  if(!points.length){g.fillText('Enable a trace; waiting for calibrated vision or console telemetry.',left+8,98);return;}
  g.save();g.beginPath();g.rect(left,top,right-left,bottom-top);g.clip();
  for(const layer of layers.filter(l=>l.visible)){
   g.strokeStyle=layer.color;g.fillStyle=layer.color;g.lineWidth=layer.name==='fused'?2.3:1.5;
   for(const dashed of [false,true]){
    g.setLineDash(dashed?[4,4]:[]);g.beginPath();
    for(const s of layer.segments){if(s.dashed!==dashed||end-s.b.t>12)continue;g.moveTo(x(s.a.t),y(s.a.v));g.lineTo(x(s.b.t),y(s.b.v));}g.stroke();
   }
   g.setLineDash([]);
   for(const p of layer.points){if(end-p.t>12)continue;g.beginPath();g.arc(x(p.t),y(p.v),1.2,0,Math.PI*2);g.fill();}
  }
  g.restore();
 }
 const api={ComparisonStore,Visibility,angleLayers,draw};
 if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.AngleComparison=api;
})(globalThis);
