#pragma once
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include "vertical_motion.h"

// One production implementation for the board and native fault tests. No heap,
// peripheral calls, or host-time assumptions. All deadlines use 1 ms board ticks.
namespace executor {
// Installed cart profile: switch 1/D26 left, switch 2/D27 right.
// Uninstalled 3/4 remain raw telemetry only. Z has independent bounded pulses.
constexpr uint8_t REQUIRED_LIMIT_MASK=0x03;
inline uint8_t requiredLimits(uint8_t raw){return raw&REQUIRED_LIMIT_MASK;}
enum Fault : uint8_t { NONE=0,ACTION_EXPIRED=1,HOST_TIMEOUT=2,DURATION=3,ENCODER=4,LIMIT=5,DEADLINE=6,RAIL=7,CONTROLLER=8,HOMING=9 };
enum ReplyKind : uint8_t { SILENT,OK,ERROR,HELLO,STATE,INFO,PARAM,ZSTATE };
struct Reply {ReplyKind kind;const char* text;};
inline bool reached(uint32_t now,uint32_t end){return int32_t(now-end)>=0;}
inline uint16_t crc(const char* p,size_t n){uint16_t c=0xffff;while(n--){c^=uint16_t(uint8_t(*p++))<<8;for(int i=0;i<8;++i)c=c&0x8000?uint16_t((c<<1)^0x1021):uint16_t(c<<1);}return c;}
inline int frame(const char* p,char* out,size_t capacity){const size_t n=strlen(p);if(n>248)return -1;const int length=snprintf(out,capacity,"@%s*%04X\n",p,crc(p,n));return length<0 || size_t(length)>=capacity?-1:length;}
inline bool uintValue(const char* p,uint32_t& value){if(!p || !*p)return false;uint32_t v=0;for(;*p;++p){if(*p<'0'||*p>'9')return false;unsigned d=*p-'0';if(v>(UINT32_MAX-d)/10)return false;v=v*10+d;}value=v;return true;}
inline bool realValue(const char* p,float& value){
 if(!p||!*p)return false;
 // Decimal ASCII only: reject strtof's whitespace, hex, NaN and infinity forms.
 const char* q=p;if(*q=='-'||*q=='+')++q;bool digits=false;
 while(*q>='0'&&*q<='9'){digits=true;++q;}if(*q=='.'){++q;while(*q>='0'&&*q<='9'){digits=true;++q;}}
 if(!digits)return false;
 if(*q=='e'||*q=='E'){++q;if(*q=='-'||*q=='+')++q;bool exponent=false;while(*q>='0'&&*q<='9'){exponent=true;++q;}if(!exponent)return false;}
 if(*q)return false;errno=0;char* end;value=strtof(p,&end);return !*end && errno!=ERANGE && isfinite(value);
}
// Corrected hanging capture accepts a bounded signed integer only. The host
// validates its measured model; firmware independently limits the correction.
inline bool hangingOffset(const char* p,int16_t& value){
 if(!p||!*p)return false;
 const bool negative=*p=='-';if(negative)++p;uint32_t magnitude;
 if(!uintValue(p,magnitude)||magnitude>256)return false;
 value=negative?-int16_t(magnitude):int16_t(magnitude);return true;
}
struct LineBuffer {
 char data[256]{};uint16_t n=0;bool invalid=false;
 bool feed(char ch,const char*& out){out=nullptr;if(ch=='\n'){if(!invalid&&n){data[n]=0;out=data;}n=0;invalid=false;return true;}if(uint8_t(ch)<32||uint8_t(ch)>126){invalid=true;return false;}if(n>=254){invalid=true;return false;}if(!invalid)data[n++]=ch;return false;}
};
struct Core {
 vertical::Motion z;
 uint32_t sid=0,tick=0,acquisition_us=0,last_seq=0;
 uint32_t host_at=0,end_at=0,action_end=0;uint16_t rate=100;
 bool enabled=false,homed=false,healthy=false,auto_run=false;volatile bool switches=true;
 uint32_t healthy_since=0;
 uint8_t limits=0,mode=0,calibration=0;Fault fault=NONE;float demand=0;
 int16_t calibration_offset=0;bool calibration_model=false;
 // Synchronized by the stopped-only parameter transaction in the sketch.
 float acceleration_limit=25.f,manual_speed_limit=.15f;
 char parameter_name[24]{};float parameter_value=0;bool parameter_write=false;
 void stop(){z.cancel();auto_run=false;enabled=false;calibration=0;calibration_offset=0;calibration_model=false;mode=fault==NONE?0:4;demand=0;last_seq=0;action_end=0;}
 void trip(Fault f){if(fault==NONE)fault=f;if(f==ENCODER||f==LIMIT||f==DEADLINE||f==RAIL||f==CONTROLLER||f==HOMING)homed=false;stop();}
 void sample(bool health,uint8_t raw){if(!health||requiredLimits(raw)||!healthy||requiredLimits(limits))healthy_since=tick;healthy=health;limits=raw&15;if(enabled){if(!healthy)trip(ENCODER);else if(switches && requiredLimits(limits) && mode!=13)trip(LIMIT);}}
 void beginReturn(){auto_run=false;mode=14;end_at=tick+15000;demand=0;last_seq=0;action_end=0;}
 void advance(uint32_t us){
  ++tick;acquisition_us=us;if(!enabled)return;
  // A normal auto deadline never overrides a simultaneous safety loss.
  if(fault!=NONE){stop();return;}
  if(!healthy){trip(ENCODER);return;}
  if(mode!=13&&mode!=15&&!homed){trip(HOMING);return;}
  if(switches&&requiredLimits(limits)&&mode!=13){trip(LIMIT);return;}
  if(uint32_t(tick-host_at)>=200){trip(HOST_TIMEOUT);return;}
  if(reached(tick,end_at)){if(auto_run)beginReturn();else trip(DURATION);}
  else if((mode==12||mode==1)&&reached(tick,action_end))trip(ACTION_EXPIRED);
 }
 bool canStart()const{return !enabled && !calibration && mode==0 && fault==NONE && homed && healthy && (!switches||(requiredLimits(limits)==0 && uint32_t(tick-healthy_since)>=5));}
 Reply command(const char* line){
  if(!strcmp(line,"stop")){stop();return {OK,"STOP"};}
  const size_t size=strlen(line);if(size>255||size<7||line[0]!='@')return {ERROR,"frame"};
  const size_t n=size && line[size-1]=='\n'?size-1:size;if(n<7||line[n-5]!='*')return {ERROR,"frame"};
  uint16_t received=0;for(size_t i=n-4;i<n;++i){char ch=line[i];unsigned d;if(ch>='0'&&ch<='9')d=ch-'0';else if(ch>='A'&&ch<='F')d=ch-'A'+10;else if(ch>='a'&&ch<='f')d=ch-'a'+10;else return {ERROR,"crc"};received=uint16_t((received<<4)|d);}
  if(received!=crc(line+1,n-6))return {ERROR,"crc"};
  char payload[249];const size_t length=n-6;if(length>=sizeof payload)return {ERROR,"frame"};memcpy(payload,line+1,length);payload[length]=0;
  for(size_t i=0;i<length;++i)if(uint8_t(payload[i])<32||uint8_t(payload[i])>126)return {ERROR,"ascii"};
  char* tokens[8];unsigned count=0;char* p=payload;
  // Canonical single-space tokens: no blank/trailing/extra fields accepted.
  while(*p){if(count==8||*p==' ')return {ERROR,"args"};tokens[count++]=p;while(*p&&*p!=' ')++p;if(*p){*p++=0;if(!*p)return {ERROR,"args"};}}
  if(count<3||strcmp(tokens[0],"V2"))return {ERROR,"version"};uint32_t session;if(!uintValue(tokens[2],session)||!session)return {ERROR,"session"};const char* cmd=tokens[1];
  if(!strcmp(cmd,"HELLO")){if(count!=3)return {ERROR,"args"};if(enabled)return {ERROR,"enabled"};sid=session;stop();z.refs=z.bottom=0;return {HELLO,"HELLO"};}
  if(session!=sid)return {ERROR,"session"};
  if(!strcmp(cmd,"ZSTATUS")){if(count!=3)return {ERROR,"args"};return {ZSTATE,"ZSTATUS"};}
  if(!strcmp(cmd,"ZLIMIT")){
   float value=0;
   if(count!=4||!realValue(tokens[3],value)||value<10||value>200)return {ERROR,"value"};
   if(enabled||mode!=0||z.phase||calibration||fault!=NONE||!healthy)return {ERROR,"enabled"};
   for(int i=0;i<2;++i)if((z.refs&(1<<i))&&z.count[i]/vertical::STEPS_MM>value)return {ERROR,"travel"};
   z.max_mm=value;return {OK,"ZLIMIT"};
  }
  if(!strcmp(cmd,"ZSTOP")){if(count!=3)return {ERROR,"args"};z.cancel();if(mode==15)stop();return {OK,"ZSTOP"};}
  if(!strcmp(cmd,"ZREF")||!strcmp(cmd,"ZHOME")||!strcmp(cmd,"ZMOVE")||!strcmp(cmd,"ZSINE")){
   const bool ref=!strcmp(cmd,"ZREF"),home=!strcmp(cmd,"ZHOME"),move=!strcmp(cmd,"ZMOVE");
   const unsigned expected=(ref||home)?4:move?6:7;
   if(count!=expected)return {ERROR,"args"};
   uint8_t mask=3;
   if(ref||home||move){mask=!strcmp(tokens[3],"left")?1:!strcmp(tokens[3],"right")?2:!strcmp(tokens[3],"both")?3:0;if(!mask)return {ERROR,"axis"};}
   if(z.phase||calibration||fault!=NONE||!healthy||(switches&&(requiredLimits(limits)||uint32_t(tick-healthy_since)<5)))return {ERROR,"health"};
   if((ref||home)&&(enabled||mode!=0))return {ERROR,"enabled"};
   if(enabled&&mode!=1&&mode!=2&&mode!=3&&mode!=12)return {ERROR,"mode"};
   if(ref){z.reference(mask);return {OK,"ZREF"};}
   if(!home&&(z.refs&mask)!=mask)return {ERROR,"reference"};
   float a=0,b=0,c=0,d=0;
   if(move){
    if(!realValue(tokens[4],a)||!realValue(tokens[5],b)||a==0||fabsf(a)>20||b<=0||b>40||1.875*fabs(double(a))/b>60)return {ERROR,"value"};
    for(int i=0;i<2;++i)if(mask&(1<<i)){double endpoint=z.count[i]/vertical::STEPS_MM+a;if(endpoint<0||endpoint>z.max_mm)return {ERROR,"travel"};}
   }else if(!home){
    vertical::Settings settings;settings.travel=z.max_mm;
    if(!realValue(tokens[3],a)||!realValue(tokens[4],b)||!realValue(tokens[5],c)||!realValue(tokens[6],d)||!vertical::validSine(settings,a,b,c,d)||fabs(z.count[0]-z.count[1])/vertical::STEPS_MM>.25)return {ERROR,"value"};
   }
   z.begin(mask,home?2:move?1:3,tick);
   if(home){z.refs&=~mask;z.duration=5;}
   else if(move){z.duration=std::max(.5,1.875*fabs(double(a))/b);for(int i=0;i<2;++i)z.to[i]=z.from[i]+a;}
   else{z.amplitude=a;z.hz=b;z.center=c;z.seconds=d;z.duration=std::max(vertical::moveDuration(z.from[0],c),vertical::moveDuration(z.from[1],c));z.to[0]=z.to[1]=c;}
   if(!enabled){enabled=true;mode=15;host_at=tick;end_at=tick+uint32_t(ceil((z.duration+(home||move?0:z.seconds))*1000))+1000;}
   return {OK,home?"ZHOME":move?"ZMOVE":"ZSINE"};
  }
  if(!strcmp(cmd,"PING")){if(count!=3)return {ERROR,"args"};host_at=tick;return {SILENT,"PING"};}
  if(!strcmp(cmd,"INFO")){if(count!=3)return {ERROR,"args"};if(enabled)return {ERROR,"enabled"};return {INFO,"INFO"};}
  if(!strcmp(cmd,"STATUS")){if(count!=3)return {ERROR,"args"};return {STATE,"STATUS"};}
  if(!strcmp(cmd,"STOP")){if(count!=3)return {ERROR,"args"};stop();return {OK,"STOP"};}
  if(!strcmp(cmd,"CLEAR")){if(count!=3)return {ERROR,"args"};if(enabled)return {ERROR,"enabled"};if(!healthy||(switches&&(requiredLimits(limits)||uint32_t(tick-healthy_since)<5)))return {ERROR,"health"};fault=NONE;stop();return {OK,"CLEAR"};}
  if(!strcmp(cmd,"RATE")){uint32_t hz;if(count!=4||!uintValue(tokens[3],hz))return {ERROR,"args"};if(hz!=0&&hz!=100&&hz!=250&&hz!=500&&hz!=1000)return {ERROR,"rate"};rate=uint16_t(hz);return {OK,"RATE"};}
  if(!strcmp(cmd,"SWITCHES")){uint32_t on;if(count!=4||!uintValue(tokens[3],on)||on>1)return {ERROR,"args"};if(enabled)return {ERROR,"enabled"};if(on&&!switches)homed=false;switches=on;return {OK,"SWITCHES"};}
  if(!strcmp(cmd,"PARAM")){
   if(count!=4 && count!=5)return {ERROR,"args"};
   const size_t length=strlen(tokens[3]);
   if(!length||length>=sizeof parameter_name||strspn(tokens[3],"abcdefghijklmnopqrstuvwxyz0123456789_")!=length)return {ERROR,"args"};
   float value=0;const bool write=count==5;
   if(write&&!realValue(tokens[4],value))return {ERROR,"value"};
   if(enabled||calibration||(mode!=0&&mode!=4))return {ERROR,"health"};
   if(write&&(mode!=0||fault!=NONE||!healthy))return {ERROR,"health"};
   memcpy(parameter_name,tokens[3],length+1);parameter_value=value;parameter_write=write;
   return {PARAM,"PARAM"};
  }
  if(!strcmp(cmd,"CALIBRATE")){
   if((count!=4&&count!=5) || (strcmp(tokens[3],"upright") && strcmp(tokens[3],"hanging")))return {ERROR,"args"};
   int16_t offset=0;
   if(count==5 && (strcmp(tokens[3],"hanging") || !hangingOffset(tokens[4],offset)))return {ERROR,"args"};
   if(enabled||calibration||mode!=0||fault!=NONE||!healthy)return {ERROR,"health"};
   calibration=!strcmp(tokens[3],"upright")?2:1;calibration_offset=offset;calibration_model=count==5;host_at=tick;
   return {OK,"CALIBRATE"}; // Sketch defers wire ACK until capture and verified save.
  }
  if(!strcmp(cmd,"CENTER")){if(count!=3)return {ERROR,"args"};if(enabled||calibration||mode!=0||fault!=NONE||!healthy)return {ERROR,"health"};if(switches)return {ERROR,"switches"};homed=true;return {OK,"CENTER"};}
  if(!strcmp(cmd,"REHOME")){if(count!=3)return {ERROR,"args"};if(!canStart())return {ERROR,"health"};enabled=true;host_at=tick;beginReturn();return {OK,"REHOME"};}
  if(!strcmp(cmd,"HOME")){if(count!=3)return {ERROR,"args"};if(enabled||calibration||mode!=0||fault!=NONE||!healthy||requiredLimits(limits)||uint32_t(tick-healthy_since)<5)return {ERROR,"health"};if(!switches)return {ERROR,"switches"};homed=false;enabled=true;mode=13;host_at=tick;end_at=tick+75000;return {OK,"HOME"};}
  if(!strcmp(cmd,"ARM")||!strcmp(cmd,"MODE")){
   const bool arm=!strcmp(cmd,"ARM");uint32_t duration;
   if(count!=(arm?4u:5u)||!uintValue(tokens[arm?3:4],duration)||duration<1||duration>60000)return {ERROR,"duration"};
   uint8_t desired=12;if(!arm){if(!strcmp(tokens[3],"auto"))desired=2;else if(!strcmp(tokens[3],"bal"))desired=3;else if(!strcmp(tokens[3],"manual"))desired=1;else return {ERROR,"mode"};}
   if(!canStart())return {ERROR,"health"};auto_run=desired==2;enabled=true;mode=desired;host_at=tick;end_at=tick+duration;action_end=tick+20;demand=0;last_seq=0;return {OK,arm?"ARM":"MODE"};
  }
  if(!strcmp(cmd,"A")||!strcmp(cmd,"V")){
   const bool accel=!strcmp(cmd,"A");uint32_t seq,observed,ttl;float value;
   if(count!=7||!uintValue(tokens[3],seq)||!uintValue(tokens[4],observed)||!uintValue(tokens[5],ttl)||!realValue(tokens[6],value))return {ERROR,"args"};
   if(!enabled||mode!=(accel?12:1))return {ERROR,"mode"};if(!seq||seq<=last_seq)return {ERROR,"sequence"};
   if(ttl<1||ttl>50||int32_t(tick-observed)<0||uint32_t(tick-observed)>20||reached(tick,observed+ttl))return {ERROR,"stale"};if(fabsf(value)>(accel?acceleration_limit:manual_speed_limit))return {ERROR,"value"};
   last_seq=seq;action_end=observed+ttl;demand=value;return {SILENT,accel?"A":"V"};
  }
  return {ERROR,"command"};
 }
};
}
