#pragma once
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
// Installed profile: left D26, right D27. Caller passes required inputs only.
// Direction and travel bounds still need physical commissioning.
namespace cart_homing {
constexpr float steps_per_m=26666.6666667f;
constexpr float maximum_rail_half=.375f;
// rail is the governor reference half-span. Its normal/fault boundaries are
// rail-margin and rail-fault_margin. Never place the reference past a measured
// switch contact. Floor the float32 expression once to a whole commanded step.
inline float measuredRailCap(int32_t span_steps,float fraction=.95f,float margin=.015f){
 if(span_steps<10667 || span_steps>20000)return 0;
 const int32_t half=span_steps/2;
 // Volatile products force the same individual float32 roundings as the SDK;
 // otherwise ARM FMA contraction can move floor() by one commanded step.
 const volatile float fraction_steps=float(half)*fraction;
 const volatile float margin_steps=margin*steps_per_m;
 const float requested=fraction_steps+margin_steps;
 return floorf(fminf(float(half),requested))/steps_per_m;
}
struct Machine {
 enum Phase:uint8_t { OFF,SEEK_NEG,VERIFY_NEG,BACK_NEG,LATCH_NEG,VERIFY_LATCH_NEG,CLEAR_NEG,SEEK_POS,VERIFY_POS,BACK_POS,LATCH_POS,VERIFY_LATCH_POS,CLEAR_POS,CENTER,DONE,FAILED };
 Phase phase=OFF;int32_t negative=0,positive=0;
 uint32_t started=0,phase_at=0,stable_at=0;int32_t origin=0,release_pos=0;
 bool released=false;uint8_t chatter=0;
 static constexpr float seek=.03f,slow=.01f;
 // 750 mm maximum search/contact span. 27 s covers a full traverse at
 // 30 mm/s; 75 s covers a near-end start, both relatches and centering.
 static constexpr int32_t clearance=134,max_backoff=534,max_seek=20000,min_span=10667,max_span=20000;
 float safeRail(float saved,float fraction=.95f,float margin=.015f)const{
  return fminf(saved,measuredRailCap(span(),fraction,margin));
 }
 bool active()const{return phase>=SEEK_NEG&&phase<=CENTER;}
 bool done()const{return phase==DONE;}bool failed()const{return phase==FAILED;}
 void cancel(){phase=OFF;}
 void fail(){phase=FAILED;}
 int32_t span()const{return positive-negative;}int32_t center()const{return negative+(positive-negative)/2;}
 void enter(Phase p,uint32_t tick,int32_t pos){phase=p;phase_at=tick;stable_at=tick;origin=pos;released=false;}
 bool start(uint32_t tick,int32_t pos,uint8_t raw){if(raw){fail();return false;}started=tick;chatter=0;negative=positive=0;enter(SEEK_NEG,tick,pos);return true;}
 uint8_t expected()const{switch(phase){case SEEK_NEG:case VERIFY_NEG:case BACK_NEG:case LATCH_NEG:case VERIFY_LATCH_NEG:case CLEAR_NEG:return 1;case SEEK_POS:case VERIFY_POS:case BACK_POS:case LATCH_POS:case VERIFY_LATCH_POS:case CLEAR_POS:return 2;default:return 0;}}
 bool approaching()const{return phase==SEEK_NEG||phase==LATCH_NEG||phase==SEEK_POS||phase==LATCH_POS;}
 // Called by pulse ISR as well as tick: expected raw approach edges stop pulses
 // immediately; only the finite recovery phase permits travel away while HIGH.
 bool inhibit(uint8_t raw)const{return raw && (raw&~expected() || approaching() || phase==VERIFY_NEG||phase==VERIFY_LATCH_NEG||phase==VERIFY_POS||phase==VERIFY_LATCH_POS);}
 float velocity()const{switch(phase){case SEEK_NEG:return -seek;case LATCH_NEG:return -slow;case SEEK_POS:return seek;case LATCH_POS:return slow;case BACK_NEG:case CLEAR_NEG:return seek;case BACK_POS:case CLEAR_POS:return -seek;case CENTER:return 0;default:return 0;}}
 float velocity(int32_t pos)const{if(phase==CENTER)return pos<center()?seek:(pos>center()?-seek:0);return velocity();}
 void update(uint32_t tick,int32_t pos,uint8_t raw){
  if(!active())return;
  const int64_t travel=int64_t(pos)-origin;
  if(uint32_t(tick-started)>=75000||uint32_t(tick-phase_at)>=27000||llabs(travel)>max_seek|| (raw&12)|| ((raw&3)==3) || (raw&~expected())){fail();return;}
  // Clearance after the left latch must not add distance beyond the total
  // allowed contact-to-contact span during the rightward search.
  if(expected()==2 && int64_t(pos)-negative>max_span){fail();return;}
  const bool contact=(raw&expected())!=0;const bool neg=expected()==1;
  switch(phase){
   case SEEK_NEG:case SEEK_POS:case LATCH_NEG:case LATCH_POS:
    if(contact){Phase next=phase==SEEK_NEG?VERIFY_NEG:phase==SEEK_POS?VERIFY_POS:phase==LATCH_NEG?VERIFY_LATCH_NEG:VERIFY_LATCH_POS;enter(next,tick,pos);}break;
   case VERIFY_NEG:case VERIFY_POS:case VERIFY_LATCH_NEG:case VERIFY_LATCH_POS:
    if(!contact){if(++chatter>3){fail();break;}stable_at=tick;return;}
    if(uint32_t(tick-stable_at)>=5){if(phase==VERIFY_LATCH_NEG)negative=pos;if(phase==VERIFY_LATCH_POS)positive=pos;Phase next=phase==VERIFY_NEG?BACK_NEG:phase==VERIFY_POS?BACK_POS:phase==VERIFY_LATCH_NEG?CLEAR_NEG:CLEAR_POS;enter(next,tick,pos);}break;
   case BACK_NEG:case BACK_POS:case CLEAR_NEG:case CLEAR_POS:
    if(llabs(travel)>max_backoff||uint32_t(tick-phase_at)>=1500){fail();break;}
    if(contact){stable_at=tick;if(released){if(++chatter>3){fail();break;}released=false;}break;}
    if(!released && uint32_t(tick-stable_at)>=5){released=true;release_pos=pos;}
    if(released && (neg?int64_t(pos)-release_pos:int64_t(release_pos)-pos)>=clearance){
     if(phase==BACK_NEG)enter(LATCH_NEG,tick,pos);else if(phase==BACK_POS)enter(LATCH_POS,tick,pos);else if(phase==CLEAR_NEG)enter(SEEK_POS,tick,pos);else {if(span()<min_span||span()>max_span){fail();break;}enter(CENTER,tick,pos);}
    }break;
   case CENTER:if(pos==center())phase=DONE;else if((int64_t(pos)-center())*(int64_t(origin)-center())<0){fail();}break;
   default:break;
  }
 }
};
}
