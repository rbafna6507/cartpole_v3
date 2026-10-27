#pragma once
#include "vertical_profile.h"
#include <stdint.h>
namespace vertical {
constexpr double STEPS_MM=640.;
struct Motion {
 double max_mm=120;
 uint8_t phase=0,refs=0,bottom=0,selected=0;
 uint32_t started=0;double duration=0,amplitude=0,hz=0,center=0,seconds=0;
 double from[2]={0,0},to[2]={0,0};int32_t count[2]={0,0};
 void cancel(){if(phase){refs&=~selected;bottom&=~selected;}phase=0;selected=0;}
 void reference(uint8_t mask){refs|=mask;bottom|=mask;for(int i=0;i<2;++i)if(mask&(1<<i))count[i]=0;}
 void begin(uint8_t mask,uint8_t kind,uint32_t tick){selected=mask;phase=kind;started=tick;bottom&=~mask;for(int i=0;i<2;++i)from[i]=count[i]/STEPS_MM;}
 double position(int i,uint32_t tick)const {
  const double t=uint32_t(tick-started)*.001;
  if(phase==2)return from[i]-40*std::min(t,5.);
  if(phase==4)return sinePosition(t,amplitude,hz,center,seconds);
  return movePosition(t/duration,from[i],to[i]);
 }
};
}
