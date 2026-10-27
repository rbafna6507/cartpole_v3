#pragma once
#include "encoder_reference.h"

namespace encoder_reference {
// Model-qualified capture: 401 native reads span 400 ms at 1 kHz. A fixed
// signed-byte window and nine-bin histograms compute the same circular median
// and first/last-quarter median drift used by the offline calibration policy.
// There is no allocation, sorting, or unbounded work in the control tick.
struct ModelStillCapture {
  static constexpr int required_samples=401;
  int count=0,low=0,high=0;
  int32_t anchor=0;
  int8_t samples[required_samples]{};
  void reset(){count=0;low=high=0;}
  int medianTwice(int start,int length)const{
    int histogram[9]={};
    for(int i=start;i<start+length;++i)++histogram[int(samples[i])+4];
    const int first=(length-1)/2,second=length/2;
    int seen=0,result=0;
    for(int bin=0;bin<9;++bin){
      const int next=seen+histogram[bin];
      if(seen<=first&&first<next)result+=bin-4;
      if(seen<=second&&second<next)result+=bin-4;
      seen=next;
    }
    return result;
  }
  bool add(int32_t raw,bool valid){
    if(!valid||raw<0||raw>=counts_per_turn){reset();return false;}
    if(count>=required_samples)return true;
    if(!count)anchor=raw;
    int d=difference(raw,anchor);
    if(d<low)low=d;
    if(d>high)high=d;
    if(high-low>4){reset();anchor=raw;d=0;}
    samples[count++]=int8_t(d);
    if(count<required_samples)return false;
    constexpr int quarter=required_samples/4;
    const int drift=medianTwice(required_samples-quarter,quarter)-medianTwice(0,quarter);
    if(drift>4||drift< -4){reset();return false;}
    return true;
  }
  int32_t median()const{return normalize(anchor+medianTwice(0,required_samples)/2);}
};
}
