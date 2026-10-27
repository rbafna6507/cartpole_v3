#pragma once
#include <cmath>
#include <cstdint>

namespace encoder_reference {
// Fallback only; firmware loads and can replace the saved upright reference.
// Source: encoder_calibration/125mm_20260917/selected_calibration.json
constexpr int counts_per_turn = 4096;
constexpr int upright_raw = 3953; // default for an uncalibrated board
constexpr float radians_per_count = 6.2831853071795864769f / counts_per_turn;

// Uniform count scale only. The other three poses are diagnostic reference points,
// not a validated nonlinear correction. Retain ENC_INVERT for direction convention.
inline int32_t normalize(int32_t raw) {
  return (raw % counts_per_turn + counts_per_turn) % counts_per_turn;
}
inline int32_t difference(int32_t raw, int32_t reference) {
  int32_t d=normalize(raw-reference);
  return d>counts_per_turn/2?d-counts_per_turn:d;
}
inline int32_t opposite(int32_t raw) { return normalize(raw+counts_per_turn/2); }
inline bool parse_count(const char *text, int32_t &value) {
  if(!text || !*text)return false;
  int32_t n=0;
  for(const char *p=text;*p;++p){
    if(*p<'0' || *p>'9')return false;
    n=n*10+(*p-'0');if(n>=counts_per_turn)return false;
  }
  value=n;return true;
}
// A bounded whole-window span rejects slow drift as well as sudden movement.
// Differences about an anchor keep averages correct across raw 4095 -> 0.
struct StillCapture {
  int count=0, low=0, high=0;
  int32_t anchor=0, sum=0;
  void reset(){count=0;low=high=0;sum=0;}
  bool add(int32_t raw,bool valid){
    if(!valid || raw<0 || raw>=counts_per_turn){reset();return false;}
    if(!count)anchor=raw;
    int d=difference(raw,anchor);
    if(d<low)low=d;if(d>high)high=d;
    if(high-low>4){reset();anchor=raw;d=0;}
    sum+=d;++count;return count>=400;
  }
  int32_t mean()const{return normalize(anchor+(int32_t)lroundf((float)sum/count));}
};
inline float angle_from_upright(int32_t raw, bool invert, int32_t reference=upright_raw) {
  int32_t delta = (raw-reference) % counts_per_turn;
  if (delta > counts_per_turn/2) delta -= counts_per_turn;
  if (delta < -counts_per_turn/2) delta += counts_per_turn;
  return (invert ? -delta : delta) * radians_per_count;
}
}
