#pragma once
#include <cmath>
#include <algorithm>

namespace vertical {
constexpr double PROFILE_PI=3.14159265358979323846;
constexpr double MAX_RATE=100000; // pulses/s; below the 4 us timer's 125 kHz ceiling
constexpr double CLEARANCE=2;    // mm from either declared endpoint
constexpr double RAMP=2;         // sine amplitude ramp, seconds at each end
struct Settings {
    double travel=0, lead=5;
    int motorSteps=200, microsteps=16;
    bool bottomRated=false;
    double stepsPerMm() const { return motorSteps*microsteps/lead; }
    bool valid() const {
        return std::isfinite(travel) && travel>=10 && travel<=500 &&
               std::isfinite(lead) && lead>=1 && lead<=20 &&
               (motorSteps==200 || motorSteps==400) &&
               (microsteps==1 || microsteps==2 || microsteps==4 || microsteps==8 || microsteps==16);
    }
};
inline bool validBottom(const Settings&s,double speed,double seconds) {
    return s.valid() && s.bottomRated && std::isfinite(speed) && std::isfinite(seconds) &&
           speed>0 && speed<=40 && seconds>0 && seconds<=5 && speed*s.stepsPerMm()<=MAX_RATE;
}
inline bool validSine(const Settings&s,double amplitude,double hz,double center,double seconds) {
    // Envelope derivative is <= pi/(2*RAMP). Include it in the speed bound.
    return s.valid() && std::isfinite(amplitude) && std::isfinite(hz) &&
           std::isfinite(center) && std::isfinite(seconds) && amplitude>0 && hz>0 &&
           hz<=5 && seconds>=2*RAMP && seconds<=60 &&
           center-amplitude>=CLEARANCE && center+amplitude<=s.travel-CLEARANCE &&
           amplitude*(2*PROFILE_PI*hz+PROFILE_PI/(2*RAMP))*s.stepsPerMm()<=MAX_RATE;
}
inline double sinePosition(double t,double amplitude,double hz,double center,double seconds) {
    if(t<=0 || t>=seconds)return center;
    double r=std::min(1.,std::min(t,seconds-t)/RAMP);
    double envelope=.5-.5*std::cos(PROFILE_PI*r);
    return center+amplitude*envelope*std::sin(2*PROFILE_PI*hz*t);
}
inline double moveDuration(double from,double to) {
    // Quintic smoothstep peak derivative 1.875; cap transfer speed at 10 mm/s.
    return std::max(.5,1.875*std::abs(to-from)/10.);
}
inline double movePosition(double fraction,double from,double to) {
    double u=std::max(0.,std::min(1.,fraction));
    return from+(to-from)*(u*u*u*(10+u*(-15+6*u)));
}
}
