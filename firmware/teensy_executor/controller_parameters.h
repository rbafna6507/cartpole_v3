#pragma once
#include <stddef.h>
#include <string.h>
#include <math.h>
#include "swing_controller.h"

// Trusted stopped-only operator tuning. Hardware ceilings remain independent
// of these values. No direct gains, Z commands, or encoder calibration fields.
namespace controller_parameters {
struct Field {
 const char* name;
 float swing_control::Parameters::*member;
 float minimum,maximum;
 bool minimum_exclusive;
};
static const Field fields[]={
 {"leff",&swing_control::Parameters::leff,.02f,1.f,false},
 {"pw",&swing_control::Parameters::pw,.1f,30.f,false},
 {"pz",&swing_control::Parameters::pz,.05f,5.f,false},
 {"bal_pw",&swing_control::Parameters::bal_pw,1.f,20.f,false},
 {"pc1",&swing_control::Parameters::pc1,-50.f,-.1f,false},
 {"pc2",&swing_control::Parameters::pc2,-50.f,-.1f,false},
 {"vmax",&swing_control::Parameters::vmax,0.f,cart_hardware::max_speed,true},
 {"vmax_s",&swing_control::Parameters::vmax_s,0.f,cart_hardware::max_speed,true},
 {"vmax_b",&swing_control::Parameters::vmax_b,0.f,cart_hardware::max_speed,true},
 {"approach_v",&swing_control::Parameters::approach_v,0.f,cart_hardware::max_speed,true},
 {"catch_v",&swing_control::Parameters::catch_v,0.f,cart_hardware::max_speed,true},
 {"approach_angle",&swing_control::Parameters::approach_angle,.65f,1.5f,false},
 {"catch_da",&swing_control::Parameters::catch_da,0.f,200.f,true},
 {"amax_s",&swing_control::Parameters::amax_s,0.f,200.f,true},
 {"amax_b",&swing_control::Parameters::amax_b,0.f,200.f,true},
 {"jmax",&swing_control::Parameters::jmax,0.f,10000.f,true},
 {"amax_m",&swing_control::Parameters::amax_m,0.f,200.f,true},
 {"jmax_m",&swing_control::Parameters::jmax_m,0.f,10000.f,true},
 {"ke",&swing_control::Parameters::ke,0.f,100.f,false},
 {"kpx",&swing_control::Parameters::kpx,0.f,1000.f,false},
 {"kdx",&swing_control::Parameters::kdx,0.f,100.f,false},
 {"phase_soft",&swing_control::Parameters::phase_soft,0.f,50.f,true},
 {"catch_a",&swing_control::Parameters::catch_a,.05f,1.2f,false},
 {"catch_r",&swing_control::Parameters::catch_r,.1f,30.f,false},
 {"giveup",&swing_control::Parameters::giveup,.1f,2.5f,false},
 {"spin_trip_rad_s",&swing_control::Parameters::spin_trip_rad_s,0.f,150.f,true},
 {"spin_resume_rad_s",&swing_control::Parameters::spin_resume_rad_s,0.f,150.f,true},
 {"rail",&swing_control::Parameters::rail,.02f,.375f,true},
 {"bw",&swing_control::Parameters::bw,.1f,100.f,false},
 {"vman",&swing_control::Parameters::vman,0.f,cart_hardware::max_speed,false},
 {"manual_speed_max",&swing_control::Parameters::manual_speed_max,0.f,cart_hardware::max_speed,true},
 {"rail_fraction",&swing_control::Parameters::rail_fraction,.5f,.99f,false},
 {"rail_margin",&swing_control::Parameters::rail_margin,.002f,.05f,false},
 {"rail_fault_margin",&swing_control::Parameters::rail_fault_margin,.001f,.049f,false},
 {"rail_hysteresis",&swing_control::Parameters::rail_hysteresis,.001f,.05f,false},
 {"rail_recover_v",&swing_control::Parameters::rail_recover_v,0.f,cart_hardware::max_speed,true},
 {"rail_recover_a",&swing_control::Parameters::rail_recover_a,0.f,200.f,true},
 {"rail_recover_s",&swing_control::Parameters::rail_recover_s,.1f,60.f,false},
 {"rehome_v",&swing_control::Parameters::rehome_v,0.f,cart_hardware::max_speed,true},
 {"rehome_a",&swing_control::Parameters::rehome_a,0.f,200.f,true},
 {"rehome_j",&swing_control::Parameters::rehome_j,0.f,10000.f,true},
};
inline const Field* find(const char* name){
 for(const auto& field:fields)if(!strcmp(field.name,name))return &field;
 return nullptr;
}
inline bool valid(const Field& field,float value,const swing_control::Parameters& current,float rail_cap){
 if(!isfinite(value)||value>field.maximum||
    (field.minimum_exclusive?value<=field.minimum:value<field.minimum))return false;
 if(field.member==&swing_control::Parameters::spin_trip_rad_s && value<=current.spin_resume_rad_s)return false;
 if(field.member==&swing_control::Parameters::spin_resume_rad_s && value>=current.spin_trip_rad_s)return false;
 if(field.member==&swing_control::Parameters::rail && value>rail_cap)return false;
 swing_control::Parameters candidate=current;candidate.*(field.member)=value;
 if(candidate.vman>candidate.manual_speed_max)return false;
 // Tolerance covers float32 subtraction at an exact 1 mm configured gap.
 if(candidate.rail_margin-candidate.rail_fault_margin<.001f-1e-8f)return false;
 if(candidate.rail-candidate.rail_margin-candidate.rail_hysteresis<1.f/cart_hardware::steps_per_m)return false;
 return true;
}
}
