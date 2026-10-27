// C ABI over the production firmware headers. No device or transport access.
#include "controller_parameters.h"
#include <new>

struct HostController {
  swing_control::Parameters parameters;
  swing_control::Gains feedback;
  swing_control::PumpState pump;
};
extern "C" {
unsigned cp_abi(){return 1;}
unsigned cp_fields(){return sizeof(controller_parameters::fields)/sizeof(controller_parameters::fields[0]);}
const char* cp_field(unsigned i){return i<cp_fields()?controller_parameters::fields[i].name:nullptr;}
void* cp_create(const float* values,unsigned count,int upright){
  if(!values||count!=cp_fields())return nullptr;
  auto* state=new(std::nothrow) HostController;
  if(!state)return nullptr;
  for(unsigned i=0;i<count;++i){
    if(!isfinite(values[i])){delete state;return nullptr;}
    state->parameters.*(controller_parameters::fields[i].member)=values[i];
  }
  for(const auto& field:controller_parameters::fields){
    if(!controller_parameters::valid(field,state->parameters.*(field.member),state->parameters,state->parameters.rail)){
      delete state;return nullptr;
    }
  }
  state->feedback=upright?swing_control::uprightGains(state->parameters):swing_control::gains(state->parameters);
  return state;
}
void cp_destroy(void* handle){delete static_cast<HostController*>(handle);}
void cp_reset(void* handle){if(handle)static_cast<HostController*>(handle)->pump.reset();}
int cp_gains(void* handle,float* out){
  if(!handle||!out)return 0;
  auto k=static_cast<HostController*>(handle)->feedback;
  out[0]=k.k1;out[1]=k.k2;out[2]=k.k3;out[3]=k.k4;return 1;
}
int cp_advance(void* handle,float q,float w,float x,float v,float a,float dt,int balancing,float* out){
  if(!handle||!out||!isfinite(q)||!isfinite(w)||!isfinite(x)||!isfinite(v)||!isfinite(a)||!isfinite(dt)||dt<=0||dt>.02f)return 0;
  auto& state=*static_cast<HostController*>(handle);
  const auto& p=state.parameters;const auto& k=state.feedback;
  float demand;
  if(balancing){
    demand=swing_control::balance(k,q,w,x,v);
    if(fabsf(q)>p.giveup)balancing=0;
  }else{
    demand=swing_control::swing(p,state.pump,q,w,x,v,dt,a,&k);
    if(swing_control::canCapture(p,k,q,w,x,v,a)){
      balancing=1;demand=swing_control::balance(k,q,w,x,v);
    }
  }
  if(!isfinite(demand))return 0;
  out[0]=demand;out[1]=balancing?1.f:0.f;return 1;
}
float cp_velocity(float v,float a,float target,float cap,float jerk,float dt){
  return cart_motion::settledVelocityAccel(v,a,target,cap,jerk,dt);
}
}
