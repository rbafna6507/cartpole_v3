// Teensy 4.1 bounded V2 executor; controller math copied from preserved v33.
// Only framed V2 commands and the literal fallback stop are accepted.
#include <Arduino.h>
#include <Wire.h>
#include <IntervalTimer.h>
#include "teensy_reference_store.h"
#if !defined(ARDUINO_TEENSY41)
#error "Select Teensy 4.1 with USB Type Serial"
#endif
#include "cart_motion.h"
#include "rail_recovery.h"
#include "spin_recovery.h"
#include "upright_session.h"
#include "swing_controller.h"
#include "controller_parameters.h"
#include "encoder_reference.h"
#include "encoder_fault_diagnostics.h"
#include "model_encoder_reference.h"
#include "serial_frame.h"
#include "executor_core.h"
#include "cart_homing.h"
#include "saved_center_return.h"
#include <cstdarg>
#include "status_display.h"


// ============================== PINS ========================================
#define PIN_CART_STEP   2
#define PIN_CART_DIR    3
#define PIN_Z1_STEP     4
#define PIN_Z1_DIR      5
#define PIN_Z2_STEP     6
#define PIN_Z2_DIR      7
#define PIN_EN_ALL      8        // shared ENABLE, active LOW
#define PIN_SDA         17
#define PIN_SCL         16
#define I2C_HZ          400000    // drop to 100000 if the cable to the cart is
                                  // long or you see enc_err climbing

#define CART_INVERT     0         // v19: reverse cart polarity for upright trial; encoder sign unchanged
#define Z_INVERT        0         // flip if 'zv 5' lowers instead of raises
#define ENC_INVERT      0         // flip if theta goes negative when the pole
                                  // leans toward +x

// Legacy controller diagnostics stay local; USB contains only V2 frames.
static uint32_t dropped_serial_frames=0;
static uint32_t control_deadline_misses=0;
void checkedPrintln(const char*){}
void checkedPrintln(const __FlashStringHelper*){}
void checkedPrintf(const char*, ...){}
executor::Core executor_state;
static encoder_diagnostics::Snapshot encoder_fault;
static encoder_diagnostics::Transaction encoder_tick_failure;
cart_homing::Machine homing;
volatile bool limit_fault_pending=false;
volatile bool homing_active=false,homing_approach=false,homing_verify=false,homing_centering=false;
volatile uint8_t homing_expected=0,homing_edge_pending=0;
volatile int32_t homing_center_target=0;
static uint8_t rawLimits(){return (digitalReadFast(26)?1:0)|(digitalReadFast(27)?2:0)|(digitalReadFast(28)?4:0)|(digitalReadFast(29)?8:0);}
// ========================== MECHANICS =======================================
// Big Easy Driver uses 1/16 with all MS pins HIGH (also its unconnected default).
// On the 60T GT2 pulley this is 26.667 steps/mm.
// This port actively drives MS1/2/3 HIGH. Do not ground those GPIO outputs.
// Driver MS pins remain unconnected; retain the BED default 1/16 mechanics.
#define CART_MICROSTEPS cart_hardware::microsteps        // BED onboard pull-ups -> 1/16
#define Z_MICROSTEPS    16        // same
#define MOTOR_STEPS_REV cart_hardware::motor_steps       // 1.8 deg
#define PULLEY_TEETH    cart_hardware::pulley_teeth        // measured hardware: GT2 60T = 120 mm/rev
#define BELT_PITCH_MM   cart_hardware::belt_pitch_mm
#define Z_LEAD_MM       5.0f      // SFU1605

static const float CART_STEPS_PER_M =
    cart_hardware::steps_per_m;
static const float Z_STEPS_PER_MM =
    (MOTOR_STEPS_REV * Z_MICROSTEPS) / Z_LEAD_MM;

#define GRAV            9.81f
#define CTRL_HZ         1000
// A 7 us timer period gives 142857.14 ticks/s and 71428.57 steps/s.
// At 60T / 1:16, 0.8 m/s needs 21333.333 steps/s; DDS uses this same period.
#define ISR_HZ          cart_hardware::isr_hz
static const float DT = 1.0f / CTRL_HZ;

// Hard ceiling on cart speed, derived rather than guessed. Ask for more than
// this and the pulse generator clamps while the controller keeps believing the
// number it was given - so we clamp loudly at the point of entry instead.
static const float VMAX_CEIL = cart_hardware::max_speed;

#define MANUAL_TIMEOUT_MS 250     // 'v' command watchdog: no news = stop

// ===================== LIVE-TUNABLE PARAMETERS ==============================
// Saved defaults load at boot; bounded stopped-only operator tuning is volatile.
// Current arm: 125 mm / 11.05 g plus two 5.85 g end weights.
// L_eff=0.124 m from 20 low-angle cycles in the pre-175 mm captures.
swing_control::Parameters controller;
float &p_leff = controller.leff;
float &p_pw = controller.pw;
float &p_pz = controller.pz;
float &p_pc1 = controller.pc1;
float &p_pc2 = controller.pc2;
float &p_vmax = controller.vmax;
float &p_vmax_s = controller.vmax_s;
float &p_vmax_b = controller.vmax_b;
float &p_amax_s = controller.amax_s;
float &p_amax_b = controller.amax_b;
float &p_jmax = controller.jmax;
float &p_amax_m = controller.amax_m;
float &p_jmax_m = controller.jmax_m;
float &p_ke = controller.ke;
float &p_kpx = controller.kpx;
float &p_kdx = controller.kdx;
float &p_phase_soft = controller.phase_soft;
float &p_catch_a = controller.catch_a;
float &p_catch_r = controller.catch_r;
float &p_giveup = controller.giveup;
float &p_rail = controller.rail;
float &p_bw = controller.bw;
float &p_vman = controller.vman;
float p_zvmax   = 2.0f;    // mm/s
float p_zamax   = 20.0f;   // mm/s^2
float p_ztravel = 100.0f;  // mm of usable screw travel above z=0

// state-feedback gains, computed from the poles above: a = -(K.[th,thd,x,v])
float K1, K2, K3, K4;

// ==================== STEP PULSE GENERATOR (3 axes) =========================
// Same DDS rates and 7 us tick as the working ESP32 firmware.
enum { AX_CART = 0, AX_Z1, AX_Z2, N_AX };
volatile bool g_energized=false;
volatile uint32_t ax_inc[N_AX]={0,0,0};
volatile int32_t ax_pos[N_AX]={0,0,0};
volatile int8_t ax_dir[N_AX]={1,1,1};
static uint32_t ax_acc[N_AX]={0,0,0};
static const uint8_t DIR_PIN[N_AX]={PIN_CART_DIR,PIN_Z1_DIR,PIN_Z2_DIR};
static bool direction_initialized[N_AX]={false,false,false};
IntervalTimer step_timer,z_timer;
static bool z_timer_ready=false;
volatile int32_t z_count[2]={0,0},z_target[2]={0,0};
volatile uint8_t z_active=0;
volatile uint32_t z_increment[2]={0,0};
static uint32_t z_accumulator[2]={0,0};
static bool z_high[2]={false,false};
static int8_t z_direction[2]={0,0};
static bool step_timer_ready=false;
volatile uint32_t control_heartbeat_us=0;
volatile bool timing_fault_pending=false;
constexpr uint32_t CONTROL_STALL_US=5000;

static inline uint32_t interruptMask(){
  uint32_t mask;asm volatile("mrs %0, primask" : "=r"(mask) :: "memory");return mask;
}
void clearStepPins(){
  digitalWriteFast(PIN_CART_STEP,LOW);
  digitalWriteFast(PIN_Z1_STEP,LOW);
  digitalWriteFast(PIN_Z2_STEP,LOW);
}
void onStepTimer(){
  digitalWriteFast(PIN_CART_STEP,LOW);
  if(!g_energized)return;
  // Independent pulse-side timeout: a stalled I2C/foreground cannot keep moving.
  if(uint32_t(micros()-control_heartbeat_us)>CONTROL_STALL_US){
    digitalWriteFast(PIN_EN_ALL,HIGH);
    g_energized=false;
    for(int i=0;i<N_AX;++i)ax_inc[i]=0;
    timing_fault_pending=true;
    return;
  }
  // Sample raw NC channels on every 7 us pulse interrupt. Normal trips raise
  // the shared EN; expected homing edges inhibit cart pulses before debounce.
  const uint8_t raw=executor_state.switches?executor::requiredLimits(rawLimits()):0;
  if(raw){
    if(!homing_active || (raw & ~homing_expected)){
      digitalWriteFast(PIN_EN_ALL,HIGH);g_energized=false;
      for(int i=0;i<N_AX;++i)ax_inc[i]=0;
      limit_fault_pending=true;return;
    }
    if(homing_approach){homing_edge_pending|=raw & homing_expected;ax_inc[AX_CART]=0;return;}
    if(homing_verify){ax_inc[AX_CART]=0;return;}
  }
  // A bounced edge remains an inhibit even if foreground rate setting races
  // with the first ISR trip. Only the stopped debounce owner consumes it.
  if(homing_edge_pending){ax_inc[AX_CART]=0;return;}
  if(homing_centering && ax_pos[AX_CART]==homing_center_target)ax_inc[AX_CART]=0;
  // Cart-only installation: never emit Z pulses, even from a stale rate.
  const uint32_t inc=ax_inc[AX_CART];
  if(inc){
    const uint32_t prev=ax_acc[AX_CART];ax_acc[AX_CART]=prev+inc;
    if(ax_acc[AX_CART]<prev){
      ax_pos[AX_CART]+=ax_dir[AX_CART];digitalWriteFast(PIN_CART_STEP,HIGH);
    }
  }
}
// Independent 4 us Z interrupt: physical coordinates are positive UP.
// Each rising edge is followed by a full LOW tick; DIR changes get one tick setup.
void onZTimer(){
 const uint8_t pins[2]={PIN_Z1_STEP,PIN_Z2_STEP},dirs[2]={PIN_Z1_DIR,PIN_Z2_DIR};
 if(g_energized&&z_active&&(uint32_t(micros()-control_heartbeat_us)>CONTROL_STALL_US ||
    (executor_state.switches&&executor::requiredLimits(rawLimits())))){
  if(uint32_t(micros()-control_heartbeat_us)>CONTROL_STALL_US)timing_fault_pending=true;
  else limit_fault_pending=true;
  digitalWriteFast(PIN_EN_ALL,HIGH);g_energized=false;
 }
 for(int i=0;i<2;++i){
  digitalWriteFast(pins[i],LOW);
  const bool previous=z_high[i];z_high[i]=false;
  if(!g_energized||!(z_active&(1<<i)))continue;
  const int32_t delta=z_target[i]-z_count[i];if(!delta)continue;
  const int8_t direction=delta>0?1:-1;
  if(direction!=z_direction[i]){if(previous)continue;z_accumulator[i]=0;digitalWriteFast(dirs[i],((direction>0)==(i==1))?HIGH:LOW);z_direction[i]=direction;continue;}
  const uint32_t old=z_accumulator[i];z_accumulator[i]+=z_increment[i];
  if(z_accumulator[i]>=old)continue;
  if(previous){digitalWriteFast(PIN_EN_ALL,HIGH);g_energized=false;timing_fault_pending=true;continue;}
  z_count[i]+=direction;digitalWriteFast(pins[i],HIGH);z_high[i]=true;
 }
}
void axSetRate(int i,float steps_per_sec,bool invert){
  if(i!=AX_CART || !g_energized){ax_inc[i]=0;return;}
  const int8_t d=steps_per_sec>=0?1:-1;
  if(!direction_initialized[i] || ax_dir[i]!=d){
    ax_inc[i]=0;
    // Let the timer finish a previous pulse before changing DIR.
    delayMicroseconds(cart_hardware::step_timer_period_us+1);
    digitalWrite(DIR_PIN[i],((d>0)!=invert)?HIGH:LOW);
    ax_dir[i]=d;direction_initialized[i]=true;
    delayMicroseconds(1); // A4988 DIR setup, with pulse generation paused
  }
  float r=fabsf(steps_per_sec);
  if(r>ISR_HZ*0.5f)r=ISR_HZ*0.5f;
  const uint32_t inc=(uint32_t)((double)r*4294967296.0/(double)ISR_HZ);
  const uint32_t irq=interruptMask();__disable_irq();
  ax_inc[i]=g_energized?inc:0;
  if(!irq)__enable_irq();
}
void steppersInit(){
  const uint8_t outs[]={PIN_CART_STEP,PIN_CART_DIR,PIN_Z1_STEP,PIN_Z1_DIR,PIN_Z2_STEP,PIN_Z2_DIR};
  for(uint8_t p:outs){pinMode(p,OUTPUT);digitalWrite(p,LOW);}
  static_assert(CART_MICROSTEPS==16 && Z_MICROSTEPS==16,"Mechanics require BED default 1/16 microsteps");
  step_timer_ready=step_timer.begin(onStepTimer,cart_hardware::step_timer_period_us);
  z_timer_ready=z_timer.begin(onZTimer,4);z_timer.priority(48);
  step_timer.priority(32); // higher priority than normal USB/I2C interrupts
  if(!step_timer_ready)checkedPrintln("! step timer allocation failed; motion locked out");
}
void energize(bool on){
  const uint32_t irq=interruptMask();__disable_irq();
  on=on && step_timer_ready && z_timer_ready && !timing_fault_pending;
  if(on && !g_energized){
    control_heartbeat_us=micros();
    encoder_fault={};encoder_tick_failure={}; // New enabled episode only.
  }
  digitalWriteFast(PIN_EN_ALL,on?LOW:HIGH);
  g_energized=on;
  if(!on){z_active=0;for(int i=0;i<N_AX;++i)ax_inc[i]=0;clearStepPins();}
  if(!irq)__enable_irq();
}

// ============================== AS5600 ======================================
#define AS5600_ADDR   0x36
#define REG_STATUS    0x0B
#define REG_RAWANGLE  0x0C
#define REG_CONF_HI   0x07
#define REG_AGC       0x1A
#define ENC_CPR       4096

static int32_t enc_last_raw = 0, enc_ticks = 0, enc_zero = 0;
volatile int32_t enc_upright_raw = encoder_reference::upright_raw;
volatile bool reference_saved = false;
void loadEncoderReference() {
  uint16_t raw;
  if(teensy_reference_store::load(raw) && raw<ENC_CPR){
    enc_upright_raw=raw;reference_saved=true;
  }
}
volatile uint8_t enc_agc = 0;
volatile uint8_t enc_status = 0;
static bool mag_status_valid=false;
static uint32_t mag_status_us=0;
constexpr uint32_t MAG_STATUS_MAX_AGE_US=20000;
volatile uint32_t enc_errors = 0;
// Verify the sensor's operating mode, not just successful I2C transactions.
// AS5600 datasheet v1-06: PM=3 updates only every 100 ms. WD can enter that
// mode after inactivity. Preserve reserved and output-stage bits on writes.
constexpr uint16_t ENC_CONF_MASK=0x3F0F, ENC_CONF_EXPECTED=0x0300;
constexpr uint32_t ENC_CONF_MAX_AGE_US=20000;
static int32_t enc_boot_conf=-1, enc_conf=-1;
static bool enc_conf_valid=false;
static uint32_t enc_conf_us=0, enc_raw_changed_us=0;

// Capture immediately: the next Wire transaction clears hardware error flags.
// phase 1 = register-pointer TX, 2 = read request, 3 = register write.
void encTransactionFailure(uint8_t reg,uint8_t phase,uint8_t result,uint8_t expected,uint32_t started){
  if(encoder_tick_failure.phase)return;
  auto &f=encoder_tick_failure;
  f.reg=reg;f.phase=phase;f.result=result;f.expected=expected;
  f.duration_us=micros()-started;
#if defined(__IMXRT1062__)
  f.msr=IMXRT_LPI2C3.MSR; // Wire1 on Teensy 4.1; read only.
#endif
  f.pins=(digitalReadFast(PIN_SDA)?1:0)|(digitalReadFast(PIN_SCL)?2:0);
}

bool as5600Write8(uint8_t reg, uint8_t val) {
  Wire1.beginTransmission(AS5600_ADDR);
  Wire1.write(reg); Wire1.write(val);
  const uint32_t started=micros();
  const uint8_t result=Wire1.endTransmission();
  if(result)encTransactionFailure(reg,3,result,0,started);
  return result==0;
}
int16_t as5600Read8(uint8_t reg) {
  Wire1.beginTransmission(AS5600_ADDR);
  Wire1.write(reg);
  uint32_t started=micros();
  uint8_t result=Wire1.endTransmission(false);
  if(result){encTransactionFailure(reg,1,result,0,started);return -1;}
  started=micros();result=Wire1.requestFrom((uint8_t)AS5600_ADDR,(uint8_t)1);
  if(result!=1){encTransactionFailure(reg,2,result,1,started);return -1;}
  return Wire1.read();
}

int32_t as5600ReadConf() {
  Wire1.beginTransmission(AS5600_ADDR);Wire1.write(REG_CONF_HI);
  uint32_t started=micros();
  uint8_t result=Wire1.endTransmission(false);
  if(result){encTransactionFailure(REG_CONF_HI,1,result,0,started);++enc_errors;return -1;}
  started=micros();result=Wire1.requestFrom((uint8_t)AS5600_ADDR,(uint8_t)2);
  if(result!=2){encTransactionFailure(REG_CONF_HI,2,result,2,started);++enc_errors;return -1;}
  const uint16_t hi=Wire1.read(),lo=Wire1.read();
  return (hi<<8)|lo;
}
void encConfigRead() {
  enc_conf=as5600ReadConf();enc_conf_us=micros();
  enc_conf_valid=enc_conf>=0 && (uint16_t(enc_conf)&ENC_CONF_MASK)==ENC_CONF_EXPECTED;
}
// NOM power, watchdog off, hysteresis off, slow filter 2x (0.286 ms).
// Acknowledged writes alone do not prove the device accepted this mode.
bool as5600FastFilter() {
  enc_conf_valid=false;
  const int32_t before=as5600ReadConf();
  if(before<0){enc_conf=-1;return false;}
  const uint16_t desired=(uint16_t(before)&uint16_t(~ENC_CONF_MASK))|ENC_CONF_EXPECTED;
  const bool high=as5600Write8(REG_CONF_HI,uint8_t(desired>>8));
  const bool low=as5600Write8(REG_CONF_HI+1,uint8_t(desired));
  if(!high || !low)++enc_errors;
  encConfigRead();
  enc_conf_valid=enc_conf_valid && high && low;
  return enc_conf_valid;
}

// I2C is used only in setup and the foreground control tick, never in an ISR.
// Teensy Wire has its own bounded bus waits; the step ISR independently stops
// motion if the foreground fails to complete control for more than 5 ms.
volatile bool     enc_ok = false;
volatile uint32_t enc_recoveries = 0;
static   uint32_t enc_consec_err = 0;

void encBusReinit() {
  Wire1.end();
  delay(2);
  Wire1.setSDA(PIN_SDA);
  Wire1.setSCL(PIN_SCL);
  Wire1.begin();
  Wire1.setClock(I2C_HZ);
  as5600FastFilter();          // CONF is volatile, re-apply after a re-init
  enc_recoveries++;
}

int32_t encReadRaw() {
  Wire1.beginTransmission(AS5600_ADDR);
  Wire1.write(REG_RAWANGLE);
  uint32_t started=micros();
  uint8_t result=Wire1.endTransmission(false);
  if(result){encTransactionFailure(REG_RAWANGLE,1,result,0,started);++enc_errors;return -1;}
  started=micros();result=Wire1.requestFrom((uint8_t)AS5600_ADDR,(uint8_t)2);
  if(result!=2){encTransactionFailure(REG_RAWANGLE,2,result,2,started);++enc_errors;return -1;}
  uint16_t hi = Wire1.read(), lo = Wire1.read();
  return ((hi << 8) | lo) & 0x0FFF;
}

void encUpdate() {
  int32_t raw = encReadRaw();
  if (raw < 0) {
    // Hold the last value through a one-off glitch, but a sustained run of
    // failures means the bus is wedged (a stretched clock, a brown-out on the
    // module). Rebuild it rather than running the controller blind.
    if (++enc_consec_err >= 50) { enc_consec_err = 0; encBusReinit(); }
    enc_ok = false;
    return;
  }
  enc_consec_err = 0;
  enc_ok = true;
  int32_t d = raw - enc_last_raw;
  if (d >  ENC_CPR/2) d -= ENC_CPR;
  if (d < -ENC_CPR/2) d += ENC_CPR;
  enc_ticks += d;
  if(raw!=enc_last_raw)enc_raw_changed_us=micros();
  enc_last_raw = raw;
}

int32_t encZeroRaw() {
  // enc_ticks is relative to the first sample, not the sensor absolute zero.
  const int32_t raw_zero=enc_last_raw-(enc_ticks-enc_zero)%ENC_CPR;
  return (raw_zero%ENC_CPR+ENC_CPR)%ENC_CPR;
}

inline float wrapPi(float a) {
  while (a >  (float)M_PI) a -= 2.0f*(float)M_PI;
  while (a < -(float)M_PI) a += 2.0f*(float)M_PI;
  return a;
}

// theta: 0 = upright, +-pi = hanging, positive = pole top leaning toward +x
float encTheta() {
  // One live reference for the observer, capture gate, balance and telemetry.
  return encoder_reference::angle_from_upright(enc_last_raw, ENC_INVERT != 0, enc_upright_raw);
}

// One bounded status-register transaction at 100 Hz, independent of the 1 kHz
// angle read. A successful angle transaction alone cannot establish magnet health.
void encStatusRead(){
  const int16_t status=as5600Read8(REG_STATUS);
  mag_status_us=micros();mag_status_valid=status>=0;
  if(status<0){++enc_errors;enc_status=0;}
  else enc_status=(uint8_t)status;
}
bool encoderHealthy(){
  return enc_ok && enc_consec_err==0 && enc_conf_valid &&
    uint32_t(micros()-enc_conf_us)<=ENC_CONF_MAX_AGE_US && mag_status_valid &&
    uint32_t(micros()-mag_status_us)<=MAG_STATUS_MAX_AGE_US &&
    (enc_status & 0x20) && !(enc_status & 0x18);
}

// Magnet diagnostics. The reviews on these modules are full of people who got
// the wrong magnet - check this before blaming the controller.
//   STATUS bit5 MD = detected, bit4 ML = too weak, bit3 MH = too strong
//   AGC should sit mid-range (~64 of 0..128 at 3.3V). Railed = bad air gap.
// Bus access - control tick (or setup) only.
void encDiagRead() {
  encStatusRead();
  int16_t ag = as5600Read8(REG_AGC);
  if (ag >= 0) enc_agc = ag;
}

// Printing only - safe from loop(), reads the cached copies.
void encPrintMagnet() {
  uint8_t st = enc_status, ag = enc_agc;
  if (!enc_ok) checkedPrintln(F("! encoder not reading - values below are stale"));
  checkedPrintf("# AS5600 status=0x%02X agc=%d  %s%s%s\n", st, ag,
                (st & 0x20) ? "magnet-ok " : "NO-MAGNET ",
                (st & 0x10) ? "too-weak "  : "",
                (st & 0x08) ? "too-strong" : "");
  if (ag >= 120)     checkedPrintln(F("# AGC at max gain - magnet too far, move it CLOSER"));
  else if (ag <= 8)  checkedPrintln(F("# AGC at min gain - magnet too close"));
  checkedPrintf("# i2c errors=%lu  bus recoveries=%lu\n",
                (unsigned long)enc_errors, (unsigned long)enc_recoveries);
}

// ========================== CONTROLLER ======================================
enum Mode { M_IDLE=0, M_MANUAL, M_SWINGUP, M_BALANCE, M_FAULT, M_RAIL_BRAKE, M_RAIL_RETURN, M_SPIN_BRAKE, M_SPIN_CENTER, M_SPIN_WAIT, M_BAL_BRAKE, M_BAL_CENTER, M_EXTERNAL, M_HOMING, M_REHOME, M_VERTICAL };
volatile Mode mode = M_IDLE;
const char *MODE_NAME[] = {"IDLE","MANUAL","SWINGUP","BALANCE","FAULT","RAIL_BRAKE","RAIL_RETURN","SPIN_BRAKE","SPIN_CENTER","SPIN_WAIT","BAL_BRAKE","BAL_CENTER","EXTERNAL","HOMING","REHOME","VERTICAL"};

volatile float th=0, thd=0, xc=0, vc=0, acc_cmd=0, energy_n=-1;
volatile float z_mm=0;
volatile uint32_t loop_us=0;
static float th_hat=0, thd_hat=0;
volatile bool req_estimator_reset=false;
static swing_control::PumpState pump;
static swing_control::ResponseWatch response_watch;
static int32_t home_steps=0, zhome_steps=0;
static int32_t measured_home_span_steps=0;
static uint32_t home_epoch=0;
constexpr float SAVED_RAIL_HALF=.26f;
// Default HOME uses measured travel. An explicit stopped rail write narrows it.
static float configured_rail=cart_homing::maximum_rail_half;

// Deferred foreground requests. loop() may only SET these; the control tick is the
// only thing allowed to act on them, because it is the sole owner of the I2C
// bus once it exists. This is the whole fix for 'mag' returning nothing.
volatile bool req_diag = false, req_encoder = false;
// 0 none, 1 hanging capture, 2 upright capture. Operator-only V2 command.
volatile int calibration_request=0;
volatile int32_t requested_upright_raw=0;
volatile int16_t requested_hanging_offset=0;
volatile bool requested_model_capture=false;
volatile bool reference_save_pending=false;
static char display_fault[81]="See serial log";
volatile uint32_t dropped_control_messages=0;
void controlLog(const char *text){
  if(mode==M_FAULT && text[0]=='!')snprintf(display_fault,sizeof(display_fault),"%.80s",text+2);
  checkedPrintln(text);
}
volatile bool diag_done = false;
static volatile bool zeroing = false;
static bool capture_upright=false,capture_model=false;
static encoder_reference::StillCapture reference_capture;
static encoder_reference::ModelStillCapture model_reference_capture;
static uint32_t zero_t0=0;
static uint32_t calibration_sid=0;
void queueReply(const char* payload);
bool encoderCalibrationBusy(){return calibration_request!=0 || zeroing || reference_save_pending;}
void cancelEncoderCalibration(){
  calibration_request=0;zeroing=false;reference_save_pending=false;
  executor_state.calibration=0;executor_state.calibration_offset=0;executor_state.calibration_model=false;
  requested_hanging_offset=0;requested_model_capture=false;capture_model=false;
  reference_capture.reset();model_reference_capture.reset();calibration_sid=0;
}
void finishEncoderCalibration(const char* error){
  const uint32_t sid=calibration_sid;
  cancelEncoderCalibration();
  if(sid){
    char payload[100];
    snprintf(payload,sizeof payload,"V2 %s %lu %s",error?"ERR":"OK",
      (unsigned long)sid,error?error:"CALIBRATE");
    queueReply(payload);
  }
}
void applyEncoderReference(int32_t raw){
  // Called only after persistence has been verified, with every driver disabled.
  enc_upright_raw=encoder_reference::normalize(raw);
  enc_zero=enc_ticks+encoder_reference::difference(encoder_reference::opposite(enc_upright_raw),enc_last_raw);
  th=th_hat=encTheta();thd=thd_hat=0;energy_n=cosf(th);
  pump.reset();response_watch.armed=false;
  char message[120];snprintf(message,sizeof(message),"# upright reference applied: %ld counts, theta=%.2fdeg; saved",(long)enc_upright_raw,th*180/M_PI);
  controlLog(message);
}
// Non-real-time persistence. All motion commands remain blocked until finished.
void saveEncoderReference(){
  if(!reference_save_pending)return;
  if(g_energized || executor_state.enabled || !executor_state.calibration){
    finishEncoderCalibration("calibration_state");return;
  }
  const bool ok=teensy_reference_store::save((uint16_t)requested_upright_raw);
  if(ok){applyEncoderReference(requested_upright_raw);reference_saved=true;}
  else{
    uint16_t stored;
    reference_saved=teensy_reference_store::load(stored) && stored==enc_upright_raw;
  }
  finishEncoderCalibration(ok?nullptr:"save");
}

static float v_manual=0;
static uint32_t v_manual_deadline=0;
static float zv_target=0, zv_now=0;
static int cart_brake_direction=0;
static rail_recovery::State rail_recovery_state;
static bool recovery_was_manual=false;
static spin_recovery::State spin_recovery_state;
static upright_session::Return upright_return;
static saved_center_return::State saved_return;
static bool upright_only=false;
volatile bool req_balance=false;
bool inUprightReturn(){return mode==M_BAL_BRAKE || mode==M_BAL_CENTER;}
bool inSpinRecovery(){return mode==M_SPIN_BRAKE || mode==M_SPIN_CENTER || mode==M_SPIN_WAIT;}

// Exact pole placement. With a = -(K1*th + K2*thd + K3*x + K4*v) the closed
// loop characteristic polynomial of this plant is
//   s^4 + (K4 - K2/L)s^3 + (K3 - g/L - K1/L)s^2 - (g*K4/L)s - (g*K3/L)
// Matching that to (s^2+2*z*w*s+w^2)(s-pc1)(s-pc2) inverts in closed form.
void computeGains() {
  const auto k = swing_control::gains(controller);
  K1=k.k1; K2=k.k2; K3=k.k3; K4=k.k4;
}

// Alpha-beta tracking differentiator. Differencing a 12-bit encoder at 1 kHz
// would give ~1.5 rad/s of quantisation hash; this smooths it and handles the
// +-pi wrap without a glitch.
void estimate(float theta_meas) {
  if(req_estimator_reset){
    swing_control::resetEstimate(theta_meas,th_hat,thd_hat);
    req_estimator_reset=false;
  }
  swing_control::estimate(theta_meas, p_bw, DT, th_hat, thd_hat);
  th = theta_meas;  // telemetry retains the raw encoder angle
  thd = thd_hat;
}

swing_control::Gains currentGains() {
  if(upright_only)return swing_control::uprightGains(controller);
  return {K1,K2,K3,K4};
}
float balanceAccel() {
  return swing_control::balance(currentGains(), th_hat, thd, xc, vc);
}
float swingAccel() {
  energy_n = swing_control::energy(th_hat, thd, p_leff);
  const auto feedback=currentGains();
  return swing_control::swing(controller, pump, th_hat, thd, xc, vc, DT, acc_cmd, &feedback);
}

void eStop() {
  energize(false); // Disable immediately; ISR also suppresses pulses/counts.
  saved_return.reset();
  executor_state.stop();homing.cancel();homing_active=homing_centering=false;homing_edge_pending=0;
  cancelEncoderCalibration();
  mode = M_IDLE;
  req_balance=false; upright_only=false; upright_return.reset();
  cart_brake_direction = 0;
  rail_recovery_state.reset(); recovery_was_manual=false;
  spin_recovery_state.reset();
  pump.reset(); response_watch.armed = false;
  v_manual = 0; vc = 0; acc_cmd = 0; zv_target = 0; zv_now = 0;
  for (int i=0;i<N_AX;i++) ax_inc[i] = 0;
  clearStepPins();
}

void cancelVertical(){
 const uint32_t irq=interruptMask();__disable_irq();z_active=0;executor_state.z.cancel();
 digitalWriteFast(PIN_Z1_STEP,LOW);digitalWriteFast(PIN_Z2_STEP,LOW);
 if(!irq)__enable_irq();
}
void serviceVertical(){
 auto &z=executor_state.z;
 if(!z.phase)return;
 if(!executor_state.enabled || !g_energized || (mode!=M_VERTICAL&&mode!=M_MANUAL&&mode!=M_SWINGUP&&mode!=M_BALANCE&&mode!=M_EXTERNAL)){cancelVertical();return;}
 const uint32_t irq=interruptMask();__disable_irq();
 for(int i=0;i<2;++i)z.count[i]=z_count[i];
 if(!irq)__enable_irq();
 int32_t targets[2];uint32_t increments[2];bool lag=false,arrived=true;
 for(int i=0;i<2;++i){
  targets[i]=(z.selected&(1<<i))?int32_t(lround(z.position(i,executor_state.tick)*vertical::STEPS_MM)):z.count[i];
  if(z.phase!=2&&(z.selected&(1<<i))&&(targets[i]<0||targets[i]>int32_t(floor(z.max_mm*vertical::STEPS_MM)))){
   executor_state.trip(executor::LIMIT);eStop();mode=M_FAULT;return;
  }
  const int32_t delta=targets[i]-z.count[i];if(abs(delta)>200)lag=true;if(delta)arrived=false;
  increments[i]=uint32_t(std::min(vertical::MAX_RATE,fabs(double(delta))*1100.)*4294967296./250000.);
 }
 if(lag){executor_state.trip(executor::DEADLINE);eStop();mode=M_FAULT;return;}
 const bool elapsed=uint32_t(executor_state.tick-z.started)>=uint32_t(ceil(z.duration*1000));
 if(elapsed&&(arrived||z.phase==2)){
  if(z.phase==3){z.phase=4;z.started=executor_state.tick;z.duration=z.seconds;}
  else{if(z.phase==2)z.bottom|=z.selected;z.phase=0;z.selected=0;}
 }
 __disable_irq();
 for(int i=0;i<2;++i){z_target[i]=targets[i];z_increment[i]=increments[i];}
 if(!z.phase)z_active=0;
 if(!irq)__enable_irq();
 if(!z.phase&&mode==M_VERTICAL)eStop();
}
void beginSavedReturn(){
  cancelVertical();saved_return.begin();mode=M_REHOME;
  req_balance=false;upright_only=false;upright_return.reset();
  rail_recovery_state.reset();recovery_was_manual=false;spin_recovery_state.reset();
  response_watch.armed=false;pump.reset();v_manual=0;zv_target=0;
}

void beginUprightReturn(const char *reason) {
  cancelVertical();upright_return.begin(micros());
  rail_recovery_state.reset(); spin_recovery_state.reset();
  response_watch.armed=false; pump.reset(); v_manual=0; zv_target=0;
  mode=M_BAL_BRAKE;
  controlLog(reason);
}

void beginRailRecovery() {
  if(upright_only){beginUprightReturn("# upright trial rail protection -> BAL_BRAKE; return to center and stop");return;}
  if (mode==M_RAIL_BRAKE || mode==M_RAIL_RETURN) return;
  cancelVertical();recovery_was_manual = mode==M_MANUAL;
  rail_recovery_state.begin(xc,p_rail,cart_brake_direction,controller.rail_margin,controller.rail_hysteresis);
  // Keep the startup response watch active through recovery.
  mode=M_RAIL_BRAKE;
  controlLog("# rail recovery -> BRAKING; pendulum control paused");
}

void beginSpinRecovery() {
  cancelVertical();spin_recovery_state.begin(micros());
  rail_recovery_state.reset(); recovery_was_manual=false;
  response_watch.armed=false; pump.reset();
  v_manual=0; zv_target=0;
  mode=M_SPIN_BRAKE;
  char message[150];snprintf(message,sizeof(message),"# pendulum |theta_dot| >%.2f rad/s -> SPIN_BRAKE; center and wait below %.2f rad/s",controller.spin_trip_rad_s,controller.spin_resume_rad_s);
  controlLog(message);
}

// Freeze first enabled unhealthy sample before the existing safety trip.
// Subsequent successful reads, STOP, HELLO and INFO retain this evidence.
void latchEncoderFault(uint32_t tick_started){
  if(encoder_fault.valid)return;
  using namespace encoder_diagnostics;
  auto &f=encoder_fault;const uint32_t now=micros();
  f.valid=true;f.tick=executor_state.tick;f.at_us=now;f.mode=uint8_t(mode);
  f.status=enc_status;f.conf=enc_conf;
  f.status_age_us=now-mag_status_us;f.conf_age_us=now-enc_conf_us;
  f.health_mask=(!enc_ok?ANGLE:0)|(enc_consec_err?CONSECUTIVE:0)|
    (!enc_conf_valid?CONF_INVALID:0)|(f.conf_age_us>ENC_CONF_MAX_AGE_US?CONF_STALE:0)|
    (!mag_status_valid?STATUS_INVALID:0)|(f.status_age_us>MAG_STATUS_MAX_AGE_US?STATUS_STALE:0)|
    (!(enc_status&0x20)||(enc_status&0x18)?MAGNET:0);
  f.errors=enc_errors;f.consecutive=enc_consec_err;f.recoveries=enc_recoveries;
  f.deadline_misses=control_deadline_misses;f.timing_pending=timing_fault_pending;
  f.control_elapsed_us=now-tick_started;f.transaction=encoder_tick_failure;
}

void controlTick() {
    uint32_t t_start = micros();
    encoder_tick_failure={};
    executor_state.advance(t_start);
    if(!executor_state.enabled && g_energized)eStop();
    if(executor_state.fault!=executor::NONE)mode=M_FAULT;
    encUpdate();
    if(executor_state.tick%10==0)encStatusRead();
    // Offset from magnet status polling to bound each tick's bus work.
    if(executor_state.tick%10==5)encConfigRead();
    const bool healthy=encoderHealthy();
    if(executor_state.enabled && !healthy && executor_state.fault==executor::NONE)latchEncoderFault(t_start);
    executor_state.sample(healthy,rawLimits());
    if(!executor_state.enabled && g_energized){eStop();mode=M_FAULT;}

    serviceVertical();

    if(req_encoder){
      req_encoder=false;encDiagRead();
      char sample[160];
      snprintf(sample,sizeof(sample),"E %lu %ld %ld %u %u %lu %d %d",
        (unsigned long)millis(),(long)enc_last_raw,(long)encZeroRaw(),
        (unsigned)enc_agc,(unsigned)enc_status,(unsigned long)enc_errors,(int)enc_ok,(int)g_energized);
      controlLog(sample);
    }

    // Never continue closed-loop control on stale angle data. A single failed
    // transaction is held over; five consecutive failures (~5 ms) latch a
    // fault and stop pulse generation.
    if (mode != M_IDLE && mode != M_FAULT && enc_consec_err >= 5) {
      controlLog("! encoder stale -> FAULT");
      eStop(); mode = M_FAULT;
    }

    // Serve bus requests here, where we already own Wire1.
    if (req_diag) { req_diag = false; encDiagRead(); diag_done = true; }
    if(calibration_request){
      capture_upright=calibration_request==2;capture_model=requested_model_capture;
      calibration_request=0;zeroing=true;reference_capture.reset();model_reference_capture.reset();zero_t0=millis();
    }
    if(zeroing){
      if(!executor_state.calibration || mode!=M_IDLE || g_energized){
        finishEncoderCalibration("calibration_state");
      }else if(!encoderHealthy()){
        finishEncoderCalibration("encoder");
      }else if(uint32_t(executor_state.tick-executor_state.host_at)>=200){
        finishEncoderCalibration("calibration_watchdog");
      }else if(uint32_t(millis()-zero_t0)>=5000){
        finishEncoderCalibration("calibration_timeout");
      }else if(capture_model?model_reference_capture.add(enc_last_raw,true):reference_capture.add(enc_last_raw,true)){
        const int32_t raw=capture_model?model_reference_capture.median():reference_capture.mean();
        requested_upright_raw=capture_upright?raw:encoder_reference::normalize(encoder_reference::opposite(raw)+requested_hanging_offset);
        zeroing=false;reference_save_pending=true;
      }
    }

    estimate(encTheta());
    xc = (ax_pos[AX_CART] - home_steps) / CART_STEPS_PER_M;
    z_mm = (ax_pos[AX_Z1] - zhome_steps) / Z_STEPS_PER_MM;

    // Z-only authority still publishes fresh measured cart/pendulum state.
    if(mode==M_VERTICAL){ax_inc[AX_CART]=0;loop_us=micros()-t_start;return;}

    float rail_hard = p_rail - controller.rail_fault_margin;
    if (mode != M_IDLE && mode != M_FAULT && mode != M_HOMING && fabsf(xc) > rail_hard) {
      executor_state.trip(executor::RAIL);
      controlLog("! rail limit -> FAULT");
      mode = M_FAULT; eStop(); mode = M_FAULT;
    }

    // Homing owns only cart pulses, and never commands either Z axis.
    if(mode==M_HOMING){
      const uint32_t sample_irq=interruptMask();__disable_irq();
      const uint8_t sampled=executor::requiredLimits(rawLimits()) | homing_edge_pending;
      if(homing_edge_pending || (homing_approach && (sampled & homing_expected)))ax_inc[AX_CART]=0;
      homing_edge_pending=0;
      if(!sample_irq)__enable_irq();
      homing.update(executor_state.tick,ax_pos[AX_CART],sampled);
      if(homing.failed()){executor_state.trip(executor::HOMING);eStop();mode=M_FAULT;}
      else if(homing.done()){
        const int32_t center=homing.center();
        p_rail=homing.safeRail(configured_rail,controller.rail_fraction,controller.rail_margin);
        measured_home_span_steps=homing.span();++home_epoch;
        eStop();home_steps=center;
        executor_state.homed=true;xc=0;
      }else{
        const uint32_t irq=interruptMask();__disable_irq();
        homing_expected=homing.expected();homing_approach=homing.approaching();
        homing_verify=homing.phase==cart_homing::Machine::VERIFY_NEG || homing.phase==cart_homing::Machine::VERIFY_POS || homing.phase==cart_homing::Machine::VERIFY_LATCH_NEG || homing.phase==cart_homing::Machine::VERIFY_LATCH_POS;
        homing_centering=homing.phase==cart_homing::Machine::CENTER;
        homing_center_target=homing.center();
        if(!irq)__enable_irq();
        vc=homing.velocity(ax_pos[AX_CART]);acc_cmd=0;executor_state.demand=vc;
        axSetRate(AX_CART,vc*CART_STEPS_PER_M,CART_INVERT);
      }
      axSetRate(AX_Z1,0,Z_INVERT);axSetRate(AX_Z2,0,Z_INVERT);
      loop_us=micros()-t_start;return;
    }
    // Normal auto expiry requests this only after fresh health/rail guards.
    // Preserve inherited v/a: returning is a jerk-limited brake, not an eStop.
    if(executor_state.enabled && executor_state.mode==M_REHOME && mode!=M_REHOME)beginSavedReturn();
    if(mode==M_REHOME){
      cart_motion::State current;
      current.velocity=vc;current.acceleration=acc_cmd;current.brake_direction=cart_brake_direction;
      const uint32_t irq=interruptMask();__disable_irq();
      const bool centered=saved_return.settledAtCenter(ax_pos[AX_CART]-home_steps,current);
      if(centered){eStop();xc=0;} // Atomic count check and disable; publish that verified coordinate.
      if(!irq)__enable_irq();
      if(!centered){
        const auto next=saved_return.advance(current,xc,p_rail,DT,p_vmax,p_jmax,max(p_amax_s,p_amax_b),controller.rehome_v,controller.rehome_a,controller.rehome_j,controller.rail_margin);
        if(!isfinite(next.velocity)||!isfinite(next.acceleration)||fabsf(next.velocity)>p_vmax+.002f||fabsf(next.velocity)>VMAX_CEIL||fabsf(next.acceleration)>max(max(p_amax_s,p_amax_b),controller.rehome_a)+.001f){
          executor_state.trip(executor::CONTROLLER);eStop();mode=M_FAULT;
        }else{
          vc=next.velocity;acc_cmd=next.acceleration;cart_brake_direction=next.brake_direction;
          executor_state.demand=saved_return.demand;
          axSetRate(AX_CART,vc*CART_STEPS_PER_M,CART_INVERT);
        }
      }
      axSetRate(AX_Z1,0,Z_INVERT);axSetRate(AX_Z2,0,Z_INVERT);
      loop_us=micros()-t_start;return;
    }
    // Consume the one-shot start in the control tick using fresh sensor state.
    // Rejection never arms a future automatic capture.
    if(req_balance){
      req_balance=false;
      if(mode!=M_IDLE || encoderCalibrationBusy() || !upright_session::canStart(th,th_hat,thd,xc,vc,acc_cmd,enc_ok && enc_consec_err==0)){
        char reason[160];snprintf(reason,sizeof(reason),"! bal rejected: angle=%.2fdeg estimate=%.2fdeg rate=%.2f x=%.1fmm; need +/-10deg, rate<=1, x<=30mm, stopped + healthy",th*180/M_PI,th_hat*180/M_PI,thd,xc*1000);
        controlLog(reason);
      }else{
        upright_only=true; energize(true); mode=M_BALANCE;
        controlLog("# BALANCE");
        controlLog("# upright-only; >50deg -> brake, center, disable; no automatic restart");
      }
    }
    if(upright_only && mode==M_BALANCE && upright_session::fallen(th))
      beginUprightReturn("# upright fall >50deg -> BAL_BRAKE; return to center and stop");

    const bool automatic=mode==M_SWINGUP || mode==M_BALANCE ||
        ((mode==M_RAIL_BRAKE || mode==M_RAIL_RETURN) && !recovery_was_manual);
    if(!upright_only && automatic && enc_ok && spin_recovery::trigger(thd,controller.spin_trip_rad_s))beginSpinRecovery();

    if ((mode==M_SWINGUP || mode==M_BALANCE) &&
        rail_recovery_state.atEdge(xc,p_rail,controller.rail_margin)) beginRailRecovery();

    float a = 0.0f;
    switch (mode) {
      case M_EXTERNAL:
        a=executor_state.demand;
        break;

      case M_MANUAL:
        v_manual=executor_state.demand;
        a = cart_motion::velocityAccel(vc, v_manual, p_amax_m, p_jmax_m, DT);
        break;

      case M_SWINGUP:
        a = swingAccel();
        if (swing_control::canCapture(controller, currentGains(), th_hat, thd, xc, vc, acc_cmd)) {
          mode = M_BALANCE; response_watch.armed = false;
          a = constrain(balanceAccel(), -p_amax_b, p_amax_b);
          controlLog("# caught -> BALANCE");
        }
        break;

      case M_BALANCE: {
        float Eup = GRAV/p_leff;
        energy_n = (0.5f*thd*thd + Eup*cosf(th))/Eup;
        a = constrain(balanceAccel(), -p_amax_b, p_amax_b);
        if (!upright_only && fabsf(th_hat) > p_giveup) {
          mode = M_SWINGUP;
          controlLog("# lost it -> SWINGUP");
        }
        break;
      }

      case M_RAIL_BRAKE:
      case M_RAIL_RETURN: {
        cart_motion::State current;
        current.velocity=vc; current.acceleration=acc_cmd;
        current.brake_direction=cart_brake_direction;
        const float recovery_amax=recovery_was_manual?p_amax_m:max(p_amax_s,p_amax_b);
        const float recovery_jerk=recovery_was_manual?p_jmax_m:p_jmax;
        if(rail_recovery_state.update(xc,current,DT,controller.rail_recover_v,controller.rail_recover_a)) {
          char handoff[96];snprintf(handoff,sizeof(handoff),"# rail handoff x=%.5f v=%.4f a=%.3f",xc,vc,acc_cmd);
          controlLog(handoff);
          if(recovery_was_manual) {
            recovery_was_manual=false; mode=M_MANUAL;
            v_manual=0; v_manual_deadline=millis();
            a=cart_motion::velocityAccel(vc,0,p_amax_m,p_jmax_m,DT);
            controlLog("# back inside operating range -> MANUAL");
            break;
          }
          pump.reset();
          mode=M_SWINGUP;
          a=swingAccel();
          controlLog("# back inside operating range -> SWINGUP");
        } else if(rail_recovery_state.timedOut(controller.rail_recover_s)) {
          eStop(); mode=M_FAULT;
          controlLog("! rail recovery timeout -> FAULT");
          goto zaxis;
        } else {
          if(mode==M_RAIL_BRAKE && rail_recovery_state.phase==rail_recovery::RETURNING) {
            mode=M_RAIL_RETURN; controlLog("# rail speed and acceleration settled -> RETURN_INSIDE");
          }
          a=rail_recovery_state.demand(xc,current,p_vmax,recovery_amax,recovery_jerk,DT,controller.rail_recover_v,controller.rail_recover_a);
        }
        break;
      }

      case M_BAL_BRAKE:
      case M_BAL_CENTER: {
        cart_motion::State current;
        current.velocity=vc;current.acceleration=acc_cmd;current.brake_direction=cart_brake_direction;
        const uint32_t now=micros();
        if(upright_return.update(xc,current,now)){
          eStop();controlLog("# upright trial centered -> STOP; drivers disabled; send bal for another trial");
          goto zaxis;
        }
        if(upright_return.timedOut(now)){
          eStop();mode=M_FAULT;controlLog("! upright return timeout -> FAULT; drivers disabled");
          goto zaxis;
        }
        mode=upright_return.braking()?M_BAL_BRAKE:M_BAL_CENTER;
        a=upright_return.demand(xc,current,p_vmax,max(p_amax_s,p_amax_b),p_jmax,DT);
        break;
      }

      case M_SPIN_BRAKE:
      case M_SPIN_CENTER:
      case M_SPIN_WAIT: {
        cart_motion::State current;
        current.velocity=vc;current.acceleration=acc_cmd;current.brake_direction=cart_brake_direction;
        const uint32_t now=micros();
        const auto previous=spin_recovery_state.phase;
        if(spin_recovery_state.update(xc,current,thd,enc_ok,now,controller.spin_resume_rad_s)) {
          pump.reset();response_watch.reset(th);mode=M_SWINGUP;
          a=swingAccel();
          char message[100];snprintf(message,sizeof(message),"# centered and |theta_dot| <%.2f rad/s -> SWINGUP",controller.spin_resume_rad_s);
          controlLog(message);
        } else if(spin_recovery_state.timedOut(now)) {
          eStop();mode=M_FAULT;
          controlLog("! spin recovery centering timeout -> FAULT");
          goto zaxis;
        } else {
          mode=spin_recovery_state.phase==spin_recovery::BRAKING?M_SPIN_BRAKE:
               spin_recovery_state.phase==spin_recovery::CENTERING?M_SPIN_CENTER:M_SPIN_WAIT;
          if(previous!=spin_recovery_state.phase){char notice[64];snprintf(notice,sizeof(notice),"# spin recovery -> %s",MODE_NAME[mode]);controlLog(notice);}
          a=spin_recovery_state.demand(xc,current,p_vmax,max(p_amax_s,p_amax_b),p_jmax,DT);
        }
        break;
      }

      default:                                   // IDLE / FAULT
        vc = 0; acc_cmd = 0; cart_brake_direction = 0;
        axSetRate(AX_CART, 0, CART_INVERT);
        goto zaxis;
    }

    {
      cart_motion::State motion;
      motion.velocity = vc;
      motion.acceleration = acc_cmd;
      motion.brake_direction = cart_brake_direction;
      const bool use_manual_limits=mode==M_MANUAL ||
        ((mode==M_RAIL_BRAKE || mode==M_RAIL_RETURN) && recovery_was_manual);
      const float drive_amax = use_manual_limits ? p_amax_m :
                               ((mode==M_EXTERNAL || mode==M_RAIL_BRAKE || mode==M_RAIL_RETURN || inSpinRecovery() || inUprightReturn()) ? max(p_amax_s,p_amax_b) :
                                (mode == M_SWINGUP ? p_amax_s : p_amax_b));
      const float brake_amax = use_manual_limits ? p_amax_m : max(p_amax_s, p_amax_b);
      const float jerk = use_manual_limits ? p_jmax_m : p_jmax;
      const float mode_speed=use_manual_limits?fminf(p_vmax,controller.manual_speed_max):
          mode==M_SWINGUP?swing_control::speedLimit(controller,false):
          mode==M_BALANCE?swing_control::speedLimit(controller,true):
          swing_control::recoverySpeedLimit(controller);
      // A lower mode cap is approached through jerk limiting; only the hard
      // ceiling below is a fault. Never clip velocity on a mode transition.
      if(mode!=M_EXTERNAL && mode!=M_MANUAL)executor_state.demand=a;
      motion = cart_motion::advance(motion, xc, a, mode_speed, drive_amax,
                                    brake_amax, jerk, p_rail, DT,controller.rail_margin);
      vc = motion.velocity;
      acc_cmd = motion.acceleration;
      cart_brake_direction = motion.brake_direction;
      if (cart_brake_direction && (mode==M_SWINGUP || mode==M_BALANCE))
        beginRailRecovery();
      // Only exceptional stops bypass the jerk ramp. Never keep integrating
      // a speed the pulse generator cannot produce.
      if (fabsf(vc) > p_vmax + 0.002f || fabsf(vc)>VMAX_CEIL) {
        controlLog("! motion speed limit -> FAULT");
        eStop(); mode = M_FAULT;
      }
      if ((mode==M_SWINGUP || ((mode==M_RAIL_BRAKE || mode==M_RAIL_RETURN) && !recovery_was_manual)) && response_watch.fault(th, vc, DT)) {
        controlLog("! no pendulum response to cart command -> FAULT; check motor tracking and recenter");
        eStop(); mode = M_FAULT;
      }
      axSetRate(AX_CART, vc*CART_STEPS_PER_M, CART_INVERT);
    }

  zaxis:
    {   // both screws get the identical command so the rail cannot rack
      float dv = p_zamax*DT;
      if      (zv_now < zv_target - dv) zv_now += dv;
      else if (zv_now > zv_target + dv) zv_now -= dv;
      else                              zv_now  = zv_target;
      if (z_mm >= p_ztravel && zv_now > 0) zv_now = 0;
      if (z_mm <= 0.0f      && zv_now < 0) zv_now = 0;
      float zr = zv_now*Z_STEPS_PER_MM;
      axSetRate(AX_Z1, zr, Z_INVERT);
      axSetRate(AX_Z2, zr, Z_INVERT);
    }

    loop_us = micros() - t_start;
}

// ============================= V2 TRANSPORT ==================================
// Small bounded ACK queue and a replaceable latest snapshot. A partially sent
// frame must finish to preserve framing; unsent snapshots are always replaced.
struct TxFrame {char bytes[256];uint16_t length=0;};
static TxFrame replies[4],latest_state,active_tx;
static uint8_t reply_head=0,reply_tail=0,reply_count=0;
static uint16_t active_offset=0;
static bool latest_pending=false,status_requested=false;
static uint32_t last_emitted_tick=0;
void resetTx(){
  reply_head=reply_tail=reply_count=0;latest_pending=false;
  // Finish an already visible line. Truncating it would concatenate the new
  // HELLO with old bytes and make the handshake itself fail CRC validation.
  if(!active_tx.length || !active_offset){active_tx.length=0;active_offset=0;}
}
void queueReply(const char* payload){
  if(reply_count==4){++dropped_serial_frames;return;}
  int n=executor::frame(payload,replies[reply_head].bytes,sizeof replies[0].bytes);
  if(n<0){++dropped_serial_frames;return;}
  replies[reply_head].length=n;reply_head=(reply_head+1)%4;++reply_count;
}
// Existing OK grammar keeps host parsers unchanged. INFO remains byte-for-byte
// compatible. Two pages share the latched tick; unsigned CONF -1 is 4294967295.
// No room means omit both diagnostic pages; a later stopped INFO can retry.
void queueEncoderFault(){
  if(!encoder_fault.valid || g_energized || reply_count>2)return;
  const auto &f=encoder_fault;const auto &t=f.transaction;
  char payload[249];
  int n=snprintf(payload,sizeof payload,
    "V2 OK %lu EncFault_v1_A_%lu_%lu_%u_%u_%u_%u_%u_%u_%lu_%lu_%u",
    (unsigned long)executor_state.sid,(unsigned long)f.tick,(unsigned long)f.at_us,
    (unsigned)f.mode,(unsigned)f.health_mask,(unsigned)t.reg,(unsigned)t.phase,
    (unsigned)t.result,(unsigned)t.expected,(unsigned long)t.duration_us,
    (unsigned long)t.msr,(unsigned)t.pins);
  if(n<0 || n>=(int)sizeof payload)return;
  queueReply(payload);
  n=snprintf(payload,sizeof payload,
    "V2 OK %lu EncFault_v1_B_%lu_%u_%lu_%lu_%lu_%lu_%lu_%lu_%lu_%u_%lu",
    (unsigned long)executor_state.sid,(unsigned long)f.tick,(unsigned)f.status,
    (unsigned long)uint32_t(f.conf),(unsigned long)f.status_age_us,(unsigned long)f.conf_age_us,
    (unsigned long)f.errors,(unsigned long)f.consecutive,(unsigned long)f.recoveries,
    (unsigned long)f.deadline_misses,(unsigned)f.timing_pending,(unsigned long)f.control_elapsed_us);
  if(n>=0 && n<(int)sizeof payload)queueReply(payload);
}
void emitTelemetry(){
  if(!executor_state.sid)return;
  char payload[249];const uint8_t raw=executor_state.limits;
  const unsigned flags=(executor_state.healthy?1:0)|(g_energized?2:0)|(executor_state.homed?4:0)|(executor_state.switches?8:0)|(unsigned(raw)<<4)|(executor_state.calibration?256:0);
  const int n=snprintf(payload,sizeof payload,"V2 S %lu %lu %lu %d %u %.7g %.7g %.7g %.7g %.7g %.7g %.7g %lu %u %lu %lu %d",
    (unsigned long)executor_state.sid,(unsigned long)executor_state.tick,(unsigned long)executor_state.acquisition_us,(int)mode,flags,
    (double)th,(double)th_hat,(double)thd,(double)xc,(double)vc,(double)acc_cmd,(double)executor_state.demand,
    (unsigned long)executor_state.last_seq,(unsigned)executor_state.fault,(unsigned long)control_deadline_misses,(unsigned long)dropped_serial_frames,cart_brake_direction);
  if(n<0 || n>=(int)sizeof payload){++dropped_serial_frames;return;}
  const int length=executor::frame(payload,latest_state.bytes,sizeof latest_state.bytes);
  if(length<0){++dropped_serial_frames;return;}
  if(latest_pending)++dropped_serial_frames;
  latest_state.length=length;latest_pending=true;
}
void serviceSerialTx(){
  if(!Serial)return;
  if(!active_tx.length){
    if(reply_count){active_tx=replies[reply_tail];reply_tail=(reply_tail+1)%4;--reply_count;}
    else if(latest_pending){active_tx=latest_state;latest_pending=false;}
    else return;
    active_offset=0;
  }
  const int room=Serial.availableForWrite();if(room<=0)return;
  size_t amount=active_tx.length-active_offset;if(amount>(size_t)room)amount=room;if(amount>64)amount=64;
  active_offset+=Serial.write((const uint8_t*)active_tx.bytes+active_offset,amount);
  if(active_offset==active_tx.length)active_tx.length=0;
}
void handleV2(const char* line){
  const uint8_t previous_mode=executor_state.mode;
  const uint32_t z_irq=interruptMask();__disable_irq();
  for(int i=0;i<2;++i)executor_state.z.count[i]=z_count[i];
  if(!z_irq)__enable_irq();
  executor::Reply reply=executor_state.command(line);
  __disable_irq();
  if(reply.kind==executor::OK&&!strcmp(reply.text,"ZREF"))for(int i=0;i<2;++i)z_count[i]=z_target[i]=executor_state.z.count[i];
  if(!executor_state.z.phase)z_active=0;
  if(!z_irq)__enable_irq();
  // Strict upright-only start gate is evaluated before drivers can be enabled.
  if(reply.kind==executor::OK && executor_state.enabled && executor_state.mode==M_BALANCE && previous_mode==M_IDLE &&
     !upright_session::canStart(th,th_hat,thd,xc,vc,acc_cmd,enc_ok && enc_consec_err==0)){
    executor_state.stop();reply={executor::ERROR,"upright"};
  }
  if(reply.kind==executor::OK){
    if(!strcmp(reply.text,"ZHOME")||!strcmp(reply.text,"ZMOVE")||!strcmp(reply.text,"ZSINE")){
      if(executor_state.mode==M_VERTICAL)mode=M_VERTICAL;
      const uint32_t irq=interruptMask();__disable_irq();
      for(int i=0;i<2;++i){z_target[i]=z_count[i];z_increment[i]=0;z_accumulator[i]=0;}
      energize(true);z_active=g_energized?executor_state.z.selected:0;
      if(!irq)__enable_irq();
    }
    if(!strcmp(reply.text,"ZSTOP")){cancelVertical();if(previous_mode==M_VERTICAL)eStop();}
    if(!strcmp(reply.text,"CALIBRATE")){
      calibration_sid=executor_state.sid;
      requested_hanging_offset=executor_state.calibration_offset;
      requested_model_capture=executor_state.calibration_model;
      calibration_request=executor_state.calibration;
      return; // No success response until stable capture and verified EEPROM write.
    }
    if(!strcmp(reply.text,"STOP") || !strcmp(reply.text,"CLEAR")){eStop();mode=executor_state.fault==executor::NONE?M_IDLE:M_FAULT;}
    else if(!strcmp(reply.text,"CENTER")){home_steps=ax_pos[AX_CART];p_rail=fminf(configured_rail,SAVED_RAIL_HALF);measured_home_span_steps=0;++home_epoch;xc=0;}
    else if(!strcmp(reply.text,"REHOME")){
      if(!encoderHealthy() || (executor_state.switches && executor::requiredLimits(rawLimits()))){
        executor_state.trip(!encoderHealthy()?executor::ENCODER:executor::LIMIT);eStop();mode=M_FAULT;reply={executor::ERROR,"health"};
      }else{beginSavedReturn();energize(true);}
    }
    else if(!strcmp(reply.text,"HOME")){
      if(!homing.start(executor_state.tick,ax_pos[AX_CART],executor::requiredLimits(rawLimits()))){executor_state.trip(executor::HOMING);reply={executor::ERROR,"homing"};}
      else{homing_edge_pending=0;homing_active=true;homing_expected=1;homing_approach=true;homing_verify=false;homing_centering=false;mode=M_HOMING;energize(true);}
    }else if(!strcmp(reply.text,"ARM") || !strcmp(reply.text,"MODE")){
      mode=(Mode)executor_state.mode;upright_only=mode==M_BALANCE;
      req_estimator_reset=mode==M_SWINGUP;pump.reset();response_watch.reset(th);
      if(mode!=M_SWINGUP)response_watch.armed=false;
      v_manual=vc=acc_cmd=0;cart_brake_direction=0;
      energize(true);
    }
  }
  if(reply.kind==executor::HELLO){resetTx();status_requested=false;eStop();mode=executor_state.fault==executor::NONE?M_IDLE:M_FAULT;}
  if(executor_state.enabled&&!g_energized){executor_state.trip(executor::DEADLINE);eStop();mode=M_FAULT;reply={executor::ERROR,"timer"};}
  if(reply.kind==executor::ZSTATE){
    char payload[160];snprintf(payload,sizeof payload,"V2 Z %lu %u %u %.6f %.6f %u %.6f",(unsigned long)executor_state.sid,
      executor_state.z.phase,executor_state.z.refs,z_count[0]/vertical::STEPS_MM,z_count[1]/vertical::STEPS_MM,executor_state.z.bottom,executor_state.z.max_mm);
    queueReply(payload);return;
  }
  if(reply.kind==executor::PARAM){
    const auto* field=controller_parameters::find(executor_state.parameter_name);
    if(!field)reply={executor::ERROR,"parameter"};
    else if(g_energized || encoderCalibrationBusy())reply={executor::ERROR,"health"};
    else if(executor_state.parameter_write){
      const float value=executor_state.parameter_value;
      auto candidate=controller;candidate.*(field->member)=value;
      const bool geometry=field->member==&swing_control::Parameters::rail_fraction ||
                          field->member==&swing_control::Parameters::rail_margin;
      float rail_cap=SAVED_RAIL_HALF;
      if(executor_state.homed && measured_home_span_steps>0)
        rail_cap=cart_homing::measuredRailCap(measured_home_span_steps,candidate.rail_fraction,candidate.rail_margin);
      if(geometry)candidate.rail=fminf(configured_rail,rail_cap);
      const bool changes_boundary=geometry || field->member==&swing_control::Parameters::rail ||
                                   field->member==&swing_control::Parameters::rail_fault_margin;
      const float stopped_x=(ax_pos[AX_CART]-home_steps)/CART_STEPS_PER_M;
      if((changes_boundary && executor_state.homed && fabsf(stopped_x)>candidate.rail-candidate.rail_fault_margin) ||
         !controller_parameters::valid(*field,value,candidate,rail_cap))reply={executor::ERROR,"value"};
      else{
        controller=candidate;
        if(field->member==&swing_control::Parameters::rail)configured_rail=value;
        executor_state.acceleration_limit=max(p_amax_s,p_amax_b);
        executor_state.manual_speed_limit=controller.manual_speed_max;
        eStop();computeGains();req_estimator_reset=true;
      }
    }
    if(reply.kind==executor::PARAM){
      char payload[100];
      snprintf(payload,sizeof payload,"V2 PARAM %lu %s %.9g",(unsigned long)executor_state.sid,
        field->name,(double)(controller.*(field->member)));
      queueReply(payload);return;
    }
  }
  if(reply.kind==executor::INFO){
    encConfigRead(); // INFO is stopped-only; return an actual register readback.
    executor_state.sample(encoderHealthy(),rawLimits());
    char payload[224];
    snprintf(payload,sizeof payload,"V2 INFO %lu executor-v7-vertical %ld %u %u %lu %lu %ld %.7g %ld %ld %u %ld %lu %lu",
      (unsigned long)executor_state.sid,(long)enc_upright_raw,(unsigned)reference_saved,(unsigned)enc_status,
      (unsigned long)enc_errors,(unsigned long)home_epoch,(long)measured_home_span_steps,(double)p_rail,
      (long)enc_boot_conf,(long)enc_conf,(unsigned)enc_conf_valid,(long)enc_last_raw,
      (unsigned long)(micros()-enc_raw_changed_us),(unsigned long)enc_recoveries);
    queueReply(payload);queueEncoderFault();return;
  }
  if(reply.kind==executor::STATE){status_requested=true;return;}
  if(reply.kind==executor::SILENT)return;
  char payload[100];
  if(reply.kind==executor::HELLO)snprintf(payload,sizeof payload,"V2 HELLO %lu 1 1000",(unsigned long)executor_state.sid);
  else snprintf(payload,sizeof payload,"V2 %s %lu %s",reply.kind==executor::ERROR?"ERR":"OK",(unsigned long)executor_state.sid,reply.text);
  queueReply(payload);
}
void pollSerial(){
  static executor::LineBuffer input;unsigned budget=96;
  while(budget-- && Serial.available()){
    const char* line=nullptr;if(input.feed(char(Serial.read()),line)){if(line)handleV2(line);else queueReply("V2 ERR 0 frame");return;}
  }
}

// ============================== SETUP =======================================
void setup() {
  // Preload HIGH before enabling output, before serial delays/I2C/zeroing.
  digitalWriteFast(PIN_EN_ALL, HIGH); // preload Teensy output latch, even while input
  pinMode(PIN_EN_ALL, OUTPUT);
  digitalWriteFast(PIN_EN_ALL, HIGH);
  g_energized = false;
  for(uint8_t pin=26;pin<=29;++pin)pinMode(pin,INPUT_PULLUP);
  status_display::begin(); // Blocking LCD init is only done with drivers disabled.
  Serial.begin(115200);
  delay(300);
  checkedPrintln(F("# Teensy 4.1 cart-pole swingup-125mm-60t-v33-teensy41"));

  // I2C comes up FIRST, while nothing else is competing for the CPU. Bringing
  // the 125 kHz step interrupt up first meant the very first bus transactions
  // happened under ~20% interrupt load, on a 5 ms timeout, with no retries.
  Wire1.setSDA(PIN_SDA);
  Wire1.setSCL(PIN_SCL);
  Wire1.begin();
  Wire1.setClock(I2C_HZ);
  delay(50);

  checkedPrintln(F("# scanning i2c..."));
  for (uint8_t a = 1; a < 127; a++) {
    Wire1.beginTransmission(a);
    if (Wire1.endTransmission() == 0)
      checkedPrintf("#   device at 0x%02X%s\n", a,
                    a == AS5600_ADDR ? "  <- AS5600" :
                    a == 0x06        ? "  <- MT6701, different chip" : "");
    delay(1);
  }

  encDiagRead();
  enc_last_raw = encReadRaw();
  enc_ok = (enc_last_raw >= 0);
  if (!enc_ok) { checkedPrintln(F("! encoder not answering - check SDA=17 SCL=16, "
                                  "3V3, and DIR tied to GND")); enc_last_raw = 0; }
  encPrintMagnet();
  enc_boot_conf=as5600ReadConf(); // Capture before correcting a retained bad mode.
  enc_raw_changed_us=micros();
  if (as5600FastFilter()) checkedPrintln(F("# AS5600 normal power, watchdog off, 2x filter verified"));
  else                    checkedPrintln(F("! AS5600 CONF write failed"));
  enc_ticks = 0;

  steppersInit();

  // Exact saved working parameter set. Historical encoder calibration is excluded.
  controller.rail=0.26f;
  configured_rail=cart_homing::maximum_rail_half;
  controller.leff=0.124f;
  controller.pw=4.5f;
  controller.pz=0.85f;
  controller.pc1=-3.0f;
  controller.pc2=-4.0f;
  controller.bw=30.0f;
  controller.vmax=1.5f;
  controller.vmax_s=1.5f;
  controller.vmax_b=1.5f;
  controller.approach_v=0.3f;
  controller.catch_v=0.3f;
  controller.approach_angle=0.8f;
  controller.catch_da=4.0f;
  controller.amax_s=25.0f;
  controller.amax_b=25.0f;
  controller.jmax=150.0f;
  controller.ke=4.0f;
  controller.kpx=40.0f;
  controller.kdx=2.0f;
  controller.phase_soft=2.0f;
  controller.catch_a=0.6f;
  controller.catch_r=3.0f;
  controller.giveup=0.8f;
  controller.spin_trip_rad_s=150.0f;
  controller.spin_resume_rad_s=10.0f;
  controller.amax_m=0.5f;
  controller.jmax_m=10.0f;
  controller.vman=0.05f;
  controller.bal_pw=4.0f;
  controller.manual_speed_max=.15f;
  controller.rail_fraction=.95f;
  controller.rail_margin=.015f;
  controller.rail_fault_margin=.010f;
  controller.rail_hysteresis=.005f;
  controller.rail_recover_v=.15f;
  controller.rail_recover_a=1.5f;
  controller.rail_recover_s=8.f;
  controller.rehome_v=.08f;
  controller.rehome_a=.5f;
  controller.rehome_j=10.f;
  executor_state.acceleration_limit=max(p_amax_s,p_amax_b);
  executor_state.manual_speed_limit=controller.manual_speed_max;
  p_zvmax=2.0f;
  p_zamax=20.0f;
  p_ztravel=100.0f;
  computeGains();
  checkedPrintf("# cart %.1f steps/mm (1/%d), Z %.1f steps/mm\n",
                CART_STEPS_PER_M/1000.0f, CART_MICROSTEPS, Z_STEPS_PER_MM);
  checkedPrintf("# cart speed ceiling %.2f m/s\n", VMAX_CEIL);
  delay(300);
  loadEncoderReference();
  enc_zero=enc_ticks+encoder_reference::difference(encoder_reference::opposite(enc_upright_raw),enc_last_raw);
  checkedPrintf("# upright reference %ld counts (%s); startup retains it\n",(long)enc_upright_raw,reference_saved?"saved":"fallback; use upright or hanging to calibrate");
  th = th_hat = encTheta();
  home_steps = ax_pos[AX_CART];
  zhome_steps = ax_pos[AX_Z1];
  checkedPrintf("# K = [%.3f %.3f %.3f %.3f]\n", K1,K2,K3,K4);
  checkedPrintf("# Cart defaults: leff=%.3f m, vmax=%.3f m/s, amax_s=%.1f, amax_b=%.1f m/s^2, jmax=%.1f m/s^3\n",
                p_leff, p_vmax, p_amax_s, p_amax_b, p_jmax);
  checkedPrintln(F("# Startup position is x=0: place cart at physical center before reset; no homing motion."));
  checkedPrintln(F("# Travel 300 mm; normal range +/-135 mm. Predictive braking -> return inside range; fault +/-140 mm."));
  checkedPrintf("# Pendulum |theta_dot| >%.2f rad/s: brake, center, then resume below %.2f rad/s; no full-turn check or dwell.\n",controller.spin_trip_rad_s,controller.spin_resume_rad_s);
  checkedPrintln(F("# bal: explicit upright-only start within 10deg; >50deg fall brakes, centers and disables; no restart."));
  checkedPrintln(F("# Drivers DISABLED. Motion commands enable; stop disables all drivers."));
  checkedPrintln(F("# ready. 'help' for commands."));

  encStatusRead();encConfigRead();
  executor_state.sample(encoderHealthy(),rawLimits());
  control_heartbeat_us=micros();
}

// Run one foreground control tick at 1 kHz. Never execute catch-up bursts
// with the same sensor sample; a whole missed period during motion is a fault.
static uint32_t next_control_us=0;
void serviceControl(){
  const uint32_t now=micros();
  if(!next_control_us)next_control_us=now;
  if(limit_fault_pending){executor_state.trip(executor::LIMIT);eStop();mode=M_FAULT;limit_fault_pending=false;}
  if(timing_fault_pending){
    executor_state.trip(executor::DEADLINE);eStop();mode=M_FAULT;timing_fault_pending=false;
    controlLog("! control stalled >5ms -> FAULT; drivers disabled");
  }
  if(int32_t(now-next_control_us)<0)return;
  if(uint32_t(now-next_control_us)>=1000000UL/CTRL_HZ){
    ++control_deadline_misses;
    if(g_energized){executor_state.trip(executor::DEADLINE);eStop();mode=M_FAULT;controlLog("! missed control deadline -> FAULT; drivers disabled");}
    next_control_us=now;
  }
  next_control_us+=1000000UL/CTRL_HZ;
  controlTick();
  if(mode==M_FAULT && executor_state.fault==executor::NONE)executor_state.trip(executor::CONTROLLER);
  if(executor_state.fault!=executor::NONE)mode=M_FAULT;
  executor_state.mode=(uint8_t)mode;
  control_heartbeat_us=micros();
}
// Display work runs only in foreground, one small operation per call.
void serviceDisplay(){
  if(int32_t(next_control_us-micros())<600)return;
  static uint32_t last_snapshot=0;
  if(uint32_t(millis()-last_snapshot)>=200){
    last_snapshot=millis();
    const bool linked=executor_state.sid && uint32_t(executor_state.tick-executor_state.host_at)<200;
    const char *label=MODE_NAME[mode];
    uint16_t color=0xFFFF;
    if(mode==M_FAULT){label="FAULT";color=0xF800;}
    else if(encoderCalibrationBusy()){label="CALIBRATING";color=0xFFE0;}
    else if(mode==M_IDLE && !linked){label="DISCONNECTED";color=0x8410;}
    else if(mode==M_SWINGUP){label="SWINGING UP";color=0xFFE0;}
    else if(mode==M_BALANCE){label="BALANCING";color=0x07E0;}
    status_display::snapshot(label,color,linked,g_energized,encoderHealthy(),
      reference_saved,th*180.0f/M_PI,xc*1000.0f,vc,mode==M_FAULT?display_fault:nullptr);
    return;
  }
  status_display::service();
}

void loop(){
  serviceControl();pollSerial();serviceControl();
  saveEncoderReference(); // Disabled-only persistence, outside the control tick.
  const uint32_t tick=executor_state.tick;
  if(status_requested || (executor_state.rate && uint32_t(tick-last_emitted_tick)>=CTRL_HZ/executor_state.rate)){
    status_requested=false;last_emitted_tick=tick;emitTelemetry();
  }
  serviceControl();serviceSerialTx();serviceControl();serviceDisplay();
}
