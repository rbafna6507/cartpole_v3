// Standalone, operator-attended Z commissioning. NOT the cartpole executor.
// No cart STEP pulses. No encoder, cart balancing, or physical Z feedback.
#include <Arduino.h>
#include <IntervalTimer.h>
#include "vertical_profile.h"
#include <cstdlib>
#include <cstring>
#if !defined(ARDUINO_TEENSY41)
#error "Select Teensy 4.1, USB Serial"
#endif
constexpr uint8_t STEP[2]={4,6}, DIR[2]={5,7}, ENABLE=8, CART_STEP=2;
constexpr uint32_t TICK_US=4, HEARTBEAT_US=500000, LOOP_US=5000;
IntervalTimer pulseTimer;
vertical::Settings settings;
enum Mode { IDLE, PROBE, BOTTOM, MOVE, SINE };
Mode mode=IDLE;
volatile int32_t position[2]={0,0}, target[2]={0,0};
volatile uint32_t increment[2]={0,0}, accumulator[2]={0,0};
volatile int8_t direction[2]={1,1};
volatile bool highStep[2]={false,false}, running=false;
volatile uint8_t fault=0; // 1 host heartbeat, 2 foreground stall, 3 pulse spacing, 4 tracking
volatile uint32_t lastPing=0,lastControl=0,hardEnd=0;
bool timerReady=false,energized=false,referenced=false,bottomComplete=false;
bool probed[2]={false,false},confirmed[2]={false,false};
int8_t upSign[2]={1,1},pinSign[2]={0,0};
bool configured=false;
uint32_t began=0,lastUpdate=0;
double duration=0,startMm[2]={0,0},endMm[2]={0,0};
double amplitude=20,hz=1,center=25,sineSeconds=10,bottomSpeed=40;
int probeAxis=0;

void inhibit(uint8_t reason) {
    increment[0]=increment[1]=0;running=false;
    if(reason)fault=reason;
}
void pulses() {
    bool previous[2]={highStep[0],highStep[1]};
    digitalWriteFast(4,LOW);digitalWriteFast(6,LOW);
    highStep[0]=highStep[1]=false;
    if(!running)return;
    uint32_t now=micros();
    if(uint32_t(now-lastPing)>HEARTBEAT_US){inhibit(1);return;}
    if(uint32_t(now-lastControl)>LOOP_US){inhibit(2);return;}
    if(int32_t(now-hardEnd)>=0){inhibit(4);return;}
    for(int i=0;i<2;i++) {
        if(position[i]==target[i])continue;
        uint32_t old=accumulator[i];accumulator[i]+=increment[i];
        if(accumulator[i]<old) {
            if(previous[i]){inhibit(3);return;}
            position[i]+=direction[i];highStep[i]=true;
            if(i==0)digitalWriteFast(4,HIGH);else digitalWriteFast(6,HIGH);
        }
    }
}
void halt(bool invalidate=true) {
    noInterrupts();inhibit(0);interrupts();
    mode=IDLE;
    if(invalidate){referenced=false;bottomComplete=false;}
    // Keep ENABLE unchanged: STOP is hold, never an implicit gravity release.
}
void powerOn() {
    if(!energized){digitalWriteFast(ENABLE,LOW);energized=true;delayMicroseconds(1000);}
}
void status() {
    int32_t p[2];noInterrupts();p[0]=position[0];p[1]=position[1];interrupts();
    Serial.printf("STATUS mode=%d fault=%u enabled=%d reference=%d bottom_complete=%d directions=%d,%d confirmed=%d,%d counted_mm=%.4f,%.4f\n",
      int(mode),fault,energized,referenced,bottomComplete,upSign[0],upSign[1],confirmed[0],confirmed[1],p[0]/settings.stepsPerMm(),p[1]/settings.stepsPerMm());
}
bool number(const char*s,double&v){char*e;v=strtod(s,&e);return *s && !*e && std::isfinite(v);}
void start(Mode m,double seconds) {
    powerOn();began=micros();lastControl=began;lastUpdate=began;
    duration=seconds;mode=m;bottomComplete=false;
    // 20 ms settle allowance is pulse-side bounded, even if foreground wedges.
    noInterrupts();hardEnd=began+uint32_t((seconds+.020)*1e6);running=true;interrupts();
}
void setTargets(double left,double right) {
    double values[2]={left,right};
    for(int i=0;i<2;i++) {
        int32_t goal=int32_t(llround(values[i]*settings.stepsPerMm()));
        int32_t pos;noInterrupts();pos=position[i];interrupts();
        int32_t delta=goal-pos;
        if(std::abs(delta)>200){noInterrupts();inhibit(4);interrupts();return;}
        int8_t d=delta>=0?1:-1;
        int8_t physical=d*(mode==PROBE?1:upSign[i]);
        if(delta && pinSign[i]!=physical) {
            noInterrupts();increment[i]=0;interrupts();
            delayMicroseconds(TICK_US+1); // Finish preceding pulse before DIR.
            digitalWrite(DIR[i],physical>0?HIGH:LOW);pinSign[i]=physical;
            delayMicroseconds(2);
            noInterrupts();accumulator[i]=0;interrupts();
        }
        // Slight catch-up margin; reject sustained lag above 200 counted pulses.
        double rate=std::min(vertical::MAX_RATE,std::abs(double(delta))*1100.);
        uint32_t inc=uint32_t(rate*4294967296.0/(1e6/TICK_US));
        noInterrupts();
        if(!fault && running){direction[i]=d;target[i]=goal;increment[i]=inc;}
        interrupts();
    }
}
void updateMotion() {
    uint32_t now=micros();lastControl=now;
    if(fault){if(mode!=IDLE){halt();Serial.printf("FAULT %u; pulses stopped, holding torque retained\n",fault);}return;}
    if(mode==IDLE || uint32_t(now-lastUpdate)<1000)return;
    lastUpdate=now;
    double t=uint32_t(now-began)*1e-6;
    double z[2];
    if(mode==PROBE || mode==MOVE) {
        for(int i=0;i<2;i++)z[i]=vertical::movePosition(t/duration,startMm[i],endMm[i]);
    } else if(mode==BOTTOM) {
        for(int i=0;i<2;i++)z[i]=startMm[i]-bottomSpeed*std::min(t,duration);
    } else {
        z[0]=z[1]=vertical::sinePosition(t,amplitude,hz,center,duration);
    }
    if(mode==BOTTOM && t>=duration) {
        halt();bottomComplete=true;
        Serial.println("DONE bottoming; contact NOT sensed. Inspect both bottoms, then reference CONFIRM.");return;
    }
    setTargets(z[0],z[1]);
    if(t>=duration && !fault) {
        bool arrived;noInterrupts();arrived=position[0]==target[0] && position[1]==target[1];interrupts();
        if(arrived) {
            Mode finished=mode;halt(false);
            if(finished==PROBE){probed[probeAxis]=true;Serial.println("DONE probe; observe physical direction before confirming.");}
            else if(finished==MOVE){start(SINE,sineSeconds);Serial.println("RUN sine");}
            else Serial.println("DONE sine; holding at center");
        }
    }
}
void command(char*line) {
    char*argv[10];int argc=0;char*save=nullptr;
    for(char*p=strtok_r(line," \t",&save);p;p=strtok_r(nullptr," \t",&save)) {
        if(argc==10){Serial.println("ERR too many arguments");return;}argv[argc++]=p;
    }
    if(!argc)return;
    if(!strcmp(argv[0],"ping") && argc==1){lastPing=micros();return;}
    if(!strcmp(argv[0],"hello") && argc==1){Serial.println("VERTICAL-COMMISSIONING-1");return;}
    if(!strcmp(argv[0],"status") && argc==1){status();return;}
    if(!strcmp(argv[0],"stop") && argc==1){halt();Serial.println("OK stopped; holding torque retained; reference invalidated");return;}
    if(!strcmp(argv[0],"disable") && argc==2 && !strcmp(argv[1],"SUPPORTED")) {
        halt();digitalWriteFast(ENABLE,HIGH);energized=false;Serial.println("OK shared drivers disabled");return;
    }
    if(mode!=IDLE){Serial.println("ERR busy; stop first");return;}
    if(!strcmp(argv[0],"clear") && argc==1) {
        halt();noInterrupts();fault=0;interrupts();Serial.println("OK fault acknowledged; new reference required");return;
    }
    if(fault || !timerReady){Serial.println("ERR fault or unavailable timer");return;}
    if(!strcmp(argv[0],"configure") && argc==6) {
        double v[5];for(int i=0;i<5;i++)if(!number(argv[i+1],v[i])){Serial.println("ERR number");return;}
        vertical::Settings s;s.travel=v[0];s.lead=v[1];
        if((v[2]!=200 && v[2]!=400) || (v[3]!=1 && v[3]!=2 && v[3]!=4 && v[3]!=8 && v[3]!=16) || (v[4]!=0 && v[4]!=1)){Serial.println("ERR settings");return;}
        s.motorSteps=int(v[2]);s.microsteps=int(v[3]);s.bottomRated=bool(v[4]);
        if(!s.valid()){Serial.println("ERR settings");return;}
        settings=s;configured=true;referenced=false;bottomComplete=false;
        confirmed[0]=confirmed[1]=probed[0]=probed[1]=false;
        Serial.printf("OK configured %.3f pulses/mm; open-loop, no contact sensors\n",s.stepsPerMm());return;
    }
    if(!configured || uint32_t(micros()-lastPing)>HEARTBEAT_US){Serial.println("ERR configuration or heartbeat required");return;}
    if(!strcmp(argv[0],"probe") && (argc==3 || argc==4)) {
        int axis=!strcmp(argv[1],"left")?0:(!strcmp(argv[1],"right")?1:-1);
        int sign=!strcmp(argv[2],"+")?1:(!strcmp(argv[2],"-")?-1:0);
        if(axis<0 || !sign){Serial.println("ERR probe left|right +|-");return;}
        double distance=.5;
        if((argc==4 && !number(argv[3],distance)) || distance<=0 || distance>20){Serial.println("ERR probe distance must be >0 and <=20 mm");return;}
        referenced=false;confirmed[axis]=false;probeAxis=axis;
        for(int i=0;i<2;i++)startMm[i]=endMm[i]=position[i]/settings.stepsPerMm();
        endMm[axis]+=distance*sign;
        // Quintic peak derivative 1.875: keep peak speed <=2 mm/s.
        start(PROBE,std::max(.5,1.875*distance/2.));
        Serial.printf("RUN raw %.4f mm direction probe; max 2 mm/s\n",distance);return;
    }
    if(!strcmp(argv[0],"direction") && argc==3) {
        int axis=!strcmp(argv[1],"left")?0:(!strcmp(argv[1],"right")?1:-1);
        int sign=!strcmp(argv[2],"+")?1:(!strcmp(argv[2],"-")?-1:0);
        if(axis<0 || !sign || !probed[axis]){Serial.println("ERR complete probe first; direction left|right +|-");return;}
        upSign[axis]=sign;confirmed[axis]=true;referenced=false;
        Serial.println("OK operator-confirmed UP polarity");return;
    }
    if(!confirmed[0] || !confirmed[1]){Serial.println("ERR confirm both motor directions first");return;}
    if(!strcmp(argv[0],"bottom") && argc==4 && !strcmp(argv[3],"CONTACT_OK")) {
        double speed,seconds;
        if(!number(argv[1],speed)||!number(argv[2],seconds)||!vertical::validBottom(settings,speed,seconds)){Serial.println("ERR bottoming capability/speed/time");return;}
        for(int i=0;i<2;i++)startMm[i]=position[i]/settings.stepsPerMm();
        referenced=false;bottomSpeed=speed;start(BOTTOM,seconds);Serial.println("RUN timed bottoming; no contact detection");return;
    }
    if(!strcmp(argv[0],"reference") && argc==2 && !strcmp(argv[1],"CONFIRM")) {
        noInterrupts();position[0]=position[1]=target[0]=target[1]=0;accumulator[0]=accumulator[1]=0;interrupts();
        referenced=true;bottomComplete=false;Serial.println("OK operator-declared bottom reference; NOT sensor verified");return;
    }
    if(!strcmp(argv[0],"sine") && argc==5) {
        double v[4];for(int i=0;i<4;i++)if(!number(argv[i+1],v[i])){Serial.println("ERR number");return;}
        if(!referenced || !vertical::validSine(settings,v[0],v[1],v[2],v[3])){Serial.println("ERR reference/travel/rate/duration bounds");return;}
        amplitude=v[0];hz=v[1];center=v[2];sineSeconds=v[3];
        for(int i=0;i<2;i++)startMm[i]=position[i]/settings.stepsPerMm(),endMm[i]=center;
        double seconds=std::max(vertical::moveDuration(startMm[0],center),vertical::moveDuration(startMm[1],center));
        start(MOVE,seconds);Serial.println("RUN smooth transfer to sine center");return;
    }
    Serial.println("ERR command; use host help");
}
void setup() {
    // Establish disabled shared enable before any other output.
    digitalWriteFast(ENABLE,HIGH);pinMode(ENABLE,OUTPUT);
    for(uint8_t p:{uint8_t(2),uint8_t(3),uint8_t(4),uint8_t(5),uint8_t(6),uint8_t(7)}){digitalWrite(p,LOW);pinMode(p,OUTPUT);}
    Serial.begin(115200);lastControl=micros();
    timerReady=pulseTimer.begin(pulses,TICK_US);pulseTimer.priority(32);
}
void loop() {
    updateMotion();
    static char line[160];static size_t used=0;static bool overflow=false;
    // Bounded serial work prevents a flood from starving motion/watchdog updates.
    for(int n=0;n<32 && Serial.available();n++) {
        int c=Serial.read();
        if(c=='\r')continue;
        if(c=='\n') {
            if(overflow){halt();Serial.println("ERR line too long; stopped");}
            else {line[used]=0;command(line);}
            used=0;overflow=false;
        } else if(c<32 || c>126){overflow=true;}
        else if(used<sizeof(line)-1)line[used++]=char(c);else overflow=true;
    }
}
