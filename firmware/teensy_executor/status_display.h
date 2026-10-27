#pragma once
#include <SPI.h>
#include <ST7735_t3.h>
#include <ST7789_t3.h>
#include <cstdio>
#include <cstring>

// SK Pang / Copperhill 240x240 IPS LCD carrier vendor demo pinout.
// LCD SPI and backlight pins must not be used for external peripherals.
namespace status_display {
static ST7789_t3 lcd(10,9,11,13,32); // CS, DC, MOSI, SCLK, RESET
constexpr unsigned rows=10, cols=20;
static char desired[rows][cols], shown[rows][cols];
static uint16_t colors[rows], shown_colors[rows][cols];
static unsigned cursor=0;
static void line(unsigned r,const char *text,uint16_t color=0xFFFF){
  const size_t n=strlen(text);
  for(unsigned c=0;c<cols;++c)desired[r][c]=c<n?text[c]:' ';
  colors[r]=color;
}
static void begin(){
  pinMode(33,OUTPUT);digitalWrite(33,LOW);
  lcd.init(240,240);lcd.setRotation(0);lcd.fillScreen(0);
  lcd.setTextSize(2);lcd.setTextColor(0xFFFF,0);lcd.setCursor(0,0);
  lcd.print("CARTPOLE\n\nBOOTING");
  memset(shown,0,sizeof(shown));memset(desired,' ',sizeof(desired));
  digitalWrite(33,HIGH);
}
static void snapshot(const char *state,uint16_t color,bool host,bool enabled,
                     bool encoder,bool saved,float degrees,float mm,float speed,
                     const char *fault){
  char b[64];
  line(0,"CARTPOLE / TEENSY 4.1",0x07FF);
  line(1,state,color);
  line(2,host?"HOST: CONNECTED":"HOST: DISCONNECTED");
  line(3,enabled?"DRIVERS: ENABLED":"DRIVERS: DISABLED",enabled?0xFFE0:0xFFFF);
  line(4,encoder?"ENCODER: OK":"ENCODER: NO DATA",encoder?0x07E0:0xF800);
  snprintf(b,sizeof(b),"ANGLE: %7.1f deg",degrees);line(5,encoder?b:"ANGLE: unavailable");
  snprintf(b,sizeof(b),"CART:  %7.1f mm",mm);line(6,b);
  snprintf(b,sizeof(b),"SPEED: %6.2f m/s",speed);line(7,b);
  if(fault){
    char part[21];snprintf(part,sizeof(part),"%.20s",fault);line(8,part,0xF800);
    line(9,strlen(fault)>20?fault+20:"See serial fault log",0xF800);
  }else{
    line(8,saved?"REFERENCE: SAVED":"REF: NEEDS CALIB",saved?0xFFFF:0xFFE0);
    line(9,"USB commands control",0x8410);
  }
}
static void service(){
  // One opaque 12x16 glyph at most; never clear/redraw the entire LCD in motion.
  // Scan at most one row of unchanged cells per invocation as well.
  for(unsigned n=0;n<cols;++n){
    const unsigned r=cursor/cols,c=cursor%cols;
    cursor=(cursor+1)%(rows*cols);
    if(shown[r][c]==desired[r][c] && shown_colors[r][c]==colors[r])continue;
    lcd.drawChar(c*12,r*16,desired[r][c],colors[r],0,2);
    shown[r][c]=desired[r][c];shown_colors[r][c]=colors[r];
    return;
  }
}
}
