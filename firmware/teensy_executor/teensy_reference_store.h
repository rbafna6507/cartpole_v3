#pragma once
#include <EEPROM.h>
#include <stdint.h>
// Owns EEPROM bytes 0..7. Magic is committed last so incomplete writes fail.
// Calibration is only saved with every driver disabled.
namespace teensy_reference_store {
constexpr uint32_t magic=0x54435031; // TCP1
inline bool load(uint16_t &raw){
  uint32_t tag;uint16_t inverse;
  EEPROM.get(0,tag);EEPROM.get(4,raw);EEPROM.get(6,inverse);
  return tag==magic && raw<4096 && inverse==(uint16_t)~raw;
}
inline bool save(uint16_t raw){
  if(raw>=4096)return false;
  uint16_t previous;
  if(load(previous) && previous==raw)return true;
  const uint32_t invalid=0;
  const uint16_t inverse=(uint16_t)~raw;
  EEPROM.put(0,invalid);EEPROM.put(4,raw);EEPROM.put(6,inverse);EEPROM.put(0,magic);
  uint16_t check;return load(check) && check==raw;
}
}
