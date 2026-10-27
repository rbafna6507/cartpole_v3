#pragma once
#include <cstddef>
#include <cstdint>

// CRC-16/CCITT-FALSE: polynomial 0x1021, initial value 0xffff.
// Protects dropped/changed bytes, including plausible-looking numeric corruption.
namespace serial_frame {
inline uint16_t checksum(const char *data, size_t length) {
  uint16_t crc=0xffff;
  for(size_t n=0;n<length;++n){
    crc ^= uint16_t(uint8_t(data[n])) << 8;
    for(int bit=0;bit<8;++bit)crc=(crc&0x8000)?uint16_t((crc<<1)^0x1021):uint16_t(crc<<1);
  }
  return crc;
}
}
