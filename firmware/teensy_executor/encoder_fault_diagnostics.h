#pragma once
#include <cstdint>

// Post-INFO OK token pages (decimal fields, underscore separated):
// A: tick, at_us, mode, health_mask, reg, phase, result, expected,
//    transaction duration_us, raw LPI2C3 MSR, pins (SDA=1, SCL=2).
// B: tick, status, uint32 CONF (-1 => 4294967295), status_age_us,
//    conf_age_us, errors, consecutive, recoveries, deadline_misses,
//    timing_pending, control_elapsed_us. Matching tick joins the pages.
// The host records these as ordinary OK replies; no parser extension needed.
// Diagnostic-only state: never used to authorize motion or retry a transaction.
namespace encoder_diagnostics {
enum Health : uint16_t {
  ANGLE=1, CONSECUTIVE=2, CONF_INVALID=4, CONF_STALE=8,
  STATUS_INVALID=16, STATUS_STALE=32, MAGNET=64
};
struct Transaction {
  uint8_t reg=0, phase=0, result=0, expected=0, pins=0;
  uint32_t duration_us=0, msr=0;
};
struct Snapshot {
  bool valid=false, timing_pending=false;
  uint8_t mode=0, status=0;
  uint16_t health_mask=0;
  uint32_t tick=0, at_us=0, status_age_us=0, conf_age_us=0;
  uint32_t errors=0, consecutive=0, recoveries=0, deadline_misses=0;
  uint32_t control_elapsed_us=0;
  int32_t conf=-1;
  Transaction transaction;
};
}
