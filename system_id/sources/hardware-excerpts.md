# Historical hardware source excerpts

These are excerpts from the existing v2 source, inspected 2026-10-27. They establish documented configuration and owner reports, not a fresh physical inspection. Source hashes bind the original complete files.

## firmware/teensy_executor/teensy_executor.ino:27–42

SHA-256: `df21ecdc8643d77b77b59b96a2a0df7a987d6ffc44f1dc489f5b693a66554229`

```text
27: // ============================== PINS ========================================
28: #define PIN_CART_STEP   2
29: #define PIN_CART_DIR    3
30: #define PIN_Z1_STEP     4
31: #define PIN_Z1_DIR      5
32: #define PIN_Z2_STEP     6
33: #define PIN_Z2_DIR      7
34: #define PIN_EN_ALL      8        // shared ENABLE, active LOW
35: #define PIN_SDA         17
36: #define PIN_SCL         16
37: #define I2C_HZ          400000    // drop to 100000 if the cable to the cart is
38:                                   // long or you see enc_err climbing
39: 
40: #define CART_INVERT     0         // v19: reverse cart polarity for upright trial; encoder sign unchanged
41: #define Z_INVERT        0         // flip if 'zv 5' lowers instead of raises
42: #define ENC_INVERT      0         // flip if theta goes negative when the pole
```

## firmware/teensy_executor/teensy_executor.ino:256–263

SHA-256: `df21ecdc8643d77b77b59b96a2a0df7a987d6ffc44f1dc489f5b693a66554229`

```text
256: // ============================== AS5600 ======================================
257: #define AS5600_ADDR   0x36
258: #define REG_STATUS    0x0B
259: #define REG_RAWANGLE  0x0C
260: #define REG_CONF_HI   0x07
261: #define REG_AGC       0x1A
262: #define ENC_CPR       4096
263: 
```

## firmware/teensy_executor/cart_mechanics.h:1–23

SHA-256: `04d5d54363baaaeb0e1df9827e06ef3c2a4be3fcb9b967e8a2c48d6b94e442b6`

```text
1: #pragma once
2: 
3: // Shared pulse/drive geometry. This ceiling describes pulse generation,
4: // not the motor's measured torque or ability to follow the commands.
5: namespace cart_hardware {
6: constexpr int motor_steps = 200, microsteps = 16, pulley_teeth = 60;
7: constexpr float belt_pitch_mm = 2.0f;
8: // Use one exact integer-microsecond timer period everywhere, including DDS.
9: constexpr unsigned step_timer_period_us = 7;
10: constexpr double isr_hz = 1000000.0 / step_timer_period_us;
11: constexpr float mm_per_rev = pulley_teeth * belt_pitch_mm;
12: // Linear limits stay fixed when pulley geometry changes (60T, 2026-09-17).
13: constexpr float default_speed = 1.5f;         // m/s
14: constexpr float default_acceleration = 25.0f; // m/s^2
15: constexpr float default_jerk = 100.0f;         // m/s^3
16: constexpr float requested_speed_rpm = default_speed * 60000.0f / mm_per_rev;
17: constexpr float requested_acceleration_rpm_s = default_acceleration * 60000.0f / mm_per_rev;
18: constexpr float requested_jerk_rpm_s2 = default_jerk * 60000.0f / mm_per_rev;
19: constexpr float steps_per_m = motor_steps * microsteps * 1000.0f / mm_per_rev;
20: constexpr float max_speed = (isr_hz * 0.5f) * mm_per_rev /
21:                             (motor_steps * microsteps * 1000.0f);
22: constexpr float max_rpm = isr_hz * 0.5f * 60.0f / (motor_steps * microsteps);
23: }
```

## docs/pcb-switches.md:1–35

SHA-256: `b3293e1e1fd644a1726c6b5457507f41b3fa35009d9f0778f750ce33023694c7`

```text
1: # PCB switch wiring and bounded homing
2: 
3: Electrical inspection performed 2026-10-13 against the current original design, read-only. Source hashes, net nodes, pad UUIDs, routed segment UUIDs, and Gerber comparisons are recorded in `evidence/phase3/pcb-switches.json`. This establishes design intent and connectivity; it does not establish which revision was assembled or commission the hardware. On 2026-10-14 the owner confirmed switches 1 and 2 are installed, with 1 at the left and 2 at the right. Switches 3 and 4 are not installed. Electrical polarity, disconnect response and homing direction still need physical commissioning.
4: 
5: ## Verified wiring
6: 
7: | Switch label | PCB connector | Signal pad | Teensy GPIO / carrier pad | Parallel screw-terminal signal | Pull-up |
8: |---|---|---|---|---|---|
9: | limit_switch_1 | J8 | 1 | D26 / U3 J6_4 | J5 pin 9 | R5, 1 kΩ to 3.3 V |
10: | limit_switch_2 | J11 | 1 | D27 / U3 J6_3 | J5 pin 10 | R6, 1 kΩ to 3.3 V |
11: | limit_switch_3 | J10 | 1 | D28 / U3 J6_2 | J5 pin 11 | R7, 1 kΩ to 3.3 V |
12: | limit_switch_4 | J9 | 1 | D29 / U3 J6_1 | J5 pin 12 | R8, 1 kΩ to 3.3 V |
13: 
14: For every J8–J11 connector, solder pad **1 (square, SIG) to the switch NC terminal** and pad **2 (round, GND) to COM**. Leave NO unused. The footprint description explicitly states this wiring. Each signal is independently routed; these are not a series chain. J5 provides parallel access to the same signals, with ground at J5 pin 1; J5 pin 2 carries +12 V and is not a switch supply. Use connector/pad numbers when working from the underside because the view reverses.
15: 
16: The pull-ups connect to U3 J7_2, the carrier's 3.3 V output, and not to 5 V or motor power. A closed NC contact reads LOW (healthy, unpressed), and an actuated NC contact reads HIGH (opened). An unplugged/broken wire also reads HIGH. Nominal closed-contact current is 3.3 mA per input. A short to ground cannot be distinguished from a healthy closed contact; homing must require observed open and close transitions to catch a stuck-low channel. The PCB contains no series input resistor, capacitor, RC debounce, optoisolator, or added clamp on these channels. Configure GPIOs as inputs, never outputs; internal pull-up can supplement the fitted external pull-up but must not substitute for confirming R5–R8 are populated.
17: 
18: [The PJRC reference card](https://www.pjrc.com/teensy/card11b_rev3_web.pdf) confirms Teensy 4.1 GPIOs are not 5 V tolerant. These connectors are for dry contacts to ground; an externally powered switch output must meet the 3.3 V input requirement.
19: 
20: ## Driver enable and pin reservations
21: 
22: GPIO D8 (U3 J7_10 / J4 pin 10) is a single direct copper net connected to Enable pin 1 on cart U2, left U4, and right U5. There is no AND/OR gate, switch-to-enable connection, or independent per-driver enable on this board. A limit opening does not directly stop a driver: firmware must suppress all step pulses and raise D8. The Big Easy Driver enable convention is LOW enabled, HIGH disabled, consistent with the archived working firmware and [the SparkFun hardware guide](https://learn.sparkfun.com/tutorials/big-easy-driver-hookup-guide/hardware-overview). The carrier board adds no pull-up/down to D8. Driver module logic jumpers and module-local bias are outside this PCB and were not physically inspected; reset/high-impedance D8 is therefore not proven fail-safe. Firmware must establish disabled D8 before enabling step outputs. Shared disable de-energizes the Z drivers too; mechanical holding/support cannot be inferred from this schematic.
23: 
24: | Function | Reserved Teensy GPIOs |
25: |---|---|
26: | Cart step / direction | 2 / 3 |
27: | Left Z1 step / direction | 4 / 5 |
28: | Right Z2 step / direction | 6 / 7 |
29: | Shared driver enable | 8 |
30: | Magnetic encoder SDA / SCL | 17 / 16 |
31: | MPU6050 SDA / SCL / interrupt | 25 / 24 / 38 |
32: | Carrier LCD CS / DC / MOSI / SCLK / reset / backlight | 10 / 9 / 11 / 13 / 32 / 33 |
33: | Switch inputs | 26 / 27 / 28 / 29 |
34: 
35: Switch GPIOs do not overlap these functions. Carrier J6_* and J7_* identifiers are PCB pad identifiers; they are not Arduino GPIO numbers. The magnetic encoder measures pendulum angle; there is no cart linear position feedback in this PCB. Homing positions and measured stroke are based on issued step counts, so lost steps are not independently detected.
```

## docs/VERTICAL_CONTROL.md:1–40

SHA-256: `7a9b5a90d1143bf289c50c11ed5d24682a1ef7436056e0bfa4172950ebd59dd5`

```text
1: # Integrated vertical rail control
2: 
3: This feature requires **executor-v7-vertical** and the matching SDK. It adds the
4: vertical axes to the normal cartpole executor and interactive `cartpole-console`;
5: it is separate from the earlier standalone commissioning sketch.
6: 
7: ## Confirmed installation
8: 
9: The owner verified on 2026-10-23 that raw `+` on the left driver moves down, and
10: raw `+` on the right moves up. The firmware therefore maps physical UP to left
11: DIR LOW / right DIR HIGH. Operator commands use `up` and `down`, not raw polarity.
12: 
13: | Axis / function | Teensy GPIO |
14: |---|---|
15: | Left Z1 STEP / DIR | D4 / D5 |
16: | Right Z2 STEP / DIR | D6 / D7 |
17: | Shared active-low driver enable | D8 |
18: | Cart STEP / DIR, unchanged | D2 / D3 |
19: 
20: The owner confirmed 200 mm usable stroke, supported rail with drivers disabled,
21: and bottom stops suitable for timed bottoming. Geometry is 200 full steps/rev,
22: Big Easy Driver default 1/16 microstepping, 5 mm travel/revolution: 640 pulses/mm.
23: These values match the commissioning defaults; NEMA 17 alone does not establish
24: the motor step angle. No Z encoder, contact input or stall sensor is commissioned.
25: 
26: ## Operator commands
27: 
28: After installing the matching firmware and launching the new release:
29: 
30: ```text
31: z status
32: z reference both
33: z move left up 5
34: z move right up 5
35: z move both up 20
36: z move left down 5
37: z move right down 5
38: z stop
39: ```
40: 
```

## docs/VERTICAL_COMMISSIONING.md:31–38

SHA-256: `401292a49c369fcce6b53ce5fb9f7a2b149a6cf55c89b6fe6094419e44b66c07`

```text
31: 
32: MS1/MS2/MS3 are left in their existing hardware configuration. Big Easy Driver
33: unconnected defaults are 1/16 microstepping, per the
34: [SparkFun product specification](https://www.sparkfun.com/big-easy-driver.html).
35: The sketch does not drive these pins or change motor current. NEMA 17 specifies
36: frame size, not step angle: **verify 1.8°/200 full steps per revolution**; use
37: `--motor-steps 400` for 0.9° motors. The supplied 5 mm pitch is interpreted as
38: **5 mm travel per revolution (lead)**. Multi-start screws require actual lead.
```
