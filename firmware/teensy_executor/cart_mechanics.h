#pragma once

// Shared pulse/drive geometry. This ceiling describes pulse generation,
// not the motor's measured torque or ability to follow the commands.
namespace cart_hardware {
constexpr int motor_steps = 200, microsteps = 16, pulley_teeth = 60;
constexpr float belt_pitch_mm = 2.0f;
// Use one exact integer-microsecond timer period everywhere, including DDS.
constexpr unsigned step_timer_period_us = 7;
constexpr double isr_hz = 1000000.0 / step_timer_period_us;
constexpr float mm_per_rev = pulley_teeth * belt_pitch_mm;
// Linear limits stay fixed when pulley geometry changes (60T, 2026-09-17).
constexpr float default_speed = 1.5f;         // m/s
constexpr float default_acceleration = 25.0f; // m/s^2
constexpr float default_jerk = 100.0f;         // m/s^3
constexpr float requested_speed_rpm = default_speed * 60000.0f / mm_per_rev;
constexpr float requested_acceleration_rpm_s = default_acceleration * 60000.0f / mm_per_rev;
constexpr float requested_jerk_rpm_s2 = default_jerk * 60000.0f / mm_per_rev;
constexpr float steps_per_m = motor_steps * microsteps * 1000.0f / mm_per_rev;
constexpr float max_speed = (isr_hz * 0.5f) * mm_per_rev /
                            (motor_steps * microsteps * 1000.0f);
constexpr float max_rpm = isr_hz * 0.5f * 60.0f / (motor_steps * microsteps);
}
