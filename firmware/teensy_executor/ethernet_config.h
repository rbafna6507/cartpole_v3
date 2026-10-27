#pragma once
// Copperhill carrier uses Teensy 4.1 native Ethernet, not SPI/W5500.
// Dedicated cable: computer 192.168.50.1/24, Teensy 192.168.50.2/24.
#define PENDULUM_IP 192,168,50,2
#define PENDULUM_NETMASK 255,255,255,0
#define PENDULUM_GATEWAY 0,0,0,0
#define PENDULUM_TCP_PORT 9000
