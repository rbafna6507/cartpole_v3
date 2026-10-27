#pragma once
#include <QNEthernet.h>
#include "ethernet_config.h"
namespace pendulum_net {
using namespace qindesign::network;
static EthernetServer server(PENDULUM_TCP_PORT);
static EthernetClient client;
static bool active=false, ready=false;
static char rx[96], tx[8192];
static size_t rxn=0, head=0, tail=0;
static bool overflow=false;
static uint32_t dropped=0;
inline void close() {
  client.abort(); active=false; rxn=0; overflow=false; head=tail=0;
}
inline bool begin() {
  ready=Ethernet.begin(IPAddress(PENDULUM_IP),IPAddress(PENDULUM_NETMASK),IPAddress(PENDULUM_GATEWAY));
  if(ready)server.begin();
  return ready;
}
inline void enqueue(const char *frame,size_t n) {
  if(!active)return;
  size_t used=(head+sizeof(tx)-tail)%sizeof(tx);
  if(n>=sizeof(tx)-used){++dropped;return;}
  for(size_t i=0;i<n;++i){tx[head]=frame[i];head=(head+1)%sizeof(tx);}
}
// Application RX/TX work is bounded; network-stack execution time must still
// be measured on the actual board. Never wait for link, connection, or TX space.
inline void poll(bool canAccept,void (*command)(char*),void (*lost)()) {
  if(!ready)return;
  if(active && (!Ethernet.linkState() || !static_cast<bool>(client))) {
    lost(); close(); return; // stop before discarding pending commands
  }
  EthernetClient incoming=server.accept();
  if(incoming) {
    if(active || !canAccept)incoming.abort();
    else {
      client=incoming; client.setNoDelay(true); active=true;
      rxn=0;overflow=false;head=tail=0;
    }
  }
  if(!active)return;
  unsigned budget=96;
  while(budget-- && client.available()) {
    int value=client.read();if(value<0)break;
    char c=(char)value;
    if(c=='\r' || c=='\n') {
      if(!overflow && rxn){rx[rxn]=0;command(rx);}
      rxn=0;overflow=false;return;
    }
    if(!overflow){if(rxn<sizeof(rx)-1)rx[rxn++]=c;else overflow=true;}
  }
}
inline void serviceTx() {
  if(!active || head==tail)return;
  int space=client.availableForWrite();if(space<=0)return;
  size_t n=head>tail?head-tail:sizeof(tx)-tail;
  if(n>(size_t)space)n=space;
  if(n>256)n=256;
  size_t sent=client.write((const uint8_t*)tx+tail,n);
  tail=(tail+sent)%sizeof(tx);
}
}
