"""Bounded pull-driven sensing stream. Camera and encoder timestamps stay intact."""
import json
import time

MAX_LINE_BYTES=4*1024*1024


def encode_envelope(vision,encoder,sequence,host_t,max_bytes=MAX_LINE_BYTES):
    encoder=dict(encoder)
    packets=encoder.pop('packets',[])
    envelope=dict(schema='cartpole-sensing-stream-v1',stream_seq=sequence,
                  host_t=host_t,vision=vision,encoder=encoder)
    dump=lambda value:json.dumps(value,separators=(',',':'),allow_nan=False).encode()
    # Reserve metadata and a small margin before serializing bounded packets.
    base=len(dump(envelope))+64
    budget=max_bytes-base-128
    kept=[];used=0
    for packet in reversed(packets):
        encoded=dump(packet)
        if used+len(encoded)+1>budget:break
        kept.append(packet);used+=len(encoded)+1
    kept.reverse();encoder['packets']=kept
    if len(kept)<len(packets):
        encoder['history_truncated']=True
        encoder['oldest_seq']=kept[0]['seq'] if kept else None
    line=dump(envelope)+b'\n'
    if len(line)>max_bytes:raise ValueError('Sensing metadata exceeds maximum line size')
    return line


def envelopes(engine,after_seq=None,session=None,max_hz=60):
    previous_frame=None;last_emit=None;sequence=0
    period=1/max_hz;heartbeat=.1
    while True:
        # Condition.wait releases the lock. Encoder IPC and network output never
        # occur inside this section. A camera stall still produces a heartbeat.
        with engine.changed:
            while True:
                now=time.monotonic()
                elapsed=now-last_emit if last_emit is not None else heartbeat
                fresh=engine.processed!=previous_frame
                if last_emit is None or (fresh and elapsed>=period) or elapsed>=heartbeat:break
                delay=period-elapsed if fresh else heartbeat-elapsed
                engine.changed.wait(timeout=max(.0001,min(delay,heartbeat-elapsed)))
            vision=engine.estimator.snapshot(now)
            vision['worker_error']=engine.error
            previous_frame=engine.processed
        encoder=engine.comparison.packet(after_seq=after_seq,session=session)
        last_emit=time.monotonic()
        yield encode_envelope(vision,encoder,sequence,last_emit)
        after_seq=encoder.get('latest_seq');session=encoder.get('session');sequence+=1
