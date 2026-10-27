"""System-Python GStreamer helper. Stdin: timestamped BGR. Stdout: H.264 packets.

Separate process uses the Jetson's existing GI runtime, without adding packages
or requiring GI in the vision virtual environment. Diagnostics go to stderr.
"""
import struct
import sys
import threading
import gi
gi.require_version('Gst','1.0')
from gi.repository import Gst


def exact(stream,n):
    out=bytearray()
    while len(out)<n:
        data=stream.read(n-len(out))
        if not data:
            if not out:return None
            raise EOFError('Truncated raw frame')
        out.extend(data)
    return bytes(out)


def main():
    width,height,fps,bitrate=map(int,sys.argv[1:])
    Gst.init(None)
    pipeline=Gst.parse_launch(
        f'appsrc name=source is-live=true format=time block=true max-buffers=2 max-bytes=0 '
        f'caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 '
        '! videoconvert n-threads=2 ! video/x-raw,format=I420 '
        f'! x264enc speed-preset=ultrafast tune=zerolatency bitrate={bitrate} '
        f'threads=2 sliced-threads=true bframes=0 key-int-max={max(1,fps//2)} byte-stream=true '
        '! video/x-h264,profile=constrained-baseline '
        '! h264parse config-interval=-1 ! video/x-h264,stream-format=byte-stream,alignment=au '
        '! appsink name=sink sync=false max-buffers=2 drop=false')
    source=pipeline.get_by_name('source');sink=pipeline.get_by_name('sink')
    errors=[]
    def output():
        base=None
        try:
            while True:
                sample=sink.emit('pull-sample')
                if sample is None:break
                buf=sample.get_buffer();payload=buf.extract_dup(0,buf.get_size())
                if base is None:base=buf.pts
                timestamp=max(0,(buf.pts-base)//1000)
                key=not buf.has_flags(Gst.BufferFlags.DELTA_UNIT)
                sys.stdout.buffer.write(struct.pack('!IQB',len(payload),timestamp,int(key)))
                sys.stdout.buffer.write(payload);sys.stdout.buffer.flush()
        except Exception as error:
            errors.append(error)
            print(error,file=sys.stderr,flush=True)
    pipeline.set_state(Gst.State.PLAYING)
    reader=threading.Thread(target=output,daemon=True);reader.start()
    base=None
    try:
        while not errors:
            header=exact(sys.stdin.buffer,12)
            if header is None:break
            timestamp,size=struct.unpack('!dI',header)
            if size!=width*height*3:raise ValueError('Unexpected raw frame dimensions')
            frame=exact(sys.stdin.buffer,size)
            if frame is None:raise EOFError('Missing raw frame')
            if base is None:base=timestamp
            buf=Gst.Buffer.new_allocate(None,size,None);buf.fill(0,frame)
            buf.pts=buf.dts=round((timestamp-base)*Gst.SECOND)
            buf.duration=Gst.SECOND//fps
            result=source.emit('push-buffer',buf)
            if result!=Gst.FlowReturn.OK:raise RuntimeError(f'Encoder input: {result}')
            message=pipeline.get_bus().pop_filtered(Gst.MessageType.ERROR)
            if message:
                error,debug=message.parse_error();raise RuntimeError(f'{error}: {debug}')
        source.emit('end-of-stream');reader.join(timeout=5)
        if errors:raise errors[0]
    finally:pipeline.set_state(Gst.State.NULL)

if __name__=='__main__':main()
