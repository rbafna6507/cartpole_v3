"""Run the observe-only cartpole vision dashboard."""
import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
import signal
from pathlib import Path
import threading
import time
import cv2
from werkzeug.serving import make_server
from .app import Engine,create_app
from .http import StreamingRequestHandler,ViewServer
from .capture import Camera,BufferedRecorder
from .preview import PreviewWorker
from .stream import VideoStream
from .comparison_process import ComparisonProcess
from .runtime import JsonlLog, camera_failure, prune_sessions


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera',default='0');p.add_argument('--width',type=int,default=960)
    p.add_argument('--height',type=int,default=600);p.add_argument('--fps',type=int,default=30)
    p.add_argument('--rotate',type=int,choices=(0,90,180,270),default=180)
    p.add_argument('--view-bind',help='Optional exact tailnet IPv4 for remote viewing; calibration stays on loopback')
    p.add_argument('--view-port',type=int,default=3091)
    p.add_argument('--view-tcp-mss',type=int,help='Optional private listener TCP MSS, 536..1220 bytes; use only after network measurement')
    p.add_argument('--port',type=int,default=3090);p.add_argument('--preview-fps',type=float,default=1)
    p.add_argument('--detector-width',type=int,default=640,help='Detection image width; endpoints remain in camera coordinates')
    p.add_argument('--opencv-threads',type=int,default=2)
    p.add_argument('--video-fps',type=int,help='H.264 frame-rate cap; defaults to capture FPS')
    p.add_argument('--no-video',action='store_true',help='Disable H.264 to measure tracking without video encoding load')
    p.add_argument('--video-bitrate',type=int,default=4000,help='H.264 target bitrate, kbit/s')
    p.add_argument('--gst-python',default='/usr/bin/python3',help='Python with the installed GStreamer GI runtime')
    p.add_argument('--calibration-file',type=Path,help='Restore saved vision reference for identical camera geometry; update it on calibration changes')
    p.add_argument('--encoder-port',type=int,default=3096,help='Loopback UDP feed from console; 0 disables')
    p.add_argument('--encoder-sign',type=int,choices=(-1,1),default=1,help='Encoder direction relative to image-right')
    p.add_argument('--vision-delay-ms',type=float,default=0,help='Measured camera delay to subtract from host-read timestamp')
    p.add_argument('--camera-timeout',type=float,default=2,help='Exit on stalled camera so a supervisor can restart')
    p.add_argument('--keep-sessions',type=int,default=4,help='Bounded telemetry sessions retained; recorded sessions are never pruned')
    p.add_argument('--log-file',type=Path,help='Optional rotating runtime diagnostics, 8 MiB maximum')
    p.add_argument('--sdk-port',type=int,default=3097,help='Loopback SDK shadow telemetry; 0 disables')
    p.add_argument('--out',type=Path,default=Path('runs/vision'));p.add_argument('--record',action='store_true')
    p.add_argument('--duration',type=float,help='Stop automatically after this many seconds')
    p.add_argument('--exposure-raw',type=float,help='Backend-specific exposure value; acceptance/readback logged')
    p.add_argument('--auto-exposure-raw',type=float,help='Backend-specific auto-exposure mode')
    p.add_argument('--gain-raw',type=float,help='Backend-specific gain value')
    a=p.parse_args(argv)
    if a.video_fps is None:a.video_fps=a.fps
    import math
    if min(a.width,a.height,a.fps,a.video_fps,a.video_bitrate,a.detector_width,a.opencv_threads)<=0 or not math.isfinite(a.preview_fps) or a.preview_fps<=0:p.error('Dimensions and rates must be positive')
    if a.duration is not None and (not math.isfinite(a.duration) or a.duration<=0):p.error('duration must be positive')
    for value in (a.exposure_raw,a.auto_exposure_raw,a.gain_raw):
        if value is not None and not math.isfinite(value):p.error('Camera control values must be finite')
    if not 0<=a.encoder_port<=65535 or not 0<=a.sdk_port<=65535:p.error('UDP ports must be 0..65535')
    if not 1<=a.port<=65535 or not 1<=a.view_port<=65535:p.error('HTTP ports must be 1..65535')
    if a.view_tcp_mss is not None and not 536<=a.view_tcp_mss<=1220:p.error('view TCP MSS must be 536..1220')
    if a.view_bind:
        import ipaddress
        try:
            if ipaddress.ip_address(a.view_bind) not in ipaddress.ip_network('100.64.0.0/10'):raise ValueError()
        except ValueError:p.error('view-bind must be an exact Tailscale IPv4 address in 100.64.0.0/10')
    if not math.isfinite(a.camera_timeout) or a.camera_timeout<=0:p.error('camera timeout must be positive')
    if a.keep_sessions<1:p.error('keep sessions must be positive')
    if not math.isfinite(a.vision_delay_ms) or not 0<=a.vision_delay_ms<=200:p.error('vision delay must be 0..200 ms')
    return a


def main(argv=None):
    a=arguments(argv)
    cv2.setNumThreads(a.opencv_threads)
    setup_error=None
    if a.log_file:
        try:
            a.log_file.parent.mkdir(parents=True,exist_ok=True)
            handler=RotatingFileHandler(a.log_file,maxBytes=2*1024*1024,backupCount=3)
            handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
            logging.getLogger().addHandler(handler)
            logging.getLogger().setLevel(logging.INFO)
        except OSError as error:setup_error=str(error)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    directory=a.out/(time.strftime('%Y%m%d_%H%M%S')+f'_{time.time_ns()%1000000:06d}')
    try:
        a.out.mkdir(parents=True,exist_ok=True)
        # Reclaim expired sessions before allocating another one on a full disk.
        prune_sessions(a.out,max(0,a.keep_sessions-1))
        directory.mkdir(exist_ok=False)
        if not a.record:(directory/'.cartpole-vision-session').touch()
        (directory/'session.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},indent=2))
    except OSError as error:
        setup_error=str(error)
        logging.warning('Recording setup unavailable; continuing camera: %s',error)
    stop=threading.Event()
    old_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    for sig in old_handlers:signal.signal(sig,lambda *_:stop.set())
    cam=engine=comparison=recorder=thread=preview=server=sdk=view_server=view_thread=None
    started=time.monotonic();failure=None
    try:
        comparison=ComparisonProcess(directory,a.encoder_sign,a.vision_delay_ms,a.encoder_port)
        cam=Camera(a.camera,a.width,a.height,a.fps,a.rotate,a.exposure_raw,a.auto_exposure_raw,a.gain_raw)
        engine=Engine(a.fps,directory,a.detector_width,a.encoder_sign,a.vision_delay_ms,comparison)
        engine.camera=cam
        engine.recording_setup_error=setup_error
        from .sdk_feed import SDKFeed
        sdk=SDKFeed(a.sdk_port) if a.sdk_port else None
        engine.sdk=sdk
        if a.calibration_file:engine.restore_calibration(a.calibration_file,a.width,a.height,a.rotate)
        recorder=BufferedRecorder(directory,a.fps) if a.record else None
        engine.recorder=recorder
        def worker():
            last=0;flushed=time.monotonic()
            try:
                with JsonlLog(directory/'states.jsonl') as log:
                    engine.telemetry_log=log
                    while not stop.is_set():
                        item=cam.get()
                        if item is None or item[0]==last:stop.wait(.001);continue
                        seq,t,frame=item;last=seq
                        state=engine.process(frame,t,seq)
                        log.write(json.dumps(state,allow_nan=False)+'\n')
                        if recorder:recorder.write(frame,seq,t)
                        now=time.monotonic()
                        if now-flushed>1:log.flush();flushed=now
            except Exception as error:
                with engine.lock:engine.error=str(error)
                logging.exception('Tracking worker stopped')
        thread=threading.Thread(target=worker,name='vision-worker',daemon=True);thread.start()
        preview=PreviewWorker(engine,a.preview_fps)
        if not a.no_video:
            video_width,video_height=(a.height,a.width) if a.rotate in (90,270) else (a.width,a.height)
            engine.video=VideoStream(cam,engine,video_width,video_height,a.video_fps,a.video_bitrate,a.gst_python)
        server=make_server('127.0.0.1',a.port,create_app(engine),threaded=True,request_handler=StreamingRequestHandler)
        server.timeout=.25
        if a.view_bind:
            view_server=ViewServer(a.view_bind,a.view_port,create_app(engine,read_only=True),a.view_tcp_mss)
            view_thread=threading.Thread(target=lambda:view_server.serve_forever(poll_interval=.25),name='tailnet-viewer',daemon=True)
            view_thread.start()
        message=f'Vision: http://127.0.0.1:{a.port} | requested {a.width}x{a.height} @ {a.fps} FPS | {directory}'
        print(message,flush=True);logging.info(message)
        while not stop.is_set() and (a.duration is None or time.monotonic()-started<a.duration):
            server.handle_request()
            reason=camera_failure(cam.report(),time.monotonic()-started,a.camera_timeout)
            if reason:raise RuntimeError(reason)
            if engine.error:raise RuntimeError(engine.error)
            if engine.video and engine.video.error:raise RuntimeError(engine.video.error)
    except Exception as error:
        failure=str(error);logging.exception('Vision runtime stopped; supervisor may restart')
    finally:
        stop.set()
        if view_server:
            view_server.shutdown();view_server.server_close()
        if view_thread:view_thread.join(timeout=2)
        if server:server.server_close()
        if thread:thread.join(timeout=5)
        if preview:preview.close()
        if engine and engine.video:engine.video.close()
        report=engine.snapshot() if engine else {}
        report['runtime_error']=failure
        try:(directory/'final_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        except OSError as error:logging.warning('Final recording report unavailable: %s',error)
        if sdk:sdk.close()
        if comparison:comparison.close()
        if cam:cam.close()
        if recorder:recorder.close()
        for sig,handler in old_handlers.items():signal.signal(sig,handler)
        print(json.dumps({k:report.get(k) for k in ('camera','tracking','processed_frames','skipped_camera_frames','runtime_error')},indent=2),flush=True)
    if failure:raise SystemExit(1)


if __name__=='__main__':main()
