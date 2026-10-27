"""Thread-safe measurement engine and local calibration dashboard API."""
import json
import math
from pathlib import Path
import threading
import time
import cv2
import numpy as np
from flask import Flask,Response,jsonify,request
from .cart import CartTemplate
from .capture import ArrivalStats
from .detector import ArmDetector
from .state import StateEstimator
from .metrics import MeasurementMetrics
from .history import MeasurementHistory
from .encoder_feed import Comparison
from .runtime import JsonlLog


class Engine:
    def __init__(self, target_fps=80, directory=None, detector_width=640, encoder_sign=1, vision_delay_ms=0, comparison=None):
        self.lock=threading.RLock()
        self.changed=threading.Condition(self.lock);self.sensing_slots=threading.BoundedSemaphore(4)
        self.detector=ArmDetector(processing_width=detector_width);self.estimator=StateEstimator();self.cart=CartTemplate()
        self.metrics=MeasurementMetrics(target_fps)
        self.history=MeasurementHistory()
        self.comparison=comparison or Comparison(Path(directory) if directory else None,encoder_sign,vision_delay_ms)
        self.stats=ArrivalStats(target_fps);self.frame=None;self.pose=None;self.mask=None
        self.jpg=None;self.mask_jpg=None;self.last_seq=None;self.skipped=0;self.processed=0
        self.calibration_path=None;self.calibration_geometry=None
        self.camera=None;self.error=None;self.directory=Path(directory) if directory else None
        self.sdk=None;self.preview_seq=None;self.recorder=None;self.telemetry_log=None;self.recording_setup_error=None;self.video=None;self.video_frame=None

    def process(self,frame,t,seq):
        tick=time.perf_counter()
        with self.lock:
            acquired=time.perf_counter()
            if self.frame is not None and frame.shape!=self.frame.shape:
                self.estimator.reset();self.cart=CartTemplate();self.detector.seed=None
                self.history.clear();self.comparison.reset()
            if self.calibration_geometry is not None and (frame.shape[1],frame.shape[0])!=self.calibration_geometry[:2]:
                self.estimator.reset();self.comparison.reset()
                self.calibration_path=None;self.calibration_geometry=None
            detect_start=time.perf_counter()
            detection=self.detector.detect(frame)
            detection_ms=(time.perf_counter()-detect_start)*1000
            cart_pivot,score,cart_reason=self.cart.detect(frame)
            self.frame=frame;self.pose=detection.pose;self.mask=detection.mask
            # Trust the screw for cart selection only when it passes the same
            # length gate as StateEstimator; rejected arms cannot disable a template.
            home_length=self.estimator.home_length
            arm_accepted=detection.pose is not None and (home_length is None or
                abs(detection.pose.length/home_length-1)<=.20)
            if arm_accepted:
                if cart_pivot is not None and np.linalg.norm(np.array(cart_pivot)-detection.pose.pivot)>8:
                    cart_reason='cart template disagrees with screw; reselect patch'
                    self.cart.invalid_reason=cart_reason
                cart_pivot=None
            state=self.estimator.update(detection.pose,t,cart_pivot)
            if self.last_seq is not None:self.skipped+=max(0,seq-self.last_seq-1)
            self.last_seq=seq;self.processed+=1;self.stats.observe(t)
            state.update(seq=seq,detection_status=detection.reason,cart_template_status=cart_reason,
                         cart_match_score=score,processing_ms=(time.perf_counter()-tick)*1000,
                         frame_width=frame.shape[1],frame_height=frame.shape[0],
                         detected=detection.pose is not None,lock_wait_ms=(acquired-tick)*1000,
                         detection_ms=detection_ms)
            fusion_tick=time.perf_counter()
            self.comparison.vision(state)
            state["comparison_ms"]=(time.perf_counter()-fusion_tick)*1000
            state['processing_ms']=(time.perf_counter()-tick)*1000
            state['host_read_to_result_ms']=max(0.,(time.monotonic()-t)*1000)
            state['detector_width']=self.detector.processing_width
            self.metrics.observe(state)
            self.estimator.last_state.update(state)
            self.history.append(state)
            self.video_frame=(seq,t,frame,detection.pose,state['status'],detection.mask)
            self.changed.notify_all()
            return state

    def snapshot(self, include_history=False, after=None, epoch=None, comparison_after=None, comparison_epoch=None):
        with self.lock:
            state=self.estimator.snapshot(time.monotonic())
            tracking=self.stats.report()
            if state['stale']:tracking['target_rate_met']=False
            state.update(tracking=tracking,processed_frames=self.processed,skipped_camera_frames=self.skipped,
                         worker_error=self.error,cart_template_selected=self.cart.template is not None,
                         preview_seq=self.preview_seq)
            state['measurement_quality']=self.metrics.report(time.monotonic())
            if include_history:state['history']=self.history.read(after,epoch)
        state['comparison']=self.comparison.snapshot(after=comparison_after,epoch=comparison_epoch,include_history=include_history)
        if self.camera is not None:state['camera']=self.camera.report()
        if self.video is not None:state['video']=self.video.report()
        if self.telemetry_log is not None:state['telemetry_recording']=self.telemetry_log.report()
        if self.recording_setup_error:state['recording_setup_error']=self.recording_setup_error
        if self.recorder is not None:
            state['recording']=dict(dropped_frames=self.recorder.dropped,error=self.recorder.error,
                                    written_frames=self.recorder.recorder.count if self.recorder.recorder else 0,queued_frames=self.recorder.queue.qsize())
        return state

    def fresh(self):
        if self.estimator.last_t is None or time.monotonic()-self.estimator.last_t>.15:
            raise RuntimeError('No fresh camera measurement')

    def restore_calibration(self,path,width,height,rotation):
        path=Path(path);data=json.loads(path.read_text())
        geometry=(data.get('frame_width'),data.get('frame_height'),data.get('rotation'))
        if geometry!=(width,height,rotation):raise ValueError('Saved calibration camera geometry differs')
        c=data['calibration']
        angle=c.get('home_angle_rad');length=c.get('home_length_px')
        if (angle is None)!=(length is None):raise ValueError('Incomplete angle calibration')
        if angle is not None and (not math.isfinite(angle) or not math.isfinite(length) or length<=0):
            raise ValueError('Invalid angle calibration')
        a=c.get('cart_a_px');b=c.get('cart_b_px');span=c.get('cart_span_mm')
        for point in (a,b):
            if point is not None and (not isinstance(point,list) or len(point)!=2 or not all(math.isfinite(v) for v in point)):
                raise ValueError('Invalid cart reference')
        if b is not None and (a is None or span is None or not math.isfinite(span) or not 10<=span<=2000 or np.linalg.norm(np.array(a)-b)<40):
            raise ValueError('Invalid cart span')
        self.estimator.home_angle=angle;self.estimator.home_length=length
        self.estimator.cart_a=a;self.estimator.cart_b=b;self.estimator.cart_span_mm=span
        self.calibration_path=path;self.calibration_geometry=geometry
        self.estimator.invalidate_measurement()
        self.history.clear();self.comparison.reset()

    def save_event(self,event):
        if self.calibration_path is not None:
            width,height,rotation=self.calibration_geometry
            saved=dict(frame_width=width,frame_height=height,rotation=rotation,calibration=self.estimator.calibration())
            temporary=self.calibration_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(saved,indent=2,allow_nan=False))
            temporary.replace(self.calibration_path)
        if self.directory is not None:
            payload=dict(t=time.monotonic(),event=event,calibration=self.estimator.calibration(),
                         comparison_settings=dict(encoder_sign=self.comparison.encoder_sign,
                                                  vision_delay_ms=self.comparison.vision_delay*1000))
            (self.directory/'calibration.json').write_text(json.dumps(payload,indent=2))
            with JsonlLog(self.directory/'events.jsonl',max_bytes=1024*1024,backups=1) as log:log.write(json.dumps(payload)+'\n')


def create_app(engine, read_only=False):
    app=Flask(__name__)
    app.config['MAX_CONTENT_LENGTH']=16*1024
    page=Path(__file__).with_name('dashboard.html')

    @app.after_request
    def compress_json(response):
        if response.is_json and request.accept_encodings['gzip']>0:
            import gzip
            response.set_data(gzip.compress(response.get_data(),compresslevel=1))
            response.headers['Content-Encoding']='gzip'
            response.vary.add('Accept-Encoding')
        return response

    @app.before_request
    def restrict_remote_calibration():
        if read_only and request.method not in ('GET','HEAD') and request.path!='/api/video-mode':
            return jsonify(error='Calibration changes require the local observation endpoint'),403

    @app.get('/api/access')
    def access():return jsonify(read_only=read_only)

    @app.get('/')
    def index():return page.read_text()

    @app.get('/api/state')
    def state():
        if request.args.get('measurement')=='1':
            with engine.lock:
                result=engine.estimator.snapshot(time.monotonic())
                result['worker_error']=engine.error
            return jsonify(result),200,{'Cache-Control':'no-store'}
        include_history=request.args.get('history')=='1'
        after=request.args.get('after') if include_history else None
        if after is not None:
            try:
                after=int(after)
                if after<0:raise ValueError()
            except ValueError:return jsonify(error='after must be a nonnegative sequence number'),400
        comparison_after=request.args.get('comparison_after') if include_history else None
        if comparison_after is not None:
            try:
                comparison_after=int(comparison_after)
                if comparison_after<0:raise ValueError()
            except ValueError:return jsonify(error='comparison_after must be a nonnegative sequence number'),400
        result=engine.snapshot(include_history,after,request.args.get('epoch'),
                               comparison_after,request.args.get('comparison_epoch'))
        limit=request.args.get('history_limit')
        if limit is not None:
            try:
                limit=int(limit)
                if not 1<=limit<=4096:raise ValueError()
            except ValueError:return jsonify(error='history_limit must be 1..4096'),400
            if 'history' in result:result['history']['samples']=result['history']['samples'][-limit:]
            for key in ('encoder','fused'):
                result['comparison'][key]['samples']=result['comparison'][key]['samples'][-limit:]
        return jsonify(result)

    @app.get('/api/encoder')
    def encoder_packet():
        after=request.args.get('after_seq')
        if after is not None:
            try:
                after=int(after)
                if after<0:raise ValueError()
            except ValueError:return jsonify(error='after_seq must be a nonnegative integer'),400
        return jsonify(engine.comparison.packet(after_seq=after,session=request.args.get('session'))),200,{'Cache-Control':'no-store'}

    @app.get('/api/sensing-stream')
    def sensing_stream():
        from .sensing_stream import envelopes
        try:
            max_hz=int(request.args.get('max_hz','60'))
            after=request.args.get('after_seq')
            after=None if after is None else int(after)
            if not 10<=max_hz<=100 or (after is not None and after<0):raise ValueError()
        except ValueError:return jsonify(error='max_hz must be 10..100 and after_seq nonnegative'),400
        session=request.args.get('session')
        if session is not None and len(session)>64:return jsonify(error='session is too long'),400
        connection=request.environ.get('werkzeug.socket')
        if connection is not None:
            import socket
            connection.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,32768)
            connection.settimeout(5)
        if not engine.sensing_slots.acquire(blocking=False):return jsonify(error='Too many sensing subscribers'),429
        released=False;release_lock=threading.Lock()
        def release():
            nonlocal released
            with release_lock:
                if not released:released=True;engine.sensing_slots.release()
        def generate():
            try:yield from envelopes(engine,after,session,max_hz)
            finally:release()
        # A slow consumer blocks only its own bounded generator, never capture.
        response=Response(generate(),mimetype='application/x-ndjson',
                          headers={'Cache-Control':'no-store','X-Accel-Buffering':'no'})
        response.call_on_close(release)
        return response

    from .developer_streams import install as install_developer_streams
    install_developer_streams(app, engine)

    @app.get('/api/sdk-state')
    def sdk_state():
        from .sdk_feed import empty_snapshot
        result=engine.sdk.snapshot() if engine.sdk else empty_snapshot()
        after=request.args.get('after')
        if after is not None:
            try:after=int(after)
            except ValueError:return jsonify(error='after must be an integer'),400
            if (result.get('packet') or {}).get('session_id')==request.args.get('session'):
                result['samples']=[p for p in result['samples'] if p['sequence']>after]
        if request.args.get('compact')=='1':result['samples']=result['samples'][-120:]
        return jsonify(result),200,{'Cache-Control':'no-store'}

    @app.get('/sdk.js')
    def sdk_script():
        return Response(page.with_name('sdk.js').read_text(),mimetype='text/javascript')

    @app.get('/api/clock')
    def clock():
        return jsonify(monotonic_s=time.monotonic()),200,{'Cache-Control':'no-store'}

    @app.get('/snapshot.png')
    def snapshot():
        with engine.lock:
            try:engine.fresh()
            except RuntimeError as e:return jsonify(error=str(e)),409
            frame=engine.frame.copy()
        ok,data=cv2.imencode('.png',frame)
        return Response(data.tobytes(),mimetype='image/png') if ok else (jsonify(error='PNG encoding failed'),500)

    @app.get('/preview.jpg')
    def preview():
        with engine.lock:
            data=engine.mask_jpg if request.args.get('mask')=='1' else engine.jpg
        if data is None:return jsonify(error='Waiting for camera'),503
        return Response(data,mimetype='image/jpeg',headers={'Cache-Control':'no-store'})

    @app.get('/video.js')
    def video_script():
        return Response(page.with_name('video.js').read_text(),mimetype='text/javascript')

    @app.get('/timing.js')
    def timing_script():
        return Response(page.with_name('timing.js').read_text(),mimetype='text/javascript')

    @app.get('/traces.js')
    def traces_script():
        return Response(page.with_name('traces.js').read_text(),mimetype='text/javascript',
                        headers={'Cache-Control':'no-store'})

    @app.post('/api/comparison')
    def comparison_settings():
        body=request.get_json(silent=True)
        if not isinstance(body,dict):return jsonify(error='Expected a JSON object'),400
        sign=body.get('encoder_sign');delay=body.get('vision_delay_ms')
        if type(sign) is not int or sign not in (-1,1):return jsonify(error='Encoder sign must be +1 or -1'),400
        if type(delay) not in (int,float) or not math.isfinite(delay) or not 0<=delay<=200:
            return jsonify(error='Camera delay must be between 0 and 200 ms'),400
        with engine.lock, engine.comparison.lock:
            engine.comparison.configure(sign,delay)
            engine.save_event('comparison alignment changed')
        return jsonify(ok=True)

    @app.get('/comparison.js')
    def comparison_script():
        return Response(page.with_name('comparison.js').read_text(),mimetype='text/javascript',
                        headers={'Cache-Control':'no-store'})

    @app.get('/video.h264')
    def video_stream():
        if engine.video is None:return jsonify(error='Video encoder unavailable'),503
        video=engine.video
        viewer=video.hub.subscribe()
        def packets():
            import queue
            try:
                while True:
                    try:packet=viewer.get(timeout=5)
                    except queue.Empty:return
                    if packet is None:return
                    yield packet
            finally:video.hub.unsubscribe(viewer)
        return Response(packets(),mimetype='application/octet-stream',
                        headers={'Cache-Control':'no-store','X-Accel-Buffering':'no'})

    @app.post('/api/video-mode')
    def video_mode():
        if engine.video is None:return jsonify(error='Video encoder unavailable'),503
        body=request.get_json(silent=True)
        if not isinstance(body,dict):return jsonify(error='Expected a JSON object'),400
        try:engine.video.set_mode(body.get('mode'))
        except ValueError as error:return jsonify(error=str(error)),400
        return jsonify(ok=True,mode=engine.video.mode)

    def action(fn):
        with engine.lock:
            try:
                engine.fresh()
                body=request.get_json(silent=True)
                if body is None:body={}
                if not isinstance(body,dict):raise ValueError('Expected a JSON object')
                fn(body)
                engine.estimator.invalidate_measurement()
                engine.history.clear();engine.comparison.reset()
                engine.save_event(request.path)
            except RuntimeError as error:return jsonify(error=str(error)),409
            except (ValueError,TypeError,KeyError,OverflowError) as error:return jsonify(error=str(error)),400
        return jsonify(ok=True,calibration=engine.estimator.calibration())

    @app.post('/api/home')
    def home():return action(lambda body:engine.estimator.home())

    @app.post('/api/cart/<which>')
    def mark_cart(which):
        return action(lambda body:engine.estimator.mark_cart(which,float(body.get('span_mm',0))))

    @app.post('/api/select')
    def select():
        def apply(body):
            x,y=float(body['x']),float(body['y'])
            if not (math.isfinite(x) and math.isfinite(y) and 0<=x<=1 and 0<=y<=1):
                raise ValueError('Select a point inside the image')
            h,w=engine.frame.shape[:2]
            engine.detector.select(x*w,y*h)
            engine.estimator.reset();engine.cart=CartTemplate()
        return action(apply)

    @app.post('/api/template')
    def template():
        def apply(body):
            if engine.pose is None:raise ValueError('Arm and pivot must be visible while selecting the cart')
            rect=np.array(body['rectangle'],float)
            if rect.shape!=(4,) or not np.all(np.isfinite(rect)) or np.any(rect<0) or np.any(rect>1):
                raise ValueError('Invalid normalized rectangle')
            h,w=engine.frame.shape[:2]
            engine.cart.select(engine.frame,rect*np.array([w,h,w,h]),engine.pose.pivot)
            engine.estimator.cart_velocity.clear();engine.estimator.cart_samples.clear()
        return action(apply)

    @app.post('/api/reset')
    def reset():
        with engine.lock:
            engine.estimator.reset();engine.detector.seed=None;engine.cart=CartTemplate()
            engine.history.clear();engine.comparison.reset()
            engine.save_event('reset')
        return jsonify(ok=True)

    return app
