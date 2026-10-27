"""Bounded loopback-first HTTP API; browser credentials never appear in URLs."""
from contextlib import closing
import argparse
import base64
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import json
from pathlib import Path
import re
import socket
import sqlite3
import threading
import time
from urllib.parse import urlsplit, parse_qs
from cartpole_sdk.recording import canonical_json, strict_json_loads
from .queue import Store, QueueError, name

from .artifacts import inspect_artifact, ArtifactError, MAX_ENVELOPE_BYTES

MAX_BODY=100_000

class _LimitedReader:
    def __init__(self,stream,connection,seconds):
        self.stream=stream;self.connection=connection
        self.deadline=time.monotonic()+seconds;self.pending=bytearray()
    def _chunk(self,size):
        remaining=self.deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('absolute request deadline')
        self.connection.settimeout(remaining)
        return self.stream.read1(size)
    def readline(self,size=-1):
        maximum=min(8193,size) if size>=0 else 8193
        while True:
            newline=self.pending.find(b'\n')
            if newline>=0 and newline<maximum:
                line=bytes(self.pending[:newline+1]);del self.pending[:newline+1]
                if len(line)>8192:raise http.client.LineTooLong('request line')
                return line
            if len(self.pending)>=maximum:raise http.client.LineTooLong('request line')
            chunk=self._chunk(min(4096,maximum-len(self.pending)))
            if not chunk:
                line=bytes(self.pending);self.pending.clear();return line
            self.pending.extend(chunk)
    def read(self,size):
        value=bytearray(self.pending[:size]);del self.pending[:len(value)]
        while len(value)<size:
            chunk=self._chunk(min(4096,size-len(value)))
            if not chunk:break
            value.extend(chunk)
        return bytes(value)
    def __getattr__(self,key):return getattr(self.stream,key)

class PortalServer(ThreadingHTTPServer):
    daemon_threads=True
    request_timeout=3
    allow_reuse_address=True
    def __init__(self,*args,**kwargs):
        self.slots=threading.BoundedSemaphore(16)
        self.video_slots=threading.BoundedSemaphore(4)
        super().__init__(*args,**kwargs)
    def server_close(self):
        try:super().server_close()
        finally:
            anchor=getattr(self,'_database_anchor',None)
            if anchor is not None:
                self._database_anchor=None
                anchor.close()
    def process_request(self,request,address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request);return
        try:super().process_request(request,address)
        except BaseException:self.slots.release();raise
    def process_request_thread(self,*args):
        try:super().process_request_thread(*args)
        finally:self.slots.release()

class Handler(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.0'
    def setup(self):
        self.request.settimeout(3)
        super().setup()
        self.rfile=_LimitedReader(self.rfile,self.request,self.server.request_timeout)
    def log_message(self,*args):pass
    def end_headers(self):
        self.send_header('Vary','Origin')
        origin=self.headers.get('Origin')
        if origin in self.server.origins:
            self.send_header('Access-Control-Allow-Origin',origin)
            self.send_header('Access-Control-Expose-Headers','Content-Disposition')
        super().end_headers()
    def do_OPTIONS(self):
        origin=self.headers.get('Origin')
        requested={h.strip().lower() for h in self.headers.get('Access-Control-Request-Headers','').split(',') if h.strip()}
        if (origin not in self.server.origins or
                self.headers.get('Access-Control-Request-Method') not in ('GET','POST') or
                not requested <= {'authorization','content-type','x-worker-lease'}):
            self._respond(403,{'error':'preflight denied'});return
        self.send_response(204)
        self.send_header('Access-Control-Allow-Methods','GET, POST')
        self.send_header('Access-Control-Allow-Headers','Authorization, Content-Type, X-Worker-Lease')
        self.send_header('Access-Control-Max-Age','600')
        self.send_header('Content-Length','0')
        self.end_headers()
    def _respond(self,status,value,kind='application/json'):
        body=canonical_json(value).encode() if kind=='application/json' else value
        self.send_response(status)
        self.send_header('Content-Type',kind+('; charset=utf-8' if kind.startswith('text/') or kind=='application/json' else ''))
        if kind=='application/zip':self.send_header('Content-Disposition','attachment; filename=cartpole-run.zip')
        self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' blob:; media-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        self.end_headers();self.wfile.write(body)
    def _principal(self):
        header=self.headers.get('Authorization','')
        candidate=header[7:] if header.startswith('Bearer ') else ''
        selected=None
        for token,principal in self.server.credentials.items():
            if hmac.compare_digest(candidate.encode(),token.encode()):selected=principal
        return selected
    def _body(self,maximum=MAX_BODY):
        if self.headers.get('Transfer-Encoding') is not None:raise QueueError('unsupported body framing')
        lengths=self.headers.get_all('Content-Length',[])
        if len(lengths)!=1 or not re.fullmatch('[0-9]{1,8}',lengths[0]):raise QueueError('invalid body length')
        size=int(lengths[0])
        if size>maximum:raise QueueError('body too large')
        if size>MAX_BODY:self.rfile.deadline=time.monotonic()+30
        if self.headers.get_content_type()!='application/json':raise QueueError('JSON body required')
        body=self.rfile.read(size)
        if len(body)!=size:raise QueueError('incomplete body')
        try:data=strict_json_loads(body.decode())
        except (ValueError,UnicodeError) as exc:raise QueueError('invalid JSON') from exc
        if not isinstance(data,dict):raise QueueError('JSON object required')
        return data
    def _fields(self,data,fields):
        if set(data)!=set(fields):raise QueueError('request fields mismatch')
    def do_GET(self):self._dispatch(False)
    def do_POST(self):self._dispatch(True)
    def _dispatch(self,mutating):
        try:
            if len(self.headers)>64 or sum(len(k)+len(v) for k,v in self.headers.items())>16384:
                self._respond(431,{'error':'headers too large'});return
            if self.headers.get('Origin') is not None and self.headers['Origin'] not in self.server.origins:
                self._respond(403,{'error':'origin denied'});return
            path=urlsplit(self.path).path
            if not mutating and path in ('/','/app.js','/style.css','/video.js','/config.js'):
                filename={'/':'index.html','/app.js':'app.js','/style.css':'style.css','/video.js':'video.js','/config.js':'config.js'}[path]
                kind={'/':'text/html','/app.js':'text/javascript','/style.css':'text/css','/video.js':'text/javascript','/config.js':'text/javascript'}[path]
                self._respond(200,(Path(__file__).parent/'web'/filename).read_bytes(),kind);return
            principal=self._principal()
            if principal is None:self._respond(401,{'error':'authentication required'});return
            owner,role=principal
            recording_upload=mutating and re.fullmatch(r'/api/jobs/[a-f0-9]{32}/recording',path) is not None
            upload=mutating and path=='/api/jobs' and role=='submit'
            body=self._body(MAX_ENVELOPE_BYTES if upload else MAX_BODY) if mutating and not recording_upload else None
            store=self.server.store
            if path=='/api/demos' and not mutating:
                if role not in ('submit','operator'):self._respond(403,{'error':'role denied'});return
                from .demos import catalog
                self._respond(200,{'demos':catalog()});return
            demo_match=re.fullmatch(r'/api/demos/(controller|policy)/run',path)
            if demo_match and mutating:
                if role not in ('submit','operator'):self._respond(403,{'error':'role denied'});return
                self._fields(body,('idempotency_key',))
                from .demos import demo_artifact
                artifact,filename=demo_artifact(demo_match[1])
                job=store.submit(owner,body['idempotency_key'],artifact,filename,'magnetic-step',20,recording='detailed')
                self._respond(201,job);return
            if path=='/api/angles' and not mutating:
                if role not in ('submit','operator'):self._respond(403,{'error':'role denied'});return
                from .camera_proxy import camera_request
                data=camera_request(self.server.camera_url,'/api/state?history=1&history_limit=4096')
                comparison=data.get('comparison',{})
                def samples(rows):
                    rows=[r for r in rows[-4096:] if isinstance(r,dict)]
                    count=min(120,len(rows));result=[];previous=-1
                    for n in range(count):
                        index=n*(len(rows)-1)//max(1,count-1)
                        row={k:rows[index][k] for k in ('t','theta_rad','angle_valid','gap') if k in rows[index]}
                        if any(r.get('gap') or r.get('angle_valid') is False or r.get('theta_rad') is None
                               for r in rows[previous+1:index+1]):row['gap']=True
                        result.append(row);previous=index
                    return result
                self._respond(200,{'now':comparison.get('now'),
                    'vision':{'stale':data.get('stale',True),'samples':samples(data.get('history',{}).get('samples',[]))},
                    'encoder':{'samples':samples(comparison.get('encoder',{}).get('samples',[]))}});return
            if path in ('/api/camera','/api/camera/mode','/api/camera/preview'):
                if role not in ('submit','operator'):self._respond(403,{'error':'role denied'});return
                from .camera_proxy import camera_request, camera_source, MODES
                source,query=camera_source(self.path,allow_mode=path=='/api/camera/preview')
                if source=='emeet':
                    if mutating:raise QueueError('webcam has no presentation controls')
                    if path=='/api/camera':
                        data=camera_request(self.server.webcam_url,'/status')
                        self._respond(200,{k:data[k] for k in ('live','width','height','fps','frame_age_seconds') if k in data});return
                    if path=='/api/camera/preview':
                        if query:raise QueueError('webcam has no presentation modes')
                        data=camera_request(self.server.webcam_url,'/snapshot.jpg',jpeg=True)
                        self._respond(200,data,'image/jpeg');return
                if path=='/api/camera' and not mutating:
                    data=camera_request(self.server.camera_url,'/api/state')
                    mode=data.get('video',{}).get('mode')
                    if mode=='tracked':mode='overlay'
                    if mode not in MODES:raise QueueError('camera mode unavailable')
                    self._respond(200,{'mode':mode,'modes':list(MODES)});return
                if path=='/api/camera/mode' and mutating:
                    self._fields(body,('mode',))
                    if not isinstance(body['mode'],str) or body['mode'] not in MODES:raise QueueError('invalid camera mode')
                    data=camera_request(self.server.camera_url,'/api/video-mode',body={'mode':'tracked' if body['mode']=='overlay' else body['mode']})
                    if data.get('mode')=='tracked':data['mode']='overlay'
                    if data.get('mode') not in MODES:raise QueueError('camera mode unavailable')
                    self._respond(200,{'mode':data['mode'],'modes':list(MODES)});return
                if path=='/api/camera/preview' and not mutating:
                    if query and (set(query)!={'mode'} or len(query['mode'])!=1 or query['mode'][0] not in MODES):raise QueueError('invalid preview mode')
                    endpoint='/preview.jpg?mask=1' if query.get('mode')==['mask'] else '/preview.jpg'
                    data=camera_request(self.server.camera_url,endpoint,jpeg=True)
                    self._respond(200,data,'image/jpeg');return
            if path=='/api/video' and not mutating:
                if role not in ('submit','operator'):self._respond(403,{'error':'role denied'});return
                from .video_proxy import relay_video
                from .camera_proxy import camera_source
                source,_=camera_source(self.path)
                relay_video(self,source=source);return
            if path=='/api/config' and not mutating:
                self._respond(200,{'camera_url':'/api/video' if self.server.camera_url else None,'camera_available':bool(self.server.camera_url),'camera_sources':([{'id':'arducam','label':'Arducam','format':'h264'}] if self.server.camera_url else [])+([{'id':'emeet','label':'EMEET webcam','format':'mjpeg'}] if self.server.webcam_url else []),'reservations_enabled':store.reservations_enabled,'rig_id':self.server.rig_id,'recording_modes':['standard','detailed']});return
            if path=='/api/reservations':
                if role not in ('submit','operator'):self._respond(403,{'error':'role denied'});return
                if not mutating:
                    self._respond(200,{'enabled':store.reservations_enabled,'reservations':store.reservations(owner if role=='submit' else None)});return
                if role!='submit':self._respond(403,{'error':'role denied'});return
                self._fields(body,('rig','starts_at','duration_minutes'))
                if body['rig']!=self.server.rig_id:raise QueueError('unknown reservation rig')
                self._respond(201,store.reserve(owner,**body));return
            reservation_match=re.fullmatch(r'/api/reservations/([a-f0-9]{32})/cancel',path)
            if reservation_match and mutating:
                if role!='submit':self._respond(403,{'error':'role denied'});return
                self._fields(body,())
                self._respond(200,store.cancel_reservation(reservation_match[1],owner));return
            if path=='/api/me' and not mutating:
                self._respond(200,{'principal':owner,'role':role});return
            if path=='/api/jobs':
                if not mutating:
                    if role not in ('submit','operator'):self._respond(403,{'error':'role denied'});return
                    self._respond(200,{'jobs':store.list(owner if role=='submit' else None)});return
                if role!='submit':self._respond(403,{'error':'role denied'});return
                fields={'filename','sensor_source','duration','idempotency_key'}
                options=(fields|{encoding}|optional for encoding in ('source','artifact_base64') for optional in (set(),{'reservation_id'},{'recording'},{'reservation_id','recording'}))
                if set(body) not in options:raise QueueError('request fields mismatch')
                if 'source' in body:
                    if not isinstance(body['source'],str) or not isinstance(body['filename'],str) or not body['filename'].endswith('.py'):raise QueueError('source must be Python text')
                    artifact=body['source'].encode('utf-8')
                else:
                    if not isinstance(body['artifact_base64'],str):raise QueueError('artifact_base64 must be text')
                    try:artifact=base64.b64decode(body['artifact_base64'],validate=True)
                    except ValueError as exc:raise QueueError('invalid artifact base64') from exc
                job=store.submit(owner,body['idempotency_key'],artifact,body['filename'],body['sensor_source'],body['duration'],reservation_id=body.get('reservation_id'),recording=body.get('recording','standard'))
                self._respond(201,job);return
            if path=='/api/claim' and mutating:
                if role!='worker':self._respond(403,{'error':'role denied'});return
                if set(body) not in ({'rig'},{'rig','sensor_source'}):raise QueueError('request fields mismatch')
                self._respond(200,{'job':store.claim(owner,body['rig'],sensor_source=body.get('sensor_source'))});return
            match=re.fullmatch(r'/api/jobs/([a-f0-9]{32})(?:/(cancel|approve|artifact|recording|start|heartbeat|finish))?',path)
            if match:
                identifier,operation=match.groups()
                if operation is None and not mutating:
                    if role not in ('submit','operator'):self._respond(403,{'error':'role denied'});return
                    self._respond(200,store.get(identifier,owner if role=='submit' else None));return
                if operation=='recording':
                    if not mutating:
                        if role not in ('submit','operator'):self._respond(403,{'error':'role denied'});return
                        self._respond(200,store.recording(identifier,owner if role=='submit' else None),'application/zip');return
                    if role!='worker':self._respond(403,{'error':'role denied'});return
                    from .run_recording import MAX_BUNDLE_BYTES
                    lengths=self.headers.get_all('Content-Length',[])
                    if self.headers.get('Transfer-Encoding') is not None or len(lengths)!=1 or not re.fullmatch('[0-9]{1,8}',lengths[0]):raise QueueError('invalid recording body framing')
                    size=int(lengths[0])
                    if not 1<=size<=MAX_BUNDLE_BYTES:raise QueueError('invalid recording size')
                    if self.headers.get_content_type()!='application/zip':raise QueueError('ZIP body required')
                    lease=self.headers.get('X-Worker-Lease','')
                    # Authorize before accepting a potentially large body.
                    with closing(store._connect()) as db:
                        row=db.execute('SELECT worker,lease,status FROM jobs WHERE id=?',(identifier,)).fetchone()
                    if row is None or row['status'] not in ('succeeded','failed','cancelled') or row['worker']!=owner or not isinstance(row['lease'],str) or not hmac.compare_digest(row['lease'],lease):
                        self._respond(403,{'error':'recording worker identity mismatch'});return
                    self.rfile.deadline=time.monotonic()+30
                    raw=self.rfile.read(size)
                    if len(raw)!=size:raise QueueError('incomplete recording body')
                    self._respond(200,store.put_recording(identifier,owner,lease,raw));return
                if operation=='approve' and mutating:
                    if role!='operator':self._respond(403,{'error':'role denied'});return
                    self._fields(body,('artifact','sensor_source','duration'))
                    self._respond(200,store.approve(identifier,owner,**body));return
                if operation=='cancel' and mutating:
                    if role!='submit':self._respond(403,{'error':'role denied'});return
                    self._fields(body,());self._respond(200,store.cancel(identifier,owner));return
                if operation=='artifact' and not mutating:
                    if role not in ('worker','operator'):self._respond(403,{'error':'role denied'});return
                    with closing(store._connect()) as db:
                        if role=='operator':row=db.execute('SELECT artifact,filename FROM jobs WHERE id=?',(identifier,)).fetchone()
                        else:row=db.execute("SELECT artifact,filename FROM jobs WHERE id=? AND worker=? AND status IN ('claimed','running')",(identifier,owner)).fetchone()
                    if row is None:raise QueueError('job unavailable')
                    source=store.artifact(row['artifact'])
                    inspection=inspect_artifact(source,row['filename'])
                    self._respond(200,{'sha256':row['artifact'],'source_base64':base64.b64encode(source).decode(),'filename':row['filename'],'kind':inspection['kind'],'inspection':inspection});return
                if operation in ('start','heartbeat','finish') and mutating:
                    if role!='worker':self._respond(403,{'error':'role denied'});return
                    self._fields(body,('lease','status','result') if operation=='finish' else ('lease',))
                    if operation=='finish':result=store.finish(identifier,owner,body['lease'],body['status'],body['result'])
                    else:result=getattr(store,operation)(identifier,owner,body['lease'])
                    self._respond(200,result);return
            match=re.fullmatch(r'/api/rigs/([A-Za-z0-9_.-]{1,80})/recover',path)
            if match and mutating:
                if role!='operator':self._respond(403,{'error':'role denied'});return
                self._fields(body,('stopped_confirmed',))
                store.release_block(match[1],stopped_confirmed=body['stopped_confirmed'])
                self._respond(200,{'recovered':True});return
            self._respond(404,{'error':'resource unavailable'})
        except (QueueError,ArtifactError) as exc:
            self._respond(404 if str(exc) in ('job unavailable','recording unavailable','reservation unavailable') else 400,{'error':str(exc)})
        except (TimeoutError,socket.timeout):
            self._respond(408,{'error':'request timeout'})
        except (BrokenPipeError,ConnectionResetError):pass
        except Exception:
            self._respond(500,{'error':'service error'})

def make_server(store,credentials,*,host='127.0.0.1',port=8765,origins=(),camera_url=None,webcam_url=None,rig_id='cartpole'):
    if not isinstance(credentials,dict) or not credentials:raise QueueError('credentials required')
    for token,principal in credentials.items():
        if not isinstance(token,str) or len(token)<12 or not isinstance(principal,(tuple,list)) or len(principal)!=2 or principal[1] not in ('submit','worker','operator'):raise QueueError('invalid credential configuration')
        name(principal[0],'principal')
    name(rig_id,'rig')
    for origin in origins:
        parsed=urlsplit(origin)
        if (parsed.scheme not in ('http','https') or not parsed.hostname or parsed.username or parsed.password
                or parsed.path or parsed.query or parsed.fragment or re.search(r'[\s;\"<>]',origin)):
            raise QueueError('allowed origins must be exact HTTP(S) origins without a trailing slash')
    for label,origin in (('camera',camera_url),('webcam',webcam_url)):
        if origin is None:continue
        parsed=urlsplit(origin)
        if (parsed.scheme not in ('http','https') or not parsed.hostname or parsed.username or parsed.password
                or parsed.path not in ('','/') or parsed.query or parsed.fragment
                or any(c.isspace() or c in ";'\"<>" for c in origin)):
            raise QueueError(label+' URL must be a plain HTTP(S) origin')
        try:parsed.port
        except ValueError as exc:raise QueueError('invalid camera port') from exc
    camera_url=camera_url.rstrip('/') if camera_url else None
    webcam_url=webcam_url.rstrip('/') if webcam_url else None
    server=PortalServer((host,port),Handler)
    try:
        # Keep WAL open between short request transactions. Otherwise the final
        # connection close checkpoints on every heartbeat, adding storage stalls
        # to the control lease path. This connection never holds a transaction;
        # Configured synchronous writes and normal size-based checkpoints stay enabled.
        server._database_anchor=sqlite3.connect(store.database,check_same_thread=False)
        server._database_anchor.execute('SELECT name FROM sqlite_master LIMIT 1').fetchall()
    except BaseException:
        server.server_close();raise
    server.camera_url=camera_url;server.webcam_url=webcam_url;server.rig_id=rig_id
    server.store=store;server.credentials=credentials.copy()
    server.origins=set(origins)|{f'http://{host}:{server.server_port}'}
    return server

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--credentials',required=True)
    parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--origin',action='append',default=[])
    parser.add_argument('--camera-url',help='trusted camera HTTP(S) origin; only video, preview, and display mode are relayed')
    parser.add_argument('--webcam-url',help='trusted EMEET HTTP(S) origin; read-only status, snapshot and MJPEG')
    parser.add_argument('--enable-reservations',action='store_true')
    parser.add_argument('--rig-id',default='cartpole')
    args=parser.parse_args(argv)
    credentials=strict_json_loads(Path(args.credentials).read_text())
    server=make_server(Store(args.root,reservations_enabled=args.enable_reservations),credentials,host=args.host,port=args.port,origins=args.origin,camera_url=args.camera_url,webcam_url=args.webcam_url,rig_id=args.rig_id)
    print(f'portal listening at http://{args.host}:{server.server_port}; public hosting is not configured')
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
    return 0
