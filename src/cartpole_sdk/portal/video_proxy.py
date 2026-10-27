"""Fixed-origin, read-only video relay; no browser-supplied URLs or credentials."""
import http.client
import time
import re
from urllib.parse import urlsplit
from cartpole_sdk.sensing.stream import _DeadlineSocket, StreamDeadline

VIDEO_SECONDS=60


def relay_video(handler, *, source='arducam'):
    webcam=source=='emeet'
    origin=handler.server.webcam_url if webcam else handler.server.camera_url
    if origin is None:
        handler._respond(503,{'error':'camera not configured'});return
    if not handler.server.video_slots.acquire(blocking=False):
        handler._respond(429,{'error':'camera viewer limit reached'});return
    connection=None;response=None;started=False
    deadline=time.monotonic()+VIDEO_SECONDS
    try:
        parts=urlsplit(origin)
        factory=http.client.HTTPSConnection if parts.scheme=='https' else http.client.HTTPConnection
        connection=factory(parts.hostname,parts.port,timeout=min(3,VIDEO_SECONDS))
        connection.response_class=lambda sock,*args,**kwargs: http.client.HTTPResponse(
            _DeadlineSocket(sock,deadline=deadline,timeout_s=3,clock=time.monotonic),*args,**kwargs)
        # Fixed endpoint only. No redirects, cookies, or portal bearer forwarded.
        kind='multipart/x-mixed-replace' if webcam else 'application/octet-stream'
        connection.request('GET','/stream.mjpg' if webcam else '/video.h264',headers={'Accept':kind})
        response=connection.getresponse()
        content_type=response.getheader('Content-Type','')
        if (response.status!=200 or content_type.split(';')[0]!=kind or
                (webcam and re.fullmatch(r'multipart/x-mixed-replace;\s*boundary=[A-Za-z0-9_-]{1,70}',content_type) is None)):
            handler._respond(502,{'error':'camera video unavailable'});return
        handler.send_response(200)
        handler.send_header('Content-Type',content_type)
        handler.send_header('Cache-Control','no-store')
        handler.send_header('X-Content-Type-Options','nosniff')
        handler.send_header('X-Accel-Buffering','no')
        handler.send_header('Connection','close')
        handler.end_headers();started=True
        handler.connection.settimeout(3)
        remaining_bytes=64*1024*1024
        while time.monotonic()<deadline and remaining_bytes>0:
            data=response.read1(min(65536,remaining_bytes))
            if not data:break
            handler.wfile.write(data);handler.wfile.flush();remaining_bytes-=len(data)
    except (OSError,http.client.HTTPException,StreamDeadline):
        if not started:handler._respond(502,{'error':'camera video unavailable'})
    finally:
        handler.close_connection=True
        if response is not None:response.close()
        if connection is not None:connection.close()
        handler.server.video_slots.release()
