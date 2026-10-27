"""Bounded requests to the configured camera; never forward browser credentials."""
import http.client
import time
from urllib.parse import urlsplit, parse_qs
from cartpole_sdk.recording import canonical_json, strict_json_loads
from cartpole_sdk.sensing.stream import _DeadlineSocket, StreamDeadline
from .queue import QueueError

MODES=('camera','overlay','mask')
CAMERA_SECONDS=3
MAX_CAMERA_BYTES=2*1024*1024

def camera_request(origin,path,*,body=None,jpeg=False):
    if origin is None:raise QueueError('camera not configured')
    parts=urlsplit(origin)
    factory=http.client.HTTPSConnection if parts.scheme=='https' else http.client.HTTPConnection
    connection=factory(parts.hostname,parts.port,timeout=CAMERA_SECONDS)
    deadline=time.monotonic()+CAMERA_SECONDS
    connection.response_class=lambda sock,*args,**kwargs:http.client.HTTPResponse(
        _DeadlineSocket(sock,deadline=deadline,timeout_s=CAMERA_SECONDS,clock=time.monotonic),*args,**kwargs)
    try:
        headers={'Accept':'image/jpeg' if jpeg else 'application/json'}
        data=None
        if body is not None:
            data=canonical_json(body).encode();headers['Content-Type']='application/json'
        connection.request('POST' if body is not None else 'GET',path,body=data,headers=headers)
        with connection.getresponse() as response:
            expected='image/jpeg' if jpeg else 'application/json'
            if response.status!=200 or response.getheader('Content-Type','').split(';')[0]!=expected:
                raise QueueError('camera response unavailable')
            data=response.read(MAX_CAMERA_BYTES+1)
            if len(data)>MAX_CAMERA_BYTES:raise QueueError('camera response too large')
            if jpeg:return data
            result=strict_json_loads(data.decode())
            if not isinstance(result,dict):raise ValueError('expected object')
            return result
    except (OSError,http.client.HTTPException,StreamDeadline,ValueError) as exc:
        raise QueueError('camera response unavailable') from exc
    finally:connection.close()


def camera_source(path, *, allow_mode=False):
    query=parse_qs(urlsplit(path).query,keep_blank_values=True)
    if set(query)-({'source','mode'} if allow_mode else {'source'}):
        raise QueueError('invalid camera query')
    sources=query.pop('source',['arducam'])
    if len(sources)!=1 or sources[0] not in ('arducam','emeet'):
        raise QueueError('invalid camera source')
    return sources[0],query
