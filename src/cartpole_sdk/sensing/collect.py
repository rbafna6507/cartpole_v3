"""Bounded read-only NDJSON collection; no hardware commands or clock rewriting."""
import argparse
import http.client
import json
import math
import sys
import time
from urllib.parse import urlencode, urlsplit
from .stream import SensingStreamClient, StreamDeadline, _DeadlineSocket
from ..recording import strict_json_loads


def collect(url, output, *, streams=None, duration=30, max_hz=30, limit=128,
            cursors=None, max_records=100000, max_bytes=256*1024*1024,
            legacy_sensing=False):
    if not math.isfinite(duration) or not .001 <= duration <= 3600:
        raise ValueError('duration must be .001..3600 seconds')
    if not 1 <= max_hz <= 100 or not 1 <= limit <= 128 or max_records < 1 or max_bytes < 1:
        raise ValueError('invalid rate, sample, record or byte bound')
    parsed = urlsplit(url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ('', '/'):
        raise ValueError('URL must be an HTTP(S) service origin without credentials/query/path')
    deadline = time.monotonic() + duration
    count = size = 0
    def write(envelope):
        nonlocal count, size
        line = json.dumps(envelope, allow_nan=False, separators=(',', ':')) + '\n'
        length = len(line.encode())
        if size + length > max_bytes: return False
        output.write(line); output.flush(); size += length; count += 1
        return count < max_records
    if legacy_sensing:
        if streams or cursors: raise ValueError('legacy mode exports complete v1 vision+encoder envelopes; selection/cursors require the new endpoint')
        if max_hz < 10: raise ValueError('legacy max_hz must be 10..100')
        iterator = SensingStreamClient(url.rstrip('/')+'/api/sensing-stream', timeout_s=min(5,duration)).messages(deadline=deadline, max_hz=max_hz)
        try:
            for envelope, _ in iterator:
                if not write(envelope): break
        except StreamDeadline: pass
        finally: iterator.close()
        return count
    query = dict(duration=duration, max_hz=max_hz, limit=limit)
    if streams: query['streams'] = ','.join(streams)
    if cursors: query['cursors'] = json.dumps(cursors, separators=(',', ':'))
    factory = http.client.HTTPSConnection if parsed.scheme == 'https' else http.client.HTTPConnection
    connection = factory(parsed.hostname, parsed.port, timeout=min(5, duration))
    connection.response_class = lambda sock, *a, **kw: http.client.HTTPResponse(
        _DeadlineSocket(sock, deadline=deadline, timeout_s=5, clock=time.monotonic), *a, **kw)
    response = None
    try:
        connection.request('GET', '/api/streams/ndjson?'+urlencode(query), headers={'Accept':'application/x-ndjson', 'Accept-Encoding':'identity'})
        response = connection.getresponse()
        if response.status != 200: raise ValueError(f'stream HTTP {response.status}; old service: explicitly use --legacy-sensing')
        if response.getheader('Content-Encoding', 'identity') != 'identity': raise ValueError('compressed response unsupported')
        while time.monotonic() < deadline:
            line = response.readline(4*1024*1024+1)
            if not line: break
            if len(line) > 4*1024*1024 or not line.endswith(b'\n'): raise ValueError('oversized or incomplete stream record')
            record = strict_json_loads(line.decode())
            if not isinstance(record, dict) or record.get('schema') != 'cartpole-developer-stream-v1': raise ValueError('unexpected stream schema')
            if time.monotonic() >= deadline: break
            if not write(record): break
    except (StreamDeadline, TimeoutError):
        if time.monotonic() < deadline: raise
    finally:
        if response: response.close()
        connection.close()
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:3090')
    parser.add_argument('--streams', help='comma-separated catalog channel ids')
    parser.add_argument('--duration', type=float, default=30)
    parser.add_argument('--max-hz', type=int, default=30)
    parser.add_argument('--limit', type=int, default=128)
    parser.add_argument('--max-records', type=int, default=100000)
    parser.add_argument('--max-bytes', type=int, default=256*1024*1024)
    parser.add_argument('--cursors', help='JSON cursor map copied from a previous envelope')
    parser.add_argument('--output', default='-', help='- for stdout; otherwise creates a NEW JSONL file')
    parser.add_argument('--legacy-sensing', action='store_true', help='export unchanged v1 vision+encoder envelopes from an existing service; SDK shadow is not included')
    args = parser.parse_args(argv)
    output = None
    try:
        output = sys.stdout if args.output == '-' else open(args.output, 'x', encoding='utf-8')
        count = collect(args.url, output, streams=args.streams.split(',') if args.streams else None,
                duration=args.duration, max_hz=args.max_hz, limit=args.limit,
                cursors=strict_json_loads(args.cursors) if args.cursors else None,
                max_records=args.max_records, max_bytes=args.max_bytes, legacy_sensing=args.legacy_sensing)
        if count == 0:
            print('collection failed: no complete records received within the duration/output bounds', file=sys.stderr)
            return 1
        return 0
    except BrokenPipeError: return 0
    except (OSError, ValueError, http.client.HTTPException) as exc:
        print(f'collection failed: {exc}', file=sys.stderr); return 1
    finally:
        if output is not None and output is not sys.stdout: output.close()


if __name__ == '__main__':
    raise SystemExit(main())
