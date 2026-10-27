"""Authenticated owner API for agents and scripts; never grants worker authority.

Submission uploads the immutable artifact and requests a run atomically. Physical
jobs still require operator approval. No retry is automatic for mutations.
"""
import base64
import hashlib
import io
import math
from pathlib import Path
import re
import time
import zipfile
from .artifacts import MAX_UPLOAD_BYTES, inspect_artifact
from .run_recording import MAX_BUNDLE_BYTES, MAX_TELEMETRY_BYTES
from .worker import RemoteQueue, WorkerError
from cartpole_sdk.recording import strict_json_loads


class PortalClientError(ValueError):
    pass


class PortalClient:
    def __init__(self, origin, token, *, timeout=30):
        self._transport = RemoteQueue(origin, token, timeout=timeout)
        self.origin = self._transport.url

    def _request(self, path, body=None, *, binary=False, timeout=None):
        try:
            return self._transport._request(path, body, response_limit=MAX_BUNDLE_BYTES if binary else 2*1024*1024,
                                            raw_response=binary, deadline_timeout=timeout)
        except WorkerError as exc:
            raise PortalClientError(str(exc)) from exc

    @staticmethod
    def _id(identifier):
        if not isinstance(identifier, str) or re.fullmatch('[a-f0-9]{32}', identifier) is None:
            raise PortalClientError('invalid identifier')
        return identifier

    def me(self):
        return self._request('/api/me')

    def config(self):
        return self._request('/api/config')

    def jobs(self):
        return self._request('/api/jobs')['jobs']

    def get(self, identifier):
        return self._request('/api/jobs/'+self._id(identifier))

    def submit(self, artifact, *, sensor_source, duration, idempotency_key, reservation_id=None, recording='standard'):
        """Upload .py/.zip/actor .json and request execution using a stable retry key."""
        path = Path(artifact)
        with path.open('rb') as stream:
            data = stream.read(MAX_UPLOAD_BYTES+1)
        inspect_artifact(data, path.name)
        body = dict(artifact_base64=base64.b64encode(data).decode(), filename=path.name,
                    sensor_source=sensor_source, duration=duration, idempotency_key=idempotency_key,
                    recording=recording)
        modes=self.config().get('recording_modes',[])
        if recording not in modes:
            if recording!='standard':raise PortalClientError('backend does not support requested recording mode')
            body.pop('recording')
        if reservation_id is not None:
            body['reservation_id'] = self._id(reservation_id)
        return self._request('/api/jobs', body)

    def cancel(self, identifier):
        return self._request('/api/jobs/'+self._id(identifier)+'/cancel', {})

    def reservations(self):
        return self._request('/api/reservations')

    def reserve(self, rig, starts_at, *, duration_minutes=30):
        return self._request('/api/reservations', dict(rig=rig, starts_at=starts_at, duration_minutes=duration_minutes))

    def cancel_reservation(self, identifier):
        return self._request('/api/reservations/'+self._id(identifier)+'/cancel', {})

    @staticmethod
    def _deadline(timeout, poll_interval):
        if any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in (timeout,poll_interval)):
            raise PortalClientError('timeout and poll interval must be finite and positive')
        return time.monotonic()+timeout

    def _poll(self, identifier, deadline, poll_interval, *, recording=False):
        path='/api/jobs/'+self._id(identifier)
        while True:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('portal collection deadline exceeded')
            job=self._request(path, timeout=remaining)
            if job['status'] in ('succeeded','failed','cancelled'):
                metadata=(job.get('result') or {}).get('recording') or {}
                if not recording or metadata.get('status')=='available':return job
                if metadata.get('status')!='local_pending_upload':
                    raise PortalClientError('terminal job has no downloadable recording')
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('portal collection deadline exceeded')
            time.sleep(min(poll_interval,remaining))

    def wait(self, identifier, *, timeout=120, poll_interval=.5):
        """Wait for execution outcome. Recording packaging may still be pending."""
        return self._poll(identifier,self._deadline(timeout,poll_interval),poll_interval)

    def download(self, identifier, destination, *, job=None, timeout=None):
        """Verify ZIP and telemetry hashes/identity before exclusive file creation."""
        identifier=self._id(identifier)
        job=self.get(identifier) if job is None else job
        metadata=(job.get('result') or {}).get('recording') or {}
        if job.get('id')!=identifier or metadata.get('status')!='available':
            raise PortalClientError('recording unavailable')
        raw=self._request('/api/jobs/'+identifier+'/recording',binary=True,timeout=timeout)
        if len(raw)!=metadata.get('bytes') or hashlib.sha256(raw).hexdigest()!=metadata.get('sha256'):
            raise PortalClientError('recording integrity failure')
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                entries=archive.infolist()
                if len(entries)!=2 or {e.filename for e in entries}!={'manifest.json','telemetry.jsonl'}:raise ValueError()
                if any(e.compress_type!=zipfile.ZIP_STORED or e.flag_bits&1 for e in entries):raise ValueError()
                if archive.getinfo('manifest.json').file_size>1024*1024 or archive.getinfo('telemetry.jsonl').file_size>MAX_TELEMETRY_BYTES:raise ValueError()
                manifest=strict_json_loads(archive.read('manifest.json').decode())
                if manifest['schema']!='cartpole-run-bundle-v1' or manifest['job']['id']!=identifier or manifest['job']['artifact']!=job['artifact']:raise ValueError()
                telemetry=archive.read('telemetry.jsonl')
                if hashlib.sha256(telemetry).hexdigest()!=manifest['recording']['telemetry_sha256']:raise ValueError()
        except (ValueError,KeyError,TypeError,UnicodeError,zipfile.BadZipFile,RuntimeError) as exc:
            raise PortalClientError('invalid recording bundle') from exc
        destination=Path(destination)
        with destination.open('xb') as stream:stream.write(raw)
        return manifest

    def collect(self, identifier, destination, *, timeout=120, poll_interval=.5):
        """Wait for outcome AND completed upload, then verify and save the bundle."""
        deadline=self._deadline(timeout,poll_interval)
        job=self._poll(identifier,deadline,poll_interval,recording=True)
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('portal collection deadline exceeded')
        return self.download(identifier,destination,job=job,timeout=remaining)
