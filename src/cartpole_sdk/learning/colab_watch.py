"""Local checkpoint downloads; never allocate, restart, or keep a runtime alive."""
import json
import math
from pathlib import Path
import time


def watch_job(status, download, output, *, poll_seconds=60., max_seconds=28800.):
    for value in (poll_seconds, max_seconds):
        if not math.isfinite(value) or value <= 0:
            raise ValueError('watch intervals must be finite and positive')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    last_snapshot = None
    seen = ()
    started = time.monotonic()

    def record(event, **fields):
        row = dict(time=time.time(), event=event, **fields)
        with (output/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        return row

    def finish(state, **fields):
        value = dict(status=state, last_snapshot=last_snapshot, time=time.time(), **fields)
        temporary = output/'watch-status.tmp'
        temporary.write_text(json.dumps(value, indent=2)+'\n')
        temporary.replace(output/'watch-status.json')
        return value

    try:
        while True:
            current = status()
            if not isinstance(current, dict) or current.get('status') not in {'installed','training','evaluating','completed','failed','interrupted'}:
                raise ValueError('missing structured job status; watch requires a newly prepared bundle')
            record('status', remote=current)
            checkpoints = tuple(current.get('checkpoints', []))
            terminal = current['status'] in {'completed','failed','interrupted'}
            if (checkpoints and checkpoints != seen) or terminal:
                destination = output/f'snapshot-{time.time_ns()}.zip'
                receipt = download(destination)
                last_snapshot = destination.name
                record('snapshot', path=last_snapshot, **receipt)
                seen = checkpoints
            finish('watching', remote=current)
            if terminal:
                return finish(current['status'], remote=current)
            if time.monotonic()-started >= max_seconds:
                return finish('timeout', note='Watcher stopped; remote job was not stopped.')
            time.sleep(max(0.,min(poll_seconds, max_seconds-(time.monotonic()-started))))
    except KeyboardInterrupt:
        finish('interrupted_locally', note='Remote job was not stopped.')
        raise
    except Exception as error:
        record('error', error=str(error))
        finish('connection_error', error=str(error))
        raise
