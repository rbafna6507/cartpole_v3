"""Persistent, loopback-only source-policy preview. Never starts a worker."""
import argparse
import json
import os
from pathlib import Path
import secrets
import stat
from urllib.parse import urlsplit

from .queue import Store
from .server import Handler, make_server


class PreviewHandler(Handler):
    def do_GET(self):
        if urlsplit(self.path).path == '/api/dev-info':
            self._respond(200, {'local_preview': True, 'camera_url': 'http://127.0.0.1:3090'})
            return
        super().do_GET()


def prepare_credentials(root):
    """Create private credentials once; subsequent launches reuse the same file."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'credentials.json'
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
            raise ValueError('credentials must be a regular private file')
        os.chmod(path, 0o600)
    else:
        credentials = {
            secrets.token_hex(32): ['local-submitter', 'submit'],
            secrets.token_hex(32): ['local-operator', 'operator'],
        }
        with os.fdopen(fd, 'w') as stream:
            json.dump(credentials, stream, indent=2)
            stream.write('\n')
    credentials = json.loads(path.read_text())
    if (not isinstance(credentials, dict) or len(credentials) != 2
            or any(not isinstance(p, list) or len(p) != 2 for p in credentials.values())
            or {p[1] for p in credentials.values()} != {'submit', 'operator'}
            or any(len(token) != 64 or any(c not in '0123456789abcdef' for c in token)
                   for token in credentials)):
        raise ValueError('invalid local preview credentials; existing file was preserved')
    return path, credentials


def create_preview(root, *, port=8765):
    path, credentials = prepare_credentials(root)
    server = make_server(Store(Path(root) / 'store', reservations_enabled=True), credentials, host='127.0.0.1', port=port, camera_url='http://127.0.0.1:3090')
    server.RequestHandlerClass = PreviewHandler
    return server, path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runs/portal-dev'))
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args(argv)
    server, path = create_preview(args.root, port=args.port)
    print(f'Local portal: http://127.0.0.1:{server.server_port}', flush=True)
    print(f'Private credentials: {path.resolve()}', flush=True)
    print('Load that file in the portal, then choose submitter or operator. No worker is running.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
