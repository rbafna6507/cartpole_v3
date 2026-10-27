"""Persistent portal service configured by a private JSON file; no worker starts."""
import argparse
import json
from pathlib import Path
from .runtime_store import RuntimeStore
from .server import make_server


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args(argv)
    config=json.loads(args.config.read_text())
    allowed={'root','credentials','origins','camera_url','webcam_url','rig_id','port','reservations_enabled'}
    if set(config)-allowed:raise ValueError('unknown portal service setting')
    credentials=json.loads(Path(config['credentials']).read_text())
    store=RuntimeStore(config['root'],reservations_enabled=config.get('reservations_enabled',False))
    store.recover()  # a prior API process cannot retain live runtime authority
    server=make_server(store,
        credentials,host='127.0.0.1',port=config.get('port',8780),
        origins=config.get('origins',[]),camera_url=config.get('camera_url'),webcam_url=config.get('webcam_url'),rig_id=config.get('rig_id','cartpole'))
    print(f'Authenticated portal API on loopback:{server.server_port}; no execution worker started',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        server.server_close()
        store.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
