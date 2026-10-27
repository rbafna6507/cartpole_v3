#!/usr/bin/env python3
"""Export frozen learned weights to one source file for the isolated portal."""
import argparse
import json
from pathlib import Path
from cartpole_sdk.learning.export import export_policy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',choices=('simulation','magnetic-step'),required=True)
    args=parser.parse_args()
    manifest=export_policy(args.artifact,args.output,source=args.source)
    manifest_path=args.output.with_suffix('.manifest.json')
    manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(f"Saved {args.output} ({manifest['source_bytes']} bytes)")
    print(f"Source SHA-256: {manifest['source_sha256']}")
    print(f"Manifest: {manifest_path}; hardware_accepted=false")


if __name__=='__main__':main()
