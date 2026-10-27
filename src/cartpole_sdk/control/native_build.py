"""Explicit trusted build; never invoked by policy loading or a running worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile


def build(source_root, output, *, compiler='c++'):
    root=Path(source_root).resolve();output=Path(output).resolve()
    headers=root/'firmware/teensy_executor'
    source=root/'src/cartpole_sdk/control/native_controller.cpp'
    if not source.is_file() or not (headers/'swing_controller.h').is_file():
        raise ValueError('full cartpole source checkout required')
    header_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(headers.glob('*.h'))}
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    flags=['-std=c++17','-O2','-fno-fast-math','-fPIC','-dynamiclib' if platform.system()=='Darwin' else '-shared']
    compiler_version=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0]
    provenance=dict(schema=1,abi=1,source_sha256=source_hash,headers_sha256=header_hashes,
                    compiler=compiler_version,flags=flags,machine=platform.machine(),system=platform.system())
    digest=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()[:20]
    output.mkdir(parents=True,exist_ok=True)
    name=f'controller-{digest}'+('.dylib' if platform.system()=='Darwin' else '.so')
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        binary=Path(temporary)/name
        subprocess.run([compiler,*flags,'-I',str(headers),str(source),'-o',str(binary)],check=True,capture_output=True,text=True)
        provenance['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
        manifest=Path(temporary)/(name+'.json');manifest.write_text(json.dumps(provenance,indent=2)+'\n')
        os.replace(binary,output/name);os.replace(manifest,output/(name+'.json'))
        pointer=Path(temporary)/'current.json';pointer.write_text(json.dumps({'library':name})+'\n')
        os.replace(pointer,output/'current.json')
    return output/name


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'_native')
    parser.add_argument('--compiler',default='c++')
    args=parser.parse_args()
    print(build(args.source_root,args.output,compiler=args.compiler))

if __name__=='__main__':main()
