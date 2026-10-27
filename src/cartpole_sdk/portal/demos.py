"""Fixed, reviewable demonstration artifacts; never browser-supplied paths."""
from pathlib import Path
from .artifacts import inspect_artifact
NATIVE_AUTO_SOURCE=b'"""Trusted native AUTO demo marker. Exact bytes select firmware execution.\nThe sandbox returns no host actions; all normal approval and lease checks apply.\n"""\nclass Policy:\n    def reset(self, session): pass\n    def act(self, observation): return None\ndef create_policy(): return Policy()\n'

def demo_artifact(identifier):
    if identifier=='controller':return NATIVE_AUTO_SOURCE,'native-auto-demo.py'
    if identifier=='policy':
        data=(Path(__file__).resolve().parents[3]/'policies/working.json').read_bytes()
        inspect_artifact(data,'working.json')
        return data,'working.json'
    raise ValueError('unknown demo')

def catalog():
    result=[]
    for identifier,label in (('controller','Physics-based controller'),('policy','Learned policy')):
        try:demo_artifact(identifier);available=True
        except (OSError,ValueError):available=False
        result.append(dict(id=identifier,label=label,duration=20,available=available))
    return result
