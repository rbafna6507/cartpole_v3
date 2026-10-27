"""Explicit, fail-closed binding to a prebuilt production-header controller."""
import ctypes as ct
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import weakref
import threading
from .parameters import ControllerParameters


def _numbers(*values):
    if any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>3.4028234e38 for v in values):
        raise ValueError('native inputs must be finite float32 numbers')
    return values


class NativeController:
    def __init__(self, parameters, *, library=None, upright=False):
        if not isinstance(parameters,ControllerParameters) or type(upright) is not bool:
            raise ValueError('explicit validated parameters and upright mode required')
        if library is None:
            root=Path(__file__).parent/'_native'
            name=json.loads((root/'current.json').read_text())['library']
            if not isinstance(name,str) or Path(name).name!=name:
                raise ValueError('invalid native library selection')
            library=root/name
        library=Path(library).resolve()
        self.provenance=json.loads(Path(str(library)+'.json').read_text())
        if (self.provenance.get('schema')!=1 or self.provenance.get('abi')!=1 or
            hashlib.sha256(library.read_bytes()).hexdigest()!=self.provenance.get('binary_sha256')):
            raise ValueError('native library provenance mismatch')
        lib=ct.CDLL(str(library));self._lib=lib
        lib.cp_abi.argtypes=[];lib.cp_abi.restype=ct.c_uint
        lib.cp_fields.argtypes=[];lib.cp_fields.restype=ct.c_uint
        lib.cp_field.argtypes=[ct.c_uint];lib.cp_field.restype=ct.c_char_p
        lib.cp_create.argtypes=[ct.POINTER(ct.c_float),ct.c_uint,ct.c_int];lib.cp_create.restype=ct.c_void_p
        lib.cp_destroy.argtypes=[ct.c_void_p];lib.cp_destroy.restype=None
        lib.cp_reset.argtypes=[ct.c_void_p];lib.cp_reset.restype=None
        lib.cp_gains.argtypes=[ct.c_void_p,ct.POINTER(ct.c_float)];lib.cp_gains.restype=ct.c_int
        lib.cp_advance.argtypes=[ct.c_void_p]+[ct.c_float]*6+[ct.c_int,ct.POINTER(ct.c_float)];lib.cp_advance.restype=ct.c_int
        lib.cp_velocity.argtypes=[ct.c_float]*6;lib.cp_velocity.restype=ct.c_float
        values=asdict(parameters)
        if lib.cp_abi()!=1 or lib.cp_fields()!=len(values):raise ValueError('native ABI/catalog mismatch')
        names=[lib.cp_field(i).decode('ascii') for i in range(lib.cp_fields())]
        if set(names)!=set(values):raise ValueError('native parameter catalog mismatch')
        supplied=_numbers(*(values[n] for n in names))
        handle=lib.cp_create((ct.c_float*len(names))(*supplied),len(names),int(upright))
        if not handle:raise ValueError('native controller creation failed')
        self._lock=threading.RLock()
        self._handle=handle;self._output=(ct.c_float*4)()
        self._finalizer=weakref.finalize(self,lib.cp_destroy,handle)
        self._upright=upright

    def _require(self):
        if not self._finalizer.alive:raise ValueError('native controller closed')
    def close(self):
        with self._lock:self._finalizer()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def reset(self):
        with self._lock:
            self._require();self._lib.cp_reset(self._handle)
    @property
    def gains(self):
        with self._lock:
            self._require()
            if not self._lib.cp_gains(self._handle,self._output):raise ValueError('native gains failed')
            return tuple(self._output)
    def advance(self,theta,omega,x,v,acceleration,dt,*,balancing):
        with self._lock:
            self._require();_numbers(theta,omega,x,v,acceleration,dt)
            if type(balancing) is not bool or not 0<dt<=.02:raise ValueError('invalid controller cadence/mode')
            if not self._lib.cp_advance(self._handle,theta,omega,x,v,acceleration,dt,int(balancing),self._output):
                raise ValueError('native controller rejected state')
            return float(self._output[0]),bool(self._output[1])
    def velocity_accel(self,v,a,target,cap,jerk,dt):
        with self._lock:
            self._require();_numbers(v,a,target,cap,jerk,dt)
            if cap<=0 or jerk<=0 or not 0<dt<=.02:raise ValueError('invalid motion planner limits')
            value=float(self._lib.cp_velocity(v,a,target,cap,jerk,dt))
            if not math.isfinite(value):raise ValueError('native motion planner rejected state')
            return value
