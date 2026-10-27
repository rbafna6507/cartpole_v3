"""Optional pyserial adapter; importing the SDK never opens a device."""
from typing import Protocol

class Transport(Protocol):
    def read(self, size: int) -> bytes: ...
    def write(self, data: bytes) -> int: ...
    def close(self) -> None: ...

class SerialTransport:
    def __init__(self, port: str):
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError('live serial requires pip install cartpole-sdk[hardware]') from exc
        self._serial = serial.Serial(port=port, baudrate=115200, timeout=.005,
                                     write_timeout=.05, exclusive=True)
        self._serial.reset_input_buffer()
    def read(self, size):
        # Wait only for the first byte when idle, then drain what is available.
        # pyserial.read(size) otherwise delays every short frame until timeout.
        return self._serial.read(min(size, max(1, self._serial.in_waiting)))
    def write(self, data):
        return self._serial.write(data)
    def close(self):
        self._serial.close()
