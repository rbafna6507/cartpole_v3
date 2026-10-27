"""Pure source adapters; providers require explicit clocks and calibration."""
from .vision import ClockMapping, VisionProvider, VisionHTTPClient
__all__ = ['ClockMapping', 'VisionProvider', 'VisionHTTPClient']

from .frames import AngularAlignment
from .fusion import AngleFusion, FusionConfig, SourceIdentity
from .shadow import ShadowObserver
from .telemetry import EncoderPublisher
__all__ += ["AngularAlignment", "AngleFusion", "FusionConfig", "SourceIdentity", "ShadowObserver", "EncoderPublisher"]

from .stream import SensingStreamClient, StreamDeadline, StreamDisconnected
__all__ += ["SensingStreamClient", "StreamDeadline", "StreamDisconnected"]
