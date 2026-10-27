from dataclasses import replace
import math
import pytest
from cartpole_sdk.contracts import Observation, Sample, Session
from cartpole_sdk.sensing.fusion import AngleFusion, FusionConfig, SourceIdentity
from cartpole_sdk.sensing.frames import AngularAlignment

SESSION = Session('shadow', 'host')
ENC = SourceIdentity('encoder-rig-a', 'upright-370', 'board-host-map-v1')
VIS = SourceIdentity('camera-a', 'camera-anchor-v1', 'camera-host-map-v1')


def observation(source, t, value, *, received=None, frame='upright', valid=True):
    received=t if received is None else received
    sample=Sample('pole','angle',source,value if valid else None,'rad',frame,'test',
                  'valid' if valid else 'invalid',None if valid else 'sensor unavailable',
                  'measured',t,t,received)
    return Observation('shadow','host',t,received,'explicit test clock',(sample,))


def fusion(**changes):
    return AngleFusion(SESSION,frame_id='upright',encoder_identity=ENC,vision_identity=VIS,
                       config=FusionConfig(**dict(dict(bias_time_constant_s=.1,
                           max_bias_rate_rad_s=100.,max_innovation_rad=.5),**changes)))


def test_delayed_camera_compares_history_not_latest_encoder():
    f=fusion()
    for t,v in [(0,0.),(10_000_000,.1),(20_000_000,.2),(30_000_000,.3)]:
        f.update_encoder(observation('magnetic',t,v),identity=ENC)
    assert f.update_vision(observation('vision',15_000_000,.25,received=35_000_000),identity=VIS)
    assert f.diagnostics['innovation_rad']==pytest.approx(.1)
    assert 0 < f.diagnostics['bias_rad'] < .1
    out=f.observation(now_ns=35_000_000)
    assert out.samples[-1].value==pytest.approx(.3+f.diagnostics['bias_rad'])
    assert out.samples[-1].effective_ns==30_000_000
    assert out.samples[-1].received_ns==35_000_000
    assert out.samples[1].effective_ns==15_000_000
    assert f.diagnostics['control_eligible'] is False


def test_interpolation_and_innovation_wrap_at_pi():
    f=fusion()
    f.update_encoder(observation('magnetic',0,math.pi-.02),identity=ENC)
    f.update_encoder(observation('magnetic',10_000_000,-math.pi+.02),identity=ENC)
    assert f.update_vision(observation('vision',5_000_000,-math.pi+.03,received=10_000_000),identity=VIS)
    assert f.diagnostics['innovation_rad']==pytest.approx(.03)


def test_no_extrapolation_or_large_encoder_gap():
    f=fusion(max_interpolation_gap_ns=20_000_000)
    f.update_encoder(observation('magnetic',0,0),identity=ENC)
    f.update_encoder(observation('magnetic',50_000_000,.2),identity=ENC)
    assert not f.update_vision(observation('vision',25_000_000,.1,received=50_000_000),identity=VIS)
    assert 'gap' in f.diagnostics['reason']
    assert not f.update_vision(observation('vision',55_000_000,.1,received=60_000_000),identity=VIS)
    assert f.observation(now_ns=60_000_000).samples[-1].validity=='invalid'


def test_dropout_preserves_sources_but_invalidates_fusion():
    f=fusion(max_vision_age_ns=50_000_000,max_encoder_age_ns=20_000_000)
    f.update_encoder(observation('magnetic',0,.1),identity=ENC)
    f.update_vision(observation('vision',0,.2),identity=VIS)
    f.update_encoder(observation('magnetic',60_000_000,.15),identity=ENC)
    out=f.observation(now_ns=60_000_000)
    assert out.samples[0].value==.15
    assert out.samples[-1].value is None
    assert 'vision' in out.samples[-1].reason
    out=f.observation(now_ns=100_000_000)
    assert out.samples[0].validity=='invalid'


def test_outlier_and_stale_camera_do_not_adjust_bias():
    f=fusion()
    f.update_encoder(observation('magnetic',0,0),identity=ENC)
    assert not f.update_vision(observation('vision',0,2),identity=VIS)
    assert f.diagnostics['bias_rad']==0
    f.update_encoder(observation('magnetic',1_000_000,0),identity=ENC)
    assert not f.update_vision(observation('vision',1_000_000,.1,received=1_000_000_000),identity=VIS)
    assert f.diagnostics['bias_rad']==0


@pytest.mark.parametrize('which', ['frame','clock','identity','rollback'])
def test_incompatible_provenance_or_clock_reset_latches_until_reconstruction(which):
    f=fusion();f.update_encoder(observation('magnetic',10_000_000,.1),identity=ENC)
    obs=observation('magnetic',20_000_000,.2)
    ident=ENC
    if which=='frame':obs=replace(obs,samples=(replace(obs.samples[0],frame_id='hanging'),))
    if which=='clock':obs=replace(obs,clock_id='other-host')
    if which=='identity':ident=replace(ENC,calibration_id='new-upright')
    if which=='rollback':obs=observation('magnetic',5_000_000,.1,received=20_000_000)
    with pytest.raises(ValueError):f.update_encoder(obs,identity=ident)
    assert f.diagnostics['status']=='reset_required'
    assert f.observation(now_ns=20_000_000).samples[-1].validity=='invalid'
    with pytest.raises(ValueError,match='reset'):
        f.update_encoder(observation('magnetic',30_000_000,.2),identity=ENC)


def test_invalid_encoder_clears_bias_and_requires_new_vision():
    f=fusion();f.update_encoder(observation('magnetic',0,0),identity=ENC)
    f.update_vision(observation('vision',0,.1),identity=VIS)
    f.update_encoder(observation('magnetic',10_000_000,0,valid=False),identity=ENC)
    f.update_encoder(observation('magnetic',20_000_000,0),identity=ENC)
    assert f.diagnostics['bias_rad']==0
    assert f.observation(now_ns=20_000_000).samples[-1].validity=='invalid'


def test_history_bounded_and_bias_rate_limited():
    f=fusion(history_capacity=3,max_bias_rate_rad_s=.2)
    for t in range(10):f.update_encoder(observation('magnetic',t*10_000_000,0),identity=ENC)
    assert f.diagnostics['history_size']==3
    assert f.update_vision(observation('vision',90_000_000,.4),identity=VIS)
    assert abs(f.diagnostics['bias_rad'])<=.0021
    assert not f.update_vision(observation('vision',90_000_000,.4),identity=VIS)


def test_angular_alignment_is_explicit_and_changes_values_not_just_labels():
    anchor=AngularAlignment(source='vision',source_id='camera-a',source_frame_id='legacy-hanging',
        source_calibration_id='old-hash',target_frame_id='upright',reference_id='direct-upright-2026',
        zero_rad=.0319,sign=-1)
    obs=observation('vision',10_000_000,.1319,frame='legacy-hanging')
    out=anchor.apply(obs,source_id='camera-a',source_calibration_id='old-hash')
    assert out.samples[0].value==pytest.approx(-.1)
    assert out.samples[0].frame_id=='upright'
    with pytest.raises(ValueError,match='identity'):
        anchor.apply(obs,source_id='camera-b',source_calibration_id='old-hash')
    with pytest.raises(ValueError,match='frame'):
        anchor.apply(replace(obs,samples=(replace(obs.samples[0],frame_id='upright'),)),source_id='camera-a',source_calibration_id='old-hash')


def test_output_clock_rollback_cannot_resurrect_a_stale_correction():
    f=fusion()
    f.update_encoder(observation('magnetic',0,0),identity=ENC)
    f.update_vision(observation('vision',0,0),identity=VIS)
    f.observation(now_ns=1_000_000_000)
    with pytest.raises(ValueError,match='rollback'):
        f.observation(now_ns=1)
    assert f.diagnostics['status']=='reset_required'
