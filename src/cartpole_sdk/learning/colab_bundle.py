"""Immutable selected-source bundles for the official Colab CLI (no network)."""
from hashlib import sha256
import json
from pathlib import Path
import zipfile

from .model_experiment import build_experiment, read_json, digest

INPUTS = ('model.json', 'executor-parameters.json', 'observation-config.json', 'prior-parameters.json')
SOURCE_SUFFIXES = {'.py', '.json', '.xml', '.html', '.js', '.css', '.sh'}


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def _hash(path):
    return sha256(path.read_bytes()).hexdigest()


def prepare_bundle(repo, profile, output, *, steps=2_097_152, seed=20261017, seconds=25.,
                   checkpoint_every=102400, evaluation_seeds=tuple(range(20000,20016))):
    repo, profile, output = map(Path, (repo, profile, output))
    if output.exists():
        raise FileExistsError(output)
    if type(steps) is not int or steps < 1024 or steps % 1024:
        raise ValueError('steps must be a positive multiple of the 1024-step PPO rollout')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an unsigned 32-bit integer')
    if type(checkpoint_every) is not int or checkpoint_every < 1024 or checkpoint_every % 1024:
        raise ValueError('checkpoint_every must be a positive multiple of1024')
    from .model_search import _seeds
    evaluation_seeds = _seeds(evaluation_seeds, 'evaluation seeds')
    if set(evaluation_seeds) & set(range(10000,10128)):
        raise ValueError('training and evaluation seeds overlap')
    extra_profile = []
    if (profile/'bundle.json').exists():
        manifest,_ = read_json(profile/'bundle.json')
        content=dict(manifest);expected=content.pop('content_sha256',None)
        profile_extras = {
            'cartpole-passive-training-bundle-v1': {'provenance.json', 'frozen-candidate.json', 'assumptions.json'},
            'cartpole-evidence-training-bundle-v1': {'provenance.json', 'randomization.json'},
        }
        extras = profile_extras.get(manifest.get('schema'))
        allowed = set(INPUTS) | (extras or set())
        if (extras is None or digest(content)!=expected or
                set(manifest.get('files_sha256',{}))!=allowed):
            raise ValueError('source profile hash manifest is invalid or unsupported')
        for name,expected in manifest['files_sha256'].items():
            if _hash(profile/name)!=expected:
                raise ValueError('source profile input hash mismatch: '+name)
        extra_profile = ['bundle.json', *sorted(extras)]
    randomization = None
    if (profile/'randomization.json').exists():
        from .training_recipe import validate_randomization
        randomization = validate_randomization(read_json(profile/'randomization.json')[0])
        if 'randomization.json' not in extra_profile:
            extra_profile.append('randomization.json')
    values = {name: read_json(profile/name)[0] for name in INPUTS}
    executor = values['executor-parameters.json']
    if executor.get('schema_version') == 1:
        executor = executor['parameters']
    experiment = build_experiment(profile/'model.json', executor_parameters=executor,
        observation_config=values['observation-config.json'], prior_parameters=values['prior-parameters.json'],
        development_only=True, seconds=seconds)
    recipe = dict(schema='cartpole-colab-recipe-v1', steps=steps, seed=seed, seconds=seconds,
        n_steps=1024, batch_size=64, epochs=10, gamma=.999, gae_lambda=.98,
        learning_rate=.0003, reward_scale=.01, randomize=randomization is None, training_seeds=list(range(10000,10128)),
        evaluation_seeds=list(evaluation_seeds), hold_seconds=5.,
        checkpoint_every=checkpoint_every, angle_source=experiment['observation_config']['angle_source'],
        run='ppo-001', python='3.11.7', uv='0.8.11', colab_cli='0.7.4',
        hardware_accepted=False, task_qualified=False)
    if randomization is not None:
        recipe['randomization'] = randomization
    selected = [repo/name for name in ('pyproject.toml','README.md','uv.lock')]
    for folder in ('src','examples','scripts'):
        for path in sorted((repo/folder).rglob('*')):
            relative = path.relative_to(repo)
            if (path.is_file() and not path.is_symlink() and
                    not any(part.startswith('.') or part == '__pycache__' for part in relative.parts) and
                    (path.suffix in SOURCE_SUFFIXES or (folder == 'scripts' and not path.suffix))):
                selected.append(path)
    if any(not p.is_file() or p.is_symlink() for p in selected):
        raise ValueError('required source is missing or a symlink')
    output.mkdir(parents=True)
    with zipfile.ZipFile(output/'source.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for path in selected:
            z.write(path, path.relative_to(repo))
    _json(output/'recipe.json', recipe)
    with zipfile.ZipFile(output/'inputs.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for name in INPUTS:
            z.write(profile/name,name)
        z.write(output/'recipe.json', 'recipe.json')
        # Keep the selected source profile's provenance with the training inputs.
        for name in extra_profile:
            z.write(profile/name,name)
    template = (Path(__file__).with_name('colab_remote.py')).read_text()
    pins = {name: _hash(output/name) for name in ('source.zip','inputs.zip')}
    (output/'remote.py').write_text('EXPECTED_ARCHIVES = '+repr(pins)+'\n'+template)
    manifest = dict(schema='cartpole-colab-bundle-v1', hardware_accepted=False,
        task_qualified=False, recipe=recipe, source_files=len(selected),
        files_sha256={name:_hash(output/name) for name in ('source.zip','inputs.zip','recipe.json','remote.py')})
    _json(output/'bundle.json', manifest)
    return manifest


def verify_bundle(root):
    root = Path(root)
    manifest, _ = read_json(root/'bundle.json')
    if manifest.get('schema') != 'cartpole-colab-bundle-v1':
        raise ValueError('unsupported Colab bundle')
    pins = manifest.get('files_sha256',{})
    if set(pins) != {'source.zip','inputs.zip','recipe.json','remote.py'}:
        raise ValueError('incomplete bundle hash manifest')
    for name, expected in pins.items():
        if _hash(root/name) != expected:
            raise ValueError(f'bundle hash mismatch: {name}')
    return manifest
