import subprocess

_CHECK_ONLY = 'check' in COMMAND_LINE_TARGETS
_CONTINUE = 'continue' in COMMAND_LINE_TARGETS
_SUBJECT = 'src'


def _exec(cmd: str) -> None:
    print(f'>>> {cmd}')
    exit_code = subprocess.call(cmd, shell=True)
    if exit_code != 0 and not _CONTINUE:
        print(f'Exiting with {exit_code}')
        Exit(exit_code)


# Combinations
if 'all' in COMMAND_LINE_TARGETS:
    COMMAND_LINE_TARGETS += ['format', 'quality']

if 'quality' in COMMAND_LINE_TARGETS:
    COMMAND_LINE_TARGETS += ['test', 'doctest', 'coverage', 'pycodestyle', 'pylint', 'mypy', 'complexity']

if 'format' in COMMAND_LINE_TARGETS:
    COMMAND_LINE_TARGETS += ['autoflake', 'isort']

# Quality
if 'test' in COMMAND_LINE_TARGETS:
    _exec('python -m unittest discover tests.unit')

if 'coverage' in COMMAND_LINE_TARGETS:
    _exec('coverage run --source=src -m unittest discover tests.unit')
    _exec('coverage report -m --fail-under=50')

if 'pycodestyle' in COMMAND_LINE_TARGETS:
    _exec(f'pycodestyle {_SUBJECT} -v --config=setup.cfg')

if 'pylint' in COMMAND_LINE_TARGETS:
    _exec(f'pylint --rcfile=setup.cfg {_SUBJECT}')

if 'mypy' in COMMAND_LINE_TARGETS:
    _exec(f'mypy {_SUBJECT}')

if 'complexity' in COMMAND_LINE_TARGETS:
    _exec(f'radon cc {_SUBJECT} -nc --total-average')
    _exec(f'xenon {_SUBJECT} --max-absolute B --max-modules A --max-average A')

# Format
if 'autoflake' in COMMAND_LINE_TARGETS:
    cmd = f'autoflake {_SUBJECT} --recursive --in-place --remove-unused-variables'
    if _CHECK_ONLY:
        cmd += ' --check'
    _exec(cmd)

if 'isort' in COMMAND_LINE_TARGETS:
    cmd = f'isort {_SUBJECT} --recursive --quiet'
    if _CHECK_ONLY:
        cmd += ' --check'
    _exec(cmd)

Exit(0)
