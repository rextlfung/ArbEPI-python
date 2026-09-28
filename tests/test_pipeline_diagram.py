"""Keeps README's pipeline diagram (docs/pipeline/) in step with its generator
and with the repo's files."""

import importlib.util
import os
import re

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, 'docs', 'pipeline', 'make_pipeline.py')
SKIP_DIRS = {'output', '__pycache__', 'node_modules'}


def _load_generator():
    spec = importlib.util.spec_from_file_location('make_pipeline', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mp = _load_generator()


@pytest.mark.parametrize('theme', sorted(mp.THEMES))
def test_committed_svg_matches_generator(theme):
    with open(mp.svg_path(theme), encoding='utf-8') as f:
        committed = f.read()
    assert committed == mp.render(theme), (
        f'docs/pipeline/pipeline-{theme}.svg is stale: run '
        '`uv run python docs/pipeline/make_pipeline.py` and commit the result'
    )


def test_source_files_named_in_diagram_exist():
    # .jl only with a directory part: bare names like MRIFieldmaps.jl are Julia packages
    named = set(re.findall(r'[\w./-]+\.py\b|[\w.-]*/[\w./-]+\.jl\b', mp.render('light')))
    assert 'julia/b0map.jl' in named and 'lib/mask2epi.py' in named
    repo_files = set()
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.')]
        rel = os.path.relpath(root, REPO)
        repo_files.update(os.path.normpath(os.path.join(rel, f)) for f in files)
    missing = sorted(
        name for name in named
        if not any(p == name or p.endswith(os.sep + name) for p in repo_files)
    )
    assert not missing, f'diagram names files that no longer exist: {missing}'
