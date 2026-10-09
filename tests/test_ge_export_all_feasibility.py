"""ge/ge_export.py's check_all_ge_feasibility (review item 235) reports every
infeasible sequence, not just the first; ge/coppe.py keeps ssh/scp stderr
(no -q, item 171)."""

import inspect

import pytest

import ge.coppe as coppe
import ge.ge_export as ge_export


def test_check_all_reports_every_failure(monkeypatch):
    calls = []

    def fake_check(path, params):
        calls.append(path)
        if path in ('b.seq', 'd.seq'):
            raise RuntimeError(f'GE feasibility check failed for {path}:\nFAIL')
        return ('seq-' + path, 'report-' + path)

    monkeypatch.setattr(ge_export, 'check_ge_feasibility', fake_check)
    paths = {'A': 'a.seq', 'B': 'b.seq', 'C': 'c.seq', 'D': 'd.seq'}
    with pytest.raises(RuntimeError) as exc:
        ge_export.check_all_ge_feasibility(paths, params=None)
    assert calls == ['a.seq', 'b.seq', 'c.seq', 'd.seq']  # did not stop at B
    msg = str(exc.value)
    assert 'b.seq' in msg and 'd.seq' in msg
    assert '2 of 4' in msg


def test_check_all_returns_results_when_all_pass(monkeypatch):
    monkeypatch.setattr(ge_export, 'check_ge_feasibility', lambda p, params: (p, 'r'))
    out = ge_export.check_all_ge_feasibility({'A': 'a.seq', 'B': 'b.seq'}, params=None)
    assert out == {'A': ('a.seq', 'r'), 'B': ('b.seq', 'r')}


def test_coppe_ssh_calls_do_not_use_quiet_flag():
    for target in coppe._JUMP_HOSTS:
        assert '-q' not in coppe.build_ssh_prefix('user', target)
        assert '-q' not in coppe.build_ssh_prefix('user', target, relay='relay.example')
    src = inspect.getsource(coppe)
    assert "'-q'" not in src
