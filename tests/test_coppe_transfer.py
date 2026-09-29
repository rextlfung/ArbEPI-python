"""Runs ge/coppe.py's _TRANSFER_SCRIPT locally under bash, with a fake `scp`
first on PATH that plays phobos's Okta keyboard-interactive exchange
through the script's askpass (the way ssh would) instead of connecting
anywhere. Checks that the Okta prompts, including the number to tap,
reach stderr, and that the throwaway askpass is removed whether or not the
copy succeeds."""

import os
import stat
import subprocess
import tarfile
from pathlib import Path

from ge.coppe import _TRANSFER_SCRIPT

_FAKE_SCP = r"""#!/bin/bash
set -eu
echo "$SSH_ASKPASS" > "$FAKE_SCP_LOG/askpass_path"
[ "$SSH_ASKPASS_REQUIRE" = force ] || { echo "SSH_ASKPASS_REQUIRE not forced" >&2; exit 90; }
case " $* " in *BatchMode=yes*) echo "BatchMode set" >&2; exit 91;; esac
case " $* " in *keyboard-interactive*) ;; *) echo "no keyboard-interactive" >&2; exit 92;; esac
"$SSH_ASKPASS" "Okta passcode (leave blank to initiate a push): " > "$FAKE_SCP_LOG/answer1"
"$SSH_ASKPASS" "The correct answer is 46
Press enter to continue: " > "$FAKE_SCP_LOG/answer2"
[ "${FAKE_SCP_FAIL:-0}" = 1 ] && { echo "Permission denied (keyboard-interactive)." >&2; exit 1; }
src="${@: -2:1}"
cp "${src#*:}" "${@: -1}"
"""


def _run_transfer(tmp_path: Path, fail: bool) -> tuple[subprocess.CompletedProcess, Path, Path]:
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    scp = bindir / 'scp'
    scp.write_text(_FAKE_SCP)
    scp.chmod(scp.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / 'log'
    log.mkdir()

    staging = tmp_path / 'staging'
    staging.mkdir()
    (staging / 'ArbEPI.pge').write_bytes(b'pge')
    (staging / 'pge7.entry').write_text('1\n/run/ArbEPI.pge\n')
    tar_path = staging / 'coppe-scanfiles.tgz'
    with tarfile.open(tar_path, 'w:gz') as tar:
        for name in ('ArbEPI.pge', 'pge7.entry'):
            tar.add(staging / name, arcname=name)

    basedir = tmp_path / 'basedir'
    (basedir / 'pulseq' / 'v7' / '.lock-pge7').mkdir(parents=True)
    run_dir = basedir / 'rexfung' / 'run1'

    env = {**os.environ, 'PATH': f'{bindir}:{os.environ["PATH"]}', 'FAKE_SCP_LOG': str(log),
           'FAKE_SCP_FAIL': '1' if fail else '0'}
    env.pop('DISPLAY', None)
    result = subprocess.run(
        ['bash', '-s', '--', str(basedir), str(run_dir), 'rexfung', '1.2.3.4', str(tar_path), '7'],
        input=_TRANSFER_SCRIPT.encode(), capture_output=True, env=env,
    )
    return result, basedir, log


def test_transfer_echoes_okta_prompts_and_installs(tmp_path):
    result, basedir, log = _run_transfer(tmp_path, fail=False)
    assert result.returncode == 0, result.stderr.decode()

    stderr = result.stderr.decode()
    assert 'OKTA PROMPT: Okta passcode (leave blank to initiate a push)' in stderr
    assert 'The correct answer is 46' in stderr
    # Empty answers: a blank passcode starts a push, a blank line continues.
    assert (log / 'answer1').read_text() == '\n'
    assert (log / 'answer2').read_text() == '\n'

    v7 = basedir / 'pulseq' / 'v7'
    assert (v7 / 'pge7.entry').read_text() == '1\n/run/ArbEPI.pge\n'
    assert not (v7 / '.lock-pge7').exists()
    run_dir = basedir / 'rexfung' / 'run1'
    assert (run_dir / 'ArbEPI.pge').read_bytes() == b'pge'
    assert not (run_dir / 'coppe-scanfiles.tgz').exists()

    askpass = Path((log / 'askpass_path').read_text().strip())
    assert askpass.parent == run_dir
    assert not askpass.exists()


def test_transfer_removes_askpass_when_scp_fails(tmp_path):
    result, basedir, log = _run_transfer(tmp_path, fail=True)
    assert result.returncode != 0
    assert 'Permission denied (keyboard-interactive)' in result.stderr.decode()
    askpass = Path((log / 'askpass_path').read_text().strip())
    assert not askpass.exists()
    assert not (basedir / 'pulseq' / 'v7' / 'pge7.entry').exists()
