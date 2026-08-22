"""Copy a folder of .pge sequences (e.g. output/*.pge) to a GE scanner, for
**internal University of Michigan fMRI lab use only**.

Python port of ../toppe/+toppe/+utils/coppe.m (+ ../toppe/+toppe/
writeentryfile.m's v7 branch), which automates the manual workflow
documented in TOPPEpsdSourceCode's UserGuide branch, v7/README.md
("Running a sequence on the scanner"): copy a .pge file onto the scanner,
hand-write a pgeN.entry file (two lines: the literal "1", then the
absolute path to the .pge), and type N into the scanner console.

Upgrades over the MATLAB original (this repo only ever produces v7/pge2
.pge files, so the MATLAB script's v5/v6 legacy-format branches are
dropped entirely):
  - Transfers a whole folder of .pge files in one run, not one at a time.
  - Auto-allocates each file's entry number N (0-9999) by scanning the
    scanner for unused numbers first, falling back to overwriting the
    oldest existing entry only once 0-9999 is exhausted -- so no one's
    sequence is overwritten unless every slot is genuinely taken.
  - Prints the resulting filename -> N mapping so it can be typed into
    the scanner console.

See toppe/README.md for usage and the SSH key setup required for this to
work (two hops, in the same spirit as the MATLAB original: this host ->
lab jump server, and scanner -> this host, since the transfer is a *pull*
initiated from the scanner side).

If this host isn't itself allowed to connect to epyc/goliath directly
(e.g. running from a laptop instead of an already-whitelisted machine like
phobos.engin.umich.edu), a ProxyJump hop through phobos is used
automatically (see default_relay) -- pass --relay to point at a different
relay, or --relay '' to force none. See build_ssh_prefix and README.md's
"Running from an unwhitelisted machine" section.
"""

import argparse
import getpass
import re
import secrets
import socket
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

# UM-lab-specific infrastructure (mirrors coppe.m's own disclosure that
# server names would need updating for another site).
_BASEDIR = '/srv/nfs/psd/usr/psd'
_JUMP_HOSTS = {'inside': 'epyc', 'outside': 'goliath'}
_SCANNER_HOST = 'sdc@10.0.1.1'
_DEFAULT_RELAY_HOST = 'phobos.engin.umich.edu'

_ENTRY_RE = re.compile(r'^(?P<mtime>\d+(?:\.\d+)?)\s+pge(?P<n>\d+)\.entry$')

# Each script runs on the far end of the double-SSH hop via `bash -s --`
# (see run_remote): dynamic values travel as positional args ($1, $2, ...)
# rather than being interpolated into the script text, so they need no
# shell-escaping regardless of content.

_QUERY_SCRIPT = r"""
set -eu
basedir="$1"
mkdir -p "$basedir/pulseq/v7"
find "$basedir/pulseq/v7" -maxdepth 1 -name 'pge*.entry' -printf '%T@ %f\n' 2>/dev/null || true
"""

# Finds which existing pulseq/v7 entry numbers, if any, already point into
# run_dir -- i.e. what an earlier coppe.py run to this same --run-id
# already installed, so a rerun can reuse them instead of claiming fresh
# ones. Prints "N basename" per match.
_RUN_ENTRIES_SCRIPT = r"""
set -eu
basedir="$1"; run_dir="$2"
for f in "$basedir/pulseq/v7"/pge*.entry; do
    [ -e "$f" ] || continue
    path=$(sed -n '2p' "$f")
    case "$path" in
        "$run_dir"/*)
            base="$(basename "$f" .entry)"
            echo "${base#pge} $(basename "$path")"
            ;;
    esac
done
"""

# Claims entry numbers atomically via `mkdir` (atomic w.r.t. concurrent
# claims from another lab member's coppe.py run). Self-heals lock dirs
# older than 10 minutes first, so a crash between claiming and installing
# doesn't permanently shrink the usable number range. Tries candidates in
# order, stopping once `count` are claimed; prints each claimed number.
_CLAIM_SCRIPT = r"""
set -u
basedir="$1"; count="$2"; shift 2
lockdir="$basedir/pulseq/v7"
mkdir -p "$lockdir"
find "$lockdir" -maxdepth 1 -name '.lock-pge*' -mmin +10 -exec rmdir {} \; 2>/dev/null

claimed=0
for n in "$@"; do
    if [ "$claimed" -ge "$count" ]; then
        break
    fi
    if mkdir "$lockdir/.lock-pge${n}" 2>/dev/null; then
        echo "$n"
        claimed=$((claimed + 1))
    fi
done
"""

# Pulls the tarball back from this host (scp initiated from the scanner
# side -- see toppe/README.md for why), unpacks it into the per-user/
# per-run directory, and moves just the .entry files into the shared
# pulseq/v7/ namespace (the .pge files stay where they land).
_TRANSFER_SCRIPT = r"""
set -eu
basedir="$1"; run_dir="$2"; user="$3"; host_ip="$4"; tar_abspath="$5"; shift 5
mkdir -p "$run_dir"
cd "$run_dir"
tar_name="${tar_abspath##*/}"
scp -q "${user}@${host_ip}:${tar_abspath}" ./
tar -xzf "$tar_name"
rm -f "$tar_name"
for n in "$@"; do
    mv "pge${n}.entry" "$basedir/pulseq/v7/pge${n}.entry"
    rmdir "$basedir/pulseq/v7/.lock-pge${n}" 2>/dev/null || true
done
"""

_RELEASE_SCRIPT = r"""
basedir="$1"; shift
for n in "$@"; do
    rmdir "$basedir/pulseq/v7/.lock-pge${n}" 2>/dev/null || true
done
"""


def find_pge_files(pge_dir: str) -> list[Path]:
    files = sorted(Path(pge_dir).glob('*.pge'))
    if not files:
        raise RuntimeError(f'no .pge files found in {pge_dir!r} -- run `main.py --ge` first')
    if len(files) > 10000:
        raise RuntimeError(
            f'{len(files)} .pge files found, but only 10000 entry numbers (0-9999) exist'
        )
    return files


def discover_host_ip(
    override: str | None, user: str | None = None, relay: str | None = None,
) -> str:
    """The public IP the scanner-side scp pull (see _TRANSFER_SCRIPT) should
    connect back to. Without --relay this is this host's own public IP;
    with --relay it's relay's public IP instead (the tarball is staged
    there -- see stage_tarball_on_relay -- since relay, not this host, is
    what the scanner's hop-2 keys are provisioned for). Pass --host-ip to
    skip the lookup entirely (e.g. if it's unreachable, or the IP it
    reports isn't the one the scanner can actually route to)."""
    if override:
        return override
    if relay:
        return discover_relay_ip(user, relay)
    try:
        with urllib.request.urlopen('https://ifconfig.me/ip', timeout=5) as resp:
            return resp.read().decode().strip()
    except (urllib.error.URLError, TimeoutError) as e:
        raise RuntimeError(
            f"could not auto-discover this host's public IP ({e}) -- pass --host-ip explicitly"
        ) from e


_RELAY_IP_SCRIPT = r"""
set -eu
if command -v curl >/dev/null 2>&1; then
    curl -s https://ifconfig.me/ip
elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://ifconfig.me/ip
else
    python3 -c "import urllib.request as u
print(u.urlopen('https://ifconfig.me/ip', timeout=5).read().decode().strip())"
fi
"""


def _decode_output(result: subprocess.CompletedProcess) -> str:
    return (
        (result.stdout or b'').decode(errors='replace')
        + (result.stderr or b'').decode(errors='replace')
    )


def discover_relay_ip(user: str, relay: str) -> str:
    """relay's own public IP, queried by running the lookup directly on
    relay over a plain single-hop ssh (not through ssh_prefix's chain into
    the scanner) -- reproduces exactly what discover_host_ip would return
    if coppe.py were run locally on relay itself."""
    result = subprocess.run(
        ['ssh', '-q', f'{user}@{relay}', 'bash', '-s'],
        input=_RELAY_IP_SCRIPT.encode(), capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"could not auto-discover {relay}'s public IP (exit {result.returncode})\n"
            f'{_decode_output(result)}-- pass --host-ip explicitly'
        )
    ip = result.stdout.decode().strip()
    if not ip:
        raise RuntimeError(f"could not auto-discover {relay}'s public IP (empty response) "
                            '-- pass --host-ip explicitly')
    return ip


def stage_tarball_on_relay(user: str, relay: str, tar_path: Path) -> str:
    """scp's tar_path up to a fresh temp directory on relay -- a direct,
    single-hop this-host -> relay operation, unlike ssh_prefix's multi-hop
    chain into the scanner. Returns that remote directory; the
    scanner-side pull (_TRANSFER_SCRIPT) then scp's the tarball from
    there, reusing relay's already-provisioned hop-2 keys (see
    toppe/README.md) rather than needing new ones set up for this host."""
    result = subprocess.run(['ssh', '-q', f'{user}@{relay}', 'mktemp', '-d'], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f'could not create a staging directory on {relay} (exit {result.returncode})\n'
            f'{_decode_output(result)}'
        )
    remote_dir = result.stdout.decode().strip()
    if not remote_dir:
        raise RuntimeError(f'could not create a staging directory on {relay} (empty response)')

    scp_result = subprocess.run(
        ['scp', '-q', str(tar_path), f'{user}@{relay}:{remote_dir}/'], capture_output=True,
    )
    if scp_result.returncode != 0:
        cleanup_relay_staging(user, relay, remote_dir)
        raise RuntimeError(
            f'could not copy the tarball to {relay} (exit {scp_result.returncode})\n'
            f'{_decode_output(scp_result)}'
        )
    return remote_dir


def cleanup_relay_staging(user: str, relay: str, remote_dir: str) -> None:
    """Best-effort cleanup of the directory stage_tarball_on_relay created;
    never raises, mirroring release_locks's best-effort style."""
    try:
        subprocess.run(
            ['ssh', '-q', f'{user}@{relay}', 'rm', '-rf', remote_dir], capture_output=True,
        )
    except OSError:
        pass


def default_relay() -> str | None:
    """--relay's default: phobos, unless this host already *is* phobos (in
    which case no relay hop is needed at all -- see build_ssh_prefix).
    Compares only the hostname's first label (gethostname() rather than
    getfqdn(), to avoid a reverse-DNS lookup that can hang on a
    misconfigured resolver), so this matches whether the local hostname is
    the bare 'phobos' or the full 'phobos.engin.umich.edu'. Explicitly
    passing --relay '' overrides this to force no relay from anywhere."""
    local_host = socket.gethostname().split('.')[0].lower()
    relay_host = _DEFAULT_RELAY_HOST.split('.')[0].lower()
    return None if local_host == relay_host else _DEFAULT_RELAY_HOST


def build_ssh_prefix(user: str, target: str, relay: str | None = None) -> list[str]:
    """When relay is given, reaches epyc/goliath via ProxyJump (-J) through
    it instead of connecting directly -- for running coppe.py from a
    machine epyc/goliath won't accept connections from, but that can reach
    a whitelisted relay (e.g. phobos.engin.umich.edu). -J (not a third
    nested `ssh ... ssh ...` hop) matters here: it makes this host's own
    ssh client the one authenticating to epyc/goliath (so an interactive
    Duo prompt still has a tty to prompt on), with only the TCP connection
    itself routed through relay -- a nested remote-command hop would run
    that auth on relay with no tty, and Duo can't prompt there. The final
    epyc/goliath -> scanner hop stays a nested remote-command ssh as
    before: the scanner's sdc account trusts a key provisioned on
    epyc/goliath, not on this host or relay."""
    if relay:
        return ['ssh', '-q', '-J', f'{user}@{relay}', f'{user}@{_JUMP_HOSTS[target]}',
                'ssh', '-q', _SCANNER_HOST]
    return ['ssh', '-q', f'{user}@{_JUMP_HOSTS[target]}', 'ssh', '-q', _SCANNER_HOST]


def run_remote(
    ssh_prefix: list[str], script: str, args: list[str] = (), verbose: bool = False,
) -> str:
    """Runs `script` (plain bash, piped via stdin) on the far end of the
    double-SSH hop in `ssh_prefix`, with `args` as its positional
    parameters. SSH forwards stdin transparently through nested non-pty
    hops, so this needs only one layer of remote shell parsing (the final
    `bash -s` on the scanner), unlike a nested heredoc built as one string.

    When verbose, output isn't captured so it streams live to the terminal
    (progress, any interactive password/Duo prompt) -- note that SSH
    normally reads the actual secret from /dev/tty directly regardless of
    whether stdout/stderr are captured, so this mainly matters for
    *watching* prompts/progress (e.g. a Duo push), not for the prompt
    itself to function."""
    argv = [*ssh_prefix, 'bash', '-s', '--', *args]
    result = subprocess.run(argv, input=script.encode(), capture_output=not verbose)
    if result.returncode != 0:
        detail = '' if verbose else (
            (result.stdout or b'').decode(errors='replace')
            + (result.stderr or b'').decode(errors='replace')
        )
        raise RuntimeError(f'remote command failed (exit {result.returncode})\n{detail}')
    return (result.stdout or b'').decode() if not verbose else ''


def query_existing_entries(ssh_prefix: list[str]) -> dict[int, float]:
    stdout = run_remote(ssh_prefix, _QUERY_SCRIPT, [_BASEDIR])
    entries = {}
    for line in stdout.splitlines():
        m = _ENTRY_RE.match(line.strip())
        if m:
            entries[int(m['n'])] = float(m['mtime'])
    return entries


def query_run_entries(ssh_prefix: list[str], remote_run_dir: str) -> dict[str, int]:
    """Returns {pge basename: entry number} for every existing pulseq/v7
    entry file whose recorded path already points into remote_run_dir --
    i.e. what a previous coppe.py run to this same --run-id installed."""
    stdout = run_remote(ssh_prefix, _RUN_ENTRIES_SCRIPT, [_BASEDIR, remote_run_dir])
    existing = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        n_str, basename = line.split(' ', 1)
        existing[basename] = int(n_str)
    return existing


def split_reused_files(
    pge_files: list[Path], existing: dict[str, int],
) -> tuple[dict[Path, int], list[Path]]:
    """Splits pge_files into (reused, to_claim) by basename match against
    `existing` (see query_run_entries): reused maps a file to the entry
    number a previous run under this --run-id already assigned it, so its
    .pge content can just be overwritten in place with no new entry file
    needed; to_claim lists files with no such match, still needing a
    freshly claimed entry number."""
    reused: dict[Path, int] = {}
    to_claim: list[Path] = []
    for f in pge_files:
        if f.name in existing:
            reused[f] = existing[f.name]
        else:
            to_claim.append(f)
    return reused, to_claim


def confirm_replace(run_id: str, existing: dict[str, int]) -> bool:
    print(f"Run '{run_id}' already has sequence(s) installed on the scanner:")
    for basename, n in sorted(existing.items(), key=lambda item: item[1]):
        print(f'  {basename} (entry {n})')
    answer = input(
        'Replace the existing .pge file(s) in place, reusing their entry numbers? [y/N] '
    )
    return answer.strip().lower() in ('y', 'yes')


def assign_entry_numbers(
    pge_files: list[Path], entries: dict[int, float],
) -> tuple[list[int], list[int]]:
    """Returns (primary, padding). primary[i] is the preferred entry number
    for pge_files[i]: unused numbers 0-9999 first (ascending), falling
    back to the oldest-mtime existing entries only once every number is
    taken (prints a WARNING per reused number -- never silent, even though
    no confirmation prompt is required). padding is a list of backup
    candidates for claim_entry_numbers to fall through to if a concurrent
    coppe.py run claims one of the primary picks first between this query
    and the claim step -- any successfully claimed number is equally valid
    for any file at that point, so there's no need to preserve a strict
    pge_files[i] <-> primary[i] pairing once claiming is underway."""
    count = len(pge_files)
    used = set(entries)
    unused = [n for n in range(10000) if n not in used]

    primary = unused[:count]
    n_reused = count - len(primary)
    if n_reused > 0:
        oldest = sorted(entries, key=lambda n: entries[n])
        reuse_picks = oldest[:n_reused]
        for n in reuse_picks:
            age_h = (time.time() - entries[n]) / 3600
            print(
                f'WARN reusing entry {n} (last written {age_h:.1f}h ago) '
                '-- all 10000 entry numbers are currently taken'
            )
        primary = primary + reuse_picks

    pad_count = max(10, count)
    padding_unused = unused[count:count + pad_count]
    padding_reused = sorted(
        (n for n in entries if n not in primary), key=lambda n: entries[n],
    )[:pad_count]
    padding = padding_unused + padding_reused

    return primary, padding


def claim_entry_numbers(
    ssh_prefix: list[str], primary: list[int], padding: list[int], count: int,
) -> list[int]:
    candidates = primary + [n for n in padding if n not in primary]
    stdout = run_remote(
        ssh_prefix, _CLAIM_SCRIPT, [_BASEDIR, str(count), *map(str, candidates)],
    )
    claimed = [int(line) for line in stdout.splitlines() if line.strip()]
    if len(claimed) < count:
        release_locks(ssh_prefix, claimed)
        raise RuntimeError(
            f'could only claim {len(claimed)}/{count} entry numbers on the scanner '
            '(likely a concurrent coppe.py run) -- try again'
        )
    return claimed


def release_locks(ssh_prefix: list[str], numbers: list[int]) -> None:
    """Best-effort cleanup for claimed-but-not-installed lock directories;
    never raises (the stale-lock self-heal in _CLAIM_SCRIPT is the
    backstop if this doesn't run at all, e.g. the process is killed)."""
    if not numbers:
        return
    try:
        run_remote(ssh_prefix, _RELEASE_SCRIPT, [_BASEDIR, *map(str, numbers)])
    except RuntimeError:
        pass


def stage_entry_files(
    assignment: dict[Path, int], remote_pge_dir: str, staging_dir: Path,
) -> list[Path]:
    entry_files = []
    for pge_file, n in assignment.items():
        entry_path = staging_dir / f'pge{n}.entry'
        entry_path.write_text(f'1\n{remote_pge_dir}/{pge_file.name}\n')
        entry_files.append(entry_path)
    return entry_files


def build_tarball(pge_files: list[Path], entry_files: list[Path], tar_path: Path) -> None:
    with tarfile.open(tar_path, 'w:gz') as tar:
        for f in (*pge_files, *entry_files):
            tar.add(f, arcname=f.name)


def transfer_and_install(
    ssh_prefix: list[str], tar_abspath: str, user: str, host_ip: str,
    remote_run_dir: str, claimed: list[int], verbose: bool,
) -> None:
    args = [_BASEDIR, remote_run_dir, user, host_ip, tar_abspath, *map(str, claimed)]
    run_remote(ssh_prefix, _TRANSFER_SCRIPT, args, verbose=verbose)


def print_report(assignment: dict[Path, int], remote_run_dir: str) -> None:
    print()
    print(f'SUCCESS: {len(assignment)} sequence(s) copied to {remote_run_dir}/')
    print()
    width = max(len(p.name) for p in assignment) + 2
    print(f'{"file":<{width}}entry #')
    for pge_file, n in assignment.items():
        print(f'{pge_file.name:<{width}}{n}')
    print()
    print('On the scanner console: prescribe a product GRE (e.g. a localizer), set PSD')
    print("Name to 'pge2', then enter the entry number above for the sequence to run.")


def main(args: argparse.Namespace) -> None:
    pge_files = find_pge_files(args.pge_dir)
    ssh_prefix = build_ssh_prefix(args.user, args.target, args.relay)
    remote_run_dir = f'{_BASEDIR}/{args.user}/{args.run_id}'

    # Printed before the first SSH call into ssh_prefix (query_run_entries,
    # right below) rather than right before transfer_and_install -- every
    # call through ssh_prefix can trigger its own Duo push, not just the
    # final transfer, so the warning needs to be visible before any of them
    # fire, not just the last one.
    print('Contacting the scanner (keep an eye out for a Duo push)...')

    # A rerun to the same --run-id can reuse entries a previous run already
    # claimed instead of allocating fresh ones -- but only after confirming
    # that's what's wanted, since it overwrites those .pge files in place.
    existing = query_run_entries(ssh_prefix, remote_run_dir)
    reused, to_claim = split_reused_files(pge_files, existing)
    if existing:
        if not confirm_replace(args.run_id, existing):
            raise RuntimeError(
                f"run-id '{args.run_id}' already has sequence(s) on the scanner -- "
                'choose a different --run-id, or rerun and confirm to replace them'
            )

    claimed: list[int] = []
    if to_claim:
        entries = query_existing_entries(ssh_prefix)
        primary, padding = assign_entry_numbers(to_claim, entries)
        claimed = claim_entry_numbers(ssh_prefix, primary, padding, count=len(to_claim))
    new_assignment = dict(zip(to_claim, claimed))

    # Preserve pge_files' order for the printed report, regardless of which
    # files were reused vs. newly claimed.
    assignment = {**reused, **new_assignment}
    assignment = {f: assignment[f] for f in pge_files}

    try:
        with tempfile.TemporaryDirectory() as tmp:
            staging_dir = Path(tmp)
            # Only newly claimed files need a new entry file staged --
            # reused ones already have a correct entry file in place on the
            # scanner (same run_dir, same path), untouched by this run.
            entry_files = stage_entry_files(new_assignment, remote_run_dir, staging_dir)
            tar_path = staging_dir / 'coppe-scanfiles.tgz'
            build_tarball(pge_files, entry_files, tar_path)

            relay_dir = None
            try:
                if args.relay:
                    relay_dir = stage_tarball_on_relay(args.user, args.relay, tar_path)
                    tar_abspath = f'{relay_dir}/{tar_path.name}'
                    host_ip = discover_host_ip(args.host_ip, args.user, args.relay)
                else:
                    tar_abspath = str(tar_path.resolve())
                    host_ip = discover_host_ip(args.host_ip)

                print(f'Copying {len(pge_files)} sequence(s) to the scanner...')
                transfer_and_install(
                    ssh_prefix, tar_abspath, args.user, host_ip, remote_run_dir, claimed,
                    args.verbose,
                )
            finally:
                if relay_dir:
                    cleanup_relay_staging(args.user, args.relay, relay_dir)
    except Exception:
        release_locks(ssh_prefix, claimed)
        raise

    print_report(assignment, remote_run_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument(
        '--target', choices=['inside', 'outside'], default='inside',
        help="which scanner to copy to (default: 'inside')",
    )
    parser.add_argument(
        '--run-id', default=f'{time.strftime("%Y%m%dT%H%M%S")}-{secrets.token_hex(3)}',
        help='override the default timestamp-based per-run subfolder name',
    )
    parser.add_argument(
        '--pge-dir', default='output',
        help="folder of .pge files to transfer (default: 'output')",
    )
    parser.add_argument(
        '--host-ip', default=None,
        help="override auto-discovery of this host's (or --relay's) public IP",
    )
    parser.add_argument(
        '--relay', default=default_relay(),
        help='ProxyJump through this host to reach epyc/goliath, for running coppe.py from a '
             "machine they won't accept direct connections from. Defaults to "
             f"'{_DEFAULT_RELAY_HOST}' unless this host's hostname is already 'phobos' "
             "(no relay needed); pass --relay '' to force no relay -- see toppe/README.md",
    )
    parser.add_argument(
        '--user', default=getpass.getuser(),
        help='SSH username on the lab server/scanner (default: current user)',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='stream the final transfer step live instead of capturing it, '
             'for interactive password/Duo prompts',
    )
    parser.add_argument(
        '-h', '--help', action='help', default=argparse.SUPPRESS,
        help='show this help message and exit',
    )
    main(parser.parse_args())
