# ge/coppe.py — copy .pge sequences to the scanner

For **internal University of Michigan fMRI lab use only** — the server
names below (`epyc`, `goliath`, `sdc@10.0.1.1`) are specific to this lab's
infrastructure.

## What this does

`coppe.py` is a Python port (with upgrades) of the MATLAB lab utility
[`../toppe/+toppe/+utils/coppe.m`](../../toppe/+toppe/+utils/coppe.m). It
takes a folder of compiled `.pge` sequences (e.g. `output/*.pge`, written
by [`main.py --ge`](../README.md#ge-export-pge)) and:

1. Scans the scanner's `pulseq/v7/` directory for entry numbers (`N`,
   0–9999) that aren't already in use by anyone else's sequence.
2. Copies every `.pge` file to a directory unique to you and this run
   (`/srv/nfs/psd/usr/psd/<your username>/<run id>/`), so repeated runs
   never collide or overwrite each other.
3. Writes a `pgeN.entry` file for each `.pge`, pointing at its final
   location, and installs it into the scanner's shared `pulseq/v7/`
   namespace.
4. Prints the resulting filename → entry number mapping.

If every one of the 10000 possible entry numbers is already taken (should
essentially never happen), it falls back to overwriting the **oldest**
existing entry — always with a printed `WARNING`, never silently.

Unlike the MATLAB original, this only supports the pge2/tv7 interpreter
(the only format this repo ever produces), transfers a whole folder in one
run instead of one file at a time, and manages entry numbers automatically
instead of taking a manually-chosen `cv` number.

## SSH key setup (required once per user)

The file transfer is a **pull**: the scanner-side script copies the
tarball of `.pge`/`.entry` files *from* your machine, rather than your
machine pushing to the scanner directly. This mirrors the MATLAB
original's own assumption and needs two separate SSH key hops set up
before `coppe.py` will work.

### Hop 1: this host → lab jump server (`epyc` / `goliath`)

`coppe.py` logs into the scanner through one of the lab's jump servers —
`epyc` for `--target inside` (the default), `goliath` for
`--target outside`. Set up key-based login to both:

```
ssh-keygen -t ed25519          # skip if you already have a key
ssh-copy-id <your-username>@epyc
ssh-copy-id <your-username>@goliath
```

Verify with `ssh <your-username>@epyc echo ok` (should print `ok` with no
password prompt).

### Hop 2: scanner → this host (the "pull" leg)

The scanner (via the jump server) needs to `scp` the tarball back from
your machine, which means the scanner's `sdc` account needs a key
authorized on **this host**. In practice this is usually shared,
lab-wide infrastructure rather than something each user sets up
individually — check with a labmate or your PI before assuming you need to
provision this yourself. To verify it's already working, run `coppe.py`
once (see [Usage](#usage) below); the specific failure to watch for is the
transfer step's `scp` erroring with `Permission denied
(publickey)`, which means this hop isn't set up for your account/host yet.

**If you do end up setting this up yourself**, a few things that aren't
obvious the first time through:

- `sdc` is a single account shared by everyone in the lab — generate a key
  named for *you* specifically (e.g. `id_ecdsa_<username>`), not a
  default-named one, so it doesn't collide with another lab member's setup.
- The scanner host commonly runs in **FIPS mode**, which rejects Ed25519
  keys (`Key type ed25519 not allowed in FIPS mode`) — use RSA or ECDSA
  with a NIST curve instead:
  ```
  ssh-keygen -t ecdsa -b 384 -f ~/.ssh/id_ecdsa_<username> -N ""
  ```
- Because it's a shared account, don't rely on SSH's automatic default-key
  lookup — pin your key explicitly for your host in `sdc`'s
  `~/.ssh/config`, listing **both** an alias and the raw IP as patterns on
  the same `Host` line (`coppe.py`'s default `--host-ip` auto-discovery
  connects using the bare IP, which won't match a block that only lists an
  alias):
  ```
  Host <your-alias> <your-host-ip>
      HostName <your-host-fqdn>
      User <your-username>
      IdentityFile ~/.ssh/id_ecdsa_<username>
  ```
  (Alternatively, always pass `--host-ip <your-alias>` explicitly instead
  of relying on auto-discovery, and you only need the alias pattern.)
- Copy the public half to this host as usual, then double-check
  permissions here if it still prompts for a password afterward — `sshd`
  silently falls back to password auth if `~`, `~/.ssh`, or
  `~/.ssh/authorized_keys` are group/world-writable:
  ```
  ssh-copy-id -i ~/.ssh/id_ecdsa_<username>.pub <you>@<your-host>
  chmod 700 ~ ~/.ssh
  chmod 600 ~/.ssh/authorized_keys
  ```

### (Not your responsibility) jump server → scanner

The `epyc`/`goliath` → `sdc@10.0.1.1` hop is pre-provisioned lab
infrastructure — nothing to configure on your end.

## Running from an unwhitelisted machine (e.g. your laptop)

`epyc`/`goliath` only accept SSH connections from machines the lab network
has whitelisted (e.g. `phobos.engin.umich.edu`) — running `coppe.py`
directly from an arbitrary machine (a personal laptop, off-campus) fails at
the very first SSH hop. **`coppe.py` relays through phobos automatically**
whenever it detects it isn't already running on phobos itself (`--relay`'s
default, from `default_relay()` — checked via local hostname, no network
call), so as long as you can SSH from your machine to phobos, no extra flag
is needed:

```
uv run python ge/coppe.py
```

is equivalent to explicitly passing `--relay phobos.engin.umich.edu`. Pass
`--relay <other-host>` to relay through somewhere else instead, or
`--relay ''` to force no relay even from a non-phobos machine.

The relay hop adds phobos as a
[`ProxyJump`](https://man.openbsd.org/ssh_config#ProxyJump) for the
`epyc`/`goliath` connection (`ssh -J`, not a third nested `ssh` — see
`build_ssh_prefix`'s docstring for why that distinction matters for Duo),
and stages the transfer tarball on phobos rather than locally, so the
scanner's pull-back leg (hop 2 above) reuses phobos's already-provisioned
keys instead of needing new ones set up for your laptop.

**Setup required**: just SSH key auth from your machine to the relay itself
(`ssh-copy-id <your-username>@phobos.engin.umich.edu`, same as any normal
SSH login) — no new keys needed for the relay → `epyc`/`goliath` or
scanner → relay hops, since those already work today from the relay
directly. One extra one-time step: `ssh -J` verifies `epyc`/`goliath`'s host
key using *your local machine's* `known_hosts`, which (unlike the relay
itself) has probably never seen it before, since you've only ever reached
it as a remote command run on the relay. Run this once and accept the
prompt:

```
ssh -J <your-username>@phobos.engin.umich.edu <your-username>@epyc echo ok
ssh -J <your-username>@phobos.engin.umich.edu <your-username>@goliath echo ok
```

If either one prompts for a password instead of logging straight in, that's
hop 1's key (see [Hop 1](#hop-1-this-host--lab-jump-server-epyc--goliath)
above) missing from `epyc`/`goliath` for *your laptop* specifically —
because `-J` authenticates from your laptop's own SSH client (that's what
lets an interactive Duo prompt work), the key needs to be on `epyc`/
`goliath` themselves, not on the relay, even though the relay is where the
TCP connection is routed through:

```
ssh-copy-id -o ProxyJump=<your-username>@phobos.engin.umich.edu <your-username>@epyc
```

## Usage

```
uv run python ge/coppe.py
```

Transfers every `output/*.pge` file to the "inside" scanner using your
current username. Other examples:

```
# Transfer to the "outside" scanner instead
uv run python ge/coppe.py --target outside

# Transfer .pge files from a different folder
uv run python ge/coppe.py --pge-dir /path/to/some/other/folder

# Show live output during the transfer (e.g. to watch for a Duo push)
uv run python ge/coppe.py -v

# Skip the automatic public-IP lookup (see Troubleshooting)
uv run python ge/coppe.py --host-ip 141.213.x.x

# Use a specific SSH username, or a specific run subfolder name
uv run python ge/coppe.py --user labmate --run-id my-test-run

# Relay through a different host instead of the default phobos
# (see "Running from an unwhitelisted machine" above)
uv run python ge/coppe.py --relay some-other-whitelisted-host.engin.umich.edu
```

Run `uv run python ge/coppe.py --help` for the full flag list.

## Entry-number allocation

Each `.pge` file gets its own entry number, chosen as follows:

1. **Unused numbers first**, ascending from 0 — this is the common case,
   and never touches anyone else's existing entry.
2. **Oldest-entry reuse**, only once every number 0–9999 is already taken
   — the least-recently-written entries are overwritten first. This always
   prints a `WARNING: reusing entry N (last written <age> ago)` line so
   it's visible even though there's no confirmation prompt.

Claiming a number is done atomically on the scanner (via `mkdir`, which
either succeeds or fails outright, never partially) so two lab members
running `coppe.py` at the same moment can't both land on the same number —
whoever loses the race automatically falls back to the next candidate.
That said, avoid deliberately running two transfers at once if you can
help it; it's handled safely, but adds an extra network round trip if a
collision does occur.

## Rerunning with the same `--run-id`

If you rerun `coppe.py` with a `--run-id` that already has files installed
on the scanner (e.g. re-copying `output/` after regenerating a sequence),
it prompts before touching anything:

```
Run 'my-test-run' already has sequence(s) installed on the scanner:
  ArbEPI.pge (entry 42)
  EPIcal.pge (entry 17)
Replace the existing .pge file(s) in place, reusing their entry numbers? [y/N]
```

Answering `y` overwrites those `.pge` files in place and **reuses their
existing entry numbers** — no new entry is claimed, and the `.entry` file
on the scanner is left untouched (it already points at the right path).
Any file in the current batch that *doesn't* have a matching entry from
before (e.g. a new sequence added to the folder since the last run) still
gets a freshly claimed entry number as usual. Answering anything else
aborts with no changes made — pick a different `--run-id` if you didn't
mean to overwrite that run.

## Running the sequence on the scanner console

This is the same manual console workflow as any other pge2 sequence (see
`TOPPEpsdSourceCode`'s `v7/README.md`, "Running a sequence on the
scanner"):

1. Prescribe by copying and pasting a **product sequence**, e.g. a 3-plane
   localizer — do not copy an existing pge2 series. Select
   2D / Gradient echo / GRE.
2. Set the PSD Name field to `pge2`.
3. In the User CVs menu, enter the entry number `coppe.py` printed for the
   sequence you want to run.

## Troubleshooting

- **`no .pge files found in 'output'`**: run `uv run python main.py --ge`
  first (see the [main README](../README.md#ge-export-pge)).
- **`Permission denied (publickey)` on the first ssh command**: hop 1
  isn't set up — see [SSH key setup](#ssh-key-setup-required-once-per-user)
  above.
- **The transfer step hangs, or `scp` fails with `Permission denied
  (publickey)`**: hop 2 (scanner → this host) isn't set up, or the
  auto-discovered public IP (`--host-ip`) isn't actually reachable from
  the scanner's network — try passing `--host-ip` explicitly, and confirm
  with your lab's network setup whether this host is reachable at all from
  the scanner subnet.
- **A Duo push doesn't seem to be arriving**: pass `-v`/`--verbose` to
  stream the transfer step's output live instead of capturing it, so you
  can see what's actually happening rather than a silent wait.
- **You're asked to 2FA multiple times per run**: `coppe.py` makes up to 4
  separate SSH connections per invocation (entry-number lookup, claiming,
  and the transfer itself), each authenticated independently. Enable SSH
  connection multiplexing on the machine you run it from so only the first
  one needs a fresh prompt:
  ```
  # ~/.ssh/config, on the machine you run coppe.py from:
  Host epyc goliath phobos.engin.umich.edu
      ControlMaster auto
      ControlPath ~/.ssh/cm-%r@%h:%p
      ControlPersist 10m
  ```
  (include the relay host too if using `--relay` — it's a separate
  connection from the `ssh -J` hop into `epyc`/`goliath`)
- **`could only claim N/M entry numbers ... try again`**: another
  `coppe.py` run grabbed one of the same candidates at the same moment
  (see [Entry-number allocation](#entry-number-allocation)) — just rerun.
- **Every one of the 10000 entries is genuinely taken**: `coppe.py` will
  still succeed by overwriting the oldest entries, printing a `WARNING`
  for each — if that's not what you want, coordinate with your lab to
  clean up old/unused entries on the scanner first.

## Limitations

- The SSH topology (`epyc`/`goliath` → `sdc@10.0.1.1`) is fixed to this
  lab's infrastructure, not configurable via a flag — see
  `../toppe/+toppe/+utils/coppe.m`'s own docstring if adapting this for
  another site.
- Assumes GNU `find`/`coreutils` on the scanner (same assumption the
  MATLAB original and this repo's other GE-facing code already make).
- pge2/tv7 only — this repo never produces the older toppe v5/v6 file
  format, so that branch of the MATLAB original isn't ported.
