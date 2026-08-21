"""Pure-logic tests for toppe/coppe.py's entry-number assignment and
entry-file formatting -- no live SSH/network needed. The remote-facing
functions (run_remote/query_existing_entries/claim_entry_numbers) are thin
wrappers around subprocess calls to a real scanner and aren't exercised
here; see toppe/README.md for how to smoke-test those against a real
scanner."""

from pathlib import Path

from toppe.coppe import (
    assign_entry_numbers,
    find_pge_files,
    split_reused_files,
    stage_entry_files,
)


def _files(*names: str) -> list[Path]:
    return [Path(n) for n in names]


def test_assign_prefers_unused_numbers_ascending():
    primary, padding = assign_entry_numbers(_files('a.pge', 'b.pge', 'c.pge'), entries={})
    assert primary == [0, 1, 2]
    assert 0 not in padding and 1 not in padding and 2 not in padding


def test_assign_skips_numbers_already_taken():
    entries = {0: 100.0, 1: 200.0}
    primary, _ = assign_entry_numbers(_files('a.pge'), entries)
    assert primary == [2]


def test_assign_falls_back_to_oldest_when_exhausted(capsys):
    # Every number 0-9999 taken; ages increase with N so N=0 is oldest.
    entries = {n: float(n) for n in range(10000)}
    primary, padding = assign_entry_numbers(_files('a.pge', 'b.pge'), entries)
    assert primary == [0, 1]
    # Reusing an entry is never silent, even without a confirmation prompt.
    out = capsys.readouterr().out
    assert 'WARN' in out and 'reusing entry 0' in out and 'reusing entry 1' in out


def test_assign_padding_excludes_primary_and_is_ordered_by_preference():
    primary, padding = assign_entry_numbers(_files('a.pge'), entries={0: 1.0})
    # primary[0] is the first unused number (1, since 0 is taken).
    assert primary == [1]
    assert 1 not in padding
    # Next-best unused numbers come first in padding, ascending.
    assert padding[:3] == [2, 3, 4]


def test_assign_padding_includes_reused_fallback_when_exhausted():
    # All 10000 numbers taken, ages increasing with N. Requesting 1 file
    # forces primary to reuse the oldest (0); padding's reused portion
    # should offer the next-oldest (1) as a backup candidate for the claim
    # step to fall through to if 0 is grabbed by a concurrent run first.
    entries = {n: float(n) for n in range(10000)}
    primary, padding = assign_entry_numbers(_files('a.pge'), entries)
    assert primary == [0]
    assert padding[0] == 1


def test_stage_entry_files_writes_two_line_v7_format(tmp_path):
    pge_file = Path('ArbEPI.pge')
    assignment = {pge_file: 42}
    entry_files = stage_entry_files(assignment, remote_pge_dir='/srv/x/y', staging_dir=tmp_path)

    assert len(entry_files) == 1
    entry_path = entry_files[0]
    assert entry_path.name == 'pge42.entry'
    assert entry_path.read_text() == '1\n/srv/x/y/ArbEPI.pge\n'


def test_find_pge_files_raises_when_empty(tmp_path):
    try:
        find_pge_files(str(tmp_path))
        raise AssertionError('expected RuntimeError')
    except RuntimeError as e:
        assert 'no .pge files found' in str(e)


def test_find_pge_files_returns_sorted_matches(tmp_path):
    (tmp_path / 'b.pge').write_text('')
    (tmp_path / 'a.pge').write_text('')
    (tmp_path / 'not_a_pge.txt').write_text('')

    files = find_pge_files(str(tmp_path))
    assert [f.name for f in files] == ['a.pge', 'b.pge']


def test_split_reused_files_matches_by_basename():
    files = _files('a.pge', 'b.pge', 'c.pge')
    existing = {'a.pge': 5, 'c.pge': 9}  # no entry for b.pge

    reused, to_claim = split_reused_files(files, existing)

    assert reused == {Path('a.pge'): 5, Path('c.pge'): 9}
    assert to_claim == [Path('b.pge')]


def test_split_reused_files_with_no_existing_entries():
    files = _files('a.pge', 'b.pge')
    reused, to_claim = split_reused_files(files, existing={})

    assert reused == {}
    assert to_claim == files
