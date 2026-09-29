"""Draws the README's pipeline diagram: docs/pipeline/pipeline-{light,dark}.svg.

The diagram is generated, not hand-edited: change the layout in `draw()` below,
then regenerate both SVGs with

    uv run python docs/pipeline/make_pipeline.py

and commit them together with this file. tests/test_pipeline_diagram.py fails
if the committed SVGs differ from this script's output, or if a .py/.jl file
the diagram names no longer exists in the repo.

README.md embeds the two files in a <picture> element, so GitHub shows the one
matching the reader's theme. Each SVG carries its own <style> and no web fonts
(an SVG shown through <img> cannot load any), so text is sized for common
system fonts: `Drawing.box`/`Drawing.tag` raise if a line is likely to overflow
its box (or its box is too short), using per-character widths that are
generous for Helvetica/Arial. A box's ('para', ...) lines are wrapped to its
width; each section is placed relative to the one above, so resizing a box
moves everything below it.

Layout: one column of steps (config -> masks -> partitioning) that fans out
into four lanes, one per sequence (deGRE, noise, EPIcal, ArbEPI). Each lane
keeps its x position from sequence generation through the scanner into the
preprocessing step that consumes that sequence's raw data. scan_info.mat is the
dashed line on the right: it goes around the scanner.
"""

import os
from html import escape

HERE = os.path.dirname(os.path.abspath(__file__))
W, LEGEND_H = 1100, 40

THEMES = {
    'light': {
        'bg': '#f5f6f8', 'fg': '#19212c', 'muted': '#586273', 'line': '#3d4757',
        'box': '#ffffff', 'chip': '#e6eaf0',
        'seq-band': '#e7edf7', 'seq': '#2f5d9e',
        'scan-band': '#f2ece2', 'scan': '#8d5f22',
        'pre-band': '#e4f1ea', 'pre': '#2a7250',
        'rec-band': '#eee8f6', 'rec': '#664a9c',
        'hand': '#bb4a24', 'hand-bg': '#fbe9e1',
    },
    'dark': {
        'bg': '#10141a', 'fg': '#e3e7ed', 'muted': '#9ba5b3', 'line': '#b4bdc9',
        'box': '#19202a', 'chip': '#232b37',
        'seq-band': '#15213a', 'seq': '#8cb2ee',
        'scan-band': '#282116', 'scan': '#d7a864',
        'pre-band': '#13281e', 'pre': '#78c8a0',
        'rec-band': '#221b33', 'rec': '#b7a0e8',
        'hand': '#f0906b', 'hand-bg': '#3a2119',
    },
}

SANS = ('"IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, '
        '"Liberation Sans", sans-serif')
MONO = ('"IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", '
        'monospace')

STYLE = f"""
.bg {{ fill: var(--bg); }}
.band {{ stroke: none; }}
.band.seq {{ fill: var(--seq-band); }} .band.scan {{ fill: var(--scan-band); }}
.band.pre {{ fill: var(--pre-band); }} .band.rec {{ fill: var(--rec-band); }}
.rail {{ font: 600 11px {SANS}; letter-spacing: .12em; }}
.rail.seq {{ fill: var(--seq); }} .rail.scan {{ fill: var(--scan); }}
.rail.pre {{ fill: var(--pre); }} .rail.rec {{ fill: var(--rec); }}
.box {{ fill: var(--box); stroke-width: 1.5; }}
.box.seq {{ stroke: var(--seq); }} .box.scan {{ stroke: var(--scan); }}
.box.pre {{ stroke: var(--pre); }} .box.rec {{ stroke: var(--rec); }}
.box.plain {{ stroke: var(--line); }}
.box.key {{ stroke-width: 2.5; }}
.box.dashed {{ stroke-dasharray: 5 4; }}
.group {{ fill: none; stroke: var(--pre); stroke-width: 1; stroke-dasharray: 3 3; }}
.chip {{ fill: var(--chip); stroke: none; }}
.chip.hand {{ fill: var(--hand-bg); stroke: var(--hand); stroke-width: 1.5; }}
.flow {{ fill: none; stroke: var(--line); stroke-width: 1.5; }}
.flow.dashed {{ stroke-dasharray: 5 4; }}
.hand {{ fill: none; stroke: var(--hand); stroke-width: 2; stroke-dasharray: 7 5; }}
.ah {{ fill: var(--line); }}
.ah-hand {{ fill: var(--hand); }}
.t-title {{ font: 600 14px {SANS}; fill: var(--fg); }}
.t-mono {{ font: 400 12px {MONO}; fill: var(--muted); }}
.t-mono-s {{ font: 400 11px {MONO}; fill: var(--muted); }}
.t-mono-b {{ font: 600 12.5px {MONO}; fill: var(--fg); }}
.t-text {{ font: 400 12.5px {SANS}; fill: var(--fg); }}
.t-chip {{ font: 500 12px {MONO}; fill: var(--fg); }}
.t-edge {{ font: 600 11.5px {MONO}; fill: var(--fg); }}
.t-section {{ font: 600 12.5px {SANS}; fill: var(--muted); letter-spacing: .02em; }}
.t-legend {{ font: 400 12.5px {SANS}; fill: var(--muted); }}
.t-note {{ font: 400 11.5px {SANS}; fill: var(--hand); }}
.t-note-b {{ font: 600 11.5px {SANS}; fill: var(--hand); }}
.t-note-mono {{ font: 400 11px {MONO}; fill: var(--hand); }}
.t-small {{ font: 400 11.5px {SANS}; fill: var(--muted); }}
"""

# Upper-bound pixel width per character for each text class, used only to
# catch lines that would overflow their box after an edit.
CHAR_PX = {'title': 7.8, 'mono': 7.2, 'mono-s': 6.6, 'mono-b': 7.5, 'text': 6.6, 'indent': 6.6}
INDENT = 14  # px, for 'indent' lines in a box

ARIA = (
    'ArbEPI workflow: scanner specs and scan parameters feed sampling-mask generation (bypassed '
    'when a custom mask is given) and trajectory computation, which produce four Pulseq sequences '
    '(deGRE, noise, EPIcal, ArbEPI); these are exported to GE .pge files and run on the scanner. '
    'Preprocessing whitens with the noise scan, estimates the readout delay and odd/even mismatch '
    'from EPIcal, regrids, corrects and scatters the ArbEPI data, and derives coil compression, '
    'sensitivity maps, B0 and R2* from the deGRE before compressing the k-space, extracting the '
    'calibration region and writing one preprocessed file. Reconstruction is root-sum-of-squares '
    'or iterative SENSE with optional B0 and R2* models. scan_info.mat bypasses the scanner and '
    'carries the sampling schedule to preprocessing.'
)


class Drawing:
    def __init__(self):
        self.back = []  # environment bands, drawn behind everything else
        self.out = []

    def add(self, s):
        self.out.append(s)

    def box(self, x, y, w, h, cls, lines, dashed=False):
        """A processing step. `lines` are (kind, text) pairs, usually one or more 'title'
        lines, a 'mono' module path, then details: 'text' (one line), 'indent' (one
        line, indented) or 'para' (wrapped to the box width)."""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" '
                 f'class="box {cls}{" dashed" if dashed else ""}"/>')
        ty, prev = y, None
        for kind, text in self._wrap(lines, w - 32):
            ty += 22 if prev is None else 18 if prev in ('title', 'mono', 'mono-s') else 16
            dx = INDENT if kind == 'indent' else 0
            self._check_fit(kind, text, w - 32 - dx)
            cls_t = 'text' if kind == 'indent' else kind
            self.add(f'<text x="{x + 16 + dx}" y="{ty}" class="t-{cls_t}">{escape(text)}</text>')
            prev = kind
        if ty > y + h - 12:
            raise ValueError(f'{lines[0][1]!r}: text runs past the bottom of its box '
                             f'(needs a height of at least {ty - y + 12})')

    def chip(self, cx, cy, text, cls=''):
        """A file or array passed between steps, centered on an arrow."""
        w = round(len(text) * 7.2 + 22)
        self.add(f'<rect x="{cx - w / 2:.1f}" y="{cy - 11}" width="{w}" height="22" rx="11" '
                 f'class="chip {cls}"/>')
        self.add(f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" class="t-chip">'
                 f'{escape(text)}</text>')
        return w

    def tag(self, x, y, w, h, lines, cls=''):
        """A multi-line file tag (same fill as a chip)."""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" class="chip {cls}"/>')
        ty = y
        for i, (kind, text) in enumerate(lines):
            ty += 22 if i == 0 else 16
            self._check_fit(kind, text, w - 28)
            self.add(f'<text x="{x + 14}" y="{ty}" class="t-{kind}">{escape(text)}</text>')
        if ty > y + h - 8:
            raise ValueError(f'{lines[0][1]!r}: text runs past the bottom of its tag')

    def arrow(self, d, cls='flow'):
        marker = 'ah-hand' if cls == 'hand' else 'ah'
        self.add(f'<path d="{d}" class="{cls}" marker-end="url(#{marker})"/>')

    def line(self, d, cls='flow'):
        self.add(f'<path d="{d}" class="{cls}"/>')

    def label(self, x, y, text, anchor='middle', cls='t-edge'):
        self.add(f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="{cls}">'
                 f'{escape(text)}</text>')

    def band(self, y0, y1, cls, name):
        """A full-width background band for one environment, named on the left rail."""
        self.back.append(f'<rect x="0" y="{y0}" width="{W}" height="{y1 - y0}" rx="10" '
                         f'class="band {cls}"/>')
        cy = (y0 + y1) / 2
        self.back.append(f'<text x="30" y="{cy}" transform="rotate(-90 30 {cy})" '
                         f'text-anchor="middle" class="rail {cls}">{escape(name)}</text>')

    @staticmethod
    def _wrap(lines, avail):
        per_line = int(avail // CHAR_PX['text'])
        for kind, text in lines:
            if kind != 'para':
                yield kind, text
                continue
            cur = ''
            for word in text.split():
                if cur and len(cur) + 1 + len(word) > per_line:
                    yield 'text', cur
                    cur = word
                else:
                    cur = f'{cur} {word}' if cur else word
            if cur:
                yield 'text', cur

    @staticmethod
    def _check_fit(kind, text, avail):
        if len(text) * CHAR_PX[kind] > avail:
            raise ValueError(f'{text!r} is likely wider than its box ({avail}px); '
                             'shorten it or widen the box')


# x centers of the four sequence lanes, kept from generation to preprocessing
LANE = {'gre': 150, 'noise': 360, 'cal': 570, 'epi': 780}


def draw_legend(d):
    y = 22
    d.add(f'<rect x="20" y="{y - 11}" width="26" height="16" rx="4" class="box plain"/>')
    d.label(54, y + 2, 'processing step', 'start', 't-legend')
    d.add(f'<rect x="170" y="{y - 11}" width="26" height="16" rx="8" class="chip"/>')
    d.label(204, y + 2, 'file or array passed on', 'start', 't-legend')
    d.add(f'<rect x="368" y="{y - 11}" width="26" height="16" rx="4" class="box plain dashed"/>')
    d.label(402, y + 2, 'optional input or step', 'start', 't-legend')
    d.line(f'M566 {y - 3} H600', 'hand')
    d.label(608, y + 2, 'scan_info.mat, which goes around the scanner', 'start', 't-legend')


def draw(d):
    """Draws the diagram body and returns its height. Each section is placed
    relative to the one above it, so a taller box pushes everything below down."""
    # ---- one column: scanner specs / parameters -> masks -> trajectory ----
    sx, sw = 380, 440
    sc = sx + sw // 2
    top, row_h, step = 28, 92, 148  # step: top of one box to the top of the next
    mid = top + row_h // 2
    d.box(60, top, 270, row_h, 'seq', [
        ('title', 'Scanner specs'),
        ('mono', 'scanners.py'),
        ('text', 'ScannerSpec: GE_MR750 | GE_UHP'),
        ('text', 'gradient, slew, B1 and PNS limits'),
    ])
    d.arrow(f'M330 {mid} H378')
    d.box(sx, top, sw, row_h, 'seq', [
        ('title', 'Scan parameters'),
        ('mono', 'params.py · load_params() → Params'),
        ('text', 'resolution, matrix, R, ETL, TE, volume TR, readout slews'),
        ('text', 'derives Nshots, then TR and flip angle'),
    ])
    d.box(860, top, 225, row_h, 'seq', [
        ('title', 'Custom mask (optional)'),
        ('mono-s', 'sampling/external_mask.py'),
        ('text', '(ky, kz) or (ky, kz, t) .mat'),
        ('text', 'sets Nshots and effective R'),
    ], dashed=True)
    d.arrow(f'M860 {mid} H822', 'flow dashed')

    y_mask, y_traj = top + step, top + 2 * step
    d.arrow(f'M{sc} {top + row_h} V{y_mask - 2}')
    d.chip(sc, top + row_h + 28, 'Params (+ sys, spec)')
    d.box(sx, y_mask, sw, row_h, 'seq', [
        ('title', 'Sampling mask generation'),
        ('mono', 'sampling/ · resolve_omegas()'),
        ('text', 'one (ky, kz) mask per frame: pd · caipi · ticaipi · rand'),
        ('text', 'skipped when a custom mask is given'),
    ])
    y_omegas = y_mask + row_h + 28
    d.arrow(f'M{sc} {y_mask + row_h} V{y_traj - 2}')
    w_omegas = d.chip(sc, y_omegas, 'omegas · Ny × Nz × Nframes, bool')
    # a custom mask goes straight to omegas
    d.arrow(f'M972 {top + row_h} V{y_omegas} H{sc + w_omegas / 2 + 2:.0f}', 'flow dashed')
    for i, text in enumerate(['custom mask:', 'bypasses mask', 'generation']):
        d.label(962, y_mask + 8 + 16 * i, text, 'end', 't-small')
    d.box(sx, y_traj, sw, row_h, 'seq', [
        ('title', 'Trajectory computation'),
        ('mono', 'lib/mask2epi.py · radial | laminar'),
        ('text', "split each frame's samples into Nshots trains of ETL echoes"),
        ('text', 'order: min-sum TSP → bottleneck 2-opt → uncross (radial)'),
    ])

    # ---- four sequences, one lane each ----
    seq_y, seq_h, lw = y_traj + row_h + 90, 168, 190
    d.arrow(f'M{sc} {y_traj + row_h} V{seq_y - 26} H{LANE["epi"]} V{seq_y - 2}')
    d.chip(sc, y_traj + row_h + 30, 'schedules · Nframes × Nshots × ETL × (ky, kz, t)')
    d.label(55, seq_y - 14, 'Sequence generation · sequences/ (pypulseq)', 'start', 't-section')
    seqs = [
        ('gre', 'deGRE', 'deGRE.py',
         'dual-echo 3D GRE scan for estimating coil sensitivity maps, B0 field maps, and '
         'coil compression matrices'),
        ('noise', 'noise', 'noise.py', 'ADC-only noise scan. Same readout length as ArbEPI'),
        ('cal', 'EPIcal', 'EPIcal.py',
         'Short snippet of the ArbEPI sequence without phase-encoding blips. Used for odd/even '
         'echo mismatch calibration and receiver gain tuning'),
        ('epi', 'ArbEPI', 'ArbEPI.py',
         'Main 3D-EPI sequence with arbitrary phase-encoding described by schedules'),
    ]
    for key, name, mod, desc in seqs:
        d.box(LANE[key] - lw // 2, seq_y, lw, seq_h, 'seq' + (' key' if key == 'epi' else ''),
              [('title', name), ('mono', mod), ('para', desc)])
    info_y = seq_y + 12
    d.tag(905, info_y, 180, 84, [
        ('mono-b', 'scan_info.mat'),
        ('text', 'schedules, kxo/kxe,'),
        ('text', 'echo times, TE_degre,'),
        ('text', 'scan scalars (v7.3)'),
    ], 'hand')
    d.arrow(f'M{LANE["epi"] + lw // 2} {info_y + 42} H903')
    notes = [
        ('t-note-b', 'read by preprocess from'),
        ('t-note-mono', '<datdir>/seqs/<seq>/'),
        ('t-note', 'schedules → (ky, kz)'),
        ('t-note', 'kxo/kxe → gridding'),
        ('t-note', 'echo times → B0 model'),
    ]
    for i, (cls, text) in enumerate(notes):
        d.label(905, info_y + 116 + i * 16 + (6 if i >= 2 else 0), text, 'start', cls)

    # ---- GE export ----
    ge_y, ge_h = seq_y + seq_h + 56, 88
    for key, name, _, _ in seqs:
        d.arrow(f'M{LANE[key]} {seq_y + seq_h} V{ge_y - 2}')
        d.chip(LANE[key], seq_y + seq_h + 28, f'{name}.seq')
    d.box(55, ge_y, 820, ge_h, 'seq', [
        ('title', 'GE export'),
        ('mono', 'ge/ · main.py --ge'),
        ('text', 'checks all four first: gradient, slew, B1 limits · PNS (fails above 100%, '
                 'warns above 80%) · gradient acoustics (warns)'),
        ('text', 'then seq2ceq → writeceq writes one .pge per sequence '
                 '(pure Python port of PulCeq)'),
    ])
    d.band(0, ge_y + ge_h + 20, 'seq', 'SEQUENCE DESIGN · main.py')

    # ---- scanner ----
    sc_y, sc_h = ge_y + ge_h + 86, 60
    pge_y = ge_y + ge_h + 42
    for key, name, _, _ in seqs:
        d.arrow(f'M{LANE[key]} {ge_y + ge_h} V{sc_y - 2}')
        d.chip(LANE[key], pge_y, f'{name}.pge')
    d.label(905, pge_y - 6, 'copy to the scanner:', 'start', 't-note')
    d.label(905, pge_y + 10, 'ge/coppe.py (optional)', 'start', 't-note-mono')
    d.box(55, sc_y, 820, sc_h, 'scan', [
        ('title', 'GE scanner · pge2/tv7 interpreter'),
        ('text', 'runs each .pge as a pge2 entry; raw data saved as GE ScanArchives '
                 '(HDF5, read only by the Orchestra SDK)'),
    ])
    d.band(sc_y - 22, sc_y + sc_h + 20, 'scan', 'SCANNER')

    # ---- preprocess, Stage A: one step per lane, then the EPI chain down lane 4 ----
    pre_y0 = sc_y + sc_h + 70
    sa_y0 = pre_y0 + 18
    sb_y, sbw = sa_y0 + 36, 170
    corr_y, scat_y = sb_y + 156, sb_y + 280
    sa_y1 = scat_y + 104 + 18
    for key in ('noise', 'cal', 'epi'):
        d.arrow(f'M{LANE[key]} {sc_y + sc_h} V{sb_y - 2}')
    archives = {'gre': 'gre.h5 · shared', 'noise': '<seq>_noise.h5 (opt.)',
                'cal': '<seq>_cal.h5', 'epi': '<seq>_epi.h5'}
    for key, text in archives.items():
        d.chip(LANE[key], sc_y + sc_h + 45, text)

    d.add(f'<rect x="255" y="{sa_y0}" width="644" height="{sa_y1 - sa_y0}" rx="8" '
          'class="group pre"/>')
    d.label(262, sa_y1 + 22, 'Stage A (dotted) · all coils, cached, resumable', 'start',
            't-section')
    ex, ew = LANE['epi'] - sbw // 2, sbw + 24  # lane-4 boxes: left edge, width (wider titles)
    d.box(LANE['noise'] - sbw // 2, sb_y, sbw, 104, 'pre', [
        ('title', 'Whitening'), ('mono', 'coils.py'),
        ('text', 'noise covariance → W'), ('text', 'noise_var measured'),
        ('text', 'after gridding'),
    ])
    d.box(LANE['cal'] - sbw // 2, sb_y, sbw, 120, 'pre', [
        ('title', 'Odd/even mismatch'), ('title', 'estimation'), ('mono', 'oephase.py'),
        ('text', 'readout-delay sweep,'), ('text', 'then ghost phase a'),
        ('text', 'on whitened cal data'),
    ])
    d.box(ex, sb_y, ew, 120, 'pre', [
        ('title', 'Ramp-sample regrid'), ('mono', 'epi_gridding.py'),
        ('para', 'whitened readouts onto Cartesian kx (1D NUFFT); odd/even kx shifted by the '
                 'calibrated delay'),
    ])
    d.box(ex, corr_y, ew, 88, 'pre', [
        ('title', 'Odd/even correction'), ('mono', 'oephase.py'),
        ('text', 'subtracts a[0] + a[1]·x'), ('text', 'from every even echo'),
    ])
    d.box(ex, scat_y, ew, 104, 'pre', [
        ('title', 'Scatter to (ky, kz)'), ('mono', 'preprocess.py'),
        ('text', 'each (shot, echo) into'), ('text', 'its slot of the zero-'),
        ('text', 'filled grid, per frame'),
    ])
    my = sb_y + 52
    d.arrow(f'M{LANE["noise"] + sbw // 2} {my} H{LANE["cal"] - sbw // 2 - 2}')
    d.label((LANE['noise'] + LANE['cal']) / 2, my - 6, 'W')
    d.arrow(f'M{LANE["cal"] + sbw // 2} {my} H{ex - 2}')
    d.label((LANE['cal'] + LANE['epi']) / 2, my - 6, 'delay')
    d.arrow(f'M{LANE["cal"]} {sb_y + 120} V{corr_y + 44} H{ex - 2}')
    d.label((LANE['cal'] + ex) / 2, corr_y + 38, 'a')
    d.arrow(f'M{LANE["noise"] - sbw // 2} {my} H{LANE["gre"] + 3}')
    d.label((LANE['noise'] - sbw // 2 + LANE['gre']) / 2 + 8, my - 6, 'W')
    d.arrow(f'M{LANE["epi"]} {sb_y + 120} V{corr_y - 2}')
    d.arrow(f'M{LANE["epi"]} {corr_y + 88} V{scat_y - 2}')

    # scan_info.mat goes around the scanner: kxo/kxe to the regrid, schedules to the scatter
    d.arrow(f'M1070 {info_y + 84} V{scat_y + 52} H{ex + ew + 2}', 'hand')
    d.arrow(f'M1070 {sb_y + 60} H{ex + ew + 2}', 'hand')

    # ---- preprocess: deGRE maps | coil compression, R2* | k-space; then Stage B ----
    col_l, col_m, col_r = (55, 250), (345, 240), (625, 250)  # (x, width)
    ya, rh = sa_y1 + 66, 104
    yb, yc = ya + rh + 40, ya + 2 * (rh + 40)
    d.arrow(f'M{LANE["epi"]} {scat_y + 104} V{ya - 2}')
    d.chip(LANE['epi'], sa_y1 + 32, '<seq>_gridded.h5')
    # the whitened deGRE feeds ESPIRiT (down its lane) and GCC (branch)
    gcc_cx = col_m[0] + col_m[1] // 2
    d.arrow(f'M{LANE["gre"]} {sc_y + sc_h} V{ya - 2}')
    d.arrow(f'M{LANE["gre"]} {ya - 18} H{gcc_cx} V{ya - 2}')
    d.box(col_l[0], ya, col_l[1], rh, 'pre', [
        ('title', 'Sensitivity maps (ESPIRiT)'), ('mono', 'smaps.py'),
        ('para', 'ESPIRiT on echo 1, resized to the EPI grid, masked and smoothed; then '
                 'compressed with T'),
    ])
    d.box(col_m[0], ya, col_m[1], rh, 'pre', [
        ('title', 'Coil compression fit (GCC)'), ('mono', 'coils.py'),
        ('para', 'geometric decomposition on the whitened deGRE: per-x matrices keeping 99.9% '
                 'of the energy'),
    ])
    d.box(col_r[0], ya, col_r[1], rh, 'pre', [
        ('title', 'Compress k-space'), ('mono', 'Stage B · write_output()'),
        ('para', 'the cached k-space frame by frame with T, and the gridded noise → noise_var'),
    ])
    mid_a = ya + rh // 2
    d.arrow(f'M{col_m[0]} {mid_a} H{col_l[0] + col_l[1] + 2}')
    d.label((col_m[0] + col_l[0] + col_l[1]) / 2, mid_a - 6, 'T')
    d.arrow(f'M{col_m[0] + col_m[1]} {mid_a} H{col_r[0] - 2}')
    d.label((col_m[0] + col_m[1] + col_r[0]) / 2, mid_a - 6, 'T')

    # B0 sits right of a lane that carries the smaps straight down to the output
    smaps_x, b0_x = 72, 88
    b0_w = col_l[0] + col_l[1] - b0_x
    b0_cx = b0_x + b0_w // 2
    d.box(b0_x, yb, b0_w, rh, 'pre', [
        ('title', 'B0 map (optional)'), ('mono-s', 'b0map.py + julia/b0map.jl'),
        ('para', 'MRIFieldmaps.jl on both echoes from a ROMEO.jl start; skip with --no-b0'),
    ], dashed=True)
    d.box(col_m[0], yb, col_m[1], rh, 'pre', [
        ('title', 'R2* map (optional)'), ('mono', 'r2star.py'),
        ('para', 'log-linear fit in the B0 mask (placeholder); skip with --no-r2star'),
    ], dashed=True)
    d.box(col_r[0], yb, col_r[1], rh, 'pre', [
        ('title', 'Calibration region'), ('mono', 'find_calib_region()'),
        ('para', '(ky, kz) block sampled in every frame, grown from the center; cut out as '
                 'ksp_calib'),
    ])
    d.arrow(f'M{b0_cx} {ya + rh} V{yb - 2}')
    d.label(b0_cx + 8, yb - 14, 'smaps, emap', 'start')
    mid_b = yb + rh // 2
    d.arrow(f'M{col_l[0] + col_l[1]} {mid_b} H{col_m[0] - 2}')
    d.label((col_l[0] + col_l[1] + col_m[0]) / 2, mid_b - 6, 'mask')
    r_cx = col_r[0] + col_r[1] // 2
    d.arrow(f'M{r_cx} {ya + rh} V{yb - 2}')

    d.arrow(f'M{smaps_x} {ya + rh} V{yc - 2}')
    d.arrow(f'M{b0_cx} {yb + rh} V{yc - 2}', 'flow dashed')
    d.arrow(f'M{gcc_cx} {yb + rh} V{yc - 2}', 'flow dashed')
    d.arrow(f'M{r_cx} {yb + rh} V{yc - 2}')
    wh = 92
    d.box(55, yc, 820, wh, 'pre', [
        ('title', 'Stage B · write <seq>_preprocessed.h5'),
        ('mono', 'preprocess.py · write_output()'),
        ('text', 'ksp_epi_zf, ksp_calib, smaps, b0map_hz, r2star, omegas, echo_times, W, '
                 'cc_matrix; attrs noise_var, t_ref_s'),
        ('text', 'the three maps also go to .nii.gz + .json for viewing'),
    ])
    pre_y1 = yc + wh + 22
    d.band(pre_y0, pre_y1, 'pre', 'PREPROCESS · .venv-preprocessing')

    # ---- recon ----
    rec_y0 = pre_y1 + 50
    bus_y, r_y, r_h = rec_y0 + 12, rec_y0 + 28, 152
    rss_cx, sense_cx = 200, 620
    d.line(f'M470 {yc + wh} V{bus_y}')
    d.line(f'M{rss_cx} {bus_y} H{sense_cx}')
    d.arrow(f'M{rss_cx} {bus_y} V{r_y - 2}')
    d.arrow(f'M{sense_cx} {bus_y} V{r_y - 2}')
    d.chip(470, pre_y1 + 25,
           '<seq>_preprocessed.h5 · ksp_epi_zf, smaps, b0map_hz, r2star, echo_times')
    d.box(55, r_y, 290, r_h, 'rec', [
        ('title', 'Root-sum-of-squares'), ('title', 'coil-combined IFT'),
        ('title', 'reconstruction'), ('mono', 'rss.py'),
        ('text', 'zero-filled inverse FFT per coil,'), ('text', 'then RSS over coils'),
        ('text', 'first look, aliasing kept'),
    ])
    d.box(365, r_y, 510, r_h, 'rec', [
        ('title', 'Iterative SENSE reconstruction'),
        ('mono', 'sense.py · operators.py · regularizers.py · solvers.py'),
        ('text', 'min over x of ½‖Ax − y‖² + g(x), y = the sampled k-space points only'),
        ('text', 'A: SENSE (coil maps · 3D FFT · sampling), always'),
        ('indent', '+ B0 off-resonance, optional (--B0): time-segmented, L = 32'),
        ('indent', '+ R2* decay, optional (--R2star, with B0): image at nominal TE'),
        ('text', 'g → solver: none → CG · low rank → POGM · wavelet + TV → PDHG'),
        ('text', 'k-space scaled to unit noise variance; λ defaults to R'),
    ])
    d.arrow(f'M875 {r_y + 76} H903')
    d.tag(905, r_y + 18, 180, 116, [
        ('mono-b', '<datdir>/recon/'),
        ('mono-s', 'sense_<reg>/'),
        ('mono-s', 'sense_<reg>_b0/'),
        ('mono-s', 'sense_<reg>_b0r2star/'),
        ('mono-s', 'rss/'),
        ('text', '.h5 + .nii.gz + .json'),
    ])
    rec_y1 = r_y + r_h + 20
    d.band(rec_y0, rec_y1, 'rec', 'RECON · .venv-recon')
    return rec_y1 + 12


def render(theme):
    tokens = ''.join(f'--{k}: {v}; ' for k, v in THEMES[theme].items())
    d = Drawing()
    draw_legend(d)
    legend = d.out
    d.out = []
    h = draw(d) + LEGEND_H
    head = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" width="{W}" '
        f'height="{h}" role="img" aria-label="{escape(ARIA)}">',
        f'<title>ArbEPI pipeline</title>\n<style>svg {{ {tokens}}}{STYLE}</style>',
        '<defs>',
    ]
    for mid in ('ah', 'ah-hand'):
        head.append(f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                    f'markerHeight="7" orient="auto-start-reverse">'
                    f'<path d="M0,0 L10,5 L0,10 z" class="{mid}"/></marker>')
    head += ['</defs>', f'<rect width="{W}" height="{h}" rx="12" class="bg"/>']
    body = [f'<g transform="translate(0 {LEGEND_H})">', *d.back, *d.out, '</g>\n</svg>\n']
    return '\n'.join(head + legend + body)


def svg_path(theme):
    return os.path.join(HERE, f'pipeline-{theme}.svg')


def main():
    for theme in THEMES:
        with open(svg_path(theme), 'w', encoding='utf-8') as f:
            f.write(render(theme))
        print(f'wrote {os.path.relpath(svg_path(theme))}')


if __name__ == '__main__':
    main()
