"""Draws the README's pipeline diagram: docs/pipeline/pipeline-{light,dark}.svg.

The diagram is generated, not hand-edited: change the layout in `draw()` below,
then regenerate both SVGs with

    uv run python docs/pipeline/make_pipeline.py

and commit them together with this file. tests/test_pipeline_diagram.py fails
if the committed SVGs differ from this script's output, or if a .py/.jl file
the diagram names no longer exists in the repo.

README.md embeds the two files in a <picture> element, so GitHub shows the one
matching the reader's theme. Each SVG carries its own <style> and no web fonts
(an SVG shown through <img> cannot load any). Text uses Helvetica, or its
metric twins Arial and Liberation Sans, and is measured with the Helvetica
metrics bundled with matplotlib, so ('para', ...) lines fill each box's width
before wrapping and boxes size their height to their text. LaTeX lines
('math', 'mathc') are drawn as paths with matplotlib's mathtext. Each section is
placed relative to the one above, so longer text moves everything below it.

Layout: one column of steps (config -> masks -> partitioning) that fans out
into four lanes, one per sequence (deGRE, noise, EPIcal, ArbEPI). Each lane
keeps its x position from sequence generation through the scanner into the
preprocessing step that consumes that sequence's raw data. scan_info.mat is the
dashed line on the right: it goes around the scanner. Iterative SENSE is a
dotted group of its parts: the objective, then one row per option of the
encoding operator A, the regularizer g and the solver that goes with g.
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

SANS = 'Helvetica, Arial, "Liberation Sans", sans-serif'  # metric-compatible fonts
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
.group.rec {{ stroke: var(--rec); }}
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
.t-colhead {{ font: 600 11px {SANS}; fill: var(--muted); letter-spacing: .06em; }}
.t-math {{ fill: var(--fg); stroke: none; }}
.rule {{ stroke: var(--line); stroke-opacity: .25; stroke-width: 1; }}
"""

# Text is measured with the Helvetica metrics that ship with matplotlib; Arial
# and Liberation Sans (the rest of SANS) share them, so a line measured to fit
# does fit on macOS, Windows and Linux. Monospace fonts are 0.6 em per character.
TEXT_STYLE = {  # kind -> (bold, font size px, monospace)
    'title': (True, 14, False), 'text': (False, 12.5, False), 'indent': (False, 12.5, False),
    'mono': (False, 12, True), 'mono-s': (False, 11, True), 'mono-b': (True, 12.5, True),
}
FIT = 0.98  # use at most this fraction of a box's inner width
INDENT = 14  # px, for 'indent' lines in a box
MATH_SIZE = {'math': 15, 'mathc': 18}  # px, LaTeX lines in a box
_GLYPH_NAMES = {'·': 'periodcentered', '×': 'multiply', '–': 'endash', '—': 'emdash',
                '’': 'quoteright', '‘': 'quoteleft', '“': 'quotedblleft', '”': 'quotedblright'}
_AFM = {}


def text_width(kind, text):
    """Rendered width in px of `text` in a box line of this kind."""
    bold, size, mono = TEXT_STYLE[kind]
    if mono:
        return len(text) * 0.6 * size
    if bold not in _AFM:
        import matplotlib
        from matplotlib import _afm
        name = 'Helvetica-Bold.afm' if bold else 'Helvetica.afm'
        path = os.path.join(matplotlib.get_data_path(), 'fonts', 'pdfcorefonts', name)
        with open(path, 'rb') as f:
            _AFM[bold] = _afm.AFM(f)
    afm = _AFM[bold]
    units = 0
    for ch in text:
        try:
            units += (afm.get_width_char(ord(ch)) if ord(ch) < 128
                      else afm.get_width_from_char_name(_GLYPH_NAMES[ch]))
        except KeyError:
            units += 900  # arrows, Greek, math symbols: wider than an average letter
    return units * size / 1000


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
        self.front = []  # chips, drawn over the arrows they sit on

    def add(self, s):
        self.out.append(s)

    def box(self, x, y, w, h, cls, lines, dashed=False):
        """A processing step; returns its height. `lines` are (kind, text) pairs, usually
        a 'title' (wrapped if long), a 'mono' module path, then details: 'para'
        (wrapped to fill the box width), 'text' (one line), 'indent', or LaTeX as
        'math' (left-aligned) or 'mathc' (centered, larger). h=None sizes the box
        to its text; a given h must be tall enough."""
        placed, need = self._layout(y, w, lines)
        h = need if h is None else h
        if need > h:
            raise ValueError(f'{lines[0][1]!r}: text runs past the bottom of its box '
                             f'(needs a height of at least {need})')
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" '
                 f'class="box {cls}{" dashed" if dashed else ""}"/>')
        for kind, text, ty in placed:
            if kind in ('math', 'mathc'):
                size = MATH_SIZE[kind]
                if kind == 'mathc':
                    self.math(x + w / 2, ty, text, size=size, anchor='middle')
                else:
                    self.math(x + 16, ty, text, size=size)
                continue
            dx = INDENT if kind == 'indent' else 0
            cls_t = 'text' if kind == 'indent' else kind
            self.add(f'<text x="{x + 16 + dx}" y="{ty}" class="t-{cls_t}">{escape(text)}</text>')
        return h

    def box_height(self, w, lines):
        """The height box() would give these lines at width w."""
        return self._layout(0, w, lines)[1]

    def _layout(self, y, w, lines):
        """([(kind, text, baseline)], height needed) for a box's lines."""
        placed, ty, prev, prev_descent = [], y, None, 0
        for kind, text in self._wrap(lines, w - 32):
            if kind in ('math', 'mathc'):
                _, width, ascent, descent = _math_path(text, MATH_SIZE[kind])
                if width > w - 32:
                    raise ValueError(f'math {text!r} is wider than its box ({w - 32}px)')
                ty += max(22, ascent + 8) if prev is None else max(18, prev_descent + ascent + 6)
            else:
                dx = INDENT if kind == 'indent' else 0
                self._check_fit(kind, text, w - 32 - dx)
                descent = 4
                if prev is None:
                    ty += 22
                elif prev in ('math', 'mathc'):
                    ty += max(16, prev_descent + 14)
                else:
                    ty += 18 if prev in ('title', 'mono', 'mono-s') else 16
            placed.append((kind, text, ty))
            prev, prev_descent = kind, descent
        return placed, ty - y + max(14, prev_descent + 10)

    def chip(self, cx, cy, text, cls=''):
        """A file or array passed between steps, centered on an arrow."""
        w = round(len(text) * 7.2 + 22)
        self.front.append(f'<rect x="{cx - w / 2:.1f}" y="{cy - 11}" width="{w}" height="22" '
                          f'rx="11" class="chip {cls}"/>')
        self.front.append(f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" class="t-chip">'
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

    def math(self, x, baseline, tex, size=15, anchor='start'):
        """LaTeX (matplotlib mathtext, Computer Modern) drawn as SVG paths, so it
        needs no fonts or MathJax when GitHub shows the SVG as an image. Returns
        the rendered width in px."""
        path, width, _, _ = _math_path(tex, size)
        x0 = x - width / 2 if anchor == 'middle' else x
        self.add(f'<path transform="translate({x0:.1f} {baseline})" class="t-math" d="{path}"/>')
        return width

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
        """Wraps 'para' lines, and 'title' lines too long for the box, at word
        breaks, filling each line as far as it measurably fits."""
        for kind, text in lines:
            if kind not in ('para', 'title'):
                yield kind, text
                continue
            out = 'text' if kind == 'para' else 'title'
            cur = ''
            for word in text.split():
                trial = f'{cur} {word}' if cur else word
                if cur and text_width(out, trial) > avail * FIT:
                    yield out, cur
                    cur = word
                else:
                    cur = trial
            if cur:
                yield out, cur

    @staticmethod
    def _check_fit(kind, text, avail):
        if text_width(kind, text) > avail:
            raise ValueError(f'{text!r} is wider than its box ({avail}px); '
                             'shorten it or widen the box')


def _math_path(tex, size):
    """(SVG path data with y pointing down and the baseline at 0, width, ascent,
    descent) for `tex`, in px."""
    import matplotlib
    from matplotlib.font_manager import FontProperties
    from matplotlib.path import Path
    from matplotlib.textpath import TextPath

    with matplotlib.rc_context({'mathtext.fontset': 'cm'}):
        tp = TextPath((0, 0), f'${tex}$', size=size, prop=FontProperties(size=size))
    cmd = {Path.MOVETO: 'M', Path.LINETO: 'L', Path.CURVE3: 'Q', Path.CURVE4: 'C'}
    parts = []
    for verts, code in tp.iter_segments(curves=True, simplify=False):
        if code == Path.CLOSEPOLY:
            parts.append('Z')
            continue
        pts = ' '.join(f'{vx:.2f} {-vy:.2f}' for vx, vy in verts.reshape(-1, 2))
        parts.append(f'{cmd[code]}{pts}')
    ext = tp.get_extents()
    return ''.join(parts), ext.x1, round(max(ext.y1, 0)), round(max(-ext.y0, 0))


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
    """Draws the diagram body and returns its height. Boxes size to their text
    (h=None), boxes in a row share the tallest one's height, and each section is
    placed relative to the one above it, so longer text pushes everything down."""

    def row(w_lines):
        """Common height for boxes drawn side by side: [(width, lines), ...]."""
        return max(d.box_height(w, lines) for w, lines in w_lines)

    # ---- one column: scanner specs / parameters -> masks -> trajectory ----
    sx, sw = 365, 470
    sc = sx + sw // 2
    top, gap = 28, 56  # gap: between boxes in the column (room for a chip)
    specs = [
        ('title', 'Scanner specs'), ('mono', 'scanners.py'),
        ('para', 'ScannerSpec: GE_MR750 | GE_UHP; gradient, slew, B1 and PNS limits'),
    ]
    params = [
        ('title', 'Scan parameters'), ('mono', 'params.py · load_params() → Params'),
        ('para', 'resolution, matrix, R, ETL, TE, volume TR, readout slews; derives Nshots, '
                 'then TR, then flip angle (Ernst angle)'),
    ]
    custom = [
        ('title', 'Custom mask (optional)'), ('mono-s', 'sampling/external_mask.py'),
        ('para', '(ky, kz) or (ky, kz, t) .mat; sets Nshots and effective R'),
    ]
    top_h = row([(270, specs), (sw, params), (225, custom)])
    mid = top + top_h // 2
    d.box(60, top, 270, top_h, 'seq', specs)
    d.arrow(f'M330 {mid} H{sx - 2}')
    d.box(sx, top, sw, top_h, 'seq', params)
    d.box(860, top, 225, top_h, 'seq', custom, dashed=True)
    d.arrow(f'M860 {mid} H{sx + sw + 2}', 'flow dashed')

    y_mask = top + top_h + gap
    d.arrow(f'M{sc} {top + top_h} V{y_mask - 2}')
    d.chip(sc, top + top_h + gap // 2, 'Params (+ sys, spec)')
    mask_h = d.box(sx, y_mask, sw, None, 'seq', [
        ('title', 'Sampling mask generation'), ('mono', 'sampling/ · resolve_omegas()'),
        ('para', 'independently generate (ky, kz) masks per frame; skipped when a custom '
                 'mask is given'),
    ])
    y_traj = y_mask + mask_h + gap
    y_omegas = y_mask + mask_h + gap // 2
    d.arrow(f'M{sc} {y_mask + mask_h} V{y_traj - 2}')
    w_omegas = d.chip(sc, y_omegas, 'omegas · Ny × Nz × Nframes, bool')
    # a custom mask goes straight to omegas
    d.arrow(f'M972 {top + top_h} V{y_omegas} H{sc + w_omegas / 2 + 2:.0f}', 'flow dashed')
    for i, text in enumerate(['custom mask:', 'bypasses mask', 'generation']):
        d.label(962, y_mask + 8 + 16 * i, text, 'end', 't-small')
    traj_h = d.box(sx, y_traj, sw, None, 'seq', [
        ('title', 'Trajectory computation'), ('mono', 'lib/mask2epi.py · radial | laminar'),
        ('para', "split each frame's samples into Nshots trains of ETL echoes; 3-step sorting: "
                 'min-sum TSP → bottleneck 2-opt → uncross (radial)'),
    ])

    # ---- four sequences, one lane each ----
    lw = 190
    seq_y = y_traj + traj_h + 90
    d.arrow(f'M{sc} {y_traj + traj_h} V{seq_y - 26} H{LANE["epi"]} V{seq_y - 2}')
    d.chip(sc, y_traj + traj_h + 30, 'schedules · Nframes × Nshots × ETL × (ky, kz, t)')
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
    seq_lines = {key: [('title', name), ('mono', mod), ('para', desc)]
                 for key, name, mod, desc in seqs}
    seq_h = row([(lw, lines) for lines in seq_lines.values()])
    for key, lines in seq_lines.items():
        d.box(LANE[key] - lw // 2, seq_y, lw, seq_h, 'seq' + (' key' if key == 'epi' else ''),
              lines)
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
    ge_y = seq_y + seq_h + 56
    for key, name, _, _ in seqs:
        d.arrow(f'M{LANE[key]} {seq_y + seq_h} V{ge_y - 2}')
        d.chip(LANE[key], seq_y + seq_h + 28, f'{name}.seq')
    ge_h = d.box(55, ge_y, 820, None, 'seq', [
        ('title', 'GE export'), ('mono', 'ge/ · main.py --ge'),
        ('para', 'checks all four first: gradient, slew, B1 limits · PNS (fails above 100%, '
                 'warns above 80%) · gradient acoustics (warns); then seq2ceq → writeceq writes '
                 'one .pge per sequence'),
    ])
    d.band(0, ge_y + ge_h + 20, 'seq', 'SEQUENCE DESIGN · main.py')

    # ---- scanner ----
    sc_y = ge_y + ge_h + 86
    pge_y = ge_y + ge_h + 42
    for key, name, _, _ in seqs:
        d.arrow(f'M{LANE[key]} {ge_y + ge_h} V{sc_y - 2}')
        d.chip(LANE[key], pge_y, f'{name}.pge')
    sc_h = d.box(55, sc_y, 820, None, 'scan', [
        ('title', 'GE scanner · pge2 interpreter'),
        ('para', 'runs each .pge as a pge2 entry; raw data saved as GE ScanArchives '
                 '(HDF5, read only by the Orchestra SDK)'),
    ])
    d.band(sc_y - 22, sc_y + sc_h + 20, 'scan', 'SCANNER')

    # ---- preprocess, raw data: two rows of steps in lanes 2-4 ----
    # row 1: whitening -> odd/even delay estimation -> gridding
    # row 2:             linear phase correction    -> placement in (ky, kz)
    pre_y0 = sc_y + sc_h + 70
    sa_y0 = pre_y0 + 18
    r1_y, sbw = sa_y0 + 36, 170
    ex, ew = LANE['epi'] - sbw // 2, sbw + 24  # lane-4 boxes: left edge, width
    cx0 = LANE['cal'] - sbw // 2  # lane-3 boxes: left edge
    whiten = [('title', 'Whitening'), ('mono', 'coils.py'),
              ('para', 'Estimate noise covariance matrix W')]
    oe_est = [('title', 'Odd/even delay estimation'), ('mono', 'oephase.py'),
              ('para', 'readout-delay sweep and linear phase estimation, on whitened '
                       'calibration data')]
    grid = [('title', 'Grid ramp-sampled data'), ('mono', 'epi_gridding.py'),
            ('para', 'whitened readouts onto Cartesian kx (1D NUFFT); odd/even kx shifted by '
                     'the calibrated delay')]
    corr = [('title', 'Linear phase correction'), ('mono', 'oephase.py'),
            ('para', 'subtracts a[0] + a[1]·x from every even echo')]
    place = [('title', 'Place samples at their (ky, kz) coordinates'), ('mono', 'preprocess.py'),
             ('para', 'place each (shot, echo) indexed sample into their respective k-space '
                      'location, zero-filling unsampled locations')]
    r1_h = row([(sbw, whiten), (sbw, oe_est), (ew, grid)])
    r2_y = r1_y + r1_h + 44
    r2_h = row([(sbw, corr), (ew, place)])
    sa_y1 = r2_y + r2_h + 18
    for key in ('noise', 'cal', 'epi'):
        d.arrow(f'M{LANE[key]} {sc_y + sc_h} V{r1_y - 2}')
    archives = {'gre': 'gre.h5', 'noise': '<seq>_noise.h5',
                'cal': '<seq>_cal.h5', 'epi': '<seq>_epi.h5'}
    for key, text in archives.items():
        d.chip(LANE[key], sc_y + sc_h + 45, text)
    d.add(f'<rect x="255" y="{sa_y0}" width="644" height="{sa_y1 - sa_y0}" rx="8" '
          'class="group pre"/>')
    # the group's name sits in the free slot under the whitening box
    for i, text in enumerate(['Raw-data processing', '(dotted): all physical', 'coils, cached per',
                              'frame, resumable']):
        d.label(LANE['noise'] - sbw // 2, r2_y + 16 + 18 * i, text, 'start', 't-section')
    d.box(LANE['noise'] - sbw // 2, r1_y, sbw, r1_h, 'pre', whiten)
    d.box(cx0, r1_y, sbw, r1_h, 'pre', oe_est)
    d.box(ex, r1_y, ew, r1_h, 'pre', grid)
    d.box(cx0, r2_y, sbw, r2_h, 'pre', corr)
    d.box(ex, r2_y, ew, r2_h, 'pre', place)
    my, my2 = r1_y + 44, r2_y + r2_h // 2
    d.arrow(f'M{LANE["noise"] + sbw // 2} {my} H{cx0 - 2}')
    d.label((LANE['noise'] + LANE['cal']) / 2, my - 6, 'W')
    d.arrow(f'M{cx0 + sbw} {my} H{ex - 2}')
    d.label((cx0 + sbw + ex) / 2, my - 6, 'delay')
    d.arrow(f'M{LANE["noise"] - sbw // 2} {my} H{LANE["gre"] + 3}')
    d.label((LANE['noise'] - sbw // 2 + LANE['gre']) / 2 + 8, my - 6, 'W')
    # gridded data drops to the correction; the phase model a comes straight down
    gap_y = r1_y + r1_h + 22
    d.arrow(f'M{LANE["epi"]} {r1_y + r1_h} V{gap_y} H{LANE["cal"] + 40} V{r2_y - 2}')
    d.arrow(f'M{LANE["cal"] - 40} {r1_y + r1_h} V{r2_y - 2}')
    d.label(LANE['cal'] - 48, gap_y + 4, 'a', 'end')
    d.arrow(f'M{cx0 + sbw} {my2} H{ex - 2}')

    # scan_info.mat goes around the scanner: kxo/kxe to the gridding, schedules to the placement
    d.arrow(f'M1070 {info_y + 84} V{my2} H{ex + ew + 2}', 'hand')
    d.arrow(f'M1070 {r1_y + r1_h // 2} H{ex + ew + 2}', 'hand')

    # ---- preprocess: deGRE maps | coil compression, R2* | k-space; then the output ----
    col_l, col_m, col_r = (55, 250), (345, 240), (625, 250)  # (x, width)
    smaps_x, b0_x = 72, 88  # B0 sits right of a lane carrying the smaps down to the output
    b0_w = col_l[0] + col_l[1] - b0_x
    espirit = [('title', 'Sensitivity maps (ESPIRiT)'), ('mono', 'smaps.py'),
               ('para', 'ESPIRiT on echo 1, resized to the EPI grid, masked and smoothed; then '
                        'compressed with GCC')]
    gcc = [('title', 'Coil compression fit (GCC)'), ('mono', 'coils.py'),
           ('para', 'geometric-decomposition coil compression (GCC) on the whitened deGRE data: '
                    'per-x SVD-based compression with default threshold at keeping 99.9% '
                    'of coil energy')]
    compress = [('title', 'Compress EPI k-space using GCC'),
                ('mono', 'preprocess.py · write_output()'),
                ('para', 'compress the cached k-space frame by frame with GCC')]
    b0 = [('title', 'B0 map (optional)'), ('mono-s', 'b0map.py + julia/b0map.jl'),
          ('para', 'MRIFieldmaps.jl on both echoes from a ROMEO.jl start; skip with --no-b0')]
    r2s = [('title', 'R2* map (optional)'), ('mono', 'r2star.py'),
           ('para', 'log-linear fit in the B0 mask (placeholder); skip with --no-r2star')]
    calib = [('title', 'Calibration region extraction'), ('mono', 'find_calib_region()'),
             ('para', 'extract the (ky, kz) block sampled in every frame, cut out as ksp_calib')]
    ya = sa_y1 + 66
    rha = row([(col_l[1], espirit), (col_m[1], gcc), (col_r[1], compress)])
    yb = ya + rha + 40
    rhb = row([(b0_w, b0), (col_m[1], r2s), (col_r[1], calib)])
    yc = yb + rhb + 40
    d.arrow(f'M{LANE["epi"]} {r2_y + r2_h} V{ya - 2}')
    d.chip(LANE['epi'], sa_y1 + 32, '<seq>_gridded.h5')
    # the whitened deGRE feeds ESPIRiT (down its lane) and GCC (branch)
    gcc_cx = col_m[0] + col_m[1] // 2
    d.arrow(f'M{LANE["gre"]} {sc_y + sc_h} V{ya - 2}')
    d.arrow(f'M{LANE["gre"]} {ya - 18} H{gcc_cx} V{ya - 2}')
    d.box(col_l[0], ya, col_l[1], rha, 'pre', espirit)
    d.box(col_m[0], ya, col_m[1], rha, 'pre', gcc)
    d.box(col_r[0], ya, col_r[1], rha, 'pre', compress)
    mid_a = ya + rha // 2
    d.arrow(f'M{col_m[0]} {mid_a} H{col_l[0] + col_l[1] + 2}')
    d.label((col_m[0] + col_l[0] + col_l[1]) / 2, mid_a - 6, 'GCC')
    d.arrow(f'M{col_m[0] + col_m[1]} {mid_a} H{col_r[0] - 2}')
    d.label((col_m[0] + col_m[1] + col_r[0]) / 2, mid_a - 6, 'GCC')

    b0_cx = b0_x + b0_w // 2
    d.box(b0_x, yb, b0_w, rhb, 'pre', b0, dashed=True)
    d.box(col_m[0], yb, col_m[1], rhb, 'pre', r2s, dashed=True)
    d.box(col_r[0], yb, col_r[1], rhb, 'pre', calib)
    d.arrow(f'M{b0_cx} {ya + rha} V{yb - 2}')
    d.label(b0_cx + 8, yb - 14, 'smaps, emap', 'start')
    mid_b = yb + rhb // 2
    d.arrow(f'M{col_l[0] + col_l[1]} {mid_b} H{col_m[0] - 2}')
    d.label((col_l[0] + col_l[1] + col_m[0]) / 2, mid_b - 6, 'mask')
    r_cx = col_r[0] + col_r[1] // 2
    d.arrow(f'M{r_cx} {ya + rha} V{yb - 2}')

    d.arrow(f'M{smaps_x} {ya + rha} V{yc - 2}')
    d.label(smaps_x + 6, yc - 14, 'smaps', 'start')
    d.arrow(f'M{b0_cx} {yb + rhb} V{yc - 2}', 'flow dashed')
    d.label(b0_cx + 8, yc - 14, 'b0_map', 'start')
    d.arrow(f'M{gcc_cx} {yb + rhb} V{yc - 2}', 'flow dashed')
    d.label(gcc_cx + 8, yc - 14, 'r2star_map', 'start')
    d.arrow(f'M{r_cx} {yb + rhb} V{yc - 2}')
    d.label(r_cx + 8, yc - 14, 'ksp_calib', 'start')
    r_x1 = col_r[0] + col_r[1]  # compressed k-space goes around the calibration box
    wh = d.box(55, yc, 820, None, 'pre', [
        ('title', 'Write all output to a single HDF5 file'),
        ('mono', 'write_output() → <seq>_preprocessed.h5'),
        ('para', 'ksp_epi_zf, ksp_calib, smaps, b0_map, r2star_map, omegas, echo_times, W, '
                 'GCC; attrs noise_var, t_ref_s. The three maps also go to .nii.gz + .json '
                 'for viewing'),
    ])
    d.arrow(f'M{r_x1} {mid_a} H{r_x1 + 22} V{yc + wh // 2} H{r_x1 + 2}')
    d.label(r_x1 + 30, mid_b + 4, 'ksp_epi_zf', 'start')
    pre_y1 = yc + wh + 22
    d.band(pre_y0, pre_y1, 'pre', 'PREPROCESS · .venv-preprocessing')

    # ---- recon: iterative SENSE (dotted: objective, then A, R and solver) | RSS ----
    rec_y0 = pre_y1 + 50
    bus_y, r_y = rec_y0 + 12, rec_y0 + 28
    gx0, gx1 = 55, 815  # the SENSE group
    ix0, iw = gx0 + 18, gx1 - gx0 - 36  # inside it
    rss_x, rss_w = gx1 + 20, 1085 - gx1 - 20
    rss_cx = rss_x + rss_w // 2
    obj_y = r_y + 36
    obj_cx = ix0 + iw // 2
    d.chip(470, pre_y1 + 25,
           '<seq>_preprocessed.h5 · ksp_epi_zf, smaps, b0_map, r2star_map, echo_times')
    d.line(f'M470 {yc + wh} V{bus_y}')
    d.line(f'M{obj_cx} {bus_y} H{rss_cx}')
    d.arrow(f'M{obj_cx} {bus_y} V{obj_y - 2}')
    d.arrow(f'M{rss_cx} {bus_y} V{r_y - 2}')
    rss_h = d.box(rss_x, r_y, rss_w, None, 'rec', [
        ('title', 'Root-sum-of-squares (RSS) coil-combined IFT reconstruction'),
        ('mono', 'rss.py'),
        ('para', 'zero-filled inverse FFT per coil, then RSS over coils; first look, '
                 'aliasing kept'),
    ])
    d.label(gx0 + 14, r_y + 22, 'Iterative SENSE (dotted) · sense.py', 'start', 't-section')
    obj_h = d.box(ix0, obj_y, iw, None, 'rec', [
        ('title', 'Objective'), ('mono', 'sense.py · run_sense()'),
        ('mathc', r'\hat{x} = \arg\min_x \ \dfrac{1}{2}\,\|Ax - y\|_2^2 + R(x)'),
        ('para', 'x: one 3D image per frame. y: only the sampled k-space points, scaled to unit '
                 'noise variance. R(x) is the regularizer. For mslr, A is divided by its largest '
                 'singular value, and λ defaults to the acceleration factor.'),
    ])
    cw = 228  # three columns: A | R | solver
    cg = (iw - 3 * cw) // 2
    cols = [ix0, ix0 + cw + cg, ix0 + 2 * (cw + cg)]
    head_y = obj_y + obj_h + 28
    for x, text in zip(cols, ['A · ENCODING OPERATOR', 'R · REGULARIZER  (--reg)', 'SOLVER']):
        d.label(x, head_y, text, 'start', 't-colhead')
    operators = [  # (lines, optional?); one definition per 'text' line
        ([('title', 'SENSE'), ('mono', 'operators.SENSE'),
          ('math', r'A = \Omega\,F\,S'),
          ('text', 'S: coil sensitivity maps (smaps)'),
          ('text', 'F: centered 3D FFT'),
          ('text', 'Ω: keep only the sampled (ky, kz)')], False),
        ([('title', '+ B0 off-resonance (--B0)'), ('mono', 'operators.SENSE_B0'),
          ('math', r'A = \sum_{l=1}^{L} W_l\, \Omega\,F\,S\, \Phi_l'),
          ('math', r'\Phi_l = \mathrm{diag}(e^{\,i 2\pi \Delta f(\mathbf{r})\, t_l})'),
          ('text', 'Φₗ: B0 phase at segment time tₗ'),
          ('text', 'Δf: field map (b0_map)'),
          ('text', 'Wₗ: per-sample segment weights'),
          ('text', 'L: number of time segments (32)')], True),
        ([('title', '+ R2* decay (--R2star)'), ('mono', 'operators.SENSE_B0_R2star'),
          ('math', r'\Phi_l = \mathrm{diag}(e^{\,\psi(\mathbf{r})\,(t_l - \mathrm{TE})})'),
          ('math', r'\psi = i 2\pi \Delta f(\mathbf{r}) - R_2^*(\mathbf{r})'),
          ('text', 'R₂*: R2* map (r2star_map)'),
          ('text', 'TE: nominal echo time'),
          ('text', 'A as in + B0; needs --B0')], True),
    ]
    regularizers = [  # (R, the solver it uses)
        ([('title', 'None (--reg none)'), ('math', r'R(x) = 0'),
          ('para', 'plain least-squares SENSE')],
         [('title', 'Conjugate gradient'), ('mono', 'solvers.cg'),
          ('math', r'A^H A\, x = A^H y'), ('para', 'fast when there is no regularizer')]),
        ([('title', 'Wavelet + TV (--reg wavelet-tv)'), ('mono', 'WaveletTV'),
          ('math', r'R(x) = \lambda_{\ell_1}\|Wx\|_1 + \lambda_{TV}\|Dx\|_1'),
          ('para', 'per frame: sparse 3D wavelet coefficients and image gradients')],
         [('title', 'Primal-dual (PDHG)'), ('mono', 'solvers.pdhg'),
          ('para', 'via mirtorch FBPD; TV has no closed-form prox')]),
        ([('title', 'Multi-scale low rank (--reg mslr)'), ('mono', 'MultiScaleLowRank'),
          ('math', r'R(x) = \sum_k \lambda_k \sum_b \|P_b(x_k)\|_*'),
          ('math', r'x = \sum_k x_k'),
          ('para', 'one component per patch scale; nuclear norm of each space × time patch')],
         [('title', 'POGM with restart'), ('mono', 'solvers.pogm_restart'),
          ('para', 'proximal gradient; the prox is singular-value soft-thresholding of each '
                   'patch. --mom fpgm or pgm also available')]),
    ]
    y = head_y + 14
    for (a_lines, a_opt), (r_lines, s_lines) in zip(operators, regularizers):
        h = row([(cw, a_lines), (cw, r_lines), (cw, s_lines)])
        d.box(cols[0], y, cw, h, 'rec', a_lines, dashed=a_opt)
        d.box(cols[1], y, cw, h, 'rec', r_lines)
        d.box(cols[2], y, cw, h, 'rec', s_lines)
        d.arrow(f'M{cols[1] + cw} {y + h // 2} H{cols[2] - 2}')
        y += h + 20
    g_y1 = y - 20 + 18
    d.add(f'<rect x="{gx0}" y="{r_y}" width="{gx1 - gx0}" height="{g_y1 - r_y}" rx="8" '
          'class="group rec"/>')

    # ---- recon output, below both methods ----
    out_y = g_y1 + 40
    d.arrow(f'M{obj_cx} {g_y1} V{out_y - 2}')
    d.arrow(f'M{rss_cx} {r_y + rss_h} V{out_y - 2}')
    out_h = 84
    d.tag(55, out_y, 1085 - 55, out_h, [
        ('mono-b', '<datdir>/recon/'),
        ('mono-s', 'rss/<seq>_recon.nii.gz + .json'),
        ('mono-s', 'sense_<reg>[_b0 | _b0r2star]/<seq>_recon.h5 + .nii.gz + .json'),
        ('text', '.nii.gz: magnitude image for viewing · .json: every setting, acceleration '
                 'factor, σ1(A), runtime '
                 '· .h5: complex image and per-iteration solver traces'),
    ])
    rec_y1 = out_y + out_h + 20
    d.band(rec_y0, rec_y1, 'rec', 'RECON · uv (.venv-recon)')
    return rec_y1 + 12


def render(theme):
    tokens = ''.join(f'--{k}: {v}; ' for k, v in THEMES[theme].items())
    d = Drawing()
    draw_legend(d)
    legend = d.out + d.front
    d.out, d.front = [], []
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
    body = [f'<g transform="translate(0 {LEGEND_H})">', *d.back, *d.out, *d.front,
            '</g>\n</svg>\n']
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
