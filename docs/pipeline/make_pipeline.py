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
its box, using per-character widths that are generous for Helvetica/Arial.

Layout: one column of steps (config -> masks -> partitioning) that fans out
into four lanes, one per sequence (deGRE, noise, EPIcal, ArbEPI). Each lane
keeps its x position from sequence generation through the scanner into the
preprocessing step that consumes that sequence's raw data. scan_info.mat is the
dashed line on the right: it goes around the scanner.
"""

import os
from html import escape

HERE = os.path.dirname(os.path.abspath(__file__))
W, H_BODY, LEGEND_H = 1100, 1560, 40
H = H_BODY + LEGEND_H

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
"""

# Upper-bound pixel width per character for each text class, used only to
# catch lines that would overflow their box after an edit.
CHAR_PX = {'title': 7.8, 'mono': 7.2, 'mono-s': 6.6, 'mono-b': 7.5, 'text': 6.6}

ARIA = (
    'ArbEPI workflow: scan configuration, sampling masks and echo-train partitioning produce '
    'four Pulseq sequences (deGRE, noise, EPIcal, ArbEPI); these are exported to GE .pge files '
    "and run on the scanner; each sequence's raw ScanArchive feeds its own preprocessing step "
    '(whitening, odd/even calibration, gridding, coil maps and B0); the preprocessed file feeds '
    'RSS or iterative SENSE reconstruction. scan_info.mat bypasses the scanner and carries the '
    'sampling schedule to preprocessing.'
)


class Drawing:
    def __init__(self):
        self.out = []

    def add(self, s):
        self.out.append(s)

    def box(self, x, y, w, h, cls, lines, dashed=False):
        """A processing step: first line is its title, then module path, then details."""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" '
                 f'class="box {cls}{" dashed" if dashed else ""}"/>')
        ty, prev = y, None
        for kind, text in lines:
            ty += 22 if prev is None else 18 if prev in ('title', 'mono') else 16
            self._check_fit(kind, text, w - 32)
            self.add(f'<text x="{x + 16}" y="{ty}" class="t-{kind}">{escape(text)}</text>')
            prev = kind
        if ty > y + h - 12:
            raise ValueError(f'{lines[0][1]!r}: text runs past the bottom of its box')

    def chip(self, cx, cy, text, cls=''):
        """A file or array passed between steps, centered on an arrow."""
        w = round(len(text) * 7.2 + 22)
        self.add(f'<rect x="{cx - w / 2:.1f}" y="{cy - 11}" width="{w}" height="22" rx="11" '
                 f'class="chip {cls}"/>')
        self.add(f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" class="t-chip">'
                 f'{escape(text)}</text>')

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
        self.add(f'<rect x="0" y="{y0}" width="{W}" height="{y1 - y0}" rx="10" '
                 f'class="band {cls}"/>')
        cy = (y0 + y1) / 2
        self.add(f'<text x="30" y="{cy}" transform="rotate(-90 30 {cy})" text-anchor="middle" '
                 f'class="rail {cls}">{escape(name)}</text>')

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
    d.label(402, y + 2, 'optional input', 'start', 't-legend')
    d.line(f'M512 {y - 3} H546', 'hand')
    d.label(554, y + 2, 'scan_info.mat, which goes around the scanner', 'start', 't-legend')


def draw(d):
    # ---- environment bands ----
    d.band(0, 778, 'seq', 'SEQUENCE DESIGN · main.py')
    d.band(822, 924, 'scan', 'SCANNER')
    d.band(974, 1340, 'pre', 'PREPROCESS · .venv-preprocessing')
    d.band(1390, 1548, 'rec', 'RECON · .venv-recon')

    # ---- one column: configuration -> masks -> partitioning ----
    sx, sw = 380, 440
    sc = sx + sw // 2
    d.box(60, 28, 270, 92, 'seq', [
        ('mono-b', 'scanners.py'),
        ('text', 'ScannerSpec: GE_MR750 | GE_UHP'),
        ('text', 'max gradient, slew, B1, PNS;'),
        ('text', 'ge/ checks read the same spec'),
    ])
    d.arrow('M330 74 H378')
    d.box(sx, 28, sw, 92, 'seq', [
        ('title', 'Scan configuration'),
        ('mono', 'params.py · load_params() → Params'),
        ('text', 'resolution, matrix, R, ETL, TE, volume TR, readout slews'),
        ('text', 'derives Nshots, then TR and flip angle'),
    ])
    d.box(860, 28, 225, 92, 'seq', [
        ('title', 'Custom mask (optional)'),
        ('mono-s', 'sampling/external_mask.py'),
        ('text', '(ky, kz) or (ky, kz, t) .mat'),
        ('text', 'sets Nshots and effective R'),
    ], dashed=True)
    d.arrow('M860 74 H822', 'flow dashed')

    d.arrow(f'M{sc} 120 V174')
    d.chip(sc, 148, 'Params (+ sys, spec)')
    d.box(sx, 176, sw, 92, 'seq', [
        ('title', 'Sampling masks'),
        ('mono', 'sampling/ · resolve_omegas()'),
        ('text', 'one (ky, kz) mask per frame: pd · caipi · ticaipi · rand'),
        ('text', 'or the custom mask, passed through from Params'),
    ])
    d.arrow(f'M{sc} 268 V322')
    d.chip(sc, 296, 'omegas · Ny × Nz × Nframes, bool')
    d.box(sx, 324, sw, 92, 'seq', [
        ('title', 'Echo-train partitioning'),
        ('mono', 'lib/mask2epi.py · radial | laminar'),
        ('text', "split each frame's samples into Nshots trains of ETL echoes"),
        ('text', 'order: min-sum TSP → bottleneck 2-opt → uncross (radial)'),
    ])
    d.arrow(f'M{sc} 416 V480 H{LANE["epi"]} V504')
    d.chip(sc, 446, 'schedules · Nframes × Nshots × ETL × (ky, kz, t)')

    # ---- four sequences, one lane each ----
    d.label(55, 492, 'Sequence generation · sequences/ (pypulseq)', 'start', 't-section')
    seq_y, seq_h, lw = 506, 108, 190
    seqs = [
        ('gre', 'deGRE', 'deGRE.py', ['dual-echo 3D GRE', 'patches TE_degre in', 'scan_info.mat']),
        ('noise', 'noise', 'noise.py',
         ['ADC only: no RF or', 'gradients, same readout', 'geometry as ArbEPI']),
        ('cal', 'EPIcal', 'EPIcal.py',
         ['ArbEPI readout with', 'every blip at 0', '(ghost calibration)']),
        ('epi', 'ArbEPI', 'ArbEPI.py',
         ['POPE readout, blips', 'from schedules,', 'TE/TR delays, trap4ge']),
    ]
    for key, name, mod, det in seqs:
        d.box(LANE[key] - lw // 2, seq_y, lw, seq_h, 'seq' + (' key' if key == 'epi' else ''),
              [('title', name), ('mono', mod)] + [('text', t) for t in det])
    d.tag(905, 518, 180, 84, [
        ('mono-b', 'scan_info.mat'),
        ('text', 'schedules, kxo/kxe,'),
        ('text', 'echo times, TE_degre,'),
        ('text', 'scan scalars (v7.3)'),
    ], 'hand')
    d.arrow(f'M{LANE["epi"] + lw // 2} 560 H903')

    # ---- GE export ----
    ge_y, ge_h = 670, 88
    for key, name, _, _ in seqs:
        d.arrow(f'M{LANE[key]} {seq_y + seq_h} V{ge_y - 2}')
        d.chip(LANE[key], 642, f'{name}.seq')
    d.box(55, ge_y, 820, ge_h, 'seq', [
        ('title', 'GE export'),
        ('mono', 'ge/ · main.py --ge'),
        ('text', 'checks all four first: gradient, slew, B1 limits · PNS (fails above 100%, '
                 'warns above 80%) · acoustics (warns)'),
        ('text', 'then seq2ceq → writeceq writes one .pge per sequence '
                 '(pure Python port of PulCeq)'),
    ])

    # ---- scanner ----
    sc_y, sc_h = 844, 60
    for key, name, _, _ in seqs:
        d.arrow(f'M{LANE[key]} {ge_y + ge_h} V{sc_y - 2}')
        d.chip(LANE[key], 800, f'{name}.pge')
    d.label(905, 794, 'copy to the scanner:', 'start', 't-note')
    d.label(905, 810, 'ge/coppe.py (optional)', 'start', 't-note-mono')
    d.box(55, sc_y, 820, sc_h, 'scan', [
        ('title', 'GE scanner · pge2/tv7 interpreter'),
        ('text', 'runs each .pge as a pge2 entry; raw data saved as GE ScanArchives '
                 '(HDF5, read only by the Orchestra SDK)'),
    ])

    # ---- preprocess: Stage A per lane, then compression + deGRE maps ----
    archives = {'gre': 'gre.h5 · shared', 'noise': '<seq>_noise.h5 (opt.)',
                'cal': '<seq>_cal.h5', 'epi': '<seq>_epi.h5'}
    sa_y0, sa_y1 = 992, 1150
    sb_y, sb_h, sbw = 1028, 104, 170
    c_y, c_h = 1210, 108
    for key in ('noise', 'cal', 'epi'):
        d.arrow(f'M{LANE[key]} {sc_y + sc_h} V{sb_y - 2}')
    d.arrow(f'M{LANE["gre"]} {sc_y + sc_h} V{c_y - 2}')
    for key, text in archives.items():
        d.chip(LANE[key], 949, text)

    d.add(f'<rect x="255" y="{sa_y0}" width="620" height="{sa_y1 - sa_y0}" rx="8" '
          'class="group pre"/>')
    d.label(262, 1172, 'Stage A (dotted) · all coils, cached, resumable', 'start', 't-section')
    stage_a = [
        ('noise', 'Whitening', 'coils.py',
         ['noise covariance → W', 'noise_var measured', 'after gridding']),
        ('cal', 'Odd/even phase', 'oephase.py',
         ['readout-delay sweep,', 'then ghost phase a', 'on whitened cal data']),
        ('epi', 'Grid + scatter', 'epi_gridding.py',
         ['ramp-sample regrid,', 'odd/even correction,', 'scatter by schedules']),
    ]
    for key, title, mod, det in stage_a:
        d.box(LANE[key] - sbw // 2, sb_y, sbw, sb_h, 'pre',
              [('title', title), ('mono', mod)] + [('text', t) for t in det])
    my = sb_y + sb_h // 2 - 8
    d.arrow(f'M{LANE["noise"] + sbw // 2} {my} H{LANE["cal"] - sbw // 2 - 2}')
    d.label((LANE['noise'] + LANE['cal']) / 2, my - 6, 'W')
    d.arrow(f'M{LANE["cal"] + sbw // 2} {my} H{LANE["epi"] - sbw // 2 - 2}')
    d.label((LANE['cal'] + LANE['epi']) / 2, my - 6, 'W, a')
    d.arrow(f'M{LANE["noise"] - sbw // 2} {my} H{LANE["gre"] + 3}')
    d.label((LANE['noise'] - sbw // 2 + LANE['gre']) / 2 + 8, my - 6, 'W')

    d.arrow(f'M{LANE["epi"]} {sb_y + sb_h} V{c_y - 2}')
    d.chip(LANE['epi'], 1180, '<seq>_gridded.h5')
    d.box(55, c_y, 820, c_h, 'pre', [
        ('title', 'Coil compression, deGRE maps, Stage B output'),
        ('mono', 'coils.py · smaps.py · b0map.py + julia/b0map.jl · r2star.py'),
        ('text', 'GCC fit on the whitened deGRE: per-x matrices keeping 99.9% of the '
                 'eigenvalue energy'),
        ('text', 'deGRE → ESPIRiT sensitivity maps · B0 field map (MRIFieldmaps.jl, ROMEO start)'
                 ' · R2*'),
        ('text', 'Stage B: compress the cached k-space; write it with the calibration region, '
                 'maps and noise_var'),
    ])

    # scan_info.mat goes around the scanner, straight to preprocessing
    d.arrow(f'M1070 602 V{my + 22} H{LANE["epi"] + sbw // 2 + 2}', 'hand')
    notes = [
        ('t-note-b', 'read by preprocess from'),
        ('t-note-mono', '<datdir>/seqs/<seq>/'),
        ('t-note', 'schedules → (ky, kz)'),
        ('t-note', 'kxo/kxe → gridding'),
        ('t-note', 'echo times → B0 model'),
    ]
    for i, (cls, text) in enumerate(notes):
        d.label(905, 634 + i * 16 + (6 if i >= 2 else 0), text, 'start', cls)

    # ---- recon ----
    r_y, r_h = 1418, 108
    d.line(f'M470 {c_y + c_h} V1402')
    d.line('M180 1402 H600')
    d.arrow('M180 1402 V1416')
    d.arrow('M600 1402 V1416')
    d.chip(470, 1365, '<seq>_preprocessed.h5 · ksp_epi_zf, smaps, b0map_hz, r2star, echo_times')
    d.box(55, r_y, 250, r_h, 'rec', [
        ('title', 'RSS'),
        ('mono', 'rss.py'),
        ('text', 'zero-filled IFFT per coil,'),
        ('text', 'root-sum-of-squares'),
        ('text', 'first look, aliasing kept'),
    ])
    d.box(325, r_y, 550, r_h, 'rec', [
        ('title', 'Iterative SENSE'),
        ('mono', 'sense.py · operators.py · regularizers.py · solvers.py'),
        ('text', 'min over x of ½‖Ax − y‖² + g(x)'),
        ('text', 'A: SENSE · + B0 (time-segmented, L = 32) · + B0 + R2*'),
        ('text', 'g → solver: none → CG · lowrank → POGM · wavelet-tv → PDHG'),
    ])
    d.arrow('M875 1472 H903')
    d.tag(905, 1416, 180, 116, [
        ('mono-b', '<datdir>/recon/'),
        ('mono-s', 'sense_<reg>/'),
        ('mono-s', 'sense_<reg>_b0/'),
        ('mono-s', 'sense_<reg>_b0r2star/'),
        ('mono-s', 'rss/'),
        ('text', '.h5 + .nii.gz + .json'),
    ])


def render(theme):
    tokens = ''.join(f'--{k}: {v}; ' for k, v in THEMES[theme].items())
    d = Drawing()
    d.add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
          f'height="{H}" role="img" aria-label="{escape(ARIA)}">')
    d.add(f'<title>ArbEPI pipeline</title>\n<style>svg {{ {tokens}}}{STYLE}</style>')
    d.add('<defs>')
    for mid, cls in (('ah', 'ah'), ('ah-hand', 'ah-hand')):
        d.add(f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
              f'markerHeight="7" orient="auto-start-reverse">'
              f'<path d="M0,0 L10,5 L0,10 z" class="{cls}"/></marker>')
    d.add('</defs>')
    d.add(f'<rect width="{W}" height="{H}" rx="12" class="bg"/>')
    draw_legend(d)
    d.add(f'<g transform="translate(0 {LEGEND_H})">')
    draw(d)
    d.add('</g>\n</svg>\n')
    return '\n'.join(d.out)


def svg_path(theme):
    return os.path.join(HERE, f'pipeline-{theme}.svg')


def main():
    for theme in THEMES:
        with open(svg_path(theme), 'w', encoding='utf-8') as f:
            f.write(render(theme))
        print(f'wrote {os.path.relpath(svg_path(theme))}')


if __name__ == '__main__':
    main()
