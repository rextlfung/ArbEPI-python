function ceq = write_to_ge_from_seq(seqfile, filepath, sysPGE2, PNSwt, pislquant)
%% write_to_ge_from_seq  Convert a Pulseq .seq file to GE TOPPE .pge format.
%   Adapted from ../ArbEPI/lib/write_to_ge.m to start from a .seq filename
%   (as written by the Python port) instead of an in-memory mr.Sequence
%   object, so the existing, already-tested GE export logic — including
%   check_grad_acoustics.m from ../ArbEPI/lib — can be reused unchanged.
%
%   seqfile:   path to a .seq file (e.g. as written by arbepi.sequences.arbepi)
%   filepath:  full output path without extension (e.g. fullfile(outputDir, seqname))
%   sysPGE2:   GE system struct from pge2.opts(...)
%   PNSwt:     PNS direction weights [x y z]
%   pislquant: number of ADC events at scan start for receive gain calibration
%
% Requires pulseq, toppe, PulCeq (+pge2, seq2ceq), and ArbEPI/lib
% (check_grad_acoustics) on the MATLAB path — see arbepi/ge_export.py,
% which sets this up before invoking this function.
%
% Returns ceq (compact sequence representation) for optional downstream use
% (acoustics check, plotting).

seq = mr.Sequence();
seq.read(seqfile);

ceq = seq2ceq(seq);
hfig = figure('Visible', 'off');
S = pge2.plot(ceq, sysPGE2, 'blockRange', [1 10], 'rotate', false, 'interpolate', true, 'wt', PNSwt);
close(hfig);

% Check for forbidden gradient frequencies on GE MR750 (xrm)
check_grad_acoustics(reshape([S.gx.signal S.gy.signal S.gz.signal], [length(S.gx.signal), 1, 3])/100, 'xrm', [0, 0]);

pge_params = pge2.check(ceq, sysPGE2, 'wt', PNSwt);
pge2.writeceq(ceq, [filepath '.pge'], 'pislquant', pislquant, 'params', pge_params);
end
