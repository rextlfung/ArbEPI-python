function [pge_params, ceq] = ge_feasibility_check(seqfile, sysPGE2, PNSwt, coil)
%% ge_feasibility_check  Run GE hardware/PNS/acoustic-resonance checks on a .seq file, without writing a .pge.
%
%   Factored out of write_to_ge_from_seq.m so it can run as a fast,
%   standalone pre-flight right after .seq generation -- surfacing
%   hardware-limit, PNS, or acoustic-resonance infeasibility in seconds,
%   instead of only being discovered at the end of a full .pge export.
%
%   seqfile:   path to a .seq file (e.g. as written by arbepi.sequences.arbepi)
%   sysPGE2:   GE system struct from pge2.opts(...)
%   PNSwt:     PNS direction weights [x y z]
%   coil:      gradient coil code, e.g. 'xrm' (MR750) or 'hrmbuhp' (UHP) --
%              see pge2.opts.m's header comment for the full table. Must
%              match the coil sysPGE2 was built for.
%
% Requires pulseq, toppe, PulCeq (+pge2, seq2ceq), and ArbEPI/lib
% (check_grad_acoustics) on the MATLAB path -- see arbepi/ge_export.py,
% which sets this up before invoking this function.
%
% Returns pge_params (from pge2.check, needed by pge2.writeceq) and ceq
% (the compact sequence representation) for optional downstream use.

seq = mr.Sequence();
seq.read(seqfile);

ceq = seq2ceq(seq);
hfig = figure('Visible', 'off');
S = pge2.plot(ceq, sysPGE2, 'blockRange', [1 10], 'rotate', false, 'interpolate', true, 'wt', PNSwt);
close(hfig);

check_grad_acoustics(reshape([S.gx.signal S.gy.signal S.gz.signal], [length(S.gx.signal), 1, 3])/100, coil, [0, 0]);

pge_params = pge2.check(ceq, sysPGE2, 'wt', PNSwt);
end
