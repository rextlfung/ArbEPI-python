function ceq = write_to_ge_from_seq(seqfile, filepath, sysPGE2, PNSwt, pislquant, coil)
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
%   coil:      gradient coil code, e.g. 'xrm' (MR750) or 'hrmbuhp' (UHP) --
%              must match the coil sysPGE2 was built for. See
%              ge_feasibility_check.m / pge2.opts.m for the full table.
%
% Requires pulseq, toppe, PulCeq (+pge2, seq2ceq), and ArbEPI/lib
% (check_grad_acoustics) on the MATLAB path — see arbepi/ge_export.py,
% which sets this up before invoking this function.
%
% Returns ceq (compact sequence representation) for optional downstream use
% (acoustics check, plotting).

[pge_params, ceq] = ge_feasibility_check(seqfile, sysPGE2, PNSwt, coil);
pge2.writeceq(ceq, [filepath '.pge'], 'pislquant', pislquant, 'params', pge_params);
end
