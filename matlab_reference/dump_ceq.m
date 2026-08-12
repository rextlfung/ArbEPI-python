function dump_ceq(seqfile, matfile)
%% dump_ceq  Run seq2ceq on a .seq file and dump the Ceq struct to a v7.3 .mat file.
%
%   Used to validate the from-scratch Python port of seq2ceq.m
%   (ge/seq2ceq.py) against the reference MATLAB implementation, field by
%   field, independent of the .pge binary format (so a mismatch can be
%   localized to seq2ceq vs writeceq).
%
%   seqfile:  path to a .seq file
%   matfile:  output .mat path (written with '-v7.3' so hdf5storage.loadmat
%             can read it from Python -- see CLAUDE.md's .mat file format note)
%
% Requires pulseq, toppe, PulCeq on the MATLAB path -- see ge_export.py,
% which sets this up before invoking this function.

ceq = seq2ceq(seqfile);

nParentBlocks = ceq.nParentBlocks;
nSegments = ceq.nSegments;
nMax = ceq.nMax;
nReadouts = ceq.nReadouts;
duration = ceq.duration;
loop = ceq.loop;

% Parent blocks: flatten to arrays indexable by parent block ID (1-based)
parentBlockDuration = zeros(nParentBlocks, 1);
for p = 1:nParentBlocks
    parentBlockDuration(p) = ceq.parentBlocks(p).block.blockDuration;
end

% Segments
segmentTRID = zeros(nSegments, 1);
segmentNBlocks = zeros(nSegments, 1);
segmentEmaxN = zeros(nSegments, 1);
segmentBlockIDs = cell(nSegments, 1);
for i = 1:nSegments
    segmentTRID(i) = ceq.segments(i).TRID;
    segmentNBlocks(i) = ceq.segments(i).nBlocksInSegment;
    segmentEmaxN(i) = ceq.segments(i).Emax.n;
    segmentBlockIDs{i} = ceq.segments(i).blockIDs;
end

save(matfile, 'nParentBlocks', 'nSegments', 'nMax', 'nReadouts', 'duration', ...
    'loop', 'parentBlockDuration', 'segmentTRID', 'segmentNBlocks', ...
    'segmentEmaxN', 'segmentBlockIDs', '-v7.3');

fprintf('Wrote %s (nParentBlocks=%d, nSegments=%d, nMax=%d)\n', matfile, nParentBlocks, nSegments, nMax);
end
