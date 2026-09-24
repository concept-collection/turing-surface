% Solve the implicit diffusion system
%
%   (I - dtD*lap_g) X = B,  i.e.  A*X = B  with  A*x = M.*x - dtD*dlap(x)
%
% exactly, to fp32 round-off, by a dense LU factorization of A. Where the
% other solvers apply the operator through dlap and iterate, this one has
% the host write the operator down: since dlap(x) = (analys(lap_g x) +
% (lam./jhat).*x).*filt, the preconditioner's jhat cancels on the band and
% A = I - dtD*K, where K is the matrix of x -> dlap(x) - (lam./jhat).*x.*filt
% -- the geometry alone, independent of dt, D and jhat. lusolve
% (src/mgpu/exact.ts) assembles K once per surface by pushing each of the
% 2*nlm unit vectors through the very same compiled dlap (an edited
% lib/dlap.m included), factors A with partial pivoting on the GPU whenever
% dtD or jhat changes, and answers each call with a pair of triangular
% solves in a single dispatch. Nothing is iterated, so niter plays no part.
%
% The cost is the dense matrix: (2*nlm)^2 floats, one copy for K and one
% factorization per solve call site (per species). That is 0.3 MB at lmax
% 15 and 4.5 MB at lmax 31, where assembling K takes well under a second;
% at lmax 63 it is 69 MB per copy, the assembly runs 2*nlm = 4160 operator
% evaluations, and each refactorization is about a second of GPU time. It is
% a small-lmax solver, and the reference the iterative ones are measured
% against: their iterates converge toward this answer as niter grows.
%
% The arguments after dtD are the operator's, passed through unchanged; the
% host builds K from the same buffers, so it refuses anything else there.

function X = exact(B, dtD, lam, filt, jhat, p2, r, dp1, dq2, jinv)
  X = lusolve(B, dtD, lam, filt, jhat, p2, r, dp1, dq2, jinv);
end
