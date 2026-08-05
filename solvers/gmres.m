% Solve the implicit diffusion system
%
%   (I - dtD*lap_g) X = B,  i.e.  A*X = B  with  A*x = M.*x - dtD*dlap(x)
%
% by right-preconditioned GMRES(niter) — one Arnoldi sweep of niter
% iterations, no restart — with the round-sphere operator M = 1 + dtD*lam as
% the preconditioner. Same operator and preconditioner as the other two
% solvers; what GMRES adds over bicgstab is optimality (the residual is
% minimized over the whole Krylov space, monotonically nonincreasing) at the
% price of storing the basis and the O(niter^2) orthogonalization sweep.
%
% The bookkeeping no other solver needs — the basis, the Hessenberg matrix,
% the Givens rotations — lives in banks and small matrices accessed through
% getslab/setslab and getat/setat (src/mgpu/externals.ts): functional
% updates the planner turns into static-offset buffer copies once the
% unrolled loop's variable makes every index a literal. That is also why the
% triangular loops below (`for i = 1:j`) compile: each unrolled j plans its
% own inner trip count.
%
% Scalars are GPU-resident 1-element values throughout, as in bicgstab, and
% every ratio is guarded the same way (a*b/(b*b + 1e-30), and 1/sqrt(x) as
% 1/sqrt(x + 1e-30)), so a converged or broken-down iteration contributes
% zero coefficients instead of NaNs. sqrt takes abs() of its argument
% because the compiler cannot see that <w, w> is nonnegative.
%
% niter is fixed at compile time. X0 is the round-sphere answer, and the
% correction added at the end is X = X0 + M \ (V*y) accumulated slab by
% slab.

function X = gmres(B, dtD, lam, filt, wlm, Vtx, Vty, Vtz, Vpx, Vpy, Vpz, nlm, niter)
  M = 1 + dtD * lam;
  X = B ./ M;
  dL0 = dlap(X, filt, Vtx, Vty, Vtz, Vpx, Vpy, Vpz, lam);
  R = B - (M .* X - dtD * dL0);

  % The Krylov basis (niter+1 spectral fields), the Hessenberg matrix, the
  % rotated right-hand side, the Givens coefficients, and the solve's y.
  VB = zeros(2, nlm * (niter + 1));
  H = zeros(niter + 1, niter);
  g = zeros(niter + 1, 1);
  c = zeros(niter, 1);
  s = zeros(niter, 1);
  y = zeros(niter, 1);

  Rw = R .* wlm;
  r2 = dot(Rw, R);
  nr = sqrt(abs(r2));
  invnr = nr / (r2 + 1e-30);
  V1 = R * invnr;
  VB = setslab(VB, V1, 1);
  g = setat(g, nr, 1);

  for j = 1:niter
    % w = A * M \ v_j
    Vj = getslab(VB, j);
    Zj = Vj ./ M;
    dLz = dlap(Zj, filt, Vtx, Vty, Vtz, Vpx, Vpy, Vpz, lam);
    W = M .* Zj - dtD * dLz;

    % Modified Gram-Schmidt against every basis vector so far.
    for i = 1:j
      Vi = getslab(VB, i);
      Viw = Vi .* wlm;
      hij = dot(Viw, W);
      H = setat(H, hij, i, j);
      W = W - hij * Vi;
    end
    Ww = W .* wlm;
    w2 = dot(Ww, W);
    hn = sqrt(abs(w2));
    H = setat(H, hn, j + 1, j);
    invh = hn / (w2 + 1e-30);
    Vn = W * invh;
    VB = setslab(VB, Vn, j + 1);

    % Apply the previous Givens rotations to column j, then form the new
    % one that zeroes H(j+1, j), and rotate g with it.
    for i = 1:j-1
      a = getat(H, i, j);
      b = getat(H, i + 1, j);
      ci = getat(c, i);
      si = getat(s, i);
      t1 = ci * a + si * b;
      t2 = ci * b - si * a;
      H = setat(H, t1, i, j);
      H = setat(H, t2, i + 1, j);
    end
    a = getat(H, j, j);
    b = getat(H, j + 1, j);
    rr = a * a + b * b;
    invr = 1 / sqrt(abs(rr) + 1e-30);
    cj = a * invr;
    sj = b * invr;
    c = setat(c, cj, j);
    s = setat(s, sj, j);
    t1 = cj * a + sj * b;
    H = setat(H, t1, j, j);
    gj = getat(g, j);
    t3 = cj * gj;
    t4 = -(sj * gj);
    g = setat(g, t3, j);
    g = setat(g, t4, j + 1);
  end

  % Back-substitute the rotated (upper triangular) system H*y = g.
  for j = niter:-1:1
    acc = getat(g, j);
    for i = j+1:niter
      rji = getat(H, j, i);
      yi = getat(y, i);
      acc = acc - rji * yi;
    end
    rjj = getat(H, j, j);
    yj = (acc * rjj) / (rjj * rjj + 1e-30);
    y = setat(y, yj, j);
  end

  % X = X0 + M \ (V * y), slab by slab.
  for j = 1:niter
    Vj = getslab(VB, j);
    yj = getat(y, j);
    X = X + yj * (Vj ./ M);
  end
end
