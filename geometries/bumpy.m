% Lobes around the equator, plus a gentle pear-shaped offset along z.
%
% The radial deformation sin(theta)^4 * cos(nlobe*phi) is the shape of a
% sectoral harmonic of order nlobe, concentrated at the equator and vanishing
% at both poles; the cos(theta) term breaks the north-south symmetry so the two
% ends differ.
%
% `nlobe` should be a whole number. A fraction makes cos(nlobe*phi) disagree
% with itself across the phi = 0 seam, which is not a closed surface; the
% analysis into coefficients will band-limit whatever results, but what comes
% back is not what this file says.

function [gx, gy, gz] = shape(theta, phi, amp, nlobe, pear)
  st = sin(theta);
  r = 1 + amp * ((st .^ 4) .* cos(nlobe * phi)) + pear * cos(theta);
  gx = r .* (st .* cos(phi));
  gy = r .* (st .* sin(phi));
  gz = r .* cos(theta);
end
