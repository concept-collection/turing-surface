% Lobes around the equator, plus a pear-shaped offset along z.
% `nlobe` should be a whole number, or the surface does not close at the
% phi = 0 seam.

function [gx, gy, gz] = shape(theta, phi, amp, nlobe, pear)
  st = sin(theta);
  r = 1 + amp * ((st .^ 4) .* cos(nlobe * phi)) + pear * cos(theta);
  gx = r .* (st .* cos(phi));
  gy = r .* (st .* sin(phi));
  gz = r .* cos(theta);
end
