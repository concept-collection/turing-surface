% A dumbbell: a radial profile that pinches at the equator, stretched along z.
%
% The radius is 1 - waist*sin(theta)^2, so the poles keep radius 1 and the
% equator narrows to 1 - waist. Multiplying it into the unit-sphere coordinates
% raises the degree to 3, which is still far below any lmax the app offers, so
% the band-limited surface is again the one written here.
%
% Two regions of positive curvature joined by a saddle — the first geometry in
% this list where the Laplace-Beltrami operator differs from the round one in a
% way that should visibly change where a pattern wants to put its spots.

function [gx, gy, gz] = shape(theta, phi, waist, stretch)
  st = sin(theta);
  r = 1 - waist * (st .^ 2);
  gx = r .* (st .* cos(phi));
  gy = r .* (st .* sin(phi));
  gz = (1 + stretch) * (r .* cos(theta));
end
