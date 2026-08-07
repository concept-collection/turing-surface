% Smooth random function on the unit sphere — chebfun's randnfunsphere,
% evaluated at the given (theta, phi) instead of returned as a spherefun.
%
%   F = randnfunsphere(LAMBDA, THETA, PHI) is a combination of all spherical
%   harmonics up to degree floor(2*pi/LAMBDA) with independent N(0,1)
%   coefficients, normalized so the variance is 1 at each point.
%
%   randnfunsphere(LAMBDA, THETA, PHI, 'monochromatic') uses only the
%   harmonics of that one degree, so every component has the same wave
%   number — chebfun's 'monochrome' option.
%
% Seed the draw with rng(...) before calling. This project has no chebfun
% objects: what would be a spherefun there is returned here as values on the
% grid the caller passes in.

function f = randnfunsphere(lambda, theta, phi, type)
  if ( nargin < 4 )
    type = 'white';
  end
  % The unit sphere has circumference 2*pi, matching randnfun's deg = L/lambda.
  deg = floor(2*pi/lambda);
  if ( strncmpi(type, 'm', 1) )
    c = randn(2*deg+1, 1);
    c = sqrt(4*pi/numel(c)) * c;      % normalize so the variance is 1
    f = sphHarmSumFixedDeg(theta, phi, deg, c);
  else
    c = randn((deg+1)^2, 1);
    c = sqrt(4*pi/numel(c)) * c;      % normalize so the variance is 1
    f = sphHarmSum(theta, phi, deg, c);
  end
end

% All spherical harmonics up to degree deg, with coefficients ordered by
% degree and order (0, -1,0,1, -2,-1,0,1,2, ...). Order +m carries
% cos(m*phi), order -m carries sin(m*phi).
function f = sphHarmSum(theta, phi, deg, c)
  f = 1/sqrt(4*pi) * c(1) * ones(size(theta));
  k = 1;                              % coefficients consumed so far
  for l = 1:deg
    cl = c(k+1 : k+2*l+1);            % this degree's orders, -l..l
    k = k + 2*l + 1;
    f = f + sphHarmSumFixedDeg(theta, phi, l, cl);
  end
end

% All spherical harmonics of the single degree l.
function f = sphHarmSumFixedDeg(theta, phi, l, c)
  m = (0:l).';
  a = (-1).^m ./ sqrt((1 + double(m==0)) * pi);
  costh = cos(theta(:)).';            % legendre wants cos(theta), in a row
  G = legendre(l, costh, 'norm');     % (l+1) x npts
  f = 0 * theta;
  for mm = 0:l
    f = f + a(mm+1) * c(l+1+mm) * (G(mm+1,:).' .* cos(mm*phi));
    if mm > 0
      f = f + a(mm+1) * c(l+1-mm) * (G(mm+1,:).' .* sin(mm*phi));
    end
  end
end
