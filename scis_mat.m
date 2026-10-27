function [P,cov,N] = scis_mat(a,b,m,C,Nmax,Nmin,Nint,covmax)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Matrix-based SCIS (Sequential Conditioned Importance Sampling)        %
%                                                                         %
%   Copyright (C) 2009  Young Joo Lee, and Junho Song                     %
%                                                                         %
%   This program is free software: you can redistribute it and/or modify  %
%   it under the terms of the GNU General Public License as published by  %
%   the Free Software Foundation, either version 3 of the License, or     % 
%   (at your option) any later version.                                   %
%                                                                         %
%   This program is distributed in the hope that it will be useful,       %
%   but WITHOUT ANY WARRANTY; without even the implied warranty of        %
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         %
%   GNU General Public License for more details.                          %
%                                                                         %
%   You should have received a copy of the GNU General Public License     % 
%   along with this program.  If not, see <http://www.gnu.org/licenses/>. %
%                                                                         % 
%   This programe is originated from an SCIS code developed by Junho Song %
%   (Assistance professor, University of Illinois, Urbana-Champaign)      %
%   and it was modified by Young Joo Lee (Graduate student, University of %
%   Illinois, Urbana-Champaign) for more efficient calculations through   %
%   vectorizations. The original SCIS code was a part of the FERUM        % 
%   (Finite Element Reliability Using Matlab) developed at U.C.Berkeley.  %                                                  %                                                                         %
%                                                                         %
%   For more information, visit:                                          %
%   (1) Junho Song's website: https://netfiles.uiuc.edu/junho/www/        %
%   (2) FERUM website: http://www.ce.berkeley.edu/FERUM/                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   [P,cov,N] = scis_mat(a,b,m,C,Nmax,Nmin,Nint,covmax)
%
%   Evaluates the probability of multinormal variable
%   in the rectangular domain by SCIS
%
%    Output: - P       = Estimate of probability of multinormal variable N(m,C)
%                        in the n-dimensional rectangular domain [a_k,b_k]
%            - cov     = Coefficient of variation of estimate P
%            - N       = Total number of SCIS trials
%

%    Input:  - a       = Lower bounds
%            - b       = Upper bounds
%            - m       = Mean vector
%            - C       = Covariance matrix
%            - Nmax    = Maximum trial number of SCIS
%            - Nmin    = Minimum trial number of SCIS
%            - Nint    = Interval of increasing trial number of SCIS
%            - covmax  = Maximum coefficent of variation
%   "Multinormal Probability by Sequential Conditioned Importance Sampling"
%      by R.Ambartzumian, A.Der Kiureghian, V.Ohanian, and H.Sukiasian
%      Probabilistic Engineering Mechanics, Vol.13, No.4, pp.299-308, 1998.

% n : Number of random variables
n = length(a);

% Preallocating
% for optimized dll(MEX) file(-i option), all arrays should be preallocated
% and should not grow dynamically.

Dk = zeros(n,n);
d = zeros(n,n);

% Preprocessings
% d : Matrix containing parts of inverse of covariance matrix
%     (Note) d(k,1:k) - k-th row of inverse matrix of C(1:k,1:k)
for ii = 1:n
    Dk(1:ii,1:ii) = inv(C(1:ii,1:ii));
    for j = 1:ii,
        d(ii,j) = Dk(ii,j);
    end
end

% ICIS algorithm
% y    : Product of probabilities
% P    : Estimate of probability (sum(y)/N)
% em   : Mean of conditional multinormal distribution
% dd   : Standard deviation of conditional multinormal distribution
% x    : Sampled value for each random variable

%------- Initializations -------%
cov = inf;
P = 0;

%%%%%%%%%%%%%%%%%%%%%%%%% 1st step (until Nmin) %%%%%%%%%%%%%%%%%%%%%%%%%%%
NN = Nmin;
NN_sum = Nmin;
y = [];
y_temp = ones(1,NN);
x = [];

for k = 1:n
    em = m(k)*ones(1,NN);
    dd = 1/sqrt(d(k,k))*ones(1,NN);
    if (d(k,k) <= 0)    % To avoid a numerical problem
        dd = eps*ones(1,NN);
    end
    if (k~=1)
        for j = 1:(k-1)
            sm = (d(k,j)*(x(j,:) - m(j)*ones(1,NN)))/d(k,k);
            em = em - sm;
        end
    end
    za = normcdf(a(k)*ones(1,NN),em,dd);
    zb = normcdf(b(k)*ones(1,NN),em,dd);
    y_temp = y_temp.*(zb-za);

    zz = norminv((0.9999*zb-0.9999*za).*rand(1,NN)+0.9999*za);
    x_temp = dd.*zz + em;
    x_temp(find(x_temp==inf)) = 1/eps;   % To avoid a numerical problem
    x_temp(find(x_temp==-inf)) = -1/eps; % To avoid a numerical problem
    x = [x; x_temp];
end
y = [y y_temp];

P = sum(y) / NN;
if (P ~=0)
    cov = norm(y-P*ones(1,NN))/NN/P;
end

%%%%%%%%%%%%%%%%%%%%%%%%% 2nd step (until Nint) %%%%%%%%%%%%%%%%%%%%%%%%%%%
if (cov > covmax)
    NN = Nint - Nmin;
    NN_sum = Nint;
    y_temp = ones(1,NN);
    x = [];

    for k = 1:n
        em = m(k)*ones(1,NN);
        dd = 1/sqrt(d(k,k))*ones(1,NN);
        if (d(k,k) <= 0)    % To avoid a numerical problem
            dd = eps*ones(1,NN);
        end
        if (k~=1)
            for j = 1:(k-1),
                sm = (d(k,j)*(x(j,:) - m(j)*ones(1,NN)))/d(k,k);
                em = em - sm;
            end
        end
        za = normcdf(a(k)*ones(1,NN),em,dd);
        zb = normcdf(b(k)*ones(1,NN),em,dd);
        y_temp = y_temp.*(zb-za);

        zz = norminv((0.9999*zb-0.9999*za).*rand(1,NN)+0.9999*za);
        x_temp = dd.*zz + em;
        x_temp(find(x_temp==inf)) = 1/eps;   % To avoid a numerical problem
        x_temp(find(x_temp==-inf)) = -1/eps; % To avoid a numerical problem
        x = [x; x_temp];
    end
    y = [y y_temp];

    P = sum(y) / NN_sum;
    if (P ~=0)
        cov = norm(y-P*ones(1,NN_sum))/NN_sum/P;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%% 3rd step (until Nmax) %%%%%%%%%%%%%%%%%%%%%%%%%%%
if (cov > covmax)
    while (NN_sum < Nmax && cov > covmax)
        NN = Nint;
        NN_sum = NN_sum + Nint;
        y_temp = ones(1,NN);
        x = [];

        for k = 1:n
            em = m(k)*ones(1,NN);
            dd = 1/sqrt(d(k,k))*ones(1,NN);
            if (d(k,k) <= 0)    % To avoid a numerical problem
                dd = eps*ones(1,NN);
            end
            if (k~=1)
                for j = 1:(k-1),
                    sm = (d(k,j)*(x(j,:) - m(j)*ones(1,NN)))/d(k,k);
                    em = em - sm;
                end
            end
            za = normcdf(a(k)*ones(1,NN),em,dd);
            zb = normcdf(b(k)*ones(1,NN),em,dd);
            y_temp = y_temp.*(zb-za);

            zz = norminv((0.9999*zb-0.9999*za).*rand(1,NN)+0.9999*za);
            x_temp = dd.*zz + em;
            x_temp(find(x_temp==inf)) = 1/eps;   % To avoid a numerical problem
            x_temp(find(x_temp==-inf)) = -1/eps; % To avoid a numerical problem
            x = [x; x_temp];
        end
        y = [y y_temp];

        P = sum(y) / NN_sum;
        if (P ~=0)
            cov = norm(y-P*ones(1,NN_sum))/NN_sum/P;
        end
    end
end

P = P;
N = NN_sum;
cov = cov;