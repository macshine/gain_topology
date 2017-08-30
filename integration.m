%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear

%% Step 1: Set-Up

%addpath to scripts
addpath('/path/to/file/2014_04_05 BCT/') % https://sites.google.com/site/bctnet/

%load simulated data (rows = regions & columns = time)
data = dlmread('data.tsv','\t',1,1);

%identify variable sizes
[nNodes,1] = size(data);
nParams = 40 * 20; %number of unique sigma and gamma parameter pairs


%%Step 2: Time-Averaged Functional Connectivity

%time-averaged codnnectivity matrix
stat_avg = corr(data');
%collapse across parameter to give stat_grp = nNodes x nNodes x nParams matrix.


%% Step 3: Graph Theoretical Measures

%Modularity
ci = zeros(nNodes,nParams);
q = zeros(nParams,1);
gamma = 1;
tau = 0.1;
nReps = 10;

for p = 1:Params
    %iterate multiple times due to stochasticity of Louvain algorithm
    for x = 1:500
      [ci_temp(:,x),q_temp(x,1)] = community_louvain(stat_grp(:,:,p),gamma,1:1:nNodes,'negative_asym'); 
    end
    
    %estimate a 'consensus' partition (tau and nReps can be altered to change threshold - see https://sites.google.com/site/bctnet/)
    D = agreement(ci_temp);
    ci(:,p) = consensus_und(D,tau,nReps);
    q(p,1) = nanmean(q_temp);
    
end


%Participation index (BA)
BA = zeros(nNodes,nParams);

for p = 1:nParams
  BA(:,p) = participation_coef_sign(stat_grp(:,:,p),ci(:,p));
end


%Communicability (hat tip: Bratislav Misic)
Comm = zeros(nNodes,nNodes,nParams);

for p = 1:nParams
  CIJ = stat_grp(:,:,p);
  N = size(CIJ,1);
  B = sum(CIJ')';
  C = diag(B);
  D = C^(-(1/2));
  E = D * CIJ * D;
  F = expm(E);
  Comm(:,:,p) = F.*~eye(N);
end

log10_Comm = log10(Comm);


