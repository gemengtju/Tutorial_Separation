function [SDR,SIR,SAR,perm]=bss_eval_sources(se,s)

% BSS_EVAL_SOURCES Ordering and measurement of the separation quality for
% estimated source signals in terms of filtered true source, interference
% and artifacts.
% 
% The decomposition allows a time-invariant filter distortion of length
% 512, as described in Section III.B of the reference below.
%
% [SDR,SIR,SAR,perm]=bss_eval_sources(se,s)
%
% Inputs:
% se: nsrc x nsampl matrix containing estimated sources
% s: nsrc x nsampl matrix containing true sources
%
% Outputs:
% SDR: nsrc x 1 vector of Signal to Distortion Ratios
% SIR: nsrc x 1 vector of Source to Interference Ratios
% SAR: nsrc x 1 vector of Sources to Artifacts Ratios
% perm: nsrc x 1 vector containing the best ordering of estimated sources
% in the mean SIR sense (estimated source number perm(j) corresponds to
% true source number j)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2008 Emmanuel Vincent
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
% If you find it useful, please cite the following reference:
% Emmanuel Vincent, R¨¦mi Gribonval, and C¨¦dric F¨¦votte, "Performance
% measurement in blind audio source separation," IEEE Trans. on Audio,
% Speech and Language Processing, 14(4):1462-1469, 2006.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%% Errors %%%
if nargin<2, error('Not enough input arguments.'); end
[nsrc,nsampl]=size(se);
[nsrc2,nsampl2]=size(s);
if nsrc2~=nsrc, error('The number of estimated sources and reference sources must be equal.'); end
if nsampl2~=nsampl, error('The estimated sources and reference sources must have the same duration.'); end

%%% Performance criteria %%%
% Computation of the criteria for all possible pair matches
SDR=zeros(nsrc,nsrc);
SIR=zeros(nsrc,nsrc);
SAR=zeros(nsrc,nsrc);
for jest=1:nsrc,
    for jtrue=1:nsrc,
        [s_true,e_spat,e_interf,e_artif]=bss_decomp_mtifilt(se(jest,:),s,jtrue,512);
        [SDR(jest,jtrue),SIR(jest,jtrue),SAR(jest,jtrue)]=bss_source_crit(s_true,e_spat,e_interf,e_artif);
    end
end
% Selection of the best ordering
perm=perms(1:nsrc);
nperm=size(perm,1);
meanSIR=zeros(nperm,1);
for p=1:nperm,
    meanSIR(p)=mean(SIR((0:nsrc-1)*nsrc+perm(p,:)));
end
[meanSIR,popt]=max(meanSIR);
perm=perm(popt,:).';
SDR=SDR((0:nsrc-1).'*nsrc+perm);
SIR=SIR((0:nsrc-1).'*nsrc+perm);
SAR=SAR((0:nsrc-1).'*nsrc+perm);

return;



function [s_true,e_spat,e_interf,e_artif]=bss_decomp_mtifilt(se,s,j,flen)

% BSS_DECOMP_MTIFILT Decomposition of an estimated source image into four
% components representing respectively the true source image, spatial (or
% filtering) distortion, interference and artifacts, derived from the true
% source images using multichannel time-invariant filters.
%
% [s_true,e_spat,e_interf,e_artif]=bss_decomp_mtifilt(se,s,j,flen)
%
% Inputs:
% se: nchan x nsampl matrix containing the estimated source image (one row per channel)
% s: nsrc x nsampl x nchan matrix containing the true source images
% j: source index corresponding to the estimated source image in s
% flen: length of the multichannel time-invariant filters in samples
%
% Outputs:
% s_true: nchan x nsampl matrix containing the true source image (one row per channel)
% e_spat: nchan x nsampl matrix containing the spatial (or filtering) distortion component
% e_interf: nchan x nsampl matrix containing the interference component
% e_artif: nchan x nsampl matrix containing the artifacts component

%%% Errors %%%
if nargin<4, error('Not enough input arguments.'); end
[nchan2,nsampl2]=size(se);
[nsrc,nsampl,nchan]=size(s);
if nchan2~=nchan, error('The number of channels of the true source images and the estimated source image must be equal.'); end
if nsampl2~=nsampl, error('The duration of the true source images and the estimated source image must be equal.'); end

%%% Decomposition %%%
% True source image
s_true=[reshape(s(j,:,:),nsampl,nchan).',zeros(nchan,flen-1)];
% Spatial (or filtering) distortion
e_spat=project(se,s(j,:,:),flen)-s_true;
% Interference
e_interf=project(se,s,flen)-s_true-e_spat;
% Artifacts
e_artif=[se,zeros(nchan,flen-1)]-s_true-e_spat-e_interf;

return;



function sproj=project(se,s,flen)

% SPROJ Least-squares projection of each channel of se on the subspace
% spanned by delayed versions of the channels of s, with delays between 0
% and flen-1

[nsrc,nsampl,nchan]=size(s);
s=reshape(permute(s,[3 1 2]),nchan*nsrc,nsampl);

%%% Computing coefficients of least squares problem via FFT %%%
% Zero padding and FFT of input data
s=[s,zeros(nchan*nsrc,flen-1)];
se=[se,zeros(nchan,flen-1)];
fftlen=2^nextpow2(nsampl+flen-1);
sf=fft(s,fftlen,2);
sef=fft(se,fftlen,2);
% Inner products between delayed versions of s
G=zeros(nchan*nsrc*flen);
for k1=0:nchan*nsrc-1,
    for k2=0:k1,
        ssf=sf(k1+1,:).*conj(sf(k2+1,:));
        ssf=real(ifft(ssf));
        ss=toeplitz(ssf([1 fftlen:-1:fftlen-flen+2]),ssf(1:flen));
        G(k1*flen+1:k1*flen+flen,k2*flen+1:k2*flen+flen)=ss;
        G(k2*flen+1:k2*flen+flen,k1*flen+1:k1*flen+flen)=ss.';
    end
end
% Inner products between se and delayed versions of s
D=zeros(nchan*nsrc*flen,nchan);
for k=0:nchan*nsrc-1,
    for i=1:nchan,
        ssef=sf(k+1,:).*conj(sef(i,:));
        ssef=real(ifft(ssef,[],2));
        D(k*flen+1:k*flen+flen,i)=ssef(:,[1 fftlen:-1:fftlen-flen+2]).';
    end
end

%%% Computing projection %%%
% Distortion filters
C=G\D;
C=reshape(C,flen,nchan*nsrc,nchan);
% Filtering
sproj=zeros(nchan,nsampl+flen-1);
for k=1:nchan*nsrc,
    for i=1:nchan,
        sproj(i,:)=sproj(i,:)+fftfilt(C(:,k,i).',s(k,:));
    end
end

return;



function [SDR,SIR,SAR]=bss_source_crit(s_true,e_spat,e_interf,e_artif)

% BSS_SOURCE_CRIT Measurement of the separation quality for a given source
% in terms of filtered true source, interference and artifacts.
%
% [SDR,SIR,SAR]=bss_source_crit(s_true,e_spat,e_interf,e_artif)
%
% Inputs:
% s_true: nchan x nsampl matrix containing the true source image (one row per channel)
% e_spat: nchan x nsampl matrix containing the spatial (or filtering) distortion component
% e_interf: nchan x nsampl matrix containing the interference component
% e_artif: nchan x nsampl matrix containing the artifacts component
%
% Outputs:
% SDR: Signal to Distortion Ratio
% SIR: Source to Interference Ratio
% SAR: Sources to Artifacts Ratio

%%% Errors %%%
if nargin<4, error('Not enough input arguments.'); end
[nchant,nsamplt]=size(s_true);
[nchans,nsampls]=size(e_spat);
[nchani,nsampli]=size(e_interf);
[nchana,nsampla]=size(e_artif);
if ~((nchant==nchans)&&(nchant==nchani)&&(nchant==nchana)), error('All the components must have the same number of channels.'); end
if ~((nsamplt==nsampls)&&(nsamplt==nsampli)&&(nsamplt==nsampla)), error('All the components must have the same duration.'); end

%%% Energy ratios %%%
s_filt=s_true+e_spat;
% SDR
SDR=10*log10(sum(sum(s_filt.^2))/sum(sum((e_interf+e_artif).^2)));
% SIR
SIR=10*log10(sum(sum(s_filt.^2))/sum(sum(e_interf.^2)));
% SAR
SAR=10*log10(sum(sum((s_filt+e_interf).^2))/sum(sum(e_artif.^2)));

return;
