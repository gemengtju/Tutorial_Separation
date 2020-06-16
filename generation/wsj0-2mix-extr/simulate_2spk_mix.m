function simulate_2spk_mix(data_type, wsj0root, output_dir, fs8k, min_max)
% Simulate 2-speaker mixture data for speaker extraction.
% Call:
%     simulate_2spk_mix(data_type, wsj0root, output_dir, fs8k, min_max)
%     e.g., simulate_2spk_mix('tt', '/media/clx214/data/wsj/', '/media/clx214/data/wsj0_2mix_extr_tmp/wav8k', 8000, 'max')
% Paras:
%     data_type: data set to generate, (tr|cv|tt), e.g., 'tt'
%     wsj0root: YOUR_PATH/, the folder containing converted wsj0/, e.g., '/media/clx214/data/wsj/'
%     output_dir: the folder to save simulated data for extraction, e.g., '/media/clx214/data/wsj0_2mix_extr_tmp/wav8k'
%     fs8k: sampling rate of the simulated data, e.g., 8000
%     min_max: get the mininium or maximum wav length, when simulating mixture data, e.g, 'max'
%
% The code is based on "create_wav_2speakers_extr.m" from "http://www.merl.com/demos/deep-clustering"
%
% 1. Assume that WSJ0's wv1 sphere files is converted to wav files. The folder
%    structure and file name are kept same under wsj0/, e.g.,
%    ORG_PATH/wsj0/si_tr_s/01t/01to030v.wv1 is converted to wav and
%    stored in YOUR_PATH/wsj0/si_tr_s/01t/01to030v.wv1.
%    Relevant data ('si_tr_s', 'si_dt_05' and 'si_et_05') are under YOUR_PATH/wsj0/
% 2. Put 'voicebox' toolbox in current folder. (http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html)
% 3. Set your 'YOUR_PATH' and 'OUTPUT_PATH' properly, then run this script in Matlab.
%    (The max lenght of the wav will be kept when generate the mixture. The sampling rate will be 8kHz.)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Copyright (C) 2016 Mitsubishi Electric Research Labs
%                          (Jonathan Le Roux, John R. Hershey, Zhuo Chen)
%   Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Copyright 2018 Chenglin Xu, Nanyang Technological University, Singapore
%
%Licensed under the Apache License, Version 2.0 (the "License");
%you may not use this file except in compliance with the License.
%You may obtain a copy of the License at
%
%    http://www.apache.org/licenses/LICENSE-2.0
%
%Unless required by applicable law or agreed to in writing, software
%distributed under the License is distributed on an "AS IS" BASIS,
%WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
%See the License for the specific language governing permissions and
%limitations under the License.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~exist([output_dir '/' min_max '/' data_type],'dir')
    mkdir([output_dir '/' min_max '/' data_type]);
end
mkdir([output_dir  '/' min_max '/' data_type '/s1/']);
mkdir([output_dir  '/' min_max '/' data_type '/aux/']);
mkdir([output_dir  '/' min_max '/' data_type '/mix/']);

TaskFile = ['mix_2_spk_' data_type '_extr.txt'];
fid = fopen(TaskFile,'r');
C = textscan(fid,'%s %f %s %f %s');
num_files = length(C{1});

fprintf(1,'Start to generate data for %s\n', [min_max '_' data_type]);
for i = 1:num_files
    [inwav1_dir,invwav1_name,inwav1_ext] = fileparts(C{1}{i});
    [inwav2_dir,invwav2_name,inwav2_ext] = fileparts(C{3}{i});
    [inwav_aux_dir,invwav_aux_name,inwav_aux_ext] = fileparts(C{5}{i});
    
    inwav1_snr = C{2}(i);
    inwav2_snr = C{4}(i);
    mix_name = [invwav1_name,'_',num2str(inwav1_snr),'_',invwav2_name,'_',num2str(inwav2_snr),'_',invwav_aux_name];
    
    % get input wavs
    [s1, fs] = audioread([wsj0root C{1}{i}]);
    s2       = audioread([wsj0root C{3}{i}]);
    s_aux    = audioread([wsj0root C{5}{i}]);
    
    % resample, normalize to 8 kHz file
    s1_8k = resample(s1,fs8k,fs);
    [s1_8k,lev1] = activlev(s1_8k,fs8k,'n'); % y_norm = y /sqrt(lev);
    s2_8k = resample(s2,fs8k,fs);
    [s2_8k,lev2] = activlev(s2_8k,fs8k,'n');
    s_aux_8k = resample(s_aux,fs8k,fs);
    [s_aux_8k,lev_aux] = activlev(s_aux_8k,fs8k,'n');
    
    weight_1 = 10^(inwav1_snr/20);
    weight_2 = 10^(inwav2_snr/20);
    
    s1_8k = weight_1 * s1_8k;
    s2_8k = weight_2 * s2_8k;
    
    switch min_max
        case 'max'
            mix_8k_length = max(length(s1_8k),length(s2_8k));
            s1_8k = cat(1,s1_8k,zeros(mix_8k_length - length(s1_8k),1));
            s2_8k = cat(1,s2_8k,zeros(mix_8k_length - length(s2_8k),1));
        case 'min'
            mix_8k_length = min(length(s1_8k),length(s2_8k));
            s1_8k = s1_8k(1:mix_8k_length);
            s2_8k = s2_8k(1:mix_8k_length);
    end
    mix_8k = s1_8k + s2_8k;
    
    max_amp_8k = max(cat(1,abs(mix_8k(:)),abs(s1_8k(:)),abs(s2_8k(:)),abs(s_aux_8k(:))));
    mix_scaling_8k = 1/max_amp_8k*0.9;
    s1_8k = mix_scaling_8k * s1_8k;
    mix_8k = mix_scaling_8k * mix_8k;
    s_aux_8k = mix_scaling_8k * s_aux_8k;
    
    audiowrite([output_dir '/' min_max '/' data_type '/s1/' mix_name '.wav'],s1_8k,fs8k);
    audiowrite([output_dir '/' min_max '/' data_type '/aux/' mix_name '.wav'],s_aux_8k,fs8k);
    audiowrite([output_dir '/' min_max '/' data_type '/mix/' mix_name '.wav'],mix_8k,fs8k);
end
fclose(fid);
fprintf(1,'End of generating data for %s\n', [min_max '_' data_type]);
end
