function eval_sdr(dataset, eval_mix, eval_pesq, model_name, fmix, fclean)
addpath('composite_pesq');
%% WSJ0_2mix_extr
%mixed_wav_dir = ['/export/home/clx214/data/wsj0_2mix_extr/wav8k/max/' dataset '/' fmix '/'];
%spk1_dir = ['/export/home/clx214/data/wsj0_2mix_extr/wav8k/max/' dataset '/' fclean '/'];

%% WSJ0_2mix
mixed_wav_dir = ['/export/home/clx214/gm/ntu_project/SpEx_SincNetAuxCNNEncoder_MultiOriEncoder_share_min_2spk/data/wsj0_2mix_min_mix_6k/'];
spk1_dir = ['/export/home/clx214/gm/ntu_project/SpEx_SincNetAuxCNNEncoder_MultiOriEncoder_share_min_2spk/data/wsj0_2mix_min_clean_6k/'];

%% WHAM
%mixed_wav_dir = ['/export/home/clx214/gm/ntu_project/SpEx+_WHAM/data/WHAM_mix_6k/'];
%spk1_dir = ['/export/home/clx214/gm/ntu_project/SpEx+_WHAM/data/WHAM_clean_6k/'];

% WHAMR reverb
%mixed_wav_dir = ['/export/home/clx214/gm/ntu_project/SpEx+_WHAMR/data/WHAMR_mix_6k/'];
%spk1_dir = ['/export/home/clx214/gm/ntu_project/SpEx+_WHAMR/data/WHAMR_clean_6k/'];

% WHAMR noise
%mixed_wav_dir = ['/export/home/clx214/gm/ntu_project/SpEx+_WHAMR2/data/WHAMR_noise_mix_6k/'];
%spk1_dir = ['/export/home/clx214/gm/ntu_project/SpEx+_WHAMR/data/WHAMR_clean_6k/']; % all WHAMR clean data is same

% WHAMR noise + reverb
%mixed_wav_dir = ['/export/home/clx214/gm/ntu_project/SpEx+_WHAMR_noise/data/WHAMR_noise_reverb_mix_6k/'];
%spk1_dir = ['/export/home/clx214/gm/ntu_project/SpEx+_WHAMR/data/WHAMR_clean_6k/']; % all WHAMR clean data is same


sprintf('start, %s\n', model_name)
%rec_wav_dir = ['../data/rec/' dataset '/' model_name  '/'];
rec_wav_dir = ['/export/home/clx214/gm/ntu_project/SpEx_SincNetAuxCNNEncoder_MultiOriEncoder_share_min_2spk/rec_aux60/spk1/'];
lists = dir(rec_wav_dir);
len = length(lists) - 2;
SDR = zeros(len, 1);
SIR = zeros(len, 1);
SAR = zeros(len, 1);
SDR_Mix = zeros(len, 1);
SIR_Mix = zeros(len, 1);
SAR_Mix = zeros(len, 1);
PESQ = zeros(len, 1);
PESQ_Mix = zeros(len, 1);
SISDR = zeros(len, 1);
SISDR_Mix = zeros(len, 1);

target_durs=textscan(fopen('target_ref_dur.txt'), '%s %f');

for i = 3:len+2
    name = lists(i).name;
    part_name = name(1:end-4);
    [rec_wav, Fs] = audioread([rec_wav_dir part_name '.wav']);
    ori_wav = audioread([spk1_dir part_name '.wav']);
    mix_wav = audioread([mixed_wav_dir part_name '.wav']);

    % get ground truth length
    utt_tokens = strsplit(part_name, '_');
    idx = find(strcmp(target_durs{1}, utt_tokens{1}));
    dur = int32(target_durs{2}(idx)*Fs);
    
    min_len = min(size(ori_wav, 1), size(rec_wav, 1));
    min_len = int32(min(min_len, dur));
    
    rec_wav = rec_wav(1:min_len);
    ori_wav = ori_wav(1:min_len);
    mix_wav = mix_wav(1:min_len);

    [SDR(i-2),SIR(i-2),SAR(i-2),perm]=bss_eval_sources(rec_wav',ori_wav');
    SISDR(i-2)=cal_SISDR(ori_wav', rec_wav');

    if eval_pesq
        fprintf('PESQINDEX: %d\n', i);
        fprintf('PESQINDEX: %s,%s\n', [spk1_dir part_name '.wav'], [rec_wav_dir part_name '.wav']);
        PESQ(i-2)=pesq(8000, [spk1_dir part_name '.wav'], [rec_wav_dir part_name '.wav']);
    end

    if eval_mix
        [SDR_Mix(i-2),SIR_Mix(i-2),SAR_Mix(i-2),perm]=bss_eval_sources(mix_wav',ori_wav');
        SISDR_Mix(i-2)=cal_SISDR(ori_wav', mix_wav');

        if eval_pesq
            fprintf('PESQINDEX_MIX: %d\n', i);
            PESQ_Mix(i-2)=pesq(8000, [spk1_dir part_name '.wav'], [mixed_wav_dir part_name '.wav']);
        end
    end

    if mod(i, 200) == 0
        fprintf('the number of sample is evaluated: %d\n', i);
        fprintf('%s, %s, target:%d, org:%d, rec:%d\n', part_name, utt_tokens{1}, dur, size(ori_wav,1), size(rec_wav,1));
    end
end
mean_SDR = mean(SDR);
mean_SIR = mean(SIR);
mean_SAR = mean(SAR);
mean_PESQ = mean(PESQ);
mean_SISDR = mean(SISDR);
fprintf('The mean SDR, SIR, SAR, PESQ, SISDR are: %f ,\t %f ,\t %f ,\t %f, \t %f \n', mean_SDR, mean_SIR, mean_SAR, mean_PESQ, mean_SISDR);
if eval_mix
    mean_SDR_Mix = mean(SDR_Mix);
    mean_SIR_Mix = mean(SIR_Mix);
    mean_SAR_Mix = mean(SAR_Mix);
    mean_PESQ_Mix = mean(PESQ_Mix);
    mean_SISDR_Mix = mean(SISDR_Mix);
    fprintf('The mean SDR, SIR, SAR, PESQ, SISDR of mixture are: %f ,\t %f ,\t %f ,\t %f, \t %f \n', mean_SDR_Mix, mean_SIR_Mix, mean_SAR_Mix, mean_PESQ_Mix, mean_SISDR_Mix);
end

% Calculte different gender case
if dataset == 'cv'
    [spk, gender] = textread('spk2gender_cv', '%s%d');
else
    [spk, gender] = textread('spk2gender', '%s%d');
end
cmm = 1;
cmf = 1;
cff = 1;
csame = 1;
for i = 1:size(SDR, 1)
    mix_name = lists(i+2).name;
    spk1 = mix_name(1:3);
    tmp = regexp(mix_name, '_');
    spk2 = mix_name(tmp(2)+1:tmp(2)+3);
    for j = 1:length(spk)
        if spk1 == spk{j}
            break
        end
    end
    for k = 1:length(spk)
        if spk2 == spk{k}
            break
        end
    end
    
    if gender(k) == 0 & gender(j) == 0
        SDR_FF(cff) = SDR(i); 
        SIR_FF(cff) = SIR(i);
        SAR_FF(cff) = SAR(i);
        PESQ_FF(cff) = PESQ(i); 

        SDR_Same(csame) = SDR(i); 
        SIR_Same(csame) = SIR(i); 
        SAR_Same(csame) = SAR(i); 
        PESQ_Same(csame) = PESQ(i); 
    
        if eval_mix
            SDR_FF_Mix(cff) = SDR_Mix(i); 
            SIR_FF_Mix(cff) = SIR_Mix(i); 
            SAR_FF_Mix(cff) = SAR_Mix(i); 
            PESQ_FF_Mix(cff) = PESQ_Mix(i); 

            SDR_Same_Mix(csame) = SDR_Mix(i); 
            SIR_Same_Mix(csame) = SIR_Mix(i); 
            SAR_Same_Mix(csame) = SAR_Mix(i); 
            PESQ_Same_Mix(csame) = PESQ_Mix(i);
        end

        lists_FF{cff} = lists(i).name;
        cff = cff +1;
        csame = csame +1;
    
    elseif gender(k) == 1 & gender(j) == 1
        SDR_MM(cmm)= SDR(i); 
        SIR_MM(cmm)= SIR(i);
        SAR_MM(cmm)= SAR(i);
        PESQ_MM(cmm) = PESQ(i); 

        SDR_Same(csame) = SDR(i); 
        SIR_Same(csame) = SIR(i); 
        SAR_Same(csame) = SAR(i); 
        PESQ_Same(csame) = PESQ(i); 

        if eval_mix
            SDR_MM_Mix(cmm) = SDR_Mix(i); 
            SIR_MM_Mix(cmm) = SIR_Mix(i); 
            SAR_MM_Mix(cmm) = SAR_Mix(i); 
            PESQ_MM_Mix(cmm) = PESQ_Mix(i); 

            SDR_Same_Mix(csame) = SDR_Mix(i); 
            SIR_Same_Mix(csame) = SIR_Mix(i); 
            SAR_Same_Mix(csame) = SAR_Mix(i); 
            PESQ_Same_Mix(csame) = PESQ_Mix(i);
        end

        lists_MM{cmm} = lists(i).name;
        cmm = cmm + 1;
        csame = csame +1;
    else
        SDR_MF(cmf) = SDR(i);
        SIR_MF(cmf) = SIR(i);
        SAR_MF(cmf) = SAR(i);
        PESQ_MF(cmf) = PESQ(i); 

        if eval_mix
            SDR_MF_Mix(cmf) = SDR_Mix(i); 
            SIR_MF_Mix(cmf) = SIR_Mix(i); 
            SAR_MF_Mix(cmf) = SAR_Mix(i); 
            PESQ_MF_Mix(cmf) = PESQ_Mix(i); 
        end

        lists_MF{cmf} = lists(i).name;
        cmf = cmf + 1;
    end
end
mean_SDR_MF = mean(SDR_MF);
mean_SDR_FF = mean(SDR_FF);
mean_SDR_MM = mean(SDR_MM);
mean_SDR_Same = mean(SDR_Same);

mean_SIR_MF = mean(SIR_MF);
mean_SIR_FF = mean(SIR_FF);
mean_SIR_MM = mean(SIR_MM);
mean_SIR_Same = mean(SIR_Same);

mean_SAR_MF = mean(SAR_MF);
mean_SAR_FF = mean(SAR_FF);
mean_SAR_MM = mean(SAR_MM);
mean_SAR_Same = mean(SAR_Same);

mean_PESQ_MF = mean(PESQ_MF);
mean_PESQ_FF = mean(PESQ_FF);
mean_PESQ_MM = mean(PESQ_MM);
mean_PESQ_Same = mean(PESQ_Same);

if eval_mix
    mean_SDR_MF_Mix = mean(SDR_MF_Mix);
    mean_SDR_FF_Mix = mean(SDR_FF_Mix);
    mean_SDR_MM_Mix = mean(SDR_MM_Mix);
    mean_SDR_Same_Mix = mean(SDR_Same_Mix);

    mean_SIR_MF_Mix = mean(SIR_MF_Mix);
    mean_SIR_FF_Mix = mean(SIR_FF_Mix);
    mean_SIR_MM_Mix = mean(SIR_MM_Mix);
    mean_SIR_Same_Mix = mean(SIR_Same_Mix);

    mean_SAR_MF_Mix = mean(SAR_MF_Mix);
    mean_SAR_FF_Mix = mean(SAR_FF_Mix);
    mean_SAR_MM_Mix = mean(SAR_MM_Mix);
    mean_SAR_Same_Mix = mean(SAR_Same_Mix);

    mean_PESQ_MF_Mix = mean(PESQ_MF_Mix);
    mean_PESQ_FF_Mix = mean(PESQ_FF_Mix);
    mean_PESQ_MM_Mix = mean(PESQ_MM_Mix);
    mean_PESQ_Same_Mix = mean(PESQ_Same_Mix);
end

fprintf('The mean SDR, SIR, SAR, PESQ for Male & Female are : %f ,\t %f ,\t %f ,\t %f \n', mean_SDR_MF, mean_SIR_MF, mean_SAR_MF, mean_PESQ_MF);
fprintf('The mean SDR, SIR, SAR, PEESQ for Female & Female are : %f ,\t %f ,\t %f ,\t %f \n', mean_SDR_FF, mean_SIR_FF, mean_SAR_FF, mean_PESQ_FF);
fprintf('The mean SDR, SIR, SAR, PESQ for Male & Male are : %f ,\t %f ,\t %f ,\t %f \n', mean_SDR_MM, mean_SIR_MM, mean_SAR_MM, mean_PESQ_MM);
fprintf('The mean SDR, SIR, SAR, PESQ for same gender are : %f ,\t %f ,\t %f ,\t %f \n', mean_SDR_Same, mean_SIR_Same, mean_SAR_Same, mean_PESQ_Same);

if eval_mix
    fprintf('The mean SDR, SIR, SAR, PESQ for Male & Female mixture are : %f ,\t %f ,\t %f ,\t %f \n', mean_SDR_MF_Mix, mean_SIR_MF_Mix, mean_SAR_MF_Mix, mean_PESQ_MF_Mix);
    fprintf('The mean SDR, SIR, SAR, PEESQ for Female & Female mixture are : %f ,\t %f ,\t %f ,\t %f \n', mean_SDR_FF_Mix, mean_SIR_FF_Mix, mean_SAR_FF_Mix, mean_PESQ_FF_Mix);
    fprintf('The mean SDR, SIR, SAR, PESQ for Male & Male mixture are : %f ,\t %f ,\t %f ,\t %f \n', mean_SDR_MM_Mix, mean_SIR_MM_Mix, mean_SAR_MM_Mix, mean_PESQ_MM_Mix);
    fprintf('The mean SDR, SIR, SAR, PESQ for same gender mixture are : %f ,\t %f ,\t %f ,\t %f \n', mean_SDR_Same_Mix, mean_SIR_Same_Mix, mean_SAR_Same_Mix, mean_PESQ_Same_Mix);
end

if eval_mix
    save(['sdr_' model_name '_' dataset '.mat'], 'SDR', 'SIR', 'SAR', 'PESQ', 'SDR_Mix', 'SIR_Mix', 'SAR_Mix', 'PESQ_Mix', 'SISDR', 'SISDR_Mix', 'lists', 'mean_SISDR', 'mean_SISDR_Mix', 'mean_SDR', 'mean_SDR_MF', 'mean_SDR_FF', 'mean_SDR_MM', 'mean_SDR_Same','mean_SIR', 'mean_SIR_MF', 'mean_SIR_FF', 'mean_SIR_MM', 'mean_SIR_Same','mean_SAR', 'mean_SAR_MF', 'mean_SAR_FF', 'mean_SAR_MM', 'mean_SAR_Same', 'mean_PESQ', 'mean_PESQ_MF', 'mean_PESQ_FF', 'mean_PESQ_MM', 'mean_PESQ_Same', 'mean_SDR_Mix', 'mean_SDR_MF_Mix', 'mean_SDR_FF_Mix', 'mean_SDR_MM_Mix', 'mean_SDR_Same_Mix', 'mean_SIR_Mix', 'mean_SIR_MF_Mix', 'mean_SIR_FF_Mix', 'mean_SIR_MM_Mix', 'mean_SIR_Same_Mix', 'mean_SAR_Mix', 'mean_SAR_MF_Mix', 'mean_SAR_FF_Mix', 'mean_SAR_MM_Mix', 'mean_SAR_Same_Mix', 'mean_PESQ_Mix', 'mean_PESQ_MF_Mix', 'mean_PESQ_FF_Mix', 'mean_PESQ_MM_Mix', 'mean_PESQ_Same_Mix');
else
    save(['sdr_' model_name '_' dataset '.mat'], 'SDR', 'SIR', 'SAR', 'PESQ', 'SISDR', 'SDR_MF', 'SDR_FF', 'SDR_MM', 'SDR_Same', 'lists', 'mean_SISDR', 'mean_SDR', 'mean_SDR_MF', 'mean_SDR_FF', 'mean_SDR_MM', 'mean_SDR_Same','mean_SIR', 'mean_SIR_MF', 'mean_SIR_FF', 'mean_SIR_MM', 'mean_SIR_Same','mean_SAR', 'mean_SAR_MF', 'mean_SAR_FF', 'mean_SAR_MM', 'mean_SAR_Same', 'mean_PESQ', 'mean_PESQ_MF', 'mean_PESQ_FF', 'mean_PESQ_MM', 'mean_PESQ_Same');
end

end

function SISDR = cal_SISDR(clean_sig, rec_sig)
clean_sig = clean_sig-mean(clean_sig);
rec_sig = rec_sig-mean(rec_sig);
s_target = dot(rec_sig, clean_sig)*clean_sig/dot(clean_sig, clean_sig);
e_noise = rec_sig - s_target;
SISDR = 10*log10(dot(s_target, s_target)/dot(e_noise, e_noise));

end
