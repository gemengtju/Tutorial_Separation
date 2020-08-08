# Speech Separation and Extraction via Deep Learning

This repo summarizes the tutorials, datasets, papers, codes and tools for speech separation and speaker extraction task. You are kindly invited to pull requests. 


## Table of Contents

- [Tutorials](#tutorials)
- [Datasets](#datasets)
- [Papers](#papers)
    - [Speech Separation based on Brain Studies](#Speech-Separation-based-on-Brain-Studies)
    - [Pure Speech Separation](#Pure-Speech-Separation)
    - [Multi-Model Speech Separation](#Multi-Model-Speech-Separation)
    - [Multi-Channel Speech Separation](#Multi-channel-Speech-Separation)
    - [Speaker Extraction](#Speaker-Extraction)
- [Tools](#Tools)
    - [System Tool](#System-Tools)
    - [Evaluation Tool](#Evaluation-Tools)
- [Results on WSJ0-2mix](#Results-on-WSJ0-2mix)


## Tutorials

- [Speech Separation, Hung-yi Lee, 2020] [[Video (Subtitle)]](https://www.bilibili.com/video/BV1Cf4y1y7FN?from=search&seid=17392360823608929388) [[Video]](https://www.youtube.com/watch?v=tovg5ZxNgIo&t=8s) [[Slide]](http://speech.ee.ntu.edu.tw/~tlkagk/courses/DLHLP20/SP%20(v3).pdf)

- [Advances in End-to-End Neural Source Separation, Yi Luo, 2020] [[Video (BiliBili)]](https://www.bilibili.com/video/BV11T4y1774e) [[Video]](https://www.shenlanxueyuan.com/open/course/62/lesson/57/liveToVideoPreview) [[Slide]](https://github.com/gemengtju/Tutorial_Separation/blob/master/slides/Advances_in_end-to-end_neural_source_separation.pdf)

- [Audio Source Separation and Speech Enhancement, Emmanuel Vincent, 2018] [[Book]](https://github.com/gemengtju/Tutorial_Separation/tree/master/book)

- [Audio Source Separation, Shoji Makino, 2018] [[Book]](https://github.com/gemengtju/Tutorial_Separation/tree/master/book)

- [Overview Papers] [[Paper (DeLiang Wang)]](https://arxiv.org/ftp/arxiv/papers/1708/1708.07524.pdf) [[Paper (Bo Xu)]](http://www.aas.net.cn/article/zdhxb/2019/2/234) [[Paper (Zafar Rafii)]](https://arxiv.org/pdf/1804.08300.pdf) [[Paper (Sharon Gannot)]](https://hal.inria.fr/hal-01414179v2/document)

- [Overview Slides] [[Slide (DeLiang Wang)]](https://github.com/gemengtju/Tutorial_Separation/blob/master/slides/DeLiangWang_ASRU19.pdf) [[Slide (Haizhou Li)]](https://github.com/gemengtju/Tutorial_Separation/blob/master/slides/HaizhouLi_CCF.pdf) [[Slide (Meng Ge)]](https://github.com/gemengtju/Tutorial_Separation/blob/master/slides/overview-GM.pdf)

## Datasets

- [WSJ0] [[Dataset]](https://catalog.ldc.upenn.edu/LDC93S6A)

- [WSJ0-2mix] [[Script]](https://github.com/gemengtju/Tutorial_Separation/tree/master/generation/wsj0-2mix)

- [WSJ0-2mix-extr] [[Script]](https://github.com/xuchenglin28/speaker_extraction)

- [WHAM & WHAMR] [[Paper (WHAM)]](https://arxiv.org/pdf/1907.01160.pdf) [[Paper (WHAMR)]](https://arxiv.org/pdf/1910.10279.pdf) [[Dataset]](http://wham.whisper.ai/)

- [LibriMix] [[Paper]](https://arxiv.org/pdf/2005.11262.pdf) [[Script]](https://github.com/JorisCos/LibriMix)

- [LibriCSS] [[Paper]](https://arxiv.org/pdf/2001.11482.pdf) [[Script]](https://github.com/chenzhuo1011/libri_css)

- [SparseLibriMix] [[Script]](https://github.com/popcornell/SparseLibriMix)

- [VCTK-2Mix] [[Script]](https://github.com/JorisCos/VCTK-2Mix)

- [CHIME5 & CHIME6 Challenge] [[Dataset]](https://chimechallenge.github.io/chime6/)

- [AudioSet] [[Dataset]](https://research.google.com/audioset/download.html)

- [Microsoft DNS Challenge] [[Dataset]](https://github.com/microsoft/DNS-Challenge)

- [AVSpeech] [[Dataset]](https://looking-to-listen.github.io/avspeech/download.html)

- [LRW] [[Dataset]](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html)

- [LRS2] [[Dataset]](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html)

- [LRS3] [[Dataset]](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) [[Script]](https://github.com/JusperLee/LRS3-For-Speech-Separationhttps://github.com/JusperLee/LRS3-For-Speech-Separation)

- [VoxCeleb] [[Dataset]](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)


## Papers

### Speech Separation based on Brain Studies

- [Attentional Selection in a Cocktail Party Environment Can Be Decoded from Single-Trial EEG, James, Cerebral Cortex 2012] [[Paper]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4481604/pdf/bht355.pdf)

- [Selective cortical representation of attended speaker in multi-talker speech perception, Nima Mesgarani, Nature 2012] [[Paper]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3870007/pdf/nihms445767.pdf)

- [Neural decoding of attentional selection in multi-speaker environments without access to clean sources, James, Journal of Neural Engineering 2017] [[Paper]](https://europepmc.org/article/pmc/pmc5805380#free-full-text)

- [Speech synthesis from neural decoding of spoken sentences, Gopala K. Anumanchipalli, Nature 2019] [[Paper]](https://www.univie.ac.at/mcogneu/lit/anumanchipalli-19.pdf)

- [Towards reconstructing intelligible speech from the human auditory cortex, HassanAkbari, Scientific Reports 2019] [[Paper]](https://www.nature.com/articles/s41598-018-37359-z.pdf) [[Code]](http://naplab.ee.columbia.edu/naplib.html)

### Pure Speech Separation

- [Joint Optimization of Masks and Deep Recurrent Neural Networks for Monaural Source Separation, Po-Sen Huang, TASLP 2015] [[Paper]](https://arxiv.org/pdf/1502.04149) [[Code (posenhuang)]](https://github.com/posenhuang/deeplearningsourceseparation)

- [Complex Ratio Masking for Monaural Speech Separation, DS Williamson, TASLP 2015] [[Paper]](https://ieeexplore.ieee.org/abstract/document/7364200/)

- [Deep clustering: Discriminative embeddings for segmentation and separation, JR Hershey,  ICASSP 2016] [[Paper]](https://arxiv.org/abs/1508.04306) [[Code (Kai Li)]](https://github.com/JusperLee/Deep-Clustering-for-Speech-Separation) [[Code (Jian Wu)]](https://github.com/funcwj/deep-clustering) [[Code (asteroid)]](https://github.com/mpariente/asteroid/blob/master/egs/wsj0-mix/DeepClustering)

- [Single-channel multi-speaker separation using deep clustering, Y Isik, Interspeech 2016] [[Paper]](https://arxiv.org/pdf/1607.02173) [[Code (Kai Li)]](https://github.com/JusperLee/Deep-Clustering-for-Speech-Separation) [[Code (Jian Wu)]](https://github.com/funcwj/deep-clustering)

- [Permutation invariant training of deep models for speaker-independent multi-talker speech separation, Dong Yu, ICASSP 2017] [[Paper]](https://arxiv.org/pdf/1607.00325) [[Code (Kai Li)]](https://github.com/JusperLee/UtterancePIT-Speech-Separation) [[Code (Sining Sun)]](https://github.com/snsun/pit-speech-separation)

- [Recognizing Multi-talker Speech with Permutation Invariant Training, Dong Yu, ICASSP 2017] [[Paper]](https://arxiv.org/pdf/1704.01985)

- [Multitalker speech separation with utterance-level permutation invariant training of deep recurrent neural networks, M Kolbæk, TASLP 2017] [[Paper]](https://arxiv.org/pdf/1703.06284) [[Code (Kai Li)]](https://github.com/JusperLee/UtterancePIT-Speech-Separation)

- [Deep attractor network for single-microphone speaker separation, Zhuo Chen, ICASSP 2017] [[Paper]](https://arxiv.org/abs/1611.08930) [[Code (Kai Li)]](https://github.com/JusperLee/DANet-For-Speech-Separation)

- [Alternative Objective Functions for Deep Clustering, Zhong-Qiu Wang, ICASSP 2018] [[Paper]](http://www.merl.com/publications/docs/TR2018-005.pdf)

- [Listen, Think and Listen Again: Capturing Top-down Auditory Attention for Speaker-independent Speech Separation, Jing Shi, IJCAI 2018] [[Paper]](https://www.ijcai.org/Proceedings/2018/0605.pdf) 

- [End-to-End Speech Separation with Unfolded Iterative Phase Reconstructioni, Zhong-Qiu Wang et al. 2018] [[Paper]](https://arxiv.org/pdf/1804.10204.pdf)

- [Modeling Attention and Memory for Auditory Selection in a Cocktail Party Environment, Jiaming Xu, AAAI 2018] [[Paper]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16670/15950) [[Code]](https://github.com/jacoxu/ASAM)

- [Speaker-independent Speech Separation with Deep Attractor Network, Luo Yi, TASLP 2018] [[Paper]](https://arxiv.org/pdf/1707.03634) [[Code (Kai Li)]](https://github.com/JusperLee/DANet-For-Speech-Separation)

- [Listening to Each Speaker One by One with Recurrent Selective Hearing Networks, Keisuke Kinoshita, ICASSP 2018] [[Paper]](http://150.162.46.34:8080/icassp2018/ICASSP18_USB/pdfs/0005064.pdf)

- [Tasnet: time-domain audio separation network for real-time, single-channel speech separation, Luo Yi, ICASSP 2018] [[Paper]](https://arxiv.org/pdf/1711.00541) [[Code (Kai Li)]](https://github.com/JusperLee/Conv-TasNet) [[Code (asteroid)]](https://github.com/mpariente/asteroid/blob/master/egs/whamr/TasNet)

- [Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation, Luo Yi, TASLP 2019] [[Paper]](https://ieeexplore.ieee.org/iel7/6570655/6633080/08707065.pdf) [[Code (Kai Li)]](https://github.com/JusperLee/Conv-TasNet) [[Code (asteroid)]](https://github.com/mpariente/asteroid/blob/master/egs/wham/ConvTasNet)

- [Divide and Conquer: A Deep CASA Approach to Talker-independent Monaural Speaker Separation, Yuzhou Liu, TASLP 2019] [[Paper]](https://arxiv.org/pdf/1904.11148) [[Code]](https://github.com/yuzhou-git/deep-casa) [[Code]](https://github.com/yuzhou-git/deep-casa)

- [Improved Speech Separation with Time-and-Frequency Cross-domain Joint Embedding and Clustering, Gene-Ping Yang, Interspeech 2019] [[Paper]](https://arxiv.org/pdf/1904.07845v1.pdf) [[Code]](https://github.com/r06944010/Speech-Separation-TF2)

- [Dual-path RNN: efficient long sequence modeling for time-domain single-channel speech separation, Luo Yi, Arxiv 2019] [[Paper]](https://arxiv.org/pdf/1910.06379) [[Code (Kai Li)]](https://github.com/JusperLee/Dual-Path-RNN-Pytorch)

- [A comprehensive study of speech separation: spectrogram vs waveform separation, Fahimeh Bahmaninezhad, Interspeech 2019] [[Paper]](https://arxiv.org/pdf/1905.07497)

- [Discriminative Learning for Monaural Speech Separation Using Deep Embedding Features, Cunhang Fan, Interspeech 2019] [[Paper]](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1940.pdf)

- [Interrupted and cascaded permutation invariant training for speech separation, Gene-Ping Yang, ICASSP, 2020][[Paper]](https://arxiv.org/abs/1910.12706)

- [FurcaNeXt: End-to-end monaural speech separation with dynamic gated dilated temporal convolutional networks, Liwen Zhang, MMM 2020] [[Paper]](https://arxiv.org/pdf/1902.04891)

- [Filterbank design for end-to-end speech separation, Manuel Pariente et al., ICASSP 2020] [[Paper]](https://128.84.21.199/abs/1910.10400)

- [Voice Separation with an Unknown Number of Multiple Speakers, Eliya Nachmani, Arxiv 2020] [[Paper]](https://arxiv.org/pdf/2003.01531.pdf) [[Demo]](https://enk100.github.io/speaker_separation/)

- [AN EMPIRICAL STUDY OF CONV-TASNET, Berkan Kadıoglu , Arxiv 2020] [[Paper]](https://arxiv.org/pdf/2002.08688.pdf) [[Code]](https://github.com/JusperLee/Deep-Encoder-Decoder-Conv-TasNet)

- [Voice Separation with an Unknown Number of Multiple Speakers, Eliya Nachmani, Arxiv 2020] [[Paper]](https://arxiv.org/pdf/2003.01531.pdf)

- [Wavesplit: End-to-End Speech Separation by Speaker Clustering, Neil Zeghidour et al. Arxiv 2020 ] [[Paper]](https://arxiv.org/abs/2002.08933)

- [La Furca: Iterative Context-Aware End-to-End Monaural Speech Separation Based on Dual-Path Deep Parallel Inter-Intra Bi-LSTM with Attention, Ziqiang Shi, Arxiv 2020] [[Paper]](https://arxiv.org/pdf/2001.08998.pdf)

- [Deep Attention Fusion Feature for Speech Separation with End-to-End Post-ﬁlter Method, Cunhang Fan, Arxiv 2020] [[Paper]](https://arxiv.org/abs/2003.07544)

- [Identify Speakers in Cocktail Parties with End-to-End Attention, Junzhe Zhu, Arxiv 2018] [[Paper]](https://arxiv.org/pdf/2005.11408v1.pdf) [[Code]](https://github.com/JunzheJosephZhu/Identify-Speakers-in-Cocktail-Parties-with-E2E-Attention)

- [Sequence to Multi-Sequence Learning via Conditional Chain Mapping for Mixture Signals, Jing Shi, Arxiv 2020] [[Paper]](https://arxiv.org/pdf/2006.14150.pdf) [[Code/Demo]](https://demotoshow.github.io/)

- [Speaker-Conditional Chain Model for Speech Separation and Extraction, Jing Shi, Arxiv 2020] [[Paper]](https://arxiv.org/pdf/2006.14149.pdf) [[Code/Demo]](https://shincling.github.io/)

- [Improving Voice Separation by Incorporating End-to-end Speech Recognition, Naoya Takahashi, ICASSP 2020] [[Paper]](https://arxiv.org/pdf/1911.12928v2.pdf) [[Code]](https://github.com/pragyak412/Improving-Voice-Separation-by-Incorporating-End-To-End-Speech-Recognition)

- [A Multi-Phase Gammatone Filterbank for Speech Separation via TasNet, David Ditter, ICASSP 2020] [[Paper]](https://arxiv.org/pdf/1910.11615v2.pdf) [[Code]](https://github.com/sp-uhh/mp-gtf)

- [Two-Step Sound Source Separation: Training on Learned Latent Targets, Efthymios Tzinis, ICASSP 2020] [[Paper]](https://arxiv.org/pdf/1910.09804v2.pdf) [[Code (Asteroid)]](https://github.com/mpariente/asteroid) [[Code (Tzinis)]](https://github.com/etzinis/two_step_mask_learning)

- [Unsupervised Sound Separation Using Mixtures of Mixtures, Scott Wisdom, Arxiv] [[Paper]](https://arxiv.org/pdf/2006.12701.pdf)

### Multi-Model Speech Separation

- [Deep Audio-Visual Learning: A Survey, Hao Zhu, Arxiv 2020] [[Paper]](https://arxiv.org/pdf/2001.04758.pdf)

- [Audio-Visual Speech Enhancement Using Multimodal Deep Convolutional Neural Networks, Jen-Cheng Hou, TETCI 2017] [[Paper]](https://arxiv.org/pdf/1703.10893) [[Code]](https://github.com/avivga/audio-visual-speech-enhancement)

- [The Sound of Pixels, Hang Zhao, ECCV 2018] [[Paper/Demo]](http://sound-of-pixels.csail.mit.edu/)

- [Learning to Separate Object Sounds by Watching Unlabeled Video, Ruohan Gao, ECCV 2018] [[Paper]](https://arxiv.org/pdf/1804.01665.pdf)

- [The Conversation: Deep Audio-Visual Speech Enhancement, Triantafyllos Afouras, Interspeech 2018] [[Paper]](https://arxiv.org/pdf/1804.04121)

- [End-to-end audiovisual speech recognition, Stavros Petridis, ICASSP 2018] [[Paper]](https://arxiv.org/pdf/1802.06424) [[Code]](https://github.com/mpc001/end-to-end-lipreading)

- [The Sound of Pixels, Hang Zhao, ECCV 2018] [[Paper]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Hang_Zhao_The_Sound_of_ECCV_2018_paper.pdf) [[Code]](https://github.com/hangzhaomit/Sound-of-Pixels)

- [Looking to Listen at the Cocktail Party: A Speaker-Independent Audio-Visual Model for Speech Separation, ARIEL EPHRAT, ACM Transactions on Graphics 2018] [[Paper]](https://arxiv.org/pdf/1804.03619) [[Code]](https://github.com/JusperLee/Looking-to-Listen-at-the-Cocktail-Party)

- [Learning to Separate Object Sounds by Watching Unlabeled Video, Ruohan Gao, ECCV 2018] [[Paper]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ruohan_Gao_Learning_to_Separate_ECCV_2018_paper.pdf)

- [Time domain audio visual speech separation, Jian Wu, Arxiv 2019] [[Paper]](https://arxiv.org/pdf/1904.03760)

- [Co-Separating Sounds of Visual Objects, Ruohan Gao, ICCV 2019] [[Paper]](https://arxiv.org/pdf/1904.07750.pdf)

- [Recursive Visual Sound Separation Using Minus-Plus Net, Xudong Xu, ICCV 2019] [[Paper]](https://arxiv.org/pdf/1908.11602.pdf)

- [The Sound of Motions, Hang Zhao, ICCV 2019] [[Paper]](https://arxiv.org/pdf/1904.05979.pdf)

- [Audio-Visual Speech Separation and Dereverberation with a Two-Stage Multimodal Network, Ke Tan, Arxiv 2019] [[Paper]](https://arxiv.org/pdf/1909.07352)

- [Co-Separating Sounds of Visual Objects, Ruohan Gao, ICCV 2019] [[Paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Gao_Co-Separating_Sounds_of_Visual_Objects_ICCV_2019_paper.pdf) [[Code]](https://github.com/rhgao/co-separation)

- [Face Landmark-based Speaker-Independent Audio-Visual Speech Enhancement in Multi-Talker Environments, Giovanni Morrone, Arxiv 2019] [[Paper]](https://arxiv.org/pdf/1811.02480v3.pdf) [[Code]](https://github.com/dr-pato/audio_visual_speech_enhancement)

- [Music Gesture for Visual Sound Separation, Chuang Gao, CVPR 2020] [[Paper]](https://arxiv.org/pdf/2004.09476.pdf)

- [FaceFilter: Audio-visual speech separation using still images, Soo-Whan Chung, Arxiv 2020] [[Paper]](https://arxiv.org/pdf/2005.07074.pdf)

- [Awesome Audio-Visual, Github, Kranti Kumar Parida] [[Github Link]](https://github.com/krantiparida/awesome-audio-visual)

### Multi-channel Speech Separation

- [FaSNet: Low-latency Adaptive Beamforming for Multi-microphone Audio Processing, Yi Luo , Arxiv 2019] [[Paper]](https://arxiv.org/abs/1909.13387)

- [MIMO-SPEECH: End-to-End Multi-Channel Multi-Speaker Speech Recognition, Xuankai Chang et al., ASRU 2020] [[Paper]](https://arxiv.org/pdf/1910.06522.pdf)

- [End-to-end Microphone Permutation and Number Invariant Multi-channel Speech Separation, Yi Luo et al., ICASSP 2020] [[Paper]](https://arxiv.org/pdf/1910.14104.pdf) [[Code]](https://github.com/yluo42/TAC)

- [Enhancing End-to-End Multi-channel Speech Separation via Spatial Feature Learning, Rongzhi Guo, ICASSP 2020] [[Paper]](https://arxiv.org/pdf/2003.03927.pdf)

- [Multi-modal Multi-channel Target Speech Separation, Rongzhi Guo, J-STSP 2020] [[Paper]](https://arxiv.org/pdf/2003.07032.pdf)

### Speaker Extraction

- [Single channel target speaker extraction and recognition with speaker beam, Marc Delcroix, ICASSP 2018] [[Paper]](http://150.162.46.34:8080/icassp2018/ICASSP18_USB/pdfs/0005554.pdf)

- [VoiceFilter: Targeted Voice Separation by Speaker-Conditioned Spectrogram Masking, Quan Wang, INTERSPEECH 2018] [[Paper]](https://arxiv.org/pdf/1810.04826.pdf) [[Code (Jian Wu)]](https://github.com/funcwj/voice-filter)

- [Single-Channel Speech Extraction Using Speaker Inventory and Attention Network, Xiong Xiao et al, ICASSP 2019] [[Paper]](http://150.162.46.34:8080/icassp2019/ICASSP2019/pdfs/0000086.pdf)

- [Optimization of Speaker Extraction Neural Network with Magnitude and Temporal Spectrum Approximation Loss, Chenglin Xu, ICASSP 2019] [[Paper]](https://arxiv.org/pdf/1903.09952.pdf) [[Code]](https://github.com/xuchenglin28/speaker_extraction)

- [Time-domain speaker extraction network, Chenglin Xu, ASRU 2019] [[Paper]](https://arxiv.org/pdf/2004.14762.pdf)

- [SpEx: Multi-Scale Time Domain Speaker Extraction Network, Chenglin Xu, TASLP 2020] [[Paper]](https://arxiv.org/pdf/2004.08326.pdf)

- [Improving speaker discrimination of target speech extraction with time-domain SpeakerBeam, Marc Delcroix, ICASSP 2020] [[Paper]](https://arxiv.org/pdf/2001.08378.pdf)

- [SpEx+: A Complete Time Domain Speaker Extraction Network, Meng Ge, Arxiv 2020] [[Paper]](https://arxiv.org/pdf/2005.04686.pdf) [[Code]](https://github.com/gemengtju/SpEx_Plus/tree/master/nnet)


## Tools

### System Tools

- [Asteroid: the PyTorch-based audio source separation toolkit for researchers, Manuel Pariente et al., ICASSP 2020] [[Tool Link]](https://github.com/mpariente/asteroid)

### Evaluation Tools

- [Performance measurement in blind audio sourceseparation, Emmanuel Vincent et al., TASLP 2004] [[Paper]](https://hal.inria.fr/inria-00544230/document) [[Tool Link]](https://github.com/gemengtju/Tutorial_Separation/tree/master/evaluation/sdr_pesq_sisdr)

- [SDR – Half-baked or Well Done?, Jonathan Le Roux, ICASSP 2019] [[Paper]](https://arxiv.org/pdf/1811.02508) [[Tool Link]](https://github.com/gemengtju/Tutorial_Separation/tree/master/evaluation/sdr_pesq_sisdr)


## Results on WSJ0-2mix

Speech separation (SS) and speaker extraction (SE) on the WSJ0-2mix (8k, min) dataset.

|  Task | Methods  | Model Size  | SDRi  | SI-SDRi  |
| :------------: | :------------: | :------------: | :------------: | :------------: |
| SS  | DPCL++  | 13.6M  | -  | 10.8   | 
| SS  | uPIT-BLSTM-ST  | 92.7M  | 10.0  | -   | 
| SS  | DANet  | 9.1M  | -  | 10.5   | 
| SS  | cuPIT-Grid-RD  | 53.2M  | 10.2  | -   | 
| SS  | SDC-G-MTL  | 53.9M  | 10.5  | -   | 
| SS  | CBLDNN-GAT  | 39.5M  | 11.0  | -   | 
| SS  | Chimera++  | 32.9M  | 12.0  | 11.5   | 
| SS  | WA-MISI-5  | 32.9M  | 13.1  | 12.6   | 
| SS  | BLSTM-TasNet  | 23.6M  | 13.6  | 13.2   | 
| SS  | Conv-TasNet  | 5.1M  | 15.6  | 15.3   | 
| SE  | SpEx  | 10.8M  | 17.0  | 16.6   | 
| SE  | SpEx+  | 11.1M  | 17.6  | 17.4   | 
| SS  | DeepCASA  | 12.8M  | 18.0  | 17.7   | 
| SS  | FurcaNeXt  | 51.4M  | 18.4  | -   | 
| SS  | DPRNN-TasNet  | 2.6M  | 19.0  | 18.8   | 
| SS  | Wavesplit  | -  | 19.2  | 19.0   | 
| SS  | Wavesplit + Dynamic mixing  | -  | 20.6  | 20.4   | 
