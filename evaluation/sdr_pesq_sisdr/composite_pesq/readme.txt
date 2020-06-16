==========================================================================
PESQ measure:
-------------
Usage of the PESQ objective measure is as follows:
   [pesq_mos]=pesq(sfreq,cleanfile.wav,enhanced.wav) 
where 'sfreq' is the sampling frequency in Hz (8000 or 16000 Hz), 'cleanfile.wav' contains the 
clean speech file and 'enhanced.wav'contains the enhanced file.

Example:
To run the PESQ objective measure with the example files provided, type 
in MATLAB:
>> pesq(8000,'sp09.wav','enhanced_logmmse.wav')

ans =

    2.2557


Note that you might encounter an error message (e.g., file is corrupt) if you're using an old version of MATLAB, 
because of p-code incompatibility. The latest version of MATLAB needs to be run.

MATLAB source code for the PESQ implementation is available from a CD-ROM  included 
in the following book:
Loizou, P. (2007) "Speech enhancement: Theory and Practice", CRC Press.

========================================================================
COMPOSITE MEASURE:
-----------------
Usage: [Csig,Cbak,Covl]=composite(cleanfile.wav,enhanced.wav)
where 'Csig' is the predicted rating of speech distortion
      'Cbak' is the predicted rating of background distortion
      'Covl' is the predicted rating of overall quality.
      

You may run example files included in the zip file.
In MATLAB, type:
      
>> [c,b,o]=composite('sp09.wav','enhanced_logmmse.wav')

 LLR=0.681368   SNRseg=3.991727   WSS=49.671978   PESQ=2.255732

c =

    3.3050


b =

    2.6160


o =

    2.7133


where 'sp09.wav' is the clean file and 'enhanced_logmmse.wav' is the enhanced file.
The predicted ratings for overall quality was 2.7133, for background distortion it was 2.61 and for 
signal distortion it was 3.3050.


Any questions, please email: loizou@utdallas.edu

References:
Hu, Y. and Loizou, P. (2008). “Evaluation of objective quality measures for speech enhancement,” 
IEEE Transactions on Speech and Audio Processing, 16(1), 229-238.
