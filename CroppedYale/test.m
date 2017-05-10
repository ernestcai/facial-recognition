clc;
filePath ='C:\Users\caiji\Desktop\CroppedYale';    %TODO apply for windows dir
numTrainSamples = 25
[ACCURACY] = SRBFR_l2_correction_PCA(numTrainSamples, filePath)
clear;