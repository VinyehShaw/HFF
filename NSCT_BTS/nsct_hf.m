%% NSCT high frequency subband extraction and preservation (4 high frequency results per modality, processing multiple patients in parallel)
clear; clc;

% Patient data root directory (please change to your actual path)
baseDir = '/home/mcga/phd/brats2020/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData';

% Add NSCT toolbox (recursively add all subdirectories)
nsct_tbx_dir = '/home/mcga/phd/bra23/NSCT_BTS/nsct_toolbox';
addpath(genpath(nsct_tbx_dir));
rehash;

% NSCT 参数设置 
% NSCT parameter settings
nlevels = [2, 2];      % 这里只取第一层高频分解，共4个方向 Here we only take the first layer of high frequency decomposition, a total of 4 directions
pfilt   = 'pyrexc';    % 金字塔滤波器 pyramid filter
dfilt   = 'cd';        % 方向滤波器 Directional filter

% 获取所有患者文件夹（假设文件夹名称以 'BraTS20_Training_' 开头）
% Get all patient folders (assuming folder names start with 'BraTS20_Training_')
patientDirs = dir(fullfile(baseDir, 'BraTS20_Training_*'));
% patientDirs = dir(fullfile(baseDir, 'BraTS-MEN-*'));
% patientDirs = dir(fullfile(baseDir, 'BraTS19*'));

% 转换为 cell 数组便于 parfor
% Convert to cell array for parfor
patientFolderNames = {patientDirs([patientDirs.isdir]).name};

% 使用 parfor 并行处理每个患者（注意不要在内部再嵌套 parfor）
% Use parfor to process each patient in parallel (be careful not to nest parfor inside)
parfor p = 1:length(patientFolderNames)
    patientFolder = fullfile(baseDir, patientFolderNames{p});
    fprintf('Processing patient folders:\n %s\n', patientFolder);
    
    % 获取该患者文件夹下的 nii 文件（排除包含 'seg' 或 '_h' 或 '_l' 的文件）
    % Get the nii files in the patient folder (excluding files containing 'seg' or '_h' or '_l')
    niiFiles = dir(fullfile(patientFolder, '*.nii*'));
    for f = 1:length(niiFiles)
        niiName = niiFiles(f).name;
        if contains(lower(niiName), 'seg') || contains(lower(niiName), '_h') || contains(lower(niiName), '_l')
            continue; % 跳过分割文件或已处理文件 Skip split files or processed files
        end
        
        fprintf(' Processing modal files: \n %s\n', niiName);
        
        % 使用 fileparts 正确提取 baseName（支持 nii 和 nii.gz） Use fileparts to correctly extract baseName (supports nii and nii.gz)
        [~, name, ext] = fileparts(niiName);
        if strcmp(ext, '.gz')
            [~, name, ~] = fileparts(name); % 如果是 .nii.gz，再次解析
        end
        
        % 构造输出文件名，如 "BraTS20_Training_036_t1_H1.nii.gz"
        % Construct output file name, such as "BraTS20_Training_036_t1_H1.nii.gz"
        outName1 = fullfile(patientFolder, [name, '_H1.nii.gz']);
        outName2 = fullfile(patientFolder, [name, '_H2.nii.gz']);
        outName3 = fullfile(patientFolder, [name, '_H3.nii.gz']);
        outName4 = fullfile(patientFolder, [name, '_H4.nii.gz']);
        
        % 如果已经存在处理结果，则跳过
        % If the processing result already exists, skip it
        if exist(outName1, 'file')
            fprintf('    Processing results already exist:%s, skip processing this modal file.\n', outName1);
            continue;
        end
        
        % 读取 nii 数据（假设为 3D 数据）
        % Read nii data (assuming 3D data)
        filePath = fullfile(patientFolder, niiName);
        nii_data = niftiread(filePath);
        [~, ~, nslices] = size(nii_data);
        
        % 预处理第一张 slice 获取高频子带尺寸（假设所有 slice 尺寸一致）
        % Preprocess the first slice to get the high frequency subband size (assuming all slices have the same size)
        slice_img = double(nii_data(:,:,1));
        coef_tmp = nsctdec(slice_img, nlevels, dfilt, pfilt);
        [r_hf, c_hf] = size(coef_tmp{2}{1});
        
        % 初始化 3D 矩阵存放 4 个高频子带
        % Initialize the 3D matrix to store 4 high frequency subbands
        HF1 = zeros(r_hf, c_hf, nslices);
        HF2 = zeros(r_hf, c_hf, nslices);
        HF3 = zeros(r_hf, c_hf, nslices);
        HF4 = zeros(r_hf, c_hf, nslices);
        
        % 顺序处理每个 slice
        % Process each slice sequentially
        for k = 1:nslices
            slice_img = double(nii_data(:,:,k));
            coef = nsctdec(slice_img, nlevels, dfilt, pfilt);
            
            % 提取第一层高频子带的 4 个方向
            % Extract the 4 directions of the first layer high frequency subband
            hf1 = coef{2}{1};
            hf2 = coef{2}{2};
            hf3 = coef{2}{3};
            hf4 = coef{2}{4};
            
            % 对每个子带进行对比度调整
            % Adjust the contrast of each subband
            adj_hf1 = imadjust(hf1, stretchlim(hf1, [0.01, 0.99]), []);
            adj_hf2 = imadjust(hf2, stretchlim(hf2, [0.01, 0.99]), []);
            adj_hf3 = imadjust(hf3, stretchlim(hf3, [0.01, 0.99]), []);
            adj_hf4 = imadjust(hf4, stretchlim(hf4, [0.01, 0.99]), []);
            
            HF1(:,:,k) = adj_hf1;
            HF2(:,:,k) = adj_hf2;
            HF3(:,:,k) = adj_hf3;
            HF4(:,:,k) = adj_hf4;
        end
        
        % 保存输出文件
        % Save output file
        niftiwrite(single(HF1), outName1, 'Compressed', true);
        niftiwrite(single(HF2), outName2, 'Compressed', true);
        niftiwrite(single(HF3), outName3, 'Compressed', true);
        niftiwrite(single(HF4), outName4, 'Compressed', true);

        fprintf('    Saved:\n      %s\n      %s\n      %s\n      %s\n', ...
            outName1, outName2, outName3, outName4);
    end
end
