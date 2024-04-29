% transfor a series of png files to a mp4 file

% get the list of files
simu_NO = 1;

% png files directory
png_dir = "../Documents/figures/simu"+num2str(simu_NO)+"/";
png_files = dir(png_dir+"*.png");

% sort the files, the files are 1.png, 2.png, 3.png, ..., 150.png

% 首先，你需要将文件名提取到一个单独的 cell array 中
fileNames = {png_files.name};
% 提取文件名中的数字
numbers = cellfun(@(x) sscanf(x, '%d.png'), {png_files.name});

% 对数字进行排序，并获取排序后的索引
[~, idx] = sort(numbers);

% 使用排序后的索引对文件名进行排序
sortedFileNames = {png_files(idx).name};

% 现在，sortedFileNames 包含了排序后的文件名

mp4_file = png_dir+"simu"+num2str(simu_NO)+".mp4";
v = VideoWriter(mp4_file,'MPEG-4');
v.FrameRate = 10;
open(v);
for i = 1:length(sortedFileNames)
    img = imread(png_dir + sortedFileNames{i});
    writeVideo(v,img);
end
close(v);
