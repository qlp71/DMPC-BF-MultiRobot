function simu1dataS3 = importfile(workbookFile, sheetName, dataLines)
%IMPORTFILE 导入电子表格中的数据
%  SIMU1DATAS3 = IMPORTFILE(FILE) 读取名为 FILE 的 Microsoft Excel
%  电子表格文件的第一张工作表中的数据。  返回数值数据。
%
%  SIMU1DATAS3 = IMPORTFILE(FILE, SHEET) 从指定的工作表中读取。
%
%  SIMU1DATAS3 = IMPORTFILE(FILE, SHEET,
%  DATALINES)按指定的行间隔读取指定工作表中的数据。对于不连续的行间隔，请将 DATALINES 指定为正整数标量或 N×2
%  正整数标量数组。
%
%  示例:
%  simu1dataS3 = importfile("D:\Codes\DMPC-BP-MultiRobot\data\simu1_data.xlsx", "robot003", [2, Inf]);
%
%  另请参阅 READTABLE。
%
% 由 MATLAB 于 2024-03-29 11:09:25 自动生成

%% 输入处理

% 如果未指定工作表，则将读取第一张工作表
if nargin == 1 || isempty(sheetName)
    sheetName = 1;
end

% 如果未指定行的起点和终点，则会定义默认值。
if nargin <= 2
    dataLines = [2, Inf];
end

%% 设置导入选项并导入数据
opts = spreadsheetImportOptions("NumVariables", 6);

% 指定工作表和范围
opts.Sheet = sheetName;
opts.DataRange = dataLines(1, :);

% 指定列名称和类型
opts.VariableNames = ["VarName1", "VarName2", "VarName3", "VarName4", "VarName5", "VarName6"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double"];

% 导入数据
simu1dataS3 = readtable(workbookFile, opts, "UseExcel", false);

for idx = 2:size(dataLines, 1)
    opts.DataRange = dataLines(idx, :);
    tb = readtable(workbookFile, opts, "UseExcel", false);
    simu1dataS3 = [simu1dataS3; tb]; %#ok<AGROW>
end

%% 转换为输出类型
simu1dataS3 = table2array(simu1dataS3);
end