simunum=18;
% 判断文件夹是否存在，不存在则创建
if ~exist("../Documents/figures/simu"+num2str(simunum), 'dir')
    mkdir("../Documents/figures/simu"+num2str(simunum));
%     else
%         % 存在则新建文件夹，避免覆盖，文件夹名加上时间戳
%         mkdir("../Documents/figures/simu"+num2str(simunum)+"_"+string(datetime("now", "Format","MM-dd-HH-mm-ss")));
end



file_name = "D:\Codes\DMPC-BP-MultiRobot\data\simu"+num2str(simunum)+"_data.xlsx";

parameter_file = "D:\Codes\DMPC-BP-MultiRobot\data\simu"+num2str(simunum)+"_parameters.json";
obstacle_parameter = get_obstacles(parameter_file);


robot_names = sheetnames(file_name);
r1_history = importfile(file_name, robot_names{1});
r2_history = importfile(file_name, robot_names{2});
r3_history = importfile(file_name, robot_names{3});
r4_history = importfile(file_name, robot_names{4});

all_history = {r1_history, r2_history, r3_history, r4_history};
sq22 = sqrt(2)*2;
d_ref = [0, 2, sq22, 2;
         2, 0, 2, sq22;
         sq22, 2, 0, 2;
         2, sq22, 2, 0];

robot_width = 0.6;

color1 = "#0072BD";
color2 = "#D95319";
color3 = "#EDB120";
color4 = "#77AC30";

% calculate the distance between the robots
N = size(r1_history, 1);
dt = 0.1;
t = (0:N-1)*dt;

robots_num = 4;
distance_errors = zeros(N, nchoosek(robots_num, 2));
k1=1;
for i = 1:robots_num-1
    for j = i+1:robots_num
        for k = 1:N
            distance_errors(k, k1) = norm(all_history{i}(k, 2:3) - all_history{j}(k, 2:3)) - d_ref(i, j);
        end
        k1=k1+1;
    end
end


% rx_history: [index, x, y, theta, v_left, v_right]
% plot the trajectory of the robot
% figure(1)
% hold on;
% plot(r1_history(:,2), r1_history(:,3));
% plot(r2_history(:,2), r2_history(:,3));
% plot(r3_history(:,2), r3_history(:,3));
% plot(r4_history(:,2), r4_history(:,3));
% hold off;
% xlabel('$x(m)$');
% ylabel('$y(m)$');
% title('Trajectories of the robots');
% legend('robot1', 'robot2', 'robot3', 'robot4');
% grid on
% box on
% axis equal

% the animation of the robot
% the robot is a rectangle, the length is 0.8m, the width is 0.6m
% the robot is heading to the positive x-axis when theta = 0

figure(2)
% 分辨率：1750*1117
set(gcf, 'Position', [0, 0, 1120, 715]);

for i=1:N
    figure(2)
    clf
    % 默认渲染器：painters
    set(gcf, 'Renderer', 'painters');
    % 默认字体：times new roman, 20
    set(gca, 'FontName', 'times new roman', 'FontSize', 14);
    % latex解释器
    set(0, 'DefaultTextInterpreter', 'latex');
    % 白色背景
    set(gcf, 'Color', 'w');
    subplot(2,2,1)
    % 坐标轴字体：times new roman
    set(gca, 'FontName', 'times new roman','FontSize',12);
    hold on;
    % plot the trajectory of the robot
    plot(r1_history(1:i,2), r1_history(1:i,3), "Color", color1, "LineWidth", 1.5);
    plot(r2_history(1:i,2), r2_history(1:i,3), "Color", color2, "LineWidth", 1.5);
    plot(r3_history(1:i,2), r3_history(1:i,3), "Color", color3, "LineWidth", 1.5);
    plot(r4_history(1:i,2), r4_history(1:i,3), "Color", color4, "LineWidth", 1.5);
    % plot the robot
    x1 = r1_history(i, 2);
    y1 = r1_history(i, 3);
    theta1 = r1_history(i, 4);
    x2 = r2_history(i, 2);
    y2 = r2_history(i, 3);
    theta2 = r2_history(i, 4);
    x3 = r3_history(i, 2);
    y3 = r3_history(i, 3);
    theta3 = r3_history(i, 4);
    x4 = r4_history(i, 2);
    y4 = r4_history(i, 3);
    theta4 = r4_history(i, 4);
    plot_robot(x1, y1, theta1, color1);
    plot_robot(x2, y2, theta2, color2);
    plot_robot(x3, y3, theta3, color3);
    plot_robot(x4, y4, theta4, color4);
    plot_obstacles(obstacle_parameter)

    hold off;
    axis equal
    axis([-2,9,-3,4])
    grid on
    box on
    xlabel('$x(m)$');
    ylabel('$y(m)$');
    title('Trajectories of the robots', "FontName", "Times New Roman", 'FontSize', 14);
    legend('robot1', 'robot2', 'robot3', 'robot4', "FontName", "Times New Roman", 'FontSize', 14, "NumColumns", 2, "Location", "northeast");
    subplot(2,2,2)
    set(gca, 'FontName', 'times new roman','FontSize',12);
    % plot the distance errors between the robots
    hold on;
    for k = 1:length(distance_errors(1,:))
        plot(t(1:i), distance_errors(1:i, k), "LineWidth", 1);
    end
    hold off
    grid on
    box on
    xlabel('$t(s)$');
    ylabel('$error(m)$');
    title('Distance errors between the robots', "FontName","Times New Roman", 'FontSize', 14);

    subplot(2,2,3)
    set(gca, 'FontName', 'times new roman','FontSize',12);
    hold on;
    % plot the control inputs of the robot
    plot(t(1:i), r1_history(1:i, 5), "Color", color1, "LineWidth", 1.5);
    plot(t(1:i), r1_history(1:i, 6), "Color", color1, "LineWidth", 1.5, "LineStyle", "--");
    plot(t(1:i), r2_history(1:i, 5), "Color", color2, "LineWidth", 1.5);
    plot(t(1:i), r2_history(1:i, 6), "Color", color2, "LineWidth", 1.5, "LineStyle", "--");
    plot(t(1:i), r3_history(1:i, 5), "Color", color3, "LineWidth", 1.5);
    plot(t(1:i), r3_history(1:i, 6), "Color", color3, "LineWidth", 1.5, "LineStyle", "--");
    plot(t(1:i), r4_history(1:i, 5), "Color", color4, "LineWidth", 1.5);
    plot(t(1:i), r4_history(1:i, 6), "Color", color4, "LineWidth", 1.5, "LineStyle", "--");
    hold off;
    grid on
    box on
    xlabel('$t(s)$');
    ylabel('$v(m/s)$');
    title('Control inputs of the robots', "FontName", "Times New Roman", 'FontSize', 14);
    legend('$v_{1,l}$', '$v_{1,r}$', '$v_{2,l}$', '$v_{2,r}$', '$v_{3,l}$', '$v_{3,r}$', '$v_{4,l}$', '$v_{4,r}$', "FontName", "Times New Roman", 'FontSize', 14, "NumColumns", 2, "Interpreter", "latex", "Location", "southeast");

    subplot(2,2,4)
    set(gca, 'FontName', 'times new roman','FontSize',12);
    hold on;
    % plot the v, w of the robot
    plot(t(1:i), (r1_history(1:i, 5)+r1_history(1:i, 6))/2, "Color", color1, "LineWidth", 1.5);
    plot(t(1:i), (r1_history(1:i, 5)-r1_history(1:i, 6))/robot_width, "Color", color1, "LineWidth", 1.0, "LineStyle", "--");
    plot(t(1:i), (r2_history(1:i, 5)+r2_history(1:i, 6))/2, "Color", color2, "LineWidth", 1.5);
    plot(t(1:i), (r2_history(1:i, 5)-r2_history(1:i, 6))/robot_width, "Color", color2, "LineWidth", 1.0, "LineStyle", "--");
    plot(t(1:i), (r3_history(1:i, 5)+r3_history(1:i, 6))/2, "Color", color3, "LineWidth", 1.5);
    plot(t(1:i), (r3_history(1:i, 5)-r3_history(1:i, 6))/robot_width, "Color", color3, "LineWidth", 1.0, "LineStyle", "--");
    plot(t(1:i), (r4_history(1:i, 5)+r4_history(1:i, 6))/2, "Color", color4, "LineWidth", 1.5);
    plot(t(1:i), (r4_history(1:i, 5)-r4_history(1:i, 6))/robot_width, "Color", color4, "LineWidth", 1.0, "LineStyle", "--");
    hold off;
    grid on
    box on
    xlabel('$t(s)$');
    ylabel('$v(m/s), \omega(rad/s)$');
    title('Control inputs of the robots', "FontName", "Times New Roman", 'FontSize', 14);
    legend('$v_1$', '$\omega_1$', '$v_2$', '$\omega_2$', '$v_3$', '$\omega_3$', '$v_4$', '$\omega_4$', "FontName", "Times New Roman", 'FontSize', 14, "NumColumns", 2, "Interpreter", "latex", "Location", "southeast");
       
    pause(dt/10);
    % save the figure in the folder "../Documents/figures/simu1"
    saveas(gcf, strcat("../Documents/figures/simu"+num2str(simunum)+"/", num2str(i), ".png"));
end

%%
function plot_robot(x, y, theta, color)
    % the robot is a rectangle, the length is 0.8m, the width is 0.6m
    % the robot is heading to the positive x-axis when theta = 0
    length = 0.8;
    width = 0.6;
    l_cos = length/2*cos(theta);
    l_sin = length/2*sin(theta);
    w_cos = width/2*cos(theta);
    w_sin = width/2*sin(theta);
    x1 = x - l_cos + w_sin;
    y1 = y - l_sin - w_cos;
    x2 = x + l_cos + w_sin;
    y2 = y + l_sin - w_cos;
    x3 = x + l_cos - w_sin;
    y3 = y + l_sin + w_cos;
    x4 = x - l_cos - w_sin;
    y4 = y - l_sin + w_cos;
    plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1], "Color", color, "LineWidth", 1);
end

%%
function plot_obstacle(x, y, r)
    % plot the obstacle
    % the obstacle is a circle
    % x, y is the center of the circle
    % r is the radius of the circle
    % fill the circle with the gray color
    theta = 0:0.01:2*pi;
    x1 = r*cos(theta) + x;
    y1 = r*sin(theta) + y;
    fill(x1, y1, [0.3, 0.3, 0.3]);
end
%%
function plot_obstacles(obstacle_parameter)
    % plot the obstacle
    % the obstacle is a circle
    % x, y is the center of the circle
    % r is the radius of the circle
    % fill the circle with the gray color
    for i = 1:size(obstacle_parameter, 1)
        x = obstacle_parameter(i, 1);
        y = obstacle_parameter(i, 2);
        r = obstacle_parameter(i, 3);
        plot_obstacle(x, y, r);
    end
end
%%
function obstacle_position = get_obstacles(file_name)
    % file_name: the file name of the obstacle, a .json file
    % *.json: {"obstacle": [{"x": 1, "y": 2, "r": 0.5}, {"x": 2, "y": 3, "r": 0.6}]}

    % read the obstacle information from the file
    fid = fopen(file_name, 'r');
    raw = fread(fid, inf);
    str = char(raw');
    fclose(fid);
    data = jsondecode(str);
    obstacle = data.Obstacle;
    obstacle_position = zeros(length(obstacle), 3);
    for i = 1:length(obstacle)
        obstacle_position(i, 1) = obstacle(i).x;
        obstacle_position(i, 2) = obstacle(i).y;
        obstacle_position(i, 3) = obstacle(i).r;
    end
end