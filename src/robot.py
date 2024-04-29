import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd
import json

w_position = 5.0
w_theta = 5.0
w_inputs = 0.5
w_l = 10
w_f = 2
lambda_u = 0.3
lambda_p = 0.1 # keep formation and distance (avoid collision between the robots)
lambda_obstacle = 0.1
default_std = 0.2
d_min = 1.0
beta_rate = 2


class Robot:
    num_robots = 0

    def __init__(self, x=0.0, y=0.0, theta=0.0, color='b'):
        self.x = x
        self.y = y
        self.theta = theta
        self.v_l = 0.0
        self.v_r = 0.0
        self.v_l_next = []
        self.v_r_next = []
        self.v_l_next_possibility = []
        self.v_r_next_possibility = []
        self.v_l_distribution = {'method': "gaussian", 'horizon': 10, 'mean': [
        ], 'std': [], 'low': [], 'high': []}
        self.v_r_distribution = {'method': "gaussian", 'horizon': 10, 'mean': [
        ], 'std': [], 'low': [], 'high': []}
        self.width = 0.6
        self.length = 0.8
        self.L = 0.6
        self.delta_t = 0.1
        self.v_max = 1.2
        self.v_min = -1.2
        self.color = color
        self.history = []
        self.obstacles = []
        self.id = Robot.num_robots
        # name: robot_id, "0" is used to fill the number to 3 digits
        self.name = "robot" + str(self.id).zfill(3)
        Robot.num_robots += 1
        # the next attributes are used for the belief propagation algorithm
        self.neighbors = {}
        # the neighbors of the robot, the key is the id of the neighbor, the value is a list [robot, random_trajectories, p_trajectories, ref_trajectory, delta_x, delta_y]，
        self.neighbors_num = 0
        self.random_trajectory_num = 200
        self.random_inputs = []
        # self.random_inputs: [[v_l_next, v_r_next], ...],
        self.random_trajectories = []
        # self.random_trajectories: [np.ndarray, np.ndarray, ...], the shape of the np.ndarray is (steps, 5), the 5 elements are x, y, theta, v_l, v_r
        self.possibility_trajectories = []
        # self.possibility_trajectories: [float, float, ...], the length is the number of the trajectories, the element is the possibility of the trajectory
        self.messages_tobe_sent = {}
        # self.random_trajectories_message:
        # {neighbor_id: [float, float, ...], ...}]
        # in the paper, message is a function $m_{j,i}(·)$ input · is a trajctory of j-th robot, output is a float
        # for the robot j, the demension of the message is the number of the trajectories * the number of the neighbors
        # this variable is used for itself to store the messages and update the messages, it will be sent to the neighbors
        self.messages_been_received = {}
        # self.messages_store: {neighbor_id: [float, float, ...], ...}
        # the messages from the neighbors, this will be used to update the messages_tobe_sent, and messages_tobe_sent will be sent to the neighbors in the next step
        self.predict_horizon = 10
        self.ref_trajectory = np.zeros((self.predict_horizon, 5))

    def re_initial(self):
        # when the robot system is established, some variables need redefine
        # the main change is the demension of the variables related to self.random_trajectory_num, self.predict_horizon
        self.ref_trajectory = np.zeros((self.predict_horizon, 5))
        self.v_l_distribution["horizon"] = self.predict_horizon
        self.v_r_distribution["horizon"] = self.predict_horizon
        if self.v_l_distribution["method"] == "gaussian" or "normal":
            self.v_l_distribution["mean"] = [0.0] * self.predict_horizon
            self.v_l_distribution["std"] = [default_std] * self.predict_horizon
        elif self.v_l_distribution["method"] == "uniform":
            self.v_l_distribution["low"] = [self.v_min] * self.predict_horizon
            self.v_l_distribution["high"] = [self.v_max] * self.predict_horizon
        if self.v_r_distribution["method"] == "gaussian" or "normal":
            self.v_r_distribution["mean"] = [0.0] * self.predict_horizon
            self.v_r_distribution["std"] = [default_std] * self.predict_horizon
        elif self.v_r_distribution["method"] == "uniform":
            self.v_r_distribution["low"] = [self.v_min] * self.predict_horizon
            self.v_r_distribution["high"] = [self.v_max] * self.predict_horizon
        # self.gen_random_speed_next_based_on_distribution()

    def set_position(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def set_speed(self, v_l, v_r):
        self.v_l = v_l
        self.v_r = v_r
        if self.v_l > self.v_max:
            self.v_l = self.v_max
        if self.v_l < self.v_min:
            self.v_l = self.v_min
        if self.v_r > self.v_max:
            self.v_r = self.v_max
        if self.v_r < self.v_min:
            self.v_r = self.v_min

    def add_speed_next(self, v_l, v_r, p_l=1.0, p_r=1.0):
        if isinstance(v_l, list):
            self.v_l_next.extend(v_l)
        else:
            self.v_l_next.append(v_l)
        if isinstance(v_r, list):
            self.v_r_next.extend(v_r)
        else:
            self.v_r_next.append(v_r)
        if isinstance(p_l, list):
            self.v_l_next_possibility.extend(p_l)
        else:
            self.v_l_next_possibility.append(p_l)
        if isinstance(p_r, list):
            self.v_r_next_possibility.extend(p_r)
        else:
            self.v_r_next_possibility.append(p_r)

    def set_speed_next(self, v_l, v_r, p_l=1.0, p_r=1.0):
        self.v_l_next = v_l
        self.v_r_next = v_r
        self.v_l_next_possibility = p_l
        self.v_r_next_possibility = p_r

    def check_speed_next(self):
        if len(self.v_l_next) != len(self.v_r_next):
            return False
        return True

    def is_empty_speed_next(self):
        if self.check_speed_next():
            return len(self.v_l_next) == 0
        else:
            print("The length of v_l_next and v_r_next is not equal.")
            return False

    def clear_speed_next(self):
        self.v_l_next = []
        self.v_r_next = []
        self.v_l_next_possibility = []
        self.v_r_next_possibility = []

    def __list__(self):
        return [self.x, self.y, self.theta, self.v_l, self.v_r]

    def __str__(self):
        # x, y, theta, v_l, v_r formate %.2f, keep 2 decimal places
        return "x: %.2f,\t y: %.2f,\t theta: %.2f,\t v_l: %.2f,\t v_r: %.2f" % (self.x, self.y, self.theta, self.v_l, self.v_r)

    def move(self):
        # change the position of the robot according to the speed
        self.v_l = self.v_l_next.pop(
            0) if len(self.v_l_next) > 0 else self.v_l
        self.v_r = self.v_r_next.pop(
            0) if len(self.v_r_next) > 0 else self.v_r
        self.x += (self.v_l + self.v_r) / 2 * np.cos(self.theta) * self.delta_t
        self.y += (self.v_l + self.v_r) / 2 * np.sin(self.theta) * self.delta_t
        self.theta += (self.v_r - self.v_l) / self.L * self.delta_t
        self.v_l_next_possibility.pop(0) if self.v_l_next_possibility else None
        self.v_r_next_possibility.pop(0) if self.v_r_next_possibility else None
        self.history.append(self.__list__())
        return self.__list__()

    def move_steps(self, steps):
        for _ in range(steps):
            self.move()
        return self.__list__()

    def predict(self):
        # output the position and the speed of the robot after delta_t, the position is not changed
        x = self.x + (self.v_l + self.v_r) / 2 * \
            np.cos(self.theta) * self.delta_t
        y = self.y + (self.v_l + self.v_r) / 2 * \
            np.sin(self.theta) * self.delta_t
        theta = self.theta + (self.v_r - self.v_l) / self.L * self.delta_t
        return [x, y, theta, self.v_l, self.v_r]

    def predict_steps(self, steps):
        output = []
        v_idx = 0
        v_l = self.v_l
        v_r = self.v_r
        x0 = self.x
        y0 = self.y
        theta0 = self.theta
        for _ in range(steps):
            x0 = x0 + (v_l + v_r) / 2 * np.cos(theta0) * self.delta_t
            y0 = y0 + (v_l + v_r) / 2 * np.sin(theta0) * self.delta_t
            theta0 = theta0 + (v_r - v_l) / self.L * self.delta_t
            if v_idx < len(self.v_l_next):
                v_l = self.v_l_next[v_idx]
                v_r = self.v_r_next[v_idx]
                v_idx += 1
            else:
                v_l = self.v_l_next[-1]
                v_r = self.v_r_next[-1]
            output.append([x0, y0, theta0, v_l, v_r])
        return output, self.v_l_next_possibility, self.v_r_next_possibility

    def gen_random_speed(self, method, **kwargs):
        # method = 'uniform' or 'normal'/'gaussian'
        # kwargs = {'low_l': , 'high_l': , 'low_r': , 'high_r': } or {'mean_l': , 'std_l': , 'mean_r': , 'std_r': }
        # return [v_l, v_r] and the possibility of v_l and v_r
        if method == 'uniform':
            low_l = kwargs.get('low_l', self.v_min)
            high_l = kwargs.get('high_l', self.v_max)
            low_r = kwargs.get('low_r', self.v_min)
            high_r = kwargs.get('high_r', self.v_max)
            v_l = np.random.uniform(low_l, high_l)
            v_r = np.random.uniform(low_r, high_r)
            p_v_l = 1 / (high_l - low_l)
            p_v_r = 1 / (high_r - low_r)
        elif method == 'normal' or method == 'gaussian':
            mean_l = kwargs.get('mean_l', 0.0)
            # std_l = kwargs.get('std_l', (self.v_max-self.v_min)/6)
            std_l = kwargs.get('std_l', default_std)
            mean_r = kwargs.get('mean_r', 0.0)
            # std_r = kwargs.get('std_r', (self.v_max-self.v_min)/6)
            std_r = kwargs.get('std_r', default_std)
            v_l = np.random.normal(mean_l, std_l)
            v_r = np.random.normal(mean_r, std_r)
            p_v_l = 1 / (std_l * np.sqrt(2 * np.pi)) * \
                np.exp(-0.5 * (v_l - mean_l) ** 2 / std_l ** 2)
            p_v_r = 1 / (std_r * np.sqrt(2 * np.pi)) * \
                np.exp(-0.5 * (v_r - mean_r) ** 2 / std_r ** 2)
        else:
            return self.gen_random_speed('uniform', low_l=self.v_min, high_l=self.v_max, low_r=self.v_min, high_r=self.v_max)
        return [v_l, v_r, p_v_l, p_v_r]

    def set_random_speed(self, method, **kwargs):
        [v_l, v_r, _, _] = self.gen_random_speed(method, **kwargs)
        self.set_speed(v_l, v_r)
        return [self.v_l, self.v_r]

    def gen_distribution_based_on_ref_trajectory(self):
        # the distribution of the speed is based on the ref_trajectory
        # the mean of the speed is the same as the ref_trajectory
        # the std of the speed is the same as the ref_trajectory
        # the horizon of the distribution is the same as the ref_trajectory
        self.v_l_distribution['mean'] = [state[3] for state in self.ref_trajectory]
        self.v_l_distribution['std'] = [default_std] * self.predict_horizon
        self.v_r_distribution['mean'] = [state[4] for state in self.ref_trajectory]
        self.v_r_distribution['std'] = [default_std] * self.predict_horizon

    def gen_random_speed_next(self, steps=1, method="uniform", **kwargs):
        self.clear_speed_next()
        if method == 'uniform':
            low_l = kwargs.get('low_l', self.v_min)
            high_l = kwargs.get('high_l', self.v_max)
            low_r = kwargs.get('low_r', self.v_min)
            high_r = kwargs.get('high_r', self.v_max)
            low_l = [low_l] * steps if isinstance(low_l, float) else low_l
            high_l = [high_l] * steps if isinstance(high_l, float) else high_l
            low_r = [low_r] * steps if isinstance(low_r, float) else low_r
            high_r = [high_r] * steps if isinstance(high_r, float) else high_r
            for i in range(steps):
                [v_l, v_r, p_v_l, p_v_r] = self.gen_random_speed(
                    method, low_l=low_l[i], high_l=high_l[i], low_r=low_r[i], high_r=high_r[i])
                self.add_speed_next(v_l, v_r, p_v_l, p_v_r)
        elif method == 'normal' or method == 'gaussian':
            mean_l = kwargs.get('mean_l', [0.0] * steps)
            # if len(mean_l) == 0:
            #     mean_l = [0.0] * steps
            # std_l = kwargs.get('std_l', (self.v_max-self.v_min)/6)
            std_l = kwargs.get('std_l', [default_std] * steps)
            # if len(std_l) == 0:
            #     std_l = [default_std] * steps
            mean_r = kwargs.get('mean_r', [0.0] * steps)
            # if len(mean_r) == 0:
            #     mean_r = [0.0] * steps
            # std_r = kwargs.get('std_r', (self.v_max-self.v_min)/6)
            std_r = kwargs.get('std_r', [default_std] * steps)
            # if len(std_r) == 0:
            #     std_r = [default_std] * steps
            mean_l = [mean_l] * steps if isinstance(mean_l, float) else mean_l
            std_l = [std_l] * steps if isinstance(std_l, float) else std_l
            mean_r = [mean_r] * steps if isinstance(mean_r, float) else mean_r
            std_r = [std_r] * steps if isinstance(std_r, float) else std_r
            for i in range(steps):
                [v_l, v_r, p_v_l, p_v_r] = self.gen_random_speed(
                    method, mean_l=mean_l[i], std_l=std_l[i], mean_r=mean_r[i], std_r=std_r[i])
                self.add_speed_next(v_l, v_r, p_v_l, p_v_r)
        else:
            return self.gen_random_speed_next(steps, 'uniform', low_l=[self.v_min]*steps, high_l=[self.v_max]*steps, low_r=[self.v_min]*steps, high_r=[self.v_max]*steps)
        return [self.v_l_next, self.v_r_next, self.v_l_next_possibility, self.v_r_next_possibility]

    def gen_random_speed_next_based_on_distribution(self):
        # self.clear_speed_next()
        # v_*_next are cleared
        # print("The speed_next is cleared.")
        method = self.v_l_distribution['method']
        kwargs = {'low_l': self.v_l_distribution['low'], 'high_l': self.v_l_distribution['high'],
                  'low_r': self.v_r_distribution['low'], 'high_r': self.v_r_distribution['high'],
                  'mean_l': self.v_l_distribution['mean'], 'std_l': self.v_l_distribution['std'],
                  'mean_r': self.v_r_distribution['mean'], 'std_r': self.v_r_distribution['std']}
        self.gen_random_speed_next(self.predict_horizon, method, **kwargs)

    def random_move(self, steps=1, method="uniform", **kwargs):
        output = []
        # kwargs = {'low_l': [], 'high_l': [], 'low_r': [], 'high_r': []} or {'mean_l': [], 'std_l': [], 'mean_r': [], 'std_r': []}
        if not self.is_empty_speed_next():
            self.clear_speed_next()
            print("The speed_next is not empty, it has been cleared.")
        self.gen_random_speed_next(steps=steps, method=method, **kwargs)
        self.move_steps(steps)
        return output

    def random_predict_move(self, steps=1, method="uniform", **kwargs):
        if not self.is_empty_speed_next():
            self.clear_speed_next()
            print("The speed_next is not empty, it has been cleared.")
        self.gen_random_speed_next(steps, method, **kwargs)
        return self.predict_steps(steps)

    def plot_robot(self, ax=None, **kwargs):
        # plot a rectangle with the center at (self.x, self.y) and angle self.theta, width self.width, length self.length
        # kwargs: "clear" to clear the plot, "show" to show the plot, "save" to save the plot, "name" to specify the name of the plot, "x_lim" to set the x limit, "y_lim" to set the y limit
        if ax is None:
            fig, ax = plt.subplots()
        if kwargs.get("clear", False):
            ax.clear()
        x = self.x
        y = self.y
        theta = self.theta
        width = self.width
        length = self.length
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        dw_cos = width / 2 * cos_theta
        dw_sin = width / 2 * sin_theta
        dl_cos = length / 2 * cos_theta
        dl_sin = length / 2 * sin_theta
        x1 = x + dl_cos - dw_sin
        y1 = y + dl_sin + dw_cos
        x2 = x - dl_cos - dw_sin
        y2 = y - dl_sin + dw_cos
        x3 = x - dl_cos + dw_sin
        y3 = y - dl_sin - dw_cos
        x4 = x + dl_cos + dw_sin
        y4 = y + dl_sin - dw_cos
        ax.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1], color=self.color)
        ax.scatter(x, y, color=self.color)
        ax.set_aspect('equal')
        ax.grid(True)
        if kwargs.get("show", False):
            plt.show()
        if kwargs.get("save", False):
            name = kwargs.get("name", "robot.png")
            plt.savefig(name)
        if kwargs.get("x_lim", False):
            x_lim = kwargs.get("x_lim")
            ax.set_xlim(x_lim)
        if kwargs.get("y_lim", False):
            y_lim = kwargs.get("y_lim")
            ax.set_ylim(y_lim)
        return ax

    def plot_history(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        x = [i[0] for i in self.history]
        y = [i[1] for i in self.history]
        ax.plot(x, y, color=self.color)
        ax.scatter(self.x, self.y, color=self.color)
        return ax

    def plot_ref_trajectory(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.ref_trajectory[:, 0],
                self.ref_trajectory[:, 1], color=self.color)
        ax.scatter(self.x, self.y, color=self.color)
        return ax

    def plot_velocity(self, ax=None, **kwargs):
        # plot the velocity and angular velocity of the robot
        # kwargs: "clear" to clear the plot, "show" to show the plot, "save" to save the plot, "name" to specify the name of the plot
        if ax is None:
            fig, ax = plt.subplots()
        if kwargs.get("clear", False):
            ax.clear()
        v_l = [state[3] for state in self.history]
        v_r = [state[4] for state in self.history]
        ax.plot(v_l, color=self.color, linestyle='-', label='v_l')
        ax.plot(v_r, color=self.color, linestyle='--', label='v_r')
        ax.grid(True)
        ax.legend()
        return ax

    def plot_velocity_and_omega(self, ax=None, **kwargs):
        # plot the velocity and angular velocity of the robot
        # kwargs: "clear" to clear the plot, "show" to show the plot, "save" to save the plot, "name" to specify the name of the plot
        if ax is None:
            fig, ax = plt.subplots()
        if kwargs.get("clear", False):
            ax.clear()
        v = [(state[3]+state[4])/2 for state in self.history]
        omega = [(state[4]-state[3])/self.L for state in self.history]
        ax.plot(v, color=self.color, label='v')
        ax.plot(omega, color=self.color, linestyle='--', label='omega')
        ax.grid(True)
        ax.legend()
        return ax

    def gen_random_trajectory(self):
        # kwargs = {'low_l': [], 'high_l': [], 'low_r': [], 'high_r': []} or {'mean_l': [], 'std_l': [], 'mean_r': [], 'std_r': []}
        steps = self.v_l_distribution['horizon']
        if steps == 0:
            print("please set the v_l_distribution and v_r_distribution")
            return None
        # method = self.v_l_distribution['method']
        # kwargs = {'low_l': self.v_l_distribution['low'], 'high_l': self.v_l_distribution['high'],
        #           'low_r': self.v_r_distribution['low'], 'high_r': self.v_r_distribution['high'],
        #           'mean_l': self.v_l_distribution['mean'], 'std_l': self.v_l_distribution['std'],
        #           'mean_r': self.v_r_distribution['mean'], 'std_r': self.v_r_distribution['std']}
        # self.gen_random_speed_next(steps, method, **kwargs)
        self.gen_random_speed_next_based_on_distribution()
        trajectory, p_l, p_r = self.predict_steps(steps)
        p_trajectory = np.prod(p_l) * np.prod(p_r)
        # self.random_trajectories = trajectory
        # self.possibility_trajectories = p_trajectory
        return trajectory, p_trajectory

    def gen_random_trajectories(self):
        # generate the random trajectories based on the distribution of the speed
        # the number of the trajectories is self.random_trajectory_num
        self.random_trajectories = []
        self.possibility_trajectories = []
        self.random_inputs = []
        for _ in range(self.random_trajectory_num):
            trajectory, p_trajectory = self.gen_random_trajectory()
            self.random_trajectories.append(trajectory)
            self.possibility_trajectories.append(p_trajectory)
            self.random_inputs.append([self.v_l_next, self.v_r_next])

    # still need to be implemented, not sure whether it is useful
    def initial_message(self):
        # the message is set to 1 before the message passing
        self.gen_random_trajectories()
        for neighbor in self.neighbors.values():
            self.messages_tobe_sent[neighbor[0].id] = [1.0] * self.random_trajectory_num

    # still need to be implemented, not sure whether it is useful
    def send_receive_messages(self):
        # in the simulation, the robots could get the messages directly without communication through the network
        # only one function is needed
        # if the neighbors are not empty, send the message to the neighbors
        # the massage includes:
        # 1. sampled trajectories
        # 2. the messages from the neighbors except the current robot
        # supposing the message is from j to i, the i-th robot will sotore the message in robot_i.neighbors[j][2], and the trajectory in robot_i.neighbors[j][1]
        for neighbor_item in self.neighbors.values():
            neighbor_item[0].neighbors[self.id][1] = copy.copy(self.random_trajectories)
            neighbor_item[0].neighbors[self.id][2] = copy.copy(self.possibility_trajectories)
            # neighbor.neighbors[self.id][2] = self.message_tobe_sent[neighbor.id]
            # self.message_tobe_sent[neighbor.id] is a list, the length is the number of the trajectories, the element is \prod_{k \in N(j) \ i} m_{k,j}(\tau_j^l), it will be calculated in self.update_message_tobe_sent()
            neighbor_item[0].neighbors[self.id][3] = copy.copy(self.ref_trajectory)
            neighbor_item[0].messages_been_received.update(
                {self.id: copy.copy(self.messages_tobe_sent[neighbor_item[0].id])})

    def update_message_tobe_sent(self):
        # update the message to be sent to the neighbors
        # the size of self.message_tobe_sent is self.neighbors_num * self.random_trajectory_num
        for i in range(self.random_trajectory_num):
            for rb_neighbor1 in self.neighbors.values():
                j = rb_neighbor1[0].id
                messages_neighbors_except_i = []
                for message_item in self.messages_been_received.items():
                    if message_item[0] != j:
                        messages_neighbors_except_i.append(message_item[1])
                # messages_neighbors_except_i = [self.neighbors[k][2] for k in range(self.neighbors_num) if self.neighbors[k][0].id != j]
                # self.neighbors: a list [robot, random_trajectories, p_trajectories, ref_trajectory, delta_x, delta_y]，
                self.messages_tobe_sent[j][i] = message_process(self.random_trajectories[i], self.neighbors[j][1], self.neighbors[j][2], self.neighbors[j][3], messages_neighbors_except_i, self.neighbors[j][0], self.neighbors[j][4], self.neighbors[j][5])
        # unit the message to be sent, the average of the messages is 1.0
        for i in self.messages_tobe_sent.keys():
            self.messages_tobe_sent[i] = self.messages_tobe_sent[i] / np.sum(self.messages_tobe_sent[i]) * self.random_trajectory_num

    def weight_trajectory_private_and_neighbors(self, trajectory, ref_trajectory, obstacles, trajectory_index):
        '''
        ### $C(\\tau_i)$ in the paper
        ### C(\\tau_i) = -log p(O_i | \\tau_i) - \\sum_{j \\in N(i)} log m_{j,i}(\\tau_i)
        ### the next calculation is based on exp(-C(\\tau_i))
        ### the function will return the exp(-C(\\tau_i)), which is p(O_i | \\tau_i) * \\prod_{j \\in N(i)} m_{j,i}(\\tau_i)
        ### costs, exponents = p_optimal_given_trajectory(trajectory, ref_trajectory, obstacles)
        '''
        costs, _ = p_optimal_given_trajectory(trajectory, ref_trajectory, obstacles)
        messages = []
        for message in self.messages_been_received.items():
            messages.append(message[1][trajectory_index])
        prod_messages = np.prod(messages)
        return np.prod(costs) * prod_messages

    def weight_trajectories_private_and_neighbors(self, trajectories, ref_trajectory, obstacles):
        # for each trajectory in trajectories, calculate the weight of the trajectory
        # and then unite the weights of the inputs
        weights_list = []
        for i in range(self.random_trajectory_num):
            weight_single_trajectory = self.weight_trajectory_private_and_neighbors(
                trajectories[i], ref_trajectory, obstacles, i)
            weights_list.append(weight_single_trajectory)
        # unite the weights of the inputs
        weights_list = np.array(weights_list)
        return weights_list / np.sum(weights_list)

    def update_distribution_of_U(self):
        weights_list = self.weight_trajectories_private_and_neighbors(
            self.random_trajectories, self.ref_trajectory, self.obstacles)
        # update the distribution of the inputs based on the weights_matrix
        # 1.self.random_inputs:         [[v_l_next0, v_r_next0], ...]   length is self.random_trajectory_num
        # 2.self.random_trajectories:   [trajectory0, ...]              length is self.random_trajectory_num
        # 3.weights_matrix:             np.ndarray((self.random_trajectory_num, self.predict_horizon))
        # the length of v_l_next0, trajectory0, weights_array are the same, the predict horizon.
        # \bar{U}^{m+1} = \sum_{l=1}^{L} w_l^m u_l^l
        mean_l = []
        mean_r = []
        std_l = []
        std_r = []
        random_inputs = np.array(self.random_inputs) # shape: (self.random_trajectory_num, 2, self.predict_horizon)
        for i in range(self.predict_horizon):
            v_l_all = random_inputs[:, 0, i]
            v_r_all = random_inputs[:, 1, i]
            # plt.plot(v_l_all, weights_list, '.')
            # plt.plot(v_r_all, weights_list, '.')
            # plt.show()
            mean_l.append(np.sum(v_l_all * weights_list))
            mean_r.append(np.sum(v_r_all * weights_list))
            # std_l.append(0.3 * np.sqrt(np.sum((v_l_all-mean_l[i]) ** 2 * weights_list)))
            # std_r.append(0.3 * np.sqrt(np.sum((v_r_all-mean_r[i]) ** 2 * weights_list)))
            # print(mean_l[i], mean_r[i], std_l[i], std_r[i])
            std_l.append(default_std)
            std_r.append(default_std)
        self.v_l_distribution["mean"] = mean_l
        self.v_r_distribution["mean"] = mean_r
        
        self.v_l_distribution["std"] = std_l
        self.v_r_distribution["std"] = std_r

        # for i in range(self.v_l_distribution["horizon"]):
        #     v_l_all = np.array([self.random_inputs[j][0][i]
        #                        for j in range(self.random_trajectory_num)])
        #     v_r_all = np.array([self.random_inputs[j][1][i]
        #                        for j in range(self.random_trajectory_num)])
        #     ax = plt.gcf().add_subplot(2, 2, 1)
        #     ax.plot(v_l_all, weights_matrix[:, i], '.')
        #     # plt.show()
        #     ax = plt.gcf().add_subplot(2, 2, 3)
        #     # plot the histogram of the v_l_all
        #     ax.hist(v_l_all, bins=20, weights=weights_matrix[:, i])
        #     # plt.show()
        #     ax = plt.gcf().add_subplot(2, 2, 4)
        #     # plot the histogram of the v_l_all
        #     ax.hist(v_r_all, bins=20, weights=weights_matrix[:, i])
        #     plt.show()
        #     mean_l.append(np.sum(v_l_all * weights_matrix[:, i]))
        #     mean_r.append(np.sum(v_r_all * weights_matrix[:, i]))
        #     std_l.append(default_std)
        #     std_r.append(default_std)
            # std_l.append(np.sqrt(np.sum((v_l_all-mean_l[i]) ** 2 * weights_matrix[:, i])))
            # std_r.append(np.sqrt(np.sum((v_r_all-mean_r[i]) ** 2 * weights_matrix[:, i])))
        # self.v_l_distribution["mean"] = mean_l
        # self.v_r_distribution["mean"] = mean_r
        # self.v_l_distribution["std"] = std_l
        # self.v_r_distribution["std"] = std_r


class Obstacle:
    def __init__(self, x:float, y:float, r:float):
        self.x = x
        self.y = y
        self.r = r
        self.safe_distance = 0.5

    def __str__(self):
        return "x: %.2f,\t y: %.2f,\t r: %.2f" % (self.x, self.y, self.r)

    def cost(self, x, y):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        distance = np.sqrt((x-self.x)**2 + (y-self.y)**2) - self.r - self.safe_distance
        # count the nagative values in the distance
        count = len(distance[distance < 0])
        # In the paper, the cost is 10.
        # A function of distance will be tried in the future.
        return 10*count
    
    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        circle = plt.Circle((self.x, self.y), self.r, color='k')
        ax.add_artist(circle)
        return ax
    
    def get_parameters(self):
        return {"x": self.x, "y": self.y, "r": self.r, "safe_distance": self.safe_distance}


def cost_position(trajectory, ref_trajectory):
    if not isinstance(trajectory, np.ndarray):
        trajectory = np.array(trajectory)
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    theta = trajectory[:, 2]
    x_ref = ref_trajectory[:, 0]
    y_ref = ref_trajectory[:, 1]
    theta_ref = ref_trajectory[:, 2]
    v_l = trajectory[:, 3]
    v_r = trajectory[:, 4]
    # v_l = trajectory[:, 3] - ref_trajectory[:, 3]
    # v_r = trajectory[:, 4] - ref_trajectory[:, 4]
    cost = 0.5 * (w_position*np.sum((x-x_ref)**2 + (y-y_ref)**2) + w_theta *
                  np.sum((theta-theta_ref)**2) + w_inputs*np.sum(v_l**2 + v_r**2))
    return cost


def cost_obstacle(trajectory, obstacles:list[Obstacle]):
    if not isinstance(trajectory, np.ndarray):
        trajectory = np.array(trajectory)
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    # theta = trajectory[:, 2]
    # v_l = trajectory[:, 3]
    # v_r = trajectory[:, 4]
    cost = 0
    for obstacle in obstacles:
        cost += obstacle.cost(x, y)
    if cost > 0:
        aaa=1
    return cost


def p_optimal_given_x_u(states, ref_states, obstacles):
    '''
    $p(\bm{O}_{i,t} | x_{i,t}, u_{i,t})$ in the paper
    states: np.ndarray, [x, y, theta, v_l, v_r]
    ref_states: np.ndarray, [x, y, theta, v_l, v_r]
    obstacles: [Obstacle, Obstacle, ...]
    '''
    exponent = 0
    # trajectory: np.ndarray, ref_trajectory: np.ndarray, obstacles: [Obstacle, Obstacle, ...]
    if not isinstance(states, np.ndarray):
        states = np.array([states])
    elif len(states.shape) == 1:
        states = np.array([states])
    if not isinstance(ref_states, np.ndarray):
        ref_states = np.array([ref_states])
    elif len(ref_states.shape) == 1:
        ref_states = np.array([ref_states])
    cost_p = cost_position(states, ref_states)
    cost_o = cost_obstacle(states, obstacles)
    exponent = -(cost_p + cost_o)/lambda_u
    cost = np.exp(exponent)
    return cost, exponent


def p_optimal_given_trajectory(trajectory, ref_trajectory, obstacles):
    '''
    states: np.ndarray, [[x, y, theta, v_l, v_r], ...]
    ref_states: np.ndarray, [[x, y, theta, v_l, v_r], ...]
    obstacles: [Obstacle, Obstacle, ...]
    '''
    costs = []
    exponents = []
    for states, ref_states in zip(trajectory, ref_trajectory):
        cost, exponent = p_optimal_given_x_u(states, ref_states, obstacles)
        costs.append(cost)
        exponents.append(exponent)
    return costs, exponents

# def unary_potential(trajectories, ref_trajectories, p_trajectories, obstacles):
#     '''
#     $\phi(\tau,\bm{O}) in the paper$
#     result is exp(-1/lambda_u* \sum cost) * \prod p(u_{i, t})
#     also write as $p(O_{i,t} | \bm{x}_{i,t}, \bm{u}_{i,t})$ in the paper
#     $\bm{x}_{i,t}$ is the state of the i-th robot at time $t$
#     $\bm{u}_{i,t}$ is the control input of the i-th robot at time $t$
#     $O_{i,t}$ is the observation of the i-th robot at time $t$
#     this is a function to evaluate the distribution of the optimal likehood of the observation given the state and the control input
#     In fact, the function is the product of trajectories' cost of function p_optimal_given_trajctory, and multiply the possibility of the trajectories. But to reduce the calculation, product of a seirers of exponentials is replaced by the exponential of the sum of the cost.
#     The function p_optimal_given_trajctory is needed in other places.
#     '''
#     exponent = 0
#     if isinstance(trajectories, np.ndarray):
#         # if the trajectories is a single trajectory, convert it to a list that only contains one trajectory
#         trajectories = [trajectories]
#     elif isinstance(ref_trajectories, list):
#         # if the ref_trajectories is a list, judge whether the elements in the list are numpy.ndarray
#         # [np.ndarray, np.ndarray, ...] -> right
#         if not isinstance(ref_trajectories[0], np.ndarray):
#             # [[float, float, ...], [float, float, ...], ...] -> [np.ndarray, np.ndarray, ...]
#             ref_trajectories = np.array(ref_trajectories)
#     for trajectory, ref_trajectory in zip(trajectories, ref_trajectories):
#         cost_p = cost_position(trajectory, ref_trajectory)
#         cost_o = cost_obstacle(trajectory, obstacles)
#         exponent += cost_p + cost_o
#     cost = np.exp(-exponent/lambda_u) * np.prod(p_trajectories)
#     return cost


def unary_potential(trajectory, ref_trajectory, p_trajectory, obstacles):
    '''
    $\phi(\tau,\bm{O}) in the paper$
    result is exp(-1/lambda_u* \sum cost) * \prod p(u_{i, t})
    also write as $p(O_{i,t} | \bm{x}_{i,t}, \bm{u}_{i,t})$ in the paper
    $\bm{x}_{i,t}$ is the state of the i-th robot at time $t$
    $\bm{u}_{i,t}$ is the control input of the i-th robot at time $t$
    $O_{i,t}$ is the observation of the i-th robot at time $t$
    this is a function to evaluate the distribution of the optimal likehood of the observation given the state and the control input
    In fact, the function is the product of trajectories' cost of function p_optimal_given_trajctory, and multiply the possibility of the trajectories. But to reduce the calculation, product of a seirers of exponentials is replaced by the exponential of the sum of the cost.
    The function p_optimal_given_trajctory is needed in other places.
    '''
    if isinstance(trajectory, np.ndarray):
        trajectory = np.array(trajectory)
    cost_p = cost_position(trajectory, ref_trajectory)
    cost_o = cost_obstacle(trajectory, obstacles)
    exponent = -(cost_p + cost_o) / lambda_u
    cost = np.exp(exponent) * np.prod(p_trajectory)
    return cost, exponent


def cost_keep_distance(trajectory_i, trajectory_j):
    if not isinstance(trajectory_i, np.ndarray):
        trajectory_i = np.array(trajectory_i)
    if not isinstance(trajectory_j, np.ndarray):
        trajectory_j = np.array(trajectory_j)
    x_i = trajectory_i[:, 0]
    y_i = trajectory_i[:, 1]
    x_j = trajectory_j[:, 0]
    y_j = trajectory_j[:, 1]
    distance = np.sqrt((x_i-x_j)**2 + (y_i-y_j)**2)
    # find the distance less than d_min, C_l=w_l*(1-(d/d_min)^b) if d<d_min, 0 otherwise
    # w_l=10, b=2
    dist_d_min = distance/d_min
    # delete the values greater than 1
    # dist_d_min = dist_d_min[dist_d_min <= 1]
    cost_l = 0
    for dist_d_min_i in dist_d_min:
        if dist_d_min_i < 1:
            cost_l += w_l * (1 - dist_d_min_i**beta_rate)
    return cost_l


def cost_formations(trajectory_i, trajectory_j, delta_x, delta_y):
    if not isinstance(trajectory_i, np.ndarray):
        trajectory_i = np.array(trajectory_i)
    if not isinstance(trajectory_j, np.ndarray):
        trajectory_j = np.array(trajectory_j)
    x_ij = trajectory_j[:, 0] - trajectory_i[:, 0]
    y_ij = trajectory_j[:, 1] - trajectory_i[:, 1]
    x_ij = x_ij - delta_x
    y_ij = y_ij - delta_y
    cost_f = 0.5 * w_f * np.sum(x_ij**2 + y_ij**2)
    return cost_f


def pairwise_potential(trajectory_i, trajectory_j, delta_x, delta_y):
    cost_l = cost_keep_distance(trajectory_i, trajectory_j)
    cost_f = cost_formations(trajectory_i, trajectory_j, delta_x, delta_y)
    exponent = -1/lambda_p*(cost_l+cost_f)
    cost = np.exp(exponent)
    return cost, exponent


def message_process(trajectory_i, trajectories_j, p_trajectiories_j, ref_trajectory_j, messages_neighbors_except_i, robot_j: Robot, delta_x_ij, delta_y_ij):
    '''
    ### \hat{m}_{j,i}(\tau_{i}^s)
    trajectory_i = $\tau_i^s$
    trajectories_j = $\tau_j^l$ for l in 1, 2, ..., L, L is the number of the trajectories sampled by the j-th robot, for trajectory_j in trajectories_j, trajectory_j = $\tau_j^l$
    ref_trajectories_j is the reference trajectories of the j-th robot, for ref_trajectory_j in ref_trajectories_j, ref_trajectory_j = $\hat{\tau}_j^l$
    messages_neighbors_except_i is the messages from the neighbors of the j-th robot except the i-th robot
    they are calculated by the function message_process (this function)
    ### the importance of the i-th robot's s-th trajectory, and the result will be sent to the j-th robot
    ### thus, this function is used to update the importance of the i-th robot's s-th trajectory for the j-th robot, the trajectories of the j-th robot (trajectories_j in the function, and \tau_j^l in the paper) are considered 
    ### message_neighbors is the messages from the neighbors of the j-th robot except the i-th robot
    ### these messages are sent from the j-th robot to the i-th robot
    '''
    mean_potential = 0
    prod_messages = np.array(messages_neighbors_except_i)
    for trajectory_j, p_trajectory_j, i in zip(trajectories_j, p_trajectiories_j, range(robot_j.random_trajectory_num)):
        factor_1, _ = unary_potential(
            trajectory_j, ref_trajectory_j, p_trajectory_j, robot_j.obstacles)
        factor_2, _ = pairwise_potential(
            trajectory_i, trajectory_j, delta_x=delta_x_ij, delta_y=delta_y_ij)
        # print(p_trajectory_j * np.prod(message_neighbors_except_i))
        # if p_trajectory_j * np.prod(message_neighbors_except_i) < 1e-6:
            # a=1
        # mean_potential += factor_1 * factor_2 / p_trajectory_j * np.prod(prod_messages[:, i])
        mean_potential += factor_1 * factor_2 * np.prod(prod_messages[:, i])
        # mean_potential += factor_1 * factor_2 * np.prod(message_neighbors_except_i)
    return mean_potential / len(trajectories_j)


class MapForRobots:
    def __init__(self, obstacles):
        # obstacles : [Obstacle, Obstacle, ...]
        self.obstacles = obstacles


class DistributeRobotSystem:
    def __init__(self, robots: list[Robot]):
        self.robots = robots
        self.robots_num = len(robots)
        self.formations = []    # the x,y of the robots in the reference coordinate system
        # the topology of the robots, the i-th element is the neighbors of the i-th robot
        self.topology = []
        self.prediction_horizon = 10
        self.random_trajectory_num = 200
        self.delta_t = 0.1
        self.random_method = "gaussian"
        self.ref_trajectory = np.zeros((self.prediction_horizon, 5))
        self.distances = {}
        self.ref_distances = {}
        self.err_distances = {}

    def set_formation(self, formation: list[list[float]]):
        # formations: [[x1, y1], [x2, y2], ...]
        # or the np.ndarray with the shape of (n, 2)
        if not isinstance(formation, np.ndarray):
            formation = np.array(formation)
        self.formations = formation
        # calculate the referece distances between the robots
        for i in range(self.robots_num):
            for j in range(i+1, self.robots_num):
                self.ref_distances[(i, j)] = np.sqrt(
                    (formation[i][0]-formation[j][0])**2 + (formation[i][1]-formation[j][1])**2)
                
    def calculate_distances(self):
        for i in range(self.robots_num):
            for j in range(i+1, self.robots_num):
                self.distances[(i, j)] = np.sqrt(
                    (self.robots[i].x-self.robots[j].x)**2 + (self.robots[i].y-self.robots[j].y)**2)
                self.err_distances[(i, j)] = self.distances[(i, j)] - self.ref_distances[(i, j)]

    def set_topology(self, topology: dict[int, list[int]]):
        '''
        # topology: {0: [1, 2], 1: [0, 2], 2: [0, 1]}
        the key is the id of the robot, the value is the list of the ids of the neighbors
        this is a undirected graph
        '''
        self.topology = topology
        # set the neighbors of the robots based on the topology
        for i, neighbors_id in topology.items():
            for neighbor_id in neighbors_id:
                # {neighbor_id: [neighbot_address*, random_trajectories, p_trajectories, ref_trajectory, delta_x*, delta_y*]}
                # _* means the value is updated
                self.robots[i].neighbors.update({neighbor_id: [self.robots[neighbor_id], np.array([0]), np.array([0]), np.array([0]), self.robots[neighbor_id].x-self.robots[i].x, self.robots[neighbor_id].y-self.robots[i].y]})
                self.robots[i].neighbors_num += 1
        # return self.topology

    def initial_robots(self):
        for robot in self.robots:
            robot.delta_t = self.delta_t
            robot.v_l_distribution["horizon"] = self.prediction_horizon
            robot.v_r_distribution["horizon"] = self.prediction_horizon
            robot.v_l_distribution["method"] = self.random_method
            robot.v_r_distribution["method"] = self.random_method
            robot.random_trajectory_num = self.random_trajectory_num
            robot.predict_horizon = self.prediction_horizon
            robot.re_initial()
            robot.initial_message()
            # initial robot.neighbors: [robot, random_trajectories*, p_trajectories*, ref_trajectory*, delta_x, delta_y]
            # * means the value is updated
            for neighbor in robot.neighbors.values():
                neighbor[1] = neighbor[0].random_trajectories
                neighbor[2] = neighbor[0].possibility_trajectories
                neighbor[3] = neighbor[0].ref_trajectory
            print(robot.name + " has been intialized.")

    def set_ref_trajectory(self, ref_trajectory):
        '''
        set the reference trajectory of the robots based on ref_trajectory and the formations
        ref_trajectory: [[x1, y1, theta1, v_l1, v_r1], [x2, y2, theta2, v_l2, v_r2], ...]
        the length of the ref_trajectory is self.prediction_horizon
        The algorithm in the paper seems unsupportive for rotation of the formation, 
        the theta_i here is the tangent of the trajectory at the i-th point,
        also the same as theta in every robot
        '''

        self.ref_trajectory = ref_trajectory
        if not isinstance(ref_trajectory, np.ndarray):
            ref_trajectory = np.array(ref_trajectory)
        for i, robot in enumerate(self.robots):
            for m in range(self.prediction_horizon):
                robot.ref_trajectory[m, :] = ref_trajectory[m, :]+np.array(
                    [self.formations[i][0], self.formations[i][1], 0, 0, 0])

    def VI_MPC_BP_algorithm(self):
        pass

    def save_all_data(self,filename):
        # save the data of the robots in filename as .xlsx: robot.history for each robot
        if not filename.endswith(".xlsx"):
            filename += ".xlsx"
        writer = pd.ExcelWriter(filename)
        for robot in self.robots:
            df = pd.DataFrame(robot.history)
            df.to_excel(writer, sheet_name=robot.name)
        writer.close()

def save_parameters(filename, other_parameters={}):
    # save the parameters in the file filename as .json
    # the parameters are the global variables in the file
    if not filename.endswith(".json"):
        filename += ".json"
    parameters = {
        "w_position": w_position,
        "w_theta": w_theta,
        "w_inputs": w_inputs,
        "w_l": w_l,
        "w_f": w_f,
        "lambda_u": lambda_u,
        "lambda_p": lambda_p,
        "d_min": d_min,
        "beta_rate": beta_rate,
        "default_std": default_std
    }
    parameters.update(other_parameters)
    with open (filename, 'w') as f:
        json.dump(parameters, f)

