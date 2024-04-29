from src.robot import Robot
from src.robot import DistributeRobotSystem
from src.robot import save_parameters
from src.robot import Obstacle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

simulation_NO = 1

R1 = Robot(x=1.0, y=1.0, theta=0.0, color='b')
R2 = Robot(x=-1.0, y=1.0, theta=0.0, color='r')
R3 = Robot(x=-1.0, y=-1.0, theta=0.0, color='g')
R4 = Robot(x=1.0, y=-1.0, theta=0.0, color='y')

O1 = Obstacle(x=2.5, y= 1.58, r=0.2)
O2 = Obstacle(x=4.5, y=-1.58, r=0.2)
obstacles = [O1, O2]

R1.obstacles.extend(obstacles)
R2.obstacles.extend(obstacles)
R3.obstacles.extend(obstacles)
R4.obstacles.extend(obstacles)

print(R1.id, R2.id, R3.id, R4.id)
v_max = 0.5
a_max = 1.5
delta_t = 0.1
v = v_max

prediction_horizon = 20
random_trajectory_num = 200
simulation_N = 150
RobotSystem = DistributeRobotSystem(robots=[R1, R2, R3, R4])
RobotSystem.prediction_horizon = prediction_horizon
RobotSystem.random_trajectory_num = random_trajectory_num
robtos_formation = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]
# robots_topology = {0: [1, 2, 3], 1: [0, 2, 3], 2: [0, 1, 3], 3: [0, 1, 2]}
robots_topology = {0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [0, 2]}
RobotSystem.set_formation(formation=robtos_formation)
RobotSystem.set_topology(topology=robots_topology)
RobotSystem.initial_robots()

ax_robot = plt.gcf().add_subplot(2, 2, 1)

ax_v = plt.gcf().add_subplot(2, 2, 2)
ax_vl_disribution = plt.gcf().add_subplot(2, 2, 3)

ax_distance = plt.gcf().add_subplot(2, 2, 4)
distance_error_history = []


for rb in RobotSystem.robots:
    rb.plot_robot(ax=ax_robot)


ref_tr_system = [[j*v*delta_t, 0, 0, v, v]
                 for j in range(prediction_horizon)]
RobotSystem.set_ref_trajectory(ref_trajectory=ref_tr_system)
for rb in RobotSystem.robots:
    rb.gen_distribution_based_on_ref_trajectory()

#     R1.plot_history(ax_history_tr)
flag1 = 300
x0 = 0
for i in tqdm(range(simulation_N)):
    x0 = x0 + v*delta_t
    ref_tr_system = [[(j+i+1)*v*delta_t, 0, 0, v, v] for j in range(prediction_horizon)]
    RobotSystem.set_ref_trajectory(ref_trajectory=ref_tr_system)
    for rb in RobotSystem.robots:
        rb.gen_random_trajectories()

    for rb in RobotSystem.robots:
        rb.send_receive_messages()
    for rb in RobotSystem.robots:
        rb.update_message_tobe_sent()

    # calculate the distance between robots

    ax_robot.clear()
    ax_v.clear()
    ax_vl_disribution.clear()
    for rb in RobotSystem.robots:
        rb.update_distribution_of_U()
        rb.v_l_next.insert(0, rb.v_l_distribution["mean"][0])
        rb.v_r_next.insert(0, rb.v_r_distribution["mean"][0])
        rb.v_l_next_possibility.insert(0, 1.0)
        rb.v_r_next_possibility.insert(0, 1.0)
        rb.move()
        rb.plot_robot(ax=ax_robot)
        rb.plot_ref_trajectory(ax=ax_robot)
        rb.plot_history(ax=ax_robot)
        ax_robot.title.set_text("Robots\s trajectories")
        rb.plot_velocity(ax=ax_v)
        ax_v.title.set_text("Velocity")
        ax_vl_disribution.plot(rb.v_l_distribution["mean"], color=rb.color)
        ax_vl_disribution.plot(
            rb.v_l_distribution["std"], linestyle='--', color=rb.color)
        ax_vl_disribution.grid(True)
        ax_vl_disribution.title.set_text("Velocity distribution")
        rb.gen_random_trajectories()
    for ob in obstacles:
        ob.plot(ax=ax_robot)
    ax_robot.set_xlim(-2, 9.5)
    ax_robot.set_ylim(-3.5, 3.5)
    RobotSystem.calculate_distances()
    ax_distance.clear()
    distance_error_history.append(
        [item[1] for item in RobotSystem.err_distances.items()])
    distance_error_history_array = np.array(distance_error_history)
    clms = distance_error_history_array.shape[1]
    for j in range(clms):
        ax_distance.plot(distance_error_history_array[:, j])
    ax_distance.plot(distance_error_history_array)
    ax_distance.grid()
    ax_distance.title.set_text("Distance error history")
    plt.pause(0.01)
plt.show()
print("Simulation is done, saving data...")
RobotSystem.save_all_data(filename="./data/simu"+str(simulation_NO)+"_data")
save_parameters(filename="./data/simu"+str(simulation_NO)+"_parameters", other_parameters={"v": v, "delta_t": delta_t, "prediction_horizon": prediction_horizon, "random_trajectory_num": random_trajectory_num,
                "simulation_N": simulation_N, "robtos_formation": robtos_formation, "robots_topology": robots_topology, "Obstacle": [obstacle.get_parameters() for obstacle in obstacles]})

print("Data is saved.")
