"""
Kohei Honda, 2023.
"""
from __future__ import annotations
# add for jupyter notebook visualization.
from IPython.display import display, clear_output 
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from typing import Tuple, Union
from matplotlib import pyplot as plt
import matplotlib.patches as patches

import torch
import numpy as np
import os
import yaml

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from src.envs.obstacle_map_2d import ObstacleMap, generate_random_obstacles

from src.envs.lane_map_2d import LaneMap
from src.envs.circuit_generator.path_generate import make_side_lane, make_csv_paths, make_csv_simple_path

from src.dynamics.bicycle_like_dynamics import Bicycle_Dynamics

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
setting_path = os.path.join(BASE_DIR, 'setting.yaml')
setting_csv_path = os.path.join(BASE_DIR, 'src/envs/circuit_generator/path.csv') # circuit
with open(setting_path) as f:
    param = yaml.full_load(f)

@torch.jit.script
def angle_normalize(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi


class Navigation2DEnv:
    def __init__(
        self, device=torch.device("cuda"), dtype=torch.float32, seed: int = 42
    ) -> None:
        # device and dtype
        if torch.cuda.is_available() and device == torch.device("cuda"):
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        self._dtype = dtype
        
        self.map_size = (param['map_size'], param['map_size'])
        self.cell_size=0.1
        self._obstacle_map = ObstacleMap(
            map_size=self.map_size, cell_size=self.cell_size, device=self._device, dtype=self._dtype, detect_range=param['range']
        )
        self._seed = seed

        # racing_center_path, _, _ = make_csv_paths(setting_csv_path)
        racing_center_path = make_csv_simple_path(setting_csv_path)
        self.line_width = 6.5
        self.right_lane, self.left_lane = make_side_lane(racing_center_path, lane_width=self.line_width)
        # numpy array to tensor
        self.racing_center_path = torch.tensor(racing_center_path, device=self._device, dtype=self._dtype)

        self._lane_map = LaneMap(
            lane=racing_center_path,
            lane_width=self.line_width*0.8,
            map_size=self.map_size,
            cell_size=self.cell_size,
            device=self._device,
            dtype=self._dtype,
        )

        self._bicycle_dynamics = Bicycle_Dynamics()

        generate_random_obstacles(
            obstacle_map=self._obstacle_map,
            random_x_range=(-(param['map_size']/2-4), (param['map_size']/2-4)),
            random_y_range=(-(param['map_size']/2-4), (param['map_size']/2-4)),
            num_circle_obs=param['num_circle_obs'],
            radius_range=(1, 5),
            num_rectangle_obs=param['num_rectangle_obs'],
            width_range=(2, 2),
            height_range=(2, 2),
            max_iteration=1000,
            seed=seed,
        )

        self._obstacle_map.convert_to_torch()

        self._start_pos = torch.tensor(
            [self.racing_center_path[0][0], self.racing_center_path[0][1]], device=self._device, dtype=self._dtype
        )
        self._goal_pos = torch.tensor(
            [self.racing_center_path[-1][0], self.racing_center_path[-1][1]], device=self._device, dtype=self._dtype
        )

        self._robot_state = torch.zeros(6, device=self._device, dtype=self._dtype)
        self._robot_state[:2] = self._start_pos
        self._robot_state[2] = angle_normalize(
            torch.atan2(
                self.racing_center_path[2][1] - self._start_pos[1],
                self.racing_center_path[2][0] - self._start_pos[0],
            )
        )
        self._robot_state[3] = 0
        self._robot_state[4] = 0
        self._robot_state[5] = 0

        # u: [v, omega] (m/s, rad/s)
        # self.u_min = torch.tensor(param['u_min'], device=self._device, dtype=self._dtype)
        # self.u_max = torch.tensor(param['u_max'], device=self._device, dtype=self._dtype)
        
        # u: [accel, steer] (m/s2, rad)
        self.u_min = torch.tensor(param['u_min'], device=self._device, dtype=self._dtype)
        self.u_max = torch.tensor(param['u_max'], device=self._device, dtype=self._dtype)
        self.L = torch.tensor(1, device=self._device, dtype=self._dtype)
        self.V_MAX = torch.tensor(param['v_max'], device=self._device, dtype=self._dtype)


    def reset(self) -> torch.Tensor:
        """
        Reset robot state.
        Returns:
            torch.Tensor: shape (5,) [x, y, theta, vel, vel, w]
        """
        self._robot_state[:2] = self._start_pos
        self._robot_state[2] = angle_normalize(
            torch.atan2(
                self.racing_center_path[2][1] - self._start_pos[1],
                self.racing_center_path[2][0] - self._start_pos[0],
            )
        )
        self._robot_state[3] = 0
        self._robot_state[4] = 0
        self._robot_state[5] = 0

        self._fig = plt.figure(layout="tight")
        self._ax = self._fig.add_subplot()
        self._ax.set_xlim(self._obstacle_map.x_lim)
        self._ax.set_ylim(self._obstacle_map.y_lim)
        self._ax.set_aspect("equal")

        self._rendered_frames = []
        

        return self._robot_state

    def step(self, u: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Update robot state based on differential drive dynamics.
        Args:
            u (torch.Tensor): control batch tensor, shape (2) [v, omega]
        Returns:
            Tuple[torch.Tensor, bool]: Tuple of robot state and is goal reached.
        """
        u = torch.clamp(u, self.u_min, self.u_max)

        self._robot_state = self.dynamics(
            state=self._robot_state.unsqueeze(0), action=u.unsqueeze(0)
        ).squeeze(0)

        # goal check
        goal_threshold = 0.5
        is_goal_reached = (
            torch.norm(self._robot_state[:2] - self._goal_pos) < goal_threshold
        )

        return self._robot_state, is_goal_reached

    def render(
        self,
        action: torch.Tensor = None,
        predicted_trajectory: torch.Tensor = None,
        is_collisions: torch.Tensor = None,
        is_robot_collision: torch.Tensor = None,
        top_samples: Tuple[torch.Tensor, torch.Tensor] = None,
        u_ref: torch.Tensor = None,
        mode: str = "human",
    ) -> None:
        self._ax.set_xlabel("x [m]")
        self._ax.set_ylabel("y [m]")
        
        # add or jupyter notebook visualization.
        # if mode == "human": 
        #     self._ax.cla()
        self._ax.cla()
        
        # obstacle map
        self._obstacle_map.render(self._ax, zorder=10)

        # start and goal
        self._ax.scatter(
            self._start_pos[0].item(),
            self._start_pos[1].item(),
            marker="o",
            color="red",
            zorder=10,
        )
        self._ax.scatter(
            self._goal_pos[0].item(),
            self._goal_pos[1].item(),
            marker="o",
            color="orange",
            zorder=10,
        )

        # robot
        robot_x = self._robot_state[0].item()
        robot_y = self._robot_state[1].item()
        robot_theta = self._robot_state[2].item()
        robot_v = self._robot_state[3].item()
        accel = action[0].item()
        steer = action[1].item()

        self._ax.scatter(
            self._robot_state[0].item(),
            self._robot_state[1].item(),
            marker="o",
            color="green",
            zorder=100,
        )
        # robot direction
        self._ax.quiver(
            robot_x,
            robot_y,
            robot_v * np.cos(robot_theta),
            robot_v * np.sin(robot_theta),
            color="green",
            zorder=100,
        )
        # steering direction
        self._ax.quiver(
            robot_x,
            robot_y,
            self.L.cpu() * np.cos(robot_theta + steer),
            self.L.cpu() * np.sin(robot_theta + steer),
            color="blue",
            zorder=100,
        )

        self._ax.plot(
            self.right_lane[:, 0],
            self.right_lane[:, 1],
            color="black",
            linestyle="--",
            zorder=5,
        )
        self._ax.plot(
            self.left_lane[:, 0],
            self.left_lane[:, 1],
            color="black",
            linestyle="--",
            zorder=5,
        )

        # visualize sensory range
        circle = patches.Circle((self._robot_state[0].cpu().numpy(), self._robot_state[1].cpu().numpy()),
            radius=self._obstacle_map._detect_range,
            edgecolor="black",
            facecolor="none",
            linestyle="--",
            linewidth=0.5,
            zorder=100)                         
        self._ax.add_patch(circle)  

        # visualize collision occur
        if is_robot_collision is not None:
            if is_robot_collision[0] > 0:
                circle = patches.Circle((self._robot_state[0].cpu().numpy(), self._robot_state[1].cpu().numpy()),
                    radius=self._obstacle_map._detect_range/2,
                    edgecolor="red",
                    facecolor="red",
                    linewidth=2,
                    zorder=100)                         
                self._ax.add_patch(circle) 

        # display velocity information
        vel_u = self._robot_state[3]
        vel_v = self._robot_state[4]
        vel = (vel_u.pow(2) + vel_v.pow(2)).pow(0.5)
        velocity_disp = f"vel: {vel:.2f}, vel_u: {vel_u:.2f}, vel_v: {vel_v:.2f}"
        # self._ax.text(35, 6, velocity_disp, fontsize=12, color='blue')
        self._ax.set_title(
            f"v: {robot_v:.2f} m/s, accel: {accel:.2f} m/s2, steer: {steer:.2f} rad"
        )
        # visualize top samples with different alpha based on weights
        if top_samples is not None:
            top_samples, top_weights = top_samples
            top_samples = top_samples.cpu().numpy()
            top_weights = top_weights.cpu().numpy()
            top_weights = 0.7 * top_weights / np.max(top_weights)
            top_weights = np.clip(top_weights, 0.1, 0.7)
            for i in range(top_samples.shape[0]):
                self._ax.plot(
                    top_samples[i, :, 0],
                    top_samples[i, :, 1],
                    color="lightblue",
                    alpha=top_weights[i],
                    zorder=1,
                )


        # predicted trajectory
        if predicted_trajectory is not None:

            # if is collision color is red
            colors = np.array(["darkblue"] * predicted_trajectory.shape[1])
            if is_collisions is not None:
                is_collisions = is_collisions.cpu().numpy()
                is_collisions = np.any(is_collisions, axis=0)
                colors[is_collisions] = "red"

            self._ax.scatter(
                predicted_trajectory[0, :, 0].cpu().numpy(),
                predicted_trajectory[0, :, 1].cpu().numpy(),
                color=colors,
                marker="o",
                s=3,
                zorder=2,
            )
    
        inital_state = self._robot_state.unsqueeze(0)
        step_size = predicted_trajectory.shape[1]-1
 
        for i in range(u_ref.shape[1]):
            u_seq = u_ref[:,i].reshape(step_size, 2)
            state_seq = torch.zeros(step_size + 1,6)
            state = inital_state
            state_seq[0,:] = inital_state
            for t in range(step_size):
                u_seq_clamp = torch.clamp(u_seq[t,:], self._bicycle_dynamics.u_min, self._bicycle_dynamics.u_max)
                u_seq_clamp = u_seq[t,:]
                next_state = self._bicycle_dynamics.linearized_forward(state, u_seq_clamp.unsqueeze(0)) # D.dynamics(state, u_seq[t,:].unsqueeze(0))
                next_state = next_state.unsqueeze(0)
                state_seq[t + 1, :] = next_state
                state = next_state

            x = state_seq[:,0]
            y = state_seq[:,1]
            self._ax.scatter(x,y,
                marker="o",
                s=3,
                zorder=2,)

        # delete for jupyter notebook visualization
        # if mode == "human":
        #     # online rendering
        #     plt.pause(0.001)
        #     plt.cla()

        if mode == "rgb_array":
            # offline rendering for video
            # TODO: high resolution rendering
            self._fig.canvas.draw()
            data = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
            plt.cla()
            self._rendered_frames.append(data)

    def get_current_frame(self) -> np.ndarray:
        canvas = FigureCanvas(self._fig)
        self._ax.set_xlabel("x [m]")
        self._ax.set_ylabel("y [m]")
        canvas.draw()
        image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        image = image.reshape(self._fig.canvas.get_width_height()[::-1] + (4,))
        return image

    def close(self, path: str = None) -> None:
        if path is None:
            # mkdir video if not exists

            if not os.path.exists("video"):
                os.mkdir("video")
            path = "video/" + "navigation_2d_" + str(self._seed) + ".gif"

        if len(self._rendered_frames) > 0:
            # save animation
            clip = ImageSequenceClip(self._rendered_frames, fps=10)
            # clip.write_videofile(path, fps=10)
            clip.write_gif(path, fps=10)

    def dynamics(
        self, state: torch.Tensor, action: torch.Tensor, delta_t: float = 0.1
    ) -> torch.Tensor:
        """
        Update robot state based on differential drive dynamics.
        Args:
            state (torch.Tensor): state batch tensor, shape (batch_size, 3) [x, y, theta, v]
            action (torch.Tensor): control batch tensor, shape (batch_size, 2) [accel, steer]
            delta_t (float): time step interval [s]
        Returns:
            torch.Tensor: shape (batch_size, 3) [x, y, theta]
        """
        result = self._bicycle_dynamics.dynamics(state,action,delta_t)
        return result

    def cost_function(self, state: torch.Tensor, action: torch.Tensor, info: dict) -> torch.Tensor:
        """
        Calculate cost function
        Args:
            state (torch.Tensor): state batch tensor, shape (batch_size, 3) [x, y, theta]
            action (torch.Tensor): control batch tensor, shape (batch_size, 2) [v, omega]
        Returns:
            torch.Tensor: shape (batch_size,)
        """
        
        #goal_cost = torch.norm(state[:, :2] - self._goal_pos, dim=1)

        pos_batch = state[:, :2].unsqueeze(1)  # (batch_size, 1, 2)
        
        inital_state = info["initial_state"]
        obstacle_cost, _ = self._obstacle_map.compute_cost(pos_batch, inital_state)
        obstacle_cost = obstacle_cost.squeeze(1)  # (batch_size,)

        vel_u = state[:,3]
        vel_v = state[:,4]
        vel = (vel_u.pow(2) + vel_v.pow(2)).pow(0.5)
        vel = state[:,3]
        velocity_cost = (vel - param["target_v"]).pow(2)

        cost = 1 * velocity_cost + 1000000 * obstacle_cost

        return cost


    def collision_check(self, state: torch.Tensor) -> torch.Tensor:
        """

        Args:
            state (torch.Tensor): state batch tensor, shape (batch_size, traj_size , 3) [x, y, theta]
        Returns:
            torch.Tensor: shape (batch_size,)
        """
        pos_batch = state[:, :, :2]
        _, is_collisions = self._obstacle_map.compute_cost(pos_batch) # .squeeze(1)
        is_collisions = is_collisions.squeeze(1)
        return is_collisions
