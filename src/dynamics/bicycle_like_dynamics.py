
import torch
import numpy as np
import os
import yaml
from src.envs.obstacle_map_2d import ObstacleMap
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
setting_path = os.path.join(BASE_DIR, 'setting.yaml')
with open(setting_path) as f:
    param = yaml.full_load(f)
    
@torch.jit.script
def angle_normalize(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi

class Bicycle_Dynamics:
    def __init__(
        self, device=torch.device("cuda"), dtype=torch.float32, seed: int = 42
    ) -> None :
        # device and dtype
        if torch.cuda.is_available() and device == torch.device("cuda"):
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        self._dtype = dtype
        self.u_min = torch.tensor(param['u_min'], device=self._device, dtype=self._dtype)
        self.u_max = torch.tensor(param['u_max'], device=self._device, dtype=self._dtype)
        self.V_MAX = torch.tensor(param['v_max'], device=self._device, dtype=self._dtype)
        self.W_MAX = torch.tensor(param['w_max'], device=self._device, dtype=self._dtype)
        
        self.map_size = (param['map_size'], param['map_size'])
        self.cell_size=0.1
        self._obstacle_map = ObstacleMap(
            map_size=self.map_size, cell_size=self.cell_size, device=self._device, dtype=self._dtype, detect_range=param['range']
        )

        # self.Iz = torch.tensor(1536.7/10, device=self._device, dtype=self._dtype)      # yaw inertia of vehicle body
        # self.kf = torch.tensor(-128916/100, device=self._device, dtype=self._dtype)     # front axle equivalent sideslip stiffness, 값이 작을수록 미끄러움
        # self.kr = torch.tensor(-85944/100, device=self._device, dtype=self._dtype)      # rear axle equivalent sideslip stiffness
        # self.lf = torch.tensor(0.5, device=self._device, dtype=self._dtype)        # distance between C.G. and front axle
        # self.lr = torch.tensor(0.5, device=self._device, dtype=self._dtype)        # distance between C.G. and rear axle
        # self.m = torch.tensor(1412/10, device=self._device, dtype=self._dtype)         # mass of the vehicle
        
        self.m  = torch.tensor(1.5,    device=self._device, dtype=self._dtype)   # kg
        self.lf = torch.tensor(0.13,   device=self._device, dtype=self._dtype)   # m
        self.lr = torch.tensor(0.15,   device=self._device, dtype=self._dtype)   # m  (WB=0.28 m)
        self.Iz = torch.tensor(0.014,  device=self._device, dtype=self._dtype)   # kg·m^2
        self.kf = torch.tensor(-120.0, device=self._device, dtype=self._dtype)   # N/rad
        self.kr = torch.tensor(-140.0, device=self._device, dtype=self._dtype)   # N/rad
        self.L = torch.tensor(self.lf + self.lr, device=self._device, dtype=self._dtype)
    def linearized_state_space(
        self, state: torch.Tensor, action: torch.Tensor, delta_t: float = 0.1
    ):
        """
        torch.Tensor: shape (batch_size, 6) [x, y, theta, u, v, w]
        """
        T = delta_t
        Iz = self.Iz 
        kf = self.kf
        kr = self.kr 
        lf = self.lf 
        lr = self.lr
        m = self.m

        x = state[:, 0].view(-1, 1)
        y = state[:, 1].view(-1, 1)
        theta = state[:, 2].view(-1, 1)
        t = angle_normalize(theta)
        u = state[:, 3].view(-1, 1)
        v = state[:, 4].view(-1, 1)
        w = state[:, 5].view(-1, 1)
        X = torch.Tensor([x,y,t,u,v,w])

        # current action
        accel = torch.clamp(action[:, 0].view(-1, 1), self.u_min[0], self.u_max[0])
        steer = torch.clamp(action[:, 1].view(-1, 1), self.u_min[1], self.u_max[1])
        U = torch.Tensor([accel, steer])

        b1 = (m*v - T*kf*steer - 2*T*m*u*w)*(m*u - T*(kf+kr)) - m*(m*u*v + T*(lf*kf - lr*kr)*w - T*kf*steer*u - T*m*u*u*w) / ((m*u - T*(T*(kf+kr)))**2)
        b2 = (Iz*w - T*lf*kf*steer) * (Iz*u - T*(lf*lf*kf + lr*lr*kr)) - Iz*(Iz*u*w + T*(lf*kf - lr*kr)*v - T*lf*kf*steer*u) / ((Iz*u - T*(lf*lf*kf + lr*lr*kr))**2)

        self.A = torch.Tensor([
            [1, 0, -T*(u*torch.sin(t) + v*torch.cos(t)), T*torch.cos(t), -T*torch.sin(t), 0],
            [0, 1, -T*(v*torch.sin(t) - u*torch.cos(t)), T*torch.sin(t), T*torch.cos(t), 0],
            [0, 0, 1, 0, 0, T],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, b1, m*u/(m*u-T*(kf+kr)), (T*(lf*kf - lr*kr) - T*m*u*u)/(m*u-T*(kf+kr))],
            [0, 0, 0, b2, T*(lf*kf - lr*kr)/(Iz*u - T*(lf*lf*kf + lr*lr*kr)), Iz*u/(Iz*u - T*(lf*lf*kf + lr*lr*kr))]
        ]) 

        self.B = torch.Tensor([
            [0, 0],
            [0, 0],
            [0, 0],
            [T, 0],
            [0, -(T*kf*u)/(m*u-T*(kf+kr))],
            [0, -T*lf*kf*u/(Iz*u - T*(lf*lf*kf + lr*lr*kr))]
        ])
        
        Xnext = self.dynamics(state, action, T)
        Xnext = Xnext.squeeze(0)
        self.C = Xnext - self.A @ X - self.B @ U
        
    def linearized_forward(self, state: torch.Tensor, action: torch.Tensor, delta_t: float = 0.1)-> torch.Tensor:
        self.linearized_state_space(state, action, delta_t)
        next_state = self.A @ state.squeeze(0) + self.B @ action.squeeze(0) + self.C
        return next_state

    def compute_uncontrolled_state(self, inital_state, A_list, C_list):
        H = len(A_list)
        A_unctrl = torch.eye(len(A_list[0]), dtype=A_list[0].dtype, device=A_list[0].device)

        for j in range(0,H):
            A_unctrl = A_list[j] @ A_unctrl

        C_unctrl = torch.zeros_like(C_list[0])

        for k in range(0,H-1):
            prod = torch.eye(A_list[0].shape[0], dtype=A_list[0].dtype, device=A_list[0].device)
            for j in range(k+1, H):
                prod = A_list[j] @ prod 
            C_unctrl = C_unctrl + prod @ C_list[k]
        C_unctrl = C_unctrl + C_list[H-1]

        uncontrolled_state = A_unctrl @ inital_state.T + C_unctrl.unsqueeze(-1) 
        return uncontrolled_state

    def dynamics(
        self, state: torch.Tensor, action: torch.Tensor, delta_t: float = 0.1
    ) -> torch.Tensor:
        """
        Update robot state based on differential drive dynamics.
        Args:
            state (torch.Tensor): state batch tensor, shape (batch_size, 6) [x, y, theta, u, v, w]
            action (torch.Tensor): control batch tensor, shape (batch_size, 2) [accel, steer]
            delta_t (float): time step interval [s]
        Returns:
            torch.Tensor: shape (batch_size, 6) [x, y, theta, u, v, w]
        """

        # current state
        x = state[:, 0].view(-1, 1)
        y = state[:, 1].view(-1, 1)
        theta = state[:, 2].view(-1, 1)
        theta = angle_normalize(theta)
        u = state[:, 3].view(-1, 1)
        v = state[:, 4].view(-1, 1)
        w = state[:, 5].view(-1, 1)

        # current action
        accel = torch.clamp(action[:, 0].view(-1, 1), self.u_min[0], self.u_max[0])
        steer = torch.clamp(action[:, 1].view(-1, 1), self.u_min[1], self.u_max[1])
        
        # Next state
        new_x = x + delta_t * (u * torch.cos(theta) - v * torch.sin(theta))
        new_y = y + delta_t * (v * torch.cos(theta) + u * torch.sin(theta))
        new_theta = angle_normalize(theta + delta_t * w)
        new_u = u + delta_t * accel
        new_v = (self.m * u * v + delta_t * (self.lf * self.kf - self.lr * self.kr) * w - delta_t * self.kf * steer * u - delta_t * self.m * u * u * w) \
            / (self.m * u - delta_t * (self.kf + self.kr))
        new_w = (self.Iz * u * w + delta_t * (self.lf * self.kf - self.lr * self.kr) * v - delta_t * self.lf * self.kf * steer * u) \
            / (self.Iz * u - delta_t * (self.lf * self.lf * self.kf + self.lr * self.lr * self.kr))


        # Clamp x and y to the map boundary
        x_lim = torch.tensor(
            self._obstacle_map.x_lim, device=self._device, dtype=self._dtype
        )
        y_lim = torch.tensor(
            self._obstacle_map.y_lim, device=self._device, dtype=self._dtype
        )
        
        clamped_x = torch.clamp(new_x, x_lim[0], x_lim[1])
        clamped_y = torch.clamp(new_y, y_lim[0], y_lim[1])
        clamped_u = torch.clamp(new_u, -self.V_MAX, self.V_MAX)
        clamped_v = torch.clamp(new_v, -self.V_MAX, self.V_MAX)
        clamped_w = torch.clamp(new_w, -self.W_MAX, self.W_MAX)

        result = torch.cat([clamped_x, clamped_y, new_theta, clamped_u, clamped_v, clamped_w], dim=1)

        return result

    def spectral_expansion(self, step_size, state_, inputs):
        inital_state = state_
        # input_scale_energy = self.u_max
        # input_scale_energy = torch.Tensor([1,1])
        input_scale_energy = torch.Tensor([2 / (self.u_max[0] - self.u_min[0]), 2 / (self.u_max[1] - self.u_min[1])])
        A_list = []
        C_list = []
        B_scaled_list = []

        # forward
        for i in range(0,step_size): 
            input_ = inputs[i,:].unsqueeze(0)
            self.linearized_state_space(state_,input_)
            A_list.append(self.A)
            C_list.append(self.C)
            B_scaled = self.B @ torch.diag(input_scale_energy)
            B_scaled_list.append(B_scaled)
            state_ = self.dynamics(state_,input_)

        A = torch.eye(self.A.shape[0])
        C_energy = torch.eye(self.A.shape[0])
        H = len(A_list)
        for i in range(1,H):
            for j in range(i,H):
                A = A_list[j] @ A
            if i == 1:
                C_energy = A @ B_scaled_list[i-1]
            else:
                C_energy = torch.hstack([C_energy, A @ B_scaled_list[i-1]])
        C_energy = torch.hstack([C_energy, B_scaled_list[H-1]])

        uncontrolled_state = self.compute_uncontrolled_state(inital_state, A_list, C_list)

        eig_val, eig_vec = torch.linalg.eig(C_energy @ C_energy.T)
        torch.linalg.norm(eig_vec)
        magnitude = torch.sqrt(eig_val) * np.sqrt(step_size * 0.1) # sqrt(E), E = integral(||u(t)||^2)
        vec = magnitude.T * eig_vec 
        vec = torch.cat([vec,-vec], dim=1)

        z = vec + uncontrolled_state

        C_energy_inv = torch.linalg.pinv(C_energy)
        u_ref = C_energy_inv.real @ z.real
        print(u_ref.shape)
        # u_seq = u_ref[:,i].reshape(step_size, 2)
        # u_seq_clamp = torch.clamp(u_seq, self.u_min, self.u_max)
        return vec, u_ref
    

