import envpool
import torch as pt

TWO_PI = 2 * pt.pi

class LLWrapper:
    def __init__(self, n_agents, n_threads, device="cpu"):
        self.device = device
        self.env = envpool.make('LunarLander-v2', 'gymnasium', num_envs=n_agents, num_threads=n_threads)
        self.obs_labels = ('x', 'y', 'vx', 'vy', 'a', 'va', 'lr', 'll')

        # max/min possible values each observation can take 
        # (according to specification; in practice higher/lower values are possible)
        self.obs_high = pt.tensor([1.5, 1.5, 5.0, 5.0, 3.14, 5.0, 1.0, 1.0], device=device) 
        self.obs_low = pt.tensor([-1.5, -1.5, -5.0, -5.0, -3.14, -5.0, -0.0, -0.0], device=device)
        self.obs_range = self.obs_high - self.obs_low

        self.obs_angles = pt.tensor([0, 0, 0, 0, 1, 0, 0, 0], dtype=pt.bool, device=device) # indices of observations that represent an angle
        self.binary_obs = pt.tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=pt.bool, device=device) # indices of observations that take binary values (0.0 or 1.0)
    
    def reset(self):
        return self.normalize_obs(pt.tensor(self.env.reset()[0], device=self.device))
    
    def step(self, action):
        obs, rew, term, trunc = [pt.tensor(x, device=self.device) for x in self.env.step(action.cpu().numpy())[:-1]]
        obs = self.normalize_obs(obs)
        return obs, rew, term, trunc

    def normalize_obs(self, obs):
        """
        Normailizes the observations from the envrionment to the range [-1, 1].
        """
        # first normalize angles so they properly "wrap around" instead of clipping
        obs[..., self.obs_angles] = (obs[..., self.obs_angles] - pt.pi) % TWO_PI - pt.pi # periodic "sawtooth"-function from [-pi, -pi] to [pi, pi]
        # normalize observations to [-1, 1]
        inputs = 2*(obs - self.obs_low) / self.obs_range - 1

        # ensure that inputs are within the valid range
        inputs = pt.clip(inputs, -1, 1)

        return inputs