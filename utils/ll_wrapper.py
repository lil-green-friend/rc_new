import envpool
import torch as pt

TWO_PI = 2 * pt.pi
OBS_LABELS = ("x", "y", "vx", "vy", "a", "va", "lr", "ll")


class LLWrapper:
    def __init__(
        self,
        n_agents,
        n_threads,
        device="cpu",
        render_mode=None,
        render_width=None,
        render_height=None,
        render_callback=None,
    ):
        self.device = device
        self.n_agents = n_agents
        render_kw = {}
        if render_width is not None:
            render_kw["render_width"] = render_width
        if render_height is not None:
            render_kw["render_height"] = render_height

        self.env = envpool.make(
            "LunarLander-v2",
            "gymnasium",
            num_envs=n_agents,
            num_threads=n_threads,
            render_mode=render_mode,
        )
        self.render_mode = render_mode
        self.render_callback = render_callback
        self.video_frames = []  # Accumulate frames for video creation
        self.obs_labels = OBS_LABELS

        # max/min possible values each observation can take
        # (according to specification; in practice higher/lower values are possible)
        self.obs_high = pt.tensor(
            [1.5, 1.5, 5.0, 5.0, 3.14, 5.0, 1.0, 1.0], device=device
        )
        self.obs_low = pt.tensor(
            [-1.5, -1.5, -5.0, -5.0, -3.14, -5.0, -0.0, -0.0], device=device
        )
        self.obs_range = self.obs_high - self.obs_low

        self.obs_angles = pt.tensor(
            [0, 0, 0, 0, 1, 0, 0, 0], dtype=pt.bool, device=device
        )  # indices of observations that represent an angle
        self.binary_obs = pt.tensor(
            [0, 0, 0, 0, 0, 0, 1, 1], dtype=pt.bool, device=device
        )  # indices of observations that take binary values (0.0 or 1.0)

    def reset(self):
        obs_data = self.normalize_obs(
            pt.tensor(self.env.reset()[0], device=self.device)
        )
        self._render_frames()
        return obs_data

    def step(self, action):
        obs, rew, term, trunc = [
            pt.tensor(x, device=self.device)
            for x in self.env.step(action.cpu().numpy())[:-1]
        ]
        obs = self.normalize_obs(obs)
        self._render_frames()
        return obs, rew, term, trunc

    def normalize_obs(self, obs):
        """
        Normailizes the observations from the envrionment to the range [-1, 1].
        """
        # first normalize angles so they properly "wrap around" instead of clipping
        obs[..., self.obs_angles] = (
            obs[..., self.obs_angles] - pt.pi
        ) % TWO_PI - pt.pi  # periodic "sawtooth"-function from [-pi, -pi] to [pi, pi]
        # normalize observations to [-1, 1]
        inputs = 2 * (obs - self.obs_low) / self.obs_range - 1

        # ensure that inputs are within the valid range
        inputs = pt.clip(inputs, -1, 1)

        return inputs

    def _render_frames(self):
        """Render all environments and either pass to callback or accumulate frames."""
        if self.render_mode != "rgb_array":
            return

        # Get render IDs for all environments and call render
        render_ids = list(range(self.n_agents))
        if len(render_ids) == 1:
            render_ids = render_ids[0]

        frames = self.env.render(render_ids)  # Returns (n_envs, height, width, 3)

        if self.render_callback is not None:
            # Pass frames to callback for real-time rendering
            self.render_callback(frames)
        else:
            # Accumulate frames for later video creation
            self.video_frames.append(frames)

    def get_video_frames(self):
        """Retrieve accumulated video frames."""
        return self.video_frames

    def reset_video_frames(self):
        """Clear accumulated video frames."""
        self.video_frames = []
