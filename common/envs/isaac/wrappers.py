import gym
import numpy as np


class FrameSkip(gym.Wrapper):
    """Return every `skip`-th frame and repeat given action during skip.
    Note that this wrapper does not "maximize" over the skipped frames.
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.called = 0
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            self.called += 1
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class EpisodeLengthWrapper(gym.Wrapper):
    def __init__(self, env, max_length=6000):
        self.cnt = 0
        self.max_length = max_length
        super().__init__(env)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.cnt = 0
        return observation

    def step(self, action):
        self.cnt += 1
        observation, reward, done, info = self.env.step(action)
        return observation, reward, self.done(done), info

    def done(self, done):
        if done:
            return True
        if self.cnt >= self.max_length:
            return True
        return False


class OneHot(gym.spaces.Discrete):
    def __init__(self, n):
        super().__init__(n)
        self.eye = np.eye(self.n, dtype=np.float32)
        self.shape = (n,)

    def sample(self):
        return self.eye[super().sample()]

    def noop(self):
        return self.eye[1]


class ManyHot(gym.spaces.MultiDiscrete):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)
        self.eyes = [np.eye(s, dtype=np.float32) for s in self.nvec]
        self.shape = (sum(self.nvec,),)

    def sample(self):
        sample = []
        idx = super().sample()
        for i in range(len(self.eyes)):
            sample.append(self.eyes[i][idx[i]])
        sample = np.concatenate(sample)
        return sample


class ThreeJointsOneHot(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # {-1, 0, 1} for pan, lift and elbow joints
        self.action_space = OneHot(27)

    def action(self, action):
        action = action.argmax().item()
        pan_move = (action % 3) - 1
        action //= 3
        lift_move = (action % 3) - 1
        action //= 3
        elbow_move = (action % 3) - 1
        action = [0, pan_move, lift_move, elbow_move, 0, 0, 0]
        return np.array(action)


class ThreeJointsManyHot(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # {-1, 0, 1} for pan, lift and elbow joints
        self.action_space = ManyHot([3, 3, 3])

    def action(self, action):
        pan, lift, elbow = action[:3], action[3:6], action[6:9]
        pan_move = pan.argmax() - 1
        lift_move = lift.argmax() - 1
        elbow_move = elbow.argmax() - 1
        action = [0, pan_move, lift_move, elbow_move, 0, 0, 0]
        return np.array(action)


class ThreeJointsCont(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # {-1, 0, 1} for pan, lift and elbow joints
        self.action_space = gym.spaces.Box(-2, 2, shape=(3,), dtype=np.float32)

    def action(self, action):
        pan_move, lift_move, elbow_move = action
        action = [0, pan_move, lift_move, elbow_move, 0, 0, 0]
        return np.array(action)