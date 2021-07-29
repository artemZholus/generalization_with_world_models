import tensorflow as tf
import numpy as np
from tensorflow.python.ops import state_ops
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

import pathlib
import agent
import common
import time
import math

class DyneTrainer:
    def __init__(self, config, dyne_encoder):
        self.config = config
        self.timed = common.Timed()
        self.dyne_encoder = dyne_encoder
        self.dyne_opt = common.Optimizer('dyne', **config.dyne_opt)
        self.modules = [self.dyne_encoder]

    def train(self, batch):
        with tf.GradientTape() as dyne_tape:
            obs = self.preprocess(batch)
            if self.config.mode == 'sa_dyne':
                input_obs = self.dyne_encoder.conv_obs(obs)
            elif self.config.mode == 'dyne':
                input_obs = obs['image']

            action = batch['action']
            input_obs = input_obs[:, 0]
            target_obs = obs['image'][:, -1]
            outs = self.dyne_encoder(input_obs, action)
            if self.config.mode == 'dyne':
                loss, metrics = self.loss(target_obs, *outs)
            elif self.config.mode == 'sa_dyne':
                loss, metrics = self.sa_loss(target_obs, *outs)
        metrics.update(self.dyne_opt(dyne_tape, loss, self.modules))
        return metrics

    def loss(self, true_obs, pred_obs, mean, logvar):
        true_obs = tf.cast(true_obs, tf.float32)
        obs_size = np.prod(true_obs.shape[1:])
        like = tf.keras.losses.mean_squared_error(true_obs, pred_obs)

        kld = -0.5 * tf.reduce_sum(1 + logvar - tf.math.pow(mean, 2) - tf.math.exp(logvar))

        metrics = {'likelihood_loss': like,
                   'kld_loss': kld}
        dyne_loss = like / obs_size + self.config.action_kld_scale * kld
        return dyne_loss, metrics

    def sa_loss(self, true_obs, pred_obs, s_mean, s_logvar, z_mean, z_logvar):
        true_obs = tf.cast(true_obs, tf.float32)
        pred_obs = tf.cast(pred_obs, tf.float32)
        obs_size = np.prod(true_obs.shape[1:])
        like = tf.reduce_sum(tf.keras.losses.mean_squared_error(true_obs, pred_obs))

        state_kld = -0.5 * tf.reduce_sum(1 + s_logvar - tf.math.pow(s_mean, 2) - tf.math.exp(s_logvar))
        action_kld = -0.5 * tf.reduce_sum(1 + z_logvar - tf.math.pow(z_mean, 2) - tf.math.exp(z_logvar))

        metrics = {'likelihood_loss': like,
                   'action_kld_loss': action_kld,
                   'state_kld_loss': state_kld}
        dyne_loss = like / obs_size + \
                    self.config.action_kld_scale * action_kld + \
                    self.config.state_kld_scale * state_kld
        return dyne_loss, metrics

    @tf.function
    def preprocess(self, obs):
        dtype = prec.global_policy().compute_dtype
        obs = obs.copy()
        obs['image'] = tf.cast(obs['image'], dtype) / 255.0 - 0.5
        obs['image'] = tf.stack([obs['image'][:, 0], obs['image'][:, -1]], 1)
        return obs


class DyneEncoder(common.Module):
    def __init__(self, config, traj_len, action_space):
        self.traj_len = traj_len
        self.action_size = np.prod(action_space.shape)
        self.act_embed_size = config.act_embed_size
        self.obs_embed_size = config.obs_embed_size
        self.action_net = config.action_net
        self.recon_net = config.recon_net

    @tf.function
    def __call__(self, state, action):
        z, mean, logvar = self.embed_action(action)
        return self.predict_state(state, z), mean, logvar

    @tf.function
    def embed_obs(self, obs):
        return obs

    @tf.function
    def embed_action(self, action):
        # assume action is tf.Tensor with shape
        # [batch_size, traj_len, action_dim]
        assert action.shape[1] == self.traj_len
        assert action.shape[2] == self.action_size
        batch_size = action.shape[0]
        traj_size = self.traj_len * self.action_size

        action = tf.reshape(action, [batch_size, traj_size])
        z = self.get(f'action_net', common.MLP, self.act_embed_size, 
                        **self.action_net)(action)
        return z.sample(), z.mean(), tf.math.log(z.variance())

    @tf.function
    def predict_state(self, state, z):
        # assume state is tf.Tensor with shape
        # [batch_size, *obs_dim]
        # and z is tf.Tensor with shape
        # [batch_size, embed_size]
        batch_size = state.shape[0]

        state = tf.reshape(state, [batch_size, self.obs_embed_size])
        input = tf.concat([state, z], 1)
        next_state = self.get(f'recon_net', common.MLP, self.obs_embed_size, 
                        **self.recon_net)(input)
        return next_state.mode()

    
class SADyneEncoder(DyneEncoder):
    def __init__(self, config, traj_len, action_space):
        super().__init__(config, traj_len, action_space)

        self.action_repeat = self.obs_embed_size // (2 * self.act_embed_size)
        self.action_input_dim = self.action_repeat * self.act_embed_size

        self.obs_net_encoder = common.ConvEncoder(**config.obs_net_encoder)
        self.obs_net_dist = common.MLP(self.obs_embed_size, **config.obs_net_dist)
        self.image_net = common.ConvDecoder(**config.image_net)

    @tf.function
    def __call__(self, state, action):
        s, s_mean, s_logvar = self.embed_obs(state)
        z, z_mean, z_logvar = self.embed_action(action)
        pred_state = self.predict_state(s, z)
        pred_state = self.deconv_obs(pred_state)
        return pred_state, s_mean, s_logvar, z_mean, z_logvar

    @tf.function
    def conv_obs(self, obs):
        conved_obs = self.obs_net_encoder(obs)
        return conved_obs

    @tf.function
    def deconv_obs(self, obs):
        image = self.image_net(obs)
        return image.mode()
    
    @tf.function
    def embed_obs(self, obs):
        out = self.obs_net_dist(obs)
        return out.sample(), out.mean(), tf.math.log(out.variance())

    @tf.function
    def predict_state(self, state, z):
        # assume state is tf.Tensor with shape
        # [batch_size, *obs_dim]
        # and z is tf.Tensor with shape
        # [batch_size, action_embed_size]
        batch_size = state.shape[0]

        state = tf.reshape(state, [batch_size, self.obs_embed_size])
        input = tf.concat([state, tf.repeat(z, self.action_repeat, 1)], 1)
        next_state = self.get(f'recon_net', common.MLP, 400, 
                              **self.recon_net)(input).mode()
        return next_state
        
