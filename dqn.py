import logging
import os

import numpy as np
import torch
import torch.utils.data as D
from torch import nn

from q_net import QNet


class DQN_Trainer:
    def __init__(self, use_gpu=True, output_dir="./output"):
        # hyper parameters
        self.batch_size = 512
        self.lr = 0.0003
        self.epsilon = 0.3
        self.gamma = 0.96
        self.q_network_iteration = 100
        self.soft_update_theta = 0.9
        self.clip_norm_max = 1.0

        self.train_interval = 1000
        self.train_overall_interval = 5000
        self.initial_epsilon = self.epsilon = 0.15

        self.output_dir = output_dir
        self.use_gpu = use_gpu
        self.logger = logging.getLogger("dqn")

        self.num_action = 4  # 4 actions: up down left right

        # eval_net for output action, target_net for calculating target Q value
        self.eval_net, self.target_net = QNet(), QNet()
        if self.use_gpu:
            self.eval_net.cuda()
            self.target_net.cuda()

        self.learn_epoch_counter = 0
        self.step_counter = 0
        self.replay_memory = []
        self.memory_limit = 1000000
        self.train_data_length = 10000

        self.build_optimizer()
        self.build_loss()

    def build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, [10000, 20000, 30000, 50000], gamma=0.3
        )

    def build_loss(self):
        self.loss_fn = nn.MSELoss()

    def get_action(self, state: np.ndarray, random=False, deterministic=False):
        state = torch.tensor(
            state, dtype=torch.float32, device="cuda" if self.use_gpu else "cpu"
        )
        state = torch.unsqueeze(state, 0)
        state = torch.unsqueeze(state, 0)
        if deterministic or (not random and np.random.random() > self.epsilon):
            # greedy policy
            action = self.eval_net.get_action(state)
        else:
            # random policy
            action = np.random.randint(0, self.num_action)
        return action

    def store_step(self, state, action, reward, next_state):
        self.replay_memory.append((state, action, reward, next_state))
        if len(self.replay_memory) > self.memory_limit * 1.2:
            self.logger.info("clear replay memory")
            self.replay_memory = self.replay_memory[-self.memory_limit :]

    def update(self) -> float:
        # soft update the parameters
        if (
            self.learn_epoch_counter > 0
            and self.learn_epoch_counter % self.q_network_iteration == 0
        ):
            for p_e, p_t in zip(
                self.eval_net.parameters(), self.target_net.parameters()
            ):
                p_t.data = (
                    self.soft_update_theta * p_e.data
                    + (1 - self.soft_update_theta) * p_t.data
                )

        self.learn_epoch_counter += 1

        state_dataset = StateDataset(
            replay_memory=self.replay_memory[
                self.replay_memory
                if self.learn_epoch_counter % self.train_overall_interval == 0
                else -min(len(self.replay_memory), self.train_data_length) :
            ],
            use_gpu=self.use_gpu,
        )
        data_loader = D.DataLoader(
            state_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # set 0 to use main process
            drop_last=True,
        )
        loss_value = 0
        for state, action, reward, next_state in data_loader:
            q_eval_total = self.eval_net(state)
            q_eval = torch.gather(q_eval_total, -1, action)
            q_next = self.target_net(next_state).detach()
            q_max = q_next.max(1)[0].view(self.batch_size, 1)
            q_target = reward + self.gamma * q_max
            loss = self.loss_fn(q_eval, q_target)
            loss_value = loss.detach().cpu().item()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.eval_net.parameters(), self.clip_norm_max)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.step_counter += 1
        return loss_value

    def save(
        self,
    ):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        ckpt_file_path = os.path.join(self.output_dir, f"{self.step_counter}.ckpt")
        torch.save(self.eval_net.state_dict(), ckpt_file_path)

    def load(self, filepath):
        self.eval_net.load_state_dict(torch.load(filepath))

    def epsilon_decay(self, episode, total_episode):
        self.epsilon = self.epsilon * 0.6


class StateDataset(D.Dataset):
    def __init__(self, replay_memory, use_gpu=False):
        super().__init__()
        self.replay_memory = replay_memory
        self.use_gpu = use_gpu

    def __getitem__(self, index):
        state, action, reward, next_state = self.replay_memory[index]
        state = torch.tensor(
            state, dtype=torch.float32, device="cuda" if self.use_gpu else "cpu"
        )
        state = torch.unsqueeze(state, 0)
        next_state = torch.tensor(
            next_state, dtype=torch.float32, device="cuda" if self.use_gpu else "cpu"
        )
        next_state = torch.unsqueeze(next_state, 0)
        reward = torch.tensor(
            reward, dtype=torch.float32, device="cuda" if self.use_gpu else "cpu"
        )
        reward = torch.reshape(reward, [1])
        action = torch.tensor(
            action, dtype=torch.int64, device="cuda" if self.use_gpu else "cpu"
        )
        action = torch.reshape(action, [-1])
        return state, action, reward, next_state

    def __len__(self):
        return len(self.replay_memory)
