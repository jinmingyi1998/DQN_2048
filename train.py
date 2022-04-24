import logging
import time

import numpy as np
from tensorboardX import SummaryWriter

from dqn import DQN_Trainer
from gym2048 import Game2048Env


def set_logger():
    """
    This method should be call before everything
    :return:
    """
    from colorlog_handler import ColorLoggerHandler

    logger = logging.getLogger("dqn")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers = [ColorLoggerHandler()]
    return logger


class Trainer:
    def __init__(
        self,
        train_episodes=20000000,
        test_episodes=20,
        ifrender=False,
        eval_interval=4000,
        epsilon_decay_interval=2000,
        log_interval=50,
    ):
        self.train_episodes = train_episodes
        self.test_episodes = test_episodes
        self.if_render = ifrender
        self.render_interval = 100
        self.eval_interval = eval_interval
        self.epsilon_decay_interval = epsilon_decay_interval
        self.num_random_episode = 20000

        self.log_interval = log_interval
        self.logger = logging.getLogger("dqn")
        self.output_dir = "./output"
        self.tflogger = SummaryWriter(self.output_dir)

        self.save_ckpt_interval = 1000

    def train(self):
        episodes = self.train_episodes
        agent = DQN_Trainer(use_gpu=False, output_dir=self.output_dir)
        env = Game2048Env()

        for i in range(episodes):
            state, reward, done, info = env.reset()

            start_time = time.time()
            while True:
                if i < self.num_random_episode:
                    action = agent.get_action(state, random=True)
                else:
                    action = agent.get_action(state)

                next_state, reward, done, info = env.step(action)

                agent.store_step(state, action, reward, next_state)
                state = next_state

                if self.if_render and i % self.render_interval == 1:
                    print(f"after action {action}")
                    print(f"reward {reward}")
                    env.render()

                if (
                    len(agent.replay_memory) % agent.train_interval == 0
                    and len(agent.replay_memory) > self.num_random_episode
                ):
                    loss_value = agent.update()
                    self.logger.info(f"loss_value {loss_value}")
                    self.tflogger.add_scalar("0_loss", loss_value, agent.step_counter)

                if done:
                    if i % self.log_interval == 0:
                        self.logger.info(f"episode: {i} done")
                        self.logger.info(
                            f"game status: score: {info['score']} steps:{info['steps']} highest:{info['highest']}"
                        )
                        episode_time = time.time() - start_time
                        self.logger.info(f"episode time:{episode_time} s")
                        self.tflogger.add_scalar("9_episode_time(s)", episode_time, i)
                        self.tflogger.add_scalar("2_train_score", info["score"], i)
                        self.tflogger.add_scalar("2_train_steps", info["steps"], i)
                        self.tflogger.add_scalar("2_train_highest", info["highest"], i)

                    if i % self.epsilon_decay_interval == 0:  # episilon decay
                        agent.epsilon_decay(i, episodes)
                    break

            if i > 0 and i % self.save_ckpt_interval == 0:
                agent.save()

            # eval
            if (
                i % self.eval_interval == 0
                and agent.learn_epoch_counter > 50
                and i > 100
            ):
                eval_info = self.test(
                    episodes=self.test_episodes, agent=agent, ifrender=self.if_render
                )
                average_score, max_score, score_lis = (
                    eval_info["mean"],
                    eval_info["max"],
                    eval_info["list"],
                )
                self.logger.info(f"eval average score {average_score}")
                self.logger.info(f"eval max socre {max_score}")
                self.tflogger.add_scalar("1_test_mean_score", average_score, i)
                self.tflogger.add_scalar("1_test_max_score", max_score, i)

    def test(self, episodes=20, agent=None, load_path=None, ifrender=True, log=False):
        self.logger.info("start eval")
        if agent is None:
            agent = DQN_Trainer(output_dir=self.output_dir, use_gpu=True)
            if load_path:
                agent.load(load_path)
            else:
                agent.load()
        env = Game2048Env()
        score_list = []
        highest_list = []
        for i in range(episodes):
            state, _, done, info = env.reset()
            while True:
                action = agent.get_action(state, deterministic=True)
                next_state, _, done, info = env.step(action)
                state = next_state
                if ifrender:
                    env.render()
                if done:
                    break
            score_list.append(info["score"])
            highest_list.append(info["highest"])
        result_info = {
            "mean": np.mean(score_list),
            "max": np.max(score_list),
            "list": score_list,
        }
        return result_info


def main():
    trainer = Trainer(ifrender=False)
    trainer.train()


if __name__ == "__main__":
    set_logger()
    main()
