import argparse
import os
import numpy as np
import pandas as pd
import torch
import csv
from torch.utils.tensorboard import SummaryWriter
from normalization import Normalization, RewardScaling
from ppo import PPO_continuous
from replaybuffer import ReplayBuffer
from uav import MakeEnv


def evaluate_policy(args, env, agent, state_norm):
    times = 2 ** 3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s, update=False)
        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action
            else:
                action = a
            s_, r, done, _ = env.step(action)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times


def main(args, seed, speed, target_rate, ROOT_PATH=None, load_path=None, s_mean_std=None):
    # Tạo môi trường với UAV cố định
    env = MakeEnv(set_num=args.car_num, target_rate=target_rate)
    env_evaluate = MakeEnv(set_num=args.car_num, target_rate=target_rate)

    # Đặt seed
    env.seed(seed)
    env_evaluate.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Các thông số môi trường
    args.state_dim = env.observation_space.shape[0]  # Sẽ là car_num (5)
    args.action_dim = env.action_space.shape[0]      # Sẽ là 1 (divergence angle)
    args.max_action = float(env.action_space.high[0])
    args.max_episode_steps = env.max_episode_steps
    args.max_train_steps = args.max_episode_steps * args.max_train_episodes

    if not load_path:
        print(ROOT_PATH)
        if not os.path.exists(ROOT_PATH):
            os.makedirs(ROOT_PATH)
        
        # Ghi settings
        with open(ROOT_PATH + '/setting.txt', 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in args.__dict__.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')
        f.close()

        evaluate_num = 0
        all_evaluate_rewards = []
        total_steps = 0
        episode_num = 0
        state_norm_info = {"mean": [], 'std': []}

        replay_buffer = ReplayBuffer(args)
        agent = PPO_continuous(args)

        writer = SummaryWriter(log_dir=ROOT_PATH + '/runs')
        
        state_norm = Normalization(shape=args.state_dim)
        if args.use_reward_norm:
            reward_norm = Normalization(shape=1)
        elif args.use_reward_scaling:
            reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

        while episode_num < args.max_train_episodes:
            s = env.reset()
            if args.use_state_norm:
                s = state_norm(s)
            if args.use_reward_scaling:
                reward_scaling.reset()
            
            ep_r = 0
            done = False
            
            while not done:
                a, a_logprob = agent.choose_action(s)

                if args.policy_dist == "Beta":
                    action = 2 * (a - 0.5) * args.max_action
                else:
                    action = a

                s_, r, done, _ = env.step(action)
                ep_r += r

                if args.use_state_norm:
                    s_ = state_norm(s_)

                if args.use_reward_norm:
                    r = reward_norm(r)
                elif args.use_reward_scaling:
                    r = reward_scaling(r)

                dw = done  # Vì UAV cố định, không có khái niệm max_steps riêng

                replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
                s = s_
                total_steps += 1

                if replay_buffer.count == args.batch_size:
                    agent.update(replay_buffer, total_steps, writer)
                    replay_buffer.count = 0
                    state_norm_info["mean"].append(state_norm.running_ms.mean)
                    state_norm_info["std"].append(state_norm.running_ms.std)

            writer.add_scalar('train/reward_ep', ep_r, global_step=total_steps)
            episode_num = episode_num + 1

            if episode_num % args.evaluate_episode_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
                all_evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t train_episodes:{} \t".format(
                    evaluate_num, evaluate_reward, episode_num))
                writer.add_scalar('evaluate/reward_ep', evaluate_reward, global_step=total_steps)
                
                # Save best model
                if (evaluate_reward >= np.mean(all_evaluate_rewards[-5:])) and (evaluate_num >= 5):
                    path = ROOT_PATH + '/data_train'
                    if not os.path.exists(path):
                        os.makedirs(path)
                    np.save(path+'/uav_fso_{}_num_{}_seed_{}_rewards{}.npy'.format(
                        evaluate_num, args.policy_dist, seed, evaluate_reward), 
                        np.array(all_evaluate_rewards))
                    env_evaluate.buffer.save(path=ROOT_PATH+'/flydata/', episode=episode_num, target_rate=target_rate)
                    agent.save_policy(reward=evaluate_reward, path=ROOT_PATH + '/model/', episode_num=episode_num)

        # Lưu kết quả
        with open(ROOT_PATH + '/episode_rewards.csv', mode='w', newline='') as file:
            writer_csv = csv.writer(file)
            writer_csv.writerow(['Episode Number', 'Episode Reward'])
            for i, reward in enumerate(all_evaluate_rewards):
                writer_csv.writerow([i + 1, reward])

    else:
        # Load model và test
        agent = PPO_continuous(args)
        agent.load_policy(name=load_path)
        rew_all = []
        
        for episode_num in range(1):  # Test 1 episode
            s = env.reset()
            done = False
            ep_r = 0.
            
            while not done:
                if s_mean_std:
                    s = (s - s_mean_std[0]) / s_mean_std[1]
                
                a, a_logprob = agent.choose_action(s)
                
                if args.policy_dist == "Beta":
                    action = 2 * (a - 0.5) * args.max_action
                else:
                    action = a
                
                next_state, reward, done, info = env.step(action)
                ep_r += reward
                s = next_state
            
            rew_all.append(ep_r)
            env.buffer.save(path='./', episode=2, target_rate=target_rate)
        
        print(rew_all)
        print(np.mean(rew_all))


if __name__ == '__main__':
    env_steps = 150  # Số frame tối đa
    parser = argparse.ArgumentParser("Hyper-parameters Setting for PPO-continuous")
    parser.add_argument("--max_train_episodes", type=int, default=200, help="Maximum number of training steps")
    parser.add_argument("--evaluate_episode_freq", type=int, default=10 * 1, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--policy_dist", type=str, default="Beta", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=env_steps * 1, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=env_steps, help="Minibatch size")
    parser.add_argument("--car_num", type=int, default=5, help="Number of cars")

    parser.add_argument("--hidden_width", type=int, default=128, help="The number of neurons in hidden layers")
    parser.add_argument("--lr_a", type=float, default=2e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=4e-4, help="Learning rate of critic")

    parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.98, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.25, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=2 ** 3, help="PPO parameter")

    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")

    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=False, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=False, help="Trick 8: orthogonal initialization")

    parser.add_argument("--set_adam_eps", type=float, default=False, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=False, help="Trick 10: tanh activation function")

    args = parser.parse_args()

    main(args, seed=1, speed=None, target_rate=1200)  # speed không cần dùng nữa