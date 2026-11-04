import numpy as np
import torch

import argparse
import os

import tqdm.auto  

import utils
import os   
 
from tqdm import trange 
import numpy as np
import torch
import argparse

import gym_dmc as gym
import utils
torch.backends.cudnn.benchmark = True
torch.set_num_threads(8)
 
 
is_log = True
def compute_discounted_return(rewards, discount):
    """Compute discounted return from a list of rewards."""
    G = 0
    discounted_returns = []
    for r in reversed(rewards):
        G = r + discount * G
        discounted_returns.insert(0, G)
    return discounted_returns
def eval_policy(policy, eid,args, seed, eval_episodes=10):
    eval_env = gym.make(
        eid=eid,
        domain_name=args.domain_name,
        task_name=args.task_name,
        
        visualize_reward=False,
        from_pixels=False,
        frame_skip=args.action_repeat,
        
    )
    true_list = []
    pred_list = []
    diversity_list  =[]
    Qs = []
    eval_env.seed(seed + 100)
    avg_reward = 0.
    episode_errors = []  # Store prediction errors for each episode
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        # truncated=done
        # state = state[0]
        while not done:
            action = policy.select_action(np.array(state),deterministic=True)
            p_state = np.array(state)
            next_state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            if hasattr(policy, 'watch'):
                # adaptive rollout
                policy.watch(p_state, action, reward)
            if is_log:
                true_list.append(reward)
                obs = torch.FloatTensor(state.reshape(1, -1)).to(policy.device)
                action_t  = torch.FloatTensor(action.reshape(1, -1)).to(policy.device)
                with torch.no_grad():
                    if hasattr(policy, 'fixed_encoder'):
                        fixed_zs = policy.fixed_encoder.zs(obs)
                        fixed_zsa = policy.fixed_encoder.zsa(fixed_zs, action_t)
                        if args.policy!='DivTD':
                            
                            pred_1, pred_2 = policy.critic(obs, action_t, fixed_zsa, fixed_zs)
                            pred = torch.cat([pred_1, pred_2], axis=1)
                        else:
                            pred = policy.critic(obs, action_t, fixed_zsa, fixed_zs)

                    else:
                        pred = policy.critic(obs, action_t)
                    pred = pred.detach().cpu().numpy()
                    pred_list.append(pred)
                    
            state = next_state
                # After episode ends, compute errors
        if is_log and len(true_list) > 0:
            # Compute discounted returns for this episode
            discounted_returns = compute_discounted_return(true_list, args.discount)
            Qs.append(discounted_returns[0])
            # Compute prediction errors at each timestep
            for t, (pred, G) in enumerate(zip(pred_list, discounted_returns)):
                # pred shape: [1, n_nets, n_quantiles], take mean over nets and quantiles
                diversity_list.append(np.std(pred,axis=1))
                pred_value = np.mean(pred)
                error = pred_value - G
                episode_errors.append(error)  
    avg_reward /= eval_episodes
        # Compute error statistics
    if is_log and len(episode_errors) > 0:
        error_mean = np.mean(episode_errors)
        error_std = np.std(episode_errors)
        diversity_mean =  np.mean(diversity_list)
        diversity_std =  np.std(diversity_list)
        qs_mean =  np.mean(Qs)
    else:
        error_mean, error_std,qs_mean,diversity_mean,diversity_std= 0.0, 0.0,0.0, 0.0,0.0
    
 
    return avg_reward,(error_mean,error_std,qs_mean,diversity_mean,diversity_std)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()  
    parser.add_argument("--policy", default='DivTD')
    parser.add_argument('--domain_name', default="hopper")
    parser.add_argument('--task_name', default="hop")
    parser.add_argument("--hidden_dim", default=256, type=int)

    parser.add_argument('--N', default=3, type=int)
    parser.add_argument('--M', default=2, type=int)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--action_repeat', default=2, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start_timesteps", default=10000, type=int)
    parser.add_argument("--eval_freq", default=5e3, type=int)
    parser.add_argument("--max_timesteps", default=5e5, type=int)
    parser.add_argument("--expl_noise", default=0.1, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--mini_batch_size", default=None, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--policy_noise", default=0.2)
    parser.add_argument("--noise_clip", default=0.5)
    parser.add_argument("--policy_freq", default=2, type=int)
  

    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--load_model", default="")
    parser.add_argument("--prioritized", default=True,type=bool)
    parser.add_argument("--target_update_rate", default=1, type=int)  # OAC 官方设置：每步更新
    parser.add_argument("--device", default='cuda:0', type=str)
    args = parser.parse_args()
    
     

    
    eid = f"{args.domain_name.capitalize()}-{args.task_name}-v1"

    eid = f"{args.domain_name.capitalize()}-{args.task_name}-v1"

    device = args.device
    start_timesteps = args.start_timesteps
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
    eid = f"{args.domain_name.capitalize()}-{args.task_name}-v1"
    file_name = f'{args.policy}_{args.domain_name}_{args.task_name}_{args.seed}'

    eid = f"{args.domain_name.capitalize()}-{args.task_name}-v1"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.domain_name}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("results"+str(args.N)+"-"+str(args.M)+"-"+str(args.alpha)+"-"+str(args.action_repeat)+"/"):
        os.makedirs("results"+str(args.N)+"-"+str(args.M)+"-"+str(args.alpha)+"-"+str(args.action_repeat)+"/")
    if args.save_model and not os.path.exists("./models"):
        
        os.makedirs("./models")
    torch.manual_seed(args.seed)
    
    np.random.seed(args.seed)

    env = gym.make(
        eid=eid,
        domain_name=args.domain_name,
        task_name=args.task_name,
        
        visualize_reward=False,
        from_pixels=False,
        frame_skip=args.action_repeat,
        
    )
    options = {}
    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {"state_dim": state_dim, "action_dim": action_dim, "max_action": max_action, "discount": args.discount,
               "device": device}
   
     
    
   
    
    if args.policy == "DivTD":   #origin 对于
        import algos.DivTD as DivTD
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        kwargs["N"] = args.N
        kwargs["M"] = args.M
        kwargs["alpha"] = args.alpha
        args.batch_size=512
        policy = DivTD.DivTD(**kwargs)
    
    # Initialize policy
    elif args.policy == "TD7":
        import algos.TD7 as TD7
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        args.batch_size=256
        policy = TD7.TD7(**kwargs)
   
        
        
    
    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim,batch_size=args.batch_size,device=device,prioritized=args.prioritized)

    # Evaluate untrained policy
    all_eval = eval_policy(policy, eid,args, args.seed)
    evaluations = [all_eval[0]]
    td_errors = [all_eval[1]]
    state, done = env.reset(), False
    # state = state[0]
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    if args.save_model:
        policy.save(filename=f'./models/{file_name}_0')

    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    eval_bins_freq=[]
  
    for t in tqdm.auto.tqdm(range(int(args.max_timesteps)), f"Training {file_name}..."):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            
            action = (policy.select_action(np.array(state),deterministic=True) + np.random.normal(0, max_action * args.expl_noise,
                                                                               size=action_dim)).clip(-max_action,
                                                                                                      max_action)
        # Perform action
        next_state, reward, done, _ = env.step(action)
         
        done_bool = float(done)
        if done_bool>1:
            done_bool=1
        # Store data in replay buffer
        if args.policy == "TD7":
            policy.add(state, action, next_state, reward, done_bool)
        else:
            replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= start_timesteps:
            req = policy.train(replay_buffer, args.batch_size)
            if req is not None:
                eval_bins_freq.append(req)
           

        if done:
            # Reset environment
            state, done = env.reset(), False
            # state=state[0]
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:            
            all_eval = eval_policy(policy, eid,args, args.seed)
            evaluations.append(all_eval[0])
            td_errors.append(all_eval[1])
            np.save("results"+str(args.N)+"-"+str(args.M)+"-"+str(args.alpha)+"-"+str(args.action_repeat)+"/"+file_name, evaluations)
            if is_log:
                np.save("results"+str(args.N)+"-"+str(args.M)+"-"+str(args.alpha)+"-"+str(args.action_repeat)+"/true_pred_"+file_name, td_errors)
            
                if len(eval_bins_freq)>3:
                    np.save("results"+str(args.N)+"-"+str(args.M)+"-"+str(args.alpha)+"-"+str(args.action_repeat)+"//req"+file_name, eval_bins_freq)
            if args.save_model:
                policy.save(f"./models/{file_name}_{str(t + 1)}")
       
 