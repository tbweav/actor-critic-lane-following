import torch
from torch import nn
import numpy as np

from actor import Actor
from critic import Critic

from buffer import Buffer
from rollout import Rollout


def get_actor_critic(state_dim=3, action_dim=1, action_value_dim=1, hlayers=2, hwidth=64):
    """Initialize actor critic networks (live and target)"""

    live_actor = Actor(state_dim, action_dim, hlayers, hwidth)
    live_critic = Critic(state_dim, action_dim, action_value_dim, hlayers, hwidth)

    target_actor = Actor(state_dim, action_dim, hlayers, hwidth)
    target_critic = Critic(state_dim, action_dim, action_value_dim, hlayers, hwidth)
    target_actor.load_state_dict(live_actor.state_dict())
    target_critic.load_state_dict(live_critic.state_dict())

    return live_actor, live_critic, target_actor, target_critic

def get_optimizers(live_actor, live_critic, lr=1e-3):
    """Initialize optimizers"""

    actor_optimizer = torch.optim.Adam(live_actor.parameters(), lr=lr)
    critic_optimizer = torch.optim.Adam(live_critic.parameters(), lr=lr)
    return actor_optimizer, critic_optimizer

def soft_update(target_model, live_model, tau):
    """Perform soft update of target model parameters. Lags behind live model by factor of tau."""

    for target_param, live_param in zip(target_model.parameters(), live_model.parameters()):
        target_param.data.copy_(tau * live_param.data + (1.0 - tau) * target_param.data)

def update_models(live_actor, live_critic, target_actor, target_critic, actor_optimizer, critic_optimizer, buffer, batch_size, gamma, tau):
    """Update live actor and critic models using sampled batch from buffer. Also performs soft update of target models."""

    states, actions, rewards, next_states, dones = buffer.sample(batch_size) #Random sample from buffer

    with torch.no_grad():
        next_actions = target_actor(next_states)
        q_target_next = target_critic(next_states, next_actions)
        q_target = rewards + gamma * (1.0 - dones) * q_target_next

    q_pred = live_critic(states, actions)
    critic_loss = nn.MSELoss()(q_pred, q_target)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    actor_actions = live_actor(states)
    actor_loss = -live_critic(states, actor_actions).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    soft_update(target_actor, live_actor, tau)
    soft_update(target_critic, live_critic, tau)

    return actor_loss.item(), critic_loss.item()


def evaluate_policy(rollout, actor, num_rollouts=8):
    eval_returns = []
    for _ in range(num_rollouts):
        ret = rollout.perform_rollout(
            actor,
            explore=False,
            use_uncertainty=False,
            store_transition=False,
            randomize_progress=True,
        )
        eval_returns.append(ret)
    return float(np.mean(eval_returns))

def train():
    episodes = 400
    eval_every = 10
    eval_rollouts = 8
    min_buffer_size = 1000
    max_buffer_size = 25000
    batch_size = 64
    gamma = 0.99
    tau = 0.01

    policy = "policy.pt"
    logs_path = "training_logs.pt"

    live_actor, live_critic, target_actor, target_critic = get_actor_critic()
    actor_optimizer, critic_optimizer = get_optimizers(live_actor, live_critic)
    buffer = Buffer(min_size=min_buffer_size, max_size=max_buffer_size)
    rollout = Rollout(buffer, randomize_progress=True)

    train_rewards = []
    eval_rewards = []
    actor_losses = []
    critic_losses = []
    best_eval_reward = -float("inf")
    best_eval_episode = 0

    while len(buffer) < min_buffer_size:
        rollout.perform_rollout(live_actor)

    for episode in range(1, episodes+1):
        rollout.reset()
        episode_reward = 0.0
        done = False

        while not done:
            reward, done = rollout.step(live_actor)
            episode_reward += reward
            actor_loss, critic_loss = update_models(
                live_actor,
                live_critic,
                target_actor,
                target_critic,
                actor_optimizer,
                critic_optimizer,
                buffer,
                batch_size,
                gamma,
                tau,
            )

        train_rewards.append(episode_reward)
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)

        if episode % eval_every == 0:
            eval_reward = evaluate_policy(rollout, live_actor, num_rollouts=eval_rollouts)
            eval_rewards.append((episode, eval_reward))

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_eval_episode = episode
                torch.save(live_actor.state_dict(), policy)

            print(
                f"Episode {episode:3d} | train reward {episode_reward:8.3f} | eval reward {eval_reward:8.3f}"
                f" | actor loss {actor_loss:8.4f} | critic loss {critic_loss:8.4f} | buffer {len(buffer)}"
            )

    torch.save(
        {
            "train_rewards": train_rewards,
            "eval_rewards": eval_rewards,
            "actor_losses": actor_losses,
            "critic_losses": critic_losses,
            "best_eval_reward": best_eval_reward,
            "best_eval_episode": best_eval_episode,
        },
        logs_path,
    )


if __name__ == "__main__":
    train()

