import torch
import matplotlib.pyplot as plt


def main(logs_path="training_logs.pt"):
    logs = torch.load(logs_path, weights_only=False)

    train_rewards = logs["train_rewards"]
    eval_rewards = logs["eval_rewards"]
    critic_losses = logs["critic_losses"]

    # Reward plot: train and eval rewards
    train_episodes = list(range(1, len(train_rewards) + 1))
    eval_episodes = [ep for ep, _ in eval_rewards]
    eval_values = [val for _, val in eval_rewards]

    plt.figure(figsize=(8, 5))
    plt.plot(train_episodes, train_rewards, label="Train Reward")
    plt.plot(eval_episodes, eval_values, label="Eval Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward vs Episode")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reward_plot.png", dpi=150)
    plt.close()

    # Loss plot: critic loss only
    loss_episodes = list(range(1, len(critic_losses) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(loss_episodes, critic_losses, label="Critic Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Critic Loss vs Episode")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_plot.png", dpi=150)
    plt.close()

    print("Saved reward_plot.png and loss_plot.png")


if __name__ == "__main__":
    main()
