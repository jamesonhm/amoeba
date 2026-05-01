import numpy as np
import time
from petree_env import PetreeDishEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

def train_amoeba(timesteps=100000, render=False):
    """Train Amoeba AI with PPO"""
    # Create env
    render_mode = "human" if render else None
    env = PetreeDishEnv(render_mode=render_mode)

    # Check env
    check_env(env)
    # Manual inspect
    obs, info = env.reset()
    print(f"reset obs: {obs}")
    print(f"reset info: {info}")
    print(f"food has nan: {np.any(np.isnan(obs["food"]))}")
    print(f"wall has nan: {np.any(np.isnan(obs["wall"]))}")

    # Create model
    model = PPO(
        "MultiInputPolicy",
        env,
        ent_coef=0.01,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
    )

    # Train the model
    print("Starting Training...")
    model.learn(total_timesteps=timesteps)

    # Save the model
    model.save("amoeba_model")
    print("model saved as 'amoeba_model'")
    # env.close()
    return model

def play_trained_model(model_path="amoeba_model", episodes=3, fps=10):
    """Watch the trained model play"""
    print(f"Loading model: {model_path}")

    env = PetreeDishEnv(render_mode="human")

    # Load Model
    model = PPO.load(model_path, env=env)

    print(f"Watching trained agent play {episodes} episodes...")
    print("Close the window to stop early")

    frame_duration = 1.0 / fps  # seconds per frame
    scores = []
    for e in range(episodes):
        obs, info = env.reset()
        done = False
        print(f"Episode {e + 1}")

        while not done:
            start = time.perf_counter()

            # Get action from model
            action, _ = model.predict(obs, deterministic=True)

            # Take step
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc

            # sleep for remainder of frame budget
            elapsed = time.perf_counter() - start
            time.sleep(max(0, frame_duration - elapsed))

        score = info.get('score', 0)
        scores.append(score)
        print(f'Score: {score}')

    env.close()

    return scores

def main():
    import sys

    if len(sys.argv) == 1:
        # Default training
        train_amoeba()
    elif sys.argv[1] == "train":
        # Custom training
        timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
        render = "--render" in sys.argv
        train_amoeba(timesteps, render)
    elif sys.argv[1] == "play":
        # Play trained model
        model_path = sys.argv[2] if len(sys.argv) > 2 else "amoeba_model"
        episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        play_trained_model(model_path, episodes)
    else:
        print("Usage:")
        print("  python train_amoeba.py                    # Train with defaults")
        print("  python train_amoeba.py train 200000       # Train for 200k steps")
        print("  python train_amoeba.py train 50000 --render # Train with rendering")
        print("  python train_amoeba.py play               # Watch trained model")
        print("  python train_amoeba.py play snake_model 3 # Watch model play 3 games")

if __name__ == "__main__":
    main()
