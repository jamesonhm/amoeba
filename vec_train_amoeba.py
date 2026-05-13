import time
from petree_env import PetreeDishEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
        DummyVecEnv, 
        SubprocVecEnv,
        VecFrameStack,
        VecMonitor,
)

# --- Top-Level config ---
N_ENVS = 4
N_STACK = 4

def make_training_env(n_envs=N_ENVS, n_stack=N_STACK):
    """
    Build a vectorized, optionally frame-stacked training environment.

    DummyVecEnv     - runs all envs sequentially in one process.
                    Lower overhead, better for lightweight envs (like this one)
    SubprocVecEnv   - runs each env in a separate subprocess.
                    Better for computationally heavy envs, but has IPC overhead
                    and requires the `if __name__ == "__main__" guard (see main()).
    """
    vec_env = make_vec_env(
        PetreeDishEnv,
        n_envs=n_envs,
        vec_env_cls=DummyVecEnv,    # swap to SubprocVecEnv if env is heavy
    )

    if n_stack > 1:
        vec_env = VecFrameStack(vec_env, n_stack=n_stack)

    # VecMonitor tracks episode rewards/lengths across all sub-envs
    # and surfaces them in the training logs automatically.
    vec_env = VecMonitor(vec_env)

    return vec_env

def make_play_env(n_stack=N_STACK):
    """
    Build a single-env wrapper that matches the training observation shape.
    We still wrap in DummyVecEnv + VecFrameStack so the loaded model sees 
    the same obs format it was trained on.
    """
    env = DummyVecEnv([lambda: PetreeDishEnv(render_mode="human")])

    if n_stack > 1:
        env = VecFrameStack(env, n_stack=n_stack)

    return env


def train_amoeba(timesteps=100000, n_envs=N_ENVS, n_stack=N_STACK):
    """Train Amoeba AI with PPO on a vectorized environment"""
    vec_env = make_training_env(n_envs=n_envs, n_stack=n_stack)

    # With n_envs parallel envs, PPO collects (n_steps * n_envs) transitions
    # per update. n_steps here is per-env, so effective rollout = 2048 * n_envs.

    # Create model
    model = PPO(
        "MlpPolicy",
        vec_env,
        ent_coef=0.05,
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
    model.save("vec_amoeba_model")
    print("model saved as 'vec_amoeba_model'")
    vec_env.close()
    return model

def play_trained_model(model_path="vec_amoeba_model", episodes=3, fps=10, n_stack=N_STACK):
    """Watch the trained model play in a single rendered environment."""
    print(f"Loading model: {model_path}")

    env = make_play_env(n_stack=n_stack)
    model = PPO.load(model_path, env=env)

    print(f"Watching trained agent play {episodes} episodes...")
    print("Close the window to stop early")

    frame_duration = 1.0 / fps  # seconds per frame
    scores = []
    for e in range(episodes):
        # VecEnv reset() returns obs directly = no (obs, info) tuple
        obs = env.reset()
        done = False
        print(f"Episode {e + 1}")

        while not done:
            start = time.perf_counter()

            # Get action from model
            action, _ = model.predict(obs, deterministic=True)

            # VecEnv step() returns 4 values, not 5.
            # 'terminated' and 'truncated' are combined into a single 'dones' array.
            # infos is a list of dicts, one per sub-env.
            obs, _reward, dones, infos = env.step(action)
            done = dones[0]

            # sleep for remainder of frame budget
            elapsed = time.perf_counter() - start
            time.sleep(max(0, frame_duration - elapsed))

        score = infos[0].get('score', 0)
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
        n_envs = int(sys.argv[3]) if len(sys.argv) > 3 else N_ENVS
        train_amoeba(timesteps, n_envs=n_envs)
    elif sys.argv[1] == "play":
        # Play trained model
        model_path = sys.argv[2] if len(sys.argv) > 2 else "vec_amoeba_model"
        episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        play_trained_model(model_path, episodes)
    else:
        print("Usage:")
        print("  python train_amoeba.py                    # Train with defaults")
        print("  python train_amoeba.py train 200000       # Train for 200k steps")
        print("  python train_amoeba.py train 200000 8     # Train with 8 envs")
        print("  python train_amoeba.py play               # Watch trained model")
        print("  python train_amoeba.py play snake_model 5 # Watch model play 5 games")

if __name__ == "__main__":
    main()
