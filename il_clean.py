import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import shutil
import argparse
import logging
import random
from glob import glob
from pathlib import Path
from time import strftime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader, Dataset

import gymnasium as gym
from gymnasium.spaces import Box

import onnx
from onnx2pytorch import ConvertModel

import wandb
from tqdm import tqdm

# silence gymnasium warnings about deprecated kwargs etc
logging.getLogger("gymnasium").setLevel(logging.ERROR)


# ==========================================================
# policy network
# ==========================================================

class PolicyNetwork(nn.Module):
    """nature dqn style cnn for 84x84 grayscale inputs
    outputs n_units_out raw logits one per discrete action
    dropout is applied between the hidden fc and the output layer
    """
    def __init__(self, n_units_out, dropout_p=0.3):
        super().__init__()
        # three conv layers downsample 84x84 to a 64x7x7 feature map
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # flatten gives 64 times 7 times 7 equal to 3136 features
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.dropout = nn.Dropout(dropout_p)
        self.fc_out = nn.Linear(512, n_units_out)

    def forward(self, x):
        # x has shape batch by 1 by 84 by 84 with values in zero to one
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # collapse channel and spatial dims keeping batch dim
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # raw logits the agent wraps them in a categorical distribution
        return self.fc_out(x)


# ==========================================================
# dataset
# ==========================================================

class DemonstrationDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = sorted(glob(f"{data_dir}/*.npz"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        # add a channel dim so the shape becomes 1 by 84 by 84
        state = data["state"][np.newaxis, ...].astype(np.float32)
        action = data["action"]
        return state / 255.0, action.item()

    def append(self, states, actions):
        offset = len(self) + 1
        for i in range(len(states)):
            filename = f"{self.data_dir}/{offset+i:06}.npz"
            np.savez_compressed(filename, state=states[i], action=actions[i].astype(np.int32))
            self.files.append(filename)


# ==========================================================
# environment wrappers and agent
# ==========================================================

class CropObservation(gym.ObservationWrapper):
    # crops the raw rgb frame to remove the score bar at the bottom
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = shape
        obs_shape = self.shape + env.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        return observation[:self.shape[0], :self.shape[1]]


class RecordState(gym.Wrapper):
    def __init__(self, env, reset_clean=True):
        super().__init__(env)
        assert env.render_mode is not None
        self.frame_list = []
        self.reset_clean = reset_clean

    def step(self, action, **kwargs):
        output = self.env.step(action, **kwargs)
        self.frame_list.append(output[0])
        return output

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        if self.reset_clean:
            self.frame_list = []
        self.frame_list.append(result[0])
        return result

    def render(self):
        # return all buffered frames and clear the buffer for next time
        frames = self.frame_list
        self.frame_list = []
        return frames


class Agent:
    def __init__(self, model, device, deterministic=False):
        self.model = model
        self.device = device
        self.deterministic = deterministic

    def select_action(self, state):
        with torch.no_grad():
            state_arr = np.asarray(state, dtype=np.float32)
            state_t = torch.from_numpy(state_arr).unsqueeze(0).to(self.device) / 255.0
            logits = self.model(state_t)
            if isinstance(logits, tuple):
                logits = logits[0]
            if self.deterministic:
                # take the highest-probability action no sampling noise
                return logits.argmax(dim=1).cpu().numpy()[0]
            probs = Categorical(logits=logits)
            return probs.sample().cpu().numpy()[0]


def make_env(seed, video_dir=None, capture_video=False):
    env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=False)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if capture_video:
        assert video_dir is not None, "video_dir is required when capture_video is true"
        # explicit trigger so every episode of this env instance is recorded
        env = gym.wrappers.RecordVideo(env, video_folder=video_dir,
                                       episode_trigger=lambda ep: True)
    env = CropObservation(env, (84, 96))
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = RecordState(env, reset_clean=True)
    # framestack of 4 is required so the expert can take the same obs
    env = gym.wrappers.FrameStackObservation(env, 4)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def run_episode(agent, video_dir=None, capture_video=False, seed=None, show_progress=False):
    env = make_env(seed=seed, video_dir=video_dir, capture_video=capture_video)
    state, _ = env.reset()
    score = 0.0
    done = False
    pbar = tqdm(desc="score 0", leave=False) if show_progress else None
    while not done:
        # bc policy takes a single grayscale frame so use the last of the stack
        action = agent.select_action(state[-1][np.newaxis, ...])
        state, reward, terminated, truncated, _ = env.step(action)
        score += reward
        done = terminated or truncated
        if pbar is not None:
            pbar.update()
            pbar.set_description(f"score {score:.2f}")
    env.close()
    if pbar is not None:
        pbar.close()
    return score


def find_latest_mp4(video_dir):
    # helper to grab the latest recorded mp4 for wandb upload
    mp4s = sorted(glob(os.path.join(video_dir, "*.mp4")), key=os.path.getmtime)
    return mp4s[-1] if mp4s else None


def save_as_onnx(torch_model, sample_input, model_path):
    torch_model.eval()
    torch.onnx.export(
        torch_model,
        sample_input,
        f=model_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        external_data=False,
        dynamo=False,
    )
    # sanity check that the export is parseable
    ConvertModel(onnx.load(model_path))
    print(f"onnx export ok at {model_path}")


# ==========================================================
# training and evaluation helpers
# ==========================================================

@torch.no_grad()
def evaluate_on_val(model, val_loader, device, loss_fn):
    """mean cross entropy loss and top one accuracy on the val set"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for states, actions in val_loader:
        states = states.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        logits = model(states)
        loss = loss_fn(logits, actions)
        total_loss += loss.item() * states.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == actions).sum().item()
        total_samples += states.size(0)
    return total_loss / total_samples, total_correct / total_samples


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================================
# train bc mode
# ==========================================================

def train_bc(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}")
    # should make runtime faster (input shape does not change)
    torch.backends.cudnn.benchmark = True

    set_seeds(args.seed)

    # unique directory for checkpoints and videos
    run_name = "bc-" + strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = Path(args.output_dir) / run_name
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    print(f"run dir {out_dir}")

    # init wandb
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
        dir=str(out_dir),
    )

    # datasets and dataloaders
    train_set = DemonstrationDataset(os.path.join(args.data_dir, "train"))
    val_set = DemonstrationDataset(os.path.join(args.data_dir, "val"))
    print(f"train samples {len(train_set)} val samples {len(val_set)}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=True,
        drop_last=False, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False,
        drop_last=False, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # model loss optimizer scheduler
    model = PolicyNetwork(n_units_out=5, dropout_p=args.dropout).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params {num_params}")
    # spot exploding or vanishing grads
    wandb.watch(model, log="gradients", log_freq=200)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience,
    )

    # keep one sample for the onnx export at the end
    sample_state, _ = train_set[0]
    sample_state = torch.from_numpy(sample_state).unsqueeze(0).to(device)

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss_sum = 0.0
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}", leave=False)
        for states, actions in pbar:
            states = states.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)

            logits = model(states)
            loss = loss_fn(logits, actions)

            # entropy is a signal for how confident the policy is
            with torch.no_grad():
                entropy = Categorical(logits=logits).entropy().mean().item()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            wandb.log({
                "train/loss": loss.item(),
                "train/entropy": entropy,
                "train/lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch + 1,
            }, step=global_step)

            epoch_loss_sum += loss.item() * states.size(0)
            n_seen += states.size(0)
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        train_loss_avg = epoch_loss_sum / n_seen

        # validation pass at the end of each epoch
        val_loss, val_acc = evaluate_on_val(model, val_loader, device, loss_fn)
        scheduler.step(val_loss)

        wandb.log({
            "epoch/train_loss": train_loss_avg,
            "epoch/val_loss": val_loss,
            "epoch/val_accuracy": val_acc,
            "epoch": epoch + 1,
        }, step=global_step)
        print(f"epoch {epoch+1} train_loss {train_loss_avg:.4f} "
              f"val_loss {val_loss:.4f} val_acc {val_acc:.4f}")

        # save best checkpoint based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "args": vars(args),
            }, out_dir / "checkpoints" / "best.pt")

    # always save the very last checkpoint too
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": vars(args),
    }, out_dir / "checkpoints" / "last.pt")

    # reload the best weights and export onnx for the submission server
    print("loading best for onnx export")
    best = torch.load(out_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    save_as_onnx(model, sample_state, str(out_dir / "checkpoints" / "model.onnx"))

    # final environment evaluation with the best model
    print("running final evaluation in env")
    evaluate_in_env(args, model=model, device=device,
                    video_dir=str(out_dir / "videos"))

    wandb.finish()


# ==========================================================
# evaluate mode
# ==========================================================

def evaluate_in_env(args, model=None, device=None, video_dir=None):
    """
    run n eval episodes in the env and log scores plus the first video to wandb
    when called as a standalone mode it loads weights from a checkpoint file
    """
    standalone = model is None
    if standalone:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device {device}")
        torch.backends.cudnn.benchmark = True
        set_seeds(args.seed)

        run_name = "eval-" + strftime("%Y-%m-%dT%H-%M-%S")
        out_dir = Path(args.output_dir) / run_name
        video_dir = str(out_dir / "videos")
        Path(video_dir).mkdir(parents=True, exist_ok=True)
        print(f"run dir {out_dir}")

        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            dir=str(out_dir),
        )

        # use the dropout value from the saved args if present otherwise from cli
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        saved_args = ckpt.get("args", {})
        dropout_p = saved_args.get("dropout", args.dropout)
        model = PolicyNetwork(n_units_out=5, dropout_p=dropout_p).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"loaded checkpoint {args.checkpoint}")

    model.eval()
    agent = Agent(model, device, deterministic=args.deterministic)

    scores = []
    for ep in tqdm(range(args.eval_episodes), desc="eval episodes"):
        # only record the first episode to keep wandb uploads small
        capture = (ep == 0)
        score = run_episode(agent, video_dir=video_dir,
                            capture_video=capture, seed=args.seed + ep)
        scores.append(score)
        wandb.log({"eval/episode_score": score, "eval/episode": ep})
        if capture:
            mp4 = find_latest_mp4(video_dir)
            if mp4 is not None:
                wandb.log({"eval/video": wandb.Video(mp4, fps=30, format="mp4")})
        print(f"episode {ep} score {score:.2f}")

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    print(f"mean {mean_score:.2f} std {std_score:.2f}")
    wandb.log({"eval/mean_score": mean_score, "eval/std_score": std_score})

    if standalone:
        wandb.finish()
    return mean_score, std_score


# ==========================================================
# dagger
# ==========================================================

def load_expert(expert_path, device):
    """
    load a pretrained expert from an onnx file
    the expert is expected to take 4 stacked grayscale frames as input
    """
    onnx_model = onnx.load(expert_path)
    expert = ConvertModel(onnx_model).to(device).eval()
    return expert


def init_dagger_dataset_dir(source_dir, target_dir):
    """
    initialize the dagger train folder by hard linking from the bc train folder
    hard linking is near instant and avoids duplicating disk usage
    if the target already has files we reuse them to support resuming a run
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    existing = list(target.glob("*.npz"))
    if existing:
        print(f"dagger dir already has {len(existing)} files reusing")
        return
    src = Path(source_dir)
    count = 0
    for f in src.glob("*.npz"):
        try:
            os.link(f, target / f.name)
        except OSError:
            # fall back to a real copy if hard linking is not supported
            shutil.copy(f, target / f.name)
        count += 1
    print(f"linked {count} files from {source_dir} to {target_dir}")


def beta_schedule(iteration, decay):
    # exponential decay starting at 1 at iteration 0
    return decay ** iteration


def dagger_rollout(model, expert_agent, device, beta, seed,
                   video_dir=None, capture_video=False, max_warmup=0):
    """
    run one rollout collecting state expert action pairs
    actions to execute come from a mixture of expert and student via beta
    every visited state is labeled with the experts action regardless

    if max_warmup is greater than 0 the expert drives for a random
    number of steps in 0 to max_warmup before data collection begins
    this diversifies the starting state of each collected trajectory
    if the episode ends during warmup empty arrays are returned and the
    caller should treat the rollout as skipped
    """
    student_agent = Agent(model, device)
    env = make_env(seed=seed, video_dir=video_dir, capture_video=capture_video)
    state, _ = env.reset()

    # optional expert warmup to skip past the always identical starting state
    # the warmup length is randomized so different rollouts begin at different points
    if max_warmup > 0:
        warmup_steps = np.random.randint(0, max_warmup + 1)
        for _ in range(warmup_steps):
            warmup_action = expert_agent.select_action(state)
            state, _, terminated, truncated, _ = env.step(warmup_action)
            if terminated or truncated:
                # episode finished during warmup nothing to collect
                env.close()
                return np.array([]), np.array([]), 0.0

    collected_states = []
    collected_actions = []
    score = 0.0
    done = False

    while not done:
        # always ask the expert what it would do - the label
        expert_action = expert_agent.select_action(state)

        # decide who actually steps the environment this turn
        if random.random() < beta:
            action_to_execute = expert_action
        else:
            action_to_execute = student_agent.select_action(state[-1][np.newaxis, ...])

        # save the latest single frame paired with the expert label
        collected_states.append(state[-1])
        collected_actions.append(expert_action)

        state, reward, terminated, truncated, _ = env.step(action_to_execute)
        score += reward
        done = terminated or truncated

    env.close()
    return np.array(collected_states), np.array(collected_actions), score


def train_dagger(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}")
    torch.backends.cudnn.benchmark = True
    set_seeds(args.seed)

    # build a unique run directory mirroring train_bc
    run_name = "dagger-" + strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = Path(args.output_dir) / run_name
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    print(f"run dir {out_dir}")

    # init wandb live monitoring
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
        dir=str(out_dir),
    )

    # set up the dagger training folder by hard linking from the original bc train data
    # this preserves the original folder and gives a clean place to append new samples
    dagger_train_dir = out_dir / "dagger_train"
    init_dagger_dataset_dir(
        os.path.join(args.data_dir, "train"),
        str(dagger_train_dir),
    )
    train_set = DemonstrationDataset(str(dagger_train_dir))
    val_set = DemonstrationDataset(os.path.join(args.data_dir, "val"))
    print(f"initial train samples {len(train_set)} val samples {len(val_set)}")
    assert len(train_set) > 0, "dagger train set is empty check --data_dir"

    # load the bc trained student to start from
    saved_args = torch.load(args.checkpoint, map_location=device, weights_only=False).get("args", {})
    dropout_p = saved_args.get("dropout", args.dropout)
    model = PolicyNetwork(n_units_out=5, dropout_p=dropout_p).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"loaded bc student from {args.checkpoint}")

    # load the expert and wrap it in the same agent interface as the student
    expert = load_expert(args.expert_path, device)
    expert_agent = Agent(expert, device)
    print(f"loaded expert from {args.expert_path}")

    # same as in bc
    wandb.watch(model, log="gradients", log_freq=200)

    # fresh optimizer at a smaller lr since the model is already trained
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.dagger_lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # a fixed val loader is fine since val set does not change
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False,
        drop_last=False, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # keep a sample for the onnx export at the end
    sample_state_arr, _ = train_set[0]
    sample_state = torch.from_numpy(sample_state_arr).unsqueeze(0).to(device)
    
    # track the best mean env score across iterations env eval is the real selector
    best_dagger_score = -float("inf")
    global_step = 0

    for iteration in range(args.dagger_iterations):
        beta = max(args.beta_min, beta_schedule(iteration, args.beta_decay))
        print(f"\n=== dagger iter {iteration+1}/{args.dagger_iterations} beta {beta:.3f} ===")

        # 1 collect a rollout under mixed policy
        model.eval()
        capture = (iteration % args.dagger_video_every == 0)
        # 1a) screen candidate seeds with the current student to find hard tracks
        # using deterministic action selection for stable scoring
        # the worst rollouts_per_iter tracks become the dagger targets this iteration
        print(f"screening {args.hard_seed_candidates} candidate seeds for hard tracks")
        screen_agent = Agent(model, device, deterministic=True)
        cand_base = args.seed + 50000 + iteration * args.hard_seed_candidates
        seed_scores = []
        for cand_idx in range(args.hard_seed_candidates):
            cand_seed = cand_base + cand_idx
            sc = run_episode(screen_agent, seed=cand_seed)
            seed_scores.append((sc, cand_seed))

        # sort ascending pick the rollouts_per_iter worst seeds
        seed_scores.sort(key=lambda x: x[0])
        hard_pairs = seed_scores[: args.rollouts_per_iter]
        hard_seeds = [s for _, s in hard_pairs]
        hard_scores = [sc for sc, _ in hard_pairs]
        all_screen_scores = [sc for sc, _ in seed_scores]
        print(f"hard seeds {hard_seeds} screen scores "
              f"{[f'{s:.1f}' for s in hard_scores]}")

        wandb.log({
            "dagger/screen_mean": float(np.mean(all_screen_scores)),
            "dagger/screen_min": float(np.min(all_screen_scores)),
            "dagger/screen_max": float(np.max(all_screen_scores)),
            "dagger/screen_hard_mean": float(np.mean(hard_scores)),
            "dagger/iteration": iteration + 1,
        }, step=global_step)

        # 1b) collect dagger rollouts on those hard seeds with random expert warmup
        # warmup makes each rollout start at a different point in the track
        all_states = []
        all_actions = []
        rollout_scores = []
        for r, hard_seed in enumerate(hard_seeds):
            r_capture = capture and (r == 0)
            s, a, sc = dagger_rollout(
                model, expert_agent, device, beta,
                seed=hard_seed,
                video_dir=str(out_dir / "videos"),
                capture_video=r_capture,
                max_warmup=args.warmup_steps,
            )
            # episode finished during warmup skip this rollout silently
            if len(s) == 0:
                print(f"rollout on seed {hard_seed} aborted during warmup skipping")
                continue
            all_states.append(s)
            all_actions.append(a)
            rollout_scores.append(sc)

        # handle the edge case where every rollout aborted during warmup
        if len(all_states) > 0:
            states = np.concatenate(all_states)
            actions = np.concatenate(all_actions)
            rollout_score = float(np.mean(rollout_scores))
        else:
            print("all rollouts aborted during warmup no data added this iteration")
            states = np.array([])
            actions = np.array([])
            rollout_score = 0.0
        print(f"rollout score {rollout_score:.2f} collected {len(states)} new samples")

        # 2 append the new state expert action pairs to the dataset buffer
        # skip if all rollouts aborted during warmup
        if len(states) > 0:
            train_set.append(states, actions)

        wandb.log({
            "dagger/iteration": iteration + 1,
            "dagger/beta": beta,
            "dagger/rollout_score": rollout_score,
            "dagger/dataset_size": len(train_set),
            "dagger/new_samples": len(states),
        }, step=global_step)
        if capture:
            mp4 = find_latest_mp4(str(out_dir / "videos"))
            if mp4 is not None:
                wandb.log(
                    {"dagger/rollout_video": wandb.Video(mp4, fps=30, format="mp4")},
                    step=global_step,
                )

        # 3 rebuild the train loader so it picks up the newly appended files
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size,
            num_workers=args.num_workers, shuffle=True,
            drop_last=False, pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

        # 4 train for a few epochs on the aggregated dataset
        for epoch in range(args.dagger_epochs_per_iter):
            model.train()
            epoch_loss_sum = 0.0
            n_seen = 0
            pbar = tqdm(
                train_loader,
                desc=f"iter {iteration+1} epoch {epoch+1}/{args.dagger_epochs_per_iter}",
                leave=False,
            )
            for batch_states, batch_actions in pbar:
                batch_states = batch_states.to(device, non_blocking=True)
                batch_actions = batch_actions.to(device, non_blocking=True)

                logits = model(batch_states)
                loss = loss_fn(logits, batch_actions)

                # entropy monitoring
                with torch.no_grad():
                    entropy = Categorical(logits=logits).entropy().mean().item()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                wandb.log({
                    "dagger/train_loss": loss.item(),
                    "dagger/entropy": entropy,
                    "dagger/lr": optimizer.param_groups[0]["lr"],
                }, step=global_step)

                epoch_loss_sum += loss.item() * batch_states.size(0)
                n_seen += batch_states.size(0)
                global_step += 1
                pbar.set_postfix(loss=f"{loss.item():.3f}")

            train_loss_avg = epoch_loss_sum / n_seen
            val_loss, val_acc = evaluate_on_val(model, val_loader, device, loss_fn)

            wandb.log({
                "dagger/epoch_train_loss": train_loss_avg,
                "dagger/epoch_val_loss": val_loss,
                "dagger/epoch_val_accuracy": val_acc,
                "dagger/iteration": iteration + 1,
                "dagger/epoch": epoch + 1,
            }, step=global_step)
            print(f"iter {iteration+1} epoch {epoch+1} "
                  f"train_loss {train_loss_avg:.4f} "
                  f"val_loss {val_loss:.4f} val_acc {val_acc:.4f}")

        # run a proper env evaluation to pick the best dagger model
        # fixed set of 10 seeds so iterations are compared on the same tracks
        model.eval()
        # deterministic argmax for selection - pick the truly best model
        # not one that got lucky with sampling
        eval_agent = Agent(model, device, deterministic=True)
        iter_scores = []
        for ep in range(args.dagger_eval_episodes):
            # seeds offset by 10000 so they never overlap with rollout or final eval seeds
            ep_seed = args.seed + 10000 + ep
            iter_scores.append(run_episode(eval_agent, seed=ep_seed))

        mean_iter = float(np.mean(iter_scores))
        std_iter = float(np.std(iter_scores))
        min_iter = float(np.min(iter_scores))
        max_iter = float(np.max(iter_scores))
        print(f"iter {iteration+1} env eval mean {mean_iter:.2f} std {std_iter:.2f} "
              f"min {min_iter:.2f} max {max_iter:.2f}")

        wandb.log({
            "dagger/iter_eval_mean": mean_iter,
            "dagger/iter_eval_std": std_iter,
            "dagger/iter_eval_min": min_iter,
            "dagger/iter_eval_max": max_iter,
            "dagger/iteration": iteration + 1,
        }, step=global_step)

        # save best_dagger checkpoint based on mean env score
        if mean_iter > best_dagger_score:
            best_dagger_score = mean_iter
            torch.save({
                "iteration": iteration + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "env_eval_mean": mean_iter,
                "env_eval_std": std_iter,
                "args": vars(args),
            }, out_dir / "checkpoints" / "best_dagger.pt")
            print(f"new best dagger score {best_dagger_score:.2f}")
            wandb.log({"dagger/best_dagger_score": best_dagger_score},
                      step=global_step)


    # always save the last checkpoint too
    torch.save({
        "iteration": args.dagger_iterations,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }, out_dir / "checkpoints" / "last.pt")

    # reload the model that scored highest in env evaluation across iterations
    print(f"loading best_dagger (env score {best_dagger_score:.2f}) for onnx export")
    best = torch.load(out_dir / "checkpoints" / "best_dagger.pt",
                      map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    save_as_onnx(model, sample_state, str(out_dir / "checkpoints" / "best_dagger.onnx"))

    # final environment evaluation with the best model
    print("running final evaluation in env")
    evaluate_in_env(args, model=model, device=device,
                    video_dir=str(out_dir / "videos"))

    wandb.finish()


# ==========================================================
# entry point
# ==========================================================

def parse_args():
    p = argparse.ArgumentParser(description="bc training for car racing v3")
    p.add_argument("--gpu", type=str, default="0")

    p.add_argument("--mode", choices=["train_bc", "train_dagger", "evaluate"], default="train_bc",
                   help="train_bc trains the policy from scratch evaluate loads a checkpoint and train_dagger implements dagger algo with expert and trained policy")

    # data and io
    p.add_argument("--data_dir", default=".",
                   help="directory containing train and val subfolders of npz files")
    p.add_argument("--output_dir", default="runs",
                   help="where to put checkpoints and videos per run")
    p.add_argument("--checkpoint", default=None,
                   help="path to a pt checkpoint required by evaluate mode")

    # training hyperparameters
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr_patience", type=int, default=3,
                   help="reduce lr on plateau patience measured in epochs")

    # evaluation
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--deterministic", action="store_true",
               help="use argmax instead of sampling at evaluation time")

    # dagger specific
    p.add_argument("--expert_path", default=None,
                   help="path to the expert onnx file required for train_dagger mode")
    p.add_argument("--dagger_iterations", type=int, default=10,
                   help="number of dagger outer iterations")
    p.add_argument("--dagger_epochs_per_iter", type=int, default=5,
                   help="number of training epochs after each rollout")
    p.add_argument("--dagger_lr", type=float, default=1e-4,
                   help="learning rate for dagger fine tuning smaller than bc lr")
    p.add_argument("--beta_decay", type=float, default=0.5,
                   help="exponential decay base for beta the expert mixing probability")
    p.add_argument("--beta_min", type=float, default=0.0,
                   help="floor for beta keeps some expert mixing if greater than zero")
    p.add_argument("--dagger_video_every", type=int, default=1,
                   help="record a rollout video every n iterations")
    p.add_argument("--dagger_eval_episodes", type=int, default=10,
                   help="number of env episodes used to score each iteration for model selection")
    p.add_argument("--rollouts_per_iter", type=int, default=1,
                   help="number of episodes collected per dagger iteration")
    p.add_argument("--hard_seed_candidates", type=int, default=20,
                   help="number of seeds to screen each iteration we pick the worst rollouts_per_iter")
    p.add_argument("--warmup_steps", type=int, default=200,
                   help="max expert warmup steps before data collection 0 disables warmup")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", default="carracing-imitation")

    return p.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", f"{args.gpu}")

    if args.mode == "train_bc":
        train_bc(args)
    elif args.mode == "evaluate":
        assert args.checkpoint is not None, "evaluate mode requires --checkpoint"
        evaluate_in_env(args)
    elif args.mode == "train_dagger":
        assert args.checkpoint is not None, "train_dagger requires --checkpoint with a bc model"
        assert args.expert_path is not None, "train_dagger requires --expert_path"
        train_dagger(args)


if __name__ == "__main__":
    main()