import time

import jax.numpy as jnp
import numpy as np
import wandb

batch_logs = {}
log_times = []


def create_log_dict(info, config):
    to_log = {
        "episode_return": info["returned_episode_returns"],
        "episode_length": info["returned_episode_lengths"],
    }

    diffusion_keys = [
        "loss", "unweighted_loss", "accuracy", "mean_t",
        "acc_t_low", "acc_t_mid", "acc_t_high", "grad_norm",
        "action_entropy", "action_unique_frac"
    ]
    for k in diffusion_keys:
        if k in info:
            to_log[f"diffusion/{k}"] = info[k]

    sum_achievements = 0.0
    sum_val_achievements = 0.0
    has_val = False

    for k, v in info.items():
        if k.startswith("val/"):
            has_val = True
            to_log[k] = v
            if "achievements" in k.lower() and k != "val/achievements":
                sum_val_achievements += v / 100.0
        elif "achievements" in k.lower():
            to_log[k] = v
            if k != "achievements":
                sum_achievements += v / 100.0

    to_log["achievements"] = sum_achievements
    if has_val:
        to_log["val/achievements"] = sum_val_achievements

    if config.get("TRAIN_ICM") or config.get("USE_RND"):
        to_log["intrinsic_reward"] = info.get("reward_i", 0.0)
        to_log["extrinsic_reward"] = info.get("reward_e", 0.0)

        if config.get("TRAIN_ICM"):
            to_log["icm_inverse_loss"] = info.get("icm_inverse_loss", 0.0)
            to_log["icm_forward_loss"] = info.get("icm_forward_loss", 0.0)
        elif config.get("USE_RND"):
            to_log["rnd_loss"] = info.get("rnd_loss", 0.0)

    return to_log


def batch_log(update_step, log, config):
    update_step = int(update_step)
    if update_step not in batch_logs:
        batch_logs[update_step] = []

    batch_logs[update_step].append(log)

    if len(batch_logs[update_step]) == config.get("NUM_REPEATS", 1):
        agg_logs = {}
        for key in batch_logs[update_step][0]:
            agg = []
            if key in ["goal_heatmap"]:
                agg = [batch_logs[update_step][0][key]]
            else:
                for i in range(config.get("NUM_REPEATS", 1)):
                    # Use .get() to prevent KeyErrors if repeats are out of sync
                    val = batch_logs[update_step][i].get(key, float("nan"))
                    if not jnp.isnan(val):
                        agg.append(val)

            if len(agg) > 0:
                if key in [
                    "episode_length",
                    "episode_return",
                    "exploration_bonus",
                    "e_mean",
                    "e_std",
                    "rnd_loss",
                    "diffusion/loss",
                    "diffusion/unweighted_loss",
                    "diffusion/accuracy",
                    "diffusion/acc_t_low",
                    "diffusion/acc_t_mid",
                    "diffusion/acc_t_high",
                    "diffusion/action_entropy",
                    "diffusion/grad_norm"
                ] or key.startswith("val/") or "achievement" in key.lower():
                    agg_logs[key] = np.mean(agg)
                else:
                    agg_logs[key] = np.array(agg)

        log_times.append(time.time())

        if config.get("DEBUG"):
            if len(log_times) == 1:
                print("Started logging")
            elif len(log_times) > 1:
                dt = log_times[-1] - log_times[-2]
                steps_between_updates = (
                        config["NUM_STEPS"] * config["NUM_ENVS"] * config.get("NUM_REPEATS", 1)
                )
                sps = steps_between_updates / dt
                agg_logs["sps"] = sps

        wandb.log(agg_logs)

        # Clear buffer to prevent memory leaks
        del batch_logs[update_step]