# Morning report — Sun Jul 19 04:36:11 UTC 2026

[04:36:14] PHASE 1 start: A2 evals (existing checkpoint + base, all vocabs)
[04:36:14] urn A2 qwen-rl-base-q8:latest (4 vocabs x 24 seeds)
[04:40:04]   urn qwen-rl-base-q8:latest OK
[04:40:04] urn_tool A2 qwen-rl-base-q8:latest (4 vocabs x 24 seeds)
[04:47:50]   urn_tool qwen-rl-base-q8:latest OK
[04:47:50] urn A2 qwen-rl-urn-final:latest (4 vocabs x 24 seeds)
[04:50:12]   urn qwen-rl-urn-final:latest OK
[04:50:12] urn_tool A2 qwen-rl-urn-final:latest (4 vocabs x 24 seeds)
[05:01:50]   urn_tool qwen-rl-urn-final:latest OK
[05:01:50] PHASE 1 done
[05:01:50] PHASE 2 start: A2 retrain to 20 steps (resuming gated run in runs/rl_urn_pilot_a2)
[09:16:51] PHASE 2 training exited rc=0; last log lines:
  GRPO update: 122s  mean_reward(balls)=39.79 (std=9.16 min=2.00 max=52.00)  [baselines: eager=35.76 wait2=43.12]  loss=-0.6440  mean_kl=-0.043  critic_loss=0.2195 (500 iters)  mean_advantage=0.188  mean|A|=5.02  first_sight=14%  lateness=1.545  group_std(mean=5.386 max=18.886)
  step 19 total: 840s

reward history (step 0 = first completed step): [37.75, 36.95, 35.96, 37.44, 38.28, 36.7, 37.34, 38.89, 37.14, 36.38, 36.47, 35.93, 38.17, 38.64, 40.44, 40.03, 38.27, 39.54, 40.63, 39.79]
behavior history (first_sight% / lateness): [(69, 0.45), (71, 0.53), (71, 0.42), (68, 0.51), (70, 0.42), (68, 0.57), (69, 0.46), (67, 0.45), (63, 0.58), (61, 0.58), (58, 0.62), (56, 0.77), (46, 0.95), (49, 0.91), (41, 0.97), (26, 1.37), (25, 1.3), (22, 1.42), (15, 1.36), (14, 1.55)]
[09:16:51] PHASE 3 start: merge + quantize + eval new A2 checkpoint
[09:21:59]   new checkpoint served as qwen-rl-a2-final:latest
[09:21:59] urn A2 eval of new checkpoint (4 vocabs)
[09:28:33]   urn new-ckpt OK
[09:28:33] R3 (script) A2 eval of new checkpoint, EFR4
[09:28:33]   R3 new-ckpt FAILED
[09:28:33] PHASE 4: tarball

## run dirs produced
runs/rl_urn_pilot_a2
runs/urn_qwen-rl-a2-final_latest_n-announced
runs/urn_qwen-rl-a2-final_latest_n-announced_vocab-cauldron
runs/urn_qwen-rl-a2-final_latest_n-announced_vocab-quiver
runs/urn_qwen-rl-a2-final_latest_n-announced_vocab-treasure_chest
runs/urn_qwen-rl-base-q8_latest_n-announced
runs/urn_qwen-rl-base-q8_latest_n-announced_vocab-cauldron
runs/urn_qwen-rl-base-q8_latest_n-announced_vocab-quiver
runs/urn_qwen-rl-base-q8_latest_n-announced_vocab-treasure_chest
runs/urn_qwen-rl-urn-final_latest_n-announced
runs/urn_qwen-rl-urn-final_latest_n-announced_vocab-cauldron
runs/urn_qwen-rl-urn-final_latest_n-announced_vocab-quiver
runs/urn_qwen-rl-urn-final_latest_n-announced_vocab-treasure_chest
runs/urn_tool_qwen-rl-base-q8_latest_n-announced
runs/urn_tool_qwen-rl-base-q8_latest_n-announced
runs/urn_tool_qwen-rl-base-q8_latest_n-announced_vocab-cauldron
runs/urn_tool_qwen-rl-base-q8_latest_n-announced_vocab-cauldron
runs/urn_tool_qwen-rl-base-q8_latest_n-announced_vocab-quiver
runs/urn_tool_qwen-rl-base-q8_latest_n-announced_vocab-quiver
runs/urn_tool_qwen-rl-base-q8_latest_n-announced_vocab-treasure_chest
runs/urn_tool_qwen-rl-base-q8_latest_n-announced_vocab-treasure_chest
runs/urn_tool_qwen-rl-urn-final_latest_n-announced
runs/urn_tool_qwen-rl-urn-final_latest_n-announced
runs/urn_tool_qwen-rl-urn-final_latest_n-announced_vocab-cauldron
runs/urn_tool_qwen-rl-urn-final_latest_n-announced_vocab-cauldron
runs/urn_tool_qwen-rl-urn-final_latest_n-announced_vocab-quiver
runs/urn_tool_qwen-rl-urn-final_latest_n-announced_vocab-quiver
runs/urn_tool_qwen-rl-urn-final_latest_n-announced_vocab-treasure_chest
runs/urn_tool_qwen-rl-urn-final_latest_n-announced_vocab-treasure_chest
[09:30:33] ALL DONE — pull ~/night_artifacts.tar.gz and ~/MORNING_REPORT.md
