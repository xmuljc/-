# Project Summary

## What This Project Is

This workspace is a Tencent KaiwuRL competition project for `gorge_chase`.

Key metadata from `kaiwu.json`:

- `project_code = "gorge_chase"`
- `version = "15.0.1-comp-normal-lite.26comp"`

The task is a survival and path-planning RL problem:

- The agent controls Luban No.7
- Survive within a step limit
- Avoid 2 monsters
- Collect as many treasures as possible
- The map also contains speed buffs and a flash skill

## Most Important Conclusion

- The main implementation to study is `agent_ppo`
- `agent_diy` is only a template and is not the main runnable baseline
- If the user says "use PPO, not DIY", always focus on `agent_ppo`
- This workspace contains task-side business code only
- Framework packages such as `kaiwudrl`, `common_python`, and `tools` live in the user's Docker environment, where training actually runs

## Read This First

Start with:

- the official root-level `.docx` file in this workspace

That document explains:

- Task rules
- Environment config
- Observation and action space
- KaiwuDRL training flow
- Code package structure
- PPO baseline design

If a new agent knows nothing about this project, reading the doc first is faster than starting from code.

## Root-Level Files

- `train_test.py`
  - Local entry point for train/test
  - Switches between `ppo` and `diy`
- `conf/app_conf_gorge_chase.toml`
  - App/policy-level config
  - Currently uses `algo = "ppo"`
- `conf/algo_conf_gorge_chase.toml`
  - Maps `ppo` and `diy` to concrete agent/workflow implementations
- `conf/configure_app.toml`
  - Replay buffer, batch size, model save, sync, preload, and related framework config
- the official root-level `.docx` file
  - Official overview document

Main code directories:

- `agent_ppo/`
  - Complete PPO baseline implementation
- `agent_diy/`
  - User-custom algorithm template with many `pass` placeholders

## Main Execution Chain

The practical chain is:

1. `train_test.py`
2. `conf/app_conf_gorge_chase.toml`
3. `conf/algo_conf_gorge_chase.toml`
4. `agent_ppo/workflow/train_workflow.py`
5. `agent_ppo/agent.py`
6. `agent_ppo/feature/preprocessor.py`
7. `agent_ppo/feature/definition.py`
8. `agent_ppo/model/model.py`
9. `agent_ppo/algorithm/algorithm.py`

In plain words:

- `train_test.py` selects the algorithm name
- App/algo config maps that name to the PPO agent and PPO workflow
- `workflow` runs episodes and collects samples
- `agent.py` wires together preprocessor, model, optimizer, and PPO algorithm
- `preprocessor.py` builds features and step reward
- `definition.py` defines data structures and computes GAE
- `model.py` defines the Actor-Critic network
- `algorithm.py` computes PPO losses and updates parameters

## PPO Baseline Details

### Agent Layer

`agent_ppo/agent.py` does the following:

- Builds `Model`
- Builds `Adam`
- Builds `Algorithm`
- Builds `Preprocessor`
- Uses `predict()` during training
- Uses `exploit()` during evaluation
- Uses `learn()` on the learner side

Training vs evaluation:

- `predict()` samples actions stochastically for exploration
- `exploit()` uses greedy action selection

### Feature and Reward Layer

`agent_ppo/feature/preprocessor.py` is one of the most important files.

The baseline uses a 40D feature vector:

- 4D hero features
- 5D monster 1 features
- 5D monster 2 features
- 16D local map features
- 8D legal-action mask
- 2D progress features

Reward is intentionally simple:

- A small survive reward each step
- Distance-based shaping relative to the nearest monster
- Terminal reward is added in `train_workflow.py`
  - survive to truncation / win: `+10`
  - caught / terminated: `-10`

Important distinction from the doc:

- Environment score is not the same thing as RL reward
- Score is used for evaluation
- Reward is what PPO trains on

### Sample and GAE Layer

`agent_ppo/feature/definition.py` defines:

- `ObsData`
- `ActData`
- `SampleData`

It also:

- fills `next_value`
- computes `advantage`
- computes `reward_sum`

This is where GAE is applied.

### Model Layer

`agent_ppo/model/model.py` is a minimal Actor-Critic network:

- Input: 40D
- Shared backbone: `40 -> 128 -> 64`
- Actor head: `64 -> 8`
- Critic head: `64 -> 1`

This is a lightweight baseline, not a large or complex model.

### Algorithm Layer

`agent_ppo/algorithm/algorithm.py` implements standard PPO-style training:

- masked softmax with `legal_action`
- clipped policy objective
- clipped value loss
- entropy regularization

Loss form:

- `total_loss = vf_coef * value_loss + policy_loss - beta * entropy_loss`

## Training Workflow

`agent_ppo/workflow/train_workflow.py` can be summarized as:

1. Read `agent_ppo/conf/train_env_conf.toml`
2. Call `env.reset(usr_conf)`
3. Call `agent.reset(...)`
4. Call `agent.load_model(id="latest")`
5. Loop:
   - `agent.predict(...)`
   - `env.step(act)`
   - build `SampleData`
6. At episode end:
   - add terminal reward
   - run `sample_process(...)`
   - call `agent.send_sample_data(g_data)`
7. Save model periodically

This file is the best entry point for understanding:

- environment interaction
- sample construction
- handoff to learner-side training

## Why `agent_diy` Is Not the Main Path

`agent_diy` is mostly a template, not the real baseline.

Examples:

- `agent_diy/agent.py` has many `pass`
- `agent_diy/algorithm/algorithm.py` is `pass`
- `agent_diy/feature/definition.py` still leaves reward/sample processing empty
- `agent_diy/workflow/train_workflow.py` is mostly a scaffold

Therefore, if the goal is:

- understand the current project
- understand the provided baseline
- use PPO instead of writing a custom algorithm from scratch

then the main reading path must be `agent_ppo`, not `agent_diy`.

## Important Task-Side Facts From the Official Doc

- The environment's raw action space is 16D:
  - 8 move actions
  - 8 flash actions
- The current PPO baseline only uses 8 actions
  - move directions only
- The observation is local-view based
- The official task score is separate from RL reward
- The provided baseline is intentionally minimal and is meant to be extended

The "baseline only uses 8 actions" point is important:

- the environment supports more behavior than the current baseline exposes

## Config Files

### Framework Config

`conf/configure_app.toml` contains framework-level settings such as:

- `replay_buffer_capacity`
- `preload_ratio`
- `reverb_sampler`
- `train_batch_size`
- `dump_model_freq`
- `model_file_sync_per_minutes`
- pretrained model loading settings

Note:

- `train_test.py` overrides part of this config through `env_vars`
- those overrides look like quick local-test settings, not serious long-run training settings

### PPO Hyperparameters

`agent_ppo/conf/conf.py` contains:

- `GAMMA`
- `LAMDA`
- `INIT_LEARNING_RATE_START`
- `BETA_START`
- `CLIP_PARAM`
- `VF_COEF`
- `GRAD_CLIP_RANGE`

### Environment Config

`agent_ppo/conf/train_env_conf.toml` controls task-side training settings such as:

- training maps
- whether map selection is random
- treasure count
- buff count and cooldown
- talent cooldown
- second monster appearance timing
- monster speedup timing
- max step count

## Docker Boundary

The user explicitly stated that the following packages live in Docker:

- `kaiwudrl`
- `common_python`
- `tools`

That means there are two layers of understanding:

### Directly readable inside this workspace

- PPO business logic
- task-side feature design
- reward design
- model design
- workflow logic
- config files

### Only readable by entering the Docker environment

- internal implementation of `run_train_test`
- internal behavior of `BaseAgent`
- exact sample-pool handoff behind `send_sample_data`
- actor / learner / aisrv scheduling details
- `tools.*` framework-side evaluation or utility details

If a future agent needs full end-to-end framework understanding, it should inspect the Docker environment next.

## Recommended Reading Order

For a new agent, read in this exact order:

1. the official root-level `.docx` file
2. `train_test.py`
3. `conf/app_conf_gorge_chase.toml`
4. `conf/algo_conf_gorge_chase.toml`
5. `agent_ppo/workflow/train_workflow.py`
6. `agent_ppo/agent.py`
7. `agent_ppo/feature/preprocessor.py`
8. `agent_ppo/feature/definition.py`
9. `agent_ppo/model/model.py`
10. `agent_ppo/algorithm/algorithm.py`
11. `agent_ppo/conf/conf.py`
12. `agent_ppo/conf/train_env_conf.toml`
13. `conf/configure_app.toml`

Only inspect `agent_diy` later if the goal is to study extension points or implement a custom algorithm.

## One-Sentence Handoff

This is a Tencent KaiwuRL `gorge_chase` project whose real baseline is `agent_ppo`, not `agent_diy`; the main chain is `train_test.py -> config mapping -> PPO workflow -> agent -> preprocessor -> sample_process(GAE) -> model -> PPO algorithm`, while framework internals live in the user's Docker environment.
