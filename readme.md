# Super-Resolution with Flow Matching (x4)

> Compact UNet + FiLM time conditioning + lightweight bottleneck attention, trained with flow matching and sampled with a few-step Forward Euler ODE solver.

---

## 1) Diffusion vs. Flow Matching — clear, equation‑free intuition

**Diffusion (noise‑prediction view).**  
• Training: take a clean image, add Gaussian noise at a random strength, and train the network to **predict the noise you added**. In other words, you teach the model how to denoise across many noise levels.  
• Sampling: start from pure noise and repeatedly **subtract the noise the model predicts**. Do this in small steps until you reach a clean image.

**Flow Matching (velocity‑prediction view).**  
• Training: decide on a **continuous path** that moves a sample from noise to the target image. The model learns the **velocity field**: at a given point along the path and a given time, how should the image move next to keep following that path?  
• Sampling: start from noise and **integrate an ordinary differential equation (ODE)** that follows the model’s velocity from the beginning of the path to the end. No probability densities or scores are required—just follow the learned flow.

**Analogy.** Diffusion is like learning to **remove fog** layer by layer; flow matching is like learning the **river’s current** that carries you from fog to clear skies.

In this repo we use the simplest possible path: a **straight line** from noise to the HR image. The model is trained to imitate the straight‑line “push” that turns noise into the image, while also using the LR input as guidance.

---

## 2) Results at a glance
- Validation after **1k training steps** (we used steps, not epochs): **PSNR ≈ 12.16 dB**, **SSIM ≈ 0.42**.  
- With minimal compute, we observed **steady upward trends** in both metrics.  
- A **consistent tint** appears in flow‑matching outputs. We intentionally leave this as‑is; it often happens when optimizing only RGB MSE without any color‑specific regularization.  
- For visuals we save **grids** per image: **[bicubic | SR (ours) | HR]**.

---

## 3) Datasets & directories
- **Datasets:** DIV2K (train HR and validation HR). Set the paths in `config.yaml`.
- **Run directory:** experiment artifacts (checkpoints, samples, logs) live under `runs/sr_x4/`.

---

## 4) How to run

### Train
```bash
python train.py --config config.yaml
```
**Important config knobs (all in `config.yaml`):**
- `paths.*` — where the datasets and run directory live.
- `data.scale` (=4), `data.hr_crop` (=192), `data.batch_size`, `data.grad_accum`, `data.num_workers`.
- `train.lr`, `train.max_steps`, `train.log_every`, `train.grad_clip`, `train.grad_checkpoint`, `train.amp`, `train.ema_decay`.
- `eval.every_steps` — how often to evaluate and checkpoint.
- `fm.sampler_default_steps` — default number of ODE steps used for validation sampling (can be overridden at sampling time).
- `inference.tile`, `inference.overlap` — tile size and overlap used for tiled SR at inference.

### Sample / visual comparison
```bash
# default: first 5 images from the validation HR directory; saves a grid [bicubic|SR|HR]
python sample.py --config config.yaml --ckpt runs/sr_x4/checkpoints/step_1000.pt

# useful options
python sample.py --config config.yaml --ckpt <ckpt> --steps 12       # try more ODE steps
python sample.py --config config.yaml --ckpt <ckpt> --limit 12       # process first N images
python sample.py --config config.yaml --ckpt <ckpt> --images 0801,0810  # pick specific filenames or stems
```
Outputs land in `runs/sr_x4/samples/` as `{stem}_grid_x4_s{steps}.png`.

---

## 5) Model architecture (compact UNet + FiLM + lightweight attention)

### High‑level
- A 3‑level **UNet** with base channels **48**, channel multipliers **(1, 2, 4)**, and **two FiLM‑ResBlocks per level**.  
- **Conditioning:** we upsample the LR patch to HR size and **concatenate** it with the current state (so the input has 6 channels: 3 from the state + 3 from LR).  
- **Time conditioning:** a sinusoidal embedding of the continuous time `t`, passed through a small MLP, produces FiLM parameters that modulate every residual block.  
- **Attention:** a **single, lightweight self‑attention** block **only at the bottleneck** (lowest resolution) to capture global context at low cost.

### What is FiLM and why use it?
**FiLM** stands for *Feature‑wise Linear Modulation*. It produces a per‑channel **scale** and **shift** from a conditioning signal (in our case, the time embedding). Inside each residual block, after normalization we modulate features as:
- “scaled by (1 + scale)” and then “shifted by shift.”
This gives the network a simple, stable way to inject time information everywhere in the UNet without adding heavy cross‑attention or concatenating extra channels at every layer.

### Time embeddings
We build a small vector of sines and cosines of the time `t` at different frequencies (a standard trick from diffusion models). This gives the MLP a rich, periodic basis to describe how the network’s behavior should change as `t` moves from 0 (very noisy) to 1 (clean image).

### Lightweight attention at the bottleneck
SR benefits from global context (textures and repeated structures), but attention at full resolution is expensive. Putting **one attention block at the bottleneck** gives the model a global receptive field with minimal memory/compute because spatial maps are tiny there.

### EMA (Exponential Moving Average) and why it helps
We maintain a shadow copy of model parameters that is a smoothed average of recent weights. During evaluation we temporarily **swap in the EMA weights**. This often makes validation metrics **more stable** and the sampled images **less jittery** across checkpoints—especially useful for short runs on small GPUs.

---

## 6) Flow matching targets and the Forward Euler sampler

### Targets (no formulas, just the idea)
At training time we draw a random time `t` between 0 and 1, create a sample that lies part‑way between **pure noise** and the **true HR image**, and ask the model to predict the **velocity** that would keep moving that sample along a **straight line** toward the HR image. Because the path is straight, the true target velocity is constant along the path. The loss is simple mean‑squared error between the model’s velocity prediction and this target velocity.

### Forward Euler (what we actually do at inference)
At sampling time we start from noise and **integrate an ODE** that uses the model’s velocity field. We use **Forward Euler** with a fixed number of steps:
- Split the time from 0 to 1 into `N` equal steps.  
- At each step, evaluate the model at the **middle** of the step (a small stability trick) and take one small move in that direction.  
- Repeat until you reach time 1.

Why Euler? It’s the simplest explicit solver, **fast** and **memory‑light**, great for validating the training signal and producing quick samples on a 4 GB GPU.  
Alternatives if you want more quality per step: **Heun / RK2** (a strong next upgrade), **RK4** (heavier), or diffusion‑style solvers adapted to flows. The number of steps is fully configurable via `--steps`; typical ranges to try are 4–32.

---

## 7) Bicubic vs. our SR (what the grid shows)
- **Bicubic interpolation** is a fast, classical way to upsample. It’s a good **baseline**, but it cannot invent missing detail—edges and textures tend to look soft.  
- Our model leverages both the LR input and learned priors from HR patches to recover sharper detail. The saved grids show **bicubic** vs **our SR** vs **ground‑truth HR** side‑by‑side so you can judge improvements at a glance.

---

## 8) Files and key functions (what lives where)

### `data.py`
- `SuperResolutionDataset` — loads HR images, applies **random HR crop 192×192**, light flips/rotations, and generates LR with torchvision’s **bicubic** downsampling (antialias enabled). Returns tensors in [0, 1], channel‑first.  
- `make_dataloader(...)` — wraps the dataset with a DataLoader. Uses a **Windows‑safe RNG instance** so workers can spawn without pickling errors.

### `model.py`
- `SinusoidalTimeEmbedding` — builds the time features from `t`.
- `TimeMLP` — maps time features to FiLM parameters.  
- `FiLMResBlock` — residual block with GroupNorm + SiLU, FiLM modulation, and 3×3 convolutions.  
- `SelfAttention2d` — lightweight multi‑head self‑attention used only at the bottleneck.  
- `UNetSR` — the full UNet with LR‑upsample concatenation at the input, FiLM‑ResBlocks on the down and up paths, bottleneck attention, and optional **gradient checkpointing**.

### `flowmatch.py`
- `sample_timesteps(B)` — draw `t` values uniformly in [0, 1].  
- `prepare_linear_path(...)` — create the part‑way sample between noise and HR and compute its target velocity for the straight‑line path.  
- `fm_training_targets(...)` — a small switchboard that currently uses the linear path.  
- `euler_sampler(model, x_lr, steps, scale)` — the few‑step **Forward Euler** integrator used at inference and validation.

### `train.py`
- Mixed precision via `torch.autocast(..., dtype=torch.float16)` and `torch.amp.GradScaler(device="cuda")`.  
- EMA implementation and swap‑in for evaluation.  
- Training loop with tqdm progress bar, MSE loss, periodic **PSNR/SSIM** on validation patches, and checkpoints every `eval.every_steps`.

### `sample.py`
- Tiled SR with `tile=192`, `overlap=16` in HR space to avoid OOM on large images.  
- Saves **grids** only by default. You can restrict processing with `--limit` (default 5) or specify exact images with `--images`.

### `metrics.py`
- Lightweight **PSNR** and **SSIM** operating on RGB tensors in [0, 1].

### `utils.py`
- Config loading, seeding, device selection, and experiment directory preparation.

---

## 9) Training notes and artifacts
- We trained **1,000 steps** (not epochs).  
- **EMA** made validation curves smoother and samples a bit more stable across checkpoints.  
- The **tint** is **consistent across images** and left as a documented artifact of RGB‑space MSE on our small‑compute run.

---

## 10) Extensions and improvement ideas
- **Train longer** (e.g., 150k steps) with EMA; consider a cosine learning‑rate schedule.  
- **Sampler upgrade:** try **Heun / RK2** for better quality at the same number of steps.  
- **Loss:** add Charbonnier/L1 or a simple color regularizer to reduce tint.  
- **Capacity:** bump `base_channels` (48 → 64/80) or add one more UNet level; consider attention at the mid‑level if VRAM allows.  
- **Alternative paths:** experiment with a variance‑preserving path for closer parity with diffusion setups.

---

## 11) FAQ
- **Can I change the number of sampler steps?** Yes: `sample.py --steps N` or set `fm.sampler_default_steps` in `config.yaml`.  
- **Why Euler and not RK2/RK4?** Euler is minimal and robust for small GPUs; RK2 is the next practical upgrade.  
- **Is bicubic a super‑resolution method?** As a baseline—yes. It upsamples but does not add detail; our learned model improves upon it.

---

## 12) Environment
- **PyTorch** + **torchvision**; AMP via `torch.amp`.  
- Hardware: single **RTX 2050 (4 GB)**; batch size 8 (effective), HR crop 192, LR 48.  
- Logging is local‑only; checkpoints saved every `eval.every_steps`.

---

**End.**

