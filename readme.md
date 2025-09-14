# Super-Resolution with Flow Matching (x4)

> I read a blog and paper on Flowmatching so I decided to have a try at it...  I learnt A LOT about batch size, Normalisation(Group,layer,Batch), nondeterminism and their importance in training..
>https://arxiv.org/abs/2210.02747 and https://github.com/facebookresearch/flow_matching were helpful.. the examples in the repository are very helpful to get the intuition. My results are in the runs/sr_x4/samples. check section 6 below for how the grids are arranged.

---

## 1) Flow Matching (intuitive)
So in flowmatching, we have 2 distribution, noise and the target distribution . Then we define a simple path from the noise to the HR image and train the network to predict the velocity (velocity is a vector) that keeps a sample moving along that path. The neural network becomes a vector field in which we can give it a sample at a given time and were gonna get the velocity. At inference we start from noise and integrate the learned velocity field for a few steps to reach the SR output.

We purposely use the straight‑line path so the target velocity is constant along the path. Good for stability and its just easier to understand.

---

## 2) Training
- Starting with only 1k optimizer steps we saw very low metrics: PSNR ≈ 6–7 dB, SSIM ≈ 0.04–0.05.
- After 10k steps on DIV2K with our small GPU, we reached PSNR = 27.21 dB and SSIM = 0.9069.
---

## 3) Why batch size & micro‑batches matter (and how to count patches)
Let:
- `data.batch_size` = micro‑batch images per GPU step (limited by VRAM).
- `train.grad_accum` = number of micro‑batches we accumulate before an optimizer update.
- Effective batch per optimizer step = `batch_size × grad_accum`.
- Patches seen over `S` optimizer steps = `S × batch_size × grad_accum`.

**My run:** `batch_size=4`, `grad_accum=4` = effective batch = 16.  
- At 1,000 steps: `1,000 × 16 = 16,000` HR patches seen.  Might be too little to learn diverse textures and edges.
- At 10,000 steps: `10,000 × 16 = 160,000` patches about 200 patches per image.

**notes about micro‑batches:**
- For simple losses like MSE and models without stateful normalization, gradient accumulation is mathematically equivalent to using the same larger batch size (same effective batch and LR schedule), just split across micro-batches to fit VRAM (Evidence: https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html?utm_source=).
---

## 4) Model (tiny UNet, no norm)
**`model.py`**
- UNetSR with three scales, base channels configurable.
- No normalization in the residual blocks (just Conv → SiLU → Conv with residual scaling).
- Conditioning: upsample LR to HR size and concatenate with the current state (6 input channels total).

**Why we removed normalization for SR**
- In patch‑based SR, normalization layers (Batch/Group) can wash out local contrast and color statistics. (I saw this twitter thread https://x.com/xunhuang1995/status/1911931965336408357 and https://x.com/rgilman33/status/1911712029443862938).
---

## 5) Files and what they do
- **`flowmatch.py`** : linear path targets and the Forward‑Euler sampler.
- **`model.py`** :no‑norm UNet for SR.
- **`train.py`** : training loop (AMP, EMA, gradient accumulation, PSNR/SSIM eval).
- **`metrics.py`** : lightweight PSNR/SSIM on RGB tensors in [0,1].

---

## 6) How to run
```bash
python train.py --config config.yaml
```
Key knobs in `config.yaml`:
- `data.hr_crop` (default 192), `data.batch_size`, `data.grad_accum` (controls effective batch), `data.num_workers`
- `train.max_steps`, `train.lr`, `train.ema_decay`, `train.amp`, `train.grad_clip`, `train.grad_checkpoint`
- `eval.every_steps`
- `fm.sampler_default_steps` (validation/inference ODE steps)

to run sampler:
```bash
python sample.py --config config.yaml --ckpt runs/sr_x4/checkpoints/step_10000.pt
```
Press ctrl+C to stop it running, cuz it runs indefinitely.Sampling grids (bicubic | ours | HR) land under `runs/sr_x4/samples/`.

filename convention:

x4 = the upscaling factor (scale=4).
We downsample HR to LR by 4 and then super-resolve back ×4.

s8 = the number of sampler steps you used (here, 8 Euler steps).
You can change it at run time with --steps N.

---

## 7)checkpoints
  Checkpoints are in `runs/sr_x4/checkpoints/step_10000.pt` etc.

---

## 8) Tips if you want more
- Train longer (e.g., 20k–50k steps) or increase effective batch.
- If you need quality per step, swap Euler for Heun (RK2) or RK4.

---

## 9) Environment
Single **RTX 2050 (4 GB)**, HR crop 192, **effective batch 16** (`batch_size=4`, `grad_accum=4`). Took about 188.3 minutes to train.  
Mixed precision via `torch.amp`; simple local logging and periodic checkpoints.
