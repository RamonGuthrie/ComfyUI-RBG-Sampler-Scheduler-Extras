
<div align="center">

# ComfyUI-RBG-Sampler-Scheduler-Extras

[![GitHub stars](https://img.shields.io/github/stars/RamonGuthrie/ComfyUI-RBG-Sampler-Scheduler-Extras?style=flat-square&logo=github)](https://github.com/RamonGuthrie/ComfyUI-RBG-Sampler-Scheduler-Extras/stargazers)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg?style=flat-square)](https://www.gnu.org/licenses/gpl-3.0)
[![ComfyUI Compatible](https://img.shields.io/badge/ComfyUI-Compatible-green.svg?style=flat-square)](https://github.com/Comfy-Org/ComfyUI)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey?style=flat-square)]()

*Curated suite of specialized samplers, schedulers, and error-correction solvers for ComfyUI.*

</div>

---

A consolidated custom node pack for [ComfyUI](https://github.com/Comfy-Org/ComfyUI) bundling specialized samplers and schedulers from multiple community repositories into a single, clean extension.

This pack maintains **full backward compatibility** with workflows created using the original individual node packs.

---

## 📦 What's Included

### 1. Samplers

* **Sampler DPM++ 2M Sharp** (`SamplerDPMPP_2M_Sharp`):  
  An experimental variation of DPM++ 2M (originally distributed for Qwen Image 2.1) that progressively damps the 2nd-order latent history during sampling. Perceptually boosts texture and micro-contrast. Set `sharpness = 0.0` for stock DPM++ 2M behavior.
  * **Technical Note:** While branded as "sharpening," mathematically it applies global scalar attenuation to previous-step latent predictions rather than spatial frequency filtering.
* **Restart Sampler Suite** (`RestartSampler`, `KSampler With Restarts` [Simple / Advanced / Custom]):  
  Implements the research paper *"Restart Sampling for Improving Generative Processes"*. Periodically re-injects calibrated noise into the latent trajectory to climb back up sigmas and re-denoise, eliminating anatomical defects and artifacts.
* **Sampler TCD** (`SamplerTCD`):  
  Dedicated solver for Trajectory Consistency Distillation (TCD) models with an adjustable `gamma` parameter to modulate stochasticity.
* **Adaptive ODE Solver** (`ODESolver` / `ODE Solver`):  
  Integrates `torchdiffeq` solvers (`bosh3`, `fehlberg2`, `dopri5`, `adaptive_heun`). Dynamically calculates local truncation error and adjusts step sizes automatically to user-defined tolerances (`log_rtol`, `log_atol`).

---

### 2. Schedulers

* **FlowMatch Euler Discrete Scheduler** (`FlowMatchEulerDiscreteScheduler`):  
  Port of Diffusers' FlowMatch Euler Discrete scheduler (popularized by Z-Image Turbo, AI-Toolkit, and validation pipelines). Exposes granular controls over dynamic shifting, base/max sequence length scaling, and custom step ranges (`start_at_step` / `end_at_step`). Also registers directly into the default `KSampler` scheduler dropdown.
* **Sigmoid Offset Scheduler** (`SigmoidOffsetScheduler`):  
  Computes sigma noise curves using an adjustable S-curve (sigmoid) distribution instead of standard linear/exponential decays. Optimized for Chroma and Flux models to concentrate sampling density where structural details solidify.
* **Restart Scheduler** (`RestartScheduler`):  
  Outputs tailored sigma curves with embedded restart segments inserted at specific timesteps/percentages for use with `RestartSampler`.
* **TCD Scheduler** (`TCDScheduler`):  
  Computes the optimal sigma trajectory for few-step distilled models (TCD LoRAs, Hyper-SD).

---

## 🎯 Recommended Recipes & Presets

| Target Model / Goal | Recommended Sampler | Recommended Scheduler | Suggested Settings & Notes |
| :--- | :--- | :--- | :--- |
| **Artifact / Anatomy Fixes** | `RestartSampler` | `RestartScheduler` | 20–30 steps. Set restart segment around `t_min: 0.1`, `t_max: 1.0` to clean up distorted details without changing composition. |
| **Z-Image Turbo / Z-Image** | Standard `Euler` / `Euler A` | `FlowMatchEulerDiscreteScheduler` | 8–9 steps. Replicates official diffusers pipeline behavior. |
| **Flux.1 / Chroma** | `Euler` or `DPM++ 2M Sharp` | `SigmoidOffsetScheduler` | `square_k: 1.0`, `base_c: 0.5`. Concentrates steps in structure-forming phases. |
| **Ultra-Fast Speed (4–8 Steps)** | `SamplerTCD` | `TCDScheduler` | For TCD LoRAs and Hyper-SD. Use `gamma: 0.25 - 0.50` to retain fine texture. |
| **Flow Models / SD3 (Adaptive)** | `ODE Solver` (`fehlberg2`) | `sgm_uniform` | `rtol: -3`, `atol: -4`. Dynamically steps through velocity fields until error tolerance is met. |
| **Qwen Image / DiTs** | `SamplerDPMPP_2M_Sharp` | `Simple` or `Normal` | `sharpness: 0.15 - 0.20`. Counteracts multi-step smoothing. |

---

## 🚀 Installation

### Option 1: ComfyUI Manager (Easiest)
1. In ComfyUI, open **ComfyUI Manager** → **Install via Git URL**.
2. Paste: `https://github.com/RamonGuthrie/ComfyUI-RBG-Sampler-Scheduler-Extras.git`
3. Restart ComfyUI.

### Option 2: Manual Git Clone

1. Open a terminal in your ComfyUI installation directory:
   ```bash
   cd ComfyUI/custom_nodes/
   ```
2. Clone this repository:
   ```bash
   git clone https://github.com/RamonGuthrie/ComfyUI-RBG-Sampler-Scheduler-Extras.git
   ```
3. Install dependencies:
   * **Portable ComfyUI (Windows):**
     ```cmd
     ..\..\python_embeded\python.exe -m pip install -r ComfyUI-RBG-Sampler-Scheduler-Extras/requirements.txt
     ```
   * **Standard Python / Virtual Environment:**
     ```bash
     pip install -r ComfyUI-RBG-Sampler-Scheduler-Extras/requirements.txt
     ```
4. Restart ComfyUI.

> [!IMPORTANT]
> If you previously installed any of the standalone repositories (`ComfyUI-EulerDiscreteScheduler`, `ComfyUI-DPMpp-2M-Sharp`, `ComfyUI_SigmoidOffsetScheduler`, `ComfyUI_restart_sampling`, `comfyui-tcd-scheduler`, or `ComfyUI-ODE`), remove or disable them from your `custom_nodes/` directory to avoid duplicate registration warnings.

---

## 🛠️ Usage

### Standard KSampler Workflows
* **FlowMatch Euler Discrete:** Available directly under the `scheduler` dropdown selector in standard `KSampler` and `KSampler (Advanced)` nodes.
* **KSampler With Restarts:** Drop the `KSampler With Restarts` node on your canvas as a direct drop-in replacement for default KSampler.

### Modular Custom Sampling Workflows
For modular setups using `SamplerCustom` or `SamplerCustomAdvanced`:
1. Connect any included sampler (`Sampler DPM++ 2M Sharp`, `RestartSampler`, `SamplerTCD`, or `ODE Solver`) to the `sampler` input port.
2. Connect the matching scheduler node (`FlowMatch Euler Discrete Scheduler`, `Sigmoid Offset Scheduler`, `RestartScheduler`, or `TCDScheduler`) to the `sigmas` input port.

---

## 📋 Requirements

* `torch`
* `diffusers >= 0.30.0` (required for FlowMatch Euler Discrete)
* `torchdiffeq` (required for Adaptive ODE Solvers)

---

## ❓ Frequently Asked Questions

<details>
<summary><b>Why is the ODE Solver failing to import?</b></summary>
The ODE solver depends on <code>torchdiffeq</code>. Run <code>pip install torchdiffeq</code> in your ComfyUI Python environment and restart ComfyUI.
</details>

<details>
<summary><b>Why don't I see FlowMatchEulerDiscreteScheduler in the KSampler dropdown?</b></summary>
<ol>
  <li>Ensure <code>diffusers>=0.30.0</code> is installed in the Python environment powering ComfyUI and check console logs for import errors.</li>
  <li>If you have node packs that override ComfyUI core schedulers (such as <code>res4lyf</code>), the dropdown registration may be overwritten. In this case, use the modular <b>FlowMatch Euler Discrete Scheduler (Custom)</b> node and plug its <code>SIGMAS</code> into a <b>SamplerCustom</b> node.</li>
</ol>
</details>

<details>
<summary><b>Will this break existing workflows saved with the original nodes?</b></summary>
No. All internal node class identifiers are identical to upstream releases, ensuring complete backward compatibility with your saved workflows.
</details>

---

## 🙏 Credits & Acknowledgments

This pack consolidates work from the following upstream projects and creators:
* **[ComfyUI-EulerDiscreteScheduler](https://github.com/erosDiffusion/ComfyUI-EulerDiscreteScheduler)** by [erosDiffusion](https://github.com/erosDiffusion)
* **[ComfyUI-DPMpp-2M-Sharp](https://github.com/envy-ai/ComfyUI-DPMpp-2M-Sharp)** by [envy-ai](https://github.com/envy-ai)
* **[ComfyUI_SigmoidOffsetScheduler](https://github.com/silveroxides/ComfyUI_SigmoidOffsetScheduler)** by [silveroxides](https://github.com/silveroxides)
* **[ComfyUI_restart_sampling](https://github.com/ssitu/ComfyUI_restart_sampling)** by [ssitu](https://github.com/ssitu)
* **[comfyui-tcd-scheduler](https://github.com/dfl/comfyui-tcd-scheduler)** by [dfl](https://github.com/dfl)
* **[ComfyUI-ODE](https://github.com/redhottensors/ComfyUI-ODE)** by [redhottensors](https://github.com/redhottensors)

---

## 📄 License

This repository is licensed under the [GNU General Public License v3.0 (GPL-3.0)](LICENSE), incorporating upstream components licensed under GPL-3.0, MIT, and Apache 2.0. See [LICENSE](LICENSE) for full legal text and third-party notices.