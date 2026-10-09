<div align="center">

# ComfyUI-RBG-Sampler-Scheduler-Extras

[![GitHub stars](https://img.shields.io/github/stars/RamonGuthrie/ComfyUI-RBG-Sampler-Scheduler-Extras?style=flat-square&logo=github)](https://github.com/RamonGuthrie/ComfyUI-RBG-Sampler-Scheduler-Extras/stargazers)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg?style=flat-square)](https://www.gnu.org/licenses/gpl-3.0)
[![ComfyUI Compatible](https://img.shields.io/badge/ComfyUI-Compatible-green.svg?style=flat-square)](https://github.com/Comfy-Org/ComfyUI)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey?style=flat-square)]()

*Curated, high-precision samplers and schedulers consolidated for advanced diffusion workflows.*

</div>

---

A consolidated custom node pack for [ComfyUI](https://github.com/Comfy-Org/ComfyUI) bundling specialized samplers and schedulers from multiple community repositories into a single, clean extension.

This pack maintains **full backward compatibility** with workflows created using the original individual node packs.

---

## 📦 What's Included

### 1. Samplers

#### **Sampler DPM++ 2M Sharp**
* **Node ID:** `SamplerDPMPP_2M_Sharp`
* **Category:** `model/sampling/samplers`
* **Output:** `SAMPLER`
* **Overview:** An experimental variation of DPM++ 2M (originally distributed for Qwen Image 2.1) that progressively damps the 2nd-order latent history during sampling.
* **Technical Note:** While branded as "sharpening," mathematically it applies global scalar attenuation to previous-step latent predictions rather than spatial frequency filtering. This introduces a slight trajectory shift during ODE solving that perceptually enhances contrast and texture. Set `sharpness = 0.0` for standard DPM++ 2M behavior.

---

### 2. Schedulers

#### **FlowMatch Euler Discrete Scheduler**
* **Node ID:** `FlowMatchEulerDiscreteScheduler`
* **Category:** `sampling/custom_sampling/schedulers`
* **Output:** `SIGMAS` *(Also registers `FlowMatchEulerDiscreteScheduler` directly into standard `KSampler` dropdown menus)*
* **Overview:** A port of the Diffusers library's FlowMatch Euler Discrete scheduler (popularized by Z-Image Turbo, AI-Toolkit, and validation pipelines). Exposes granular controls over sigma scheduling.
* **Key Features:** Dynamic shifting, base/max image sequence length scaling, sigma inversion, stochastic sampling, and custom step ranges (`start_at_step` / `end_at_step`).

#### **Sigmoid Offset Scheduler**
* **Node ID:** `SigmoidOffsetScheduler`
* **Category:** `sampling/custom_sampling/schedulers`
* **Output:** `SIGMAS` *(Also integrates with dropdown selectors where supported)*
* **Overview:** Computes sigma noise curves using an adjustable S-curve (sigmoid) distribution instead of standard linear/exponential decays. Originally optimized for Chroma and Flux models to concentrate sampling density around high-impact denoising phases.
* **Key Parameters:**
  * `square_k`: Curve steepness.
  * `base_c`: Sigmoid midpoint offset.
  * `start_sigma`: Noise scale initialization.

---

## 🎯 Recommended Recipes & Presets

| Target Model / Architecture | Recommended Sampler | Recommended Scheduler | Suggested Settings & Notes |
| :--- | :--- | :--- | :--- |
| **Z-Image Turbo / Z-Image** | Standard `Euler` / `Euler A` | `FlowMatchEulerDiscreteScheduler` | 8–9 steps. Replicates official diffusers pipeline behavior. |
| **Flux.1 / Chroma** | `Euler` or `DPM++ 2M Sharp` | `SigmoidOffsetScheduler` | `square_k: 1.0`, `base_c: 0.5`. Densifies steps in structure-forming phases. |
| **Qwen Image / DiTs** | `SamplerDPMPP_2M_Sharp` | `Simple` or `Normal` | `sharpness: 0.15 - 0.20`. Counteracts multi-step smoothing. |
| **AI-Toolkit Validation** | Standard `Euler` | `FlowMatchEulerDiscreteScheduler` | Matches standalone Diffusers inference scripts 1:1. |
| **SDXL / SD 1.5** | `SamplerDPMPP_2M_Sharp` | `Karras` / `Exponential` | `sharpness: 0.08 - 0.12`. Subtle texture boost on eyes and fine fabrics. |

---

## 🚀 Installation

### Option 1: ComfyUI Manager (Easiest)
1. In ComfyUI, open **ComfyUI Manager** → **Install via Git URL**.
2. Paste: `https://github.com/RamonGuthrie/ComfyUI-RBG-Sampler-Scheduler-Extras.git`
3. Restart ComfyUI.

### Option 2: Manual Git Clone

1. Open a terminal in your ComfyUI installation folder:
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
> If you previously had any of the original standalone repositories installed (`ComfyUI-EulerDiscreteScheduler`, `ComfyUI-DPMpp-2M-Sharp`, or `ComfyUI_SigmoidOffsetScheduler`), remove or disable them from your `custom_nodes/` directory to avoid duplicate registration warnings.

---

## 🛠️ Usage

### Standard KSampler Workflows
* **FlowMatch Euler Discrete:** Select `FlowMatchEulerDiscreteScheduler` directly from the `scheduler` dropdown inside default `KSampler` and `KSampler (Advanced)` nodes.

### Custom Sampling Workflows
For modular setups using `SamplerCustom` or `SamplerCustomAdvanced`:
1. Connect **`Sampler DPM++ 2M Sharp`** to the `sampler` input port.
2. Connect either **`FlowMatch Euler Discrete Scheduler (Custom)`** or **`Sigmoid Offset Scheduler`** to the `sigmas` input port.

---

## 📋 Requirements

* `torch`
* `diffusers >= 0.30.0` (required for FlowMatch Euler Discrete)

---

## ❓ Frequently Asked Questions

<details>
<summary><b>Why don't I see FlowMatchEulerDiscreteScheduler in the KSampler dropdown?</b></summary>
Ensure <code>diffusers&gt;=0.30.0</code> is installed in the Python environment powering ComfyUI. Check your startup console log for any import errors and restart ComfyUI.
</details>

<details>
<summary><b>Can I use DPM++ 2M Sharp with standard SD 1.5 or SDXL?</b></summary>
Yes! It functions with all models supported by standard DPM++ 2M. Use a modest sharpness value (between <code>0.05</code> and <code>0.12</code>) to subtly enhance texture without oversharpening.
</details>

<details>
<summary><b>Will this break existing workflows saved with the original nodes?</b></summary>
No. The internal node identifier strings (<code>SamplerDPMPP_2M_Sharp</code>, <code>FlowMatchEulerDiscreteScheduler</code>, and <code>SigmoidOffsetScheduler</code>) are identical to the upstream versions, ensuring complete backward compatibility.
</details>

---

## 🙏 Credits & Acknowledgments

This pack consolidates work from the following upstream projects and creators:
* **[ComfyUI-EulerDiscreteScheduler](https://github.com/erosDiffusion/ComfyUI-EulerDiscreteScheduler)** by [erosDiffusion](https://github.com/erosDiffusion)
* **[ComfyUI-DPMpp-2M-Sharp](https://github.com/envy-ai/ComfyUI-DPMpp-2M-Sharp)** by [envy-ai](https://github.com/envy-ai)
* **[ComfyUI_SigmoidOffsetScheduler](https://github.com/silveroxides/ComfyUI_SigmoidOffsetScheduler)** by [silveroxides](https://github.com/silveroxides)

---

## 📄 License

This repository is licensed under the [GNU General Public License v3.0 (GPL-3.0)](LICENSE), incorporating upstream components licensed under GPL-3.0 and MIT. See [LICENSE](LICENSE) for full legal text and third-party notices.
```