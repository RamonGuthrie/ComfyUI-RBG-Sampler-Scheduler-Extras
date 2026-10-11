"""
ComfyUI-RBG-Sampler-Scheduler-Extras 📦 v1.1.2
Consolidated high-precision samplers, schedulers, and error-correction suite.

Upstream Credits & Attribution:
- FlowMatch Euler Discrete Scheduler: erosDiffusion & Hugging Face Diffusers (Apache 2.0)
- Sampler DPM++ 2M Sharp: envy-ai (MIT)
- Sigmoid Offset Scheduler: silveroxides (MIT)
- TCD Scheduler & Sampler: dfl & Trajectory Consistency Distillation Authors (Apache 2.0)
- Restart Sampling Suite: ssitu (GPL 3.0)
- Adaptive ODE Solvers: redhottensors & torchdiffeq authors (MIT)
- Flow-Solver 2M & Momentum Euler: Based on Flow-Solver / TADA research (GPL 3.0 compatible)

License: GNU General Public License v3.0 (GPL-3.0)
"""

import ast
import inspect
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
from tqdm.auto import trange

import comfy.k_diffusion.sampling
import comfy.model_management
import comfy.sample
import comfy.samplers

try:
    import latent_preview
    HAS_LATENT_PREVIEW = True
except ImportError:
    HAS_LATENT_PREVIEW = False

logger = logging.getLogger("ComfyUI-RBG-Sampler-Scheduler-Extras")

# ==============================================================================
# 1. FLOWMATCH EULER DISCRETE SCHEDULER
# ==============================================================================

try:
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler as DiffusersFlowMatchEulerDiscreteScheduler
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False


def calculate_flowmatch_sigmas(
    steps: int = 20,
    shift: float = 3.0,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    base_image_seq_len: int = 256,
    max_image_seq_len: int = 4096,
    num_train_timesteps: int = 1000,
    shift_terminal: Optional[float] = 0.1,
    time_shift_type: str = "exponential",
    use_beta_sigmas: bool = False,
    use_dynamic_shifting: bool = False,
    use_exponential_sigmas: bool = False,
    use_karras_sigmas: bool = False,
    invert_sigmas: bool = False,
    start_at_step: int = 0,
    end_at_step: int = -1,
    latent_image: Optional[Dict[str, torch.Tensor]] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Calculates FlowMatch sigmas using Diffusers if available, falling back to exact PyTorch math."""
    seq_len = 1024
    if latent_image is not None and "samples" in latent_image:
        samples = latent_image["samples"]
        shape = samples.shape
        if len(shape) == 5:
            _, _, t_dim, h_dim, w_dim = shape
            seq_len = (t_dim * h_dim * w_dim) // 8
        elif len(shape) == 4:
            _, _, h_dim, w_dim = shape
            seq_len = (h_dim // 2) * (w_dim // 2) if (h_dim >= 2 and w_dim >= 2) else (h_dim * w_dim)

    # Linear interpolation in shift space
    m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
    b = base_shift - m * base_image_seq_len
    mu = m * seq_len + b

    sigmas = None
    if HAS_DIFFUSERS:
        init_kwargs = {
            "num_train_timesteps": num_train_timesteps,
            "shift": shift,
            "use_dynamic_shifting": use_dynamic_shifting,
            "base_shift": base_shift,
            "max_shift": max_shift,
            "base_image_seq_len": base_image_seq_len,
            "max_image_seq_len": max_image_seq_len,
            "time_shift_type": time_shift_type,
            "shift_terminal": shift_terminal,
            "use_beta_sigmas": use_beta_sigmas,
            "use_exponential_sigmas": use_exponential_sigmas,
            "use_karras_sigmas": use_karras_sigmas,
            "invert_sigmas": invert_sigmas,
        }
        supported_params = inspect.signature(DiffusersFlowMatchEulerDiscreteScheduler.__init__).parameters
        filtered_kwargs = {k: v for k, v in init_kwargs.items() if k in supported_params}
        try:
            scheduler = DiffusersFlowMatchEulerDiscreteScheduler(**filtered_kwargs)
            set_timesteps_kwargs = {"device": device}
            if use_dynamic_shifting:
                set_timesteps_kwargs["mu"] = mu
            scheduler.set_timesteps(steps, **set_timesteps_kwargs)
            sigmas = scheduler.sigmas.to(dtype=torch.float32)
        except Exception as e:
            logger.warning(f"[RBG Extras] Diffusers FlowMatch failed ({e}). Falling back to pure PyTorch.")
            sigmas = None

    if sigmas is None:
        effective_shift = math.exp(mu) if use_dynamic_shifting else shift
        timesteps = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)
        sigmas = (effective_shift * timesteps) / (1.0 + (effective_shift - 1.0) * timesteps)
        sigmas[-1] = 0.0

    if start_at_step > 0 or (0 < end_at_step < len(sigmas)):
        end_idx = end_at_step if (0 < end_at_step <= len(sigmas)) else len(sigmas)
        sigmas = sigmas[start_at_step:end_idx]

    return sigmas


class FlowMatchEulerDiscreteScheduler:
    """Calculates FlowMatch sigmas with resolution-dependent dynamic shifting and custom step slicing."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "Total sampling steps."}),
                "shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Base time shift factor."}),
                "base_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.01}),
                "max_shift": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 100.0, "step": 0.01}),
                "base_image_seq_len": ("INT", {"default": 256, "min": 1, "max": 8192}),
                "max_image_seq_len": ("INT", {"default": 4096, "min": 1, "max": 16384}),
                "num_train_timesteps": ("INT", {"default": 1000, "min": 1, "max": 10000}),
                "shift_terminal": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "time_shift_type": (["exponential", "linear"], {"default": "exponential"}),
                "use_beta_sigmas": ("BOOLEAN", {"default": False}),
                "use_dynamic_shifting": ("BOOLEAN", {"default": False, "tooltip": "Adjusts shift based on latent token length."}),
                "use_exponential_sigmas": ("BOOLEAN", {"default": False}),
                "use_karras_sigmas": ("BOOLEAN", {"default": False}),
                "invert_sigmas": ("BOOLEAN", {"default": False}),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": -1, "min": -1, "max": 10000}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            },
            "optional": {
                "latent_image": ("LATENT", {"tooltip": "Optional latent to auto-extract token sequence length for dynamic shifting."}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    CATEGORY = "✨RBG-Suite-Pack/Schedulers"
    FUNCTION = "get_sigmas"

    def get_sigmas(self, steps: int, **kwargs) -> Tuple[torch.Tensor]:
        device = kwargs.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs["device"] = device
        sigmas = calculate_flowmatch_sigmas(steps=steps, **kwargs)
        return (sigmas,)


# ==============================================================================
# 2. SAMPLER DPM++ 2M SHARP
# ==============================================================================

@torch.no_grad()
def sample_dpmpp_2m_sharp(
    model: Callable,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: Optional[Dict[str, Any]] = None,
    callback: Optional[Callable] = None,
    disable: Optional[bool] = None,
    sharpness: float = 0.15,
) -> torch.Tensor:
    """
    DPM++ 2M with progressive 2nd-order history damping (envy-ai).
    Scales previous denoised predictions to counteract multi-step smoothing.
    """
    extra_args = {} if extra_args is None else extra_args
    sigmas = sigmas.to(device=x.device, dtype=x.dtype)
    s_in = x.new_ones([x.shape[0]])
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        denoised = model(x, sigma * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': denoised})

        t = t_fn(sigma)
        t_next = t_fn(sigma_next)
        h = t_next - t

        if old_denoised is None or sigma_next == 0:
            x = (sigma_next / sigma) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            adjustment_factor = 1.0 + sharpness * (i / (len(sigmas) - 1)) ** 2
            old_denoised_scaled = old_denoised * adjustment_factor
            denoised_d = (1.0 + 1.0 / (2.0 * r)) * denoised - (1.0 / (2.0 * r)) * old_denoised_scaled
            x = (sigma_next / sigma) * x - (-h).expm1() * denoised_d

        old_denoised = denoised
    return x


class SamplerDPMPP_2M_Sharp:
    """DPM++ 2M Sharp sampler node with progressive 2nd-order history damping."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sharpness": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 2.0, "step": 0.01,
                                        "tooltip": "Sharpness factor. 0.0 = stock DPM++ 2M. Higher values progressively damp 2nd-order history."}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    CATEGORY = "✨RBG-Suite-Pack/Samplers"
    FUNCTION = "get_sampler"

    def get_sampler(self, sharpness: float):
        def _sampler(model, x, sigmas, extra_args=None, callback=None, disable=None):
            return sample_dpmpp_2m_sharp(
                model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, sharpness=sharpness
            )
        return (comfy.samplers.KSAMPLER(_sampler),)


setattr(comfy.k_diffusion.sampling, "sample_dpmpp_2m_sharp", sample_dpmpp_2m_sharp)


# ==============================================================================
# 3. SIGMOID OFFSET SCHEDULER
# ==============================================================================

def calculate_sigmoid_offset_sigmas(
    model_sampling: Any = None,
    steps: int = 30,
    square_k: float = 1.0,
    base_c: float = 0.5,
    start_sigma: float = 1.0,
) -> torch.Tensor:
    """Calculates sigma curves using a parameterized sigmoid distribution."""
    if steps <= 0:
        return torch.tensor([start_sigma, 0.0], dtype=torch.float32)

    k = (square_k ** 2) * 6.0 + 1e-4
    x = torch.linspace(0.0, 1.0, steps + 1)
    sig = 1.0 / (1.0 + torch.exp(k * (x - base_c)))
    sig_min, sig_max = sig[-1], sig[0]
    if sig_max > sig_min:
        sig = (sig - sig_min) / (sig_max - sig_min)
    else:
        sig = 1.0 - x
    sig[-1] = 0.0

    max_model_sigma = 1.0
    if model_sampling is not None:
        if hasattr(model_sampling, "sigmas") and len(model_sampling.sigmas) > 0:
            max_model_sigma = float(model_sampling.sigmas.max())
        elif hasattr(model_sampling, "sigma_max"):
            max_model_sigma = float(model_sampling.sigma_max)

    scale = max_model_sigma * start_sigma
    return (sig * scale).to(dtype=torch.float32)


class SigmoidOffsetScheduler:
    """Sigmoid Offset Scheduler for S-curve step allocation."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Model providing the base sampling maximum sigma."}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "square_k": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "base_c": ("FLOAT", {"default": 0.5, "min": -5.0, "max": 5.0, "step": 0.01}),
                "start_sigma": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    CATEGORY = "✨RBG-Suite-Pack/Schedulers"
    FUNCTION = "get_sigmas"

    def get_sigmas(self, model, steps: int, square_k: float, base_c: float, start_sigma: float):
        model_sampling = model.get_model_object("model_sampling")
        sigmas = calculate_sigmoid_offset_sigmas(
            model_sampling=model_sampling,
            steps=steps,
            square_k=square_k,
            base_c=base_c,
            start_sigma=start_sigma,
        )
        return (sigmas,)


# ==============================================================================
# 4. TCD SCHEDULER & SAMPLER
# ==============================================================================

class TCDScheduler:
    """Uniform scheduler for Trajectory Consistency Distillation."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Model providing model_sampling specs."}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    CATEGORY = "✨RBG-Suite-Pack/Schedulers"
    FUNCTION = "get_sigmas"

    def get_sigmas(self, model, steps: int):
        model_sampling = model.get_model_object("model_sampling")
        sigmas = comfy.samplers.calculate_sigmas(model_sampling, "ddim_uniform", steps)
        return (sigmas,)


@torch.no_grad()
def sample_tcd(
    model: Callable,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: Optional[Dict[str, Any]] = None,
    callback: Optional[Callable] = None,
    disable: Optional[bool] = None,
    eta: float = 1.0,
    gamma: float = 0.3,
    noise_sampler: Optional[Callable] = None,
) -> torch.Tensor:
    """TCD ancestral sampler. When gamma=0, behaves fully deterministically; when gamma>0, injects calibrated noise."""
    extra_args = {} if extra_args is None else extra_args
    sigmas = sigmas.to(device=x.device, dtype=x.dtype)
    s_in = x.new_ones([x.shape[0]])

    if noise_sampler is None:
        noise_sampler = comfy.k_diffusion.sampling.default_noise_sampler(x)

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        denoised = model(x, sigma * s_in, **extra_args)
        sigma_down, sigma_up = comfy.k_diffusion.sampling.get_ancestral_step(sigma, sigma_next, eta=eta)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': denoised})

        d = comfy.k_diffusion.sampling.to_d(x, sigma, denoised)
        dt = sigma_down - sigma

        if sigma_next == 0 or gamma == 0:
            x = x + d * dt
        else:
            noise = noise_sampler(sigma, sigma_next)
            x = x + d * dt + noise * (gamma * sigma_up)
    return x


setattr(comfy.k_diffusion.sampling, "sample_tcd", sample_tcd)


class SamplerTCD:
    """TCD Sampler node. gamma=0 yields deterministic sampling; gamma>0 injects ancestral noise."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gamma": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01,
                                    "tooltip": "Stochasticity factor. 0.0 = deterministic DDIM trajectory; 1.0 = full Euler-A noise."}),
                "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                  "tooltip": "Sub-step ancestral scale factor."}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    CATEGORY = "✨RBG-Suite-Pack/Samplers"
    FUNCTION = "get_sampler"

    def get_sampler(self, gamma: float, eta: float):
        def _sampler(model, x, sigmas, extra_args=None, callback=None, disable=None):
            return sample_tcd(
                model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
                gamma=gamma, eta=eta
            )
        return (comfy.samplers.KSAMPLER(_sampler),)


SamplerTCDEulerA = SamplerTCD


# ==============================================================================
# 5. RESTART SAMPLING SUITE
# ==============================================================================

DEFAULT_RESTART_SEGMENTS = [(3, 2, 0.06, 0.30), (3, 1, 0.30, 0.59)]


def parse_restart_segments(segment_str: str) -> List[Tuple[int, int, float, float]]:
    """Parses restart segment tuples [N_restart, K_steps, t_min, t_max]."""
    if not segment_str or segment_str.strip().lower() in ("default", ""):
        return list(DEFAULT_RESTART_SEGMENTS)
    try:
        data = ast.literal_eval(f"[{segment_str.strip()}]")
        parsed = []
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) == 4:
                parsed.append((int(item[0]), int(item[1]), float(item[2]), float(item[3])))
        if parsed:
            # Sort segments descending by t_min to maintain insertion integrity
            parsed.sort(key=lambda seg: seg[2], reverse=True)
            return parsed
        logger.warning(f"[RBG Extras] No valid 4-tuples found in '{segment_str}'. Using default segments.")
    except Exception as e:
        logger.warning(f"[RBG Extras] Failed to parse restart segments string '{segment_str}' ({e}). Using defaults.")
    return list(DEFAULT_RESTART_SEGMENTS)


class RestartScheduler:
    """Inserts ascending restart loops into a base sigma trajectory. Requires either sigmas or model."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "segments": ("STRING", {"default": "[3,2,0.06,0.30],[3,1,0.30,0.59]", "multiline": False,
                                        "tooltip": "Segments format: [N_restart, K_steps, t_min, t_max]."}),
            },
            "optional": {
                "sigmas": ("SIGMAS", {"tooltip": "Optional incoming base sigmas."}),
                "model": ("MODEL", {"tooltip": "Model providing model_sampling to calculate base sigmas if input sigmas is empty."}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    CATEGORY = "✨RBG-Suite-Pack/Schedulers"
    FUNCTION = "get_sigmas"

    def get_sigmas(
        self, steps: int, scheduler: str, segments: str,
        sigmas: Optional[torch.Tensor] = None, model: Optional[Any] = None
    ):
        if sigmas is not None:
            base_sigmas = sigmas
        elif model is not None:
            model_sampling = model.get_model_object("model_sampling")
            base_sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps)
        else:
            raise ValueError(
                "[RBG Extras] RestartScheduler requires either a 'sigmas' input or a 'model' input to calculate base sigmas."
            )

        parsed_segments = parse_restart_segments(segments)
        out_sigmas = list(base_sigmas.cpu().numpy())

        for n_rest, k_steps, t_min, t_max in parsed_segments:
            num_restarts = max(0, n_rest - 1)
            if num_restarts == 0:
                continue
            restart_curve = torch.linspace(t_max, t_min, k_steps + 1).tolist()
            insert_idx = None
            for idx, s in enumerate(out_sigmas):
                if s <= t_min:
                    insert_idx = idx
                    break
            if insert_idx is not None:
                for _ in range(num_restarts):
                    for r_val in restart_curve:
                        out_sigmas.insert(insert_idx, r_val)
                        insert_idx += 1

        return (torch.tensor(out_sigmas, dtype=torch.float32),)


class RestartSampler:
    """Wraps a sampler to execute restart noise loops in chunked mode, preserving multistep history."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": ("SAMPLER", {"tooltip": "Base sampler to execute denoising chunks."}),
                "noise_mode": (["diffusion_ve", "rectified_flow"], {"default": "diffusion_ve",
                                "tooltip": "Noise math: 'diffusion_ve' for standard models; 'rectified_flow' for continuous linear flows (Flux/Qwen/Wan)."}),
                "chunked_mode": ("BOOLEAN", {"default": True,
                                 "tooltip": "Preserves multistep history across continuous descending chunks."}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    CATEGORY = "✨RBG-Suite-Pack/Samplers"
    FUNCTION = "get_sampler"

    def get_sampler(self, sampler, noise_mode: str = "diffusion_ve", chunked_mode: bool = True, seed: int = 0):
        def _restart_sampling_wrapper(model, x, sigmas, extra_args=None, callback=None, disable=None):
            extra_args = {} if extra_args is None else extra_args
            sigmas = sigmas.to(device=x.device, dtype=x.dtype)
            gen = torch.Generator(device=x.device).manual_seed(seed)

            extra_opts = getattr(sampler, "extra_options", {})
            combined_args = {**extra_args, **extra_opts}

            if not chunked_mode:
                # Fallback unchunked mode
                for i in range(len(sigmas) - 1):
                    curr_s = sigmas[i]
                    next_s = sigmas[i + 1]
                    if next_s > curr_s:
                        noise = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
                        if noise_mode == "rectified_flow":
                            scale = (1.0 - next_s) / (1.0 - curr_s + 1e-6)
                            add_var = max(0.0, float(next_s ** 2 - (scale * curr_s) ** 2))
                            x = scale * x + noise * math.sqrt(add_var)
                        else:
                            x = x + noise * torch.sqrt(next_s ** 2 - curr_s ** 2)
                    else:
                        step_sigmas = torch.tensor([curr_s, next_s], device=x.device, dtype=sigmas.dtype)
                        x = sampler.sampler_function(
                            model, x, step_sigmas, extra_args=combined_args, callback=callback, disable=True
                        )
                return x

            # Chunked Mode: Split sigmas into strictly descending slices
            chunks: List[List[float]] = []
            current_chunk = [float(sigmas[0])]

            for i in range(len(sigmas) - 1):
                curr_s = float(sigmas[i])
                next_s = float(sigmas[i + 1])
                if next_s > curr_s:
                    chunks.append(current_chunk)
                    current_chunk = [next_s]
                else:
                    current_chunk.append(next_s)
            chunks.append(current_chunk)

            step_offset = [0]
            for chunk_idx, chunk in enumerate(chunks):
                chunk_sigmas = torch.tensor(chunk, device=x.device, dtype=sigmas.dtype)
                if len(chunk_sigmas) > 1:
                    def _chunk_callback(step_info):
                        if callback is not None:
                            cb_data = dict(step_info)
                            cb_data['i'] = cb_data.get('i', 0) + step_offset[0]
                            callback(cb_data)

                    x = sampler.sampler_function(
                        model, x, chunk_sigmas, extra_args=combined_args, callback=_chunk_callback, disable=disable
                    )
                    step_offset[0] += len(chunk_sigmas) - 1

                # Re-noise jump before the start of the next chunk
                if chunk_idx < len(chunks) - 1:
                    curr_s = torch.tensor(chunk[-1], device=x.device, dtype=x.dtype)
                    next_s = torch.tensor(chunks[chunk_idx + 1][0], device=x.device, dtype=x.dtype)
                    if next_s > curr_s:
                        noise = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
                        if noise_mode == "rectified_flow":
                            scale = (1.0 - next_s) / (1.0 - curr_s + 1e-6)
                            add_var = max(0.0, float(next_s ** 2 - (scale * curr_s) ** 2))
                            x = scale * x + noise * math.sqrt(add_var)
                        else:
                            x = x + noise * torch.sqrt(next_s ** 2 - curr_s ** 2)

            return x

        return (comfy.samplers.KSAMPLER(_restart_sampling_wrapper),)


class KSamplerWithRestarts:
    """All-in-one KSampler implementing restart noise injection with full preview and noise mask support."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "segments": ("STRING", {"default": "[3,2,0.06,0.30],[3,1,0.30,0.59]", "multiline": False}),
                "noise_mode": (["diffusion_ve", "rectified_flow"], {"default": "diffusion_ve"}),
            },
            "optional": {
                "noise_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    CATEGORY = "✨RBG-Suite-Pack/Pipelines"
    FUNCTION = "sample"

    def sample(
        self, model, seed: int, steps: int, cfg: float, sampler_name: str, scheduler: str,
        positive, negative, latent_image, denoise: float = 1.0, segments: str = "",
        noise_mode: str = "diffusion_ve", noise_mask: Optional[torch.Tensor] = None
    ):
        device = comfy.model_management.get_torch_device()
        latent = latent_image["samples"].clone().to(device)

        if hasattr(comfy.sample, "fix_empty_latent_channels"):
            latent = comfy.sample.fix_empty_latent_channels(model, latent)

        model_sampling = model.get_model_object("model_sampling")
        base_sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps)

        if denoise < 1.0:
            steps_to_take = max(1, int(steps * denoise))
            base_sigmas = base_sigmas[-(steps_to_take + 1):]

        sched_node = RestartScheduler()
        (sigmas,) = sched_node.get_sigmas(steps, scheduler, segments, sigmas=base_sigmas)
        sigmas = sigmas.to(device)

        sampler_inst = comfy.samplers.sampler_object(sampler_name)
        sampler_node = RestartSampler()
        (wrapped_sampler,) = sampler_node.get_sampler(sampler_inst, noise_mode=noise_mode, chunked_mode=True, seed=seed)

        noise = comfy.sample.prepare_noise(latent, seed, noise_mask)

        callback = None
        if HAS_LATENT_PREVIEW:
            callback = latent_preview.prepare_callback(model, len(sigmas) - 1)

        samples = comfy.sample.sample_custom(
            model=model,
            noise=noise,
            cfg=cfg,
            sampler=wrapped_sampler,
            sigmas=sigmas,
            positive=positive,
            negative=negative,
            latent_image=latent,
            noise_mask=noise_mask,
            callback=callback,
            seed=seed,
        )
        return ({"samples": samples},)


# ==============================================================================
# 6. ADAPTIVE ODE SOLVERS
# ==============================================================================

try:
    import torchdiffeq
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False


class ODESolver:
    """Adaptive ODE solver powered by torchdiffeq. Evaluates endpoints directly to minimize memory footprint."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "solver": (["fehlberg2", "bosh3", "adaptive_heun", "dopri5"], {"default": "fehlberg2"}),
                "log_relative_tolerance": ("FLOAT", {"default": -4.0, "min": -15.0, "max": 0.0, "step": 0.1}),
                "log_absolute_tolerance": ("FLOAT", {"default": -6.0, "min": -15.0, "max": 0.0, "step": 0.1}),
                "max_steps": ("INT", {"default": 100, "min": 1, "max": 1000}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    CATEGORY = "✨RBG-Suite-Pack/Samplers"
    FUNCTION = "get_sampler"

    def get_sampler(self, solver: str, log_relative_tolerance: float, log_absolute_tolerance: float, max_steps: int):
        if not HAS_TORCHDIFFEQ:
            raise ImportError("[RBG Extras] torchdiffeq is required for ODESolver. Run: pip install torchdiffeq")

        rtol = 10.0 ** log_relative_tolerance
        atol = 10.0 ** log_absolute_tolerance

        @torch.no_grad()
        def _ode_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None):
            extra_args = {} if extra_args is None else extra_args
            sigmas = sigmas.to(device=x.device, dtype=torch.float32)

            nfe_count = [0]
            s_in = x.new_ones([x.shape[0]])

            def ode_func(t, y):
                nfe_count[0] += 1
                sigma = t.expand(y.shape[0])
                denoised = model(y, sigma * s_in, **extra_args)
                d = (y - denoised) / (t if t > 1e-6 else 1e-6)
                if callback is not None:
                    callback({'x': y, 'i': nfe_count[0], 'sigma': t, 'sigma_hat': t, 'denoised': denoised})
                return d

            t_start = sigmas[0]
            t_end = sigmas[-1] if sigmas[-1] > 1e-4 else torch.tensor(1e-4, device=x.device, dtype=torch.float32)
            t_span = torch.tensor([t_start, t_end], device=x.device, dtype=torch.float32)

            sol = torchdiffeq.odeint(
                ode_func, x, t_span, method=solver, rtol=rtol, atol=atol,
                options={"max_num_steps": max_steps}
            )
            out_x = sol[-1]

            if sigmas[-1] == 0:
                final_denoised = model(out_x, t_end * s_in, **extra_args)
                out_x = final_denoised

            logger.info(f"[RBG Extras] ODESolver ({solver}) completed in {nfe_count[0]} evaluations.")
            return out_x

        return (comfy.samplers.KSAMPLER(_ode_sampler),)


# ==============================================================================
# 7. FLOW-SOLVER 2M
# ==============================================================================

@torch.no_grad()
def sample_flow_solver_2m(
    model: Callable,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: Optional[Dict[str, Any]] = None,
    callback: Optional[Callable] = None,
    disable: Optional[bool] = None,
) -> torch.Tensor:
    """2nd-order Multistep Adams-Bashforth ODE solver for Rectified Flow velocity fields (1 NFE per step)."""
    extra_args = {} if extra_args is None else extra_args
    sigmas = sigmas.to(device=x.device, dtype=x.dtype)
    s_in = x.new_ones([x.shape[0]])
    old_v = None

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        dt = sigma_next - sigma

        denoised = model(x, sigma * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': denoised})

        v = comfy.k_diffusion.sampling.to_d(x, sigma, denoised)

        if old_v is None or sigma_next == 0:
            x = x + v * dt
        else:
            dt_prev = sigma - sigmas[i - 1]
            r = dt / (dt_prev if abs(dt_prev) > 1e-7 else 1e-7)
            v_eff = (1.0 + 0.5 * r) * v - (0.5 * r) * old_v
            x = x + v_eff * dt

        old_v = v
    return x


setattr(comfy.k_diffusion.sampling, "sample_flow_solver_2m", sample_flow_solver_2m)


class SamplerFlowSolver2M:
    """Flow-Solver 2M for velocity models (Flux, SD3.5, Wan, LTX-Video, Qwen)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    CATEGORY = "✨RBG-Suite-Pack/Samplers"
    FUNCTION = "get_sampler"

    def get_sampler(self):
        def _sampler(model, x, sigmas, extra_args=None, callback=None, disable=None):
            return sample_flow_solver_2m(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable)
        return (comfy.samplers.KSAMPLER(_sampler),)


# ==============================================================================
# 8. MOMENTUM EULER
# ==============================================================================

@torch.no_grad()
def sample_momentum_euler(
    model: Callable,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: Optional[Dict[str, Any]] = None,
    callback: Optional[Callable] = None,
    disable: Optional[bool] = None,
    momentum: float = 0.15,
) -> torch.Tensor:
    """Euler solver augmented with velocity EMA momentum to dampen high-frequency directional drift."""
    extra_args = {} if extra_args is None else extra_args
    sigmas = sigmas.to(device=x.device, dtype=x.dtype)
    s_in = x.new_ones([x.shape[0]])
    v_buffer = None

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        dt = sigma_next - sigma

        denoised = model(x, sigma * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': denoised})

        d = comfy.k_diffusion.sampling.to_d(x, sigma, denoised)

        if v_buffer is None:
            v_buffer = d
        else:
            v_buffer = (1.0 - momentum) * d + momentum * v_buffer

        x = x + v_buffer * dt
    return x


setattr(comfy.k_diffusion.sampling, "sample_momentum_euler", sample_momentum_euler)


class SamplerMomentumEuler:
    """Momentum-augmented Euler sampler."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "momentum": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.8, "step": 0.01,
                                        "tooltip": "EMA momentum factor. Blends past velocity direction into the current step."}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    CATEGORY = "✨RBG-Suite-Pack/Samplers"
    FUNCTION = "get_sampler"

    def get_sampler(self, momentum: float):
        def _sampler(model, x, sigmas, extra_args=None, callback=None, disable=None):
            return sample_momentum_euler(
                model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, momentum=momentum
            )
        return (comfy.samplers.KSAMPLER(_sampler),)


# ==============================================================================
# 9. RESOLUTION-AWARE FLOW SCHEDULER
# ==============================================================================

def calculate_resolution_aware_sigmas(
    latent_image: Optional[Dict[str, torch.Tensor]] = None,
    steps: int = 20,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    terminal: float = 0.0,
    spatial_patch_size: int = 2,
    temporal_patch_size: int = 1,
) -> torch.Tensor:
    """Resolution-aware scheduler with configurable 2D/3D DiT patch dimensions."""
    seq_len = 1024
    if latent_image is not None and "samples" in latent_image:
        samples = latent_image["samples"]
        shape = samples.shape
        if len(shape) == 5:
            _, _, t_dim, h_dim, w_dim = shape
            t_tokens = max(1, t_dim // temporal_patch_size)
            h_tokens = max(1, h_dim // spatial_patch_size)
            w_tokens = max(1, w_dim // spatial_patch_size)
            seq_len = t_tokens * h_tokens * w_tokens
        elif len(shape) == 4:
            _, _, h_dim, w_dim = shape
            h_tokens = max(1, h_dim // spatial_patch_size)
            w_tokens = max(1, w_dim // spatial_patch_size)
            seq_len = h_tokens * w_tokens

    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = m * seq_len + b
    shift = float(math.exp(mu))

    if terminal > 0.0:
        active_timesteps = torch.linspace(1.0, terminal, steps, dtype=torch.float32)
        timesteps = torch.cat([active_timesteps, torch.tensor([0.0])])
    else:
        timesteps = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)

    sigmas = shift * timesteps / (1.0 + (shift - 1.0) * timesteps)
    sigmas[-1] = 0.0
    return sigmas


class ResolutionAwareFlowScheduler:
    """Resolution-aware scheduler with model-specific patch dimension support."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "base_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                "max_shift": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 10.0, "step": 0.01}),
                "base_seq_len": ("INT", {"default": 256, "min": 1, "max": 16384}),
                "max_seq_len": ("INT", {"default": 4096, "min": 1, "max": 65536}),
                "terminal": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01,
                                       "tooltip": "Cutoff threshold before final landing step. Recommended: 0.08 for LTX/Wan."}),
                "spatial_patch_size": ("INT", {"default": 2, "min": 1, "max": 8, "tooltip": "DiT spatial patch size (Flux=2, Wan=2, LTX=1)."}),
                "temporal_patch_size": ("INT", {"default": 1, "min": 1, "max": 8, "tooltip": "DiT temporal patch size for video models."}),
            },
            "optional": {
                "latent_image": ("LATENT", {"tooltip": "Incoming 4D image or 5D video latent to calculate sequence length."}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    CATEGORY = "✨RBG-Suite-Pack/Schedulers"
    FUNCTION = "get_sigmas"

    def get_sigmas(
        self, steps: int, base_shift: float, max_shift: float,
        base_seq_len: int, max_seq_len: int, terminal: float,
        spatial_patch_size: int = 2, temporal_patch_size: int = 1,
        latent_image: Optional[Dict[str, torch.Tensor]] = None
    ):
        sigmas = calculate_resolution_aware_sigmas(
            latent_image=latent_image,
            steps=steps,
            base_shift=base_shift,
            max_shift=max_shift,
            base_seq_len=base_seq_len,
            max_seq_len=max_seq_len,
            terminal=terminal,
            spatial_patch_size=spatial_patch_size,
            temporal_patch_size=temporal_patch_size,
        )
        return (sigmas,)


# ==============================================================================
# 10. QWEN-IMAGE TURBO SPECIALIZED SCHEDULER
# ==============================================================================

def calculate_comfy_linear_quadratic_sigmas(steps: int, threshold_noise: float = 0.025) -> torch.Tensor:
    """Exact ComfyUI piecewise linear_quadratic_schedule implementation."""
    if steps <= 1:
        return torch.tensor([1.0, 0.0], dtype=torch.float32)

    linear_steps = steps // 2
    linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
    threshold_noise_step_diff = linear_steps - threshold_noise * steps
    quadratic_steps = steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * (quadratic_steps ** 2))
    linear_coef = (threshold_noise / linear_steps) - 2 * threshold_noise_step_diff / (quadratic_steps ** 2)
    const = quadratic_coef * (linear_steps ** 2)
    
    quadratic_sigma_schedule = [
        quadratic_coef * (i ** 2) + linear_coef * i + const
        for i in range(linear_steps, steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return torch.tensor(sigma_schedule, dtype=torch.float32)


QWEN_TURBO_PRESETS: Dict[str, List[float]] = {
    # 1. Alibaba Official Core (8 steps; best for text glyphs, counting, strict layout)
    "official_diffusers_8step": [
        1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0
    ],
    # 2. Reddit Community Detail Boost (8 steps; 0.9786 start for skin pores & anti-saturation)
    "community_detail_boost_8step": [
        0.9786, 0.9750, 0.9520, 0.8903, 0.7898, 0.6505, 0.4724, 0.2556, 0.0
    ],
    # 3. ComfyUI Piecewise Linear-Quadratic (8 steps; matches BasicScheduler exactly)
    "linear_quadratic_8step": [
        1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0
    ],
    # 4. Extended Quality 12-Step (Sweet spot for native 2K generation and CFG 1.2-1.4)
    "extended_quality_12step": [
        1.0, 0.9841, 0.9658, 0.9442, 0.9168, 0.8805, 0.8252, 0.7386, 0.6120, 0.4485, 0.2450, 0.0820, 0.0
    ],
    # 5. Extended Quality 14-Step (Maximum fidelity curve for complex multi-character scenes)
    "extended_quality_14step": [
        1.0, 0.9870, 0.9720, 0.9550, 0.9320, 0.9030, 0.8650, 0.8120, 0.7350, 0.6280, 0.4850, 0.3120, 0.1450, 0.0520, 0.0
    ],
}


def calculate_qwen_turbo_sigmas(preset: str = "community_detail_boost_8step", steps: int = 8) -> torch.Tensor:
    """Computes sigmas for Qwen-Image-2.1-Turbo."""
    # If linear-quadratic is chosen, compute ComfyUI's exact formula dynamically for any step count
    if preset == "linear_quadratic_8step":
        return calculate_comfy_linear_quadratic_sigmas(steps=steps)

    if preset in QWEN_TURBO_PRESETS and steps == (len(QWEN_TURBO_PRESETS[preset]) - 1):
        return torch.tensor(QWEN_TURBO_PRESETS[preset], dtype=torch.float32)

    base_list = QWEN_TURBO_PRESETS.get(preset, QWEN_TURBO_PRESETS["community_detail_boost_8step"])
    base = torch.tensor(base_list, dtype=torch.float32)
    orig_x = torch.linspace(0.0, 1.0, len(base))
    new_x = torch.linspace(0.0, 1.0, steps + 1)

    sigmas = torch.zeros(steps + 1, dtype=torch.float32)
    for idx, x_val in enumerate(new_x):
        idx_right = torch.searchsorted(orig_x, x_val)
        idx_left = max(0, idx_right - 1)
        if idx_right >= len(orig_x):
            sigmas[idx] = base[-1]
        else:
            denom = orig_x[idx_right] - orig_x[idx_left]
            weight = (x_val - orig_x[idx_left]) / (denom if denom > 1e-7 else 1e-7)
            sigmas[idx] = (1.0 - weight) * base[idx_left] + weight * base[idx_right]
    sigmas[-1] = 0.0
    return sigmas


class QwenImageTurboScheduler:
    """Step-allocation scheduler for Qwen-Image-2.1-Turbo."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (
                    list(QWEN_TURBO_PRESETS.keys()),
                    {"default": "community_detail_boost_8step",
                     "tooltip": "official_diffusers (text/layout), community_detail (skin/pores), linear_quadratic (matches BasicScheduler), or extended (2K quality)."}
                ),
                "steps": ("INT", {"default": 8, "min": 1, "max": 30, "step": 1,
                                  "tooltip": "Step count. Match the preset length for exact tables, or change to interpolate."}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    CATEGORY = "✨RBG-Suite-Pack/Schedulers"
    FUNCTION = "get_sigmas"

    def get_sigmas(self, preset: str, steps: int):
        sigmas = calculate_qwen_turbo_sigmas(preset=preset, steps=steps)
        return (sigmas,)


# ==============================================================================
# 11. MODULAR SIGMA TOOLS & PREVIEW
# ==============================================================================

class SigmaSlice:
    """Slices a sigma schedule by step index or denoise percentage."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "start_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_step": ("INT", {"default": -1, "min": -1, "max": 10000}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    CATEGORY = "✨RBG-Suite-Pack/Sigma-Tools"
    FUNCTION = "slice_sigmas"

    def slice_sigmas(self, sigmas: torch.Tensor, start_step: int, end_step: int, denoise: float):
        out = sigmas.clone()
        if denoise < 1.0:
            steps_to_keep = max(1, int((len(out) - 1) * denoise))
            out = out[-(steps_to_keep + 1):]

        if start_step > 0 or (0 < end_step < len(out)):
            end_idx = end_step if (0 < end_step <= len(out)) else len(out)
            out = out[start_step:end_idx]

        return (out,)


class SigmaScale:
    """Scales or multiplies all values in a sigma schedule."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "offset": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    CATEGORY = "✨RBG-Suite-Pack/Sigma-Tools"
    FUNCTION = "scale_sigmas"

    def scale_sigmas(self, sigmas: torch.Tensor, scale: float, offset: float):
        out = (sigmas * scale) + offset
        out = torch.clamp(out, min=0.0)
        return (out,)


class SigmaConcat:
    """Concatenates two sigma schedules seamlessly."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas_a": ("SIGMAS",),
                "sigmas_b": ("SIGMAS",),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    CATEGORY = "✨RBG-Suite-Pack/Sigma-Tools"
    FUNCTION = "concat_sigmas"

    def concat_sigmas(self, sigmas_a: torch.Tensor, sigmas_b: torch.Tensor):
        if sigmas_a[-1] == 0 and len(sigmas_b) > 0:
            sigmas_a = sigmas_a[:-1]
        out = torch.cat([sigmas_a, sigmas_b])
        return (out,)


class SigmasFromString:
    """Parses a comma-separated list of float values into a SIGMAS tensor."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas_string": ("STRING", {"default": "1.0, 0.8, 0.5, 0.2, 0.0", "multiline": True}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    CATEGORY = "✨RBG-Suite-Pack/Sigma-Tools"
    FUNCTION = "from_string"

    def from_string(self, sigmas_string: str):
        vals = [float(v.strip()) for v in sigmas_string.replace("\n", ",").split(",") if v.strip()]
        if not vals:
            vals = [1.0, 0.0]
        return (torch.tensor(vals, dtype=torch.float32),)


class SigmaPreview:
    """Renders a standalone visual plot comparing one or two sigma schedules."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"tooltip": "Primary sigma schedule (Cyan)."}),
            },
            "optional": {
                "sigmas_compare": ("SIGMAS", {"tooltip": "Optional comparison sigma schedule (Orange)."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("plot_image", "info_text")
    CATEGORY = "✨RBG-Suite-Pack/Sigma-Tools"
    FUNCTION = "render_preview"

    def render_preview(self, sigmas: torch.Tensor, sigmas_compare: Optional[torch.Tensor] = None):
        w, h = 512, 384
        img = Image.new("RGB", (w, h), color=(20, 20, 24))
        draw = ImageDraw.Draw(img)

        margin_left, margin_right = 50, 30
        margin_top, margin_bottom = 30, 40
        plot_w = w - margin_left - margin_right
        plot_h = h - margin_top - margin_bottom

        # Grid lines
        for i in range(5):
            gy = margin_top + int(plot_h * (i / 4.0))
            draw.line([(margin_left, gy), (w - margin_right, gy)], fill=(40, 40, 48), width=1)

        s1 = sigmas.cpu().numpy()
        max_val = float(s1.max())
        if sigmas_compare is not None:
            max_val = max(max_val, float(sigmas_compare.cpu().numpy().max()))
        max_val = max(max_val, 1e-4)

        def _plot_curve(curve_vals, color):
            pts = []
            n = len(curve_vals)
            for idx, val in enumerate(curve_vals):
                px = margin_left + int(plot_w * (idx / max(1, n - 1)))
                py = margin_top + int(plot_h * (1.0 - (val / max_val)))
                pts.append((px, py))
            for i in range(len(pts) - 1):
                draw.line([pts[i], pts[i + 1]], fill=color, width=2)
            for px, py in pts:
                draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=color)

        _plot_curve(s1, (0, 220, 255))  # Primary Cyan

        info_lines = [f"Schedule 1: {len(s1)} points | max: {s1.max():.4f} | min: {s1.min():.4f}"]
        if sigmas_compare is not None:
            s2 = sigmas_compare.cpu().numpy()
            _plot_curve(s2, (255, 140, 0))  # Compare Orange
            info_lines.append(f"Schedule 2: {len(s2)} points | max: {s2.max():.4f} | min: {s2.min():.4f}")

        # Render Y-axis text
        draw.text((10, margin_top - 6), f"{max_val:.2f}", fill=(180, 180, 180))
        draw.text((10, margin_top + plot_h - 10), "0.00", fill=(180, 180, 180))

        # Output ComfyUI IMAGE tensor [1, H, W, 3]
        np_img = np.array(img).astype(np.float32) / 255.0
        tensor_img = torch.from_numpy(np_img).unsqueeze(0)

        return (tensor_img, "\n".join(info_lines))


# ==============================================================================
# GLOBAL KSAMPLER DROPDOWN INTEGRATION
# ==============================================================================

EXT_SCHEDULERS = (
    "FlowMatchEulerDiscreteScheduler",
    "sigmoid_offset",
    "tcd",
    "qwen_turbo_detail_boost",
    "qwen_turbo_official",
    "qwen_turbo_linear_quadratic",
)

for sched in EXT_SCHEDULERS:
    if sched not in comfy.samplers.KSampler.SCHEDULERS:
        comfy.samplers.KSampler.SCHEDULERS.append(sched)
    if hasattr(comfy.samplers, "SCHEDULERS") and sched not in comfy.samplers.SCHEDULERS:
        comfy.samplers.SCHEDULERS.append(sched)

_orig_calculate_sigmas = comfy.samplers.calculate_sigmas


def _hooked_calculate_sigmas(model_sampling, scheduler_name, steps):
    if scheduler_name == "qwen_turbo_detail_boost":
        return calculate_qwen_turbo_sigmas(preset="community_detail_boost_8step", steps=steps)
    if scheduler_name == "qwen_turbo_official":
        return calculate_qwen_turbo_sigmas(preset="official_diffusers_8step", steps=steps)
    if scheduler_name == "qwen_turbo_linear_quadratic":
        return calculate_qwen_turbo_sigmas(preset="linear_quadratic_8step", steps=steps)
    if scheduler_name == "FlowMatchEulerDiscreteScheduler":
        return calculate_flowmatch_sigmas(steps=steps)
    if scheduler_name == "sigmoid_offset":
        return calculate_sigmoid_offset_sigmas(model_sampling=model_sampling, steps=steps)
    if scheduler_name == "tcd":
        return _orig_calculate_sigmas(model_sampling, "ddim_uniform", steps)
    return _orig_calculate_sigmas(model_sampling, scheduler_name, steps)


comfy.samplers.calculate_sigmas = _hooked_calculate_sigmas


# ==============================================================================
# NODE REGISTRATION DICTIONARIES
# ==============================================================================

NODE_CLASS_MAPPINGS = {
    "FlowMatchEulerDiscreteScheduler": FlowMatchEulerDiscreteScheduler,
    "SamplerDPMPP_2M_Sharp": SamplerDPMPP_2M_Sharp,
    "SigmoidOffsetScheduler": SigmoidOffsetScheduler,
    "TCDScheduler": TCDScheduler,
    "SamplerTCD": SamplerTCD,
    "RestartScheduler": RestartScheduler,
    "RestartSampler": RestartSampler,
    "KSamplerWithRestarts": KSamplerWithRestarts,
    "ODESolver": ODESolver,
    "SamplerFlowSolver2M": SamplerFlowSolver2M,
    "SamplerMomentumEuler": SamplerMomentumEuler,
    "ResolutionAwareFlowScheduler": ResolutionAwareFlowScheduler,
    "QwenImageTurboScheduler": QwenImageTurboScheduler,
    "SigmaSlice": SigmaSlice,
    "SigmaScale": SigmaScale,
    "SigmaConcat": SigmaConcat,
    "SigmasFromString": SigmasFromString,
    "SigmaPreview": SigmaPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlowMatchEulerDiscreteScheduler": "FlowMatch Euler Discrete Scheduler",
    "SamplerDPMPP_2M_Sharp": "Sampler DPM++ 2M Sharp",
    "SigmoidOffsetScheduler": "Sigmoid Offset Scheduler",
    "TCDScheduler": "TCD Scheduler",
    "SamplerTCD": "Sampler TCD",
    "RestartScheduler": "Restart Scheduler",
    "RestartSampler": "Restart Sampler",
    "KSamplerWithRestarts": "KSampler With Restarts",
    "ODESolver": "ODE Solver (Adaptive)",
    "SamplerFlowSolver2M": "Sampler Flow-Solver 2M (Rectified Flow)",
    "SamplerMomentumEuler": "Sampler Momentum Euler (TADA)",
    "ResolutionAwareFlowScheduler": "Resolution-Aware Flow Scheduler",
    "QwenImageTurboScheduler": "Qwen-Image Turbo Scheduler",
    "SigmaSlice": "Sigma Slice",
    "SigmaScale": "Sigma Scale",
    "SigmaConcat": "Sigma Concat",
    "SigmasFromString": "Sigmas From String",
    "SigmaPreview": "Sigma Preview & Compare",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]