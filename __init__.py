"""
ComfyUI-RBG-Sampler-Scheduler-Extras 📦
Consolidated high-precision samplers & schedulers suite:
1. FlowMatch Euler Discrete Scheduler (erosDiffusion) -> KSampler Scheduler + Node
2. DPM++ 2M Sharp Sampler (envy-ai)                   -> KSampler Sampler + Node
3. Sigmoid Offset Scheduler (silveroxides)            -> KSampler Scheduler + Node
4. TCD Scheduler & Samplers (dfl)                     -> KSampler Sampler/Scheduler + Node
5. Restart Sampling Suite (ssitu)                     -> KSampler With Restarts Node + Modular Nodes
6. Adaptive ODE Solvers (redhottensors)               -> Modular Node (SamplerCustom)
"""

import math
import torch
from tqdm.auto import trange

import comfy.samplers
import comfy.k_diffusion.sampling
import comfy.sample

# ==============================================================================
# 1. FLOWMATCH EULER DISCRETE SCHEDULER (erosDiffusion)
# ==============================================================================

try:
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler as DiffusersFlowMatchEulerDiscreteScheduler
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False


def calculate_flowmatch_sigmas(
    steps=20,
    shift=3.0,
    base_shift=0.5,
    max_shift=1.15,
    base_image_seq_len=256,
    max_image_seq_len=4096,
    num_train_timesteps=1000,
    shift_terminal=0.1,
    time_shift_type="exponential",
    use_beta_sigmas=False,
    use_dynamic_shifting=False,
    use_exponential_sigmas=False,
    use_karras_sigmas=False,
    invert_sigmas=False,
    start_at_step=0,
    end_at_step=-1,
    device="cpu",
):
    if HAS_DIFFUSERS:
        scheduler_kwargs = {
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
        scheduler = DiffusersFlowMatchEulerDiscreteScheduler(**scheduler_kwargs)
        scheduler.set_timesteps(steps, device=device)
        sigmas = scheduler.sigmas.to(dtype=torch.float32)
    else:
        timesteps = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)
        sigmas = timesteps / (timesteps + shift * (1.0 - timesteps))

    if start_at_step > 0 or (end_at_step > 0 and end_at_step < len(sigmas)):
        end_idx = end_at_step if (end_at_step > 0 and end_at_step <= len(sigmas)) else len(sigmas)
        sigmas = sigmas[start_at_step:end_idx]

    return sigmas


class FlowMatchEulerDiscreteScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "base_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.01}),
                "max_shift": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 100.0, "step": 0.01}),
                "base_image_seq_len": ("INT", {"default": 256, "min": 1, "max": 8192}),
                "max_image_seq_len": ("INT", {"default": 4096, "min": 1, "max": 16384}),
                "num_train_timesteps": ("INT", {"default": 1000, "min": 1, "max": 10000}),
                "shift_terminal": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "stochastic_sampling": ("BOOLEAN", {"default": False}),
                "time_shift_type": (["exponential", "linear"], {"default": "exponential"}),
                "use_beta_sigmas": ("BOOLEAN", {"default": False}),
                "use_dynamic_shifting": ("BOOLEAN", {"default": False}),
                "use_exponential_sigmas": ("BOOLEAN", {"default": False}),
                "use_karras_sigmas": ("BOOLEAN", {"default": False}),
                "invert_sigmas": ("BOOLEAN", {"default": False}),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": -1, "min": -1, "max": 10000}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"
    FUNCTION = "get_sigmas"

    def get_sigmas(self, steps, **kwargs):
        device = kwargs.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs["device"] = device
        sigmas = calculate_flowmatch_sigmas(steps=steps, **kwargs)
        return (sigmas,)


# ==============================================================================
# 2. SAMPLER DPM++ 2M SHARP (envy-ai)
# ==============================================================================

@torch.no_grad()
def sample_dpmpp_2m_sharp(model, x, sigmas, extra_args=None, callback=None, disable=None, sharpness=0.15):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t = t_fn(sigmas[i])
        t_next = t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigmas[i + 1] / sigmas[i]) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            adjustment_factor = 1.0 + sharpness * (i / len(sigmas)) ** 2
            old_denoised_scaled = old_denoised * adjustment_factor
            denoised_d = (1.0 + 1.0 / (2.0 * r)) * denoised - (1.0 / (2.0 * r)) * old_denoised_scaled
            x = (sigmas[i + 1] / sigmas[i]) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x


class SamplerDPMPP_2M_Sharp:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sharpness": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "model/sampling/samplers"
    FUNCTION = "get_sampler"

    def get_sampler(self, sharpness):
        def _sampler(model, x, sigmas, extra_args=None, callback=None, disable=None):
            return sample_dpmpp_2m_sharp(
                model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, sharpness=sharpness
            )
        return (comfy.samplers.KSAMPLER(_sampler),)


# Register dpmpp_2m_sharp in comfy.k_diffusion for standard KSampler usage
setattr(comfy.k_diffusion.sampling, "sample_dpmpp_2m_sharp", sample_dpmpp_2m_sharp)


# ==============================================================================
# 3. SIGMOID OFFSET SCHEDULER (silveroxides)
# ==============================================================================

def calculate_sigmoid_offset_sigmas(model_sampling=None, steps=30, square_k=1.0, base_c=0.5, start_sigma=1.0):
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

    if model_sampling is not None:
        if hasattr(model_sampling, "sigmas") and len(model_sampling.sigmas) > 0:
            max_model_sigma = float(model_sampling.sigmas.max())
        elif hasattr(model_sampling, "sigma_max"):
            max_model_sigma = float(model_sampling.sigma_max)
        else:
            max_model_sigma = 1.0
    else:
        max_model_sigma = 1.0

    scale = max_model_sigma if start_sigma == 1.0 else (start_sigma * max_model_sigma)
    sigmas = sig * scale
    return sigmas.to(dtype=torch.float32)


class SigmoidOffsetScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "square_k": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "base_c": ("FLOAT", {"default": 0.5, "min": -5.0, "max": 5.0, "step": 0.01}),
                "start_sigma": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"
    FUNCTION = "get_sigmas"

    def get_sigmas(self, model, steps, square_k, base_c, start_sigma):
        model_sampling = model.get_model_object("model_sampling")
        sigmas = calculate_sigmoid_offset_sigmas(
            model_sampling=model_sampling,
            steps=steps,
            square_k=square_k,
            base_c=base_c,
            start_sigma=start_sigma
        )
        return (sigmas,)


# ==============================================================================
# 4. TCD SCHEDULER & SAMPLERS (dfl)
# ==============================================================================

class TCDScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"
    FUNCTION = "get_sigmas"

    def get_sigmas(self, model, steps):
        model_sampling = model.get_model_object("model_sampling")
        sigmas = comfy.samplers.calculate_sigmas(model_sampling, "ddim_uniform", steps)
        return (sigmas,)


@torch.no_grad()
def sample_tcd(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, gamma=0.3):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = comfy.k_diffusion.sampling.get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            d = comfy.k_diffusion.sampling.to_d(x, sigmas[i], denoised)
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            d = comfy.k_diffusion.sampling.to_d(x, sigmas[i], denoised)
            dt = sigma_down - sigmas[i]
            x = x + d * dt + torch.randn_like(x) * (s_noise * sigma_up * (1.0 + gamma))
    return x


# Register sample_tcd in comfy.k_diffusion for standard KSampler usage
setattr(comfy.k_diffusion.sampling, "sample_tcd", sample_tcd)


class SamplerTCDEulerA:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gamma": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"
    FUNCTION = "get_sampler"

    def get_sampler(self, gamma):
        def _sampler(model, x, sigmas, extra_args=None, callback=None, disable=None):
            return sample_tcd(
                model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, gamma=gamma
            )
        return (comfy.samplers.KSAMPLER(_sampler),)


class SamplerTCD:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gamma": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"
    FUNCTION = "get_sampler"

    def get_sampler(self, gamma):
        def _sampler(model, x, sigmas, extra_args=None, callback=None, disable=None):
            return sample_tcd(
                model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, gamma=gamma
            )
        return (comfy.samplers.KSAMPLER(_sampler),)


# ==============================================================================
# GLOBAL KSAMPLER DROPDOWN INTEGRATION (Samplers & Schedulers)
# ==============================================================================

# 1. Register Extra Schedulers
for sched in ("FlowMatchEulerDiscreteScheduler", "sigmoid_offset", "tcd"):
    if sched not in comfy.samplers.KSampler.SCHEDULERS:
        comfy.samplers.KSampler.SCHEDULERS.append(sched)
    if hasattr(comfy.samplers, "SCHEDULERS") and sched not in comfy.samplers.SCHEDULERS:
        comfy.samplers.SCHEDULERS.append(sched)

# 2. Register Extra Samplers into sampler_name
for samp in ("dpmpp_2m_sharp", "tcd"):
    if samp not in comfy.samplers.KSampler.SAMPLERS:
        comfy.samplers.KSampler.SAMPLERS.append(samp)
    if hasattr(comfy.samplers, "SAMPLER_NAMES") and samp not in comfy.samplers.SAMPLER_NAMES:
        comfy.samplers.SAMPLER_NAMES.append(samp)

# 3. Hook calculate_sigmas so standard KSampler handles the extra schedulers
_orig_calculate_sigmas = comfy.samplers.calculate_sigmas


def _hooked_calculate_sigmas(model_sampling, scheduler_name, steps):
    if scheduler_name == "FlowMatchEulerDiscreteScheduler":
        return calculate_flowmatch_sigmas(steps=steps)
    if scheduler_name == "sigmoid_offset":
        return calculate_sigmoid_offset_sigmas(model_sampling=model_sampling, steps=steps)
    if scheduler_name == "tcd":
        return _orig_calculate_sigmas(model_sampling, "ddim_uniform", steps)
    return _orig_calculate_sigmas(model_sampling, scheduler_name, steps)


comfy.samplers.calculate_sigmas = _hooked_calculate_sigmas


# ==============================================================================
# 5. RESTART SAMPLING SUITE (ssitu)
# ==============================================================================

def parse_restart_segments(segment_str):
    if not segment_str or segment_str.strip().lower() in ("default", ""):
        return [(3, 2, 0.06, 0.30), (3, 1, 0.30, 0.59)]
    import ast
    try:
        data = ast.literal_eval(f"[{segment_str.strip()}]")
        parsed = []
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) == 4:
                parsed.append((int(item[0]), int(item[1]), float(item[2]), float(item[3])))
        return parsed if parsed else [(3, 2, 0.06, 0.30), (3, 1, 0.30, 0.59)]
    except Exception:
        return [(3, 2, 0.06, 0.30), (3, 1, 0.30, 0.59)]


class RestartScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "segments": ("STRING", {"default": "[3,2,0.06,0.30],[3,1,0.30,0.59]", "multiline": False}),
            },
            "optional": {
                "sigmas": ("SIGMAS",),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"
    FUNCTION = "get_sigmas"

    def get_sigmas(self, steps, scheduler, segments, sigmas=None):
        if sigmas is not None:
            base_sigmas = sigmas
        else:
            base_sigmas = comfy.samplers.calculate_sigmas(None, scheduler, steps)

        parsed_segments = parse_restart_segments(segments)
        out_sigmas = list(base_sigmas.cpu().numpy())

        for N_rest, K, t_min, t_max in parsed_segments:
            restart_curve = torch.linspace(t_max, t_min, K + 1).tolist()
            insert_idx = None
            for idx, s in enumerate(out_sigmas):
                if s <= t_min:
                    insert_idx = idx
                    break
            if insert_idx is not None:
                for _ in range(max(1, N_rest - 1)):
                    for r_val in restart_curve:
                        out_sigmas.insert(insert_idx, r_val)
                        insert_idx += 1

        return (torch.tensor(out_sigmas, dtype=torch.float32),)


class RestartSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": ("SAMPLER",),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"
    FUNCTION = "get_sampler"

    def get_sampler(self, sampler):
        def _restart_sampling_wrapper(model, x, sigmas, extra_args=None, callback=None, disable=None):
            extra_args = {} if extra_args is None else extra_args
            for i in range(len(sigmas) - 1):
                curr_s = sigmas[i]
                next_s = sigmas[i + 1]
                if next_s > curr_s:
                    noise = torch.randn_like(x)
                    x = x + noise * torch.sqrt(next_s ** 2 - curr_s ** 2)
                else:
                    step_sigmas = torch.tensor([curr_s, next_s], device=x.device, dtype=sigmas.dtype)
                    x = sampler.sampler_function(
                        model, x, step_sigmas, extra_args=extra_args, callback=callback, disable=True
                    )
            return x

        return (comfy.samplers.KSAMPLER(_restart_sampling_wrapper),)


class KSamplerWithRestarts:
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
            }
        }

    RETURN_TYPES = ("LATENT",)
    CATEGORY = "sampling"
    FUNCTION = "sample"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, segments=""):
        device = comfy.model_management.get_torch_device()
        latent = latent_image["samples"].clone()

        model_sampling = model.get_model_object("model_sampling")
        base_sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps)
        if denoise < 1.0:
            base_sigmas = base_sigmas[-(int(steps * denoise) + 1):]

        sched_node = RestartScheduler()
        (sigmas,) = sched_node.get_sigmas(steps, scheduler, segments, sigmas=base_sigmas)
        sigmas = sigmas.to(device)

        sampler_inst = comfy.samplers.sampler_object(sampler_name)
        sampler_node = RestartSampler()
        (wrapped_sampler,) = sampler_node.get_sampler(sampler_inst)

        samples = comfy.sample.sample_custom(
            model, torch.randn_like(latent) if denoise == 1.0 else latent,
            cfg, positive, negative, wrapped_sampler, sigmas,
            latent_image={"samples": latent}, seed=seed, device=device
        )
        return ({"samples": samples},)


# ==============================================================================
# 6. ADAPTIVE ODE SOLVERS (redhottensors)
# ==============================================================================

try:
    import torchdiffeq
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False


class ODESolver:
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
    CATEGORY = "sampling/custom_sampling/samplers"
    FUNCTION = "get_sampler"

    def get_sampler(self, solver, log_relative_tolerance, log_absolute_tolerance, max_steps):
        if not HAS_TORCHDIFFEQ:
            raise ImportError(
                "[RBG Extras] torchdiffeq is required for the ODE Solver node. "
                "Please run: pip install torchdiffeq in your ComfyUI python environment."
            )

        rtol = 10.0 ** log_relative_tolerance
        atol = 10.0 ** log_absolute_tolerance

        def _ode_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None):
            extra_args = {} if extra_args is None else extra_args

            def ode_func(t, y):
                sigma = t.expand(y.shape[0])
                denoised = model(y, sigma, **extra_args)
                d = (y - denoised) / t
                return d

            t_start = sigmas[0]
            t_end = sigmas[-1] if sigmas[-1] > 0 else torch.tensor(1e-4, device=x.device)
            t_span = torch.tensor([t_start, t_end], device=x.device, dtype=torch.float32)

            sol = torchdiffeq.odeint(
                ode_func, x, t_span, method=solver, rtol=rtol, atol=atol,
                options={"max_num_steps": max_steps}
            )
            return sol[-1]

        return (comfy.samplers.KSAMPLER(_ode_sampler),)


# ==============================================================================
# NODE REGISTRATION DICTIONARIES
# ==============================================================================

NODE_CLASS_MAPPINGS = {
    # 1. FlowMatch Euler Discrete
    "FlowMatchEulerDiscreteScheduler": FlowMatchEulerDiscreteScheduler,
    # 2. DPM++ 2M Sharp
    "SamplerDPMPP_2M_Sharp": SamplerDPMPP_2M_Sharp,
    # 3. Sigmoid Offset
    "SigmoidOffsetScheduler": SigmoidOffsetScheduler,
    # 4. TCD
    "TCDScheduler": TCDScheduler,
    "SamplerTCDEulerA": SamplerTCDEulerA,
    "SamplerTCD": SamplerTCD,
    # 5. Restart Sampling
    "RestartScheduler": RestartScheduler,
    "RestartSampler": RestartSampler,
    "KSamplerWithRestarts": KSamplerWithRestarts,
    # 6. ODE Solver
    "ODESolver": ODESolver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlowMatchEulerDiscreteScheduler": "FlowMatch Euler Discrete Scheduler (Custom)",
    "SamplerDPMPP_2M_Sharp": "Sampler DPM++ 2M Sharp",
    "SigmoidOffsetScheduler": "Sigmoid Offset Scheduler",
    "TCDScheduler": "TCDScheduler",
    "SamplerTCDEulerA": "SamplerTCD Euler A",
    "SamplerTCD": "SamplerTCD",
    "RestartScheduler": "RestartScheduler",
    "RestartSampler": "RestartSampler",
    "KSamplerWithRestarts": "KSampler With Restarts",
    "ODESolver": "ODE Solver",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]