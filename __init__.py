"""
ComfyUI-RBG-Sampler-Scheduler-Extras
Consolidated high-precision samplers & schedulers suite:
- FlowMatch Euler Discrete Scheduler (erosDiffusion)
- DPM++ 2M Sharp Sampler (envy-ai)
- Sigmoid Offset Scheduler (silveroxides)
- Restart Sampling Suite (ssitu)
- TCD Scheduler & Sampler (dfl)
- Adaptive ODE Solvers (redhottensors)
"""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 1. FlowMatch Euler Discrete Scheduler
try:
    from .euler_discrete import NODE_CLASS_MAPPINGS as euler_nodes
    from .euler_discrete import NODE_DISPLAY_NAME_MAPPINGS as euler_display
    NODE_CLASS_MAPPINGS.update(euler_nodes)
    NODE_DISPLAY_NAME_MAPPINGS.update(euler_display)
except Exception as e:
    print(f"[RBG Extras] Error loading Euler Discrete Scheduler: {e}")

# 2. DPM++ 2M Sharp Sampler
try:
    from .dpmpp_sharp import NODE_CLASS_MAPPINGS as dpmpp_nodes
    from .dpmpp_sharp import NODE_DISPLAY_NAME_MAPPINGS as dpmpp_display
    NODE_CLASS_MAPPINGS.update(dpmpp_nodes)
    NODE_DISPLAY_NAME_MAPPINGS.update(dpmpp_display)
except Exception as e:
    print(f"[RBG Extras] Error loading DPM++ 2M Sharp: {e}")

# 3. Sigmoid Offset Scheduler
try:
    from .sigmoid_offset import NODE_CLASS_MAPPINGS as sigmoid_nodes
    from .sigmoid_offset import NODE_DISPLAY_NAME_MAPPINGS as sigmoid_display
    NODE_CLASS_MAPPINGS.update(sigmoid_nodes)
    NODE_DISPLAY_NAME_MAPPINGS.update(sigmoid_display)
except Exception as e:
    print(f"[RBG Extras] Error loading Sigmoid Offset Scheduler: {e}")

# 4. Restart Sampling Suite
try:
    from .restart_sampling import NODE_CLASS_MAPPINGS as restart_nodes
    from .restart_sampling import NODE_DISPLAY_NAME_MAPPINGS as restart_display
    NODE_CLASS_MAPPINGS.update(restart_nodes)
    NODE_DISPLAY_NAME_MAPPINGS.update(restart_display)
except Exception as e:
    print(f"[RBG Extras] Error loading Restart Sampling: {e}")

# 5. TCD Scheduler & Sampler
try:
    from .tcd_scheduler import NODE_CLASS_MAPPINGS as tcd_nodes
    from .tcd_scheduler import NODE_DISPLAY_NAME_MAPPINGS as tcd_display
    NODE_CLASS_MAPPINGS.update(tcd_nodes)
    NODE_DISPLAY_NAME_MAPPINGS.update(tcd_display)
except Exception as e:
    print(f"[RBG Extras] Error loading TCD Scheduler: {e}")

# 6. Adaptive ODE Solvers
try:
    from .ode_solver import NODE_CLASS_MAPPINGS as ode_nodes
    from .ode_solver import NODE_DISPLAY_NAME_MAPPINGS as ode_display
    NODE_CLASS_MAPPINGS.update(ode_nodes)
    NODE_DISPLAY_NAME_MAPPINGS.update(ode_display)
except Exception as e:
    print(f"[RBG Extras] Error loading ODE Solvers: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]