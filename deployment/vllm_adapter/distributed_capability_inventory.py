"""D3B Part A: real vLLM installation and hardware capability discovery.

Everything here is produced by introspecting the actually-installed vLLM
package, the actually-installed torch build, and the actual host GPU state at
call time. Nothing is assumed from memory or hardcoded from prior vLLM
versions -- CLI flags in particular come from vLLM's own argparse registry
(``vllm.entrypoints.openai.cli_args.make_arg_parser``), not from reading
documentation or guessing.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

TRUTH_BOUNDARY = (
    "Capability discovery reflects the actual installed vLLM package, torch "
    "build, and host GPU state observed at discovery time on this "
    "development host. It is not a claim about any other host or a future "
    "install."
)


@dataclass(frozen=True)
class GPUDeviceInfo:
    index: int
    name: str
    total_memory_mb: float
    compute_capability_major: int
    compute_capability_minor: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "total_memory_mb": self.total_memory_mb,
            "compute_capability": f"{self.compute_capability_major}.{self.compute_capability_minor}",
        }


@dataclass(frozen=True)
class EnvironmentInventory:
    python_version: str
    torch_version: str | None
    vllm_version: str | None
    vllm_package_path: str | None
    transformers_version: str | None
    cuda_available: bool
    cuda_version: str | None
    bf16_supported: bool | None
    visible_gpu_count: int
    gpus: tuple[GPUDeviceInfo, ...]
    platform: str
    nvidia_smi_present: bool
    distributed_backend_availability: dict[str, bool]
    truth_boundary: str = TRUTH_BOUNDARY

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "vllm_version": self.vllm_version,
            "vllm_package_path": self.vllm_package_path,
            "transformers_version": self.transformers_version,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "bf16_supported": self.bf16_supported,
            "visible_gpu_count": self.visible_gpu_count,
            "gpus": [g.to_dict() for g in self.gpus],
            "platform": self.platform,
            "nvidia_smi_present": self.nvidia_smi_present,
            "distributed_backend_availability": self.distributed_backend_availability,
            "truth_boundary": self.truth_boundary,
        }


def discover_environment() -> EnvironmentInventory:
    """Probe the actual live host/install -- no assumptions, no memory."""
    torch_version: str | None = None
    cuda_available = False
    cuda_version: str | None = None
    bf16_supported: bool | None = None
    visible_gpu_count = 0
    gpus: list[GPUDeviceInfo] = []
    dist_backend_availability: dict[str, bool] = {}

    try:
        import torch  # noqa: PLC0415

        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
        cuda_version = torch.version.cuda
        if cuda_available:
            visible_gpu_count = torch.cuda.device_count()
            bf16_supported = bool(torch.cuda.is_bf16_supported())
            for i in range(visible_gpu_count):
                props = torch.cuda.get_device_properties(i)
                gpus.append(
                    GPUDeviceInfo(
                        index=i,
                        name=props.name,
                        total_memory_mb=round(props.total_memory / (1024 * 1024), 2),
                        compute_capability_major=props.major,
                        compute_capability_minor=props.minor,
                    )
                )
        dist_backend_availability["nccl"] = bool(
            torch.distributed.is_available() and torch.distributed.is_nccl_available()
        )
        dist_backend_availability["gloo"] = bool(
            torch.distributed.is_available() and torch.distributed.is_gloo_available()
        )
    except ImportError:
        pass

    vllm_version: str | None = None
    vllm_package_path: str | None = None
    try:
        import vllm  # noqa: PLC0415

        vllm_version = vllm.__version__
        vllm_package_path = vllm.__file__
    except ImportError:
        pass

    transformers_version: str | None = None
    try:
        import transformers  # noqa: PLC0415

        transformers_version = transformers.__version__
    except ImportError:
        pass

    try:
        subprocess.run(["nvidia-smi", "-L"], capture_output=True, check=True, timeout=10)
        nvidia_smi_present = True
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        nvidia_smi_present = False

    dist_backend_availability["ray"] = _module_importable("ray")
    dist_backend_availability["multiprocessing"] = True  # stdlib, always present

    return EnvironmentInventory(
        python_version=sys.version,
        torch_version=torch_version,
        vllm_version=vllm_version,
        vllm_package_path=vllm_package_path,
        transformers_version=transformers_version,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        bf16_supported=bf16_supported,
        visible_gpu_count=visible_gpu_count,
        gpus=tuple(gpus),
        platform=platform.platform(),
        nvidia_smi_present=nvidia_smi_present,
        distributed_backend_availability=dist_backend_availability,
    )


def _module_importable(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


@dataclass(frozen=True)
class ArgumentSpec:
    dest: str
    option_strings: tuple[str, ...]
    type_name: str | None
    default: Any
    choices: tuple[Any, ...] | None
    choices_from_metavar: str | None
    is_boolean_flag: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "dest": self.dest,
            "option_strings": list(self.option_strings),
            "type_name": self.type_name,
            "default": self.default if _is_json_safe(self.default) else repr(self.default),
            "choices": list(self.choices) if self.choices is not None else None,
            "choices_from_metavar": self.choices_from_metavar,
            "is_boolean_flag": self.is_boolean_flag,
        }


def _is_json_safe(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool, list, tuple))


REGISTRY_TRUTH_BOUNDARY = (
    "Argument registry is introspected directly from the installed vLLM "
    "package's own argparse.ArgumentParser (vllm.entrypoints.openai.cli_args."
    "make_arg_parser), not read from documentation, --help text scraping, or "
    "prior knowledge of vLLM's CLI."
)


def discover_argument_registry() -> dict[str, Any]:
    """Introspect the installed vLLM OpenAI-server argparse registry directly."""
    from vllm.entrypoints.openai.cli_args import make_arg_parser  # noqa: PLC0415
    from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: PLC0415
    import vllm  # noqa: PLC0415

    parser = make_arg_parser(FlexibleArgumentParser())
    specs: dict[str, ArgumentSpec] = {}
    for action in parser._actions:  # noqa: SLF001 -- deliberate introspection, no public API exists
        if not action.dest or action.dest == "help":
            continue
        option_strings = tuple(action.option_strings)
        is_boolean_flag = any(s.startswith("--no-") for s in option_strings) or (
            action.const is True and action.default is not None and not option_strings
        )
        choices = tuple(action.choices) if action.choices else None
        choices_from_metavar = None
        if choices is None and isinstance(action.metavar, list):
            choices_from_metavar = repr(action.metavar)
        elif choices is None and isinstance(action.metavar, str) and action.metavar.startswith("["):
            choices_from_metavar = action.metavar
        type_name = getattr(action.type, "__name__", None) if action.type else None
        specs[action.dest] = ArgumentSpec(
            dest=action.dest,
            option_strings=option_strings,
            type_name=type_name,
            default=action.default,
            choices=choices,
            choices_from_metavar=choices_from_metavar,
            is_boolean_flag=is_boolean_flag,
        )

    return {
        "vllm_version": vllm.__version__,
        "discovery_method": "argparse_introspection_of_installed_package",
        "entry_point": "vllm.entrypoints.openai.api_server (vllm.entrypoints.openai.cli_args.make_arg_parser)",
        "total_arguments_discovered": len(specs),
        "arguments": {dest: spec.to_dict() for dest, spec in sorted(specs.items())},
        "truth_boundary": REGISTRY_TRUTH_BOUNDARY,
    }


# Fields the D3B distributed launch spec actually needs to resolve. Kept
# explicit so materializer code fails loudly (KeyError) rather than silently
# proceeding if the installed vLLM version renames or removes one of these.
REQUIRED_ARGUMENT_DESTS: tuple[str, ...] = (
    "model",
    "tokenizer",
    "trust_remote_code",
    "dtype",
    "seed",
    "revision",
    "served_model_name",
    "host",
    "port",
    "master_addr",
    "master_port",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "data_parallel_size",
    "distributed_executor_backend",
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "gpu_memory_utilization",
    "enable_prefix_caching",
    "enable_chunked_prefill",
)
