"""Portable checkpoint inspection and compatibility checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class CheckpointReport:
    """A serializable summary of a TADiSR checkpoint."""

    path: str
    variant: str
    bytes: int
    groups: dict[str, int]
    valid: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_REQUIRED_GROUPS = {
    "cogview4": ("state_dict_transformer", "state_dict_vae", "js_decoder"),
    "kolors": ("state_dict_unet", "state_dict_vae"),
}


def _load(path: Path) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required to inspect TADiSR checkpoints. Install it before running this command."
        ) from error

    # `weights_only` prevents arbitrary pickled code from executing. It is supported
    # in modern PyTorch; the fallback keeps compatibility with older releases.
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, Mapping):
        raise ValueError(f"Expected a mapping checkpoint, received {type(state).__name__}.")
    return state


def detect_variant(state: Mapping[str, Any]) -> str:
    keys = set(state)
    if set(_REQUIRED_GROUPS["cogview4"]).issubset(keys):
        return "cogview4"
    if set(_REQUIRED_GROUPS["kolors"]).issubset(keys):
        return "kolors"
    return "unknown"


def inspect_checkpoint(path: str | Path, variant: str = "auto") -> CheckpointReport:
    """Load a checkpoint on CPU and validate the public TADiSR state-dict contract."""
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = _load(checkpoint_path)
    detected = detect_variant(state)
    selected = detected if variant == "auto" else variant
    if selected not in _REQUIRED_GROUPS:
        raise ValueError(f"Unsupported variant: {selected}. Choose auto, cogview4, or kolors.")

    errors: list[str] = []
    if variant != "auto" and detected != variant:
        errors.append(f"Requested {variant}, but checkpoint layout is {detected}.")

    groups: dict[str, int] = {}
    for key in _REQUIRED_GROUPS[selected]:
        value = state.get(key)
        if not isinstance(value, Mapping):
            errors.append(f"Missing or invalid state-dict group: {key}.")
            groups[key] = 0
            continue
        groups[key] = len(value)
        if not value:
            errors.append(f"State-dict group is empty: {key}.")

    if selected == "cogview4":
        transformer = state.get("state_dict_transformer", {})
        if isinstance(transformer, Mapping) and not any("lora" in name for name in transformer):
            errors.append("CogView4 transformer group contains no LoRA weights.")
        decoder = state.get("js_decoder", {})
        if isinstance(decoder, Mapping) and not any(name.startswith("out_block") for name in decoder):
            errors.append("Joint segmentation decoder does not contain its output head.")
    elif selected == "kolors":
        unet = state.get("state_dict_unet", {})
        if isinstance(unet, Mapping) and not any("lora" in name for name in unet):
            errors.append("Kolors U-Net group contains no LoRA weights.")
        decoder_key = "js_decoder" if "js_decoder" in state else "mi_decoder"
        decoder = state.get(decoder_key)
        if not isinstance(decoder, Mapping) or not decoder:
            errors.append("Kolors checkpoint is missing its joint segmentation decoder.")
            groups[decoder_key] = 0
        else:
            groups[decoder_key] = len(decoder)

    return CheckpointReport(
        path=str(checkpoint_path),
        variant=selected,
        bytes=checkpoint_path.stat().st_size,
        groups=groups,
        valid=not errors,
        errors=tuple(errors),
    )


def strict_load_joint_decoder(path: str | Path, variant: str = "auto") -> None:
    """Construct the released joint decoder and require an exact state-dict match.

    This checks the adapter-specific architecture without downloading a 6B base
    model. It intentionally runs on CPU and raises on any missing or unexpected
    parameter.
    """
    checkpoint_path = Path(path).expanduser().resolve()
    state = _load(checkpoint_path)
    detected = detect_variant(state)
    selected = detected if variant == "auto" else variant
    if selected not in _REQUIRED_GROUPS or selected != detected:
        raise ValueError(f"Checkpoint layout {detected!r} does not match requested variant {selected!r}.")

    decoder_key = "js_decoder" if selected == "cogview4" else (
        "js_decoder" if "js_decoder" in state else "mi_decoder"
    )
    decoder_state = state.get(decoder_key)
    if not isinstance(decoder_state, Mapping):
        raise ValueError(f"Missing joint decoder state dict: {decoder_key}.")

    if selected == "cogview4":
        from tadisr.pipelines import JointSegmentationDecoders

        decoder = JointSegmentationDecoders()
    else:
        from tadisr.kolors_decoder import MaskInteractionDecoderKolors640M

        decoder = MaskInteractionDecoderKolors640M()
    decoder.load_state_dict(decoder_state, strict=True)
