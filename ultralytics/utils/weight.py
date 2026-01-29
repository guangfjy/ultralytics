from ultralytics import YOLO
from pathlib import Path

from pathlib import Path
from typing import Iterable, Tuple, Dict

from ultralytics import YOLO


# -------------------------------------------------------------------------
# Core weight transfer utilities
# -------------------------------------------------------------------------

def transfer_compatible_weights(
    src_model,
    dst_model,
    *,
    exclude_keys: Iterable[str] = None,
    verbose: bool = True,
) -> Tuple[int, int]:
    """
    Transfer all compatible parameters from src_model to dst_model.

    Only parameters with matching keys and identical shapes are copied.
    This is designed for scenarios where model structures differ partially
    (e.g. different heads, tasks, or number of classes).

    Args:
        src_model: Source YOLO model (or nn.Module wrapper).
        dst_model: Destination YOLO model (or nn.Module wrapper).
        exclude_keys (Iterable[str], optional): Substrings of parameter names
            to explicitly exclude from transfer.
        verbose (bool): Whether to print transfer statistics.

    Returns:
        (transferred, total): Number of transferred parameters and total
        number of destination parameters.
    """
    src_sd = src_model.model.state_dict()
    dst_sd = dst_model.model.state_dict()

    if exclude_keys is None:
        exclude_keys = []

    transferred = 0
    for k, v in dst_sd.items():
        if any(ex_key in k for ex_key in exclude_keys):
            continue
        if k in src_sd and src_sd[k].shape == v.shape:
            dst_sd[k] = src_sd[k]
            transferred += 1

    dst_model.model.load_state_dict(dst_sd, strict=False)

    if verbose:
        print(f"Transferred parameters: {transferred} / {len(dst_sd)}")

    return transferred, len(dst_sd)


# -------------------------------------------------------------------------
# Checkpoint-level conversion utilities
# -------------------------------------------------------------------------

def convert_model_checkpoint(
    src_ckpt: str,
    dst_yaml: str,
    save_path: str,
    *,
    exclude_keys: Iterable[str] = None,
    verbose: bool = True,
) -> str:
    """
    Build a destination model from yaml, transfer compatible weights from
    a source checkpoint, and save the result as a new checkpoint.

    Args:
        src_ckpt (str): Source checkpoint path (.pt). Can be official or custom.
        dst_yaml (str): Destination model yaml file.
        save_path (str): Output .pt path.
        exclude_keys (Iterable[str], optional): Keys to exclude from transfer.
        verbose (bool): Whether to print logs.

    Returns:
        str: Path to the saved checkpoint.
    """
    src_ckpt = Path(src_ckpt)
    dst_yaml = Path(dst_yaml)
    save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    src_model = YOLO(src_ckpt)
    dst_model = YOLO(dst_yaml)

    transfer_compatible_weights(
        src_model,
        dst_model,
        exclude_keys=exclude_keys,
        verbose=verbose,
    )

    dst_model.save(save_path)

    if verbose:
        print(f"Checkpoint saved to: {save_path}")

    return str(save_path)


# -------------------------------------------------------------------------
# Task-specific thin wrappers
# -------------------------------------------------------------------------

def convert_to_multilabel_cls_checkpoint(
    yaml_path: str,
    save_dir: str = None,
    verbose: bool = True,
) -> str:
    """
    Convert a multi-label classification yaml into a .pt checkpoint by
    reusing weights from the corresponding official single-label
    classification checkpoint.

    This function is a thin wrapper around convert_model_checkpoint().
    """
    yaml_path = Path(yaml_path).resolve()

    if save_dir is None:
        save_dir = yaml_path.parent
    else:
        save_dir = Path(save_dir).resolve()

    save_dir.mkdir(parents=True, exist_ok=True)

    # infer the official single-label .pt name
    # (remove "-multi" suffix and replace yaml with pt)
    base_name = yaml_path.stem.replace("-multi", "")
    src_ckpt = save_dir / f"{base_name}.pt"  # e.g. yolo26s-cls.pt
    dst_ckpt = save_dir / f"{base_name}-multi.pt"  # e.g. yolo26s-cls-multi.pt

    return convert_model_checkpoint(
        src_ckpt=str(src_ckpt),
        dst_yaml=str(yaml_path),
        save_path=str(dst_ckpt),
        verbose=verbose,
    )
