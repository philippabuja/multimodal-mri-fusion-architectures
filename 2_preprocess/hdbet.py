import subprocess
import threading
from pathlib import Path

_lock = threading.Lock()  # HD-BET crashes when two instances run concurrently


def run_hdbet(t1_file: Path, brain_file: Path, mask_file: Path, device: str = "cuda"):
    """HD-BET brain extraction on T1 -> brain image + binary mask."""
    cmd = ["hd-bet", "-i", str(t1_file), "-o", str(brain_file), "-device", device, "--save_bet_mask"]
    with _lock:
        subprocess.run(cmd, check=True)
    # --save_bet_mask writes the mask as <stem>_bet.nii.gz
    written = brain_file.with_name(brain_file.name[:-len(".nii.gz")] + "_bet.nii.gz")
    if not brain_file.exists() or not written.exists():
        raise FileNotFoundError(f"HD-BET output missing for {t1_file}")
    if mask_file != written:
        mask_file.unlink(missing_ok=True)
        written.rename(mask_file)
    return brain_file, mask_file
