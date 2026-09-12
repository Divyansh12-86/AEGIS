import random
import uuid
from datetime import datetime, timezone

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Fix all RNG seeds (random, numpy, torch) for reproducibility (PRD §30)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_run_id() -> str:
    """Unique, sortable run identifier for experiment logging."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"
