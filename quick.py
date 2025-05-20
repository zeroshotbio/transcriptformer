import platform
import textwrap

import torch

print(
    textwrap.dedent(f"""
  • torch {torch.__version__}  (compiled with CUDA {torch.version.cuda})
  • CUDA runtime available?   {torch.cuda.is_available()}
  • #GPUs detected:           {torch.cuda.device_count()}
  • Current device name:      {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}
  • Python {platform.python_version()}
""")
)
