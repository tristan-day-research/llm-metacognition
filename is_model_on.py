# In a Python shell or add to your script:
import torch
import gc

# Check if CUDA memory is being used (indicates model might still be loaded)
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")