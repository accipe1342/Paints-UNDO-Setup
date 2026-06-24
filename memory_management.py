import os
import torch
from contextlib import contextmanager


# Set PAINTS_UNDO_HIGH_VRAM=1 to keep models resident on the GPU between stages
# (faster, needs more VRAM). setup.py enables this automatically on large GPUs.
high_vram = os.environ.get('PAINTS_UNDO_HIGH_VRAM', '0') == '1'
gpu = torch.device('cuda')
cpu = torch.device('cpu')

if not torch.cuda.is_available():
    raise SystemExit(
        "[ERROR] No CUDA-capable GPU is available to PyTorch.\n"
        "        Paints-UNDO requires an NVIDIA GPU. Check your drivers and\n"
        "        that the CUDA build of PyTorch is installed."
    )

torch.zeros((1, 1)).to(gpu, torch.float32)
torch.cuda.empty_cache()

models_in_gpu = []


@contextmanager
def movable_bnb_model(m):
    if hasattr(m, 'quantization_method'):
        m.quantization_method_backup = m.quantization_method
        del m.quantization_method
    try:
        yield None
    finally:
        if hasattr(m, 'quantization_method_backup'):
            m.quantization_method = m.quantization_method_backup
            del m.quantization_method_backup
    return


def _module_on_gpu(m):
    try:
        return next(m.parameters()).device.type == 'cuda'
    except StopIteration:
        return True  # no parameters -> nothing to move


def load_models_to_gpu(models):
    global models_in_gpu

    if not isinstance(models, (tuple, list)):
        models = [models]

    models_to_remain = [m for m in set(models) if m in models_in_gpu]
    models_to_unload = [m for m in set(models_in_gpu) if m not in models_to_remain]

    if not high_vram:
        for m in models_to_unload:
            with movable_bnb_model(m):
                m.to(cpu)
            print('Unload to CPU:', m.__class__.__name__)
        models_in_gpu = models_to_remain

    # Load by actual device, not the tracking list. In high-VRAM mode the
    # startup bookkeeping can mark models as resident before they are physically
    # moved; checking the real device guarantees they end up on the GPU.
    for m in set(models):
        if not _module_on_gpu(m):
            with movable_bnb_model(m):
                m.to(gpu)
            print('Load to GPU:', m.__class__.__name__)

    models_in_gpu = list(set(models_in_gpu + list(models)))
    torch.cuda.empty_cache()
    return


def unload_all_models(extra_models=None):
    global models_in_gpu

    if extra_models is None:
        extra_models = []

    if not isinstance(extra_models, (tuple, list)):
        extra_models = [extra_models]

    models_in_gpu = list(set(models_in_gpu + extra_models))

    return load_models_to_gpu([])
