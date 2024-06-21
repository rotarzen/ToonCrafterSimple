import time
import random
import importlib
from functools import wraps
from contextlib import contextmanager

import numpy as np
import torch
from safetensors import safe_open


###
### Timers ###
###

def timed(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        start_time = time.time()
        result = f(*args, **kwargs)
        print(f"{f.__module__}.{f.__name__} took {time.time() - start_time:.3f} seconds")
        return result
    return wrapped

@contextmanager
def timeit(s: str | None = None):
    t = [time.time()]
    yield t
    t[0] = time.time() - t[0]
    if s is not None:
        print(f'{s}: {t[0]:.3f}s')


###
### Pytorch Lightning replacements ###
###

# https://github.com/Lightning-AI/pytorch-lightning/blob/a99a6d3af1e9b8090d892dfc24b4f616853a8a40/src/lightning/fabric/utilities/seed.py#L19
def seed_everything(seed: int):
    assert isinstance(seed, int)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class ModuleWithDevice(torch.nn.Module):
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
    @property
    def device(self):
        return next(iter(self.parameters())).device


###
### Fast Safetensors Initializers ###
###

# torch.device("meta") cannot be used due to various codepaths dependent on buffers existing.
class DoNotInit(torch.overrides.TorchFunctionMode):
    """
    Internally, most torch.nn.Module.__init__ implementations do the following:
        * create weights as torch.empty(...) [free action]
        * self.reset_parameters() -> torch.nn.init.ABC(...) [costly]
    Stopping the 2nd action makes initialisation roughly costless.
    """
    def __torch_function__(self, f, _, a, k={}):
        if getattr(f, "__module__", None) == 'torch.nn.init':
            t = k.get("tensor", None)
            return a[0] if t is None else t
        return f(*a,**k)

def load_sf_ckpt(model: torch.nn.Module, ckpt: str, remap: None):
    with safe_open(ckpt, framework="pt", device="cuda") as f: # pyright: ignore
        d = {k:f.get_tensor(k) for k in f.keys()}
        if remap is not None: d = remap(d)
        model.load_state_dict(d, assign=True)


###
### https://github.com/ToonCrafter/ToonCrafter/blob/main/utils/utils.py
###

def instantiate_from_config(config: dict) -> torch.nn.Module | None:
    if not "target" in config:
        if config in ['__is_first_stage__', "__is_unconditional__"]:
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string: str, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


###
### DEBUGGING ###
###

@contextmanager
def show_global_changes():
    before = {k:v for k,v in globals().items()}
    yield # e.g. from lvdm.modules.attention_svd import *
    after = globals()
    added = after.keys() - before.keys()
    overwritten = {k for k,v in before.items() if v != after[k]}
    print('added keys:', added)
    if overwritten:
        print("!!! OVERWRITTEN keys:", overwritten)

@contextmanager
def capture_forward_outputs():
    outputs = []
    from torch.nn.modules.module import register_module_forward_hook
    def hook(module, input, output): outputs.append((module,input,output))
    handle = register_module_forward_hook(hook)
    try: yield outputs
    finally: handle.remove()