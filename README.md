# Simple ToonCrafter Implementation

This is a simplified implementation of [tooncrafter](https://github.com/ToonCrafter/ToonCrafter) for inference.

```bash
# Optional: create a venv for this project. https://github.com/astral-sh/uv
uv venv --seed tc-venv && . tc-venv/bin/activate
# install package
pip install 'tooncrafter @ git+https://github.com/rotarzen/ToonCrafterSimple'
# start webui
tc-webui --compile
```

You can also use it programmatically:
```python
# Initialize model
from tooncrafter.i2v import Image2Video
i2v = Image2Video("new.yaml", result_dir="/tmp/tc-results", resolution='320_512', fp16=True)
image = ... # e.g. numpy.asarray(PIL.Image.open(...))
video_output_path = i2v.get_image(image, "an anime scene", image, steps=20, fs=10)
```

# changes from original repo

* removed useless deps (imageio, pytorch_lightning, moviepy)
* removed most codepaths unused for inference, most unused architecture modifications
* improved model load/init speed (avoid cpu init), about 2x faster
* cleaned up UNetModel && reduce redundant computations, about 40% faster forward pass
* refactor most of the code in general. see [longer summary of changes here](./src/tooncrafter/README.md)

On an RTX 3090, I obtain:

| |`get_image()` end2end example speed (s)|
|-|-|
|original repo|47.13|
|simple implementation|40.031|
|compiled impl|30.985|

```bash
$ tc-webui --compile
...
tooncrafter.i2v.spawn_model took 12.594 seconds
precompiling. This will take a while...
...
done precompile.
Running on local URL:  http://127.0.0.1:7850

To create a public link, set `share=True` in `launch()`.
Executing: prompt='an anime scene', steps=50, cfg_scale=7.5, eta=1, fs=10, seed=789
...
tooncrafter.i2v.get_image took 30.502 seconds
```

Although this project was not developed with the goal of reducing vram usage, the optimizations used have also reduced vram consumption slightly (23.5GB -> 21.5GB). 

## On numerical differences

Most changes to the code produce no change in the numerical result of the model.

However, there are minor changes that arise due to
* refactoring AttnBlock to use nn.Linear instead of nn.Conv (~10% faster, diff `x∈[-6.106e-05, 6.104e-05]`). see [analysis](./att_numerics.py)

There is (currently) **no quantization, or fast sampling**, or any optimization tricks that lead to mathematically different operations in this repo.

# TODO
* [ ] learn about other improvements from the community
* [ ] integrate with more commonly used tools (e.g. [ComfyUI](https://github.com/AIGODLIKE/ComfyUI-ToonCrafter))
* [ ] improve att_svd.py
* [ ] consider if there are unobvious end-to-end redundancies
