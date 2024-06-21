# Simplified Tooncrafter Package Structure

```
src/tooncrafter
├── att_ae.py
├── att_svd.py
├── autoencoder.py
├── diffusion.py
├── encoders.py
├── extras.py
├── i2v.py
├── model.py
├── new.yaml
├── unet.py
├── util.py
└── webui.py
```

## extras.py
Extra utilities mostly not from the original Tooncrafter repo.

Timers / Debugging / Initialization / Lightning replacements.

## diffusion.py
Slashed combination of [ddpm3d.py](https://github.com/ToonCrafter/ToonCrafter/blob/a2b50739508ff22a74a312df4ec9021415f7dac2/lvdm/models/ddpm3d.py) & [ddim.py](https://github.com/ToonCrafter/ToonCrafter/blob/a2b50739508ff22a74a312df4ec9021415f7dac2/lvdm/models/samplers/ddim.py) into a single layer that implements DDIM sampling.

This is not a clean implementation, as I do not understand diffusion that much. Suggestions from the public are welcome.

## att_ae.py
Refactor of attention layers from [ae_modules.py](https://github.com/ToonCrafter/ToonCrafter/blob/a2b50739508ff22a74a312df4ec9021415f7dac2/lvdm/modules/networks/ae_modules.py) and [attention_svd.py](https://github.com/ToonCrafter/ToonCrafter/blob/a2b50739508ff22a74a312df4ec9021415f7dac2/lvdm/modules/attention_svd.py) to a single file.

Optimizes away conv-based attention layers for ~10% gain.

## autoencoder.py
Combination of [autoencoder.py](https://github.com/ToonCrafter/ToonCrafter/blob/a2b50739508ff22a74a312df4ec9021415f7dac2/lvdm/models/autoencoder.py), [ae_modules.py](https://github.com/ToonCrafter/ToonCrafter/blob/a2b50739508ff22a74a312df4ec9021415f7dac2/lvdm/modules/networks/ae_modules.py), and [autoencoder_dualref.py](https://github.com/ToonCrafter/ToonCrafter/blob/a2b50739508ff22a74a312df4ec9021415f7dac2/lvdm/models/autoencoder_dualref.py).

Pruned of most training-related configurations && otherwise unused codepaths. Further improvements are plausible.

## unet.py
Pruned, simplified, optimized implementation of Tooncrafter's main U-net.

It is approximately 50% faster than the original implementation, even on eager mode. I believe this can be improved further, as no effort has been made to improve the SpatialTransformer/TemporalTransformer blocks.

## model.py

Main tooncrafter model implementation. It is essentially a cut-out of the [modelling components of ddpm3d](https://github.com/ToonCrafter/ToonCrafter/blob/a2b50739508ff22a74a312df4ec9021415f7dac2/lvdm/models/ddpm3d.py#L530-L531)

## i2v.py
Reimplementation of [i2v_test_application.py](https://github.com/ToonCrafter/ToonCrafter/blob/main/scripts/gradio/i2v_test_application.py).

Refactors most of the code into segregated functions for reuse, adds some types and classes for clarity.

Includes [fp16 checkpoint](https://huggingface.co/Kijai/DynamiCrafter_pruned) download and safetensors fast initialization. Remaps state-dict keys to match ToonCrafterModel.

## new.yaml
pruned yaml definition for model. included for packaging convenience.

---

## util.py
Deduplicated utilities from [common.py](https://github.com/ToonCrafter/ToonCrafter/blob/a2b50739508ff22a74a312df4ec9021415f7dac2/lvdm/common.py), [basics.py](https://github.com/ToonCrafter/ToonCrafter/blob/a2b50739508ff22a74a312df4ec9021415f7dac2/lvdm/basics.py), [utils_diffusion.py](https://github.com/ToonCrafter/ToonCrafter/blob/a2b50739508ff22a74a312df4ec9021415f7dac2/lvdm/models/utils_diffusion.py), [distributions.py](https://github.com/ToonCrafter/ToonCrafter/blob/a2b50739508ff22a74a312df4ec9021415f7dac2/lvdm/distributions.py).

## att_svd.py
Untouched copy of [attention_svd.py](https://github.com/ToonCrafter/ToonCrafter/blob/a2b50739508ff22a74a312df4ec9021415f7dac2/lvdm/modules/attention_svd.py).

I plan on optimizing this later.

## encoders.py
Minimal copy of [condition.py](https://github.com/ToonCrafter/ToonCrafter/blob/a2b50739508ff22a74a312df4ec9021415f7dac2/lvdm/modules/encoders/condition.py) and [resampler.py](https://github.com/ToonCrafter/ToonCrafter/blob/a2b50739508ff22a74a312df4ec9021415f7dac2/lvdm/modules/encoders/resampler.py).

## webui.py
Copy of [gradio_app.py](https://github.com/ToonCrafter/ToonCrafter/blob/main/gradio_app.py)
