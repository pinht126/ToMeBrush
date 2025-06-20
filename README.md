# Token Merging for BrushNet

BrushNet is a diffusion-based text-guided image inpainting model designed with reference to the ControlNet architecture. The inpainting model generates new content within the masked regions based on the text input, while preserving the non-masked regions to remain identical to the input image.

In practice, the model first generates the entire image such that the non-masked regions closely resemble the input, and then overlays the inpainted result on top of the original image using the mask(blending operation). 

![image](https://github.com/user-attachments/assets/dd32e198-5ef0-4c9a-b28e-dcae9a7969c2)


# This raises the question: is generating the non-masked regions necessary?
While it may seem redundant, the generation of the non-masked regions is not meaningless. Changes in these areas can affect the overall composition and coherence of the final image. However, their importance is not as critical as that of the masked regions.

Therefore, we aim to reduce the computational overhead for the non-masked regions using Token Merging for Stable Diffusion. Token Merging, which merges 50% of tokens, has been shown to reduce computation without sacrificing generation quality. By applying a higher merging ratio to the non-mask regions, we can significantly reduce the overall computational load.

Based on this idea, we apply a region-aware token merging strategy to the inpainting model, assigning different merging ratios to masked and non-masked areas.

![image](https://github.com/user-attachments/assets/f1f27199-6aa9-49d1-99ef-f0b03dbb3d4f)




## 🚀 Getting Started

### Environment Requirement 🌍

The environment of ToMeBrush is largely similar to that of BrushNet.

BrushNet has been implemented and tested on Pytorch 1.12.1 with python 3.9.

Clone the repo:

```
git clone https://github.com/TencentARC/BrushNet.git
```

We recommend you first use `conda` to create virtual environment, and install `pytorch` following [official instructions](https://pytorch.org/). For example:


```
conda create -n diffusers python=3.9 -y
conda activate diffusers
python -m pip install --upgrade pip
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

Then, you can install diffusers (implemented in this repo) with:

```
pip install -e .
```

After that, you can install required packages thourgh:

```
cd examples/brushnet/
pip install -r requirements.txt
pip install tomesd
```

### Data Download ⬇️


**Dataset**

We use the same dataset as BrushNet.

You can download the BrushData and BrushBench [here](https://forms.gle/9TgMZ8tm49UYsZ9s5) (as well as the EditBench we re-processed), which are used for training and testing the BrushNet. By downloading the data, you are agreeing to the terms and conditions of the license. The data structure should be like:

```
|-- data
    |-- BrushData
        |-- 00200.tar
        |-- 00201.tar
        |-- ...
    |-- BrushDench
        |-- images
        |-- mapping_file.json
    |-- EditBench
        |-- images
        |-- mapping_file.json
```


Noted: *We only provide a part of the BrushData in google drive due to the space limit. [random123123](https://huggingface.co/random123123) has helped upload a full dataset on hugging face [here](https://huggingface.co/datasets/random123123/BrushData). Thank for his help!*


**Checkpoints**

Checkpoints of BrushNet can be downloaded from [here](https://drive.google.com/drive/folders/1fqmS1CEOvXCxNWFrsSYd_jHYXxrydh1n?usp=drive_link). The ckpt folder contains 

- BrushNet pretrained checkpoints for Stable Diffusion v1.5 (`segmentation_mask_brushnet_ckpt` and `random_mask_brushnet_ckpt`)
- pretrinaed Stable Diffusion v1.5 checkpoint (e.g., realisticVisionV60B1_v51VAE from [Civitai](https://civitai.com/)). You can use `scripts/convert_original_stable_diffusion_to_diffusers.py` to process other models downloaded from Civitai. 
- BrushNet pretrained checkpoints for Stable Diffusion XL (`segmentation_mask_brushnet_ckpt_sdxl_v1` and `random_mask_brushnet_ckpt_sdxl_v0`).  A better version will be shortly released by [yuanhang](https://github.com/yuanhangio). Please stay tuned!
- pretrinaed Stable Diffusion XL checkpoint (e.g., juggernautXL_juggernautX from [Civitai](https://civitai.com/)). You can use `StableDiffusionXLPipeline.from_single_file("path of safetensors").save_pretrained("path to save",safe_serialization=False)` to process other models downloaded from Civitai. 



The data structure should be like:


```
|-- data
    |-- BrushData
    |-- BrushDench
    |-- EditBench
    |-- ckpt
        |-- realisticVisionV60B1_v51VAE
            |-- model_index.json
            |-- vae
            |-- ...
        |-- segmentation_mask_brushnet_ckpt
        |-- segmentation_mask_brushnet_ckpt_sdxl_v0
        |-- random_mask_brushnet_ckpt
        |-- random_mask_brushnet_ckpt_sdxl_v0
        |-- ...
```

The checkpoint in `segmentation_mask_brushnet_ckpt` and `segmentation_mask_brushnet_ckpt_sdxl_v0` provide checkpoints trained on BrushData, which has segmentation prior (mask are with the same shape of objects). The `random_mask_brushnet_ckpt` and `random_mask_brushnet_ckpt_sdxl` provide a more general ckpt for random mask shape.

## 🏃🏼 Running Scripts

### Inference 📜

You can inference with the script:

```
# sd v1.5
python examples/brushnet/test_brushnet.py
```

Since BrushNet is trained on Laion, it can only guarantee the performance on general scenarios. We recommend you train on your own data (e.g., product exhibition, virtual try-on) if you have high-quality industrial application requirements. We would also be appreciate if you would like to contribute your trained model!

You can also inference through gradio demo:

```
# sd v1.5
python examples/brushnet/app_brushnet.py
```


### Evaluation 📏

You can evaluate using the script:

```
python examples/brushnet/evaluate_brushnet.py \
--brushnet_ckpt_path data/ckpt/segmentation_mask_brushnet_ckpt \
--image_save_path runs/evaluation_result/BrushBench/brushnet_segmask/inside \
--mapping_file data/BrushBench/mapping_file.json \
--base_dir data/BrushBench \
--mask_key inpainting_mask
```

The `--mask_key` indicates which kind of mask to use, `inpainting_mask` for inside inpainting and `outpainting_mask` for outside inpainting. The evaluation results (images and metrics) will be saved in `--image_save_path`. 



*Noted that you need to ignore the nsfw detector in `src/diffusers/pipelines/brushnet/pipeline_brushnet.py#1261` to get the correct evaluation results. Moreover, we find different machine may generate different images, thus providing the results on our machine [here](https://drive.google.com/drive/folders/1dK3oIB2UvswlTtnIS1iHfx4s57MevWdZ?usp=sharing).*


## Result (ToMeBrush)
Comparison of results without applying the blending operation
![image](https://github.com/user-attachments/assets/cd118ee1-24ae-4a82-abe8-1cce22734259)
Our model demonstrated superior performance compared to existing models on BrushBench in all metrics except for LPIPS. Under the SDXL framework, higher resolutions exhibited better efficiency in terms of memory usage and speed.
Compared to BrushNet, ToMeBrush achieves a 5.86% improvement in inference speed, reduces the average processing time by 0.29 seconds, and lowers memory usage by 0.06 GB.

