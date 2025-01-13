from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler
import torch
import cv2
import numpy as np
from PIL import Image
import time
import tomesd
# Set seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Measure memory usage before generating images
def get_gpu_memory():
    return torch.cuda.memory_allocated() / 1024**3  # Convert bytes to GB

# choose the base model here
base_model_path = "data/ckpt/realisticVisionV60B1_v51VAE"
# base_model_path = "runwayml/stable-diffusion-v1-5"

# input brushnet ckpt path
brushnet_path = "data/ckpt/segmentation_mask_brushnet_ckpt"

# choose whether using blended operation
blended = False

# input source image / mask image path and the text prompt
image_path="examples/brushnet/src/example_3.jpg"
mask_path="examples/brushnet/src/example_3_mask.jpg"
caption=["A bunny toy"] * 32

# conditioning scale
brushnet_conditioning_scale=1.0
set_seed(1)
initial_memory = get_gpu_memory()
brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionBrushNetPipeline.from_pretrained(
    base_model_path, brushnet=brushnet, torch_dtype=torch.float16, low_cpu_mem_usage=False
)
tomesd.apply_patch(pipe, mask_path = mask_path, ratio1=0.5, ratio2=0.75, mask=True)
# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed or when using Torch 2.0.
# pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()

init_image = cv2.imread(image_path)[:,:,::-1]
mask_image = 1.*(cv2.imread(mask_path).sum(-1)>255)[:,:,np.newaxis]
init_image = init_image * (1-mask_image)

init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3,-1)*255).convert("RGB")

generator = torch.Generator("cuda").manual_seed(1234)
start_time= time.time()
for i in range(5):
    image = pipe(
        caption, 
        init_image, 
        mask_image, 
        num_inference_steps=50, 
        generator=generator,
        brushnet_conditioning_scale=brushnet_conditioning_scale
    ).images[0]
end_time = time.time()
final_memory = get_gpu_memory()
generation_time = end_time - start_time
memory_per_image = final_memory - initial_memory 
print(f"Memory Used (GB): {memory_per_image:.4f} GB/image")
print(f"Inference Time: {generation_time/5:.2f} seconds")
if blended:
    image_np=np.array(image)
    init_image_np=cv2.imread(image_path)[:,:,::-1]
    mask_np = 1.*(cv2.imread(mask_path).sum(-1)>255)[:,:,np.newaxis]

    # blur, you can adjust the parameters for better performance
    mask_blurred = cv2.GaussianBlur(mask_np*255, (21, 21), 0)/255
    mask_blurred = mask_blurred[:,:,np.newaxis]
    mask_np = 1-(1-mask_np) * (1-mask_blurred)

    image_pasted=init_image_np * (1-mask_np) + image_np*mask_np
    image_pasted=image_pasted.astype(image_np.dtype)
    image=Image.fromarray(image_pasted)

image.save("output.png")