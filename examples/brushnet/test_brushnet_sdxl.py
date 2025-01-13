from diffusers import StableDiffusionXLBrushNetPipeline, BrushNetModel, DPMSolverMultistepScheduler, UniPCMultistepScheduler, AutoencoderKL
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
base_model_path = "data/ckpt/juggernautXL_juggernautX"
# base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"

# input brushnet ckpt path
brushnet_path = "data/ckpt/segmentation_mask_brushnet_ckpt_sdxl_v0"

# choose whether using blended operation
blended = False

# input source image / mask image path and the text prompt
image_path="examples/brushnet/src/test_image.jpg"
mask_path="examples/brushnet/src/test_mask.jpg"
caption="A cake on the table."

# conditioning scale
brushnet_conditioning_scale=1.0
set_seed(1)
initial_memory = get_gpu_memory()
brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionXLBrushNetPipeline.from_pretrained(
    base_model_path, brushnet=brushnet, torch_dtype=torch.float16, low_cpu_mem_usage=False, use_safetensors=True
)
init_image = cv2.imread(image_path)[:,:,::-1]
mask_image = 1.*(cv2.imread(mask_path).sum(-1)>255)

# resize image
h,w,_ = init_image.shape
if w<h:
    scale=2048/w
else:
    scale=2048/h
new_h=int(h*scale)
new_w=int(w*scale)

init_image=cv2.resize(init_image,(new_w,new_h))


#tomesd.apply_patch(pipe, mask_path = mask_path, ratio1=0.5, ratio2=0.75, mask=True)
# change to sdxl-vae-fp16-fix to avoid nan in VAE encoding when using fp16
pipe.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
 
# speed up diffusion process with faster scheduler and memory optimization
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed or when using Torch 2.0.
# pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()

mask_image=cv2.resize(mask_image,(new_w,new_h))[:,:,np.newaxis]
init_image = init_image * (1-mask_image)

init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3,-1)*255).convert("RGB")

generator = torch.Generator("cuda").manual_seed(4321)

start_time= time.time()
for i in range(5):
    image = pipe(
        prompt=caption, 
        image=init_image, 
        mask=mask_image, 
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
