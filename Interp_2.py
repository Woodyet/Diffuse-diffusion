import argparse
import random
# Initialize the parser
parser = argparse.ArgumentParser(description="Your script description")

# Add the arguments

parser.add_argument("--seed", default=random.randint(1000,9999), type=int, help="Seed value for random operations.")
parser.add_argument("--negative_prompt", default="", type=str, help="What you want to remove from the image")
parser.add_argument("--num_of_images", default=40, type=int, help="Number of images to produce.")
parser.add_argument("--minstrength", default=0.20, type=float, help="This indicates the minimum precentage of pixels that will change at frame 1 (value must be between 0.01 and 0.99)")
parser.add_argument("--maxstrength", default=0.90, type=float, help="This indicates the maximum precentage of pixels that will change at frame 1 (value must be between 0.01 and 0.99)")
parser.add_argument("--fps", default=2, type=int, help="Frames per second.")
parser.add_argument("--guidance_scale", default=5.5, type=float, help="How much to deviate from the original image lower = less deviation higher = more")

requiredNamed = parser.add_argument_group('required named arguments')

requiredNamed.add_argument("--output_folder_name", type=str, help="Name of the output folder.", required=True)
requiredNamed.add_argument("--save_location", type=str, help="Location to save the files.", required=True)
requiredNamed.add_argument("--seed_file", type=str, help="Full path of the initial image", required=True)
requiredNamed.add_argument("--positive_prompt", type=str, help="What you want the image to turn into", required=True)

# Parse the arguments
args = parser.parse_args()

# Assign the variables using the arguments
output_folder_name = args.output_folder_name
save_location = args.save_location
seed = args.seed
filename = args.seed_file
prompt = args.positive_prompt
negative_prompt = args.negative_prompt
num_of_images = args.num_of_images
minstrength = args.minstrength
maxstrength = args.maxstrength
fps = args.fps
guidance_scale = args.guidance_scale


import numpy as np

from tqdm import tqdm
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import logging
from diffusers.utils import load_image
import cv2
import os


#variable manipulation
guidance_scales = []
guidance_scales.append(guidance_scale)
save_prefix = save_location+output_folder_name+"/"
init_image = load_image(filename).convert("RGB")
video_name = save_prefix + output_folder_name +'.avi'
num_of_images-=1

try:
  os.mkdir(save_prefix)
except:
  print("Output directory already exists")
  import sys
  sys.exit()

def nonlinspace(xmin, xmax, n=50, power=2):
    '''Intervall from xmin to xmax with n points, the higher the power, the more dense towards the ends'''
    xm = (xmax - xmin) / 2
    x = np.linspace(-xm**power, xm**power, n)
    return np.sign(x)*abs(x)**(1/power) + xm + xmin

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=False
)
pipe = pipe.to("cuda")

strengths = np.arange(minstrength, maxstrength, (maxstrength-minstrength)/num_of_images, dtype=float)

#strengths = nonlinspace(minstrength, maxstrength, n=num_of_images, power=3)

# adds frame at start with minimal distortion
strengths = np.insert(strengths,0,0.05)

img_num = 1

pipe.set_progress_bar_config(disable=True)

for strength in tqdm(strengths):
  for guidance_scale in guidance_scales:
    generator=torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(prompt,negative_prompt=negative_prompt, image=init_image, generator=generator,strength=strength, guidance_scale=guidance_scale).images[0]
    im1 = image.save(save_prefix+"Output"+str(img_num).zfill(5)+".jpg")
    img_num+=1
    
image_folder = save_prefix
list_of_imgs = sorted(os.listdir(image_folder))
list_of_imgs_rev = sorted(os.listdir(image_folder))
list_of_imgs_rev.reverse()

images = [img for img in list_of_imgs+list_of_imgs_rev if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, fps, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()