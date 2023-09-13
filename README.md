# Diffuse-diffusion
An adaption to stable diffusion XL to allow for gradual diffsuion of an image allowing for modifications to be made 

## Requirements
you can use anaconda or just install packages yourself with pip
### Anaconda
Please use anaconda3 to install dependencies 

from the base directory run the following command dependent on your operating system

#### linux
conda create --name Diffuse-diffusion --file requirements_linux.txt

#### windows
conda create --name Diffuse-diffusion --file requirements_windows.txt

### Pip

Please install the following packages using pip

pip install transformers <br>
pip install accelerate <br>
pip install safetensors <br>
pip install opencv-python <br>
pip install tqdm <br>
pip install diffusers <br>
pip install xformers <br>

## Example use 

Run the following command from the base directory to test the software

make sure to include the whole file path in "SET_SAVE_FOLDER_HERE"

python Main.py --output_folder_name example_tree_burning --save_location SET_SAVE_FOLDER_HERE --seed_file Forest_Clean.jpg --positive_prompt "A Forest on fire, burning ravaged by wildfire" --negative_prompt "green, lush, trees"

## Low VRAM parameters
--med_VRAM MED_VRAM   Set if GPU VRAM is lower than 12GB <br>
--low_VRAM LOW_VRAM   Set if GPU VRAM is lower than 8GB <br>
