import torch
from diffusers import StableDiffusionPipeline

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype = dtype)
pipe = pipe.to(device)
print(pipe)

import torchvision, time
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

prompt = "A snowy winter village scene with quaint wooden houses covered in snow, smoke rising from chimneys, and a frozen lake in the foreground"

def callback(pipe, step, timestep, data):
  if step % 5 == 0:
    latents = data['latents']
    latents = 1 / 0.18215 * latents
    image = pipe.vae.decode(latents).sample
    image = (image / 2 + 0.5).clip(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float()
    clear_output(wait = True)
    plt.figure(); plt.title("Iteration %d" % step);
    plt.imshow(image[0]);plt.axis('off'); plt.show();
  return data

images = pipe(prompt = prompt,
              num_inference_steps = 50,
              generator = torch.Generator(device).manual_seed(123),
              guidance_scale = 7.5,
              callback_on_step_end = callback).images

clear_output(wait = True)
plt.figure();
plt.imshow(images[0]);plt.axis('off'); plt.show();
images[0].save("Winter Village.jpg")

prompts = {'restaurant': 'An elegant restaurant interior with dim lighting, wood accents, and guests enjoying a fine dining experience',
           'school': "A serene view of an empty schoolyard with modern educational buildings, green lawns, and a clear blue sky",
           'shopping mall': "A spacious and modern shopping mall interior with high ceilings, elegant storefronts, and gleaming floors, illuminated by natural light from skylights above",
           'bookstore':'A cozy bookstore interior filled with tall wooden shelves, an array of books in organized rows, and soft ambient lighting creating a warm, inviting atmosphere',
           'bedroom': 'A serene and stylish bedroom featuring a plush bed with soft linens, elegant decor, large windows with sheer curtains, and a peaceful ambiance highlighted by warm, ambient lighting',
           'casino': 'A lavish casino interior showcasing vibrant slot machines, grandiose gambling tables, and opulent decor under dazzling overhead lights',
           'classroom': 'A modern classroom interior with rows of clean desks, a large blackboard at the front, and educational posters on the walls, bathed in natural light from large windows',
           'amusement park': 'A vibrant amusement park scene showcasing colorful roller coasters, bustling game booths, and sprawling festive lights under a clear blue sky',
           'airport': 'A spacious airport terminal with sleek modern architecture, expansive glass windows revealing parked airplanes',
           'train station': 'A bustling train station featuring multiple platforms, trains ready for departure',
           'hosiptal': 'A modern hospital interior with bright, clean hallways, informative signage, and a reception area equipped with comfortable seating and ambient lighting',
           'park': 'A serene park landscape featuring winding paths, scattered empty benches, a tranquil pond, and tall, mature trees casting soft shadows on lush green grass',
           'library': "A quiet library with rows of towering bookshelves filled with books, comfortable reading nooks, and a grand, ornate ceiling with soft, diffuse lighting",
           'beach': "A pristine beach scene with clear blue waters, white sandy shores, and a tranquil horizon under a bright sunny sky, devoid of any visitors",
           'mountain view': "A breathtaking mountain landscape showcasing snow-capped peaks, a clear blue sky, and wildflowers blooming along a rocky terrain",
           'City Skyline at Night': "A panoramic view of a city skyline at night, illuminated by the vibrant lights of skyscrapers, reflecting off a calm river",
           'Desert': "A vast desert landscape under a scorching sun, featuring undulating sand dunes and sparse vegetation, extending towards a distant mountain range",
           'Forest': "A dense forest background with towering trees, a carpet of ferns and fallen leaves, and beams of sunlight piercing through the canopy",
           'Office': "A modern office space with ergonomic desks, high-tech computers, minimalistic decorations, and large, sunlit windows overlooking a cityscape",
           'Winter Village': "A snowy winter village scene with quaint wooden houses covered in snow, smoke rising from chimneys, and a frozen lake in the foreground"}

def callback(pipe, step, timestep, data):
  if step % 5 == 0:
    latents = data['latents']
    latents = 1 / 0.18215 * latents
    image = pipe.vae.decode(latents).sample
    image = (image / 2 + 0.5).clip(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float()
    clear_output(wait = True)
    plt.figure(); plt.title("Iteration %d" % step);
    plt.imshow(image[0]);plt.axis('off'); plt.show();
  return data


for key, prompt in prompts.items():
    images = pipe(prompt=prompt,
                  num_inference_steps=50,
                  generator=torch.Generator(device).manual_seed(123),
                  guidance_scale=7.5,
                  callback_on_step_end=callback).images

    clear_output(wait=True)
    plt.figure()
    plt.imshow(images[0])
    plt.axis('off')
    plt.show()
    images[0].save(f"{key}.jpg")

"""#stable-diffusion-v1-5"""

from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
	"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
generator = torch.Generator("cuda").manual_seed(123)
image = pipeline("A snowy winter village scene with quaint wooden houses covered in snow, smoke rising from chimneys, and a frozen lake in the foreground", generator=generator).images[0]
image

"""#stable-diffusion-xl-base-1.0"""

from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
generator = torch.Generator("cuda").manual_seed(123)
image = pipeline("A snowy winter village scene with quaint wooden houses covered in snow, smoke rising from chimneys, and a frozen lake in the foreground", generator=generator).images[0]
image



"""#i2vgen-xl (text + image to video)"""

import torch
from diffusers import I2VGenXLPipeline
from diffusers.utils import export_to_gif, load_image

pipeline = I2VGenXLPipeline.from_pretrained("ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16")
pipeline.enable_model_cpu_offload()

image_url = "/content/Winter Village.jpg"
image = load_image(image_url).convert("RGB")

prompt = "A snowy winter village scene with quaint wooden houses covered in snow, smoke rising from chimneys, and a frozen lake in the foreground"
negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured"
generator = torch.manual_seed(8888)

frames = pipeline(
    prompt=prompt,
    image=image,
    num_inference_steps=50,
    negative_prompt=negative_prompt,
    guidance_scale=9.0,
    generator=generator
).frames[0]
export_to_gif(frames, "i2v.gif")

pip install Pillow

from PIL import Image, ImageSequence

# Open your GIF file
gif_path = '/content/i2v.gif'
img = Image.open(gif_path)

# Create a directory to store the frames
import os
frame_dir = 'gif_frames'
os.makedirs(frame_dir, exist_ok=True)

# Extract and save each frame
for i, frame in enumerate(ImageSequence.Iterator(img)):
    frame_path = os.path.join(frame_dir, f'frame_{i:03d}.jpg')
    frame = frame.convert('RGB')  # Convert to RGB
    frame.save(frame_path, 'JPEG')

print(f"Frames are saved in '{frame_dir}' directory.")

"""#ModelscopeT2V"""

import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

pipeline = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipeline.enable_model_cpu_offload()
pipeline.enable_vae_slicing()

prompt = "A snowy winter village scene with quaint wooden houses covered in snow, smoke rising from chimneys, and a frozen lake in the foreground"
video_frames = pipeline(prompt).frames[0]
export_to_video(video_frames, "modelscopet2v.mp4", fps=10)

"""#Combine bgr and fgr"""

import cv2
import numpy as np
from PIL import Image

# Define the paths with placeholders for frame indices
path_fgr = "/content/drive/MyDrive/comp646/dataset/VideoMatte240K_JPEG_SD/train/fgr/0316/{:05d}.jpg"
path_bgr = "/content/gif_frames/frame_{:03d}.jpg"
path_pha = "/content/drive/MyDrive/comp646/dataset/VideoMatte240K_JPEG_SD/train/pha/0316/{:05d}.jpg"

# Set up output video writer
frame_example = Image.open(path_fgr.format(0))  # Load the first frame to get video dimensions
width, height = frame_example.size
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('output_video.mp4', fourcc, 10, (width, height))

# Read and composite the first 15 frames
for i in range(15):  # Adjust the range if more frames are needed
    fgr = Image.open(path_fgr.format(i))
    bgr = Image.open(path_bgr.format(i))
    pha = Image.open(path_pha.format(i))

    # Convert images to numpy arrays
    fgr = np.array(fgr)
    bgr = np.array(bgr)
    pha = np.array(pha, dtype=np.float32) / 255.0  # Normalize and ensure pha is float

    # Resize images if necessary to match the dimensions of the first frame
    if fgr.shape != (height, width, 3):
        fgr = cv2.resize(fgr, (width, height), interpolation=cv2.INTER_AREA)
    if bgr.shape != (height, width, 3):
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    if pha.shape != (height, width, 3):
        pha = cv2.resize(pha, (width, height), interpolation=cv2.INTER_AREA)

    # Ensure the alpha channel is correctly shaped for broadcasting
    if pha.ndim == 2:  # Check if pha is a single channel image
        pha = np.repeat(pha[:, :, np.newaxis], 3, axis=2)  # Repeat the alpha across the RGB channels

    # Apply alpha blending formula to composite images
    composite = pha * fgr + (1 - pha) * bgr  # Element-wise operation for blending

    # Ensure composite is uint8 before writing
    composite = composite.astype(np.uint8)

    # Change BGR to RGB
    composite = composite[:, :, ::-1]

    # Write the composited image to the video
    video.write(composite)

# Release the video writer to finalize video
video.release()
print("Composited video created successfully.")

import cv2
import os

# Define the path for the MP4 file and the directory to store frames
mp4_path = '/content/modelscopet2v.mp4'
frame_dir = 'mp4_frames'
os.makedirs(frame_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(mp4_path)

# Initialize frame count
i = 0

# Read and save each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if there are no frames left

    # Define the frame path
    frame_path = os.path.join(frame_dir, f'frame_{i:03d}.jpg')

    # Convert frame from BGR to RGB (OpenCV uses BGR by default)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Save the frame as a JPG file
    cv2.imwrite(frame_path, frame)

    # Increment frame count
    i += 1

# Release the video capture object
cap.release()

print(f"Frames are saved in '{frame_dir}' directory.")

import cv2
import numpy as np
from PIL import Image

# Define the paths with placeholders for frame indices
path_fgr = "/content/drive/MyDrive/comp646/dataset/VideoMatte240K_JPEG_SD/train/fgr/0316/{:05d}.jpg"
path_bgr = "/content/mp4_frames/frame_{:03d}.jpg"
path_pha = "/content/drive/MyDrive/comp646/dataset/VideoMatte240K_JPEG_SD/train/pha/0316/{:05d}.jpg"

# Set up output video writer
frame_example = Image.open(path_fgr.format(0))  # Load the first frame to get video dimensions
width, height = frame_example.size
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('output_video_mp4.mp4', fourcc, 10, (width, height))

# Read and composite the first 15 frames
for i in range(15):  # Adjust the range if more frames are needed
    fgr = Image.open(path_fgr.format(i))
    bgr = Image.open(path_bgr.format(i))
    pha = Image.open(path_pha.format(i))

    # Convert images to numpy arrays
    fgr = np.array(fgr)
    bgr = np.array(bgr)
    pha = np.array(pha, dtype=np.float32) / 255.0  # Normalize and ensure pha is float

    # Resize images if necessary to match the dimensions of the first frame
    if fgr.shape != (height, width, 3):
        fgr = cv2.resize(fgr, (width, height), interpolation=cv2.INTER_AREA)
    if bgr.shape != (height, width, 3):
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    if pha.shape != (height, width, 3):
        pha = cv2.resize(pha, (width, height), interpolation=cv2.INTER_AREA)

    # Ensure the alpha channel is correctly shaped for broadcasting
    if pha.ndim == 2:  # Check if pha is a single channel image
        pha = np.repeat(pha[:, :, np.newaxis], 3, axis=2)  # Repeat the alpha across the RGB channels

    # Apply alpha blending formula to composite images
    composite = pha * fgr + (1 - pha) * bgr  # Element-wise operation for blending

    # Ensure composite is uint8 before writing
    composite = composite.astype(np.uint8)

    # Change BGR to RGB
    composite = composite[:, :, ::-1]

    # Write the composited image to the video
    video.write(composite)

# Release the video writer to finalize video
video.release()
print("Composited video created successfully.")