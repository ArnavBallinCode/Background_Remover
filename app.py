"""
Background Removal Tool using BiRefNet
=====================================

This application provides an easy-to-use interface for removing backgrounds from images
using the state-of-the-art BiRefNet (Bilateral Reference Network) model. The tool offers
multiple input methods and supports both CPU and GPU acceleration.

Key Features:
- High-quality background removal using BiRefNet
- Support for uploaded images, URLs, and file processing
- GPU acceleration on CUDA and Apple Silicon (MPS)
- Interactive web interface built with Gradio
- Multiple output formats including transparent PNGs

Technical Details:
- Model: ZhengPeng7/BiRefNet from Hugging Face
- Framework: PyTorch with Transformers
- UI: Gradio with custom image sliders
- Input: PIL Images, file paths, or URLs
- Output: Transparent PNG images

Author: Arnav
Repository: https://github.com/ArnavBallinCode/Background_Remover
"""

import gradio as gr
from loadimg import load_img
import spaces
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from typing import Union, Tuple
from PIL import Image
import logging

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set PyTorch matrix multiplication precision for optimal performance
# "high" provides a good balance between speed and accuracy
torch.set_float32_matmul_precision(["high", "highest"][0])

# Load the BiRefNet model for image segmentation
# BiRefNet is a state-of-the-art model for salient object detection and background removal
# The model uses bilateral reference networks to achieve high-quality segmentation
logger.info("Loading BiRefNet model...")
birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", 
    trust_remote_code=True  # Required for custom model architecture
)

# Determine the best available device for computation
# Priority: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
if torch.backends.mps.is_available():
    device = "mps"
    logger.info("Using Apple Silicon MPS acceleration")
elif torch.cuda.is_available():
    device = "cuda"
    logger.info("Using CUDA GPU acceleration")
else:
    device = "cpu"
    logger.info("Using CPU (GPU acceleration not available)")

birefnet.to(device)

# Image preprocessing pipeline for BiRefNet model
# The model expects specific input format and normalization
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),  # Resize to model's expected input size
    transforms.ToTensor(),            # Convert PIL Image to tensor (0-1 range)
    transforms.Normalize(             # Normalize using ImageNet statistics
        [0.485, 0.456, 0.406],       # Mean values for RGB channels
        [0.229, 0.224, 0.225]        # Std values for RGB channels
    ),
])

def fn(image: Union[Image.Image, str]) -> Tuple[Image.Image, Image.Image]:
    """
    Remove the background from an image and return both the transparent version and the original.

    This function performs background removal using a BiRefNet segmentation model. It is intended for use
    with image input (either uploaded or from a URL). The function returns a transparent PNG version of the image
    with the background removed, along with the original RGB version for comparison.

    Args:
        image (PIL.Image or str): The input image, either as a PIL object or a filepath/URL string.

    Returns:
        tuple:
            - processed_image (PIL.Image): The input image with the background removed and transparency applied.
            - origin (PIL.Image): The original RGB image, unchanged.
    """
    im = load_img(image, output_type="pil")
    im = im.convert("RGB")
    origin = im.copy()
    processed_image = process(im)
    return (origin, processed_image)

@spaces.GPU
def process(image: Image.Image) -> Image.Image:
    """
    Apply BiRefNet-based image segmentation to remove the background.

    This function implements the core background removal logic:
    1. Preprocesses the input image using the transform pipeline
    2. Runs inference through the BiRefNet segmentation model
    3. Generates a binary mask for foreground/background separation
    4. Applies the mask as an alpha channel for transparency

    Args:
        image (PIL.Image): The input RGB image to process.

    Returns:
        PIL.Image: The image with background removed and transparency applied.
                  The alpha channel contains the segmentation mask.
    """
    logger.info(f"Processing image of size: {image.size}")
    image_size = image.size
    
    # Preprocess image and move to appropriate device
    input_images = transform_image(image).unsqueeze(0).to(device)
    
    # Run inference without gradient computation for efficiency
    with torch.no_grad():
        # BiRefNet returns multiple scales, we use the final one [-1]
        # Apply sigmoid to get probabilities between 0 and 1
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    
    # Extract the prediction mask and convert to PIL Image
    pred = preds[0].squeeze()  # Remove batch and channel dimensions
    pred_pil = transforms.ToPILImage()(pred)
    
    # Resize mask to match original image dimensions
    mask = pred_pil.resize(image_size)
    
    # Apply mask as alpha channel for transparency
    image.putalpha(mask)
    
    logger.info("Background removal completed")
    return image

def process_file(f: str) -> str:
    """
    Load an image file from disk, remove the background, and save the output as a transparent PNG.

    Args:
        f (str): Filepath of the image to process.

    Returns:
        str: Path to the saved PNG image with background removed.
    """
    name_path = f.rsplit(".", 1)[0] + ".png"
    im = load_img(f, output_type="pil")
    im = im.convert("RGB")
    transparent = process(im)
    transparent.save(name_path)
    return name_path

# ============================================================================
# GRADIO UI COMPONENTS
# ============================================================================

# Image slider components for before/after comparison
slider1 = gr.ImageSlider(
    label="Background Removal Result", 
    type="pil", 
    format="png"
)
slider2 = gr.ImageSlider(
    label="URL Image Processing Result", 
    type="pil", 
    format="png"
)

# Input components for different interaction modes
image_upload = gr.Image(
    label="Upload an image", 
    sources=["upload", "webcam", "clipboard"]
)
image_file_upload = gr.Image(
    label="Upload image for file output", 
    type="filepath"
)
url_input = gr.Textbox(
    label="Paste an image URL",
    placeholder="https://example.com/image.jpg"
)

# Output component for downloadable files
output_file = gr.File(
    label="Download Processed PNG", 
    file_types=[".png"]
)

# ============================================================================
# EXAMPLE CONTENT
# ============================================================================

# Load example images for demonstration
logger.info("Loading example images...")
chameleon = load_img("butterfly.jpg", output_type="pil")
url_example = "https://hips.hearstapps.com/hmg-prod/images/gettyimages-1229892983-square.jpg"

# ============================================================================
# GRADIO INTERFACE SETUP
# ============================================================================

# Tab 1: Image Upload with Before/After Slider
tab1 = gr.Interface(
    fn=fn,
    inputs=image_upload,
    outputs=slider1,
    examples=[chameleon],
    title="Upload Image",
    description="Upload an image to remove its background. Use the slider to compare before and after.",
    api_name="image"
)

# Tab 2: URL Input with Before/After Slider  
tab2 = gr.Interface(
    fn=fn,
    inputs=url_input,
    outputs=slider2,
    examples=[url_example],
    title="Process from URL",
    description="Enter an image URL to process. Perfect for web images.",
    api_name="url"
)

# Tab 3: File Processing with Download
tab3 = gr.Interface(
    fn=process_file,
    inputs=image_file_upload,
    outputs=output_file,
    examples=["butterfly.jpg"],
    title="File Processing",
    description="Upload an image and download the processed PNG file.",
    api_name="file"
)

# Main application with tabbed interface
demo = gr.TabbedInterface(
    [tab1, tab2, tab3], 
    ["🖼️ Image Upload", "🔗 URL Input", "📁 File Download"],
    title="🎭 AI Background Removal Tool",
    description="""
    Remove backgrounds from images using state-of-the-art BiRefNet AI model.
    Choose your preferred input method and get professional results instantly!
    """,
    theme=gr.themes.Soft()
)

# ============================================================================
# APPLICATION LAUNCH
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting Background Removal Tool...")
    demo.launch(
        show_error=True,
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True for public sharing
        inbrowser=True         # Open browser automatically
    )