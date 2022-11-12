"""
Functions to perform text-based masking and inpainting.
"""


import torch
from torchvision import transforms
from diffusers import StableDiffusionInpaintPipeline
from clipseg.models.clipseg import CLIPDensePredT
from PIL import Image
from matplotlib import pyplot as plt


def clipseg_model_1(device):
    """Loads clipseg model #1 in inference mode.

    Args:
        device (str): 'cpu' or 'cuda'
  
    Returns:
        clipseg.models.clipseg.CLIPDensePredT
    """
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    model.eval()
    model.load_state_dict(torch.load('weights/rd64-uni.pth', map_location=torch.device(device)), strict=False);

    return model


def clipseg_model_2(device):
    """Loads clipseg model #2 (refined) in inference mode.

    Args:
        device (str): 'cpu' or 'cuda'
  
    Returns:
        clipseg.models.clipseg.CLIPDensePredT
    """
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
    model.eval()
    model.load_state_dict(torch.load('weights/rd64-uni-refined.pth', map_location=torch.device(device)), strict=False)

    return model


def plot_masks(input_image, prompts, mask_images):
    """Plots a horizontal grid of mask images next to the input image.

    Args:
        input_image (PIL.Image.Image): input image to be plotted
        prompts (list): list of strings containing the mask prompts
        mask_images (tensor): generated mask images from the clipseg model

    Returns:
        None
    """
    n = len(prompts)
    _, ax = plt.subplots(1, n + 1, figsize=(5 * n, 5))
    [a.axis('off') for a in ax.flatten()]
    ax[0].imshow(input_image)
    ax[0].text(0, -15, 'input')
    [ax[i+1].imshow(torch.sigmoid(mask_images[i][0]), cmap='gray') for i in range(n)]
    [ax[i+1].text(0, -15, '"' + prompts[i] + '"') for i in range(n)]
    plt.show()


def create_mask(input_path, transform, prompts, model, verbose=2):
    """Create a mask image for every prompt, based on an input image.

    Args:
        input_path (str): filepath for the input image, ex. 'clipseg/example_image.jpg'
        transform (torchvision.transforms.transforms.Compose): image transformation to be applied
        prompts (list): list of strings containing the prompts
        model (clipseg.models.clipseg.CLIPDensePredT): clipseg model
        verbose (int, optional): a value of 2 means mask plots will be printed
    
    Returns:
        PIL.Image.Image: input image after resizing to (512, 512)
        tensor: mask images
    """

    # Transform image
    input_image = Image.open(input_path)
    input_image = input_image.resize((512, 512))
    input_image_trans = transform(input_image).unsqueeze(0)

    # Generate model predictions
    n = len(prompts)
    with torch.inference_mode():
        preds = model(input_image_trans.repeat(n, 1, 1, 1), prompts)[0]

    # Plot results
    if verbose == 2:

        # Raw results
        print('Image Segmentation:')
        plot_masks(input_image, prompts, preds)

        # Binary (0 or 1) results
        cutoff = preds.min() + 0.50 * (preds.max() - preds.min())
        mask_image = torch.where(preds > cutoff, 1.0, 0.0)
        print('Mask Generation:')
        plot_masks(input_image, prompts, mask_image)

    return input_image, mask_image


def mask_and_inpaint(input_filepath, mask_prompt, inpaint_prompt, verbose=1):
    """Performs prompt-based image segmentation on a user-input image, followed by prompt-based inpainting.

    Args:
        input_filepath (str): filepath for the image to be edited
        mask_prompt (str): text description of the object to be removed or replaced
        inpaint_prompt (str): text description of the object to be added; an empty string '' means no guidance will be provided for inpainting
        verbose (int, optional): 0 (no plots), 1 (prints input and output images), or 2 (additionally, prints mask plots)

    Returns:
        PIL.Image.Image: output image
    """

    # Check that the GPU is being used
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print('Warning: CPU is being used. Switch to GPU.')

    # Load a clipseg model
    model = clipseg_model_2(device)

    # Define a transform for the input image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create mask image(s), and convert to PIL Image
    input_image, mask_image = create_mask(input_filepath, transform, [mask_prompt], model, verbose=verbose)
    mask_image = transforms.ToPILImage()(mask_image.squeeze(0))

    # Perform inpainting
    pipe = StableDiffusionInpaintPipeline.from_pretrained('runwayml/stable-diffusion-inpainting', revision='fp16', torch_dtype=torch.float16).to(device)
    output_image = pipe(prompt=inpaint_prompt, image=input_image, mask_image=mask_image).images[0]

    # Print input and output plots
    if verbose >= 1:
        print('Result:')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        ax1.imshow(input_image)
        ax2.imshow(output_image)
        ax1.axis('off')
        ax2.axis('off')
        ax1.text(0, -15, 'input')
        ax2.text(0, -15, 'output')
        plt.show()

    return output_image

