
from typing import Tuple
import torch
import cv2
import numpy as np
from torchvision import transforms
from bisenetv2 import BiSeNetV2

def load_skin_segmentation_model(model_weight_path: str) -> BiSeNetV2:
    model = BiSeNetV2(['skin'])
    state_dict = torch.load(model_weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def preprocess_image(image: np.ndarray) -> torch.Tensor:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_width = (image.shape[1] // 32) * 32
    image_height = (image.shape[0] // 32) * 32
    resized_image = image[:image_height, :image_width]
    
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    transformed_image = image_transform(resized_image)
    return transformed_image, resized_image

def create_skin_mask(model: BiSeNetV2, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    transformed_image, resized_image = preprocess_image(image)
    
    with torch.no_grad():
        transformed_image = transformed_image.unsqueeze(0)
        results = model(transformed_image)['out']
        results = torch.sigmoid(results)
        
        results = results > 0.5
        mask = results[0].squeeze(0).cpu().numpy() * 255
        mask = mask.astype('uint8')
    
    image_width, image_height = resized_image.shape[1], resized_image.shape[0]
    mask = cv2.resize(mask, (image_width, image_height))

    return mask, resized_image

def refine_mask(resized_image: np.ndarray, model: BiSeNetV2, mask: np.ndarray) -> np.ndarray:
    # Apply histogram equalization directly to the resized image
    equalized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2LAB)
    equalized_image[..., 0] = cv2.equalizeHist(equalized_image[..., 0])
    equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_LAB2BGR)

    # Perform segmentation on the equalized image
    refined_mask, _ = create_skin_mask(model, equalized_image)
    return refined_mask



