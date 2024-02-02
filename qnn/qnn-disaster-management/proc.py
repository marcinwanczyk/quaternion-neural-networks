import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as F
from torchvision import transforms

SIZE = [512, 512]
MEAN = (0.485, 0.456, 0.406)
STANDARD_DEVIATION = (0.229, 0.224, 0.225)

def loadImageToTensor(imagePath):
    try:
        image = cv.imread(imagePath)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        resizedImage = cv.resize(image, tuple(SIZE), interpolation=cv.INTER_AREA)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STANDARD_DEVIATION)
        ])
        imageTensor = transform(resizedImage)
        return imageTensor
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def loadGroundTruthImage(imagePath):
    try:
        image = cv.imread(imagePath, cv.IMREAD_GRAYSCALE)
        resizedImage = cv.resize(image, tuple(SIZE), interpolation=cv.INTER_AREA)
        resizedImage = cv.resize(resizedImage, tuple(SIZE), interpolation=cv.INTER_NEAREST)
        return resizedImage
    except Exception as e:
        print(f"Error loading ground truth image: {e}")
        return None

def scale_prediction(prediction_output):
    return F.interpolate(
        prediction_output, SIZE, mode="bilinear", align_corners=True
    )

def prediction_to_classes(prediction_output):
    return prediction_output.argmax(dim=1)

def prediction_to_np_img(prediction_output):
    return prediction_output.cpu().numpy().astype(np.uint8)

