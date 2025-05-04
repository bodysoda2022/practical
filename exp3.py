import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

transform = transforms.Compose([transforms.ToTensor()])

image_path = "dog_images.jpg"
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    predictions = model(image_tensor)
    predictions = predictions[0]

def visualize_predictions(image, predictions, threshold=0.5):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    for idx in range(len(predictions['boxes'])):
        score = predictions['scores'][idx].item()
        if score > threshold:
            box = predictions['boxes'][idx].cpu().numpy()
            label = predictions['labels'][idx].cpu().numpy()
            color = np.random.rand(3)
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(box[0], box[1], f'{label}: {score:.2f}', color=color, fontsize=12)
    plt.axis('off')
    plt.show()

visualize_predictions(image, predictions)
