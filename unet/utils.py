import torch
import torch.nn as nn
from dataset import SegmentationDataset
import torchvision
from torch.utils.data import DataLoader
import numpy as np

# checkpoints
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
    
# datasets    
def create_datasets(masks_dir, slices_dir, augmentations):
    train_dataset = SegmentationDataset(
                 slices_dir=slices_dir + "/train", 
                 masks_dir=masks_dir + "/train",
                 transforms=augmentations)

    val_dataset = SegmentationDataset(
                     slices_dir=slices_dir + "/val", 
                     masks_dir=masks_dir + "/val",
                     transforms=augmentations)

    test_dataset = SegmentationDataset(
                     slices_dir=slices_dir + "/test", 
                     masks_dir=masks_dir + "/test",
                     transforms=augmentations)

    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(val_dataset)}")
    print(f"Test size:  {len(test_dataset)}")
        
    return train_dataset, val_dataset, test_dataset

# dataloaders
def create_dataloaders(train_dataset, val_dataset, test_dataset, 
                       batch_size=8, pin_memory=True, num_workers=4):

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, 
        shuffle=True, pin_memory=pin_memory, num_workers=num_workers)

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=batch_size, 
        shuffle=True, pin_memory=pin_memory, num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, 
        shuffle=False, pin_memory=pin_memory, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader


# count DICE
def get_dice(target_mask, predicted_mask):
    dice_score = (2 * (predicted_mask * target_mask).sum()) / ((predicted_mask + target_mask).sum() + 1e-8)

#     dice = Dice()
#     metric = dice(predicted_mask, target_mask)
    return dice_score


def mask_to_image(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        return np.array((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return np.array(
            (np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8)
        )


def predict_img(net: nn.Module, img: torch.Tensor, device: str, out_threshold: float = 0.5):
    net.eval()
    net.to(device)

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        probs = torch.sigmoid(output)
        full_mask = probs.cpu().squeeze()

        return full_mask > out_threshold
    
    
def get_predict(model, image, device):    
    if type(image) == SegmentationDataset:
        predicted = []
        for elem in image:
            image = elem[0] 
            mask = elem[1]
            predicted.append(predict_img(model, image, device=device))
        return predicted
    
    predicted_mask = predict_img(model, image, device=device)
    return predicted_mask


def show_three(model, img_list, device):    
    fig, axes = plt.subplots(len(img_list), 3, figsize=(15, 15))
    for (idx, sample) in enumerate(img_list):
        image, target_mask = sample
        predicted_mask = get_predict(image=image, model=model, device=device)
        dice = get_dice(target_mask, predicted_mask)
        
        axes[0].imshow(image.permute(1, 2, 0))
        axes[0].set_title("image")

        axes[1].imshow(target_mask.squeeze())
        axes[1].set_title("target mask")

        axes[2].imshow(predicted_mask.squeeze())
        axes[2].set_title(f"dice: {dice}")
        
        axes[0].get_xaxis().set_visible(False)
        axes[0].get_yaxis().set_visible(False)
        axes[1].get_xaxis().set_visible(False)
        axes[1].get_yaxis().set_visible(False)
        axes[2].get_xaxis().set_visible(False)
        axes[2].get_yaxis().set_visible(False)
    plt.show()




# 
def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1) # label does not have chanel
            
            # getting probs             
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()