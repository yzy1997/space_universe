from torchvision import my_preprocessing

# Matches the training pipeline exactly:
#   - No Resize        (images stay 256x256)
#   - No Normalize     (was commented out during training)
#   - No augmentation  (RandomRotation/Flip are train-only, skip at inference)
inference_my_preprocessing = my_preprocessing.Compose([
    my_preprocessing.ToPILImage(),
    my_preprocessing.ToTensor(),
])
