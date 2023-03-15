import torchvision.transforms as transforms


def input_transform():
    return transforms.Compose([
        transforms.ToTensor()
    ])


def get_image_by_tensor():
    return transforms.Compose([
        transforms.ToPILImage()
    ])
