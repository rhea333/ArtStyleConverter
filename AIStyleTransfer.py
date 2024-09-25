import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = models.vgg19(pretrained=True).features.to(device).eval()

def load_image(img_path, max_size=400):
    image = Image.open(img_path).convert('RGB')
    size = max_size if max(image.size) > max_size else max(image.size)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

content_image = load_image('assets/content_image.jpg')
style_image = load_image('assets/style_image.jpeg')

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = torch.nn.functional.mse_loss(x, self.target)
        return x

class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target).detach()

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = torch.nn.functional.mse_loss(G, self.target)
        return x

content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

content_losses = []
style_losses = []

model = nn.Sequential()
i = 0
for layer in vgg.children():
    if isinstance(layer, nn.Conv2d):
        i += 1
        name = f'conv_{i}'
    elif isinstance(layer, nn.ReLU):
        name = f'relu_{i}'
        layer = nn.ReLU(inplace=False)
    elif isinstance(layer, nn.MaxPool2d):
        name = f'pool_{i}'
    elif isinstance(layer, nn.BatchNorm2d):
        name = f'bn_{i}'
    else:
        raise RuntimeError('Unrecognized layer')

    model.add_module(name, layer)

    if name in content_layers:
        target = model(content_image).detach()
        content_loss = ContentLoss(target)
        model.add_module(f"content_loss_{i}", content_loss)
        content_losses.append(content_loss)

    if name in style_layers:
        target = model(style_image).detach()
        style_loss = StyleLoss(target)
        model.add_module(f"style_loss_{i}", style_loss)
        style_losses.append(style_loss)

input_img = content_image.clone()
optimizer = optim.LBFGS([input_img.requires_grad_()])

run = [0]
num_steps = 300
style_weight = 1e6
content_weight = 1

while run[0] <= num_steps:
    def closure():
        input_img.data.clamp_(0, 1)
        optimizer.zero_grad()
        model(input_img)
        style_score = 0
        content_score = 0

        for sl in style_losses:
            style_score += sl.loss
        for cl in content_losses:
            content_score += cl.loss

        loss = style_weight * style_score + content_weight * content_score
        loss.backward()

        run[0] += 1
        return style_score + content_score

    optimizer.step(closure)

input_img.data.clamp_(0, 1)

def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

imshow(input_img, title='Pixel Art Styled Image')
