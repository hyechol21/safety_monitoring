import time
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
import torch.nn.functional as F


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        pad = abs(w-h) // 2

        if w > h:
            padding = (0, 0, 0, 0, pad, pad)
            pass
        else:
            padding = (0, 0, pad, pad, 0, 0)

        X = torch.Tensor(np.asarray(image))
        X = F.pad(X, padding, "constant", value=0) # tensor 반환

        padX = X.data.numpy()
        padX = np.uint8(padX)
        padim = Image.fromarray(padX, 'RGB') # 데이터를 이미지 객체로 변환
        return padim

def preprocess(image_size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        SquarePad(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform


class Classify:
    def __init__(self, model_name, weight, labels_map, threshold=0.5):
        self.model_name = model_name
        self.weight = weight
        self.labels_map = labels_map

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.image_size = None
        self.model = self.load_model(model_name, weight, len(labels_map))
        self.tfms = preprocess(self.image_size)

    def load_model(self, model_name, weight, num_classes):
        if model_name.find("lite") != -1:  # lite 버전인 경우
            print('lite 버전!!!')
            from efficientnet_lite_pytorch import EfficientNet
        else:
            from efficientnet_pytorch import EfficientNet
            self.image_size = EfficientNet.get_image_size(model_name)

        model = EfficientNet.from_pretrained(model_name=model_name, weights_path=weight, num_classes=num_classes)
        model.load_state_dict(torch.load(weight))
        model = model.to(self.device)
        model.eval()

        if self.image_size is None:
            self.image_size = model.input_image_size
        print(self.image_size)
        return model

    def predict_image_cv2(self, roi):
        # apply transforms to the input image
        start_time = time.time()
        input = self.tfms(roi).unsqueeze(0)

        input = input.to(self.device)

        with torch.no_grad():
            outputs = self.model(input)

        preds = torch.topk(outputs, k=1).indices.squeeze(0).tolist()

        for i, ans_idx in enumerate(preds):
            label = ans_idx, self.labels_map[ans_idx]
            prob = torch.softmax(outputs, dim=1)[0, ans_idx].item()
            # print('{:<75} ({:.2f}%)'.format(label, prob*100))
        del input
        return label, prob

