import os
import json
from PIL import Image

import torch
from torchvision import transforms

from efficientnet_pytorch import EfficientNet


def get_file_list(folder, file_extensions):
    """get list of files from their extensions
    
    Args
    ----
    folder (str): folder path
    file_extensions (tuple): file extensions to search
    
    Returns 
    -------
    list of file names
    """

    file_list = []
    for file in os.listdir(folder):
        if file.endswith(file_extensions):
            file_list.append(file)
    return sorted(file_list)


def predict(model, alllabels, superclasslabels, search, imgpath, topk=1):
    model_name = model
    image_size = EfficientNet.get_image_size(model_name)

    # Preprocess image
    img = Image.open(imgpath)
    tfms = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), 
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    img = tfms(img).unsqueeze(0)

    # Load class names & superclass name to search
    labels_map = json.load(open(labels))
    labels_map = [labels_map[str(i)] for i in range(1000)]
    superclass = json.load(open(superclasslabels))[search]

    # predict with EfficientNet
    model = EfficientNet.from_pretrained(model_name)
    model.eval()
    with torch.no_grad():
        logits = model(img)
    preds = torch.topk(logits, k=1).indices.squeeze(0).tolist()

    for idx in preds:
        imgname = imgpath.split("/")[-1]
        prob = torch.softmax(logits, dim=1)[0, idx].item()
        if idx in superclass:
            print("{} is bird ({:.2f}%)".format(imgname, prob*100))
        else:
            print("{} is NOT bird ({:.2f}%)".format(imgname, prob*100))


if __name__ == "__main__":
    folder = "/Users/siyang/Desktop/birdpics/Black-capped Kingfisher"
    birdlist = get_file_list(folder, ("jpeg", "png", "bmp"))

    model='efficientnet-b0'
    labels = "labels/labels_map.json"
    superclass = "labels/superclass.json"
    search = "birds"
    imgpath = "/Users/siyang/Desktop/birdpics/Black-capped Kingfisher/Baidu_Black-capped_Kingfisher_00084.png"
    predict(model, labels, superclass, search, imgpath)
    # for image in birdlist:
    #     imgpath = os.path.join(folder, image)
    #     predict(model, labels, superclass, search, imgpath)

    import cv2
    img = cv2.imread(imgpath).shape
    print(img)