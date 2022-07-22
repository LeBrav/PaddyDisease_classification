from dataset import *
from train import *
from model import *

def main():
    CFG = {
            "batch_size" : 32,
            "num_epochs" : 50,
            'pretrained' : True,
            'init_lr' : 0.001,
            'weight_decay' : 0,
            'device' : 'cuda',
            'model_name' : 'resnet34',
            'min_lr' : 0.001,
            'patience' : 0,
            'gamma' : 0.1,
            'momentum' : 0.9,
            'optimizer' : 'SGD',
        }

    model_path = "../unet_weights/best_972accu_noall.pt"
    model = CustomModel(num_classes, 'resnet34', pretrained=CFG['pretrained'])
    model.load_state_dict(torch.load(model_path))
    model = model.to(CFG['device'])
    model.eval()

    reverse_labels = dict((v, k) for k, v in labels.items())
    images_path = glob.glob('../input/paddy-disease-classification/test_images/*.jpg')
    images_path = [path.replace("\\", "/") for path in images_path]


    val_transforms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])




    submission = []
    for img_path in tqdm(images_path):
        # process image
        img = cv2.imread(img_path)[:, :, ::-1]  # convert it to rgb
        img = img.astype('float32')
        #img /= 255  # scale img to [0, 1]
        img = val_transforms(image=img)['image'] # apply same transforms of validation set
        img = img[None, ...].to(CFG['device']) # add batch dimension to image and use device
        # predict 
        pred = model(img) 
        pred = torch.max(pred, dim=1)[1]
        label = reverse_labels[pred.item()]
        submission.append([img_path.split("/")[-1], label])
    submission = pd.DataFrame(submission, columns=['image_id', 'label'])
    submission.to_csv("./submission.csv", index=False)

if __name__=='__main__':
    main()