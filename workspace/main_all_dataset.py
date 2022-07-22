from dataset import *
from train import *
from model import *
# from IPyth
# on import display
import wandb

def get_train_val(train_path):
    labels = os.listdir(train_path)
    train_data = []
    val_data = []
    for label in labels:
        images_path = glob.glob(f'{train_path}/{label}/*.jpg')
        images_path = [path.replace("\\", "/") for path in images_path]

        train_paths, val_paths = train_test_split(images_path, test_size=0.2, random_state=42)
        val_data += val_paths
        train_data += train_paths
    return train_data, val_data


def main():
    log = {}
    # wandb.init()
    wandb.login(key="5f22a3119b1182dd2ccbb259341f94b253971862")

    CFG = {
        "batch_size" : 32,
        "num_epochs" : 40,
        'pretrained' : True,
        'init_lr' : 0.001,
        'weight_decay' : 0,
        'device' : torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'model_name' : 'resnet34',
        'min_lr' : 0.001,
        'max_lr' : 0.01,
        'patience' : 0,
        'gamma' : 0.1,
        'momentum' : 0.9,
        'optimizer' : 'SGD',
        'lr_scheduler' : 'OneCycleLR',
    }

    train_transforms = A.Compose([
        A.OneOf([
            A.Rotate(30, p=1),
            A.Rotate(-30, p=1),
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
            A.CenterCrop(height=480,width=480,p=1),
            A.Blur(p=1),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=1)
        ], p=0.8),
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    train_path = '../input/paddy-disease-classification/train_images'
    train_data, val_data = get_train_val(train_path)
    train_dataset = PaddyDiseaseClassificationDataset(train_data, dataset_name='train', transforms=train_transforms)
    val_dataset = PaddyDiseaseClassificationDataset(val_data, dataset_name='validation', transforms=val_transforms)
    train_dl = DataLoader(train_dataset, batch_size=CFG['batch_size'], shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=CFG['batch_size'], shuffle=True)

    model = CustomModel(num_classes, 'resnet34', pretrained=CFG['pretrained'])
    model = model.to(CFG['device'])



    # print("dataset has a length of: ", len(dataset))
    # print('torch: ',torch.__version__)
    # print('torchcuda available: ',torch.cuda.is_available())
    # print(torch.__file__)

    if CFG['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=CFG['init_lr'], weight_decay=CFG['weight_decay'])
    elif CFG['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=CFG['init_lr'], momentum=CFG['momentum'], weight_decay=CFG['weight_decay'])
    
    if CFG['lr_scheduler'] == None:
        scheduler = None
    elif CFG['lr_scheduler'] == 'OneCycleLR':
        scheduler = OneCycleLR(optimizer, max_lr=CFG['max_lr'], steps_per_epoch=len(train_dl), epochs=CFG['num_epochs'])
    elif CFG['lr_scheduler'] == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CFG['gamma'], patience=CFG['patience'], min_lr=CFG['min_lr'])

    project_name = "paddy-disease-classification"
    run = wandb.init(project=project_name, config=CFG)

    since = time.time()
    best_loss = 100000

    for epoch in range(0,CFG['num_epochs']):
        loss = train_epoch(model, train_dl, optimizer,scheduler, epoch, CFG, log)
        # save best model
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), "./best_ALL_DATASET.pt")
        # save logs using wandb
        wandb.log(log)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    run.finish()
    # display(ipd.IFrame(run.url, width=1000, height=720))

if __name__ == "__main__":
    main()
