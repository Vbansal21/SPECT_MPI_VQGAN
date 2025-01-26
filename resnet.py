from functools import partial, reduce
import os
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import lightning as pl
from torch.optim import AdamW
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from math import ceil
from transformers import CLIPProcessor, CLIPModel

#Miscellaneous system/environment variables
anomaly_detect = False
torch.autograd.set_detect_anomaly(anomaly_detect)
use_ckpt = 0
data_transform = transforms.Compose([
    transforms.PILToTensor(),
    torch._cast_Float
])

# Defining VQ loss function
class VQLoss(nn.Module):
    def __init__(self, commitment_cost):
        super(VQLoss, self).__init__()
        self.commitment_cost = commitment_cost

    def forward(self, z_e, z_q):
        # Compute VQ loss
        vq_loss = torch.mean((z_e.detach() - z_q) ** 2) + self.commitment_cost * torch.mean((z_e - z_q.detach()) ** 2)
        return vq_loss

#Defining helper functions/classes
def znormch(x):
    M = torch.max(x)
    m = torch.min(x)
    return ((x - m)/(M - m))

class Ckptfn():
    def __init__(self,use_ckpt=False,sequential=True):
        self.use_ckpt = use_ckpt
        self.sequential = sequential

    def __call__(self,fn,x,n=None):
        if not self.use_ckpt:
            return fn(x)
        else:
            if self.sequential:
                return torch.utils.checkpoint.checkpoint_sequential(fn,len(fn),x,use_reentrant = False)
            else:
                return torch.utils.checkpoint.checkpoint(fn,x,use_reentrant = False)

class Model(pl.LightningModule):
    def __init__(self,config: dict) -> None:
        super(Model, self).__init__()

        self.config = config

        self.lr = self.config.get('lr',0.1)
        self.epochs = self.config.get('epochs',10)
        self.num_steps = self.config.get("num_steps",self.epochs)
        self.pct_start = self.config.get("pct_start",0.3)
        self.divf = self.config.get("div_factor",25)
        self.fdivf = self.config.get("final_div_factor",1e6)
        class_weights = self.config.get("class_weights",None)

        self.register_buffer("class_weights",class_weights)

        self.vism = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.vism.requires_grad_(False)

        self.embed_dim = 256 #self.vism.config["hidden_size"]
        self.img_dim = 224 #self.vism.config["image_size"]

        self.classifier = nn.Sequential(
            nn.Linear(self.vism.fc.in_features,self.embed_dim),
            nn.Tanh(),
            nn.Linear(self.embed_dim,self.embed_dim),
            nn.Tanh(),
            nn.Linear(self.embed_dim,self.embed_dim),
            # nn.Tanh(),
            nn.Linear(self.embed_dim,2),
        )

        del self.vism.fc

        self.vism.fc = self.classifier


    def forward(self, batch: tuple[torch.Tensor], batch_idx: int = None) -> tuple[torch.Tensor]:
        x,y = batch
        B,C,W,H = x.size()
        inp = x
        inp = F.interpolate(inp,(self.img_dim,self.img_dim))
        out = self.vism(inp)
        return out

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        y_true = batch[1]
        y_hat = self(batch,batch_idx)

        lf = partial(F.binary_cross_entropy_with_logits,weight=torch.softmax(torch.tensor((100/78.125,100/21.875), device=y_true.device),dim=-1))

        loss = lf(y_hat,y_true)

        values, indices = torch.topk(torch.softmax(y_hat,dim=-1),dim=-1,k=1)
        _, label = torch.topk(y_true,dim=-1,k=1)
        results = torch._cast_Float(indices==label)
        conf = ((results*values).mean())*100
        acc = (results.mean())*100

        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_epoch=True,  logger=True)
        self.log('train_conf', conf, on_epoch=True,  logger=True)

        return loss


    def validation_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        y_true = batch[1]
        y_hat = self(batch,batch_idx)

        lf = partial(F.binary_cross_entropy_with_logits)

        loss = lf(y_hat,y_true)

        values, indices = torch.topk(torch.softmax(y_hat,dim=-1),dim=-1,k=1)
        _, label = torch.topk(y_true,dim=-1,k=1)
        results = torch._cast_Float(indices==label)
        conf = ((results*values).mean())*100
        acc = (results.mean())*100

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_epoch=True,  logger=True)
        self.log('val_conf', conf, on_epoch=True,  logger=True)

        return loss

    def test_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        y_true = batch[1]
        y_hat = self(batch,batch_idx)

        lf = partial(F.binary_cross_entropy_with_logits)

        loss = lf(y_hat,y_true)

        values, indices = torch.topk(torch.softmax(y_hat,dim=-1),dim=-1,k=1)
        _, label = torch.topk(y_true,dim=-1,k=1)
        results = torch._cast_Float(indices==label)
        conf = ((results*values).mean())*100
        acc = (results.mean())*100

        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_epoch=True,  logger=True)
        self.log('test_conf', conf, on_epoch=True,  logger=True)

        return loss

    def configure_optimizers(self) -> torch.nn.Module:
        optimizer = AdamW(
            self.parameters(),
            lr = self.lr,
            betas = self.config.get("betas",(0.9,0.999)),
            eps = self.config.get("eps",1e-8),
            weight_decay = self.config.get("weight_decay",1e-2),
            amsgrad = self.config.get("amsgrad",False),
            maximize = self.config.get("maximize",False),
            foreach = self.config.get("foreach",None),
            capturable = self.config.get("capturable",False),
            differentiable = self.config.get("differentiable",False),
            fused = self.config.get("fused",True),
            )
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.lr,
            epochs=self.epochs,
            steps_per_epoch=self.num_steps,
            pct_start=self.pct_start,
            div_factor=self.divf,
            final_div_factor=self.fdivf,
            base_momentum=self.config.get("base_momentum",0.85),
            max_momentum=self.config.get("max_momentum",0.95),
            )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]

# Define custom dataset class with preprocessing
class SubDataset(Dataset):
    def __init__(self,samples,transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path, label = self.samples[idx]
        sample = Image.open(sample_path)
        if self.transform:
            sample = znormch(self.transform(sample))
        return torch.FloatTensor(sample), torch.FloatTensor(label)

    def getitem_withname(self, idx):
        sample_path, label = self.samples[idx]
        sample = Image.open(sample_path)
        if self.transform:
            sample = znormch(self.transform(sample))
        return torch.FloatTensor(sample), torch.FloatTensor(label),sample_path

class SuperDataset(Dataset):
    def __init__(self, root_dir="./dataset/", transform=None):
        self.root_dir = root_dir
        data_split=["train","test","val"]
        self.transform = transform
        if transform is None:
            raise "Transforms not provided!!! Please check."


        self.samples = dict([(datatype,[]) for datatype in data_split])
        self.class_to_idx = {}
        self.class_weights = {}
        idx = 0

        class_folders = []
        unique_classes = list(
            reduce(
                lambda x,y: set(x) | set(y),
                [
                    set(
                        [
                            class_name for class_name in
                            os.listdir(
                            folder_path
                            ) if os.path.isdir(
                                os.path.join(
                                    folder_path,
                                    class_name
                                    )
                            )
                            ]
                        )
                        for folder_path in [
                            os.path.join(
                                self.root_dir,
                                datatype
                                )
                            for datatype in data_split
                        ] if os.path.isdir(folder_path)
                    ],
                []
                )
            )

        self.class_to_idx = dict(
            [
                (class_name,idx)
                for idx,class_name in
                enumerate(unique_classes)
                ]
            )
        for datatype in data_split:
            folder_path = os.path.join(self.root_dir,datatype)
            if os.path.isdir(folder_path):
                for class_name in unique_classes:
                    class_path = os.path.join(folder_path,class_name)
                    if os.path.exists(class_path):
                        class_folders.append(class_path)
                    else:
                        os.makedirs(class_path,exist_ok=True)

        for class_path in class_folders:
            path = Path(class_path)
            class_folder = path.parts[-1]
            datatype = path.parts[-2]

            idx = self.class_to_idx[class_folder] if self.class_to_idx[class_folder] is not None else idx
            self.class_to_idx[class_folder] = idx
            files = [
                    os.path.join(class_path, file)
                    for file in os.listdir(class_path)
                    ]
            if datatype=="train":
                self.class_weights[idx] = self.class_weights.get(idx,0) + len(files)
            self.samples[datatype].extend(
                [
                    (
                        file,
                        [1 if self.class_to_idx[class_folder]==i else 0 for i in range(len(self.class_to_idx.keys()))]
                    )
                    for file in files
                ]
            )
            idx += 1

    def __len__(self):
        return len(reduce(lambda l1,l2: l1.extend(l2),[i for i in self.samples.values()],[]))

    def get_class_weights(self):
        return self.class_weights

    def get_num_classes(self):
        return len(list(self.class_to_idx.keys()))

    def __getitem__(self, idx):
        if type(idx) is int: idx = self.data_split[idx]
        return SubDataset(self.samples[idx],self.transform)

# Initialize data module
root_dir = os.path.join(os.curdir,"SPECT_MPI_Dataset","")
SuperSet = SuperDataset(root_dir=root_dir,transform=data_transform)
class_weights = SuperSet.get_class_weights()
num_classes = SuperSet.get_num_classes()
train_dataset = SuperSet["train"]
val_dataset = SuperSet["val"]
test_dataset = SuperSet["test"]

total_files = reduce(lambda x,y: x+y,class_weights.values(),0)
class_weights = [total_files/i for i in class_weights.values()]
class_weights = tuple([i/reduce(lambda x,y: x+y,class_weights,0) for i in class_weights])

# Initialize Lightning model
input_dim = (1, 3, 80, 88, 88)
epochs = 10
batch_size = 8
accumulate_grad_batches=ceil(len(train_dataset)/batch_size)
config = {
    "in_channels" : input_dim[1],
    "classes" : num_classes,
    "epochs" : epochs,
    "lr":0.1,
    "num_steps" : epochs*len(train_dataset)//accumulate_grad_batches,
    "embedding_dim":4,
    "num_layers":6,
    "layer_repetition":-1,
    'num_embeddings':2,
    "pct_start":0.15,#7/batch_size,
    "div_factor":25,
    "final_div_factor":1e6,
    "alpha":0.5,
    "weight_decay":0.1,
    "betas":(0.9,0.999),
    "amsgrad":True,
    "fused":torch.cuda.is_available(),
    "class_weights":torch.softmax(torch.tensor(class_weights),dim=-1),
    "use_ckpt":0,
}
model = Model(config)

# Define callbacks for checkpointing and early stopping
checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
    monitor='val_acc',
    mode='max',
    dirpath=os.path.join(os.curdir,"checkpoints",""),
    filename='best_model'
)

early_stop_callback = pl.pytorch.callbacks.EarlyStopping(
    monitor='val_acc',
    patience=7,
    mode='max'
)

# Initialize Lightning trainer with callbacks
trainer = pl.Trainer(
    max_epochs=epochs,
    accelerator="gpu",
    log_every_n_steps=1,
    num_nodes=1,
    precision="32",  # Automatic Mixed Precision (AMP)
    accumulate_grad_batches=accumulate_grad_batches,  # Gradient Accumulation
    callbacks=[
        #checkpoint_callback,
        # early_stop_callback,
        ],
    detect_anomaly=anomaly_detect,
)

# Train the model

train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=11)
test_dl = DataLoader(test_dataset, batch_size=batch_size, num_workers=11)
val_dl = DataLoader(val_dataset, batch_size=batch_size, num_workers=11)

trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

# Test the model
trainer.test(model, dataloaders=test_dl)



from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Calculate metrics on test set
all_y_true = []
all_y_pred = []

for batch_idx, batch in enumerate(test_dl):
    x, y_true = batch
    with torch.no_grad():
        # tmp = model.infer_vae(x)  # Use the custom inference method
        y_hat = model((x,None))  # Extract the tensor containing predictions from the output tuple

        y_probs = F.softmax(y_hat,dim=-1)

        values, indices = torch.max(y_probs, dim=-1)  # Get predicted labels

    all_y_true.extend(y_true.cpu().numpy().tolist())  # Convert to list
    all_y_pred.extend(indices.cpu().numpy().tolist())  # Convert to list

# Ensure both all_y_true and all_y_pred are single-label class assignments
# Convert all_y_true to 1D list if it's not already
if isinstance(all_y_true[0], list):
    all_y_true = [item for sublist in all_y_true for item in sublist]

# Ensure all_y_pred has the same length as all_y_true
all_y_pred = all_y_pred[:len(all_y_true)]

# Ensure both all_y_true and all_y_pred have the same length
min_len = min(len(all_y_true), len(all_y_pred))
all_y_true = all_y_true[:min_len]
all_y_pred = all_y_pred[:min_len]

# Calculate metrics
precision = precision_score(all_y_true, all_y_pred, average=None)
recall = recall_score(all_y_true, all_y_pred, average=None)
f1 = f1_score(all_y_true, all_y_pred, average=None)
conf_matrix = confusion_matrix(all_y_true, all_y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)

# Sample output(s)

for i in range(len(test_dataset)):
    data = test_dataset.getitem_withname(i)
    x,y,path = data
    with torch.no_grad():
        # tmp = model.infer_vae(x)  # Use the custom inference method
        y_hat = model((x.unsqueeze(0),None))  # Extract the tensor containing predictions from the output tuple
        y_probs = F.softmax(y_hat,dim=-1)

        values, indices = torch.topk(torch.softmax(y_hat,dim=-1),dim=-1,k=1)
        _, label = torch.topk(y,dim=-1,k=1)
        results = torch._cast_Float(indices==label)
        conf = ((results*values).mean())*100
        acc = (results.mean())*100
    print(f"File Name:{path}, \t Class:{int(torch.max(y,dim=-1)[1])}, \t Prediction;{int(indices)} \t Confidence:{conf}")

# Save the model
torch.save(model.state_dict(), 'model.pth')

# Load the model
loaded_model = Model(config)
loaded_model.load_state_dict(torch.load('model.pth'))
loaded_model.eval()
