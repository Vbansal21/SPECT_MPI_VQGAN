import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import lightning as pl
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn import CrossEntropyLoss
from torch.nn.functional import l1_loss, mse_loss
from torch.cuda.amp import GradScaler, autocast
torch.autograd.set_detect_anomaly(True)

l2_loss = mse_loss

# Define custom dataset class with preprocessing
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []
        self.class_to_idx = {}
        idx = 0

        class_folders = [
            cf for cf in os.listdir(self.root_dir) 
            if os.path.isdir(self.root_dir)
            ]

        for class_folder in class_folders:
            class_path = os.path.join(self.root_dir, class_folder)
            self.class_to_idx[class_folder] = idx
            idx += 1
            files = [
                        os.path.join(class_path, file) 
                        for file in os.listdir(class_path)
                    ]
            self.samples.extend(
                [
                    (   
                        file,
                        self.class_to_idx[class_folder]
                    )
                    for file in files
                ]
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path, label = self.samples[idx]
        sample = torch.load(sample_path)
        if self.transform:
            sample = torch.FloatTensor(self.transform(sample))
        return sample, label

# Define your custom data module
class CustomDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        train_transforms = transforms.Compose([
            #transforms.RandomRotation(15),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            #transforms.RandomResizedCrop((40, 64, 64)),
            #transforms.ToTensor(),
            torch._cast_Float,
            #transforms.Normalize((0.5,0.5),(0.5,0.5))
        ])
        
        test_transform = transforms.Compose([
            #transforms.ToTensor(),
            torch._cast_Float,
            #transforms.Normalize((0.5,0.5),(0.5,0.5))
        ])

        train_dataset = CustomDataset(root_dir=os.path.join(self.root_dir, "train"), transform=train_transforms)
        val_dataset = CustomDataset(root_dir=os.path.join(self.root_dir, "test"), transform=train_transforms)
        test_dataset = CustomDataset(root_dir=os.path.join(self.root_dir, "test"), transform=test_transform)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

# Define your VQ loss function
class VQLoss(nn.Module):
    def __init__(self, commitment_cost):
        super(VQLoss, self).__init__()
        self.commitment_cost = commitment_cost

    def forward(self, z_e, z_q):
        # Compute VQ loss
        vq_loss = torch.mean((z_e.detach() - z_q) ** 2) + self.commitment_cost * torch.mean((z_e - z_q.detach()) ** 2)
        return vq_loss

# Define your VQ-VAE-GAN architecture
class VQVAEGAN(pl.LightningModule):
    def __init__(self, config: dict) -> None:
        super(VQVAEGAN, self).__init__()
        
        self.scaler = GradScaler()

        self.config: dict = config

        self.embedding_dim: int = self.config.get('embedding_dim',256)
        self.num_embeddings: int = self.config.get('num_embeddings',16)
        self.commitment_cost: float = self.config.get('commitment_cost',0.25)
        self.num_layers: int = self.config.get('num_layers',4)
        self.min_dim: int = self.config.get('min_dim',4)
        self.in_channels: int = self.config.get('in_channels',2)
        self.classes: int = self.config.get("classes",2)
        self.config['lr'] = self.config.get('lr',0.01)

        # VQ-VAE components
        self.gan_loss = torch.nn.BCEWithLogitsLoss()
        self.BCE_loss = torch.nn.functional.binary_cross_entropy_with_logits
        self.vq_vae_loss = VQLoss(
                commitment_cost = self.commitment_cost
            )

        # Encoder, Decoder, Discriminator

        ch_mth: function = lambda i: max(
                                self.min_dim , 
                                self.embedding_dim // 
                                    ( 
                                        2 **
                                            (
                                            self.num_layers-i
                                            )
                                    )
                                )

        # Encoder
        i: int = 1
        in_channels: int = self.in_channels
        out_channels: int = ch_mth(i)

        self.encoder: nn.Sequential = nn.Sequential()
        self.encoder.append(
                nn.Conv3d(
                        in_channels = in_channels, 
                        out_channels = out_channels, 
                        kernel_size = 4, 
                        stride = 2, 
                        padding = 1
                )
            )
        self.encoder.append(
                nn.BatchNorm3d(out_channels)
            )
        self.encoder.append(
                nn.LeakyReLU(0.2)
            )
        in_channels = out_channels

        for i in range(2, 1+self.num_layers):
            out_channels: int = ch_mth(i)
            self.encoder.append(
                    nn.Conv3d(
                        in_channels = in_channels, 
                        out_channels = out_channels, 
                        kernel_size = 4, 
                        stride = 2, 
                        padding = 1
                    )
                )
            self.encoder.append(
                    nn.BatchNorm3d(out_channels)
                )
            self.encoder.append(
                    nn.LeakyReLU(0.2)
                )
            in_channels = out_channels
        
        #Decoder
        i: int = self.num_layers
        in_channels: int = self.embedding_dim
        out_channels: int = ch_mth(i)

        self.decoder: nn.Sequential = nn.Sequential()
        self.decoder.append(
                nn.ConvTranspose3d(
                        in_channels = in_channels, 
                        out_channels = out_channels, 
                        kernel_size = 4, 
                        stride = 2, 
                        padding = 1
                )
            )
        self.decoder.append(
                nn.BatchNorm3d(out_channels)
            )
        self.decoder.append(
                nn.ReLU()
            )
        in_channels = out_channels

        for i in range(self.num_layers,0,-1):
            out_channels = ch_mth(i) if i != 1 else self.in_channels
            self.decoder.append(
                    nn.ConvTranspose3d(
                        in_channels = in_channels, 
                        out_channels = out_channels, 
                        kernel_size = 4, 
                        stride = 2, 
                        padding = 1
                    )
                )
            self.decoder.append(
                    nn.BatchNorm3d(out_channels)
                )
            self.decoder.append(
                    nn.ReLU() if i != 1 else nn.Tanh()
                )
            in_channels = out_channels

        
        # Discriminator
        i: int = 1
        in_channels: int = self.in_channels
        out_channels: int = ch_mth(i)

        self.discriminator: nn.Sequential = nn.Sequential()
        self.discriminator.append(
                nn.Conv3d(
                        in_channels = in_channels, 
                        out_channels = out_channels, 
                        kernel_size = 4, 
                        stride = 2, 
                        padding = 1
                )
            )
        self.discriminator.append(
                nn.BatchNorm3d(out_channels)
            )
        self.discriminator.append(
                nn.LeakyReLU(0.2)
            )
        in_channels = out_channels

        for i in range(2, 1+self.num_layers):
            out_channels = ch_mth(i)
            self.discriminator.append(
                    nn.Conv3d(
                        in_channels = in_channels, 
                        out_channels = out_channels, 
                        kernel_size = 4, 
                        stride = 2 if i != self.num_layers else 1, 
                        padding = 1 if i != self.num_layers else 0
                    )
                )
            self.discriminator.append(
                    nn.BatchNorm3d(out_channels)
                )
            self.discriminator.append(
                    nn.LeakyReLU(0.2) if i != self.num_layers else nn.Sigmoid()
                )
            in_channels = out_channels

        self.discriminator.append(
                nn.AdaptiveAvgPool3d((1,1,1))
            )

        # Quantization Embeddings
        self.embeddings: nn.Module = nn.Embedding(self.num_embeddings, self.embedding_dim)

        # Classifier Layer Modules
        self.pool: nn.Module = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc: nn.Module = nn.Linear(self.embedding_dim,2)

        # Exponentially Moving Average (EMA)
        #self.swa_model: nn.Module = AveragedModel(self)

        self.automatic_optimization = False

    def quantize(self, z_e: torch.tensor):
        distances = torch.cdist(z_e.unsqueeze(1), self.embeddings.weight.unsqueeze(0))
        indices = torch.argmin(distances.squeeze(1), dim=1)
        z_q = self.embeddings(indices)
        return z_q

    def forward(self, x: torch.tensor):
        z_e = self.encoder(x)
        ze = z_e
        B, C, D, H, W = z_e.size()
        z_e = z_e.permute(0,2,3,4,1).contiguous().view(-1, C)
        z_q = self.quantize(z_e)
        z_q = z_q.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        x_recon = self.decoder(z_q)
        return (x_recon,z_q,ze)

    def training_step(self, batch: tuple, batch_idx: int):
        x, y = batch
        opt1, opt2, opt3 = self.optimizers()
        schd1, schd2, schd3 = self.lr_schedulers()

        # Ensure the model is in training mode
        self.train()

        # Classifier loss calculation and update
        opt1.zero_grad()  # Reset gradients to zero for next computation
        with autocast():
            x_recon, z_q, z_e = self(x)
            # Adjust the size of x_recon to match x
            x_recon = torch.nn.functional.interpolate(x_recon, x.shape[2:])
            pooled = self.pool(z_e).squeeze(2, 3, 4)
            y_hat = self.fc(pooled)
            classifier_loss = torch.nn.functional.cross_entropy(y_hat, y)

        self.scaler.scale(classifier_loss).backward(retain_graph=1)
        self.scaler.step(opt1)
        opt1.zero_grad()  # Reset gradients to zero for next computation
        self.scaler.update()  # Update the scaler
        schd1.step()  # Step the scheduler

        # Generator (and VQ-VAE) loss calculation and update
        opt2.zero_grad()  # Reset gradients
        with autocast():
            # Reconstruction and VQ-VAE loss
            recon_loss = F.mse_loss(x_recon, x)
            vq_vae_loss = self.vq_vae_loss(z_e, z_q)
            # GAN loss for the generator
            fake_pred = self.discriminator(x_recon).view(-1, self.embedding_dim, 1, 1, 1).contiguous()
            gan_loss = self.gan_loss(fake_pred, torch.ones_like(fake_pred))

        # Accumulate generator losses
        generator_loss = recon_loss + gan_loss + vq_vae_loss
        self.scaler.scale(generator_loss).backward(retain_graph=1)
        self.scaler.step(opt2)
        opt2.zero_grad()  # Reset gradients
        self.scaler.update()  # Update the scaler
        schd2.step()  # Step the scheduler

        # Discriminator loss calculation and update
        opt3.zero_grad()  # Reset gradients
        with autocast():
            real_pred = self.discriminator(x).view(-1, self.embedding_dim, 1, 1, 1).contiguous()
            d_real_loss = self.BCE_loss(real_pred, torch.ones_like(real_pred))
            d_fake_loss = self.BCE_loss(fake_pred, torch.zeros_like(fake_pred))
            discriminator_loss = d_real_loss + d_fake_loss

        self.scaler.scale(discriminator_loss).backward()
        self.scaler.step(opt3)
        opt3.zero_grad()  # Reset gradients
        self.scaler.update()  # Update the scaler
        schd3.step()  # Step the scheduler

        # Logging losses
        self.log('train_classifier_loss', classifier_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_vq_vae_loss', vq_vae_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_gan_loss', gan_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_discriminator_loss', discriminator_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            x_recon, z_q, z_e = self(x)
            pooled = self.pool(z_e).squeeze(2,3,4)
            print(pooled.shape)
            y_hat = self.fc(pooled)
            loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer_classifier = Adam(
            list(
                self.fc.parameters()
                ) + list(
                        self.pool.parameters()
                        ), 
            lr=self.config['lr']
            )
        optimizer_generator = Adam(
            list(
                self.decoder.parameters()
                ) + list(
                    self.encoder.parameters()
                    ) + list(
                        self.embeddings.parameters()
                        ),
            lr=self.config['lr']
        )
        optimizer_discriminator = Adam(
            self.discriminator.parameters(), 
            lr=self.config['lr']
            )
        
        scheduler_classifier = {
            'scheduler': OneCycleLR(optimizer_classifier, max_lr=self.config['lr'], epochs=10, steps_per_epoch=100, pct_start=0.1),
            'interval': 'step',
            'frequency': 1
        }
        scheduler_generator = {
            'scheduler': OneCycleLR(optimizer_generator, max_lr=self.config['lr'], epochs=10, steps_per_epoch=100, pct_start=0.1),
            'interval': 'step',
            'frequency': 1
        }
        scheduler_discriminator = {
            'scheduler': OneCycleLR(optimizer_discriminator, max_lr=self.config['lr'], epochs=10, steps_per_epoch=100, pct_start=0.1),
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer_classifier, optimizer_generator, optimizer_discriminator], [scheduler_classifier, scheduler_generator, scheduler_discriminator]

# Initialize Lightning model
input_dim = (1, 1, 80, 64, 64)
root_dir = "./dataset/"
output_dim = len(set(os.listdir(os.path.join(root_dir, "train"))) | set(os.listdir(os.path.join(root_dir, "test"))))
config = {
    "in_channels" : input_dim[1],
    "classes" : output_dim,
}
model = VQVAEGAN(config)

# Define callbacks for checkpointing and early stopping
checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    dirpath='./checkpoints/',
    filename='best_model'
)

early_stop_callback = pl.pytorch.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    mode='min'
)

# Initialize Lightning trainer with callbacks
trainer = pl.Trainer(
    max_epochs=10,
    #gpus=1,
    precision=32,  # Automatic Mixed Precision (AMP)
    #accumulate_grad_batches=4,  # Gradient Accumulation
    callbacks=[checkpoint_callback, early_stop_callback]
)

# Initialize data module
batch_size = 1
data_module = CustomDataModule(root_dir=root_dir, batch_size=batch_size)

# Train the model
trainer.fit(model, datamodule=data_module)

# Test the model
trainer.test(model, datamodule=data_module)

# Applying Exponentially Moving Average (EMA)
model.swa_model = trainer.swa_model
