from functools import partial, reduce
import os
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import lightning as pl
from torch.optim import AdamW
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from math import ceil

#Miscellaneous system/environment variables
anomaly_detect = False
torch.autograd.set_detect_anomaly(anomaly_detect)
use_ckpt = 0
data_transform = transforms.Compose([
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
    # mu,sigma = torch.std_mean(x,dim=1)
    # return (x - mu.unsqueeze(1))/sigma.unsqueeze(1)
    # mu,sigma = torch.std_mean(x)
    # return (x - mu)/sigma
    M = torch.max(x)
    m = torch.min(x)
    return ((x - m)/(M - m))*2 - 1
    # return x / torch.norm(x)

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

# Defining VQ-VAE-GAN architecture
class Model(pl.LightningModule):
    def __init__(self, config: dict) -> None:
        super(Model, self).__init__()

        self.config: dict = config

        self.embedding_dim: int = self.config.get('embedding_dim',32)
        self.num_embeddings: int = self.config.get('num_embeddings',2)
        self.commitment_cost: float = self.config.get('commitment_cost',0.33)
        self.num_layers: int = max(5,self.config.get('num_layers',8))
        self.in_channels: int = self.config.get('in_channels',2)
        self.min_dim: int = self.config.get('min_dim',4)
        self.classes: int = self.config.get("classes",2)
        self.lr = self.config.get('lr',4)
        self.epochs = self.config.get("epochs",10)
        self.data_shape = self.config.get("shape",(80,88,88))
        self.encdis = self.config.get("encoder_as_discriminator",False)
        self.batch_size = self.config.get("batch_size",1)
        self.dropout_prob = self.config.get("dropout_prob",0.3)
        self.alpha = self.config.get("alpha",0.99)
        self.num_steps = self.config.get("num_steps",100)
        self.layer_rep = self.config.get("layer_repetition",6)
        self.layer_rep = max(self.layer_rep,6) if self.layer_rep != -1 else -1
        self.pct_start = self.config.get("pct_start",0.3)
        self.divf = self.config.get("div_factor",25)
        self.fdivf = self.config.get("final_div_factor",1e6)
        self.noise_pct = self.config.get("noise_pct",0.001)
        self.use_ckpt = self.config.get("use_ckpt",use_ckpt)


        # Encoder, Decoder, Discriminator

        ch_mth: function = lambda i: int( min(
            max(
                self.min_dim * 2**min(i-1,4),
                self.embedding_dim //
                    (
                        2 **
                            (
                            self.num_layers-i
                            )
                    )
                ),
                self.embedding_dim )) if i!=0 else self.in_channels

        # Encoder
        i: int = 1
        in_channels: int = self.in_channels
        out_channels: int = ch_mth(i)

        self.encoder: nn.Sequential = nn.Sequential()
        self.encoder.append(
            nn.Conv3d(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = 3,
                    stride = 1,
                    padding = 1
            )
        )
        self.encoder.append(nn.AlphaDropout(self.dropout_prob))
        self.encoder.append(
            nn.BatchNorm3d(out_channels)
        )
        self.encoder.append(
            nn.SiLU()
        )
        in_channels = out_channels

        module_init = None

        for i in range(2, 1+self.num_layers):
            out_channels: int = ch_mth(i) if i<(self.layer_rep-1) or self.layer_rep==-1 else ch_mth(self.num_layers)
            module_init = nn.Conv3d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ) if i<self.layer_rep  or self.layer_rep==-1 else module_init
            self.encoder.append(
                module_init
            )
            self.encoder.append(nn.AlphaDropout(self.dropout_prob))
            if i < 6:
                self.encoder.append(
                    nn.BatchNorm3d(out_channels)
                )
                self.encoder.append(
                    nn.SiLU()
                )
                self.encoder.append(
                    nn.Conv3d(
                        in_channels = out_channels,
                        out_channels = out_channels,
                        kernel_size = 3,
                        stride = 2,
                        padding = 1
                    )
                )
                self.encoder.append(nn.AlphaDropout(self.dropout_prob))
            self.encoder.append(
                nn.BatchNorm3d(out_channels)
            )
            self.encoder.append(
                nn.SiLU()
            )
            in_channels = out_channels

        self.enc_mu = nn.Linear(in_features=out_channels,out_features=out_channels)
        self.enc_sigma = nn.Linear(in_features=out_channels,out_features=out_channels)
        self.embed_y = nn.Linear(in_features=self.classes+out_channels,out_features=self.embedding_dim)

        #Decoder
        i: int = self.num_layers
        in_channels: int = self.embedding_dim
        out_channels: int = ch_mth(i)

        self.decoder: nn.Sequential = nn.Sequential()
        self.decoder.append(
            nn.ConvTranspose3d(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = 3,
                    stride = 1,
                    padding = 1
            )
        )
        self.decoder.append(nn.AlphaDropout(self.dropout_prob))
        self.decoder.append(
            nn.BatchNorm3d(out_channels)
        )
        self.decoder.append(
            nn.SiLU()
        )
        in_channels = out_channels

        for i in range(self.num_layers,0,-1):
            out_channels = (ch_mth(i) if i != 1 else self.in_channels) if (
                i > self.num_layers-(self.layer_rep-2)
                ) or self.layer_rep==-1 else ch_mth(0)
            func = nn.ConvTranspose3d if i <= self.num_layers-4 else nn.Conv3d
            module_init = func(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ) if (
                i > (
                self.num_layers-self.layer_rep
                )
            ) or self.layer_rep==-1 else module_init
            self.decoder.append(
                module_init
            )
            self.decoder.append(nn.AlphaDropout(self.dropout_prob))
            if (i > (self.num_layers-4)):
                # self.decoder.append(
                self.decoder.append(
                    nn.Upsample(
                        scale_factor=2,
                        mode='trilinear',
                        align_corners=True
                    )
                )
            if (i==(self.num_layers-4)):
                self.decoder.append(
                    nn.Upsample(
                        size = self.data_shape,
                        mode = 'trilinear',
                        align_corners = True
                    )
                )
            self.decoder.append(
                nn.BatchNorm3d(out_channels)
            )
            self.decoder.append(
            nn.SiLU()
            )
            in_channels = out_channels


        # Discriminator
        i: int = 1
        in_channels: int = self.in_channels
        out_channels: int = ch_mth(i)

        self.discriminator: nn.Sequential = nn.Sequential()

        if not self.encdis:
            self.discriminator.append(
                nn.Conv3d(
                        in_channels = in_channels,
                        out_channels = out_channels,
                        kernel_size = 3,
                        stride = 1,
                        padding = 1
                )
            )
            self.discriminator.append(nn.AlphaDropout(self.dropout_prob))
            self.discriminator.append(
                nn.BatchNorm3d(out_channels)
            )
            self.discriminator.append(
                nn.SiLU()
            )
            in_channels = out_channels

            module_init = None

            for i in range(2, ceil((1+self.num_layers)*1.333)):
                out_channels: int = ch_mth(i) if i<(self.layer_rep-1) or self.layer_rep==-1 else ch_mth(self.num_layers)
                module_init = nn.Conv3d(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = 3,
                    stride = 1,
                    padding = 1
                ) if i<self.layer_rep or self.layer_rep==-1 else module_init
                self.discriminator.append(
                    module_init
                )
                self.discriminator.append(nn.AlphaDropout(self.dropout_prob))
                if i < 6:
                    self.discriminator.append(
                        nn.BatchNorm3d(out_channels)
                    )
                    self.discriminator.append(
                        nn.SiLU()
                    )
                    self.discriminator.append(
                        nn.Conv3d(
                            in_channels = out_channels,
                            out_channels = out_channels,
                            kernel_size = 3,
                            stride = 2,
                            padding = 1
                        )
                    )
                    self.discriminator.append(nn.AlphaDropout(self.dropout_prob))
                self.discriminator.append(
                    nn.BatchNorm3d(out_channels)
                )
                self.discriminator.append(
                        nn.SiLU() if i != self.num_layers else nn.Sigmoid()
                )
                in_channels = out_channels
        else:
            self.discriminator.append(
                nn.Conv3d(
                        in_channels = self.embedding_dim,
                        out_channels = self.embedding_dim,
                        kernel_size = 3,
                        stride = 1,
                        padding = 1
                )
            )
            self.discriminator.append(
                nn.SiLU()
            )
            self.discriminator.append(nn.AlphaDropout(self.dropout_prob))
            self.discriminator.append(
                nn.Conv3d(
                        in_channels = self.embedding_dim,
                        out_channels = self.embedding_dim,
                        kernel_size = 3,
                        stride = 1,
                        padding = 1
                )
            )
            self.discriminator.append(
                nn.SiLU()
            )
            self.discriminator.append(nn.AlphaDropout(self.dropout_prob))

        self.discriminator_classifier: nn.Sequential = nn.Sequential(
            nn.Conv3d(
                    in_channels = self.embedding_dim,
                    out_channels = self.embedding_dim,
                    kernel_size = 3,
                    stride = 1,
                    padding = 1
            ),
            nn.SiLU(),
            nn.AlphaDropout(self.dropout_prob),
            nn.Conv3d(
                    in_channels = self.embedding_dim,
                    out_channels = self.embedding_dim,
                    kernel_size = 3,
                    stride = 1,
                    padding = 1
            ),
            nn.SiLU(),
            nn.AlphaDropout(self.dropout_prob),
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Flatten(),
            nn.Linear(self.embedding_dim,2),
            nn.Sigmoid(),
        )

        # Quantization Embeddings
        self.embeddings: nn.Module = nn.Embedding(self.num_embeddings, self.embedding_dim)

        # Classifier Layer Modules
        self.classifier: nn.Module = nn.Sequential(
            nn.Conv3d(
                    in_channels = self.embedding_dim,
                    out_channels = self.embedding_dim,
                    kernel_size = 3,
                    stride = 1,
                    padding = 1
            ),
            nn.AlphaDropout(self.dropout_prob),
            nn.SiLU(),
            nn.Conv3d(
                    in_channels = self.embedding_dim,
                    out_channels = self.embedding_dim,
                    kernel_size = 3,
                    stride = 1,
                    padding = 1
            ),
            nn.AlphaDropout(self.dropout_prob),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Flatten(),
            nn.Linear(self.embedding_dim,self.classes),
            # nn.Sigmoid(),

        )

        self.ckptfn = Ckptfn(self.use_ckpt)

        # VQ-VAE components
        vq_vae_loss = VQLoss(
                commitment_cost = self.commitment_cost
            )
        self.register_module("vq_vae_loss", vq_vae_loss)

        self.automatic_optimization = True
        self.register_buffer("class_weights",torch.tensor(
            self.config.get(
                "class_weights",
                (100/78.125,100/21.875)
                )
            ))
        self.register_buffer("log_cosh_a",torch.tensor((self.config.get("log_cosh_a",10),)))

        self.norm_dist = torch.distributions.Normal(0,1)

    def quantize(self, z_e: torch.tensor):
        distances = torch.cdist(z_e.unsqueeze(1), self.embeddings.weight.unsqueeze(0))
        indices = torch.argmin(distances.squeeze(1), dim=1)
        z_q = self.embeddings(indices)
        return z_q

    def infer_vae(self,x: torch.tensor) -> torch.tensor:
        # Generate Latents from the encoder
        latents = self.ckptfn(self.encoder,x)
        # Generate Class prediction(s) from the classifier
        y_hat = self.ckptfn(self.classifier,latents)
        # Process latents for quantization
        B,C,D,H,W = latents.size()
        ze = latents.permute(0,2,3,4,1).reshape(-1,C).contiguous()
        # Quantizing latents
        zq = self.quantize(ze)
        # Generating Probability distribution parameters from quantized latents
        # for decoder
        z_mu = self.enc_mu(zq)
        z_sigma = self.enc_sigma(zq)
        # Inverse transform of the quantized latents to accepted form
        quantized_latents = zq.reshape(B,D,H,W,C).permute(0,4,1,2,3).contiguous()
        z_mu = z_mu.reshape(B,D,H,W,C).permute(0,4,1,2,3).contiguous()
        z_sigma = z_sigma.reshape(B,D,H,W,C).permute(0,4,1,2,3).contiguous()
        #reparameterization trick
        z = z_mu + z_sigma * self.norm_dist.sample(z_mu.shape).to(x.device)
        # Concatenating class prediction tensor to input for decoder
        v, i = torch.topk(torch.softmax(y_hat,dim=-1),dim=-1,k=1)
        # v, _ = torch.topk(y_hat,dim=-1,k=1)
        i = F.one_hot(i.squeeze(-1),num_classes=self.classes)
        _y_hat = i*v
        z = z.permute(0,2,3,4,1).reshape(-1,C).contiguous()
        _y_hat = _y_hat.repeat(z.size()[0]//_y_hat.size()[0],1)
        _y_hat = torch.concat((z,_y_hat),dim=1)
        z_bar = self.embed_y(_y_hat).reshape(B,D,H,W,C).permute(0,4,1,2,3).contiguous()
        # Generating prediction image from the decoder
        x_recon = self.ckptfn(self.decoder,z_bar)
        x_hat = F.interpolate(x_recon,x.shape[2:])
        # Predicting classes from reconstructed image:
        # Generate reconstruction latents from the encoder
        latents_recon = self.ckptfn(self.encoder,x_hat)
        # Generate Class prediction(s) from the classifier for the
        # reconstructed image
        y_hat_recon = self.ckptfn(self.classifier,latents_recon)

        # Various losses for predicting
        # compute the KL divergence for the predicted prob. dist. paras.
        kl = torch.nan_to_num(z_sigma ** 2 + z_mu ** 2 - torch.log(torch.abs(z_sigma)) - 0.5).sum()
        # Log-Cosh Loss for reconstruction loss
        a = self.log_cosh_a
        difference = (x - x_hat)
        recon_loss = ((torch.log(torch.cosh(a*difference))).mean()/a)
        # Quantization Loss
        vq_vae_loss = self.vq_vae_loss(latents, quantized_latents)

        return x_hat,y_hat,y_hat_recon,quantized_latents,latents,kl,recon_loss,vq_vae_loss

    def forward(self,x: torch.Tensor, y: torch.Tensor ,batch_idx: int):
        B,C,D,H,W = x.size()

        x_hat,y_hat,y_hat_recon,quantized_latents,latents,kl,recon_loss,vq_vae_loss = self.infer_vae(x)

        fake_pred_tmp = self.ckptfn(self.encoder,fake_pred_tmp) if self.encdis else x_hat
        real_pred = self.ckptfn(self.encoder,real_pred) if self.encdis else x

        fake_pred = self.ckptfn(self.discriminator,fake_pred_tmp.clone().detach())
        fake_pred = self.ckptfn(self.discriminator_classifier,fake_pred).contiguous()

        with torch.no_grad():
            fake_pred_gan = self.ckptfn(self.discriminator,fake_pred_tmp)
            fake_pred_gan = self.ckptfn(self.discriminator_classifier,fake_pred_gan).contiguous()

        real_pred = self.ckptfn(self.discriminator,real_pred.clone().detach())
        real_pred = self.ckptfn(self.discriminator_classifier,real_pred).contiguous()


        lf = partial(F.binary_cross_entropy_with_logits,weight=self.class_weights)

        aux_cl = torch.log(torch.cosh(
            (
                (
                    (
                        torch.norm(torch.softmax(y_hat.mean(dim=0)*10,dim=-1))/torch.norm(torch.tensor((0.5,0.5)))
                        ) - 1
                    )**(1/(torch.pi**2))
                ) * 1.81227829293867844009326504039658921644118862 # solution for a (wolfram alpha) to: log(cosh(a * (|x|^(1/(pi^2))))) = y; where x = sqrt(2)-1, y = 1
            ))
        aux_cl = torch.nan_to_num(aux_cl,nan=0)
        classifier_loss = (lf(y_hat_recon,y)+lf(y_hat,y))/2

        ol = lambda x: (x,torch.ones_like(x,device=x.device,dtype=x.dtype))
        zl = lambda x: (x,torch.zeros_like(x,device=x.device,dtype=x.dtype))

        gan_loss = ((lf(*ol(fake_pred_gan))**B + lf(*zl(fake_pred_gan))**((-1)*B))/2)**(1/B)

        d_real_loss = lf(*ol(real_pred))
        d_fake_loss = lf(*zl(fake_pred))

        values, indices = torch.max(torch.softmax(y_hat,dim=-1),dim=-1)
        _, label = torch.max(y,dim=-1)
        results = torch._cast_Float(indices==label)
        conf = ((results*values).mean())*100
        acc = (results.mean())*100

        total_loss = (classifier_loss*(1-self.alpha) + (
            (recon_loss
             + vq_vae_loss
             + gan_loss
             + d_real_loss
             + d_fake_loss
            #  + kl
             )*self.alpha
            ))/6
        return classifier_loss,recon_loss,vq_vae_loss,gan_loss,d_real_loss,d_fake_loss,kl,total_loss,conf,acc

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int):
        self.train()
        x, y = batch

        sf = torch.nan_to_num((torch.tanh(torch.tensor((2 - (self.current_epoch/(self.epochs*self.pct_start/3)),),device=x.device))+1)/2)

        if batch_idx==-1:
            with torch.no_grad():
                for param in self.parameters():
                    param.add_(torch.nan_to_num(torch.randn_like(param,device=param.device)*self.noise_pct*sf))

        x = x + 0*torch.nan_to_num(torch.randn_like(x,device=x.device)*self.noise_pct*(1-sf))

        cl,rl,vql,gl,drl,dfl,kl,tl,conf,acc = self(x,y,batch_idx)

        self.log('classifier_loss', cl, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('classifier_confidence', conf, on_step=True, on_epoch=True, logger=True)
        self.log('classifier_acc', acc, on_step=True, on_epoch=True, logger=True)
        self.log('recon_loss', rl , on_step=True, on_epoch=True, logger=True)
        self.log('vq_vae_loss', vql, on_step=True, on_epoch=True, logger=True)
        self.log('gan_loss', gl, on_step=True, on_epoch=True, logger=True)
        self.log('d_real_loss', drl, on_step=True, on_epoch=True, logger=True)
        self.log('d_fake_loss', dfl, on_step=True, on_epoch=True, logger=True)
        self.log('KLDiv_Loss', kl, on_step=True, on_epoch=True, logger=True)
        self.log('total_train_loss', tl, on_epoch=True, on_step=True, logger=True)

        return tl

    def validation_step(self, batch, batch_idx):
        x,y = batch
        self.eval()
        cl,rl,vql,gl,drl,dfl,kl,tl,conf,acc = self(x,y,batch_idx)
        self.log('val_loss', cl, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_confidence', conf, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True, logger=True)
        return tl

    def test_step(self, batch, batch_idx):
        x,y = batch
        self.eval()
        cl,rl,vql,gl,drl,dfl,kl,tl,conf,acc = self(x,y,batch_idx)
        self.log('test_loss', cl, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_confidence', conf, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_epoch=True, prog_bar=True, logger=True)
        return tl

    def configure_optimizers(self):
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
        sample = torch.load(sample_path)
        if self.transform:
            sample = znormch(self.transform(sample))
        return torch.FloatTensor(sample), torch.FloatTensor(label)

class SuperDataset(Dataset):
    def __init__(self, root_dir="dataset", transform=None, data_split=["train","test","val"]):
        self.root_dir = os.path.join(os.curdir,root_dir)
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
root_dir = os.path.join(os.curdir,"dataset","") # ./dataset/
SuperSet = SuperDataset(root_dir=root_dir,transform=data_transform)
class_weights = SuperSet.get_class_weights()
num_classes = SuperSet.get_num_classes()
train_dataset = SuperSet["train"]
val_dataset = SuperSet["val"]
test_dataset = SuperSet["test"]

total_files = reduce(lambda x,y: x+y,class_weights.values(),0)
class_weights = [total_files/i for i in class_weights.values()]
# class_weights = tuple([i/reduce(lambda x,y: x+y,class_weights,0) for i in class_weights])

# Initialize Lightning model
input_dim = (1, 3, 80, 88, 88)
epochs = 100
batch_size = 2
accumulate_grad_batches=ceil(len(train_dataset)/batch_size)
config = {
    "in_channels" : input_dim[1],
    "classes" : num_classes,
    "epochs" : epochs,
    "lr":0.1,
    "num_steps" : epochs*len(train_dataset)//(accumulate_grad_batches*batch_size),
    "embedding_dim":32,
    "num_layers":8,
    "layer_repetition":-1,
    'num_embeddings':2,
    "pct_start":0.1,#7/batch_size,
    "div_factor":25,
    "final_div_factor":1e6,
    "alpha":0.9,
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
        checkpoint_callback,
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
        y_hat = model.infer_vae(x)[1]  # Extract the tensor containing predictions from the output tuple

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

# Save the model
torch.save(model.state_dict(), 'model.pth')

# Load the model
loaded_model = Model(config)
loaded_model.load_state_dict(torch.load('model.pth'))
loaded_model.eval()
