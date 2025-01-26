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
class VQVAEGAN(pl.LightningModule):
    def __init__(self, config: dict) -> None:
        super(VQVAEGAN, self).__init__()

        self.config: dict = config

        self.embedding_dim: int = self.config.get('embedding_dim',32)
        self.num_embeddings: int = self.config.get('num_embeddings',2)
        self.commitment_cost: float = self.config.get('commitment_cost',0.33)
        self.num_layers: int = max(5,self.config.get('num_layers',8))
        self.in_channels: int = self.config.get('in_channels',2)
        self.min_dim: int = max(ceil(self.in_channels*0.66),self.config.get('min_dim',4))
        self.classes: int = self.config.get("classes",2)
        self.lr = self.config.get('lr',4)
        self.epochs = self.config.get("epochs",10)
        self.data_shape = self.config.get("shape",(80,88,88))
        self.seperate_discriminator = self.config.get("seperate_discriminator",True)
        self.batch_size = self.config.get("batch_size",1)
        self.dropout_prob = self.config.get("dropout_prob",0.666)
        self.alpha = self.config.get("alpha",0.99)
        self.num_steps = self.config.get("num_steps",100)
        self.layer_rep = self.config.get("layer_repetition",6)
        self.layer_rep = max(self.layer_rep,6) if self.layer_rep != -1 else -1
        self.pct_start = self.config.get("pct_start",0.3)
        self.divf = self.config.get("div_factor",25)
        self.fdivf = self.config.get("final_div_factor",1e6)
        self.noise_percentage = self.config.get("noise_percentage",1)


        # Encoder, Decoder, Discriminator

        ch_mth: function = lambda i: int (min(
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
                nn.LeakyReLU(0.2)
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
                        nn.Tanh()
                    )
                self.encoder.append(
                        nn.Conv3d(
                            in_channels = out_channels,
                            out_channels = out_channels,
                            kernel_size = (3,3,3),
                            stride = (2,2,2),
                            padding = (1,1,1)
                        )
                    )
                self.encoder.append(nn.AlphaDropout(self.dropout_prob))
            self.encoder.append(
                    nn.BatchNorm3d(out_channels)
                )
            self.encoder.append(
                    nn.Tanh()
                )
            in_channels = out_channels

        self.enc_mu = nn.Linear(in_features=out_channels,out_features=1)
        self.enc_sigma = nn.Linear(in_features=out_channels,out_features=1)
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
                nn.Tanh()
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
                    ) if (i > (self.num_layers-self.layer_rep)) or self.layer_rep==-1 else module_init
            self.decoder.append(
                    module_init
                )
            self.decoder.append(nn.AlphaDropout(self.dropout_prob))
            if (i > (self.num_layers-4)):
                # self.decoder.append(
                #         nn.ConvTranspose3d(
                #             in_channels = out_channels,
                #             out_channels = out_channels,
                #             kernel_size = 5,
                #             stride = 2,
                #             padding = 2
                #         )
                #     )
                self.decoder.append(
                    nn.Upsample(scale_factor=(2,2,2),
                                mode='trilinear',
                                align_corners=True
                                )
                )
            if (i==(self.num_layers-4)):
                self.decoder.append(
                    nn.Upsample(size = self.data_shape,
                                mode = 'trilinear',
                                align_corners = True
                                )
                )
            self.decoder.append(
                    nn.BatchNorm3d(out_channels)
                )
            self.decoder.append(
                nn.Tanh()
                )
            in_channels = out_channels


        # Discriminator
        i: int = 1
        in_channels: int = self.in_channels
        out_channels: int = ch_mth(i)

        self.discriminator: nn.Sequential = nn.Sequential()

        if self.seperate_discriminator:
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
                    nn.LeakyReLU(0.2)
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
                            nn.Tanh()
                        )
                    self.discriminator.append(
                            nn.Conv3d(
                                in_channels = out_channels,
                                out_channels = out_channels,
                                kernel_size = (3,3,3),
                                stride = (1,2,2),
                                padding = (1,1,1)
                            )
                        )
                    self.discriminator.append(nn.AlphaDropout(self.dropout_prob))
                self.discriminator.append(
                        nn.BatchNorm3d(out_channels)
                    )
                self.discriminator.append(
                            nn.Tanh() if i != self.num_layers else nn.Sigmoid()
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
                nn.Tanh()
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
                nn.Tanh()
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
            nn.Tanh(),
            nn.AlphaDropout(self.dropout_prob),
            nn.Conv3d(
                    in_channels = self.embedding_dim,
                    out_channels = self.embedding_dim,
                    kernel_size = 3,
                    stride = 1,
                    padding = 1
            ),
            nn.Tanh(),
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
            nn.Tanh(),
            nn.Conv3d(
                    in_channels = self.embedding_dim,
                    out_channels = self.embedding_dim,
                    kernel_size = 3,
                    stride = 1,
                    padding = 1
            ),
            nn.AlphaDropout(self.dropout_prob),
            nn.Tanh(),
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Flatten(),
            nn.Linear(self.embedding_dim,self.classes),
            nn.Sigmoid(),
        )

        self.ckptfn = Ckptfn(use_ckpt)

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
        self.register_buffer("log_cosh_a",torch.tensor((self.config.get("log_cosh_a",1),)))

        self.norm_dist = torch.distributions.Normal(0,1)

    def quantize(self, z_e: torch.tensor):
        distances = torch.cdist(z_e.unsqueeze(1), self.embeddings.weight.unsqueeze(0))
        indices = torch.argmin(distances.squeeze(1), dim=1)
        z_q = self.embeddings(indices)
        return z_q

    def dep_forward(self, x: torch.tensor):
        z_e = torch.utils.checkpoint.checkpoint_sequential(self.encoder,self.num_layers,x,use_reentrant = False) if use_ckpt else self.encoder(x)
        ze = z_e
        B, C, D, H, W = z_e.size()
        z_e = z_e.permute(0,2,3,4,1).contiguous().view(-1, C)
        z_q = self.quantize(z_e)
        z_q = z_q.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        x_recon = torch.utils.checkpoint.checkpoint_sequential(self.decoder,self.num_layers,z_q,use_reentrant = False) if use_ckpt else self.decoder(z_q)
        return (x_recon,z_q,ze)

    def infer_vae(self,x: torch.tensor) -> torch.tensor:
        # Generate Latents from the encoder
        latents = self.ckptfn(self.encoder,x)
        # Generate Class prediction(s) from the classifier
        y_hat = self.ckptfn(self.classifier,latents)
        # Process latents for quantization
        B,C,D,H,W = latents.size()
        ze = latents.permute(0,2,3,4,1).view(-1,C).contiguous()
        # Quantizing latents
        zq = self.quantize(ze)
        # Generating Probability distribution parameters from quantized latents
        # for decoder
        z_mu = self.enc_mu(quantized_latents)
        z_sigma = self.enc_sigma(quantized_latents)
        # Inverse transform of the quantized latents to accepted form
        quantized_latents = zq.view(B,D,H,W,C).permute(0,4,1,2,3).contiguous()
        z_mu = z_mu.view(B,D,H,W,C).permute(0,4,1,2,3).contiguous()
        z_sigma = z_sigma.view(B,D,H,W,C).permute(0,4,1,2,3).contiguous()
        #reparameterization trick
        z = z_mu + z_sigma * self.norm_dist.sample(z_mu.shape).to(x.device)
        # Concatenating class prediction tensor to input for decoder
        _, i = torch.topk(torch.softmax(y_hat,dim=-1),dim=-1,k=1)
        v, _ = torch.topk(y_hat,dim=-1,k=1)
        i = F.one_hot(i,num_classes=self.classes)
        _y_hat = i*(v.unsqueeze(1))
        _y_hat = _y_hat.view(*_y_hat.size(),*tuple([1 for i in z.size()[2:]])).repeat(1,1,*z.size()[2:])
        z_bar = self.embed_y(torch.concat((z,_y_hat),dim=1))
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
        kl = (z_sigma ** 2 + z_mu ** 2
                    - torch.log(z_sigma) - 0.5).sum()
        # Log-Cosh Loss for reconstruction loss
        a = self.log_cosh_a
        difference = znormch(x - x_hat)
        difference = difference / (torch.max(difference) - torch.min(difference))
        difference = difference - torch.min(difference)
        difference = difference*2 - 1
        recon_loss = ((torch.log(torch.cosh(a*difference))).mean()/a)
        # Quantization Loss
        vq_vae_loss = self.vq_vae_loss(latents, quantized_latents)

        return x_hat,y_hat,y_hat_recon,quantized_latents,latents,kl,recon_loss,vq_vae_loss

    def forward(self,x: torch.Tensor, y: torch.Tensor ,batch_idx: int):
        B,C,D,H,W = x.size()

        x_hat,y_hat,y_hat_recon,quantized_latents,latents,kl,recon_loss,vq_vae_loss = self.infer_vae(x)

        if not self.seperate_discriminator:
            fake_pred_tmp = self.ckptfn(self.encoder,x_hat)

            fake_pred = self.ckptfn(self.discriminator,fake_pred_tmp)
            fake_pred = self.ckptfn(self.discriminator_classifier,fake_pred).contiguous()

            with torch.no_grad():
                fake_pred_gan = self.ckptfn(self.discriminator,fake_pred_tmp)
                fake_pred_gan = self.ckptfn(self.discriminator_classifier,fake_pred_gan).contiguous()

            real_pred = self.ckptfn(self.encoder,x)
            real_pred = self.ckptfn(self.discriminator,real_pred)
            real_pred = self.ckptfn(self.discriminator_classifier,real_pred).contiguous()
        else:
            fake_pred = self.ckptfn(self.discriminator,x_hat)
            fake_pred = self.ckptfn(self.discriminator_classifier,fake_pred).contiguous()

            with torch.no_grad():
                fake_pred_gan = self.ckptfn(self.discriminator,x_hat)
                fake_pred_gan = self.ckptfn(self.discriminator_classifier,fake_pred_gan).contiguous()

            real_pred = self.ckptfn(self.discriminator,x)
            real_pred = self.ckptfn(self.discriminator_classifier,real_pred).contiguous()

        lf = partial(F.binary_cross_entropy_with_logits,weights=self.class_weights)

        classifier_loss = (lf(y_hat_recon,y)+lf(y_hat,y))/2

        ol = lambda x: (x,torch.ones_like(x,device=x.device,dtype=x.dtype))
        zl = lambda x: (x,torch.zeros_like(x,device=x.device,dtype=x.dtype))

        gan_loss = ((lf(*ol(fake_pred_gan))**B + lf(*zl(fake_pred_gan))**((-1)*B))/2)**(1/B)

        d_real_loss = lf(*ol(real_pred))
        d_fake_loss = lf(*zl(fake_pred))

        values, indices = torch.topk(torch.softmax(y_hat,dim=-1),dim=-1,k=1)
        _, label = torch.topk(y,dim=-1,k=1)
        results = torch._cast_Float(indices==label)
        conf = ((results*values).mean())*100
        acc = (results.mean())*100

        total_loss = (classifier_loss*(1-self.alpha) + (
            (recon_loss
             + vq_vae_loss
             + gan_loss
             + d_real_loss
             + d_fake_loss
             + kl
             )*self.alpha
            ))/7
        return classifier_loss,recon_loss,vq_vae_loss,gan_loss,d_real_loss,d_fake_loss,kl,total_loss,conf,acc

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int):
        self.train()
        x, y = batch
        with torch.no_grad():
            for param in self.parameters():
                param.add_(torch.randn(param.size(),device=param.device) * self.noise_percentage/(100))
        mu,sigma = torch.std_mean(x)
        x = x + torch.normal(mu,sigma**2)*self.noise_percentage/100
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

    def dep_training_step(self, batch: tuple, batch_idx: int):
        x, y = batch
        self.train()
        with torch.no_grad():
            for param in self.parameters():
                param.add_(torch.randn(param.size(),device=param.device) * self.noise_percentage/(100))
        mu,sigma = torch.std_mean(x)
        inp = x + torch.normal(mu,sigma**2)*self.noise_percentage/100
        x_recon, z_q, z_e = self(inp)
        x_recon = torch.nn.functional.interpolate(x_recon, x.shape[2:])

        pred_recon = torch.utils.checkpoint.checkpoint_sequential(
            self.discriminator,
            len(self.discriminator),
            x_recon,
            use_reentrant = False
            ).contiguous() if use_ckpt else self.discriminator(x_recon).contiguous()
        pred_recon = self.classifier(pred_recon)

        pred = torch.utils.checkpoint.checkpoint_sequential(
            self.discriminator,
            len(self.discriminator),
            x_recon,
            use_reentrant = False
            ).contiguous() if use_ckpt else self.discriminator(x_recon).contiguous()
        pred = self.classifier(pred)

        if not self.seperate_discriminator:
            fake_pred_tmp = torch.utils.checkpoint.checkpoint_sequential(
                self.encoder,
                self.num_layers,
                x_recon,
                use_reentrant = False
                ) if use_ckpt else self.encoder(x_recon)

            fake_pred = torch.utils.checkpoint.checkpoint_sequential(
                self.discriminator,
                len(self.discriminator),
                fake_pred_tmp.detach(),
                use_reentrant = False
                ).contiguous() if use_ckpt else self.discriminator(fake_pred_tmp.detach()).contiguous()
            fake_pred = self.discriminator_classifier(fake_pred).contiguous()

            with torch.no_grad():
                fake_pred_gan = torch.utils.checkpoint.checkpoint_sequential(
                    self.discriminator,
                    len(self.discriminator),
                    fake_pred_tmp,
                    use_reentrant = False
                    ).contiguous() if use_ckpt else self.discriminator(fake_pred_tmp).contiguous()
                fake_pred_gan = self.discriminator_classifier(fake_pred_gan).contiguous()

            real_pred = torch.utils.checkpoint.checkpoint_sequential(
                self.encoder,
                self.num_layers,
                x,
                use_reentrant = False
                ) if use_ckpt else self.encoder(x)
            real_pred = torch.utils.checkpoint.checkpoint_sequential(
                self.discriminator,
                len(self.discriminator),
                real_pred,
                use_reentrant = False
                ).contiguous() if use_ckpt else self.discriminator(real_pred).contiguous()
            real_pred = self.discriminator_classifier(real_pred).contiguous()
        else:
            fake_pred = torch.utils.checkpoint.checkpoint_sequential(
                self.discriminator,
                len(self.discriminator),
                x_recon.detach(),
                use_reentrant = False
                ) if use_ckpt else self.discriminator(x_recon.detach())
            fake_pred = self.discriminator_classifier(fake_pred).contiguous()

            with torch.no_grad():
                fake_pred_gan = torch.utils.checkpoint.checkpoint_sequential(
                    self.discriminator,
                    len(self.discriminator),
                    x_recon,
                    use_reentrant = False
                    ) if use_ckpt else self.discriminator(x_recon)
                fake_pred_gan = self.discriminator_classifier(fake_pred_gan).contiguous()

            real_pred = torch.utils.checkpoint.checkpoint_sequential(
                self.discriminator,
                len(self.discriminator),
                x,
                use_reentrant = False
                ).contiguous() if use_ckpt else self.discriminator(x).contiguous()
            real_pred = self.discriminator_classifier(real_pred).contiguous()


        classifier_loss = (
            F.binary_cross_entropy_with_logits(
                pred_recon, y,weight = self.class_weights
                ) + F.binary_cross_entropy_with_logits(
                    pred, y,weight = self.class_weights
                    )
            )/2

        a = self.log_cosh_a

        difference = znormch(x - x_recon)
        difference = difference / (torch.max(difference) - torch.min(difference))
        difference = difference - torch.min(difference)
        difference = difference*2 - 1
        recon_loss = ((torch.log(torch.cosh(a*difference))).mean()/a)
        vq_vae_loss = self.vq_vae_loss(z_e, z_q)

        gan_loss = torch.pow(
            (torch.pow(
                F.binary_cross_entropy_with_logits(
                    fake_pred_gan,
                    torch.ones_like(fake_pred_gan),
                    weight = self.class_weights
                    ),
                torch.tensor((int(x.shape[0]),),device=x.device)
                ) + torch.pow(
                        F.binary_cross_entropy_with_logits(
                            fake_pred_gan,
                            torch.zeros_like(fake_pred_gan),
                            weight = self.class_weights
                            ),
                        torch.tensor(((-1)*int(x.shape[0]),),device=x.device)
                    ))/2,
            torch.tensor((int(x.shape[0]) ** -1,),device=x.device)
        )
        d_real_loss = F.binary_cross_entropy_with_logits(
            real_pred,
            torch.ones_like(real_pred),
            weight = self.class_weights
            )
        d_fake_loss = F.binary_cross_entropy_with_logits(
            fake_pred,
            torch.zeros_like(fake_pred),
            weight = self.class_weights
            )

        total_loss = classifier_loss*(1-self.alpha) + (
            (recon_loss
             + vq_vae_loss
             + gan_loss
             + d_real_loss
             + d_fake_loss
             )*self.alpha
            )

        self.log('classifier_loss', classifier_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('recon_loss', recon_loss , on_step=True, on_epoch=True, logger=True)
        self.log('vq_vae_loss', vq_vae_loss, on_step=True, on_epoch=True, logger=True)
        self.log('gan_loss', gan_loss, on_step=True, on_epoch=True, logger=True)
        self.log('d_real_loss', d_real_loss, on_step=True, on_epoch=True, logger=True)
        self.log('d_fake_loss', d_fake_loss, on_step=True, on_epoch=True, logger=True)
        self.log('total_train_loss', total_loss/6, on_epoch=True, on_step=True, logger=True)

        return classifier_loss

    def inf_fn(self,batch,batch_idx):
        x, y = batch
        self.eval()
        with torch.no_grad():
            #x_recon, z_q, z_e = self(x)
            y_hat = torch.utils.checkpoint.checkpoint_sequential(
                self.discriminator,
                len(self.discriminator),
                x,
                use_reentrant = False
                ).contiguous() if use_ckpt else self.discriminator(x).contiguous()
            y_hat = self.classifier(y_hat)
            loss = F.binary_cross_entropy_with_logits(y_hat, y, weight = self.class_weights)
            values, indices = torch.topk(torch.softmax(y_hat,dim=-1),dim=-1,k=1)
            _, label = torch.topk(y,dim=-1,k=1)
            results = torch._cast_Float(indices==label)
            acc = ((results*values).mean())*100
            acc_raw = (results.mean())*100
        return loss,acc,acc_raw

    def validation_step(self, batch, batch_idx):
        # loss, acc, acc_raw = self.inf_fn(batch,batch_idx)
        x,y = batch
        cl,rl,vql,gl,drl,dfl,kl,tl,conf,acc = self(x,y,batch_idx)
        self.log('val_loss', cl, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_confidence', conf, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True, logger=True)
        return tl

    def test_step(self, batch, batch_idx):
        # loss, acc, acc_raw = self.inf_fn(batch,batch_idx)
        x,y = batch
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
            base_momentum=self.config.get("base_momentum",0.3),
            max_momentum=max(max(self.config.get("betas",(0.9,0.999))),self.config.get("max_momentum",0.99)),
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
        return len(self.samples)

    def get_class_weights(self):
        return self.class_weights

    def get_num_classes(self):
        return max(self.class_to_idx.values())+1

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
class_weights = tuple([i/total_files for i in class_weights.values()])

# Initialize Lightning model
input_dim = (1, 3, 80, 88, 88)
epochs = 100
batch_size = 4
config = {
    "in_channels" : input_dim[1],
    "classes" : num_classes,
    "epochs" : epochs,
    "lr":0.1,
    "num_steps" : ceil(128/batch_size),
    "embedding_dim":8,
    "num_layers":12,
    "layer_repetition":-1,
    'num_embeddings':16,
    "pct_start":0.1,#7/batch_size,
    "alpha":0.5,
    "weight_decay":10,
    "betas":(0.9,0.999),
    "amsgrad":True,
    "class_weights":class_weights,
}
model = VQVAEGAN(config)

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
    log_every_n_steps=1,
    num_nodes=1,
    precision=32,  # Automatic Mixed Precision (AMP)
    accumulate_grad_batches=ceil(128/batch_size),  # Gradient Accumulation
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

