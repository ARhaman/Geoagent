"""
geoagent_model.py
==================
Pix2Pix architecture for bidirectional reservoir property translation.

Components:
  - UNetGenerator          : U-Net with skip connections (256x256)
  - PatchGANDiscriminator  : 70x70 PatchGAN discriminator
  - Pix2PixModel           : Wraps both + loss functions

Author: Abdulrahman Al-Fakih
"""

import torch
import torch.nn as nn


# ── Building Blocks ──────────────────────────────────────────────────────────

class ConvNormReLU(nn.Module):
    """Conv2d -> Norm -> Activation block."""
    def __init__(self, in_ch, out_ch, kernel=4, stride=2, padding=1,
                 norm="batch", activation="leaky", transpose=False):
        super().__init__()
        layers = []
        bias = (norm == "none")
        if transpose:
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding, bias=bias))
        else:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=bias))
        if norm == "batch":
            layers.append(nn.BatchNorm2d(out_ch))
        elif norm == "instance":
            layers.append(nn.InstanceNorm2d(out_ch))
        if activation == "leaky":
            layers.append(nn.LeakyReLU(0.2, inplace=False))
        elif activation == "relu":
            layers.append(nn.ReLU(inplace=False))
        elif activation == "tanh":
            layers.append(nn.Tanh())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ── U-Net Generator ──────────────────────────────────────────────────────────

class UNetGenerator(nn.Module):
    """
    U-Net generator for image-to-image translation.
    8 encoder + 8 decoder blocks with skip connections.
    Input/Output: (B, 3, 256, 256)
    """

    def __init__(self, in_channels=3, out_channels=3, ngf=64, norm="batch", dropout=True):
        super().__init__()

        # Encoder — halves spatial dims each step
        self.enc1 = nn.Conv2d(in_channels, ngf, 4, 2, 1, bias=False)    # 128
        self.enc2 = ConvNormReLU(ngf,    ngf*2, norm=norm)               # 64
        self.enc3 = ConvNormReLU(ngf*2,  ngf*4, norm=norm)               # 32
        self.enc4 = ConvNormReLU(ngf*4,  ngf*8, norm=norm)               # 16
        self.enc5 = ConvNormReLU(ngf*8,  ngf*8, norm=norm)               # 8
        self.enc6 = ConvNormReLU(ngf*8,  ngf*8, norm=norm)               # 4
        self.enc7 = ConvNormReLU(ngf*8,  ngf*8, norm=norm)               # 2
        self.enc8 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(ngf*8, ngf*8, 4, 2, 1, bias=False),               # 1
        )

        # Decoder — doubles spatial dims, skip-connects from encoder
        self.dec8 = self._dec(ngf*8,   ngf*8,  drop=True,  norm=norm)   # 2
        self.dec7 = self._dec(ngf*16,  ngf*8,  drop=True,  norm=norm)   # 4
        self.dec6 = self._dec(ngf*16,  ngf*8,  drop=True,  norm=norm)   # 8
        self.dec5 = self._dec(ngf*16,  ngf*8,  drop=False, norm=norm)   # 16
        self.dec4 = self._dec(ngf*16,  ngf*4,  drop=False, norm=norm)   # 32
        self.dec3 = self._dec(ngf*8,   ngf*2,  drop=False, norm=norm)   # 64
        self.dec2 = self._dec(ngf*4,   ngf,    drop=False, norm=norm)   # 128
        self.dec1 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(ngf*2, out_channels, 4, 2, 1, bias=False), # 256
            nn.Tanh(),
        )

    @staticmethod
    def _dec(in_ch, out_ch, drop=False, norm="batch"):
        layers = [
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
        ]
        if norm == "batch":
            layers.append(nn.BatchNorm2d(out_ch))
        elif norm == "instance":
            layers.append(nn.InstanceNorm2d(out_ch))
        if drop:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)

        d8 = self.dec8(e8)
        d7 = self.dec7(torch.cat([d8, e7], 1))
        d6 = self.dec6(torch.cat([d7, e6], 1))
        d5 = self.dec5(torch.cat([d6, e5], 1))
        d4 = self.dec4(torch.cat([d5, e4], 1))
        d3 = self.dec3(torch.cat([d4, e3], 1))
        d2 = self.dec2(torch.cat([d3, e2], 1))
        return self.dec1(torch.cat([d2, e1], 1))


# ── PatchGAN Discriminator ────────────────────────────────────────────────────

class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN: classifies 70x70 overlapping patches as real/fake.
    Input: concat(real_A, real_B_or_fake_B) → shape (B, 6, H, W)
    """

    def __init__(self, in_channels=6, ndf=64, n_layers=3, norm="batch"):
        super().__init__()
        NL = nn.BatchNorm2d if norm == "batch" else nn.InstanceNorm2d
        seq = [nn.Conv2d(in_channels, ndf, 4, 2, 1, bias=False),
               nn.LeakyReLU(0.2, inplace=False)]
        ch = ndf
        for n in range(1, n_layers):
            ch_prev, ch = ch, min(ch * 2, 512)
            stride = 1 if n == n_layers - 1 else 2
            seq += [nn.Conv2d(ch_prev, ch, 4, stride, 1, bias=False), NL(ch),
                    nn.LeakyReLU(0.2, inplace=False)]
        ch2 = min(ch * 2, 512)
        seq += [nn.Conv2d(ch, ch2, 4, 1, 1, bias=False), NL(ch2),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(ch2, 1, 4, 1, 1)]
        self.model = nn.Sequential(*seq)

    def forward(self, inp, tgt):
        return self.model(torch.cat([inp, tgt], 1))


# ── Full Pix2Pix Model ────────────────────────────────────────────────────────

class Pix2PixModel:
    """
    Pix2Pix training wrapper (not an nn.Module — manages two networks + optimizers).

    Usage:
        model = Pix2PixModel(device=device)
        losses = model.train_step(real_A, real_B)
        fake_B = model.predict(real_A)
    """

    def __init__(
        self,
        device,
        in_channels: int = 3,
        out_channels: int = 3,
        ngf: int = 64,
        ndf: int = 64,
        lambda_l1: float = 100.0,
        norm: str = "batch",
        lr: float = 2e-4,
        beta1: float = 0.5,
    ):
        self.device = device
        self.lambda_l1 = lambda_l1

        self.G = UNetGenerator(in_channels, out_channels, ngf, norm).to(device)
        self.D = PatchGANDiscriminator(in_channels + out_channels, ndf, norm=norm).to(device)

        self.criterion_GAN = nn.BCEWithLogitsLoss()
        self.criterion_L1  = nn.L1Loss()

        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(beta1, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(beta1, 0.999))

        # LR schedulers (linear decay after epoch 100)
        self.sched_G = torch.optim.lr_scheduler.LambdaLR(
            self.opt_G, lr_lambda=lambda e: max(0.0, 1.0 - max(0, e - 100) / 100.0))
        self.sched_D = torch.optim.lr_scheduler.LambdaLR(
            self.opt_D, lr_lambda=lambda e: max(0.0, 1.0 - max(0, e - 100) / 100.0))

        self._init_weights()

    def _init_weights(self):
        for m in list(self.G.modules()) + list(self.D.modules()):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _requires_grad(self, net, flag: bool):
        for p in net.parameters():
            p.requires_grad = flag

    def train_step(self, real_A: torch.Tensor, real_B: torch.Tensor) -> dict:
        """
        One full forward + backward pass.
        real_A: input  (B, 3, 256, 256) in [-1, 1]
        real_B: target (B, 3, 256, 256) in [-1, 1]
        Returns dict of scalar losses.
        """
        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)
        fake_B = self.G(real_A)

        # Update D
        self._requires_grad(self.D, True)
        self.opt_D.zero_grad()
        pred_fake = self.D(real_A, fake_B.detach())
        pred_real = self.D(real_A, real_B)
        loss_D = 0.5 * (
            self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake)) +
            self.criterion_GAN(pred_real,  torch.ones_like(pred_real))
        )
        loss_D.backward()
        self.opt_D.step()

        # Update G
        self._requires_grad(self.D, False)
        self.opt_G.zero_grad()
        pred_fake_G = self.D(real_A, fake_B)
        loss_G_gan = self.criterion_GAN(pred_fake_G, torch.ones_like(pred_fake_G))
        loss_G_l1  = self.criterion_L1(fake_B, real_B) * self.lambda_l1
        loss_G = loss_G_gan + loss_G_l1
        loss_G.backward()
        self.opt_G.step()

        return {
            "loss_G": loss_G.item(),
            "loss_G_gan": loss_G_gan.item(),
            "loss_G_l1": loss_G_l1.item(),
            "loss_D": loss_D.item(),
        }

    @torch.no_grad()
    def predict(self, inp: torch.Tensor) -> torch.Tensor:
        """Inference — used by GeoAgent. Returns tensor in [-1, 1]."""
        self.G.eval()
        return self.G(inp.to(self.device))

    def step_schedulers(self):
        self.sched_G.step()
        self.sched_D.step()

    def save(self, path: str):
        torch.save({"G": self.G.state_dict(), "D": self.D.state_dict()}, path)
        print(f"[Pix2PixModel] Saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt["G"])
        self.D.load_state_dict(ckpt["D"])
        print(f"[Pix2PixModel] Loaded ← {path}")


def count_params(model) -> str:
    nets = [model.G, model.D] if hasattr(model, "G") else [model]
    names = ["G", "D"] if hasattr(model, "G") else ["model"]
    out = []
    for n, net in zip(names, nets):
        total = sum(p.numel() for p in net.parameters())
        out.append(f"{n}: {total/1e6:.2f}M params")
    return "  |  ".join(out)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Pix2PixModel(device=device)
    print(count_params(model))

    x = torch.randn(2, 3, 256, 256)
    y = torch.randn(2, 3, 256, 256)
    losses = model.train_step(x, y)
    print("Losses:", {k: f"{v:.4f}" for k, v in losses.items()})
    print("Model OK.")
