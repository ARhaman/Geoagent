"""Model definition copied verbatim from the training notebook (GeoAgent_2D_Fixed.ipynb, cell 7).
The checkpoints use these layer names (e1.., d1..), not the ones in geoagent_model.py."""
import torch
import torch.nn as nn

class ConvNormReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=4, stride=2, padding=1,
                 norm="batch", activation="leaky", transpose=False):
        super().__init__()
        layers = []
        bias = (norm == "none")
        if transpose:
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding, bias=bias))
        else:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=bias))
        if norm == "batch":      layers.append(nn.BatchNorm2d(out_ch))
        elif norm == "instance": layers.append(nn.InstanceNorm2d(out_ch))
        if activation == "leaky": layers.append(nn.LeakyReLU(0.2, inplace=False))
        elif activation == "relu": layers.append(nn.ReLU(inplace=False))
        elif activation == "tanh": layers.append(nn.Tanh())
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)

class UNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ngf=64, norm="batch"):
        super().__init__()
        # Encoder
        self.e1 = nn.Conv2d(in_ch, ngf, 4, 2, 1, bias=False)
        self.e2 = ConvNormReLU(ngf,   ngf*2, norm=norm)
        self.e3 = ConvNormReLU(ngf*2, ngf*4, norm=norm)
        self.e4 = ConvNormReLU(ngf*4, ngf*8, norm=norm)
        self.e5 = ConvNormReLU(ngf*8, ngf*8, norm=norm)
        self.e6 = ConvNormReLU(ngf*8, ngf*8, norm=norm)
        self.e7 = ConvNormReLU(ngf*8, ngf*8, norm=norm)
        self.e8 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(ngf*8, ngf*8, 4, 2, 1, bias=False))
        # Decoder
        self.d8 = self._dec(ngf*8,  ngf*8,  drop=True,  norm=norm)
        self.d7 = self._dec(ngf*16, ngf*8,  drop=True,  norm=norm)
        self.d6 = self._dec(ngf*16, ngf*8,  drop=True,  norm=norm)
        self.d5 = self._dec(ngf*16, ngf*8,  drop=False, norm=norm)
        self.d4 = self._dec(ngf*16, ngf*4,  drop=False, norm=norm)
        self.d3 = self._dec(ngf*8,  ngf*2,  drop=False, norm=norm)
        self.d2 = self._dec(ngf*4,  ngf,    drop=False, norm=norm)
        self.d1 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(ngf*2, out_ch, 4, 2, 1, bias=False),
            nn.Tanh())
    @staticmethod
    def _dec(in_ch, out_ch, drop=False, norm="batch"):
        layers = [nn.ReLU(inplace=False),
                  nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        if norm == "batch":      layers.append(nn.BatchNorm2d(out_ch))
        elif norm == "instance": layers.append(nn.InstanceNorm2d(out_ch))
        if drop: layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)
    def forward(self, x):
        e1=self.e1(x); e2=self.e2(e1); e3=self.e3(e2); e4=self.e4(e3)
        e5=self.e5(e4); e6=self.e6(e5); e7=self.e7(e6); e8=self.e8(e7)
        d8=self.d8(e8)
        d7=self.d7(torch.cat([d8,e7],1))
        d6=self.d6(torch.cat([d7,e6],1))
        d5=self.d5(torch.cat([d6,e5],1))
        d4=self.d4(torch.cat([d5,e4],1))
        d3=self.d3(torch.cat([d4,e3],1))
        d2=self.d2(torch.cat([d3,e2],1))
        return self.d1(torch.cat([d2,e1],1))

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_ch=6, ndf=64, n_layers=3, norm="batch"):
        super().__init__()
        NL = nn.BatchNorm2d if norm == "batch" else nn.InstanceNorm2d
        seq = [nn.Conv2d(in_ch,ndf,4,2,1,bias=False), nn.LeakyReLU(0.2,inplace=False)]
        ch = ndf
        for n in range(1, n_layers):
            ch_prev, ch = ch, min(ch*2, 512)
            stride = 1 if n == n_layers-1 else 2
            seq += [nn.Conv2d(ch_prev,ch,4,stride,1,bias=False), NL(ch),
                    nn.LeakyReLU(0.2,inplace=False)]
        ch2 = min(ch*2, 512)
        seq += [nn.Conv2d(ch,ch2,4,1,1,bias=False), NL(ch2),
                nn.LeakyReLU(0.2,inplace=False),
                nn.Conv2d(ch2,1,4,1,1)]
        self.model = nn.Sequential(*seq)
    def forward(self, inp, tgt):
        return self.model(torch.cat([inp,tgt],1))

class Pix2PixModel:
    def __init__(self, device, in_ch=3, out_ch=3, ngf=64, ndf=64,
                 lambda_l1=100.0, norm="batch", lr=2e-4, beta1=0.5):
        self.device = device
        self.lambda_l1 = lambda_l1
        self.G = UNetGenerator(in_ch, out_ch, ngf, norm).to(device)
        self.D = PatchGANDiscriminator(in_ch+out_ch, ndf, norm=norm).to(device)
        self.cGAN = nn.BCEWithLogitsLoss()
        self.cL1  = nn.L1Loss()
        self.oG = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(beta1,0.999))
        self.oD = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(beta1,0.999))
        self.sG = torch.optim.lr_scheduler.LambdaLR(
            self.oG, lambda e: max(0.0, 1.0-max(0,e-100)/100.0))
        self.sD = torch.optim.lr_scheduler.LambdaLR(
            self.oD, lambda e: max(0.0, 1.0-max(0,e-100)/100.0))
        self._init_weights()
    def _init_weights(self):
        for m in list(self.G.modules())+list(self.D.modules()):
            if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d)):
                nn.init.normal_(m.weight,0.0,0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m,(nn.BatchNorm2d,nn.InstanceNorm2d)):
                if m.weight is not None: nn.init.normal_(m.weight,1.0,0.02)
                if m.bias  is not None: nn.init.zeros_(m.bias)
    def _rg(self, net, flag):
        for p in net.parameters(): p.requires_grad = flag
    def train_step(self, rA, rB):
        rA, rB = rA.to(self.device), rB.to(self.device)
        fB = self.G(rA)
        self._rg(self.D, True); self.oD.zero_grad()
        pf = self.D(rA, fB.detach())
        pr = self.D(rA, rB)
        lD = 0.5*(self.cGAN(pf,torch.zeros_like(pf))+self.cGAN(pr,torch.ones_like(pr)))
        lD.backward(); self.oD.step()
        self._rg(self.D, False); self.oG.zero_grad()
        pg  = self.D(rA, fB)
        lGg = self.cGAN(pg, torch.ones_like(pg))
        lGl = self.cL1(fB, rB) * self.lambda_l1
        lG  = lGg + lGl; lG.backward(); self.oG.step()
        return {"loss_G":lG.item(),"loss_G_gan":lGg.item(),
                "loss_G_l1":lGl.item(),"loss_D":lD.item()}
    @torch.no_grad()
    def predict(self, x):
        self.G.eval()
        return self.G(x.to(self.device))
    def step_schedulers(self): self.sG.step(); self.sD.step()
    def save(self, path):
        torch.save({"G":self.G.state_dict(),"D":self.D.state_dict()}, path)
        print("Saved:", path)
    def load(self, path):
        c = torch.load(path, map_location=self.device)
        self.G.load_state_dict(c["G"]); self.D.load_state_dict(c["D"])
        print("Loaded:", path)

def count_params(model):
    return "  |  ".join(
        "{}: {:.2f}M params".format(n, sum(p.numel() for p in net.parameters())/1e6)
        for n, net in [("G", model.G), ("D", model.D)])

