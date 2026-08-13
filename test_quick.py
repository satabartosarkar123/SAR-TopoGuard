import torch, sys, traceback
sys.path.insert(0, ".")
from dataset import SEN12MS_Dataset
from models import UNetGenerator, PatchGANDiscriminator
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn as nn

print("imports ok")

s1_files = sorted([p for p in Path("mini_sen12_data/train/s1").glob("*.*") if p.suffix.lower() in [".tif", ".png"]])[:8]
records = [{"sar_path": str(p), "optical_path": str(Path("mini_sen12_data/train/s2") / p.name), "roi": "mini", "patch_id": p.stem} for p in s1_files if (Path("mini_sen12_data/train/s2") / p.name).exists()]
ds = SEN12MS_Dataset(records=records, patch_size=256)
loader = DataLoader(ds, batch_size=4, num_workers=0)
print(f"dataset: {len(ds)}")

batch = next(iter(loader))
print(f"sar: {batch['sar'].shape} range=[{batch['sar'].min():.3f},{batch['sar'].max():.3f}]")
print(f"opt: {batch['optical'].shape} range=[{batch['optical'].min():.3f},{batch['optical'].max():.3f}]")

netG = UNetGenerator(in_channels=2, out_channels=3).cuda()
netD = PatchGANDiscriminator(in_channels=5).cuda()
real_A = batch["sar"].cuda()
real_B = batch["optical"].cuda()
fake_B = netG(real_A)
print(f"fake: {fake_B.shape} range=[{fake_B.min():.3f},{fake_B.max():.3f}]")

# Try a training step
optG = torch.optim.Adam(netG.parameters(), lr=1e-4)
optD = torch.optim.Adam(netD.parameters(), lr=1e-4)
criterionGAN = nn.BCEWithLogitsLoss()
criterionL1 = nn.L1Loss()

optD.zero_grad()
pred_fake = netD(torch.cat([real_A, fake_B.detach()], 1))
loss_D_fake = criterionGAN(pred_fake, torch.zeros_like(pred_fake))
pred_real = netD(torch.cat([real_A, real_B], 1))
loss_D_real = criterionGAN(pred_real, torch.full_like(pred_real, 0.9))
loss_D = (loss_D_fake + loss_D_real) * 0.5
loss_D.backward()
optD.step()

optG.zero_grad()
pred_fake = netD(torch.cat([real_A, fake_B], 1))
loss_G = criterionGAN(pred_fake, torch.ones_like(pred_fake)) + criterionL1(fake_B, real_B) * 100.0
loss_G.backward()
optG.step()

print(f"G={loss_G.item():.4f} D={loss_D.item():.4f}")
print("ALL OK - Training step works!")
