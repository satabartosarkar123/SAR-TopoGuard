import sys, traceback, torch
sys.path.insert(0, '.')
from dataset import SEN12MS_Dataset
from models import UNetGenerator, PatchGANDiscriminator
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn as nn

try:
    print('STARTING')
    data_root = Path('mini_sen12_data')
    s1_dir = data_root / 'train' / 's1'
    s2_dir = data_root / 'train' / 's2'
    s1_files = sorted([p for p in s1_dir.glob('*.*') if p.suffix.lower() in ['.tif', '.png']])
    records = []
    for p in s1_files:
        opt_path = s2_dir / p.name
        if opt_path.exists():
            records.append({'sar_path': str(p), 'optical_path': str(opt_path), 'roi': 'mini', 'patch_id': p.stem})
    records = records[:500]
    ds = SEN12MS_Dataset(records=records, patch_size=256, augment=True)
    train_loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0, pin_memory=False)

    print('Loader ready')
    netG = UNetGenerator(in_channels=2, out_channels=3).cuda()
    print('netG ready')
    netD = PatchGANDiscriminator(in_channels=5).cuda()
    print('netD ready')

    optG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.999))
    print('optG created')
    optD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.999))
    print('optD created')

    criterionGAN = nn.BCEWithLogitsLoss()
    criterionL1 = nn.L1Loss()
    lambda_L1 = 100.0
    print('Start loop')

    for epoch in range(1, 11):
        netG.train()
        netD.train()
        print(f'Epoch {epoch} start')
        for i, batch in enumerate(train_loader):
            print(f'Batch {i} loaded')
            real_A = batch['sar'].cuda()
            break
        break
except Exception as e:
    traceback.print_exc()
