
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

import visdom
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import os.path as path
import os
import utils
import shutil
from dtw import *

viz = visdom.Visdom()
# parameters
batch_size = 1
print("Use cuda:", torch.cuda.is_available())


class generator(nn.Module):

    # Weight Initialization [we initialize weights here]
    def weight_init(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def __init__(self, G_in, G_out, w1, w2, w3):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(G_in, w1)
        self.fc2 = nn.Linear(w1, w2)
        self.fc3 = nn.Linear(w2, w3)
        self.out = nn.Linear(w3, G_out)
        self.weight_init()

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x


# Discriminator also consists of DNN
class discriminator(nn.Module):

    def weight_init(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def __init__(self, D_in, D_out, w1, w2, w3):
        super(discriminator, self).__init__()
        self.fc1 = nn.Linear(D_in, w1)
        self.fc2 = nn.Linear(w1, w2)
        self.fc3 = nn.Linear(w2, w3)
        self.out = nn.Linear(w3, D_out)
        self.weight_init()

    def forward(self, y):
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc3(y))
        y = F.sigmoid(self.out(y))
        return y


def load_mfc(filepath):
    return np.fromfile(filepath, dtype=np.float32).reshape((-1, 40))


def normalize(data):
    return data - data.mean(axis=0)


def align_feats(tgt, src):
    ali = dtw(normalize(src), normalize(tgt))
    return tgt[ali.index2], src[ali.index1]


class speech_data(Dataset):

    def __init__(self, mfc_list):
        self.mfc_list = mfc_list
        with open(self.mfc_list) as f:
            self.files = [l.split()[:2]for l in f]
        self.length = len(self.files)

    def __getitem__(self, index):
        tgt_file, src_file = self.files[int(index)]
        tgt_mfc, src_mfc = align_feats(load_mfc(tgt_file), load_mfc(src_file))
        return tgt_mfc, src_mfc

    def __len__(self):
        return self.length


# Loss Functions
adversarial_loss = nn.BCELoss()
mmse_loss = nn.MSELoss()

# Initialization
Gnet = generator(40, 40, 512, 512, 512).cuda()
Dnet = discriminator(40, 1, 512, 512, 512).cuda()

# Optimizers
optimizer_G = torch.optim.Adam(Gnet.parameters(), lr=0.0001)
optimizer_D = torch.optim.Adam(Dnet.parameters(), lr=0.0001)

# dataloaders
train_dataloader = DataLoader(speech_data("jvs_ver1_feat/train.list"), batch_size=1, shuffle=True, num_workers=2)
val_dataloader = DataLoader(speech_data("jvs_ver1_feat/val.list"), batch_size=1, shuffle=True, num_workers=2)

# Training Function


def training(data_loader, n_epochs):
    Gnet.train()
    Dnet.train()

    for en, (b, a) in enumerate(data_loader):
        a = Variable(a.squeeze(0)).cuda()
        b = Variable(b.squeeze(0)).cuda()

        valid = Variable(Tensor(a.shape[0], 1).fill_(1.0), requires_grad=False).cuda()
        fake = Variable(Tensor(a.shape[0], 1).fill_(0.0), requires_grad=False).cuda()

        optimizer_G.zero_grad()
        Gout = Gnet(a)
        G_loss = adversarial_loss(Dnet(Gout), valid) + mmse_loss(Gout, b) * 10

        G_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(Dnet(b), valid)
        fake_loss = adversarial_loss(Dnet(Gout.detach()), fake)
        D_loss = (real_loss + fake_loss) / 2

        D_loss.backward()
        optimizer_D.step()

        print("[Epoch: %d] [Iter: %d/%d] [D loss: %f] [G loss: %f]" %
              (n_epochs, en, len(data_loader), D_loss, G_loss.cpu().data.numpy()))


def validating(data_loader):
    Gnet.eval()
    Dnet.eval()
    Grunning_loss = 0
    Drunning_loss = 0

    valid = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False).cuda()
    fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False).cuda()

    for en, (b, a) in enumerate(data_loader):
        a = Variable(a.squeeze(0)).cuda()
        b = Variable(b.squeeze(0)).cuda()
        valid = Variable(Tensor(a.shape[0], 1).fill_(1.0), requires_grad=False).cuda()
        fake = Variable(Tensor(a.shape[0], 1).fill_(0.0), requires_grad=False).cuda()

        # optimizer_G.zero_grad()
        Gout = Gnet(a)
        G_loss = adversarial_loss(Dnet(Gout), valid) + mmse_loss(Gout, b)

        Grunning_loss += G_loss.item()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(Dnet(b), valid)
        fake_loss = adversarial_loss(Dnet(Gout.detach()), fake)
        D_loss = (real_loss + fake_loss) / 2

        Drunning_loss += D_loss.item()

    return Drunning_loss / (en + 1), Grunning_loss / (en + 1)


# Path where you want to store your results
output_dir = "output/DNNGAN"
if not path.exists(output_dir):
    os.makedirs(output_dir)


isTrain = True
print("Trainig" if isTrain else "Testing")

if isTrain:
    epoch = 100
    dl_arr = []
    gl_arr = []
    for ep in range(epoch):

        training(train_dataloader, ep + 1)
        if (ep + 1) % 5 == 0:
            torch.save(Gnet, f"{output_dir}/gen_g_1_d_1_Ep_{ep+1}.pth")
            torch.save(Dnet, f"{output_dir}/dis_g_1_d_1_Ep_{ep+1}.pth")
        dl, gl = validating(val_dataloader)
        print(f"D_loss: {dl} G_loss: {gl}")
        dl_arr.append(dl)
        gl_arr.append(gl)
        if ep == 0:
            gplot = viz.line(Y=np.array([gl]), X=np.array([ep]), opts=dict(title='Generator'))
            dplot = viz.line(Y=np.array([dl]), X=np.array([ep]), opts=dict(title='Discriminator'))
        else:
            viz.line(Y=np.array([gl]), X=np.array([ep]), win=gplot, update='append')
            viz.line(Y=np.array([dl]), X=np.array([ep]), win=dplot, update='append')

    np.save(f'{output_dir}/discriminator_loss.npy', dl_arr)
    np.save(f'{output_dir}/generator_loss.npy', gl_arr)

    plt.switch_backend('agg')
    plt.figure(1)
    plt.plot(dl_arr)
    plt.savefig(f'{output_dir}/discriminator_loss.png')
    plt.figure(2)
    plt.plot(gl_arr)
    plt.savefig(f'{output_dir}/generator_loss.png')

else:
    test_list = "test.list"
    Gnet = torch.load(f"{output_dir}/gen_g_1_d_1_Ep_100.pth")
    test_out_dir = "test_out"
    utils.ensure_dir(test_out_dir)
    with open(test_list) as rf:
        for l in rf:
            in_prefix = path.splitext(l.strip())[0]
            out_prefix = path.join(test_out_dir, path.basename(in_prefix))
            x = load_mfc(l.strip())
            x = torch.from_numpy(x).squeeze(0).cuda()
            Gout = Gnet(x).cpu().detach().numpy()
            Gout.tofile(out_prefix + ".gen.mcc")
            shutil.copy2(in_prefix + ".wav", out_prefix + ".wav")
            cmd = "./ahodecoder16_64  {0}.f0 {1}.gen.mcc {0}.fv {1}.gen.wav".format(in_prefix, out_prefix)
            utils.run_command(cmd)
