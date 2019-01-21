import torch
import torch.nn as nn
import pickle
from matplotlib import pyplot as plt
from torch.utils import data
import torch.optim as optim
import sys

print_iter = 100
batch_size = 64
noise_size = 4
epoches = 4000

real_data_config = sys.argv[1] if len(sys.argv) == 2 else 'line_real_data.pkl'


def get_fake_sample(batch_size_i):
    return torch.rand(batch_size_i, noise_size).cuda()


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.hidden0 = nn.Sequential(
            nn.Linear(input_size, int(hidden_size / 4)),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(int(hidden_size / 4), int(hidden_size / 2)),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(int(hidden_size / 2), hidden_size),
            nn.LeakyReLU(0.2)
        )

        self.out = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )

    def forward(self, x_i):
        x_i = self.hidden0(x_i)
        x_i = self.hidden1(x_i)
        x_i = self.hidden2(x_i)
        return self.out(x_i)


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.hidden0 = nn.Sequential(
            nn.Linear(input_size, int(hidden_size / 4)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(int(hidden_size / 4), int(hidden_size / 2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(int(hidden_size / 2), hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(hidden_size, output_size),
            torch.nn.Sigmoid()
        )

    def forward(self, x_i):
        x_i = self.hidden0(x_i)
        x_i = self.hidden1(x_i)
        x_i = self.hidden2(x_i)
        return self.out(x_i)


with open(real_data_config, 'rb') as data_file:
    real_data = pickle.load(data_file)

dataset = torch.utils.data.TensorDataset(torch.from_numpy(real_data).float(),
                                         torch.ones(len(real_data)).float())
train_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

# return a binary decision 0 - fake, 1 - real
discriminator = Discriminator(2, 512, 1).cuda()

# return a point generated from random noise
generator = Generator(noise_size, 512, 2).cuda()

d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0001)

criterion = nn.BCELoss()

for e in range(epoches):
    d_real_loss_val, d_fake_loss_val, g_fake_loss_val = [0] * 3

    for i, sample in enumerate(train_loader):
        x, y = sample
        x = x.cuda()
        y = y.cuda()

        ''' Train Discriminator '''
        discriminator.zero_grad()

        # Train Discriminator on real data
        real_y_hat = discriminator(x).reshape(x.shape[0])
        real_loss = criterion(real_y_hat, y)
        real_loss.backward()

        # Train Discriminator on fake data
        # create fake points
        fake_noise = get_fake_sample(x.shape[0])
        fake_x = generator(fake_noise).detach()

        # get Discriminator response on fake points
        fake_y_hat = discriminator(fake_x).reshape(x.shape[0])
        fake_loss = criterion(fake_y_hat, torch.zeros(x.shape[0], 1).float().cuda())
        fake_loss.backward()

        d_optimizer.step()

        d_real_loss_val += real_loss.detach().cpu().numpy()
        d_fake_loss_val += fake_loss.detach().cpu().numpy()

        ''' Train Generator '''
        generator.zero_grad()

        # Train Generator on fake data as if it was real data
        fake_noise = get_fake_sample(x.shape[0])
        fake_x = generator(fake_noise)
        fake_y_hat = discriminator(fake_x).reshape(x.shape[0])
        fake_loss = criterion(fake_y_hat, torch.ones(x.shape[0], 1).float().cuda())
        fake_loss.backward()

        g_optimizer.step()

        g_fake_loss_val += fake_loss.detach().cpu().numpy()

    if e % print_iter == 0:
        print('Epoch: {}\nDiscriminator: D(real) loss = {}    D(G(noise)) loss = {}\n'
              'Generator: loss = {}\n'.format(e, 100. * d_real_loss_val / len(real_data),
                                              100. * d_fake_loss_val / len(real_data),
                                              100. * g_fake_loss_val / len(real_data)))

# Create 1000 points
fake_noise = get_fake_sample(1000)
fake_points = generator(fake_noise).detach().cpu().numpy()

plt.plot(real_data[:, 0], real_data[:, 1], 'r.', label='real samples')
plt.plot(fake_points[:, 0], fake_points[:, 1], '.', label='generated samples')
plt.title('training set')
plt.legend()
plt.show()
