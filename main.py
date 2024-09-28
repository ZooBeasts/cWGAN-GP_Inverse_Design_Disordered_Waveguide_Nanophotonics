

import random
import torch.optim as optim
import torch.utils.data
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from Dataloader import get_loader
from gradientpenalty import gradient_penalty
from Model import Critic, Generator, initialize_weights


print("CUDA is available: {}".format(torch.cuda.is_available()))
print("CUDA Device Count: {}".format(torch.cuda.device_count()))
print("CUDA Device Name: {}".format(torch.cuda.get_device_name(0)))

save_dir = 'E:/'

Batch_size = 64
nc = 1
image_size = 64
ngpu = 1
features_d = 64
features_g = 64
Z_dim = 200
num_epochs = 401
lr = 1e-4
lr2 = 1e-4
beta1 = 0.5
Samplesindex = 5
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

manualSeed = 42
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# //////////////////////////////////////////////////////////
dataloader = get_loader(img_size=image_size,
                        batch_size=Batch_size,
                        z_dim=Z_dim,
                        points_path=r'C:/Users/Administrator/',
                        img_folder=r'C:/Users/Administrator/',
                        shuffle=False, )

# /////////////////////////////////////////////////////
# /////////////////////////////////////////////////////

netG = Generator(ngpu).to(device)
netD = Critic(ngpu).to(device)
initialize_weights(netG)
initialize_weights(netD)

# Setup Adam optimizers for both G and D
opt_gen = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
opt_critic = optim.Adam(netD.parameters(), lr=lr2, betas=(beta1, 0.999))

fixed_noise = torch.randn(Samplesindex, Z_dim, 1, 1).to(device)
writer_real = SummaryWriter(f"E:/85_data/dashb/real")
writer_fake = SummaryWriter(f"E:/85_data/dashb/fake")
step = 0

netG.train()
netD.train()

print('Starting training---')
img_list = []
G_loss = []
C_loss = []
for epoch in range(num_epochs):
    for i, (points, img_real, points21) in enumerate(dataloader):
        # get data and transfer to device
        points = points.to(device, dtype=torch.float)
        points21 = points21.to(device, dtype=torch.float)
        b_size = points.size(0)
        cur_batch_size = points.shape[0]

        img_real = img_real.to(device, dtype=torch.float)

        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_dim, 1, 1).to(device)
            fake = netG(points)

            # fake = torch.where(fake >= 0.0, 1.0, -1.0)

            critic_real = netD(img_real, points21).reshape(-1)
            critic_fake = netD(fake, points21).reshape(-1)
            gp = gradient_penalty(netD, points21, img_real, fake, device=device)
            loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            opt_critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        gen_fake = netD(fake, points21).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if i % 200 == 0 and i > 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(dataloader)} \
                        Loss D: {loss_critic:.6f}, loss G: {loss_gen:.6f}"
            )

            with torch.no_grad():
                fake = netG(fixed_noise)
                # take out (up to) index examples
                img_grid_real = torchvision.utils.make_grid(img_real[:Samplesindex], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:Samplesindex], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1

            if epoch % 10 == 0:
                ##Update folder location
                torch.save(netG, save_dir + 'netG' + str(epoch) + '.pt')
                torch.save(netD, save_dir + 'netD' + str(epoch) + '.pt')

            G_loss.append(loss_gen.item())
            C_loss.append(loss_critic.item())



plt.figure(figsize=(10, 5))
plt.title("Generator and Critic Loss ")
plt.plot(G_loss, label="G Loss", color = 'blue')
plt.plot(C_loss, label="C Loss", color = 'red')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig('losse1s.png')
G_loss = np.array(G_loss)
C_loss = np.array(C_loss)
np.savetxt('G_loss1.csv', G_loss, delimiter=',')
np.savetxt('C_loss1.csv', C_loss, delimiter=',')
plt.show()
