import os
import scipy
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from .generator import Generator
from .discriminator import Discriminator
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import pickle as pkl
from tqdm import tqdm
from utils.utils import plot__multi_loss, save_images, read_mask
from utils.preprocess_data import poisson_edit


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class DCGAN(object):
    def __init__(self, nz, ngf, ndf, nc, batch_size, epochs, lrG, lrD, beta1, beta2, image_size, data_root, \
                output_dir, dataset, model_name, gpu_mode):
        """[summary]

        Args:
            nz ([int]): [length of latent vector (i.e. input size of generator]
            ngf ([int]): [Size of feature maps in generator]
            ndf ([int]): [Size of feature maps in discriminator]
            nc ([int]): [Number of channels in the training images. For color images this is 3]
            batch_size ([int]): [Batch size during training]
            epochs ([int]): [number of training epochs]
            lrG ([float]): [Learning rate for optimizers of generator]
            lrD ([float]): [Learning rate for optimizers of discriminator]
            beta1 ([float]): [Beta1 hyperparam for Adam optimizers ]
            beta2 ([float]): [Beta1 hyperparam for Adam optimizers]
            image_size ([int]): [Spatial size of training images. All images will be resized to this]
            data_root ([]): [root directory of datasets]
            dataset ([string]): [datasetname]
            model_name ([string]): [default: DCGAN]
        """
        self.nz = nz
        self.ngf = ngf
        self.ndf = ndf
        self.nc = nc
        self.batch_size = batch_size
        self.epochs = epochs
        self.lrG = lrG
        self.lrD = lrD
        self.beta1 = beta1
        self.beta2 = beta2
        self.image_size = image_size
        self.dataroot = data_root
        self.output_dir = output_dir
        self.G = Generator(self.nz, self.ngf, self.nc)
        self.D = Discriminator(self.nc, self.ndf)
        
        self.gpu_mode = gpu_mode
        if torch.cuda.is_available() and self.gpu_mode:
            self.device = torch.device("cuda:0")
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")
        
        self.dataset = dataset
        self.model_name = model_name
        self.model_dir = os.path.join(self.output_dir, self.dataset, self.model_name)
        self.result_dir = os.path.join(self.output_dir, self.dataset, self.model_name, "results")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        self.resume = os.path.join(self.model_dir, "model.pth")
        self.history = os.path.join(self.model_dir, "history.pkl")


    def load(self):
        """[fucntion to load model]
        """
        checkpoint = torch.load(self.resume)
        self.start_epoch = checkpoint['epoch']
        self.iters = checkpoint['iters']
        self.G.load_state_dict(checkpoint['G'])
        self.D.load_state_dict(checkpoint['D'])
        if os.path.exists(self.history):
            datas = pkl.load(open(self.history, 'rb'))
            self.G_losses = datas["G_loss"]
            self.D_losses = datas["D_loss"]
        else:
            self.G_losses = []
            self.D_losses = []
        print("* {} loaded.\n* {} loaded.".format(self.resume, self.history))

    def save(self, epoch):
        """[function to save model]

        Args:
            epoch ([int]): [number of epochs]
        """
        checkpoint = {}
        checkpoint['iters'] = self.iters
        checkpoint['epoch'] = epoch
        checkpoint['G'] = self.G.state_dict()
        checkpoint['D'] = self.D.state_dict()
        datas = {}
        datas["G_loss"] = self.G_losses
        datas["D_loss"] = self.D_losses
        resume = os.path.join(self.model_dir, "model_{}.pth".format(str(epoch).zfill(4)))
        torch.save(checkpoint, resume)  
        # torch.save(checkpoint, self.resume)
        history = os.path.join(self.model_dir, "history_{}.pkl".format(str(epoch).zfill(4)))
        pkl.dump(datas, open(history, 'wb')) 
        
        self.G_losses = []
        self.D_losses = []  
        print("* {} saved.\n* {} saved.".format(resume, history))


    def train(self):
        
        if os.path.exists(self.resume):
            self.load()
        else:
            self.G.apply(weights_init)
            self.D.apply(weights_init)
            self.start_epoch = 0
            self.iters = 0
            self.G_losses = []
            self.D_losses = []
        self.G.to(self.device)
        self.D.to(self.device)
        # Setup Adam optimizers for both G and D
        optimizerD = optim.Adam(self.D.parameters(), lr=self.lrG, betas=(self.beta1, self.beta2))
        optimizerG = optim.Adam(self.G.parameters(), lr=self.lrD, betas=(self.beta1, self.beta2))

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        self.sample_z_ = torch.randn(self.batch_size, self.nz, 1, 1, device=self.device)

        # Establish convention for real and fake labels during training
        real_label = 1
        fake_label = 0

        # self.writer_real = SummaryWriter(f"logs/real")
        # self.writer_fake = SummaryWriter(f"logs/fake")

        dataloader = DataLoader(
            dataset=datasets.ImageFolder(root=self.dataroot,
                                    transform=transforms.Compose([
                                         transforms.Resize(self.image_size),
                                         transforms.CenterCrop(self.image_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                     ])),
                                    batch_size=self.batch_size, shuffle=True)

        print("Starting Training Loop...")
        # Initialize BCELoss function
        criterion = nn.BCELoss()  
        # For each epoch
        for epoch in tqdm(range(self.start_epoch, self.epochs), desc="Training"):
            # For each batch in the dataloader
            for i, data in enumerate(dataloader):
                ### (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ## Train with all-real batch
                self.D.zero_grad()
                # Format batch
                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, device=self.device)
                # Forward pass real batch through D
                # print(self.D(real_cpu).size())
                output = self.D(real_cpu).view(-1)
                # Calculate loss on all-real batch
                # print(output.size(), label.size())
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
                # Generate fake image batch with G
                fake = self.G(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = self.D(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # errD.backward()
                # Update D
                optimizerD.step()

                # (2) Update G network: maximize log(D(G(z)))
                self.G.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.D(fake).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # D_fake = self.D(fake)
                # errG = -torch.mean(D_fake)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print('Epoch: [{}/{}]\t Iter: [{}/{}]\tLoss_D: {:.4f}\tLoss_G: {:.4f}\tD(x): {:.4f}\tD(G(z)): {:.4f} / {:.4f}'
                          .format(epoch, self.epochs, i, len(dataloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                    print("-"*20)
                    # with torch.no_grad():
                    #     fake = self.G(noise)
                    #     # take 32 samples
                    #     img_grid_real = make_grid(real_cpu[:32], normalize=True)
                    #     img_grid_fake = make_grid(fake[:32], normalize=True)

                    #     self.writer_real.add_image("Real", img_grid_real, global_step=self.iters)
                    #     self.writer_fake.add_image("Fake", img_grid_fake, global_step=self.iters)
                # Save Losses for plotting later
                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())
                self.iters += 1
            self.save(epoch)
            self.visualize_results(epoch, fix=True)  
            # loss_plot(self.history)
            try:
                plot__multi_loss(self.model_dir)
            except Exception:
                pass


    def save_image(self, tensor, target, mask, save_path):
        self.G.eval()
        if self.gpu_mode:
            samples = tensor.cpu().data.numpy().permute(0, 2, 3, 1)
        else:
            samples = tensor.data.numpy().permute(0, 2, 3, 1)
        sample = samples[0][:, :, ::-1]
        sample = np.array(scipy.misc.toimage(sample))
        res = poisson_edit(sample.copy(), mask.copy(), target.copy())
        cv2.imwrite(save_path, res)

    def visualize_results(self, epoch, fix=True):
        self.G.eval()
        tot_num_samples = min(64, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.rand((self.batch_size, self.nz, 1, 1))
            if self.gpu_mode:
                sample_z_ = sample_z_.cuda()

            samples = self.G(sample_z_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().permute(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().permute(0, 2, 3, 1)

        samples = (samples + 1) / 2
        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          os.path.join(self.result_dir, 'epoch{}.png'.format(str(epoch).zfill(4))))

    # Complete image 
    def complete(self, cfgs=None):
        if os.path.exists(self.resume):
            self.load()
            print("Model loaded successfully !!!")
        else:
            raise RuntimeError("No model available to load !!!")

        source_img_dir = os.path.join(self.output_dir, self.dataset, self.model_name, "results_random", "source_img")
        masked_img_dir = os.path.join(self.output_dir, self.dataset, self.model_name, "results_random", "masked_img")
        inpainted_img_dir = os.path.join(self.output_dir, self.dataset, self.model_name, "results_demo", "inpainted_img")
        # Create directory
        os.makedirs(source_img_dir, exist_ok=True)
        os.makedirs(masked_img_dir, exist_ok=True)
        os.makedirs(inpainted_img_dir, exist_ok=True)
        # Transform images
        transform_img = transforms.Compose([transforms.Resize(self.image_size),
                                        transforms.CenterCrop(self.image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
        transform_mask = transforms.Compose([transforms.ToTensor()
                                            ])
        
        criteria = nn.BCELoss()
        image_dir = cfgs.test_image_dir
        mask_dir = cfgs.test_mask_dir
        # Inpainted image and mask that correspond
        images = os.listdir(image_dir)
        aligned_img = [os.path.join(image_dir, image) for image in images]
        mask_images = [os.path.join(mask_dir, os.path.splitext(image)[0] + '.png') for image in images] 

        for i, image in enumerate(images):
            # target was edited by Poission
            target = cv2.imread(aligned_img[i])  
            # mask was edited by Poission edit
            mask = 255 - cv2.imread(mask_images[i], cv2.IMREAD_GRAYSCALE)  
            batch_images = [transform_img(pil_loader(aligned_img[i]))]
            batch_images = torch.stack(batch_images).to(self.device)
            batch_masks = [transform_mask(read_mask(mask_images[i], self.image_size, self.image_size))]
            batch_masks = torch.stack(batch_masks).to(self.device)
            z_hat = torch.rand(size=[1, self.nz, 1, 1], requires_grad=True, device=self.device)
            masked_batch_images = torch.mul(batch_images, batch_masks).to(self.device)
            # z_hat.data.mul_(2.0).sub_(1.0)
            opt = optim.Adam([z_hat], lr=cfgs.lr)
            v = torch.tensor(0, dtype=torch.float32, device=self.device)
            m = torch.tensor(0, dtype=torch.float32, device=self.device)
            for iteration in range(cfgs.num_iters+1):
                # Iter through batch
                if z_hat.grad is not None:
                    z_hat.grad.data.zero_()
                self.G.zero_grad()
                self.D.zero_grad()
                self.G.to(self.device)
                self.D.to(self.device)
                batch_images_g = self.G(z_hat)
                batch_images_g_masked = torch.mul(batch_images_g, batch_masks)
                impainting_images = torch.mul(batch_images_g, (1 - batch_masks)) + masked_batch_images

                if iteration % 5000 == 0:
                    # Save results
                    print("\nsaving impainted images for iteration:{}".format(iteration))
                    save_path = os.path.join(inpainted_img_dir, os.path.splitext(image)[0]+"_iteration_{}.png".format(iteration))
                    self.save_image(batch_images_g, target, mask, save_path)
                    loss_context = torch.norm(
                        (masked_batch_images - batch_images_g_masked), p=1)
                    dis_output = self.D(impainting_images)
                    batch_labels = torch.full((1,), 1, device=self.device)
                    loss_perceptual = criteria(dis_output, batch_labels)

                    total_loss = loss_context + cfgs.lamd * loss_perceptual
                    print("iteration : {:4} , context_loss:{:.4f},percptual_loss:{:4f}".format(iteration,
                                                                                                 loss_context,
                                                                                                 loss_perceptual))
                    total_loss.backward()
                    opt.step()
                    g = z_hat.grad
                    if g is None:
                        print("g is None")
                        continue
                    vpre = v.clone()
                    mpre = m.clone()
                    m = 0.99*mpre+(1-0.99)*g
                    v = 0.999*vpre+(1-0.999)*(g*g)
                    m_hat = m/(1-0.99**(iteration+1))
                    v_hat = v/(1-0.999**(iteration+1))
                    z_hat.data.sub_(m_hat/(torch.sqrt(v_hat)+1e-8))
                    z_hat.data = torch.clamp(z_hat.data, min=-1.0,max=1.0).to(self.device)
