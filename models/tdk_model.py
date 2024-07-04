import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks


class TDKModel(BaseModel):
  
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(dataset_mode='aligned')
        if is_train:
            parser.add_argument('--coef_L1', type=float, default=10.0, help='weight for L1 loss')
            parser.add_argument('--coef_mask', type=float, default=10.0, help='weight for dab mask')
            parser.add_argument('--coef_E', type=float, default=10.0, help='weight for training estimator')
            parser.add_argument('--coef_nuclei', type=float, default=10.0, help='weight for nuclei density map')
            parser.add_argument('--nef', type=int, default=128, help='# of nuclei density map estimator filters in the first conv layer')
            parser.add_argument('--n_estimator_blocks', type=int, default=4, help='specify nuclei density map estimator architecture')
        return parser

    def __init__(self, opt):
        """
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'G_GAN', 'G_L1', 'G_mask', 'D', 'D_real', 'D_fake', 'E', 'nuclei']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'real_B', 'fake_B']
        
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D', 'E']
        else:  # during test time, only load G
            self.model_names = ['G']
            
        # define a generator
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.n_downsampling, opt.ngf, opt.netG, opt.norm,
                                      opt.use_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # before downsampling: 4 layers, each downsampling: 3 layers, each resnet block: 1 layer(8 child layer)
        self.feat_layers = []
        for i in range(1, self.opt.n_downsampling + 1):
            self.feat_layers.append(4 + i * 3 - 1)

        if self.isTrain:
            # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                                          opt.init_type, opt.init_gain, opt.num_D, self.gpu_ids)
            # define a nuclei density map estimator
            feats_nc = 0
            for nc in range(1, opt.n_downsampling + 1):
                feats_nc += opt.ngf * (2 ** nc)
            self.netE = networks.define_E(feats_nc, 1, opt.nef, opt.n_estimator_blocks, opt.norm, opt.use_dropout,
                                        opt.init_type, opt.init_gain, self.gpu_ids)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionMask = torch.nn.L1Loss()
            self.criterionEstimator = torch.nn.MSELoss()
            self.criterionNuclei = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_E)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.img_ids = input['img_id']
        if self.isTrain:
            self.real_B_dab = input['dab'].to(self.device)
            self.dab_mask = input['dab_mask'].to(self.device)
            self.true_nuclei = input['nuclei_map'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.isTrain:
            self.fake_B = self.netG(self.real_A)
            self.fake_B_dab = self.netG(self.real_A * self.dab_mask)
        else:
            self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN, L1 and mask loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.coef_L1
        # Third, G(A * dab_mask) = B_dab
        self.loss_G_mask = self.criterionMask(self.fake_B_dab, self.real_B_dab) * self.opt.coef_mask
        # Forth, calculate nuclei density map loss
        self.calculate_nuclei_density_loss()
        # combine loss
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_mask + self.loss_nuclei
        # calculate gradients
        self.loss_G.backward()
        
    def backward_E(self):
        self.real_B_feats = self.netG(self.real_B, layers=self.feat_layers, encode_only=True)
        self.real_B_nuclei = self.netE(self.real_B_feats)
        self.real_B_nuclei = F.interpolate(self.real_B_nuclei, size=self.true_nuclei.shape[-2:], mode='bilinear', align_corners=True)
        self.loss_E = self.criterionEstimator(self.real_B_nuclei, self.true_nuclei) * self.opt.coef_E
        self.loss_E.backward()
        
    def calculate_nuclei_density_loss(self):
        self.fake_B_feats = self.netG(self.fake_B, layers=self.feat_layers, encode_only=True)
        self.fake_B_nuclei = self.netE(self.fake_B_feats)
        self.fake_B_nuclei = F.interpolate(self.fake_B_nuclei, size=self.true_nuclei.shape[-2:], mode='bilinear', align_corners=True)
        self.loss_nuclei = self.criterionNuclei(self.fake_B_nuclei, self.true_nuclei) * self.opt.coef_nuclei

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update E
        self.set_requires_grad(self.netG, False)
        self.set_requires_grad(self.netE, True)
        self.optimizer_E.zero_grad()
        self.backward_E()
        self.optimizer_E.step()
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netG, True)
        self.set_requires_grad(self.netE, False)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights
        
