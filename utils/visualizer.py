import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE
from torch.utils.tensorboard import SummaryWriter


def save_images(webpage, visuals, image_ids, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_ids[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []
    ims_dict = {}
    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    """This class includes several functions that can save images and print/save logging information.

    It uses a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: create an HTML object for saveing HTML filters
        Step 3: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.use_html = opt.isTrain and not opt.no_html
        self.use_tensorboard = opt.isTrain and opt.use_tensorboard
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.saved = False
        self.current_epoch = 0
        self.epoch_losses = {}
        self.displayed_img_ids = {}

        # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
            
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)
        
        # create a tensorboard object at <checkpoints_dir>/tensorboard/; training loss curves will be saved
        if self.use_tensorboard:
            self.tensorboard_dir = os.path.join(opt.checkpoints_dir, opt.name, 'tensorboard')
            print('create tensorboard directory %s...' % self.tensorboard_dir)
            util.mkdir(self.tensorboard_dir)
            self.tb_writer = SummaryWriter(self.tensorboard_dir)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False
        self.epoch_losses = {}
        
    def update_epoch_losses(self, epoch_iter_losses, epoch_total_iters):
        for k, v in epoch_iter_losses.items():
            if k not in self.epoch_losses.keys():
                self.epoch_losses[k] = v / epoch_total_iters
            else:
                self.epoch_losses[k] += v / epoch_total_iters
                
    def update_displayed_img_ids(self, epoch, img_id):
        if ('epoch%.3d' % epoch) not in self.displayed_img_ids.keys():
            self.displayed_img_ids['epoch%.3d' % epoch] = img_id

    def display_current_results(self, visuals, epoch, epoch_img_ids):
        """Save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to or save
            epoch (int) - - the current epoch
            epoch_img_ids (dict) - - the image of each epoch to be saved, key: ['epoch%.3d' % epoch]
        """
        if self.use_html and (not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            
            # save images to the disk
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s_%s.png' % (epoch, epoch_img_ids['epoch%.3d' % epoch], label))
                util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=0)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d] -- [%s]' % (n, epoch_img_ids['epoch%.3d' % n]))
                ims, txts, links = [], [], []
                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s_%s.png' % (n, epoch_img_ids['epoch%.3d' % n], label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, losses, tag, iters=None, t_comp=None, t_data=None):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
            tag (str) -- print epoch or iteration losses
        """
        if tag == 'iter':
            message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        elif tag == 'epoch':
            message = ''
        else:
            raise NotImplementedError('[%s] loss information is not found' % tag)
            
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    def plot_current_losses(self, losses, step, tag):
        """Add scalars to tensorboard curves
        
        Parameters:
            losses (dict): training losses stored in the format of (name, float) pairs
            step (int): which epoch or iteration
            tag (str): epoch or iteration losses
        """
        for loss_name in losses:
            self.tb_writer.add_scalar(f"Loss_{tag}/{loss_name}", losses[loss_name], step)
            self.tb_writer.flush()