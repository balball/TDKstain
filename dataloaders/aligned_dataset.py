import os
from PIL import Image, ImageFilter
from dataloaders.image_folder import make_dataset
from dataloaders.base_dataset import BaseDataset, get_params, get_transform


class AlignedDataset(BaseDataset):
    """
    This dataset class can load paired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + '_HE')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + '_IHC')  # create a path '/path/to/data/trainB'
        self.img_list = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # get the list of images 
        
        if self.opt.isTrain:
            self.dir_dab = os.path.join(opt.dataroot, opt.phase + '_IHC_dab')  # create a path '/path/to/data/trainB_dab'
            self.dir_dab_mask = os.path.join(opt.dataroot, opt.phase + '_IHC_dab_mask')  # create a path '/path/to/data/trainB_dab_mask'
            self.dir_nuclei_map = os.path.join(opt.dataroot, opt.phase + '_IHC_nuclei_map')  # create a path '/path/to/data/trainB_nuclei_map'
        
        assert (self.opt.direction == 'AtoB')  # dab mask only in domain B
        assert (self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            dab_mask (tensor)       -- its dab mask from the target domain
            img_id (str)       -- its image name
        """
        img_id = self.img_list[index]
        A_path = os.path.join(self.dir_A, img_id)
        B_path = os.path.join(self.dir_B, img_id)
        
        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')
        
        # apply the same transform to A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1), is_mask=False)
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1), is_mask=False)

        A = A_transform(A)
        B = B_transform(B)
        
        if self.opt.isTrain:
            dab_path = os.path.join(self.dir_dab, img_id)
            dab = Image.open(dab_path).convert('RGB')
            dab_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1), is_mask=False)
            dab = dab_transform(dab)
            
            dab_mask_path = os.path.join(self.dir_dab_mask, img_id)
            dab_mask = Image.open(dab_mask_path).convert('L')
            dab_mask_transform = get_transform(self.opt, transform_params, grayscale=False, is_mask=True)
            dab_mask = dab_mask_transform(dab_mask)
            
            nuclei_map_path = os.path.join(self.dir_nuclei_map, img_id)
            nuclei_map = Image.open(nuclei_map_path).convert('L')
            nuclei_map_transform = get_transform(self.opt, transform_params, grayscale=False, is_mask=True)
            nuclei_map = nuclei_map_transform(nuclei_map)
            
            return {'A': A, 'B': B, 'dab': dab, 'dab_mask': dab_mask, 'nuclei_map': nuclei_map, 'img_id': img_id}
        else:
            return {'A': A, 'B': B, 'img_id': img_id}

    def __len__(self):
        return len(self.img_list)