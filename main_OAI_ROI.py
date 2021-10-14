import glob, os
import numpy as np
import torch
from utils.images_utils import imagesc, to_8bit
import torchvision.transforms as transforms
from PIL import Image
import scipy.ndimage
import pandas as pd
import torch
import torch.nn as nn
from collections import OrderedDict
import scipy.ndimage
from skimage import measure
from utils.match_dess_tse import linear_registration, make_compare, apply_warp

# segmentation network
from models.unet import UNet_clean
from models.OaiLocator import OaiLocator
import os
from dotenv import load_dotenv
load_dotenv('.env')


def resampling(x, new_size):
    l = len(x.shape)
    if l == 2:
        x = torch.from_numpy(x).type(torch.FloatTensor).unsqueeze(0).unsqueeze(1)
    if l == 3:
        x = torch.from_numpy(x).type(torch.FloatTensor).unsqueeze(1)

    x = nn.functional.interpolate(
        input=x, size=new_size, mode='bilinear', align_corners=True)

    return x.squeeze()


def make_dir(x):
    if not os.path.isdir(x):
        os.mkdir(x)


def to_dataset(source, destination):
    """
    presently not in use
    """
    make_dir(destination)
    make_dir(destination + 'train/')
    make_dir(destination + 'train/a/')
    side = 'RIGHT'
    sequence = 'SAG_IW_TSE_'
    alist = sorted(glob.glob(source))

    for i in range(0, int(len(alist) / 10 * 7)):
        ID = alist[i].split('/')[-1].split('.')[0]
        a = np.load(alist[i])
        a[a >= 400] = 400
        for s in range(a.shape[2]):
            imagesc(a[:, :, s], show=False, save=destination + 'train/a/' + ID + '_' + side + '_' + str(s) + '.png')


def make_grid(x):
    return np.concatenate([x[:, :, i] for i in range(x.shape[2])], 1)


def match_tse_dess(tse, dess):
    """
    read the meta data and align TSE and DESS using Slice Location
    """
    dess_slice_location = dess['SliceLocation'].values
    tse_slice_location = tse['SliceLocation'].values
    match_tse_dess = [np.argmin(np.abs(dess_slice_location - x)) for x in tse_slice_location]
    return match_tse_dess


def read_slices_from_df(df, source):
    all = []
    for i in range(df.shape[0]):
        x = np.load(source + df['filename'].values[i]+'.npy', allow_pickle=True)
        all.append(np.expand_dims(x, 2))
    return np.concatenate(all, 2)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def get_model():
    #netg = torch.load('models/netG_model_epoch_360.pth')
    #netg = torch.load('models/NetG.pth')
    netg = torch.load('models/netG_Unet256.pth')
    #netg = torch.load('models/netG_Resnet6.pth')
    #
    locator = OaiLocator(args_m={'backbone': 'alexnet', 'pretrained': True, 'n_classes': 2})
    locatorckpt = torch.load('models/oai_locator_alexnet2.ckpt')
    state_dict = locatorckpt['state_dict']
    new_dict = OrderedDict()
    for k in list(state_dict.keys()):
        new_dict[k.split('net.')[-1]] = state_dict[k]
    locator.load_state_dict(new_dict)

    #unet = torch.load('models/30.pth')
    unet = torch.load('models/seg_Resnet6.pth')
    unet = unet.cuda()

    return netg, locator, unet


def npy_2_tensor(y0, threshold):
    y = 1 * y0
    if threshold:
        y[y>=threshold] = threshold
    y = y / y.max()
    y = torch.from_numpy(y)
    y = y.permute(2, 0, 1)
    y = y.unsqueeze(1)
    y = torch.cat([y] * 3, 1)
    y = y.type(torch.FloatTensor)
    return y


def tensor_2_numpy(y0):
    y = 1 * y0
    y = y[:, 0, :, :].permute(1, 2, 0)
    return y.numpy()


class OAI_preprocess():
    def __init__(self, source, side):
        self.source = source
        self.side = side
        self.meta = meta_process(meta=pd.read_csv(source + 'meta.csv'))
        self.ID_list = sorted(self.meta.loc[self.meta['side'] == side, 'ID'].unique())
        self.create_destination()

    def create_destination(self):
        """
        create folders of destinations
        """
        make_dir(self.source + 'processed/')
        make_dir(self.source + 'processed/SAG_IW_TSE_' + self.side + '_cropped/')
        make_dir(self.source + 'processed/SAG_3D_DESS_' + self.side + '_cropped/')
        make_dir(self.source + 'processed/check_' + self.side + '/')

    def get_tse(self, ID):
        """
        get MRI images of TSE
        """
        meta = self.meta
        side = self.side
        source = self.source
        df_tse = meta.loc[(meta['ID'] == ID) & (meta['sequences'] == 'TSE') & (meta['side'] == side)]
        df_tse = df_tse.drop_duplicates(subset=['filename'])
        self.df_tse = df_tse
        tse = read_slices_from_df(df_tse, source)
        if tse.shape[:2] != (444, 444):
            tse = scipy.ndimage.zoom(tse, (444 / tse.shape[0], 444 / tse.shape[1], 1))
        self.tse = tse

    def get_dess(self, ID, match=True):
        """
        get MRI images of DESS
        """
        meta = self.meta
        side = self.side
        source = self.source
        df_dess = meta.loc[(meta['ID'] == ID) & (meta['sequences'] == 'DESS') & (meta['side'] == side)]
        if match:
            match_td = match_tse_dess(tse=self.df_tse, dess=df_dess)
            dess_match = df_dess.iloc[match_td]
            dess = read_slices_from_df(dess_match, source)
        else:
            dess = read_slices_from_df(df_dess, source)
        if dess.shape[:2] != (384, 384):
            dess = scipy.ndimage.zoom(dess, (384 / dess.shape[0], 384 / dess.shape[1], 1))
        self.dess = dess

    def get_pcl(self, npys, locator, threshold, shift=0):
        """
        get slice location of PCL
        """
        # TSE PCL locator
        y = npy_2_tensor(npys, threshold=threshold)
        loc, = locator((y,))
        loc = nn.Softmax()(loc).detach().cpu().numpy()
        pcl = np.argmax(moving_average(loc[:, 1], 3)) + shift
        if pcl <= 11:
            pcl = 19
        self.pcl_tse = pcl

    def get_t2d(self, y00, netg, resample=512, padding=0):
        """
        convert tse images to dess
        """
        y = 1 * y00
        orisize = y.shape[2]

        print(y.shape)

        if resample:
            y = torch.nn.functional.interpolate(y, (resample, resample), mode='bicubic', align_corners=True)
        if padding:
            y = torch.nn.ZeroPad2d(padding)(y)

        for i in range(y.shape[0]):
            y[i, :, :, :] = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(y[i, :, :, :])

        # use TSE -> DESS Generator
        print(y.shape)
        yg = netg(y.cuda())
        if resample:
            yg = torch.nn.functional.interpolate(yg, (orisize, orisize), mode='bicubic', align_corners=True)
        if padding:
            yg = torch.nn.ZeroPad2d(padding)(yg)

        print(yg.shape)

        self.t2d = tensor_2_numpy(yg.detach().cpu())

        print(self.t2d.shape)

        return self.t2d

    def crop_npys(self, list_names, list_crop):
        """
        crop the list of npys
        """
        for name in list_names:
            npy = getattr(self, name)
            if list_crop[0] is not None:
                npy = npy[list_crop[0], :, :]
            if list_crop[1] is not None:
                npy = npy[:, list_crop[1], :]
            if list_crop[2] is not None:
                npy = npy[:, :, list_crop[2]]
            setattr(self, name, npy)

    def registraion(self, tt0, dd0, smooth_warp_matrix=True, existing_warp_matrix=None):
        """
        perform linear registration
        """
        warp_mode = 1
        tt = 1 * tt0
        tt = tt - tt.min()
        tt = tt / tt.max()
        dd = 1 * dd0
        dd = dd - dd.min()
        dd = dd / dd.max()
        print('performing registration....')
        if existing_warp_matrix is None:
            wm = []
            dess_aligned = []
            for s in range(tt.shape[2]):
                aligned, w = linear_registration(tt[:, :, s], dd[:, :, s], warp_mode=warp_mode, steps=500)
                dess_aligned.append(np.expand_dims(aligned, 2))
                wm.append(np.expand_dims(w, 2))

            dess_aligned = np.concatenate(dess_aligned, 2)
            wm = np.concatenate(wm, 2)

            if smooth_warp_matrix:  # interpolate warping matrices
                xi = np.linspace(0, 1, wm.shape[2])
                wm2 = 0 * wm
                for i in range(wm.shape[0]):
                    for j in range(wm.shape[1]):
                        pcoeff = np.polyfit(xi, wm[i, j, :], 5)
                        wm2[i, j, :] = np.array([np.polyval(pcoeff, x) for x in xi])
                wm = wm2
        else:
            wm = existing_warp_matrix
            dess_aligned2 = []
            for s in range(dd.shape[2]):
                aligned2 = apply_warp(tt[:, :, 0].shape, op.dess[:, :, s], warp_matrix=wm[:, :, s],
                                      warp_mode=warp_mode)
                dess_aligned2.append(np.expand_dims(aligned2, 2))
            dess_aligned = np.concatenate(dess_aligned2, 2)
        print('registration done')
        return dess_aligned, wm


def get_seg(y0, unet):
    with torch.no_grad:
        seg = np.zeros((y0.shape[0], y0.shape[2], y0.shape[3]))
        y = 1 * y0

        y = y - y.min()
        y = y / y.max()
        dx = (y.shape[2] - 384) // 2
        if dx > 0:
            y = y[:, :, dx:-dx, dx:-dx].cuda()
        else:
            y = y.cuda()
        y = unet(y)[0].detach().cpu().numpy()
        if dx > 0:
            seg[:, dx:-dx, dx:-dx] = np.argmax(y, 1)
        else:
            seg = np.argmax(y, 1)
    return seg


def get_seg2(y0, unet, resample):
    osize = y0.shape[2]
    y = 1 * y0
    y = torch.nn.functional.interpolate(y, (resample, resample), mode='bicubic', align_corners=True)

    y = y - y.min()
    y = y / y.max()

    y = y.cuda()
    y = unet(y)
    y = torch.nn.functional.interpolate(y, (osize, osize), mode='bicubic', align_corners=True).detach().cpu().numpy()
    seg = np.argmax(y, 1)

    seg[seg==2] = 1
    seg[seg==3] = 2

    return seg


def seg2crop(npys, cropHW, cartilage_channel):
    # TSE crop from front view
    front = (npys == cartilage_channel).sum(2)
    front[:100, :] = 0
    front[-100:, :] = 0
    front[:, :100] = 0
    front[:, -100:] = 0

    movingavg = scipy.ndimage.uniform_filter(front / front.max(), size=50)
    (cm0front, cm1front) = np.unravel_index(movingavg.argmax(), movingavg.shape)
    crop = [cm0front + cropHW[0], cm0front + cropHW[1], cm1front + cropHW[2], cm1front + cropHW[3]]

    # adjust cropping coordinate if the cropping window is out of the boundaries
    (H, W, _) = npys.shape
    if crop[0] < 0:
        crop[0] = 0
        crop[1] = (cropHW[1] - cropHW[0])
    elif crop[1] > H:
        crop[1] = H
        crop[0] = H - (cropHW[1] - cropHW[0])
    if crop[2] < 0:
        crop[2] = 0
        crop[3] = (cropHW[3] - cropHW[2])
    elif crop[3] > W:
        crop[3] = W
        crop[2] = W - (cropHW[3] - cropHW[2])

    return crop


if __name__ == '__main__':
    from utils.oai_unzip import meta_process
    # get pytorch models
    netg, locator, unet = get_model()

    # basic info
    source = os.environ.get('source') + 'OAI00womac3/Npy/'
    side = 'RIGHT'
    op = OAI_preprocess(source=source, side=side)

    # settings
    existing_warp_matrix = False
    existing_crop = False

    use_dess_segmentation_for_crop = False

    for ID in op.ID_list[29:35]:
        # get op.tse
        op.get_tse(ID)
        # get op.dess
        op.get_dess(ID, match=True)
        # get op.pcl_tsef
        op.get_pcl(npys=op.tse, locator=locator, threshold=400, shift=2)
        print(op.pcl_tse)
        # get op.t2d
        _ = op.get_t2d(npy_2_tensor(op.tse, threshold=400), netg=netg, resample=512, padding=0)
        # first crop enter slice of t2d
        op.crop_npys(list_names=['tse', 'dess', 't2d'],
                     list_crop=[None, None, range(op.pcl_tse - 11, op.pcl_tse + 12)])

        op.t2d_seg = np.transpose(get_seg2(npy_2_tensor(op.t2d, threshold=400), unet=unet, resample=448), (1, 2, 0))
        op.tse_seg = np.transpose(get_seg2(npy_2_tensor(op.tse, threshold=400), unet=unet, resample=448), (1, 2, 0))

        if use_dess_segmentation_for_crop:
            # registration op.dess -> op.dess_aligned by op.t2d
            # wrap_matrix: (2, 3, 23) matrix to define the registration
            if existing_warp_matrix:
                wm = np.load(source + 'processed/wrap_' + side + '/' + ID + '.npy')
            else:
                wm = None
            op.dess_aligned, wrap_matrix = op.registraion(tt0=op.t2d, dd0=op.dess, smooth_warp_matrix=True,
                                                          existing_warp_matrix=wm)
            op.dess_aligned_seg = np.transpose(
                get_seg2(npy_2_tensor(op.dess_aligned, threshold=400), unet=unet, resample=448), (1, 2, 0))
            op.dess_seg = np.transpose(get_seg2(npy_2_tensor(op.dess, threshold=400), unet=unet, resample=384),
                                       (1, 2, 0))
            seg_used_for_crop = op.dess_aligned_seg

        else:
            seg_used_for_crop = op.t2d_seg

        # inplane-crop coordinates
        dx = 384 // 2  # 294
        if existing_crop:
            crop = np.load(source + 'processed/crop_' + side + '/' + ID + '.npy')
        else:
            crop = seg2crop(npys=seg_used_for_crop, cropHW=[-dx-60, dx-60, -dx-30, dx-30], cartilage_channel=2)

        # inplane-cropping
        list_name = ['tse', 'tse_seg', 't2d', 't2d_seg']
        if use_dess_segmentation_for_crop:
            list_name = list_name + ['dess_aligned', 'dess_aligned_seg']
        op.crop_npys(list_names=list_name,
                     list_crop=[range(crop[0], crop[1]), range(crop[2], crop[3]), None])

        # segmentation again
        #op.dess_aligned_seg = np.transpose(get_seg2(npy_2_tensor(op.dess_aligned, threshold=400), unet=unet, resample=256), (1, 2, 0))
        _ = op.get_t2d(npy_2_tensor(op.tse, threshold=2000), netg=netg, resample=256, padding=0)
        op.t2d_seg = np.transpose(get_seg2(npy_2_tensor(op.t2d, threshold=2000), unet=unet, resample=256), (1, 2, 0))

        # save npy files
        np.save(source + 'processed/SAG_IW_TSE_' + side + '_cropped/' + ID + '.npy', op.tse)
        if use_dess_segmentation_for_crop:
            np.save(source + 'processed/SAG_3D_DESS_' + side + '_cropped/' + ID + '.npy', op.dess_aligned)

        # print outputs
        outputs = dict()
        print_out = ['tse', 'tse_seg', 't2d', 't2d_seg']
        N = op.tse.shape[2]
        outputs['t2d'] = np.concatenate([make_compare(op.t2d[:, :, s], op.t2d[:, :, s] / 8) for s in range(N)], 1)
        outputs['tse'] = np.concatenate([make_compare(op.tse[:, :, s], op.tse[:, :, s] / 8) for s in range(N)], 1)
        outputs['t2d_seg'] = np.concatenate([make_compare(op.tse[:, :, s], op.t2d_seg[:, :, s] / 8) for s in range(N)], 1)
        outputs['tse_seg'] = np.concatenate([make_compare(op.tse[:, :, s], op.tse_seg[:, :, s] / 8) for s in range(N)], 1)

        if use_dess_segmentation_for_crop:
            print_out = ['dess_aligned', 'dess_seg', 't2d', 't2d_seg']
            outputs['dess_aligned'] = np.concatenate(
                [make_compare(op.dess_aligned[:, :, s], op.dess_aligned[:, :, s] / 8) for s in
                 range(op.dess_aligned.shape[2])], 1)
            outputs['dess_aligned_seg'] = np.concatenate(
                [make_compare(op.dess_aligned[:, :, s], op.dess_aligned_seg[:, :, s] / 8) for s in
                 range(op.dess_aligned.shape[2])], 1)
            outputs['dess_seg'] = np.concatenate(
                [make_compare(op.dess[:, :, s], op.dess_seg[:, :, s] / 8) for s in
                 range(op.dess_aligned.shape[2])], 1)
            outputs['compare_tse_dess'] = np.concatenate(
                [make_compare(op.tse[:, :, s], op.dess_aligned[:, :, s]) for s in
                 range(op.dess_aligned.shape[2])], 1)

        print_out = [outputs[x] for x in print_out]
        imagesc(np.concatenate(print_out, 0), show=False, save=source + 'processed/check_' + side + '/' + ID + '.jpg')

        # save wrapping matrix and crop coordinates
        make_dir(source + 'processed/wrap_' + side + '/')
        make_dir(source + 'processed/crop_' + side + '/')
        if use_dess_segmentation_for_crop:
            if not existing_warp_matrix:
                np.save(source + 'processed/wrap_' + side + '/' + ID + '.npy', wrap_matrix)
        if not existing_crop:
            np.save(source + 'processed/crop_' + side + '/' + ID + '.npy', crop)
