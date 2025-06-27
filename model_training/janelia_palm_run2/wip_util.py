from tracemalloc import start
import numpy as np
import pandas as pd
import decode
import torch 
from skimage.io import imread


def hess_lab_to_decode(hess_lab, px_size=(130, 130)):
    """
    Convert Hess Lab coordinates to decode format.
    """
    print(f"Converting Hess Lab coordinates to decode format: {hess_lab.shape}")
    print(hess_lab.columns)

    x_peak_width = torch.from_numpy(pd.to_numeric(hess_lab['X Peak Width'], errors='coerce').fillna(0).to_numpy())
    y_peak_width = torch.from_numpy(pd.to_numeric(hess_lab['Y Peak Width'], errors='coerce').fillna(0).to_numpy())
    sigma_z = torch.from_numpy(pd.to_numeric(hess_lab['Sigma Z'], errors='coerce').fillna(0).to_numpy())

    xyz_sig = torch.stack((x_peak_width, y_peak_width, sigma_z), dim=1)

    em = decode.EmitterSet(
        xyz=torch.tensor(hess_lab[['X Position', 'Y Position', 'Z Position']].to_numpy()),
        xyz_sig=xyz_sig,
        phot=torch.tensor(hess_lab['6 N Photons'].to_numpy()),
        frame_ix=torch.tensor(hess_lab['Frame Number'].to_numpy().astype(int)),  # assuming frame starts at 1
        xy_unit='px',  # z is always in nm
        px_size=px_size  # not strictly needed but recommended in order to access xyz in both nm and px
    )

    return em

def zero_pad_index(index, width=5):
    return f"{index:0{width}d}"

def print_wip_util():
    print('wip_util.py')
    print('temp utility module')

def get_np_points(emitters, start_frame=-1, end_frame=-1, do_3D=False):
    points=emitters.xyz_px.cpu().numpy()
    if do_3D:
        points = points[:, :3]  # keep z coordinate
    else:
        points = points[:, :2]  # remove z coordinate
    frames_ix = emitters.frame_ix.cpu().numpy()
    points = np.append(frames_ix[:, np.newaxis], points, axis=1)
    points_sub = points[points[:, 0] > start_frame]

    if end_frame > -1:
        points_sub = points_sub[points_sub[:, 0] < end_frame]

    return points_sub


def smap_csv_to_emitters(smap_csv_name, xy_spacing=100, swap_xy=False, show_info=False):
    smap_data = pd.read_csv(smap_csv_name)

    if show_info:
        print(smap_data.columns)
        print(smap_data.shape)
        print(smap_data.head())
        print(smap_data.xnm.min(), smap_data.xnm.max())
        print(smap_data.ynm.min(), smap_data.ynm.max())
        print(smap_data.znm.min(), smap_data.znm.max())
    # Vectorized operations

    if swap_xy:
        smap_xyz = np.stack([
            smap_data['ynm'].to_numpy()/xy_spacing,
            smap_data['xnm'].to_numpy()/xy_spacing,
            smap_data['znm'].to_numpy()
        ], axis=1)
    else:
        smap_xyz = np.stack([
            smap_data['xnm'].to_numpy()/xy_spacing,
            smap_data['ynm'].to_numpy()/xy_spacing,
            smap_data['znm'].to_numpy()
        ], axis=1)

    smap_xyz_sig = np.stack([
        smap_data['locprecnm'].to_numpy()/xy_spacing,
        smap_data['locprecnm'].to_numpy()/xy_spacing,
        smap_data['locprecznm'].to_numpy()
    ], axis=1)

    smap_phot = smap_data['phot'].to_numpy()
    smap_frame = smap_data['frame'].to_numpy().astype('int32') - 1

    smap_emitters = decode.EmitterSet(
        xyz=torch.tensor(smap_xyz),
        xyz_sig=torch.tensor(smap_xyz_sig),
        phot=torch.tensor(smap_phot),
        frame_ix=torch.tensor(smap_frame),
        xy_unit='px', # z is always in nm
        px_size=(130, 130))  # not strictly needed but recommended in order to access xyz in both 

    print(smap_emitters)

    return smap_emitters

def reverse_emitters(emitters, source_shape, source_axis, emitters_axis):
    emitters.xyz_px[:,emitters_axis] = source_shape[source_axis]-emitters.xyz_px[:,emitters_axis]-1 # reverse x axis to match smap

import glob

def load_iter(id, iter_pattern):  
    print(f"loading iter for id: {id}")
    iter_name = iter_pattern(id)
    iters_name = glob.glob(iter_name)
    if iters_name:
        return imread(iters_name[0])
    else:
        print(f"no iters found for id: {id}")
        return None

def load_iters(start_iter, end_iter, iter_pattern):
    """
    Load iterated frames from a specified pattern.
    """
    frames = np.concatenate([load_iter(i, iter_pattern) for i in range(start_iter, end_iter)], axis=0)

    return frames
