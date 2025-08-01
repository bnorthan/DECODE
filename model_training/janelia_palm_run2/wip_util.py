from tracemalloc import start
import numpy as np
import pandas as pd
import decode
import torch 
from skimage.io import imread
import glob

def hess_lab_to_decode(hess_lab, px_size=(130, 130)):
    """
    Convert Hess Lab coordinates to decode format.
    """
    print(f"Converting Hess Lab coordinates to decode format: {hess_lab.shape}")
    print(hess_lab.columns)


    x = torch.from_numpy(pd.to_numeric(hess_lab['X Position'], errors='coerce').fillna(0).to_numpy()*px_size[0])
    y = torch.from_numpy(pd.to_numeric(hess_lab['Y Position'], errors='coerce').fillna(0).to_numpy()*px_size[1])
    z = torch.from_numpy(pd.to_numeric(hess_lab['Z Position'], errors='coerce').fillna(0).to_numpy())

    x_peak_width = torch.from_numpy(pd.to_numeric(hess_lab['Sigma X Pos Full'], errors='coerce').fillna(0).to_numpy())*1000
    y_peak_width = torch.from_numpy(pd.to_numeric(hess_lab['Sigma Y Pos Full'], errors='coerce').fillna(0).to_numpy())*1000
    sigma_z = torch.from_numpy(pd.to_numeric(hess_lab['Sigma Z'], errors='coerce').fillna(0).to_numpy())

    xyz = torch.stack((x, y, z), dim=1)
    xyz_sig = torch.stack((x_peak_width, y_peak_width, sigma_z), dim=1)

    em = decode.EmitterSet(
        xyz=xyz, #torch.tensor(hess_lab[['X Position', 'Y Position', 'Z Position']].to_numpy()*px_size),
        xyz_sig=xyz_sig,
        phot=torch.tensor(hess_lab['6 N Photons'].to_numpy()),
        frame_ix=torch.tensor(hess_lab['Frame Number'].to_numpy().astype(int)),  # assuming frame starts at 1
        xy_unit='nm',  # z is always in nm
        px_size=px_size  # not strictly needed but recommended in order to access xyz in both nm and px
    )

    return em

def zero_pad_index(index, width=5):
    return f"{index:0{width}d}"

def get_np_points(emitters, start_frame=-1, end_frame=-1, do_3D=False):
    points=emitters.xyz_px.cpu().numpy()
    if do_3D:
        points = points[:, :3]  # keep z coordinate
    else:
        points = points[:, :2]  # remove z coordinate
    frames_ix = emitters.frame_ix.cpu().numpy()
    points = np.append(frames_ix[:, np.newaxis], points, axis=1)
    points_sub = points[points[:, 0] >= start_frame]

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

    if show_info:
        print(smap_emitters)

    return smap_emitters

def reverse_emitters(emitters, source_shape, source_axis, emitters_axis):
    emitters.xyz_px[:,emitters_axis] = source_shape[source_axis]-emitters.xyz_px[:,emitters_axis]-1 # reverse x axis to match smap


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

def load_iter_results(h5s_path, start_iter, end_iter):
    """
    Load emitter results from h5 files for a range of iterations.
    """
    import os
    from tqdm import tqdm

    #print(f"Loading emitter results from {h5s_path} for iterations {start_iter} to {end_iter}")
    
    if not os.path.exists(h5s_path):
        raise FileNotFoundError(f"The path {h5s_path} does not exist.")
    # create empty emitter set for loading
    emitters = None
    
    for i in tqdm(range(start_iter, end_iter), desc="Loading iterations"):
        out_h5 = os.path.join(h5s_path, f'iter_{i}.h5')
        
        #print(i, start_iter, end_iter)
        if emitters is None:
            emitters = decode.EmitterSet.load(out_h5)
            #emitters.frame_ix += i * frames_per_iter
            #print(emitters)
        else:
            emitters_ = decode.EmitterSet.load(out_h5)
            #emitters_.frame_ix += i * frames_per_iter
            #print(emitters_)
            emitters = decode.EmitterSet.cat([emitters, emitters_])
        
        #print()

    return emitters


def print_emitters_info(emitters):
    print(emitters)
    print(f'x sig range (nm) {emitters.xyz_sig_nm[:,0].min().item():.2f} {emitters.xyz_sig_nm[:,0].max().item():.2f}')
    print(f'y sig range (nm) {emitters.xyz_sig_nm[:,1].min().item():.2f} {emitters.xyz_sig_nm[:,1].max().item():.2f}')
    print(f'z sig range (nm) {emitters.xyz_sig_nm[:,2].min().item():.2f} {emitters.xyz_sig_nm[:,2].max().item():.2f}')
    print()

def filter_coordinates(emitters, xmin=float('-inf'), xmax=float('inf'), ymin=float('-inf'), ymax=float('inf'), zmin=float('-inf'), zmax=float('inf')):
    """
    Filter emitters based on coordinate ranges.
    """
    mask = (
        (emitters.xyz_px[:, 0] >= xmin) & (emitters.xyz_px[:, 0] <= xmax) &
        (emitters.xyz_px[:, 1] >= ymin) & (emitters.xyz_px[:, 1] <= ymax) &
        (emitters.xyz_px[:, 2] >= zmin) & (emitters.xyz_px[:, 2] <= zmax)
    )

    return emitters[mask]
    
def shift_coordinates(emitters, xs, ys, zs):
    emitters.xyz_px[:, 0] = emitters.xyz_px[:, 0]+xs
    emitters.xyz_px[:, 1] = emitters.xyz_px[:, 1]+ys
    emitters.xyz_px[:, 2] = emitters.xyz_px[:, 2]+zs

    return emitters

def filter_sigmas(emitters, x_sig_min=0, x_sig_max=float('inf'), 
                  y_sig_min=0, y_sig_max=float('inf'), 
                  z_sig_min=0, z_sig_max=float('inf'), 
                  prob_min=0.0, phot_min=float('-inf'), phot_max=float('inf')):
    """
    Filter emitters based on sigma ranges.
    """
    mask = (
        (emitters.xyz_sig_nm[:, 0] >= x_sig_min) & (emitters.xyz_sig_nm[:, 0] <= x_sig_max) &
        (emitters.xyz_sig_nm[:, 1] >= y_sig_min) & (emitters.xyz_sig_nm[:, 1] <= y_sig_max) &
        (emitters.xyz_sig_nm[:, 2] >= z_sig_min) & (emitters.xyz_sig_nm[:, 2] <= z_sig_max) &
        (emitters.prob >= prob_min) &
        (emitters.phot >= phot_min) & (emitters.phot <= phot_max)
    )
    return emitters[mask]