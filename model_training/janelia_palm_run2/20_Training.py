import torch
import decode
import decode.utils
import decode.neuralfitter.train.live_engine

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

from decode.neuralfitter.train import live_engine
from decode.neuralfitter.utils import logger as logger_utils

def perform_training():
    """
    This function performs the Decode training process.

    The notebook version generates more figures and is more interactive.
    But the script version is faster to run and could be used on a server...
    """
    
    # Final output
    device = 'cuda:0'  # or 'cpu'
    device_ix = 0  # possibly change device index (only for cuda)
    threads = 4  #  number of threads, useful for CPU heavy computation. Change if you know what you are doing.
    worker = 4  # number of workers for data loading. Change only if you know what you are doing.

    torch.set_num_threads(threads)  # set num threads

    if device != 'cpu':
        if (not torch.cuda.is_available()) or (not decode.simulation.psf_kernel.CubicSplinePSF.cuda_is_available()):
            raise ValueError("You have selected a non CPU device, but CUDA is not available."
                            "Refer to CPU version or check your installation.")
        
    calib_file = r'/home/bnorthan/janelia_slm/data/calibrations/Janelia PALM RUN2/crops/cropped_0_3dcal.mat'
    frame_path = r'/home/bnorthan/janelia_slm/data/Janelia_PALM_RUN2/frames'
    frames_name = r'3DPALM488nm_Iter_0640_0001_ch0_CAM1_stack0000_405nm_0000000msec_0129447004msecAbs_000x_000y_000z_0001t.tif'
    frames_name = os.path.join(frame_path, frames_name)
    param_file = r'/home/bnorthan/janelia_slm/data/Janelia_PALM_RUN2/models/2025-06-27-2class/params.yaml'
    
    data_frames = decode.utils.frames_io.load_tif(frames_name).cpu()
    print(data_frames.shape)    
    param = decode.utils.param_io.load_params(param_file)  # change path if you load custom file

    param.Hardware.device = device
    param.Hardware.device_ix = device_ix
    param.Hardware.device_simulation = device
    param.Hardware.torch_threads = threads
    param.Hardware.num_worker_train = worker

    param.Camera.baseline = 100 #398.6
    param.Camera.e_per_adu = 0.5 #5.0
    param.Camera.em_gain = None
    param.Camera.px_size =[130.0, 130.0] # Pixel Size in nano meter
    param.Camera.qe = 1.0                # Quantum efficiency
    param.Camera.read_sigma = 5.88
    param.Camera.spur_noise = 0.0015
    param.Camera.to_dict()
        
    ## Parameters from SMAP
    param.Simulation.bg_uniform = [5.74 , 41.053]           # background range to sample from. You can also specify a const. value as 'bg_uniform = 100'
    param.Simulation.emitter_av =  10                    # Average number of emitters per frame
    param.Simulation.emitter_extent[2] = [-1200, 1200]    # Volume in which emitters are sampled. x,y values should not be changed. z-range (in nm) should be adjusted according to the PSF
    param.Simulation.intensity_mu_sig = [1500.0, 500.0]  # Average intensity and its standard deviation
    param.Simulation.lifetime_avg = 1.5971                     # Average lifetime of each emitter in frames. A value between 1 and 2 works for most experiments
    param.Simulation.to_dict()

    param.InOut.calibration_file = calib_file
    param.InOut.experiment_out = ''
    param.InOut.to_dict()

    simulator, sim_test = decode.neuralfitter.train.live_engine.setup_random_simulation(param)
    camera = decode.simulation.camera.Photon2Camera.parse(param)

    # finally we derive some parameters automatically for easy use
    param = decode.utils.param_io.autoset_scaling(param)
    tar_em, sim_frames, bg_frames = simulator.sample()
    sim_frames = sim_frames.cpu()

    print(f'Data shapes, simulation: {sim_frames.shape}, real data: {data_frames.shape}')
    print(f'Average value, simulation: {sim_frames.mean().round()}, real data: {data_frames.mean().round()}')
    print(f'Min value, simulation: {sim_frames.min().round()}, real data: {data_frames.min().round()}')
    print(f'Max value, simulation: {sim_frames.max().round()}, real data: {data_frames.max().round()}')

    data_frames = camera.backward(data_frames, device='cpu')
    print()
    print(f'Average value, simulation: {sim_frames.mean().round()}, real data: {data_frames.mean().round()}')
    print(f'Min value, simulation: {sim_frames.min().round()}, real data: {data_frames.min().round()}')
    print(f'Max value, simulation: {sim_frames.max().round()}, real data: {data_frames.max().round()}')

    if device != 'cpu':
        mem_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"Your approximate total GPU memory size on the set device {device} is {mem_gb:.2f} GB.")

    param.HyperParameter.batch_size = 2 

    model_dir = Path(r'/home/bnorthan/janelia_slm/data/Janelia_PALM_RUN2/models/2025-06-27-2class-c')

    if not model_dir.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
    param_out_path = model_dir / 'params.yaml' # or an alternative path
    decode.utils.param_io.save_params(param_out_path, param)

    device = 'cuda'

    model_path = model_dir / 'decode_model.pt'
    ckpt_path = model_dir / 'decode_ckpt.pt'


    # set up tensorboard logger and dictionary logger
    logger = [logger_utils.SummaryWriter(log_dir='logs',
                                        filter_keys=["dx_red_mu", "dx_red_sig",
                                                    "dy_red_mu", "dy_red_sig",
                                                    "dz_red_mu", "dz_red_sig",
                                                    "dphot_red_mu", "dphot_red_sig",
                                                    "f1",
                                                    ]),
            logger_utils.DictLogger()]
    logger = logger_utils.MultiLogger(logger)

    ds_train, ds_test, model, model_ls, optimizer, criterion, lr_scheduler, grad_mod, post_processor, matcher, ckpt = \
        live_engine.setup_trainer(simulator, sim_test, logger, model_path, ckpt_path, device, param)

    dl_train, dl_test = live_engine.setup_dataloader(param, ds_train, ds_test)

    # colab seems to have problems with pin_memory at the moment. So change this to False, although this can make training a bit slower
    dl_train.pin_memory = False

    epoch0 = 0

    print("Training set up successfull.")
    
    try:
        ckpt = decode.utils.checkpoint.CheckPoint.load(ckpt_path)
        model.load_state_dict(ckpt.model_state)
        optimizer.load_state_dict(ckpt.optimizer_state)
        lr_scheduler.load_state_dict(ckpt.lr_sched_state)
        epoch0 = ckpt.step + 1
        model = model.train()
    except FileNotFoundError:
        print(f"Checkpoint file {ckpt_path} not found. Starting from scratch."  )

    converges = False
    n = 0
    n_max = param.HyperParameter.auto_restart_param.num_restarts
    print(n_max,  param.HyperParameter.epochs, epoch0)

    max_epochs =  25 

    max_epochs = max_epochs + epoch0
    converges = False
    n = 0
    n_max = param.HyperParameter.auto_restart_param.num_restarts


    while not converges and n < n_max:
        n += 1

        conv_check = decode.neuralfitter.utils.progress.GMMHeuristicCheck(
            ref_epoch=1,
            emitter_avg=simulator.em_sampler.em_avg,
            threshold=param.HyperParameter.auto_restart_param.restart_treshold,
        )

        for i in range(epoch0, max_epochs):#param.HyperParameter.epochs):
            logger.add_scalar('learning/learning_rate', optimizer.param_groups[0]['lr'], i)

            if i >= 1:
                train_loss = decode.neuralfitter.train_val_impl.train(
                    model=model,
                    optimizer=optimizer,
                    loss=criterion,
                    dataloader=dl_train,
                    grad_rescale=param.HyperParameter.moeller_gradient_rescale,
                    grad_mod=grad_mod,
                    epoch=i,
                    device=torch.device(param.Hardware.device),
                    logger=logger
                )

            val_loss, test_out = decode.neuralfitter.train_val_impl.test(model=model, loss=criterion, dataloader=dl_test,
                                                                        epoch=i,
                                                                        device=torch.device(param.Hardware.device))

            if not conv_check(test_out.loss[:, 0].mean(), i):
                print(f"The model will be reinitialized and retrained due to a pathological loss. "
                        f"The max. allowed loss per emitter is {conv_check.threshold:.1f} vs."
                        f" {(test_out.loss[:, 0].mean() / conv_check.emitter_avg):.1f} (observed).")

                ds_train, ds_test, model, model_ls, optimizer, criterion, lr_scheduler, grad_mod, post_processor, matcher, ckpt = \
                    decode.neuralfitter.train.live_engine.setup_trainer(simulator, sim_test, logger, model_path, ckpt_path, device, param)
                dl_train, dl_test = decode.neuralfitter.train.live_engine.setup_dataloader(param, ds_train, ds_test)

                converges = False
                break

            else:
                converges = True

            """Post-Process and Evaluate"""
            decode.neuralfitter.train.live_engine.log_train_val_progress.post_process_log_test(loss_cmp=test_out.loss, loss_scalar=val_loss,
                                                        x=test_out.x, y_out=test_out.y_out, y_tar=test_out.y_tar,
                                                        weight=test_out.weight, em_tar=ds_test.emitter,
                                                        px_border=-0.5, px_size=1.,
                                                        post_processor=post_processor, matcher=matcher, logger=logger,
                                                        step=i)

            if i >= 1:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(val_loss)
                else:
                    lr_scheduler.step()

            model_ls.save(model, None)
            ckpt.dump(model.state_dict(), optimizer.state_dict(), lr_scheduler.state_dict(),
                            log=logger.logger[1].log_dict, step=i)

            """Draw new samples Samples"""
            if param.Simulation.mode in 'acquisition':
                ds_train.sample(True)
            elif param.Simulation.mode != 'samples':
                raise ValueError

        if converges:
            print("Training finished after reaching maximum number of epochs.")
        else:
            raise ValueError(f"Training aborted after {n_max} restarts. "
                            "You can try to reduce the learning rate by a factor of 2."
                            "\nIt is also possible that the simulated data is to challenging. "
                            "Check if your background and intensity values are correct "
                            "and possibly lower the average number of emitters.")
        


    stop = 0

if __name__ == "__main__":
    
    perform_training()