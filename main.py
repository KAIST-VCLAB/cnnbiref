import argparse, sys, time
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import *
from BrEOD_dataloader import *
from network import DeblurModuleDouble as StereoRestoreNet
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, default=None)
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--cpu', default=False, action='store_true')
    args = parser.parse_args()
    cfg = read_configure(args.cfg)

    # Load model
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = get_available_device()

    model = StereoRestoreNet().to(device).eval()

    if os.path.isfile(cfg["model.weight"]):
        cp = torch.load(cfg["model.weight"], map_location=device)
        model.load_state_dict(cp['model_I_state_dict'])
        model.eval()
    else:
        sys.exist()

    # Load data
    preprocess = transforms.Compose([
        BrEODToTensor(),
        BrEODNormalize(),
        # BrEODDisparityConversion(cfg["data.min_disp"], cfg["data.max_disp"])
    ])
    dataset = BrEODdataset(root_dir=cfg["data.root"], transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=cfg["batch.size"], shuffle=False)

    # Eval model
    if not os.path.exists(cfg["out.root"]):
        os.makedirs(cfg["out.root"])
    print("Running device:\t", device)

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            im_b = data['im_b'].to(device)
            im_e = data['im_l'].to(device)
            im_o = data['im_r'].to(device)
            # disp = data['disp'].to(device)

            print("Processing Image {0:02d} ...".format(i))
            if device == torch.device('cpu'):
                start = time.time()
            else:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

            im_e_pred, im_o_pred = model(im_b)

            if device == torch.device('cpu'):
                end = time.time()
                print(".... elapsed {0:.4f} sec.".format((end - start)))
            else:
                end.record()
                torch.cuda.synchronize()
                print(".... elapsed {0:.4f} sec.".format(start.elapsed_time(end)/1000))

            np_e_gt = detach_tensor(im_e)
            np_o_gt = detach_tensor(im_o)
            np_e_pd = detach_tensor(im_e_pred)
            np_o_pd = detach_tensor(im_o_pred)

            if args.eval:
                print("Image {0:02d}\t(E-ray) || PSNR: {1:.2f}\t|| SSIM: {2:.4f}".format(
                    i, psnr(np_o_gt, np_o_pd, data_range=1), ssim(np_o_gt, np_o_pd, data_range=1, multichannel=True)))
                print("             \t(O-ray) || PSNR: {1:.2f}\t|| SSIM: {2:.4f}".format(
                    i, psnr(np_e_gt, np_e_pd, data_range=1), ssim(np_e_gt, np_e_pd, data_range=1, multichannel=True)))

            io.imsave(os.path.join(cfg["out.root"],"Restored_e_{:02d}.png".format(i)),
                      (np_e_pd * 255).astype(np.uint8))
            io.imsave(os.path.join(cfg["out.root"], "Restored_o_{:02d}.png".format(i)),
                      (np_o_pd * 255).astype(np.uint8))
