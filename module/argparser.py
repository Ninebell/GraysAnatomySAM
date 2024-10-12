
import argparse

def creater_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        default="/data1/inqlee0704/medsam/valid/imgs",
        # default="/data1/inqlee0704/medsam/valid/imgs",
        # default="test_demo/imgs/",
        # required=True,
        help="root directory of the data",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="results/warm-durian-85",
        # default="test_demo/segs/",
        help="directory to save the prediction",
    )
    parser.add_argument(
        "-lite_medsam_checkpoint_path",
        type=str,
        # default="/home/inqlee0704/medsam/MedSAM/workdir/warm-durian-85/efficientvit_sam_best.pth",
        # default="/home/inqlee0704/medsam/MedSAM/workdir/eternal-silence-73/medsam_lite_best.pth",
        # default="inqlee/efficientvit_sam_best.pth",
        default="work_dir/LiteMedSAM/efficientvit_samMicre_best.pth",
        help="path to the checkpoint of MedSAM-Lite",
    )
    parser.add_argument(
        "-device",
        type=str,
        default="cpu",
        help="device to run the inference",
    )
    parser.add_argument(
        "-num_workers",
        type=int,
        default=4,
        help="number of workers for inference with multiprocessing",
    )
    parser.add_argument(
        "--save_overlay",
        # default=True,
        default=False,
        action="store_true",
        help="whether to save the overlay image",
    )
    parser.add_argument(
        "-png_save_dir",
        type=str,
        default="./overlay/warm-durian-85",
        help="directory to save the overlay image",
    )

    return parser