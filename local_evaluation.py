import os
from tqdm.auto import tqdm
import numpy as np

from evaluaton_utils import read_depth_image, align_monodepth_prediction
from mono_depth_estimation_aicrowd.aicrowd_wrapper import AIcrowdWrapper


def check_data(datafolder):
    """
    Checks if the data is downloaded and placed correctly
    """
    imagefolder = os.path.join(datafolder, "inputs")
    annotationsfolder = os.path.join(datafolder, "depth_annotations")
    dl_text = (
        "Please download the public data from"
        "\n https://www.aicrowd.com/challenges/scene-understanding-for-autonomous-drone-delivery-suadd-23/problems/mono-depth-perception/dataset_files"
        "\n And unzip it with ==> unzip <zip_name> -d public_dataset"
    )
    if not os.path.exists(imagefolder):
        raise NameError(f"No folder named {imagefolder} \n {dl_text}")
    if not os.path.exists(annotationsfolder):
        raise NameError(f"No folder named {annotationsfolder} \n {dl_text}")


def abs_rel(target, prediction):
    mask = np.isfinite(target) & (target > 0)
    return np.mean(np.abs(target[mask] - prediction[mask]) / target[mask])


def mae(target, prediction):
    mask = np.isfinite(target) & (target > 0)
    return np.mean(np.abs(prediction[mask] - target[mask]))


def sq_rel(target, prediction):
    mask = np.isfinite(target) & (target > 0)
    sq_rel = np.mean(((target[mask] - prediction[mask]) ** 2) / target[mask])
    return sq_rel


def si_log(target, prediction):
    # https://proceedings.neurips.cc/paper/2014/file/7bccfde7714a1ebadf06c5f4cea752c1-Paper.pdf Section 3.2
    mask = np.isfinite(target) & (target > 0)
    num_vals = mask.sum()
    log_diff = np.log(prediction[mask]) - np.log(target[mask])
    si_log_unscaled = np.sum(log_diff**2) / num_vals - (np.sum(log_diff) ** 2) / (
        num_vals**2
    )
    si_log_score = np.sqrt(si_log_unscaled) * 100
    return si_log_score


def clc_bias(target, prediction):
    mask = np.isfinite(target) & (target > 0)
    b = prediction[mask] - target[mask]
    return np.mean(b)


def calculate_metrics(depth_annotation, depth_prediction):
    mae_score = mae(depth_annotation, depth_prediction)
    abs_rel_scre = abs_rel(depth_annotation, depth_prediction)
    sq_rel_score = sq_rel(depth_annotation, depth_prediction)
    si_log_score = si_log(depth_annotation, depth_prediction)
    bias = clc_bias(depth_annotation, depth_prediction)
    metrics = {
        "si_log": float(si_log_score),
        "abs_rel": float(abs_rel_scre),
        "sq_rel": float(sq_rel_score),
        "mae": float(mae_score),
        "bias": float(bias),
    }
    return metrics


def evaluate(LocalEvalConfig):
    """
    Runs local evaluation for the model
    Final evaluation code is the same as the evaluator
    """
    datafolder = LocalEvalConfig.DATA_FOLDER

    check_data(datafolder)

    imagefolder = os.path.join(datafolder, "inputs")
    preds_folder = LocalEvalConfig.OUTPUTS_FOLDER

    model = AIcrowdWrapper(
        predictions_dir=preds_folder,
        predictions_vis_dir=LocalEvalConfig.OUTPUTS_VIS_FOLDER,
        dataset_dir=imagefolder,
    )
    file_names = LocalEvalConfig.TEST_FILES
    if file_names is None:
        file_names = sorted(os.listdir(imagefolder))
        file_names = file_names[:10]

    file_names = file_names[:10]

    # Predict on all images
    for fname in tqdm(file_names, desc="Predicting Depth Values"):
        model.predict_depth_single_image(fname)

    # Evalaute metrics
    all_metrics = {}
    annotationsfolder = os.path.join(datafolder, "depth_annotations")
    for fname in tqdm(file_names, desc="Evaluating results"):
        depth_annotation = read_depth_image(os.path.join(annotationsfolder, fname))
        depth_prediction = read_depth_image(os.path.join(preds_folder, fname))
        depth_prediction_aligned = align_monodepth_prediction(
            depth_annotation, depth_prediction
        )
        all_metrics[fname] = calculate_metrics(
            depth_annotation, depth_prediction_aligned
        )

    metric_keys = list(all_metrics.values())[0].keys()
    metrics_lists = {mk: [] for mk in metric_keys}
    for metrics in all_metrics.values():
        for mk in metrics:
            metrics_lists[mk].append(metrics[mk])

    print("Evaluation Results")
    results = {key: np.mean(metric_list) for key, metric_list in metrics_lists.items()}
    for k, v in results.items():
        print(k, v)


if __name__ == "__main__":
    # change the local config as needed
    class LocalEvalConfig:
        DATA_FOLDER = "./sample_test_data/"
        OUTPUTS_FOLDER = "./sample_test_data/predictions/"  # used for scores calc
        OUTPUTS_VIS_FOLDER = (
            "./sample_test_data/predictions_vis/"  # used for visual check
        )
        TEST_FILES = os.listdir("./sample_test_data/depth_annotations/")

    outfolder = LocalEvalConfig.OUTPUTS_FOLDER
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)

    evaluate(LocalEvalConfig)
