## How to write your own models?

We recommend that you place the code for all your agents in the `my_submisison` directory (though it is not mandatory). We have added random Depth examples in `random_depth_model.py`

**Add your model name in** [`user_config.py`](user_config.py)

## Depth Perception model format
You will have to implement a class containing the function `predict_depth_single_image`. This will recieve input image_to_segment, a single frame from onboard the flight. You need to output a 2D image with the pixels values corresponding to the label number.

The prediction on each image should complete within 10 seconds.

## What's used by the evaluator

The evaluator uses `MyDepthModel` from `user_config.py` as its entrypoint. Specify the class name for your model here.

## What's AIcrowd Wrapper

Don't change this file, it is used to save the outputs you predict and read the input images. We run it on the client side for efficiency. The AIcrowdWrapper is the actual class that is called by the evaluator, it also contains some checks to see your predictions are formatted correctly.

## MiDaS Code

We use the this [commit](https://github.com/isl-org/MiDaS/tree/1645b7e1675301fdfac03640738fe5a6531e17d6) from the [open source library MiDaS](https://github.com/isl-org/MiDaS) for the inference code. The full code is included in this repository only for submission to the competition and not intended for redistibution of software.