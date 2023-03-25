This is code to run the Gait Transformer described in:

- https://arxiv.org/abs/2203.09371
- https://www.medrxiv.org/content/10.1101/2022.11.10.22282089v1


## Installation

    pip install -r requirements.txt
    pip install -e .

## Usage

See `notebooks/demo.ipynb` for a demo of the model. See `docs` for more examples.

To run this on your own data you will need to produce similarly formatted 3D keypoint sequences as this is trained on. This is used with outputs from PosePipe [http://github.com/peabody124/PosePipeline]. The weights for the demo use the MMPose option for producing the 2D keypoints and GastNet for producing the 3D keypoints.

The Gait Transformer with these weights does not work robustly far from the frontal view. Anecodtally, it appears to still be fairly reliable for gait timing but the estimates of velocity become quite biased. I suspect this is due to viewpoint sensitive biases in the 2D->3D lifting step.

Note that the core gait transformer algorithm uses TensorFlow and the Kalman Filter uses Jax. You will need to ensure they both allow each other memory or perform this as separate steps.

Data to reproduce training this model cannot be made available due to privacy protection.

## TODO:

- [ ] create more standalone utility functions for some of the functions in gait_decoder_testing
- [ ] retrain using the outputs from MeTRAbs-ACAE
- [ ] ideally have a separate frontend for processing videos with the recommended PosePipe steps to make it easier to test
