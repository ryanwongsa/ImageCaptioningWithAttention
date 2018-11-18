# ImageCaptioningWithAttention

Example layout and directory flow to set up tensorflow model for training and inference. Example uses image captioning with attention. Purpose of repo to set up a boiler plate for future tensorflow models.

## Dev Branch

Dev branch contains a implementation which can be deployed to a web server.

```
pip install -r requirements.txt

python app/server.py serve
```

## Master Branch

Master branch contains the main components for setting up the training process.

```
conda env create -f ImageCaptioningWithAttention.yml
conda activate ImageCaptioningWithAttention
```

Main notebook for training: `ImageCaptioningWithAttention.ipynb`

## References

1. [Image Captioning with Attention](https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/contrib/eager/python/examples/generative_examples/image_captioning_with_attention.ipynb)
