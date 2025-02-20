from models.encoder import *
from torchinfo import summary


if __name__ == '__main__':
    model = RGBEncoder(14)
    summary(model=model,
            input_size=(1, 14, 128, 128),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])

