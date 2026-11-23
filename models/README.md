# Trained student checkpoints

The five `student_<category>.pth` files are included in this repository:
bottle, cable, carpet, hazelnut and screw. The app loads the matching checkpoint
when a category is selected. These are the original trained model files.

The frozen ResNet18 teacher is downloaded separately through torchvision on first
use, then cached locally. The full MVTec dataset is optional for the app; see
[the data guide](../data/README.md) for training and evaluation setup.

The training script saves into this directory and replaces matching checkpoints.
Preserve a copy before running a new experiment.
