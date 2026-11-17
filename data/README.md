# MVTec AD data

Obtain the dataset from [MVTec](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
under its CC BY-NC-SA 4.0 terms. Extract category directories here:

```text
data/
  bottle/
    train/good/*.png
    test/good/*.png
    test/<defect>/*.png
    ground_truth/<defect>/*_mask.png
  cable/
  carpet/
  hazelnut/
  screw/
```

Only normal training images are used to fit each student. The app accepts an uploaded
image without needing the full dataset. Data files are excluded from Git.
