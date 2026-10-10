# AI Usage Log

I used Claude for most of the code. My workflow is decide what to try from the data or the latest results, ask Claude to write the script, then read it, run it, check the output, and validate with tests.

## 1. Augmentation script

**What I asked:** The data had lots of rotated images but nothing blurry, out of focus, low contrast or grainy, so I asked for a script that adds those variants.

**What I kept vs. rewrote, and why:** Kept the photometric only approach, so boxes don't move and labels copy unchanged.

**What the AI got wrong that I had to catch:** Nothing major.

**How I verified it:** Looked through the augmented images in Finder. Filenames record which effects were applied.

## 2. Grayscale

**What I asked:** Every mallet in the data is the same orange, so I wanted grayscale images to force the model to learn shape. I asked Claude to add it.

**What I kept vs. rewrote, and why:** Kept the function, and Claude's suggestion to raise `hsv_h` to 0.1 for the same reason.

**What the AI got wrong that I had to catch:** Claude created the grayscale function itself, but forgot to call it from main, so I had to add.

**How I verified it:** I verified by checking some of the pictures themselves for the mallet. 

## 3. Training script and baseline

**What I asked:** A simple training script that prints per-class results.

**What I kept vs. rewrote, and why:** I trained the baseline first. And I tuned the parameters according to what I thought was best `hsv_h=0.1` and `patience=10`, and skipped using `yolov8s` since the target is an NPU.

**What the AI got wrong that I had to catch:** It hardcoded `hsv_h=0.1`, so the baseline couldn't be reproduced. I added a `--hsv-h` flag.

**How I verified it:** The printed table matches Ultralytics' own validation output.

## 4. Stress test

**What I asked:** Clean validation showed no difference between models, so I asked for a short script that tests them on blurred, dark, noisy and grayscale copies of the validation set.

**What I kept vs. rewrote, and why:** I added each new model as I trained it.

**What the AI got wrong that I had to catch:** The corruptions match what the augmented models trained on, so the test favors them. I note this in METHODOLOGY.md.

**How I verified it:** The "clean" row matches each model's normal validation scores exactly.

## 5. Error analysis

**What I asked:** A script that saves every validation image the model gets wrong.

**What I kept vs. rewrote, and why:** Kept the matching logic and color-coded images. Later cut it down to validation only.

**What the AI got wrong that I had to catch:** Nothing in the code.

**How I verified it:** Looked through the images myself and found many mistakes were mallets on dirt or rocks, with rocks boxed as objects.

## 6. Background images

**What I asked:** Because of the rock false positives, I wanted images with no objects. Instead of downloading a dataset, I asked for a function that crops empty patches from the training photos, about 10% of the data.

**What I kept vs. rewrote, and why:** Kept it. It leaves a margin around every box so no partial object becomes "background".

**What the AI got wrong that I had to catch:** Nothing, but I checked the crops before training.

**How I verified it:** Copied all 80 crops into a separate folder and checked none contained part of a bottle or mallet.

## 7. Inference and live webcam

**What I asked:** The required inference script, plus a live webcam mode, since I used my Logitech camera instead of phone video.

**What I kept vs. rewrote, and why:** Kept both. The live script saves frames as evidence.

**What the AI got wrong that I had to catch:** Both defaulted to older weights, so I pass `--weights` explicitly.

**How I verified it:** Read the `.txt` output on validation images. On the webcam, a book was detected as a bottle.

## 8. Quantization

**What I asked:** A script comparing size, accuracy and speed for smaller and quantized versions of the model.

**What I kept vs. rewrote, and why:** Added an INT8-at-416 version after 416 px turned out nearly as accurate as 512 and much faster.

**What the AI got wrong that I had to catch:** Nothing, but INT8 being slower on my Mac looked like a bug. I checked why, the M2 CPU has no fast INT8 path, unlike their NPU.

**How I verified it:** Checked file sizes on disk, and confirmed calibration uses training images, not validation.

