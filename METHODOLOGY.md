 # Methodology

## How to run

Tested on macOS (Apple M2).

**Setup**

```bash
git clone https://github.com/cthotti/yonder-ai-ml-take-home.git
cd yonder-ai-ml-take-home
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Download the data.** The script reads the Roboflow key from the `ROBOFLOW_API_KEY` environment variable (or a `.env` file in the repo root).

```bash
export ROBOFLOW_API_KEY=your_key_here
python scripts/download.py
python scripts/inspect_data.py
```

**Build the datasets and train.** Training takes about 35 minutes for the baseline and about an hour for the final model on an M2.

```bash
python scripts/augment.py --copies 0 --out data_base
python scripts/augment.py --backgrounds 80 --out data_aug_v3
python scripts/train.py --data data_base/data.yaml --name baseline --hsv-h 0.015
python scripts/train.py --data data_aug_v3/data.yaml --name aug_v3
```

**Evaluate**

```bash
python scripts/baseline_vs_augmented.py
python scripts/error_analysis.py --weights runs/aug_v3/weights/best.pt
python scripts/quantization.py --weights runs/aug_v3/weights/best.pt
```

**Inference on a folder of images.** The final weights are committed, so this runs without training:

```bash
python scripts/predict.py path/to/images path/to/output --weights runs/aug_v3/weights/best.pt --conf 0.25
```

Each image gets a `.txt` with one detection per line: `class_id x_center y_center width height confidence` (normalized 0-1, class 0 = bottle, 1 = mallet). The confidence threshold is **0.25**. An empty file means nothing was detected.

**Live webcam:** `python scripts/live.py --weights runs/aug_v3/weights/best.pt --camera 1` (press `s` to save a frame, `q` to quit).

---

## What I did, in order

### 1. Looked at the data before training anything

I scrolled through the images first to see what I was working with.

- 800 training and 199 validation images, all 512x512, two classes: bottle and mallet.
- Lots of rotated and brightness-shifted copies of the same photos.
- Most images are dark, and the objects are small in the frame.
- Very little variety: almost the same water bottle in every shot, and every mallet is the same orange color.
- Nothing that looked blurry, out of focus, low contrast or grainy, even though a camera on a moving rover will produce exactly that.
- Every image contains at least one object, so the model never sees a scene with nothing in it.

I kept the provided train/validation split. Copies of the same photo may appear on both sides, so the validation scores are probably a bit optimistic.

### 2. Trained a baseline with default settings

Before changing anything, I trained YOLOv8n on the original data with default settings. Every later result is compared against it.

| | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| bottle | 0.908 | 0.788 | 0.850 | 0.559 |
| mallet | 0.922 | 0.972 | 0.985 | 0.665 |
| all | 0.915 | 0.880 | 0.918 | 0.612 |

The baseline is already good on clean images, but bottle is clearly the weaker class.

### 3. Added augmentation, including grayscale

Based on what was missing from the data, I wrote `augment.py`. It adds one extra version of every training image with 1-2 random effects:

| effect | why |
|---|---|
| defocus blur, motion blur | a rover camera moves and isn't always in focus |
| noise | cheap camera sensors, low light |
| contrast | harsh sun vs. overcast |
| gamma | darker and brighter lighting |
| grayscale | see below |

**Why grayscale matters.** In the real world, water bottles and mallets come in many colors and shapes, but in this dataset every mallet is the same orange. A model trained on that can learn "orange = mallet" and ignore shape entirely. Taking the color away in some of the training images forces it to also learn what a mallet looks like. For the same reason I raised YOLO's hue-shift setting (`hsv_h`) from 0.015 to 0.1, which recolors objects during training.

All of these effects only change pixel values and never move anything, so the original bounding boxes stay correct and label files are copied as-is. Only training images are augmented. Validation stays untouched so every model is scored on the same real images. The augmented data is written to a new folder and the original download is never modified.

On clean validation, the augmented model scored about the same as the baseline (mAP50-95 0.604 vs 0.612). That's expected, clean validation looks just like the original training data, so it can't show what augmentation is for. That's why I built the stress test in step 4.

### 4. Built a stress test to measure robustness

`baseline_vs_augmented.py` takes the validation images, applies one corruption at a time (blur, dark, low contrast, noise, grayscale), and evaluates every model on exactly the same images.

The biggest effects (mAP50-95):

| condition | baseline | augmented |
|---|---|---|
| clean | 0.612 | 0.604 |
| blur | 0.403 | 0.578 |
| noise | 0.445 | 0.542 |
| grayscale | 0.230 | 0.412 |

- **Blur:** the baseline loses a third of its score, while the augmented model barely changes.
- **Grayscale:** this confirmed the color problem. With color removed, the baseline's mallet recall drops to 0.10 while bottle recall stays at 0.66, so it found mallets almost entirely by color. With grayscale training, mallet recall in grayscale goes up to 0.50. The model leans on color less, but not zero.
- **Dark and low contrast** improved only slightly. The dataset is already dark and Roboflow had already added brightness changes, so the baseline handled those reasonably well already.

The cost was a drop in mallet recall on clean images, from 0.972 to 0.896.

One caveat: these corruptions are the same kinds the augmented model was trained on, so the test favors it. It shows the model learned what I taught it, not that it generalizes to the real world.

### 5. First error analysis

The stress test gives scores, but not *why* the model fails. So I wrote `error_analysis.py`, which runs the model over the validation set, matches its predictions to the labels (same class, box overlap of at least 50%), and saves every image with a mistake. Green boxes are correct, red boxes are false positives, and yellow boxes are objects it missed.

Going through those images, one pattern stood out: many of the mistakes were mallets lying on dirt or rocky ground, where the model also put boxes on **rocks**. Some rocks have a similar size, shape and shading to a mallet head or a bottle lying down, and the model had never been shown ground with nothing on it, so it had no reason not to call a rock an object. That's what led to the next step.

### 6. Added background images (about 10% of the data)

Every training image has a bottle or mallet in it, so the model never learns what "nothing here" looks like. Combined with the rock false positives above, that told me the model needed to see plain ground and learn that it isn't an object.

I made background images from the training photos themselves. The objects are small, so most of every photo is empty dirt and rock. The script cuts out a square that doesn't touch any labeled box, resizes it to 512x512, and saves it with an empty label file (YOLO reads that as "no objects"). It also keeps a margin around every box, so a partly visible mallet never gets labeled as background. That would teach the model to ignore partly hidden mallets, which is the opposite of what we want.

I added 80, about 10% of the 800 original images. I kept it at that level because too many backgrounds make a model too cautious, and bottle recall, already the weak spot, would drop first.

| | bottle P | bottle R | mallet P | mallet R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| baseline | 0.908 | 0.788 | 0.922 | 0.972 | 0.918 | 0.612 |
| augmented | 0.904 | 0.781 | 0.957 | 0.896 | 0.903 | 0.604 |
| **augmented + backgrounds** | **0.939** | **0.811** | 0.924 | 0.934 | **0.920** | 0.611 |

- Bottle precision went up 3.5 points, meaning fewer false bottles on dirt and rocks, which is what the backgrounds were added for.
- Mallet recall recovered from 0.896 to 0.934, winning back most of what augmentation cost.
- Mallet precision went down a bit, from 0.957 to 0.924, which I didn't expect.

### 7. Picked the final model

On the full stress test (mAP50-95):

| condition | baseline | augmented | **+ backgrounds (final)** |
|---|---|---|---|
| clean | 0.612 | 0.604 | **0.611** |
| blur | 0.403 | 0.578 | **0.598** |
| dark | 0.559 | **0.582** | 0.561 |
| low contrast | 0.581 | 0.589 | **0.603** |
| noise | 0.445 | 0.542 | **0.549** |
| grayscale | 0.230 | 0.412 | **0.433** |
| **average** | 0.472 | 0.551 | **0.559** |

The final model (augmentation, grayscale and backgrounds) has the best average, the best worst case, and matches the baseline on clean images. Its only regression is the dark condition. For a rover, doing reasonably well in bad conditions matters more than a few hundredths on clean images.

### 8. Error analysis on the final model

I ran `error_analysis.py` again on the final model to see what was left. On validation:

- **29 missed objects** (24 bottles, 5 mallets) and **46 false positives** (24 bottle, 22 mallet), in 46 of the 199 images.
- **The model rarely mixes up the two classes.** Only 4 false positives landed on an object of the other class. Its mistakes are about finding objects at all, not telling them apart.
- **Crowded images cause many of the misses.** Two images with many objects in them account for 11 of the 29.
- **Some hard images aren't field photos at all**, like a product listing photo, a product comparison image and a drawing. Part of the error comes from image types the rover will never see.
- **Some "misses" are really loose boxes.** When the same object has both a red and a yellow box, the model found it but its box overlapped the label by less than 50%.
- **Rocks:** in the first error analysis (step 5), rocks on dirt were the most common false positive, which is why the background images were added. The rise in bottle precision suggests fewer of those now, though some rocky scenes still produce false boxes.

### 9. Real-world test with a webcam

Instead of recording phone video, I ran the model live on a Logitech webcam and saved individual frames.

When I held up a book, the model detected it as a bottle. This makes sense given the training data. The model has never seen a scene without a bottle or mallet, and it has never seen indoor objects at all, so a tall rectangle is the closest thing to a bottle it knows. The background crops from step 6 teach it that dirt and rocks aren't objects, but not that books and cups aren't. The fix would be adding frames like this, with empty labels, as extra background images.

I didn't have a hammer or other mallet-shaped object, so I couldn't test mallet detection on real footage. That's the biggest gap in this test.

### 10. Part 2: making the model smaller and faster

Since Yonder Dynamics' real target is an NPU on an OrangePi, not a laptop, I measured size, accuracy and speed for different versions of the final model. Speed is the median over 50 images on the M2's CPU.

**Why ExecuTorch.** ExecuTorch is PyTorch's runtime for running models on devices. Instead of shipping all of PyTorch, the model is exported ahead of time with `torch.export` into a fixed graph, handed to a backend (here XNNPACK, a library of optimized CPU kernels for ARM and x86), and saved as a small `.pte` file that a lightweight C++ runtime executes. That's useful for Yonder because the OrangePi also has ARM CPU cores next to the NPU: if a model, or part of one, can't run on the NPU, ExecuTorch with XNNPACK runs the PyTorch-trained model efficiently on the CPU without converting it through another framework.

| version | size | mAP50-95 | mallet recall | CPU time |
|---|---|---|---|---|
| PyTorch, 512 px | 6.2 MB | 0.611 | 0.934 | 34.0 ms |
| PyTorch, 416 px | 6.2 MB | 0.605 | 0.953 | 24.5 ms |
| PyTorch, 320 px | 6.2 MB | 0.575 | 0.905 | 16.2 ms |
| ONNX, 512 px | 12.2 MB | 0.602 | 0.934 | 24.5 ms |
| OpenVINO, 512 px | 12.3 MB | 0.602 | 0.934 | 18.4 ms |
| OpenVINO INT8, 512 px | 3.6 MB | 0.592 | 0.947 | 23.1 ms |
| **OpenVINO INT8, 416 px** | **3.5 MB** | **0.593** | **0.948** | **16.2 ms** |
| ExecuTorch (XNNPACK), 512 px | 12.2 MB | 0.602 | 0.934 | 27.5 ms |
| ExecuTorch (XNNPACK), 416 px | 12.2 MB | 0.596 | 0.953 | 19.6 ms |

The model has 3.0M parameters (8.1 GFLOPs).

- **Smaller input images are the easiest speedup.** 416 px is about 28% faster than 512 px with almost no accuracy loss. 320 px starts to hurt, because the objects are already small.
- **The runtime matters as much as the model.** The exact same model takes 34.0 ms in PyTorch, 27.5 ms in ExecuTorch and 18.4 ms in OpenVINO.
- **ExecuTorch with XNNPACK** gave identical accuracy and was 1.24x faster than PyTorch, with no quantization.
- **INT8 quantization barely costs accuracy.** I used post-training quantization (OpenVINO + NNCF), calibrated on training images so validation stays unseen. It shrinks the model 3.4x and only lowers mAP50-95 by about 0.01.
- **INT8 was slower than FP32 at 512 px on my Mac.** The M2 CPU has no fast INT8 path in OpenVINO, so it pays for the conversions. The OrangePi's NPU is built for INT8, so the size and accuracy results carry over, but my CPU timings don't.

**Recommendation: INT8 at 416 px.** It's the smallest and fastest version, 2.1x faster than the original, for a small accuracy cost.

## YOLO settings and why

I barely changed the YOLO model itself. Almost all of the work went into the data. Only two settings differ from the defaults (`hsv_h` and `patience`), and everything else was a deliberate choice to keep the default.

| setting | value | why |
|---|---|---|
| model | YOLOv8n | The smallest YOLOv8 (3.0M parameters). The real target is an onboard NPU, so I started small; a bigger model would cost about 3x the compute. |
| image size | 512 | The dataset's native size, so images aren't stretched or shrunk. The objects are small and need the resolution. |
| epochs | 40 | Matches the assignment's reference timing. The baseline had leveled off by then, and a run takes 35-60 minutes on my laptop. |
| batch size | 16 | The default, and it fits in the M2's memory. |
| patience | 10 | Stops training if validation doesn't improve for 10 epochs. It saves time, but it never actually triggered. |
| hsv_h | 0.1 (default 0.015) | Larger hue shifts recolor the mallet during training, for the same reason as grayscale. |
| other augmentations | defaults | Mosaic, flips, scale and translate were kept. Mosaic in particular helps with small objects. |
| optimizer | auto (AdamW) | Ultralytics chooses this for short training runs. |
| seed | 0 | So results are reproducible. |
| confidence threshold | 0.25 | Kept low because missing a mallet is worse for the rover than a false alarm that gets re-checked as it drives closer. |

## Metrics

I report precision, recall, mAP50 and mAP50-95 for each class instead of accuracy:

- **Recall per class** shows whether one class is quietly failing. Mallet recall matters most, since missing the mallet means failing the task.
- **mAP50-95** also rewards tight boxes, not just roughly finding the object.
- Accuracy doesn't really apply to detection, and a single number can hide one class doing badly.

## Limitations

- Validation was used both to pick the best training epoch and to choose between models, and it may share source photos with training. The scores are likely a bit optimistic.
- The training data is mostly one bottle type and one mallet color.
- The stress test uses the same kinds of corruption the model was trained on.
- The real-world test was indoors, small, and had no mallet stand-in.
- The augmented models were still improving slowly at epoch 40, so longer training might help.
