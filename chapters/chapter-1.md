---
jupyter:
  jupytext:
    default_lexer: ipython3
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: jb
    language: python
    name: python3
---

<!-- #region -->
# Introduction to bioimage analysis


```{note} Plan
- What segmentation is and why we do it
- Distinguish semantic vs instance segmentation
- Core image concepts: intensity, noise, contrast and histograms
```

In recent years, biology has become increasingly spatial: it’s no longer enough to know what molecules are present—we also want to know where they are in tissue. This shift is reflected in the choices of Nature Methods “Method of the Year”: spatially resolved transcriptomics in 2020 and spatial proteomics in 2024.
Both advances rely on images as the primary data source, which makes bioimage analysis not a “nice-to-have,” but the bridge between raw spatial maps and biological insight.

::::{grid} 2
:::{grid-item}
[![nat-2024](https://media.springernature.com/w200/springer-static/cover-hires/journal/41592/21/12)](https://www.nature.com/articles/s41592-024-02565-3)
:::
:::{grid-item}
[![nat-2020](https://media.springernature.com/w200/springer-static/cover-hires/journal/41592/18/1)](https://www.nature.com/articles/s41592-020-01042-x)
:::
::::


```{admonition} Time-capsule note
:class: dropdown
:class:

If you’re reading this from the future and these “Method of the Year” picks have been dethroned—congrats, science kept moving. In *our* timeline: **2020 = spatial transcriptomics**, **2024 = spatial proteomics**. Either way, the message still stands: biology is getting **more spatial**, and images are the data.
```




## What is bioimage analysis?

Bioimage analysis is the craft of turning biological images into numbers you can trust. A microscope gives you pixels; analysis gives you answers: how many cells, how big, how bright, how different across conditions.

A helpful way to think about it: an image is a crowded stadium photo. Your biological question is rarely “what does the stadium look like?”—it’s “how many people are there, where are they sitting, and what are they doing?” Bioimage analysis is the set of methods that turns that photo into a count, a map, and a table of measurements.

**Typical workflow of bioimage analysis**
1. Acquire image (microscope outputs a stack/series)
2. Preprocess (remove noise, correct background)
3. Segment (separate objects from background)
4. Measure (area, intensity, shape, etc.)
5. Analyze (statistics, clustering, classification)
6. Validate (biological plausibility + ground truth if available)

In this book the main focus is **Python**, because it's flexible, popular and integrates cleanly with scientific computing and machine learning.

**Key python packages:**

[![NumPy](https://img.shields.io/badge/NumPy-numerical%20python-informational?logo=numpy)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-filters%20%26%20transforms-informational?logo=scipy)](https://scipy.org/)
[![pandas](https://img.shields.io/badge/pandas-data%20frames-informational?logo=pandas)](https://pandas.pydata.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-deep%20learning-informational?logo=pytorch)](https://pytorch.org/)
[![scikit-image](https://img.shields.io/badge/scikit-image%20processing-informational)](https://scikit-image.org/)
[![opencv](https://img.shields.io/badge/OpenCV-image%20processing-informational?logo=opencv)](https://scikit-image.org/)


## Set up Python environment

Before we touch pixels, let’s make sure your tools are in place. Bioimage analysis tends to mix “scientific Python” (arrays, plots) with “image Python” (TIFF stacks, filters, segmentation). A clean environment keeps that mix from turning into dependency chaos.

### Option A: `conda`

Conda is popular in scientific imaging because it handles compiled dependencies smoothly.

```bash
# 1) Create a fresh environment
conda create -n bioimg python=3.12 -y

# 2) Activate it
conda activate bioimg

# 3) Install core packages for this book
conda install -y numpy scipy pandas matplotlib scikit-image tifffile
```

### Option B: `venv` + `pip`

If you’d rather avoid `conda`, use Python’s built-in environment tools.

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows (PowerShell)
# .\.venv\Scripts\Activate.ps1
```
Then install packages:

```bash
pip install -U pip
pip install numpy scipy pandas matplotlib scikit-image tifffile
```

## Image segmentation

Image segmentation is a fundamental task in computer vision that involves partitioning an image into meaningful regions or segments. The goal is to simplify and organize visual information by grouping pixels with similar characteristics—such as color, intensity, or texture—into coherent structures that correspond to objects or areas of interest. By transforming raw pixel data into structured representations, image segmentation enables higher-level analysis and interpretation, serving as a critical step in applications such as medical imaging, autonomous driving, object recognition, and scene understanding.


### Types of image segmentation

Image segmentation can generally be divided into two major types: semantic segmentation and instance segmentation.

Semantic segmentation
: Semantic segmentation is a computer vision task that assigns a class label to every pixel in an image, grouping pixels into categories such as cell, membrane or background. It provides pixel-level predictions but does not distinguish between separate objects of the same class--for example, all cells in an image are labeled simply as "cell", without differentiating individual cells.

Instance segmentation
: Instance segmentation extends semantic segmentation by not only classifying each pixel but also distinguishing between different instances of the same class. For example, instead of labeling all cells as one group, it identifies and separates each individual cell as a distinct object. This provides both pixel-level precision and object-level separation.

## Image basics

Before we analyze images, we need to load them correctly. In bioimage analysis, this step is often more complicated than expected because biological images come in many formats--some simple, some highly specialized.

### Image file formats

Not all image files are created equal. The format determines:
- how pixel data is stored
- whether metadata is preserved (e.g., pixel size, channels, timepoints)
- how easy it is to load the image in Python

**Common formats you'll encounter:**
| Format                                       | Description                               | Typical use                  |
| -------------------------------------------- | ----------------------------------------- | ---------------------------- |
| `.png`, `.jpg`                               | Standard image formats                    | Figures, quick visualization |
| `.tif` / `.tiff`                             | Flexible, supports multi-dimensional data | Microscopy (very common)     |
| `.ome.tif`                                   | TIFF + standardized metadata              | Bioimaging workflows         |
| `.czi`, `.lif`, `.nd2`, ...                  | Proprietary formats. Microscope-specific  | Raw acquisition data         |

There are even more of different file formats for storing images. But don't panic, usually there is already a python package that can help you to work with your data.

### Loading standard images `png`, `jpg`

In Python, there are many different libraries for working with images. This is both powerful and confusing:
- Different packages often provide similar functionality
- The same task (like loading image) can be done in multiple ways
- Each library has its own conventions (data types, color order, etc.)

```{note}
An important skill is not just knowing one tool, but understanding how to navigate between them.
```

Let's look at two widely used libraries:
- `scikit-image` &rarr; common in scientific computing
- `open-cv` &rarr; common in computer vision and industry

#### Read image using `scikit-image`

```python
from skimage import io
import matplotlib.pyplot as plt

img = io.imread("example.png")

plt.imshow(img)
plt.title("scikit-image")
plt.axis("off")

print("Shape:", img.shape)
print("Type of the object:", type(img))
print("Dtype:", img.dtype)
print("Min/Max": img.min(), img.max())
```

This code reads the image into NumPy array. Uses RGB color order (what matplotlib expects).

#### Images are arrays

In bioimage analysis, an image is not a simple "image" for us, but a multidimensional numerical array (matrix). While from naive point of view pixels correspond to colors, in data analysis we will consider them as a grid of numerical values. This distinction is the foundation of bioimage data analysis.

By treating images as raw arrays, we can immediately apply statistical operations, filter specific channels for analysis, and prepare data for machine learning models without altering the underlying data structure.

```python
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

img = io.imread("example.png") 

print(f"Data type: {img.dtype}")
print(f"Shape: {img.shape}")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img[:, :, 0], cmap='Reds')
plt.title("Channel 0 (Red)\nData Shape: (H, W)")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img[:, :, 1], cmap='Greens')
plt.title("Channel 1 (Green)\nData Shape: (H, W)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img[:, :, 2], cmap='Blues')
plt.title("Channel 2 (Blue)\nData Shape: (H, W)")
plt.axis('off')

plt.tight_layout()
plt.show()
```

Key takeways:
- Notice that while the original shape of the image is `(H, W, 3)`, each slice (e.g., `img[:, :, 0]`) is a standard 2D matrix with shape `(H, W)`. To `numpy`, there is no difference between the Red and Blue channels in terms of structure; they are just different sets of numbers at different indices.





#### Read image using `open-cv`

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("example.png")

plt.imshow(img)
plt.title("OpenCV (raw)")
plt.axis("off")

print("Shape":, img.shape)
print("Type of the object":, type(img))
print("Dtype:", img.dtype)
print("Min/Max:", img.min(), img.max())
```

At first glance, this looks the same--but the displayed image may have incorrect colors. This happens because there is a key difference between these two libraries in the way how the color channels are ordered.
- `scikit-image` &rarr; RGB
- `OpenCV` &rarr; BGR

To "fix" the `open-cv` image:
```python
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title("OpenCV (converted to RGB")
plt.axis("off")
```

````{admonition} Exercise: Reorder color channels
:class: tip

::::{tab-set}

:::{tab-item} Try it
Write a code that takes an BGR image and converts is to RGB image. In this task don't use the `cv2.cvtColor` function and manually adjust the order of color channels.
:::

:::{tab-item} Solution
```python
img_rgb = img[:, :, ::-1]
```
:::

::::

````



### Image intensity

Image intensity is the brightness value of a pixel in a digital image, representing the amount of light captured at a specific point. In grayscale images, it usually ranges from 0 (black) to 255 (white), while in color iamges it is defined by the intensity values of color channels such as red, green, and blue. Intensity variations help identify features and details in image processing.

In case of microscopy images the intensity depends on staining, exposure, gain, optics and biology. When you compare intensities across images, you're often comparing both biology and acquisition settings, so later we will talk about normalization and controls.

### Noise

In biomedical images, imagew noise appears as unwanted grainy or speckled patterns that can make cells, tissues, or other biological structures harder to see clearly. It often occurs during image capture due to low light levels, limitations of imaging equipment, or electronic interference. This noise can blur fine details and reduce the accuracy of observations or measurements. Therefore, reducing noise is important in biomedical imaging to ensure that important biological features are clearly visible and properly analyzed.

```{code-cell}
print(5 + 5)
```


<!-- #endregion -->
