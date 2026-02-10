# Introduction to bioimage analysis


```{note} Plan
- What segmentation is and why we do it
- Distinguish semantic vs instance segmentation
- Core image concepts: intensity, noise, contrast and histograms
```

## What is bioimage analysis?

Bioimage analysis is the process of turning biological images (microscopy, histology slides, fluorescent images, etc.) into quantitative measurements.

Think of a bioimage as a field of pixels. Your job is to translate it into a spreadsheet:

- How many cells?
- How big are they?
- How bright is a protein marker inside each one?
- How do these values change across conditions?

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
[![pandas](https://img.shields.io/badge/pandas-tabular%20measurements-informational?logo=pandas)](https://pandas.pydata.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-deep%20learning-informational?logo=pytorch)](https://pytorch.org/)
[![scikit-image](https://img.shields.io/badge/scikit-image%20processing-informational)](https://scikit-image.org/)
[![opencv](https://img.shields.io/badge/OpenCV-image%20processing-informational?logo=opencv)](https://scikit-image.org/)
