# Bioimage analysis

![codex](./chapters/images/akoya_codex.webp)

Turn messy microscopy images into clean, measurable biology — with **Python**.

**You’ll learn to:** segment cells, measure them, and discover patterns (from classical methods to U-Net).

---

## What’s inside

- **Segmentation that actually works**: semantic vs instance  
- **Image essentials**: intensity, noise, contrast, histograms  
- **Classic toolbox**: filtering, thresholding, morphology, labeling, watershed  
- **Deep learning jump**: U-Net + augmentation + training basics  
- **From masks to insight**: features → PCA/UMAP → clustering

---

## Tiny taste of the book

```python
from skimage import io
import matplotlib.pyplot as plt

img = io.imread("cells.tif")
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()
```

## Course prerequisites

- Proficiency in the Python programming language
- Basic familiarity with NumPy and SciPy
- Intermediate-level experience using the command-line interface (CLI)