---
kernelspec:
    name: python3
    display_name: 'Python 3'
---

# From classical to AI-based segmentation

Classical segmentation methods are powerful because they are transparent: we can usually explain what each step is doing. But real biological images can be complicated. Cells may touch, boundaries may be weak, and useful information may be spread across several channels. In such cases, AI-based tools can often provide a more flexible segmentation approach.

In this section, we will use [**Cellpose-SAM**](https://github.com/mouseland/cellpose) for cell segmentation in spatial proteomics data. Cellpose is a widely used tool in bioimage analysis, and `cellpose-sam` combines Cellpose with [Meta’s Segment Anything Model](https://aidemos.meta.com/segment-anything/) to improve segmentation robustness. The overall workflow will still look familiar: we load an image, prepare useful channels, run segmentation, inspect the masks, and export the result for downstream analysis.

The full notebook for cell segmentation using the cellpose-sam method is available here:
::::{card} Google colab notebook
:link: https://colab.research.google.com/drive/1WsrS-D0uXWkUkWccldF2eietD1cZ3wnM?usp=sharing
:link-type: url

Learn about cell segmentation using cellpose-sam method.
::::

## Package installation and imports

Before running the segmentation, we need to install and import the required packages. This example uses `cellpose[dino]`, along with packages for reading OME-TIFF files, handling OME metadata, visualization, and exporting segmentation masks.

I suggest creating a clear Python environment first. After that, execute the following commands in your shell:
```bash
echo 'numpy<2.1,>=1.22' > constraints.txt
pip install -q -c constraints.txt "numpy<2.1,>=1.22"
pip install -q -c constraints.txt cellpose[dino]
pip install -q -c constraints.txt fastremap imagecodecs roifile fill-voids segment_anything ome-types tifffile natsort
```

```python
import numpy as np
import matplotlib.pyplot as plt
import tifffile

from pathlib import Path
from ome_types import from_xml
from natsort import natsorted

from cellpose import models, core, io, utils

import json
import rasterio.features
from rasterio.transform import Affine

plt.style.use("dark_background")
```

Cellpose-SAM benefits strongly from GPU acceleration. In this notebook, we explicitly check whether a GPU is available before continuing.
```python
io.logger_setup()

if core.use_gpu() is False:
    raise ImportError("No GPU access. In Colab: Runtime → Change runtime type → GPU")

print("GPU detected.")
```

## Helper functions for reading and displaying OME-TIFF data

Spatial proteomics images are often stored as multi-channel OME-TIFF files. Before segmentation, we need to know which channels are present and how to extract the channels we want to use.


:::{dropdown} Helper functions for reading and displaying OME-TIFF data

The first helper function reads the channel names from the OME metadata.

```python
def get_ome_channel_names(path):
    with tifffile.TiffFile(path) as tif:
        ome = from_xml(tif.ome_metadata)
        channel_names = [
            c.name if c.name is not None else f"channel_{i}"
            for i, c in enumerate(ome.images[0].pixels.channels)
        ]
    return channel_names
```

Next, we define a helper function to read one named channel from the OME-TIFF file. This makes the later code easier to read because we can refer to biological marker names directly instead of manually indexing channels.

```python
def read_ome_channel(path, channel_name):
    """
    Read one named channel from an OME-TIFF.

    This follows a common layout: one page per channel in tif.series[0].
    If the file has a different OME layout, the fallback tries to load the
    full series and index the C axis.
    """
    channel_names = get_ome_channel_names(path)

    if channel_name not in channel_names:
        raise ValueError(
            f"Channel {channel_name!r} not found. Available channels:\n{channel_names}"
        )

    channel_idx = channel_names.index(channel_name)

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]

        # Fast path: common OME-TIFF layout where each channel is a page
        if len(series.pages) > channel_idx:
            img = series.pages[channel_idx].asarray()
            return img

        # Fallback: load whole series and find channel axis
        data = series.asarray()
        axes = series.axes

        if "C" not in axes:
            raise ValueError(f"No channel axis found in OME series axes: {axes}")

        c_axis = axes.index("C")
        img = np.take(data, channel_idx, axis=c_axis)

        # Squeeze singleton dimensions such as T/Z if present
        img = np.squeeze(img)
        return img
```

:::

:::{dropdown} Normalization and cropping helpers
For display and channel combination, it is useful to normalize intensities into a common range. Here we use percentile normalization, which is more robust than using the absolute minimum and maximum values.

```python
def normalize_percentile(img, p_low=1, p_high=99):
    """
    Robustly scale image to [0, 1] for display/composites.
    Cellpose also normalizes internally by default, but this is useful when
    combining several channels into one membrane channel.
    """
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [p_low, p_high])

    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)

    img = (img - lo) / (hi - lo)
    img = np.clip(img, 0, 1)
    return img.astype(np.float32)
```

```python
def crop_image(img, y0, y1, x0, x1):
    return img[y0:y1, x0:x1]
```
For large images, it is usually wise to begin with a crop. This makes experimentation faster and helps us tune parameters before running segmentation on the full image.
:::


:::{dropdown} Building a membrane-like channel
Cell boundaries are often easier to detect when membrane or boundary-like markers are available. In spatial proteomics data, we may have several markers that partially highlight cell boundaries. One practical strategy is to combine multiple markers into a single membrane-like composite channel.

The function below reads selected marker channels, normalizes each one, and combines them using either `max`, `sum`, or `mean`.

```python
def build_membrane_composite(path, marker_names, y0=None, y1=None, x0=None, x1=None, method="max"):
    """
    Build one membrane-like channel from one or more marker channels.

    method:
      - "max": good default; keeps strong boundary signal from any marker
      - "sum": emphasizes regions where multiple markers are present
      - "mean": softer composite
    """
    imgs = []

    for marker in marker_names:
        ch = read_ome_channel(path, marker)

        if None not in [y0, y1, x0, x1]:
            ch = crop_image(ch, y0, y1, x0, x1)

        imgs.append(normalize_percentile(ch))

    stack = np.stack(imgs, axis=0)

    if method == "max":
        composite = stack.max(axis=0)
    elif method == "sum":
        composite = stack.sum(axis=0)
        composite = normalize_percentile(composite)
    elif method == "mean":
        composite = stack.mean(axis=0)
    else:
        raise ValueError("method must be one of: 'max', 'sum', or 'mean'")

    return composite.astype(np.float32)
```
:::

:::{dropdown} Display helpers

Before trusting a segmentation result, we need to look at it. The next two helper functions make it easier to display grayscale images and overlay segmentation boundaries on top of the original signal.

```python
def show_gray(img, title=None, figsize=(7, 7)):
    plt.figure(figsize=figsize)
    plt.imshow(normalize_percentile(img), cmap="gray")
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()
```

```python
def show_segmentation_overlay(base_img, masks, title=None, figsize=(8, 8)):
    """
    Display grayscale image with Cellpose mask outlines overlaid.
    """
    base = normalize_percentile(base_img)
    outlines = utils.masks_to_outlines(masks)

    plt.figure(figsize=figsize)
    plt.imshow(base, cmap="gray")
    plt.contour(outlines, levels=[0.5], linewidths=0.5)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()
```

The overlay function is especially important. AI-based segmentation can look impressive, but we should still check whether the mask boundaries match the image structures.

:::

:::{dropdown} Cellpose-SAM wrapper
Different Cellpose examples and versions may return slightly different numbers of outputs from `model.eval`. To make the rest of the code cleaner, we can define a small wrapper that handles both common cases.

```python
def run_cellpose_sam(model, img, channel_axis=None, **kwargs):
    """
    Small compatibility wrapper because Cellpose examples/docs sometimes show
    three or four return values depending on version/context.
    """
    output = model.eval(
        img,
        channel_axis=channel_axis,
        **kwargs
    )

    if len(output) == 3:
        masks, flows, styles = output
        diams = None
    elif len(output) == 4:
        masks, flows, styles, diams = output
    else:
        raise RuntimeError(f"Unexpected Cellpose output length: {len(output)}")

    return masks, flows, styles, diams
```
:::

:::{dropdown} Exporting labeled masks to GeoJSON

After segmentation, the output mask is a labeled image: background is `0`, and each segmented object has a positive integer label. This is similar to the labeled images we created with classical segmentation.

For downstream work, it is often useful to export these labels to another tool. In this example, we convert labeled masks to GeoJSON so they can be opened in QuPath.

```python
def labels_to_features(
    lab: np.ndarray,
    object_type: str = "annotation",
    connectivity: int = 4,
    transform: Affine = None,
    mask=None,
    downsample: float = 1.0,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    include_labels: bool = False,
    classification: dict | None = None,
) -> list:
    """
    Create GeoJSON features from a labeled image.
    """
    if lab.ndim != 2:
        raise ValueError("lab must be a 2D labeled image")

    features = []

    if lab.dtype == bool:
        if mask is None:
            mask = lab
        lab = lab.astype(np.uint8)
    else:
        if mask is None:
            mask = lab > 0

        if not np.issubdtype(lab.dtype, np.integer):
            lab = lab.astype(np.int32)

    if transform is None:
        transform = Affine.translation(x_offset, y_offset) * Affine.scale(downsample)

    for geom, value in rasterio.features.shapes(
        lab,
        mask=mask,
        connectivity=connectivity,
        transform=transform,
    ):
        if value == 0:
            continue

        props = {
            "objectType": object_type,
        }

        if include_labels:
            props["measurements"] = [
                {
                    "name": "Label",
                    "value": int(value),
                }
            ]

        if classification is not None:
            props["classification"] = classification

        feature = {
            "type": "Feature",
            "geometry": geom,
            "properties": props,
        }

        features.append(feature)

    return features
```

```python
def labels_to_geojson(
    lab: np.ndarray,
    output_path: str,
    object_type: str = "annotation",
    connectivity: int = 4,
    downsample: float = 1.0,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    include_labels: bool = True,
    classification: dict | None = None,
) -> dict:
    """
    Convert a labeled image to a GeoJSON FeatureCollection and save it.
    """
    features = labels_to_features(
        lab=lab,
        object_type=object_type,
        connectivity=connectivity,
        downsample=downsample,
        x_offset=x_offset,
        y_offset=y_offset,
        include_labels=include_labels,
        classification=classification,
    )

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    with open(output_path, "w") as f:
        json.dump(geojson, f)

    print(f"Saved {len(features)} objects to {output_path}")

    return geojson
```

This export step is useful because it connects segmentation in Python with annotation and visualization workflows in tools such as QuPath.

:::

## Downloading the example image

For this example, we will use a tonsil image from the HuBMAP repository. The image is spatial proteomics data acquired using the Phenocycler instrument, previously known as CODEX.

In your shell:
```bash
# Download tonsil image from HuBMAP repository
# Dataset: https://portal.hubmapconsortium.org/browse/dataset/5c5ac37add144e8a9707d2cd7791e694
wget -nc https://assets.hubmapconsortium.org/5034ca0cc4b0aa7a8d4d796474e8a9ab/ometiff-pyramids/stitched/expressions/reg1_stitched_expressions.ome.tif
```
After that we can open load image in python:

```python
img_path = Path("./reg1_stitched_expressions.ome.tif")
assert img_path.exists(), f"File not found: {img_path}"
print(img_path)
```

```python
channel_names = get_ome_channel_names(img_path)

print(f"Found {len(channel_names)} channels:")
for i, name in enumerate(channel_names):
    print(f"{i:02d}: {name}", end=" ")
```
Inspecting the channel names is an important step. AI-based segmentation does not remove the need to understand the data. We still need to decide which channels contain useful nuclear or membrane information.


## Segmentation option 1: DAPI-only Cellpose-SAM

The first segmentation option uses only the DAPI channel. This is a good starting point when nuclei are clear and the goal is to obtain nucleus-centered segmentation.


### Preparing the nucleus channel

We start with a simple segmentation setup using only the nucleus channel. Here, the selected nucleus channel is `"DAPI-02"`.

Because the full image is large, we first work on a crop. This is a practical habit: test the workflow on a smaller region first, then scale up once the result looks reasonable.

```python
# Required nucleus channel
NUCLEUS_CHANNEL = "DAPI-02"

# Work on a crop first. Set USE_CROP = False to segment the full image.
USE_CROP = True

y0, y1 = 2000, 4000
x0, x1 = 3000, 5000

print("Nucleus channel:", NUCLEUS_CHANNEL)
```

```python
dapi_full = read_ome_channel(img_path, NUCLEUS_CHANNEL)

if USE_CROP:
    dapi = crop_image(dapi_full, y0, y1, x0, x1)
else:
    dapi = dapi_full

show_gray(dapi, title="DAPI")
```

### Cellpose-SAM application

We first load the model and define a small set of segmentation parameters.

```python
model = models.CellposeModel(gpu=True)
print("Cellpose-SAM model loaded.")

CP_PARAMS = dict(
    diameter=None,              # None lets Cellpose-SAM handle size automatically
    flow_threshold=0.6,         # increase if too few masks; decrease if bad/irregular masks
    cellprob_threshold=0.0,     # decrease for more/larger masks; increase to suppress dim false positives
    min_size=15,                # remove tiny objects
    batch_size=64,              # lower to 8/16 if GPU memory errors
    normalize=True,
)

```
The most important idea here is that these parameters are part of the analysis. They influence how many objects are detected, which objects are kept, and how conservative the segmentation is.

```python
masks_dapi, flows_dapi, styles_dapi, diams_dapi = run_cellpose_sam(
    model,
    dapi,
    channel_axis=None,
    **CP_PARAMS,
)

print("DAPI-only masks:", masks_dapi.shape)
print("Number of objects:", masks_dapi.max())

show_segmentation_overlay(
    dapi,
    masks_dapi,
    title=f"DAPI-only Cellpose-SAM segmentation: {masks_dapi.max()} objects"
)
```

### Exporting the DAPI-only segmentation

Once the DAPI-only segmentation has been produced, we can export it. Here, each labeled object is exported as a cell object in a GeoJSON file.

```python
classification = {
    "name": "Cellpose-SAM cell",
    "color": [255, 0, 0],
}
labels_to_geojson(
    lab=masks_dapi,
    output_path="masks_dapi_qupath.geojson",
    object_type="cell",
    connectivity=4,
    downsample=1.0,
    x_offset=x0 if USE_CROP else 0,
    y_offset=y0 if USE_CROP else 0,
    include_labels=True,
    classification=classification,
)

print("Done")
```

The `x_offset` and `y_offset` values are important when working with crops. They allow exported objects to be placed back into the coordinate system of the full image.

Now you can download `masks_dapi_qupath.geojson` geojson file to overlay segmented cells in QuPath. This provides a more natural and interactive way to explore data and perform sanity checks. For that use the following commands inside QuPath: `File -> Import objects from file...`.

## Segmentation option 2: DAPI plus membrane information

Nuclei are useful, but they do not always fully describe cell boundaries. A second option is to give the model both nuclear and membrane-like information. This can help when the goal is cell segmentation rather than only nucleus segmentation.

In this example, we use `"DAPI-02"` as the nucleus channel and combine `"CD20"` and `"CD3e"` into a membrane-like composite.

```python
# Required nucleus channel
NUCLEUS_CHANNEL = "DAPI-02"

# Candidate membrane / boundary-like markers from the panel
MEMBRANE_MARKERS = ["CD20", "CD3e"]

MEMBRANE_COMBINE_METHOD = "max"  # "max", "sum", or "mean"

# Work on a crop first. Set USE_CROP = False to segment the full image.
USE_CROP = True

y0, y1 = 2000, 4000
x0, x1 = 5000, 7000
```


```python
dapi_full = read_ome_channel(img_path, NUCLEUS_CHANNEL)

if USE_CROP:
    dapi = crop_image(dapi_full, y0, y1, x0, x1)
else:
    dapi = dapi_full

membrane = build_membrane_composite(
    img_path,
    marker_names=MEMBRANE_MARKERS,
    y0=y0 if USE_CROP else None,
    y1=y1 if USE_CROP else None,
    x0=x0 if USE_CROP else None,
    x1=x1 if USE_CROP else None,
    method=MEMBRANE_COMBINE_METHOD,
)
```

### Visualizing the input channels

Before running the model, we should inspect the channels we are about to provide. This helps us check whether the membrane composite actually contains useful boundary information.

```python
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(normalize_percentile(dapi), cmap="gray")
axs[0].set_title("DAPI / nucleus channel")
axs[0].axis("off")

axs[1].imshow(normalize_percentile(membrane), cmap="gray")
axs[1].set_title("Membrane")
axs[1].axis("off")

plt.show()
```

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(normalize_percentile(dapi), cmap="gray")
axes[0].set_title("DAPI")
axes[0].axis("off")

axes[1].imshow(normalize_percentile(membrane), cmap="gray")
axes[1].set_title("Membrane composite")
axes[1].axis("off")

# Simple RGB preview: membrane = green, DAPI = blue
rgb = np.zeros((*dapi.shape, 3), dtype=np.float32)
rgb[..., 1] = normalize_percentile(membrane)
rgb[..., 2] = normalize_percentile(dapi)

axes[2].imshow(rgb)
axes[2].set_title("Preview: membrane green, DAPI blue")
axes[2].axis("off")

plt.tight_layout()
plt.show()
```

The RGB preview is only for visualization, but it is very useful. It lets us see whether the nuclear and membrane information line up in a biologically reasonable way.

### Preparing the multi-channel Cellpose input

For the multi-channel Cellpose-SAM run, we arrange the input image as `(C, Y, X)`, where `C` is the channel dimension. Here:

- channel 0 is the membrane composite,
- channel 1 is DAPI.

Luckily, the order of the membrane and nucleus isn't important for cellpose-sam. Therefore, we can run it safely without needing to memorize what should come first.
```python
# Shape: C x Y x X
# Channel 0 = membrane composite
# Channel 1 = DAPI
img_dapi_membrane = np.stack(
    [
        normalize_percentile(membrane),
        normalize_percentile(dapi),
    ],
    axis=0,
).astype(np.float32)

print("Cellpose input shape:", img_dapi_membrane.shape)
```

### Running Cellpose-SAM on DAPI plus membrane

Now we run the same model, but this time with both membrane and nucleus information.

```python
masks_dapi_membrane, flows_dapi_membrane, styles_dapi_membrane, diams_dapi_membrane = run_cellpose_sam(
    model,
    img_dapi_membrane,
    channel_axis=0,
    **CP_PARAMS,
)

print("DAPI + membrane masks:", masks_dapi_membrane.shape)
print("Number of objects:", masks_dapi_membrane.max())

```

For inspection, we overlay the segmentation result on the membrane channel, because cell boundaries are often easier to judge there.

```python
show_segmentation_overlay(
    img_dapi_membrane[0, 1000:1500, 1000:1500],
    masks_dapi_membrane[1000:1500, 1000:1500],
    title=f"DAPI + membrane Cellpose-SAM segmentation: {masks_dapi_membrane.max()} objects"
)
```

### Exporting the DAPI plus membrane segmentation

Finally, we export the multi-channel segmentation result as GeoJSON. This gives us a file that can be opened in QuPath or used in downstream workflows.

```python
classification = {
    "name": "Cellpose-SAM cell",
    "color": [255, 0, 0],
}

labels_to_geojson(
    lab=masks_dapi_membrane,
    output_path="masks_dapi_with_membrane_qupath.geojson",
    object_type="cell",
    connectivity=4,
    downsample=1.0,
    x_offset=x0 if USE_CROP else 0,
    y_offset=y0 if USE_CROP else 0,
    include_labels=True,
    classification=classification,
)

print("Done")

```

## Practical notes

AI-assisted segmentation can be very useful, but it is not magic. The result still depends on image quality, the channels we provide, and the parameters we choose. In this example, several choices matter:

- using a crop first makes testing faster,
- DAPI-only segmentation is a useful baseline,
- adding membrane markers may improve cell boundary detection,
- `flow_threshold`, `cellprob_threshold`, and `min_size` affect the final masks,
- overlays are essential for checking whether the segmentation makes biological sense.

```{admonition} Warning
:class: warning

AI-based segmentation still needs quality control. Always inspect the segmentation overlay, document the model and parameters, and check that the output matches the biological structures you intend to measure.
```
