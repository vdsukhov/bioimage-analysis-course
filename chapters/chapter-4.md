
kernelspec:
    name: python3
    display_name: 'Python 3'


# Classical segmentations

## From binary masks to segmented objects

### Why a binary mask is only the first step

A binary mask is often the first useful result of a segmentation pipeline. After thresholding, each pixel is classified as either foreground or background, which already gives us a much simpler view of the image. In some cases, that may be enough — for example, if we only care about the total foreground area.

But in many biological applications, a binary mask is only the beginning. It tells us where the foreground is, but not how many objects are present or which pixels belong to which object. If our goal is to count cells, measure nuclei, or compare object shapes, then foreground/background separation is not yet sufficient.

### From foreground pixels to individual objects

Turning a binary mask into meaningful segmented objects is often harder than it looks. Real biological images are messy: objects may touch, small noisy fragments may appear, and single objects may contain holes or irregular boundaries. As a result, a thresholded mask may capture the foreground reasonably well while still failing to separate it into biologically meaningful units.

This is where the main goal of segmentation becomes more specific. Instead of asking only whether a pixel belongs to the foreground, we now ask which **individual object** it belongs to. In the following sections, we will build that transition step by step: first by labeling connected regions, then by separating touching objects, and finally by measuring the resulting segmented regions.



## Connected-component labeling

### What connected-component labeling does

Once we have a binary mask, the next step is often to identify which foreground pixels belong together. **Connected-component labeling** does exactly that: it scans the binary image and assigns a unique label to each connected foreground region.

The result is no longer just a mask of `True` and `False` values. Instead, each object gets its own integer label, which makes it possible to count objects and analyze them one by one.

### From one foreground mask to many objects

This is an important conceptual shift. In a binary mask, all foreground pixels are treated the same way. After connected-component labeling, the foreground is split into separate regions, and each region is treated as an individual object.

For example, if a mask contains three disconnected nuclei, labeling will assign them different values such as `1`, `2`, and `3`. Background pixels usually remain `0`.

### What “connected” means

The key idea here is **connectivity**. Two foreground pixels are considered part of the same object if they are connected through neighboring foreground pixels. In 2D images, this is usually defined in one of two ways:

* **4-connectivity**: pixels are connected only through up, down, left, and right
* **8-connectivity**: diagonal neighbors also count as connected

This choice can affect the result. With 8-connectivity, diagonally touching pixels may be grouped into the same object, while with 4-connectivity they may remain separate.

### Why labeling is useful

Connected-component labeling is often the first step that turns a binary mask into something we can measure. Once each object has its own label, we can count how many objects are present, measure their size, locate their centroids, and compute many other properties.

At the same time, labeling does not solve every segmentation problem. If two touching cells are merged in the binary mask, they will still receive a single label. So connected-component labeling is powerful, but it depends on the quality of the mask that comes before it.



## Distance transform

### What the distance transform measures

The **distance transform** takes a binary image and, for each foreground pixel, measures how far that pixel is from the nearest background pixel. As a result, pixels near the boundary of an object get small values, while pixels deeper inside the object get larger values.

This turns a binary mask into a new kind of image: not just foreground versus background, but a map of how far each foreground location lies from the object boundary.

### Why this is useful

A distance-transformed image gives us more internal structure than the original binary mask. In particular, the centers of objects often appear as local maxima, because those pixels are farthest from the background.

This is especially useful when objects touch each other. In a binary mask, two touching cells may look like one merged region. But in the distance map, each cell may still contain its own peak, which gives us a clue that more than one object is present.

### A useful geometric intuition

You can think of the distance transform as asking: *if I start at this foreground pixel, how many steps can I take before I hit the background?* Pixels near the edge run out of space quickly, while pixels near the middle of an object can go farther.

Because of that, the distance transform often highlights the approximate centers of objects, even when their boundaries are touching.

### Why it often comes before watershed

On its own, the distance transform is not a segmentation. It does not assign labels or separate objects directly. What it does provide is a very useful intermediate representation: a smooth map whose peaks can be used as markers for individual objects.

That is why the distance transform is often used together with **watershed segmentation**. The distance map helps us find likely object centers, and watershed can then use that information to split touching regions more cleanly.



## Watershed segmentation

### The basic idea

The **watershed algorithm** is a classical way to separate connected regions into individual objects. The name comes from geography: imagine the image as a landscape, where pixel values represent height. If we slowly flood this landscape with water, the water fills the low regions first. As water rises from different starting points, boundaries are placed where the growing regions meet.

That is the central idea of watershed segmentation: it divides an image into regions by simulating a flooding process.

### Why it is useful in bioimage analysis

Watershed is especially useful when objects touch each other. In a binary mask, two neighboring nuclei may appear as one merged region. Thresholding can detect the foreground, but it often cannot tell where one object ends and the next begins.

This is where watershed helps. It can split a connected foreground region into multiple objects, provided we have some information about where the likely object centers are.

### Watershed on the distance transform

A very common strategy is to apply watershed not directly to the raw image, but to the **distance transform** of a binary mask. The reason is that the distance map often has one peak near the center of each object.

These peaks can be used as **markers** or **seeds**. Conceptually, each seed represents one object. The watershed algorithm then grows regions outward from those seeds until they meet, which creates a boundary between touching objects.

So the workflow is often:

1. start with a binary mask,
2. compute the distance transform,
3. find markers in the distance map,
4. apply watershed to split touching objects.

### Why markers matter

Watershed is powerful, but it also depends strongly on the choice of markers. If we place too few markers, multiple objects may remain merged. If we place too many, one object may be split into several parts.

So in practice, watershed is often used in a **marker-controlled** way. Instead of letting the algorithm decide everything on its own, we provide starting points that reflect our best guess about where the objects are.

This makes watershed much more stable and much more useful for real biological images.

### A practical intuition

You can think of watershed as a way of drawing borders between neighboring objects by asking: *if these two regions grow outward from their centers, where should they meet?* The answer is usually somewhere near the narrowest connection between them.

That is why watershed is often so effective for separating touching roundish objects such as nuclei, cells, or blobs.

### A common limitation

Watershed does not magically solve every segmentation problem. Its result depends on the quality of the binary mask, the distance transform, and especially the markers. If the mask is poor or the markers are wrong, the segmentation may still fail.

Even so, watershed is one of the most useful classical tools for turning merged foreground regions into separate labeled objects.

## Measuring labeled regions

### From segmentation to measurement

Once segmentation has separated the image into individual labeled objects, we can start to treat the result as data rather than just as a picture. Instead of only asking *where* objects are, we can now ask how many there are, how large they are, where they are located, and what shape they have.

This is one of the main reasons segmentation matters in bioimage analysis. A labeled image is not only useful for visualization — it is the starting point for quantitative measurements.

### What kinds of properties can be measured

A labeled region can be described in many ways. Some properties are very simple, such as:

* **area** — how many pixels belong to the object
* **centroid** — the center position of the object
* **bbox** — the bounding box around the object

Other properties describe shape in more detail, for example:

* **perimeter**
* **eccentricity**
* **major and minor axis length**

And if we also have the original intensity image, we can measure intensity-based features such as:

* **mean intensity**
* **maximum intensity**
* **minimum intensity**

So once objects are labeled, we can move from segmentation to object-based analysis.

### `regionprops` and `regionprops_table`

In `scikit-image`, a common way to measure labeled objects is with `regionprops` and `regionprops_table`.

* `regionprops` returns region measurements as Python objects, which is convenient for inspection and interactive exploration.
* `regionprops_table` returns the measurements in a tabular format, which is especially useful when we want to convert the results into a `pandas` DataFrame or export them for later analysis.

Both functions work on a labeled image, and both can optionally use the original intensity image if we want to include intensity-based measurements.

### Why this step matters

This is the point where segmentation becomes biologically useful. A binary mask tells us where the foreground is, and watershed or connected-component labeling helps us split that foreground into individual objects. Measurements are what let us compare cells, summarize populations, and connect image analysis to a biological question.

At the same time, measurements are only as meaningful as the segmentation they come from. If objects are merged, split, or badly outlined, the extracted properties may be misleading. So region measurements are powerful, but they always depend on the quality of the labels underneath them.


## Morphology-driven segmentation workflows

### Segmentation is usually a pipeline

In practice, classical segmentation is rarely done with a single operation. A threshold may detect the foreground, but the result often still needs some cleanup before it becomes useful. Small noisy fragments may appear, holes may remain inside objects, and nearby structures may still be connected. This is why segmentation is often better thought of as a **workflow** rather than a single step.

Morphological operations are especially useful in these workflows because they help refine binary masks in simple and interpretable ways. Instead of replacing segmentation, they usually support it by cleaning up the result before labeling, watershed, or measurement.

### A typical morphology-driven workflow

A common classical workflow looks something like this:

1. start with the raw image,
2. threshold it to obtain a binary mask,
3. clean the mask with morphological operations,
4. label or separate objects,
5. measure the final regions.

The exact steps depend on the image and the biological question, but the general pattern is very common: detect foreground first, then use morphology to make that foreground more meaningful.

### What morphology can help with

Morphological operations are useful because different steps solve different small problems:

* **opening** can remove tiny foreground specks,
* **closing** can fill small gaps,
* **fill holes** can make objects more solid,
* **remove small objects** can discard tiny connected components,
* **erosion** and **dilation** can help break or strengthen connections.

These operations are often most helpful when the thresholded mask is already close to correct, but still contains local imperfections that would interfere with later analysis.

### Why this matters

This workflow-based view is important because real biological images are rarely perfect. A thresholded mask does not need to be flawless immediately, but it should become good enough for the next step. Morphology helps bridge that gap.

At the same time, morphological cleanup should always be guided by the biological goal. A hole may be an artifact in one case and a meaningful structure in another. A small object may be noise, or it may be exactly the feature we want to detect. So morphology is powerful, but it should always be used with intention.

### A practical way to think about it

A useful mindset is this: morphology does not usually discover objects from scratch — it helps refine what earlier steps have already found. In that sense, morphology-driven workflows are not about replacing segmentation, but about turning a rough segmentation into something more stable, cleaner, and easier to measure.

```{admonition} Warning
:class: warning

Morphological cleanup can improve a segmentation a lot, but it can also remove real structures or distort object shape if used too aggressively. A good habit is to inspect the mask after each major step and make sure the changes support your biological question.
```

### Take-home message

Morphology-driven workflows are common because segmentation is usually not one perfect step, but a sequence of smaller improvements. Thresholding finds the foreground, morphology refines it, and later steps such as labeling, watershed, and measurement turn it into useful biological data.
