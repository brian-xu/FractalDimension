# Fractal Dimension of 2D Images

This algorithm calculates the fractal dimension of a 2D image using a modified box-counting approach as described by [Wen-Li Lee and Kai-Sheng Hsieh](https://doi.org/10.1016/j.sigpro.2009.12.010).

## Usage

```python
import numpy as np
import cv2
from FractalDimension import fractal_dimension

#test data
image = cv2.imread('image.png', 0)

fd = fractal_dimension(image)
print(f"Fractal dimension of the image: {fd}")
```

### Note: OpenCV is not a strict requirement, but the image is expected to be formatted identically to OpenCV's grayscale representation.
