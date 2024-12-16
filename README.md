# Image Matching

Image matching component.

## Installation

```bash
pip install dist/image_matching-0.1.0-*.whl
```


## Usage

```python
from image_matching import match_images

matches = match_images(
    rgb_image="data/00000_V.JPG",  # RGB image path
    ir_image="data/00000_T.JPG",  # Thermal image path
)

print(matches)
```
