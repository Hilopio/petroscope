# Panorama Stitching Module

A lightweight module for stitching panoramic images using feature matching and bundle adjustment.

## Implementation Details

### Alignment Stage
- **Feature Matching**: Implemented via [LoFTR](https://github.com/zju3dv/LoFTR) in `alignment.py`
- **Refinement**: Bundle Adjustment optimization in `bundle_adjustment.py`

## Quick Start

```bash
python3 path/to/align_panorama.py path/to/folder-with-tiles path/to/output-file.jpg