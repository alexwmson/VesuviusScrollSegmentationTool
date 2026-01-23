import zarr
import numpy as np
import imageio.v2 as imageio
import os
zarr_path = "/home/alexr/Vesuvius/Data/MiniVolumes/scroll1/leftedge_54keV_7.91um_Scroll1A.zarr"
output_root = "/home/alexr/Vesuvius/Data/MiniVolumes/scroll1/leftedge_54keV_7.91um_Scroll1A.zarr/pngrenders"

axes = {
    "z": 0,  # axial
    "y": 1,  # coronal
    "x": 2,  # sagittal
}

def normalize_to_uint8(img):
    """Normalize image to uint8 for PNG."""
    img = img.astype(np.float32)
    min_val = img.min()
    max_val = img.max()

    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = np.zeros_like(img)

    return (img * 255).astype(np.uint8)

z = zarr.open(zarr_path, mode="r")
volume = z["0"]

print("Volume shape:", volume.shape)
print("Volume dtype:", volume.dtype)

for axis_name, axis in axes.items():
    axis_dir = os.path.join(output_root, axis_name)
    os.makedirs(axis_dir, exist_ok=True)

    num_slices = volume.shape[axis]
    print(f"Exporting {num_slices / 4} slices along {axis_name}-axis")

    for i in range(0, num_slices, 4):
        if axis == 0:      # Z
            slice_2d = volume[i, :, :]
        elif axis == 1:    # Y
            slice_2d = volume[:, i, :]
        elif axis == 2:    # X
            slice_2d = volume[:, :, i]

        img = normalize_to_uint8(slice_2d)

        filename = f"{i+1:04d}.png"
        imageio.imwrite(os.path.join(axis_dir, filename), img)

print("Done.")