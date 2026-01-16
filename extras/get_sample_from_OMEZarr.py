import zarr
import math
import copy
import json
import os

src = zarr.open("/home/alexr/Vesuvius/Data/scroll1.volpkg/volumes/54keV_7.91um_Scroll1A.zarr", mode="r")
store = zarr.NestedDirectoryStore("/home/alexr/Vesuvius/Data/MiniVolumes/scroll1/center_54keV_7.91um_Scroll1A.zarr")
dst = zarr.group(store=store, overwrite=True)
#center z, center y, center x
CZ, CY, CX = 7000, 4000, 4000
HALF = 1024 // 2
z0, z1 = CZ - HALF, CZ + HALF
y0, y1 = CY - HALF, CY + HALF
x0, x1 = CX - HALF, CX + HALF

dst.attrs.update(copy.deepcopy(src.attrs))

for level_name in src.array_keys():
    level = int(level_name)
    src_arr = src[level_name]
    scale = 2 ** level

    # scale bounding box
    z0s, z1s = z0 // scale, math.ceil(z1 / scale)
    y0s, y1s = y0 // scale, math.ceil(y1 / scale)
    x0s, x1s = x0 // scale, math.ceil(x1 / scale)

    cropped_shape = (z1s - z0s, y1s - y0s, x1s - x0s)
    print(f"Level {level}: {cropped_shape}")

    dst_arr = dst.create(
        name=level_name,
        shape=cropped_shape,
        chunks=src_arr.chunks,
        dtype=src_arr.dtype,
        compressor=src_arr.compressor,
        order=src_arr.order,
        overwrite=True,
    )
    dst_arr[:] = src_arr[z0s:z1s, y0s:y1s, x0s:x1s]

multiscales = dst.attrs["multiscales"][0]
meta_path = os.path.join(src.store.path, "meta.json")
with open(meta_path, "r") as f:
    meta = json.load(f)

voxelsize = meta.get("voxelsize") 
for level, dataset in enumerate(multiscales["datasets"]):
    # make sure original voxel size stays same
    transform = dataset["coordinateTransformations"][0]
    transform["translation"] = [z0 * voxelsize, y0 * voxelsize, x0 * voxelsize]

# make sure it actually worked
dst = zarr.group(store=store)
for k in dst.array_keys():
    print(k, dst[k].shape, dst[k].chunks)

#write the meta.json
level0 = dst["0"]
cropped_meta = {
    "height": level0.shape[1],          # y dimension
    "width": level0.shape[2],           # x dimension
    "slices": level0.shape[0],          # z dimension
    "voxelsize": meta.get("voxelsize"), # same voxel size
    "min": meta.get("min", 0),
    "max": meta.get("max", 65535),
    "name": meta.get("name") + "_center", #add that little bit on
    "type": meta.get("type", "vol"),
    "uuid": meta.get("uuid") + "_piece", # its a piece of the original volume so this should do the trick
    "format": meta.get("format", "zarr")
}

# write the new json
meta_out_path = os.path.join(store.path, "meta.json")
with open(meta_out_path, "w") as f:
    json.dump(cropped_meta, f, indent=4)
