AW_BIN = "/usr/local/bin/aw_segmentation_algorithm"
VC_BIN = "/usr/local/bin/vc_render_tifxyz"
PNG_BIN = "/usr/local/bin/generate_png_renders"
VOLUME_MAP = {
    "volume1": ["/home/alexr/Vesuvius/Data/scroll1.volpkg/volumes/54keV_7.91um_Scroll1A.zarr", "7.81"] #scale in microns
}

OUTPUT_ROOT = "/home/alexr/gitstuff/vesuviusscrollsegmentationtool/output"
ALGO_IMAGE = "aw_segmentation:local"
PNGRENDER_IMAGE = "pngrender:latest"