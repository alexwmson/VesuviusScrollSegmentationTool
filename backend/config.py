AW_BIN = "/usr/local/bin/aw_segmentation_algorithm"
VC_BIN = "/usr/local/bin/vc_render_tifxyz"
PNG_BIN = "/usr/local/bin/generate_png_renders"
VOLUME_MAP = { #scale in microns
    "center_scroll1": ["/home/alexr/Vesuvius/Data/MiniVolumes/scroll1/center_scroll1/center_54keV_7.91um_Scroll1A.zarr", "7.81"],
    "leftedge_scroll1": ["/home/alexr/Vesuvius/Data/MiniVolumes/scroll1/leftedge_scroll1/leftedge_54keV_7.91um_Scroll1A.zarr", "7.81"]
}

OUTPUT_ROOT = "/home/alexr/gitstuff/vesuviusscrollsegmentationtool/output"
ALGO_IMAGE = "aw_segmentation:local"
PNGRENDER_IMAGE = "pngrender:latest"