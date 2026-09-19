import json
import struct

ASSET_IMAGE = "tests/assets/test.jpg"


def write_safetensors(path, tensors_header, data=b""):
    header_bytes = json.dumps(tensors_header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(data)


def h3_shaped_header():
    return {
        "blocks.0.attn.qkv_proj.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
        "video_patch_proj.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [2, 4]},
        "__metadata__": {"format": "pt"},
    }
