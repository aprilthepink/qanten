#!/usr/bin/env python3
"""Writes the small glTF test assets (no downloads, standard library only).

A tapered tube ("tentacle") skinned to a 3-joint chain, with one morph target
and four animations covering every interpolation mode:

    Bend     LINEAR       rotation of Mid and Tip about Z
    Hop      STEP         translation of Root
    Pulse    CUBICSPLINE  scale of Root
    Breathe  LINEAR       morph weight of Body

Outputs, next to this script:

    tentacle.gltf           buffers embedded as a base64 data URI
    tentacle.glb            binary container
    tentacle_external.gltf  + tentacle.bin, buffer referenced by relative path

Run from anywhere: python3 make_assets.py
"""

import base64
import json
import math
import os
import struct

RINGS, SEGMENTS = 16, 12
HEIGHT = 2.0
JOINT_Y = [0.0, 0.7, 1.4]

FLOAT, UBYTE, USHORT = 5126, 5121, 5123


def tube():
    positions, normals, joints, weights, bulge = [], [], [], [], []
    for r in range(RINGS):
        y = HEIGHT * r / (RINGS - 1)
        radius = 0.25 * (1.0 - 0.6 * y / HEIGHT)
        if y <= JOINT_Y[1]:
            j, w = (0, 1), y / JOINT_Y[1]
        elif y <= JOINT_Y[2]:
            j, w = (1, 2), (y - JOINT_Y[1]) / (JOINT_Y[2] - JOINT_Y[1])
        else:
            j, w = (1, 2), 1.0
        for s in range(SEGMENTS):
            a = 2 * math.pi * s / SEGMENTS
            n = (math.cos(a), 0.0, math.sin(a))
            positions.append((radius * n[0], y, radius * n[2]))
            normals.append(n)
            joints.append((j[0], j[1], 0, 0))
            weights.append((1.0 - w, w, 0.0, 0.0))
            amount = 0.1 * math.sin(math.pi * y / HEIGHT)
            bulge.append((amount * n[0], 0.0, amount * n[2]))
    indices = []
    for r in range(RINGS - 1):
        for s in range(SEGMENTS):
            a = r * SEGMENTS + s
            b = r * SEGMENTS + (s + 1) % SEGMENTS
            c, d = a + SEGMENTS, b + SEGMENTS
            indices += [a, c, b, b, c, d]
    return positions, normals, joints, weights, bulge, indices


def quat_z(angle):
    return (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))


class Builder:
    def __init__(self):
        self.bin = bytearray()
        self.views, self.accessors = [], []

    def add(self, fmt, rows, ctype, kind, minmax=False):
        while len(self.bin) % 4:
            self.bin.append(0)
        offset = len(self.bin)
        for row in rows:
            row = row if isinstance(row, (tuple, list)) else (row,)
            self.bin += struct.pack("<" + fmt * len(row), *row)
        self.views.append(
            {"buffer": 0, "byteOffset": offset, "byteLength": len(self.bin) - offset}
        )
        acc = {
            "bufferView": len(self.views) - 1,
            "componentType": ctype,
            "count": len(rows),
            "type": kind,
        }
        if minmax:
            cols = list(zip(*[r if isinstance(r, (tuple, list)) else (r,) for r in rows]))
            acc["min"] = [min(c) for c in cols]
            acc["max"] = [max(c) for c in cols]
        self.accessors.append(acc)
        return len(self.accessors) - 1


def document():
    positions, normals, joints, weights, bulge, indices = tube()
    b = Builder()
    pos = b.add("f", positions, FLOAT, "VEC3", minmax=True)
    nrm = b.add("f", normals, FLOAT, "VEC3")
    jnt = b.add("B", joints, UBYTE, "VEC4")
    wgt = b.add("f", weights, FLOAT, "VEC4")
    tgt = b.add("f", bulge, FLOAT, "VEC3", minmax=True)
    idx = b.add("H", indices, USHORT, "SCALAR")
    # Inverse bind matrices: the joints sit at (0, JOINT_Y[k], 0) in the bind pose.
    ibm = [
        (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, -y, 0, 1)
        for y in JOINT_Y
    ]
    ibm_acc = b.add("f", ibm, FLOAT, "MAT4")

    t3 = b.add("f", [0.0, 1.0, 2.0], FLOAT, "SCALAR", minmax=True)
    bend_mid = b.add("f", [quat_z(0), quat_z(0.6), quat_z(0)], FLOAT, "VEC4")
    bend_tip = b.add("f", [quat_z(0), quat_z(0.8), quat_z(0)], FLOAT, "VEC4")
    t2 = b.add("f", [0.0, 1.0], FLOAT, "SCALAR", minmax=True)
    hop = b.add("f", [(0, 0, 0), (0, 0.5, 0)], FLOAT, "VEC3")
    # CUBICSPLINE: (in-tangent, value, out-tangent) per keyframe.
    pulse = b.add(
        "f",
        [(0, 0, 0), (1, 1, 1), (0, 0, 0), (0, 0, 0), (2, 2, 2), (0, 0, 0)],
        FLOAT,
        "VEC3",
    )
    breathe = b.add("f", [0.0, 1.0], FLOAT, "SCALAR")

    def sampler(inp, out, interpolation):
        return {"input": inp, "output": out, "interpolation": interpolation}

    def channel(s, node, path):
        return {"sampler": s, "target": {"node": node, "path": path}}

    doc = {
        "asset": {"version": "2.0", "generator": "qanten make_assets.py"},
        "scene": 0,
        "scenes": [{"nodes": [0, 3]}],
        "nodes": [
            {"name": "Root", "children": [1]},
            {"name": "Mid", "translation": [0, JOINT_Y[1], 0], "children": [2]},
            {"name": "Tip", "translation": [0, JOINT_Y[2] - JOINT_Y[1], 0]},
            {"name": "Body", "mesh": 0, "skin": 0},
        ],
        "meshes": [
            {
                "name": "Tentacle",
                "weights": [0.25],
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": pos,
                            "NORMAL": nrm,
                            "JOINTS_0": jnt,
                            "WEIGHTS_0": wgt,
                        },
                        "indices": idx,
                        "targets": [{"POSITION": tgt}],
                    }
                ],
            }
        ],
        "skins": [{"name": "Chain", "joints": [0, 1, 2], "inverseBindMatrices": ibm_acc}],
        "animations": [
            {
                "name": "Bend",
                "samplers": [sampler(t3, bend_mid, "LINEAR"), sampler(t3, bend_tip, "LINEAR")],
                "channels": [channel(0, 1, "rotation"), channel(1, 2, "rotation")],
            },
            {
                "name": "Hop",
                "samplers": [sampler(t2, hop, "STEP")],
                "channels": [channel(0, 0, "translation")],
            },
            {
                "name": "Pulse",
                "samplers": [sampler(t2, pulse, "CUBICSPLINE")],
                "channels": [channel(0, 0, "scale")],
            },
            {
                "name": "Breathe",
                "samplers": [sampler(t2, breathe, "LINEAR")],
                "channels": [channel(0, 3, "weights")],
            },
        ],
        "accessors": b.accessors,
        "bufferViews": b.views,
        "buffers": [{"byteLength": len(b.bin)}],
    }
    return doc, bytes(b.bin)


def write_json(path, doc):
    with open(path, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
        f.write("\n")


def glb(doc, data):
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    data += b"\0" * (-len(data) % 4)
    chunks = struct.pack("<I4s", len(js), b"JSON") + js
    chunks += struct.pack("<I4s", len(data), b"BIN\0") + data
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    doc, data = document()

    embedded = json.loads(json.dumps(doc))
    embedded["buffers"][0]["uri"] = (
        "data:application/octet-stream;base64," + base64.b64encode(data).decode()
    )
    write_json(os.path.join(here, "tentacle.gltf"), embedded)

    external = json.loads(json.dumps(doc))
    external["buffers"][0]["uri"] = "tentacle.bin"
    write_json(os.path.join(here, "tentacle_external.gltf"), external)
    with open(os.path.join(here, "tentacle.bin"), "wb") as f:
        f.write(data)

    with open(os.path.join(here, "tentacle.glb"), "wb") as f:
        f.write(glb(doc, data))


if __name__ == "__main__":
    main()
