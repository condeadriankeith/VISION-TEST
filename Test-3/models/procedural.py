"""Procedural 3D meshes — zero external assets.

Each builder returns (verts (N,3) float64 in local space ~[-1,1],
edges (M,2) int, colors (N,3) uint8 BGR). All take time `t` for animation:
  flowers   — petals bloom + orbit
  dragon    — wings flap, fiery spine
  butterfly — articulated dual wings, harmonic flap
  tree      — static trunk + gently swaying canopy
"""

from __future__ import annotations

import math
import numpy as np


def _empty():
    return [], [], []


def build_flowers(t: float = 0.0):
    """3 blooming lilies: stem + 6 orbiting petals each + center."""
    verts, edges, colors = [], [], []
    # BGR
    petal = (190, 130, 255)
    petal2 = (170, 100, 240)
    center_c = (90, 225, 255)
    stem_c = (120, 200, 90)

    def add(v, c):
        verts.append([float(v[0]), float(v[1]), float(v[2])])
        colors.append(c)
        return len(verts) - 1

    bloom = 0.5 + 0.5 * math.sin(t * 1.6)          # 0..1 open amount
    centers = [(-0.55, 0.05, 0.0), (0.0, 0.25, -0.1), (0.55, 0.05, 0.0)]
    for ci, (cx, cy, cz) in enumerate(centers):
        phase = ci * 2.1
        base = add((cx, cy - 0.55, cz), stem_c)
        top = add((cx, cy, cz), center_c)
        edges.append((base, top))
        # leaves
        l1 = add((cx + 0.16, cy - 0.35, cz + 0.05), stem_c)
        edges.append((base, l1))
        # 6 petals orbiting + blooming
        for k in range(6):
            ang = phase + t * 0.7 + k * math.pi / 3.0
            r = 0.16 + 0.22 * bloom
            lift = 0.10 + 0.30 * bloom
            px = cx + r * math.cos(ang)
            py = cy + lift + 0.05 * math.sin(t * 2.0 + k)
            pz = cz + r * math.sin(ang) * 0.7
            tip = add((px, py, pz), petal if k % 2 == 0 else petal2)
            edges.append((top, tip))
            # petal outline to neighbour handled after loop via ring
        # ring between petal tips (last 6 verts)
        ring = list(range(len(verts) - 6, len(verts)))
        for k in range(6):
            edges.append((ring[k], ring[(k + 1) % 6]))
        # stamen
        s = add((cx, cy + 0.16 + 0.04 * math.sin(t * 3 + phase), cz), center_c)
        edges.append((top, s))
    return (np.array(verts, dtype=np.float64),
            np.array(edges, dtype=np.int64),
            np.array(colors, dtype=np.uint8))


def build_dragon(t: float = 0.0):
    """Red dragon / phoenix: spine chain, head, flapping wings, tail."""
    verts, edges, colors = [], [], []
    body = (50, 50, 235)
    wing_c = (70, 110, 255)
    glow = (60, 160, 255)

    def add(v, c):
        verts.append([float(v[0]), float(v[1]), float(v[2])])
        colors.append(c)
        return len(verts) - 1

    flap = math.sin(t * 6.0)  # -1..1 wing beat
    # spine from tail (-x) to neck (+x)
    spine = []
    for i in range(7):
        x = -0.7 + i * 0.20
        y = 0.05 * math.sin(i * 0.9 + t * 2.0)
        z = 0.06 * math.cos(i * 0.7 + t * 1.5)
        spine.append(add((x, y, z), body))
        if i:
            edges.append((spine[i - 1], spine[i]))
    # head + horns + jaw
    head = add((0.78, 0.10, 0.0), glow)
    edges.append((spine[-1], head))
    jaw = add((0.72, -0.06, 0.0), body)
    edges.append((head, jaw))
    h1 = add((0.66, 0.24, 0.08), glow)
    h2 = add((0.66, 0.24, -0.08), glow)
    edges.append((head, h1)); edges.append((head, h2))
    # wings: shoulder at spine[4]; each wing = 2x2 grid flapping in z/y
    sh = verts[spine[4]]
    for side in (+1, -1):
        wing_root = add((sh[0], sh[1], sh[2]), wing_c)
        edges.append((spine[4], wing_root))
        lift = flap * 0.45 * side
        pts = []
        for ix, iy in ((0.35, 0.15), (0.62, 0.28), (0.30, 0.45), (0.60, 0.58)):
            x = sh[0] + ix * 0.9 - 0.1
            y = sh[1] + iy + abs(lift) * 0.6
            z = side * (0.35 + ix * 0.9) + lift * (0.4 + iy)
            pts.append(add((x, y, z), wing_c))
        # wing quads -> edges
        edges += [(wing_root, pts[0]), (wing_root, pts[2]),
                  (pts[0], pts[1]), (pts[2], pts[3]),
                  (pts[0], pts[2]), (pts[1], pts[3])]
        # wing fingers
        for p in pts:
            tip = add((verts[p][0] + 0.12, verts[p][1] + 0.10, verts[p][2] + side * 0.12), glow)
            edges.append((p, tip))
    # tail flame tips
    for k in range(3):
        tip = add((-0.85 - 0.08 * k, 0.08 * math.sin(t * 4 + k * 2.0) + 0.05 * k, 0.05 * k), glow)
        edges.append((spine[0], tip))
    # legs (simple)
    for i in (2, 4):
        f = add((verts[spine[i]][0], verts[spine[i]][1] - 0.22, verts[spine[i]][2] + 0.08), body)
        edges.append((spine[i], f))
    return (np.array(verts, dtype=np.float64),
            np.array(edges, dtype=np.int64),
            np.array(colors, dtype=np.uint8))


def build_butterfly(t: float = 0.0):
    """Blue morpho: slender body + two articulated wings, harmonic flap."""
    verts, edges, colors = [], [], []
    wing_c = (255, 130, 50)
    wing_edge = (255, 190, 120)
    body_c = (235, 235, 235)

    def add(v, c):
        verts.append([float(v[0]), float(v[1]), float(v[2])])
        colors.append(c)
        return len(verts) - 1

    flap = math.sin(t * 7.0)  # fast flutter
    fold = 0.25 + 0.55 * abs(flap)  # wing open amount
    # body
    b0 = add((0, -0.45, 0), body_c)
    b1 = add((0, 0.0, 0.02), body_c)
    b2 = add((0, 0.45, 0), body_c)
    edges += [(b0, b1), (b1, b2)]
    # head + antennae
    hd = add((0, 0.55, 0), body_c)
    edges.append((b2, hd))
    a1 = add((-0.12, 0.72, 0.05), body_c); a2 = add((0.12, 0.72, 0.05), body_c)
    edges += [(hd, a1), (hd, a2)]
    # wings mirrored
    for side in (-1, +1):
        # inner + outer span scaled by fold (articulation)
        x0, x1, x2 = side * 0.08, side * (0.25 + 0.55 * fold), side * (0.55 + 0.55 * fold)
        y_top, y_bot = 0.42, -0.30
        z = 0.10 * flap * side + 0.05
        p_in_top = add((x0, 0.30, 0.0), wing_c)
        p_out_top = add((x1, y_top, z), wing_c)
        p_tip_top = add((x2, 0.30, z * 1.6), wing_edge)
        p_in_bot = add((x0, -0.12, 0.0), wing_c)
        p_out_bot = add((x1 * 0.9, y_bot, z), wing_c)
        p_tip_bot = add((x2 * 0.85, -0.18, z * 1.6), wing_edge)
        edges += [(b1, p_in_top), (p_in_top, p_out_top), (p_out_top, p_tip_top),
                  (p_in_top, p_tip_top),
                  (b1, p_in_bot), (p_in_bot, p_out_bot), (p_out_bot, p_tip_bot),
                  (p_in_bot, p_tip_bot), (p_out_top, p_out_bot)]
    return (np.array(verts, dtype=np.float64),
            np.array(edges, dtype=np.int64),
            np.array(colors, dtype=np.uint8))


def build_tree(t: float = 0.0):
    """Bonsai / cosmic tree: recursive trunk + foliage canopy."""
    verts, edges, colors = [], [], []
    trunk_c = (75, 135, 205)
    leaf_c = (130, 225, 140)
    leaf_c2 = (100, 200, 170)
    glow_c = (160, 255, 200)

    def add(v, c):
        verts.append([float(v[0]), float(v[1]), float(v[2])])
        colors.append(c)
        return len(verts) - 1

    sway = 0.03 * math.sin(t * 1.2)
    root = add((0, -0.7, 0), trunk_c)

    def branch(p0, direction, length, depth, phase):
        d = np.array(direction, dtype=np.float64)
        d = d / (np.linalg.norm(d) + 1e-9)
        p1 = np.array(p0) + d * length
        p1[0] += sway * (3 - depth)
        i1 = add(p1, trunk_c)
        i0 = _index_of(p0)
        edges.append((i0, i1))
        if depth <= 0:
            # leaf cluster
            for k in range(4):
                off = np.random.RandomState(depth * 97 + int(phase * 10) + k * 13).randn(3) * 0.09
                leaf = add(p1 + off + np.array([0, 0.05, 0]), leaf_c if k % 2 == 0 else leaf_c2)
                edges.append((i1, leaf))
            tip = add(p1 + np.array([0, 0.10, 0]), glow_c)
            edges.append((i1, tip))
            return
        n = 3 if depth >= 2 else 2
        for k in range(n):
            ang = phase + k * (2 * math.pi / n) + depth
            nd = [d[0] * 0.6 + 0.5 * math.cos(ang),
                  d[1] * 0.7 + 0.55,
                  d[2] * 0.6 + 0.5 * math.sin(ang)]
            branch(p1, nd, length * 0.62, depth - 1, phase + k * 0.7)

    def _index_of(p):
        # p0 is always the last trunk node added before recursion; find by proximity.
        for i in range(len(verts) - 1, -1, -1):
            if np.linalg.norm(np.array(verts[i]) - np.array(p)) < 1e-6:
                return i
        return 0

    branch(np.array([0.0, -0.7, 0.0]), np.array([0.0, 1.0, 0.0]), 0.42, 3, 0.4)
    # ground ring
    ring = []
    for k in range(10):
        a = k * 2 * math.pi / 10
        ring.append(add((0.35 * math.cos(a), -0.7, 0.35 * math.sin(a)), trunk_c))
    for k in range(10):
        edges.append((root, ring[k]))
        edges.append((ring[k], ring[(k + 1) % 10]))
    # floating spores
    for k in range(8):
        a = t * 0.5 + k * 0.785
        sp = add((0.5 * math.cos(a), -0.2 + 0.5 * abs(math.sin(t + k)), 0.5 * math.sin(a)), glow_c)
        edges.append((ring[k % 10], sp))
    return (np.array(verts, dtype=np.float64),
            np.array(edges, dtype=np.int64),
            np.array(colors, dtype=np.uint8))


BUILDERS = {
    "flowers": build_flowers,
    "dragon": build_dragon,
    "butterfly": build_butterfly,
    "tree": build_tree,
}

MODEL_NAMES = {
    "flowers": "Blooming Lilies",
    "dragon": "Red Dragon",
    "butterfly": "Blue Morpho",
    "tree": "Cosmic Bonsai",
}
