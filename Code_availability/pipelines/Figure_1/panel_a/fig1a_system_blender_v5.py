"""
fig1a_system_blender_v5.py
==========================
Figure 1(a) "System" panel for the MIWEN manuscript — clean redraw of the
hand-arranged fig1a_system_v2.pdf, in ONE Blender GPU pass (3-D scene +
camera-locked 2-D schematic HUD).

Changes vs. v4
--------------
* TRUE screen coordinates: the HUD helpers compensate for the camera
  sensor shift, so every normalised coordinate below is the actual
  position in the rendered frame (-1..1, y up).
* Print-matched typography: font sizes are given in POINTS and converted
  for the final 62 mm x 66 mm panel slot, so the text matches panels
  (b)-(e) (Arial, 8 pt panel letters / 6.5 pt captions / 5-6 pt labels).
* Panel aspect is exactly 62:66 (the (a) slot of the 180 x 66 mm figure).
* Captions are placed automatically UNDER the projected 3-D objects
  (base station cabinet / drone) on aligned rows — no drifting labels.
* Unified palette (same arrangement as fig1a_system_raw.png):
      blue   #4a78b0  broadcast weights w  (stems, arcs, big arrow)
      violet #7d70ad  local data x         (OFDM stems, pipeline arrows)
      teal   #46907f  layer output y = Wx  (output stems, out arrow)
      gray            hardware (mast, drone), card, mixer glyph
  No green card, no green diagram.
* WiFi arcs are anchored at the antenna tip and aimed at the Weight plot.

Usage:
  "F:/Simulation Software/Blender/blender.exe" -b -P fig1a_system_blender_v5.py -- preview
  "F:/Simulation Software/Blender/blender.exe" -b -P fig1a_system_blender_v5.py -- final

Output (new files only, never overwrites v1-v4):
  fig1/fig1a_system_v5.png          3006 x 3200 px  (= 62 x 66 mm @ ~1232 dpi)
  fig1/fig1a_system_v5_preview.png  fast preview
  fig1/fig1a_system_v5.blend
"""

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------


import bpy
import math
import sys
import os
import struct
import zlib
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

# ----------------------------------------------------------------------------
# CLI / quality switch
# ----------------------------------------------------------------------------
argv = sys.argv
mode = "preview"
if "--" in argv:
    extra = argv[argv.index("--") + 1:]
    if extra:
        mode = extra[0]

FINAL = (mode == "final")
# panel slot is 62 mm x 66 mm  ->  aspect 0.93939
RES_X, RES_Y = (3006, 3200) if FINAL else (902, 960)
SAMPLES = 320 if FINAL else 32
PANEL_W_MM, PANEL_H_MM = 62.0, 66.0

OUTDIR = str(_data_dir(__file__))
FONT_REG = r"C:\Windows\Fonts\arial.ttf"
FONT_BOLD = r"C:\Windows\Fonts\arialbd.ttf"

# ----------------------------------------------------------------------------
# Colour helpers  (sRGB hex -> linear; view transform is Standard)
# ----------------------------------------------------------------------------

def srgb2lin(hexstr):
    h = hexstr.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    def f(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return (f(r), f(g), f(b))


HEX = {
    "blue":   "#4a78b0",   # weights w        (soft steel blue)
    "violet": "#7d70ad",   # data x           (soft violet)
    "teal":   "#46907f",   # output y = Wx    (soft teal)
    "ink":    "#1f2937",   # captions (same ink as panels b-e)
    "slate":  "#3f4854",   # in-card headings (neutral)
    "card":   "#eef0f4",   # frosted light-gray card
    "cardbd": "#d9dce3",   # subtle card edge
    "mixer":  "#545d69",   # neutral mixer glyph
    "sun":    "#e0a24a",
    "gray":   "#9098a2",
    "white":  "#ffffff",
}

# ----------------------------------------------------------------------------
# Materials
# ----------------------------------------------------------------------------

def flat_mat(name, hexstr, alpha=1.0):
    """Unlit flat colour — for the 2-D overlay."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emi = nt.nodes.new("ShaderNodeEmission")
    r, g, b = srgb2lin(hexstr)
    emi.inputs["Color"].default_value = (r, g, b, 1.0)
    emi.inputs["Strength"].default_value = 1.0
    if alpha < 1.0:
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix = nt.nodes.new("ShaderNodeMixShader")
        mix.inputs["Fac"].default_value = alpha
        nt.links.new(tr.outputs[0], mix.inputs[1])
        nt.links.new(emi.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
    else:
        nt.links.new(emi.outputs[0], out.inputs["Surface"])
    return m


def pbr_mat(name, base=(0.8, 0.8, 0.8, 1.0), metallic=0.0, rough=0.5,
            emission=None, emission_strength=0.0, alpha=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")

    def set_in(key, val):
        s = bsdf.inputs.get(key)
        if s is not None:
            s.default_value = val

    set_in("Base Color", base)
    set_in("Metallic", metallic)
    set_in("Roughness", rough)
    set_in("Alpha", alpha)
    if emission is not None:
        set_in("Emission Color", emission)
        set_in("Emission Strength", emission_strength)
    return m


def assign(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)

# ----------------------------------------------------------------------------
# Primitive 3-D helpers
# ----------------------------------------------------------------------------

def add_box(name, size, loc, mat, rot=(0, 0, 0), bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc, rotation=rot)
    ob = bpy.context.active_object
    ob.name = name
    ob.scale = (size[0], size[1], size[2])
    bpy.ops.object.transform_apply(scale=True)
    if bevel > 0:
        mod = ob.modifiers.new("bev", 'BEVEL')
        mod.width = bevel
        mod.segments = 4
        mod.limit_method = 'ANGLE'
    assign(ob, mat)
    return ob


def add_cyl(name, r, depth, loc, mat, rot=(0, 0, 0), verts=48):
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=depth, location=loc,
                                        rotation=rot, vertices=verts)
    ob = bpy.context.active_object
    ob.name = name
    bpy.ops.object.shade_smooth()
    assign(ob, mat)
    return ob


def add_sphere(name, r, loc, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=loc,
                                         segments=32, ring_count=16)
    ob = bpy.context.active_object
    ob.name = name
    bpy.ops.object.shade_smooth()
    assign(ob, mat)
    return ob


def add_poly_curve(name, points, mat, bevel=0.012):
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = bevel
    cu.bevel_resolution = 6
    cu.use_fill_caps = True
    sp = cu.splines.new('NURBS')
    sp.points.add(len(points) - 1)
    for i, p in enumerate(points):
        sp.points[i].co = (p[0], p[1], p[2], 1.0)
    sp.use_endpoint_u = True
    sp.order_u = 3
    ob = bpy.data.objects.new(name, cu)
    bpy.context.collection.objects.link(ob)
    assign(ob, mat)
    return ob

# ============================================================================
# BUILD 3-D SCENE
# ============================================================================
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

CBL = srgb2lin(HEX["blue"])    # broadcast weights -> soft blue
CTL = srgb2lin(HEX["teal"])    # inference/output  -> soft teal
M_metal = pbr_mat("metal", (0.70, 0.72, 0.75, 1), metallic=0.8, rough=0.38)
M_antenna = pbr_mat("antenna", (0.83, 0.85, 0.87, 1), rough=0.5)
M_light = pbr_mat("light", (0.89, 0.90, 0.92, 1), rough=0.55)
M_dark = pbr_mat("dark", (*srgb2lin("#3b4450"), 1), rough=0.44)
M_gray = pbr_mat("gray", (0.62, 0.64, 0.67, 1), rough=0.5)
M_topaccent = pbr_mat("topacc", (0.80, 0.83, 0.87, 1), rough=0.5)
M_blue_e = pbr_mat("blue_e", (*CBL, 1), emission=(*CBL, 1), emission_strength=1.5)
M_teal_e = pbr_mat("teal_e", (*CTL, 1), emission=(*CTL, 1), emission_strength=1.5)
M_glass = pbr_mat("glass", (0.7, 0.8, 0.95, 1), rough=0.15, alpha=0.14)
M_rotor = pbr_mat("rotor", (0.75, 0.77, 0.80, 1), rough=0.35, alpha=0.5)

# Ground: shadow catcher over a white world -> white bg with contact shadows
bpy.ops.mesh.primitive_plane_add(size=80, location=(0, 0, 0))
ground = bpy.context.active_object
ground.name = "ground"
ground.is_shadow_catcher = True
assign(ground, M_light)

# ------------------------------------------------------------- base station --
BS = Vector((-5.05, 1.85, 0))
add_cyl("bs_pad", 0.62, 0.07, BS + Vector((0, 0, 0.035)), M_light)
add_cyl("bs_mast", 0.062, 2.55, BS + Vector((0, 0, 1.35)), M_metal)
add_cyl("bs_collar", 0.14, 0.06, BS + Vector((0, 0, 2.40)), M_metal)
for k in range(3):
    a = math.radians(120 * k + 30)
    r = 0.23
    loc = BS + Vector((r * math.cos(a), r * math.sin(a), 2.40))
    add_box(f"bs_panel{k}", (0.15, 0.05, 0.60), loc, M_antenna,
            rot=(0, 0, a + math.pi / 2), bevel=0.015)
add_sphere("bs_tip", 0.043, BS + Vector((0, 0, 2.67)), M_gray)

# weight-server cabinet + faint blue frequency-comb hologram
SV = Vector((-4.55, 1.00, 0))
add_box("server", (0.52, 0.44, 0.66), SV + Vector((0, 0, 0.33)), M_dark, bevel=0.02)
add_box("server_slit1", (0.44, 0.02, 0.04), SV + Vector((0, -0.225, 0.50)), M_blue_e)
add_box("server_slit2", (0.44, 0.02, 0.04), SV + Vector((0, -0.225, 0.40)), M_teal_e)
comb_heights = [0.16, 0.30, 0.22, 0.40, 0.26, 0.35, 0.20]
add_box("comb_plate", (0.66, 0.30, 0.02), SV + Vector((0, 0, 0.72)), M_glass)
for i, h in enumerate(comb_heights):
    x = -0.24 + i * 0.08
    add_cyl(f"comb_w{i}", 0.020, h, SV + Vector((x, 0, 0.74 + h / 2)),
            M_blue_e, verts=14)

# ------------------------------------------------------------- edge drone ---
D = Vector((2.95, -1.50, 1.28))
d_yaw = math.radians(22)
add_box("drone_body", (0.52, 0.52, 0.16), D, M_dark, rot=(0, 0, d_yaw), bevel=0.05)
add_box("drone_top", (0.30, 0.30, 0.08), D + Vector((0, 0, 0.11)), M_topaccent,
        rot=(0, 0, d_yaw), bevel=0.03)
for k in range(4):
    a = d_yaw + math.radians(45 + 90 * k)
    dir2 = Vector((math.cos(a), math.sin(a), 0))
    arm_c = D + dir2 * 0.38
    add_cyl(f"arm{k}", 0.028, 0.55, arm_c, M_gray, rot=(0, math.pi / 2, a), verts=24)
    hub = D + dir2 * 0.62
    add_cyl(f"motor{k}", 0.055, 0.07, hub + Vector((0, 0, 0.03)), M_dark)
    add_cyl(f"rotor{k}", 0.24, 0.012, hub + Vector((0, 0, 0.075)), M_rotor)
    add_cyl(f"rotorcap{k}", 0.03, 0.03, hub + Vector((0, 0, 0.09)), M_dark)
# camera gimbal + receive antenna
gim = D + Vector((0.10, -0.14, -0.14))
add_sphere("gimbal", 0.075, gim, M_gray)
lens_dir = Vector((0.55, -0.75, -0.55)).normalized()
lens = add_cyl("gimbal_lens", 0.035, 0.09, gim + lens_dir * 0.09, M_dark)
lens.rotation_mode = 'QUATERNION'
lens.rotation_quaternion = lens_dir.to_track_quat('Z', 'Y')
add_cyl("drone_ant", 0.014, 0.30, D + Vector((-0.12, 0.10, 0.24)), M_metal)
add_sphere("drone_ant_tip", 0.033, D + Vector((-0.12, 0.10, 0.42)), M_blue_e)

# ============================================================================
# CAMERA  (explicit rotation so the HUD basis is exact)
# ============================================================================
CAM_LOC = Vector((7.4, -6.6, 3.7))
CAM_TGT = Vector((0.05, -0.05, 1.15))
SHIFT_Y = 0.10                      # slides the 3-D scene DOWN in frame
bpy.ops.object.camera_add(location=CAM_LOC)
cam = bpy.context.active_object
cam.data.lens = 62
cam.data.sensor_fit = 'HORIZONTAL'
cam.data.sensor_width = 36.0
cam.data.shift_y = SHIFT_Y
q = (CAM_TGT - CAM_LOC).to_track_quat('-Z', 'Y')
cam.rotation_euler = q.to_euler()
scene.camera = cam
scene.render.resolution_x = RES_X       # set BEFORE projecting
scene.render.resolution_y = RES_Y
bpy.context.view_layer.update()

# ============================================================================
# 2-D HUD OVERLAY  — flat, camera-facing schematic in TRUE screen coords
# ============================================================================
mw = cam.matrix_world
R = mw.col[0].xyz.normalized()          # screen +x (right)
U = mw.col[1].xyz.normalized()          # screen +y (up)
F = (-mw.col[2].xyz).normalized()       # view direction (into scene)
CAM = mw.translation

D_HUD = 3.0                              # HUD plane distance (in front of 3-D)
fovx = 2.0 * math.atan(cam.data.sensor_width / (2.0 * cam.data.lens))
HALF_W = D_HUD * math.tan(fovx / 2.0)
HALF_H = HALF_W * RES_Y / RES_X
CAM_ROT = cam.rotation_euler.copy()

# sensor shift moves the frame: HUD content authored at optical-axis ny
# appears at ny - SHIFT_NY.  Compensate so coords below are TRUE screen.
SHIFT_NY = 2.0 * SHIFT_Y * RES_X / RES_Y

# font pt -> world size (panel prints at PANEL_H_MM tall)
PANEL_H_IN = PANEL_H_MM / 25.4
PT2NY = 2.0 / (72.0 * PANEL_H_IN)        # em height as fraction of half-height


# Blender FONT curves draw glyphs ~28% smaller than a matplotlib pt of the
# same nominal size (measured on the assembled composite); calibrate so a
# "pt" here prints at the same physical size as in panels (b)-(e).
FONT_CAL = 1.386


def EM(pt):
    return pt * PT2NY * HALF_H * FONT_CAL


def AX(nx):                              # absolute screen x -> world length
    return nx * HALF_W


def AY(ny):                              # absolute screen y -> world length
    return (ny + SHIFT_NY) * HALF_H


def NLX(f):                              # length along x
    return f * HALF_W


def NLY(f):                              # length along y
    return f * HALF_H


NY_PER_NX = RES_X / RES_Y                # ny units of an equal world length in nx


def W(u, v, lift=0.0):
    """world-length HUD coords (u right, v up, about optical centre) -> world."""
    return CAM + F * (D_HUD - lift) + R * u + U * v


def project(pt):
    """world point -> TRUE screen coords (nx, ny in -1..1, y up)."""
    co = world_to_camera_view(scene, cam, Vector(pt))
    return (co.x * 2.0 - 1.0, co.y * 2.0 - 1.0)


hud_objs = []


def _finish(ob, mat):
    hud_objs.append(ob)
    assign(ob, mat)
    return ob


def fill_uv(name, uv_pts, mat, lift, cu=0.0, cv=0.0):
    """filled n-gon; uv_pts are (u,v) world-length offsets about (cu,cv)."""
    verts = [W(cu + u, cv + v, lift) for (u, v) in uv_pts]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], [list(range(len(verts)))])
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    return _finish(ob, mat)


def rect(name, cx, cy, w, h, mat, lift, r=0.0):
    """rounded rectangle centred at screen (cx, cy), size w x h (world lengths)."""
    cu, cv = AX(cx), AY(cy)
    hw, hh = w / 2.0, h / 2.0
    if r <= 0:
        pts = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    else:
        r = min(r, hw, hh)
        seg_n = 6
        pts = []
        corners = [(hw - r, hh - r, 0), (-hw + r, hh - r, math.pi / 2),
                   (-hw + r, -hh + r, math.pi), (hw - r, -hh + r, 1.5 * math.pi)]
        for ccx, ccy, a0 in corners:
            for s in range(seg_n + 1):
                a = a0 + (math.pi / 2) * s / seg_n
                pts.append((ccx + r * math.cos(a), ccy + r * math.sin(a)))
    return fill_uv(name, pts, mat, lift, cu, cv)


def disk(name, cx, cy, rad, mat, lift, n=48):
    pts = [(rad * math.cos(2 * math.pi * s / n),
            rad * math.sin(2 * math.pi * s / n)) for s in range(n)]
    return fill_uv(name, pts, mat, lift, AX(cx), AY(cy))


def seg(name, x0, y0, x1, y1, wid, mat, lift):
    """thick segment between screen points, width wid (world length)."""
    u0, v0, u1, v1 = AX(x0), AY(y0), AX(x1), AY(y1)
    dx, dy = u1 - u0, v1 - v0
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return None
    nx_, ny_ = -dy / L, dx / L
    hw = wid / 2.0
    pts = [(u0 + nx_ * hw, v0 + ny_ * hw), (u1 + nx_ * hw, v1 + ny_ * hw),
           (u1 - nx_ * hw, v1 - ny_ * hw), (u0 - nx_ * hw, v0 - ny_ * hw)]
    return fill_uv(name, pts, mat, lift, 0.0, 0.0)


def triangle(name, p0, p1, p2, mat, lift):
    """screen-coord triangle."""
    pts = [(AX(p[0]), AY(p[1])) for p in (p0, p1, p2)]
    return fill_uv(name, pts, mat, lift, 0.0, 0.0)


def text(name, body, cx, cy, size, mat, lift, bold=False,
         align_x='CENTER', align_y='CENTER', line_sp=0.85):
    cu2 = bpy.data.curves.new(name, 'FONT')
    cu2.body = body
    cu2.size = size
    cu2.align_x = align_x
    cu2.align_y = align_y
    cu2.space_line = line_sp
    path = FONT_BOLD if bold else FONT_REG
    if os.path.exists(path):
        cu2.font = bpy.data.fonts.load(path)
    ob = bpy.data.objects.new(name, cu2)
    bpy.context.collection.objects.link(ob)
    ob.location = W(AX(cx), AY(cy), lift)
    ob.rotation_euler = CAM_ROT
    return _finish(ob, mat)


# depth layers (world-length lift toward camera; larger = on top)
Z_FILL = 0.000
Z_BORDER = 0.004
Z_ITEM = 0.010
Z_ARROW = 0.016
Z_TXT = 0.022

mblue = flat_mat("m_blue", HEX["blue"])
mviolet = flat_mat("m_violet", HEX["violet"])
mteal = flat_mat("m_teal", HEX["teal"])
mink = flat_mat("m_ink", HEX["ink"])
mslate = flat_mat("m_slate", HEX["slate"])
mcard = flat_mat("m_card", HEX["card"])
mcardbd = flat_mat("m_cardbd", HEX["cardbd"])
mmix = flat_mat("m_mix", HEX["mixer"])
msun = flat_mat("m_sun", HEX["sun"])
mgray = flat_mat("m_gray", HEX["gray"])
mwhite = flat_mat("m_white", HEX["white"])


def stem_plot(tag, cx, base, heights, span, colmat, lift,
              hmax=0.15, baseline=True):
    """stem plot centred at screen nx=cx, baseline at screen ny=base."""
    n = len(heights)
    x0 = cx - span / 2.0
    step = span / (n - 1) if n > 1 else 0.0
    wid = min(NLX(step * 0.16), NLX(0.0075))
    dot_r = wid * 1.75
    if baseline:
        seg(f"{tag}_base", x0 - step * 0.4, base, x0 + step * (n - 1) + step * 0.4,
            base, NLX(0.0022), mgray, lift)
    mx = max(heights)
    for i, h in enumerate(heights):
        xu = x0 + i * step
        hv = hmax * (h / mx)
        seg(f"{tag}_s{i}", xu, base, xu, base + hv, wid, colmat, lift)
        disk(f"{tag}_d{i}", xu, base + hv, dot_r, colmat, lift, n=20)


def small_arrow(tag, x0, x1, y, colmat, lift):
    """short horizontal pipeline arrow (screen coords)."""
    head = 0.020
    seg(f"{tag}_sh", x0, y, x1 - head * 0.7, y, NLX(0.0055), colmat, lift)
    head_ny = head * NY_PER_NX
    triangle(f"{tag}_hd",
             (x1, y), (x1 - head, y + head_ny * 0.62),
             (x1 - head, y - head_ny * 0.62), colmat, lift)


def big_arrow(tag, p0, c0, c1, p1, colmat, lift, width_nx=0.016):
    """thick cubic-Bezier arrow in screen coords (weight -> mixer LO port).
    Arc-length construction: the shaft ribbon runs into the arrowhead base
    (slight overlap, no seam) and the head tip lands exactly on p1."""
    def B(t):
        mt = 1 - t
        x = (mt**3) * p0[0] + 3 * mt * mt * t * c0[0] + \
            3 * mt * t * t * c1[0] + t**3 * p1[0]
        y = (mt**3) * p0[1] + 3 * mt * mt * t * c0[1] + \
            3 * mt * t * t * c1[1] + t**3 * p1[1]
        return Vector((AX(x), AY(y)))
    N = 160
    pts = [B(i / N) for i in range(N + 1)]
    cum = [0.0]
    for i in range(1, N + 1):
        cum.append(cum[-1] + (pts[i] - pts[i - 1]).length)
    total = cum[-1]
    wid = NLX(width_nx)
    head = NLX(0.034)
    s_trim = total - head * 0.82        # shaft runs slightly into the head
    top, bot = [], []
    last_i = 0
    for i in range(N + 1):
        if cum[i] > s_trim and len(top) >= 2:
            break
        last_i = i
        j = min(i + 1, N)
        d = pts[j] - pts[max(j - 1, 0)]
        if d.length < 1e-12:
            continue
        d.normalize()
        nrm = Vector((-d.y, d.x))
        f = cum[i] / total
        taper = 1.0 if f < 0.62 else \
            max(0.58, 1.0 - (f - 0.62) / 0.38 * 0.42)
        top.append(pts[i] + nrm * wid * 0.5 * taper)
        bot.append(pts[i] - nrm * wid * 0.5 * taper)
    fill_uv(f"{tag}_rib", [(q.x, q.y) for q in top + bot[::-1]], colmat, lift)
    tip = pts[-1]
    d = (tip - pts[last_i]).normalized()
    base = tip - d * head
    nrm = Vector((-d.y, d.x))
    fill_uv(f"{tag}_hd",
            [(tip.x, tip.y),
             (base.x + nrm.x * head * 0.55, base.y + nrm.y * head * 0.55),
             (base.x - nrm.x * head * 0.55, base.y - nrm.y * head * 0.55)],
            colmat, lift)


# ------------------------------------------------------- WiFi broadcast arcs
# anchored at the antenna tip, aimed (in screen space) at the Weight plot;
# kept steep/left so the arcs never cross behind the Edge-Device card
WGT_CX, WGT_BASE = -0.05, 0.62           # weight stem plot (screen coords)
ANT = BS + Vector((0.05, -0.05, 2.62))
ant_scr = project(ANT)
tgt_scr = (WGT_CX - 0.33, WGT_BASE + 0.02)
d_nx = (tgt_scr[0] - ant_scr[0])
d_ny = (tgt_scr[1] - ant_scr[1])
emit = (R * d_nx * HALF_W + U * d_ny * HALF_H).normalized()
qn = F.cross(emit).normalized()
arc_c = ANT + emit * 0.10
for i, r in enumerate((0.25, 0.42, 0.60)):
    pts = []
    for k in range(41):
        a = math.radians(-34 + 68 * k / 40)
        pts.append(arc_c + r * (math.cos(a) * emit + math.sin(a) * qn))
    m = pbr_mat(f"wave{i}", (*CBL, 1), emission=(*CBL, 1),
                emission_strength=2.0 - 0.4 * i)
    add_poly_curve(f"wavefront{i}", pts, m, bevel=0.018)

# ============================================================================
# HUD LAYOUT  (all coords TRUE screen, -1..1, y up)
# ============================================================================

# ------------------------------------------------------------- panel tag ----
# Nature style: lowercase bold, no parentheses
text("tag_a", "a", -0.94, 0.92, EM(8.0), mink, Z_TXT, bold=True, align_x='LEFT')

# ------------------------------------------------------------- Weight plot --
text("t_weight", "Weight", WGT_CX, 0.905, EM(6.2), mblue, Z_TXT, bold=True)
stem_plot("wgt", WGT_CX, WGT_BASE, [0.55, 0.85, 0.35, 0.7, 0.25, 1.0, 0.6, 0.9],
          0.42, mblue, Z_ITEM, hmax=0.20)

# ------------------------------------------------------------- Edge card ----
C_L, C_R = -0.27, 0.965
C_T, C_B = 0.47, -0.225
cx_card = (C_L + C_R) / 2
cy_card = (C_T + C_B) / 2
cw = NLX(C_R - C_L)
ch = NLY(C_T - C_B)
rect("card_bd", cx_card, cy_card, cw + NLX(0.010), ch + NLX(0.010),
     mcardbd, Z_FILL, r=NLX(0.050))
rect("card_bg", cx_card, cy_card, cw, ch, mcard, Z_BORDER, r=NLX(0.047))
text("t_edge", "Edge Device", C_L + 0.05, C_T - 0.085, EM(6.2),
     mslate, Z_TXT, bold=True, align_x='LEFT')

ROW = 0.155          # pipeline vertical centre
SUBT = -0.015        # sub-label TOP anchor row

# data icon (centred on the pipeline row axis)
YD = -0.15
iw, ih = NLX(0.095), NLY(0.16)
rect("yd_bd", YD, ROW, iw + NLX(0.008), ih + NLX(0.008), mmix, Z_ITEM)
rect("yd_bg", YD, ROW, iw, ih, mwhite, Z_ITEM + 0.002)
mtn = Z_ITEM + 0.004
ihn = 0.16          # icon height in ny units
iwn = 0.095         # icon width in nx units
icx, icy = YD, ROW
seg("yd_m1a", icx - iwn * 0.34, icy - ihn * 0.34, icx - iwn * 0.02,
    icy + ihn * 0.14, NLX(0.0045), mgray, mtn)
seg("yd_m1b", icx - iwn * 0.02, icy + ihn * 0.14, icx + iwn * 0.16,
    icy - ihn * 0.12, NLX(0.0045), mgray, mtn)
seg("yd_m2a", icx - iwn * 0.04, icy - ihn * 0.34, icx + iwn * 0.22,
    icy + ihn * 0.20, NLX(0.0045), mgray, mtn)
seg("yd_m2b", icx + iwn * 0.22, icy + ihn * 0.20, icx + iwn * 0.40,
    icy - ihn * 0.06, NLX(0.0045), mgray, mtn)
disk("yd_sun", icx + iwn * 0.20, icy + ihn * 0.27, NLX(0.013), msun,
     mtn + 0.002, n=24)
text("t_yd", "data", YD, SUBT, EM(6.2), mink, Z_TXT, align_y='TOP')

small_arrow("a1", -0.0585, 0.0215, ROW, mviolet, Z_ARROW)

# OFDM waveform stems (data x -> violet)
OF = 0.145
stem_plot("ofdm", OF, ROW - 0.085, [0.5, 0.75, 0.45, 0.9, 0.6, 1.0, 0.55, 0.8],
          0.155, mviolet, Z_ITEM, hmax=0.16)
text("t_ofdm", "OFDM\nwaveform", OF, SUBT, EM(6.2), mink, Z_TXT, align_y='TOP')

small_arrow("a2", 0.299, 0.379, ROW, mviolet, Z_ARROW)

# Passive mixer glyph (X inscribed in the ring, equal world lengths)
MX = 0.51
mr = NLX(0.060)
disk("mix_out", MX, ROW, mr, mmix, Z_ITEM, n=56)
disk("mix_in", MX, ROW, mr - NLX(0.0075), mwhite, Z_ITEM + 0.002, n=56)
kk = 0.0354
kk_ny = kk * NY_PER_NX
seg("mix_x1", MX - kk, ROW - kk_ny, MX + kk, ROW + kk_ny, NLX(0.009),
    mmix, Z_ITEM + 0.004)
seg("mix_x2", MX - kk, ROW + kk_ny, MX + kk, ROW - kk_ny, NLX(0.009),
    mmix, Z_ITEM + 0.004)
text("t_mix", "Passive\nMixer", MX, SUBT, EM(6.2), mslate, Z_TXT,
     bold=True, align_y='TOP')

small_arrow("a3", 0.617, 0.697, ROW, mteal, Z_ARROW)

# layer output stems (result y = Wx -> teal)
LO = 0.815
stem_plot("lo", LO, ROW - 0.085, [0.9, 0.55, 0.75, 0.35, 0.6], 0.13, mteal,
          Z_ITEM, hmax=0.16)
text("t_lo", "layer\noutput", LO, SUBT, EM(6.2), mink, Z_TXT, align_y='TOP')

# ------------------------------------------------------------- big arrow ----
# weight vector -> mixer LO port (tip touches the top of the mixer ring)
big_arrow("bigarr", (0.23, 0.70), (0.53, 0.78), (0.66, 0.50), (MX, 0.216),
          mblue, Z_ARROW, width_nx=0.016)

# mixer port labels, coloured by their signals:
#   LO = broadcast weight (blue), RF = local data (violet), IF = output (teal)
text("t_port_lo", "LO", 0.435, 0.285, EM(6.2), mblue, Z_TXT)
text("t_port_rf", "RF", 0.339, 0.215, EM(6.2), mviolet, Z_TXT)
text("t_port_if", "IF", 0.657, 0.215, EM(6.2), mteal, Z_TXT)

# ------------------------------------------------------------- captions -----
# aligned rows, horizontally centred on the projected 3-D objects
# base-station caption hugs the (now more distant) station: rows are
# derived from the projected ground point; drone caption sits on a lower
# fixed row so it clears the rotors/gimbal.
bs_nx, bs_ny = project(SV + Vector((0, 0, 0.33)))
bs_gnx, bs_gny = project(SV)                      # ground point under cabinet
dr_nx, dr_ny = project(D)
BS_NAME_NY = bs_gny - 0.10
BS_SUB_NY = BS_NAME_NY - 0.085
ED_NY = -0.74
print(f"[fig1a v5] proj cabinet=({bs_nx:+.3f},{bs_ny:+.3f})  ground={bs_gny:+.3f}  "
      f"drone=({dr_nx:+.3f},{dr_ny:+.3f})  ant=({ant_scr[0]:+.3f},{ant_scr[1]:+.3f})")
text("t_bs1", "Base Station", bs_nx, BS_NAME_NY, EM(6.2), mink, Z_TXT, bold=True)
text("t_bs2", "(broadcast weight)", bs_nx, BS_SUB_NY, EM(6.2), mink, Z_TXT)
text("t_ed", "Edge Device", dr_nx, ED_NY, EM(6.2), mink, Z_TXT, bold=True)

# ============================================================================
# LIGHTS + WORLD
# ============================================================================
tgt = bpy.data.objects.new("lt_target", None)
bpy.context.collection.objects.link(tgt)
tgt.location = CAM_TGT
for name, loc, energy, size in [
    ("key", (5.5, -2.5, 8.0), 1100, 4.5),
    ("fill", (-6.0, -5.0, 5.0), 380, 6.0),
    ("rim", (-1.0, 7.5, 5.5), 420, 4.0),
]:
    bpy.ops.object.light_add(type='AREA', location=loc)
    li = bpy.context.active_object
    li.name = name
    li.data.energy = energy
    li.data.size = size
    c = li.constraints.new('TRACK_TO')
    c.target = tgt
    c.track_axis = 'TRACK_NEGATIVE_Z'
    c.up_axis = 'UP_Y'

world = bpy.data.worlds.new("world")
scene.world = world
world.use_nodes = True
wt = world.node_tree
wt.nodes.clear()
wout = wt.nodes.new("ShaderNodeOutputWorld")
lp = wt.nodes.new("ShaderNodeLightPath")
b_cam = wt.nodes.new("ShaderNodeBackground")   # pure white to camera
b_cam.inputs[0].default_value = (1, 1, 1, 1)
b_cam.inputs[1].default_value = 1.0
b_lit = wt.nodes.new("ShaderNodeBackground")   # gentle ambient for shading
b_lit.inputs[0].default_value = (1, 1, 1, 1)
b_lit.inputs[1].default_value = 0.52
wmix = wt.nodes.new("ShaderNodeMixShader")
wt.links.new(lp.outputs["Is Camera Ray"], wmix.inputs[0])
wt.links.new(b_lit.outputs[0], wmix.inputs[1])
wt.links.new(b_cam.outputs[0], wmix.inputs[2])
wt.links.new(wmix.outputs[0], wout.inputs[0])

# ============================================================================
# RENDER SETTINGS — Cycles + GPU (OptiX) + Standard view transform
# ============================================================================
scene.render.engine = 'CYCLES'
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = True
scene.render.film_transparent = False
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGB'
scene.render.filter_size = 1.4

scene.view_settings.view_transform = 'Standard'
scene.view_settings.look = 'None'
scene.display_settings.display_device = 'sRGB'

prefs = bpy.context.preferences.addons.get('cycles')
if prefs is not None:
    cprefs = prefs.preferences
    for dev_type in ('OPTIX', 'CUDA'):
        try:
            cprefs.compute_device_type = dev_type
            cprefs.get_devices()
            gpus = [d for d in cprefs.devices if d.type != 'CPU']
            if gpus:
                for d in cprefs.devices:
                    d.use = (d.type != 'CPU')
                scene.cycles.device = 'GPU'
                print(f"[fig1a v5] GPU via {dev_type}: {[d.name for d in gpus]}")
                break
        except Exception as e:
            print(f"[fig1a v5] {dev_type} unavailable: {e}")
    else:
        scene.cycles.device = 'CPU'
        print("[fig1a v5] CPU render")

suffix = "" if FINAL else "_preview"
out_png = os.path.join(OUTDIR, f"fig1a_system_v5{suffix}.png")
scene.render.filepath = out_png

bpy.ops.render.render(write_still=True)
print("[fig1a v5] wrote", out_png)

if FINAL:                      # save AFTER render; retry around Dropbox locks
    import time as _time
    blend_path = os.path.join(OUTDIR, "fig1a_system_v5.blend")
    for _try in range(6):
        try:
            if os.path.exists(blend_path):
                os.remove(blend_path)
            bpy.ops.wm.save_as_mainfile(filepath=blend_path, check_existing=False)
            print("[fig1a v5] saved", blend_path)
            break
        except Exception as e:
            if _try == 5:
                print("[fig1a v5] blend save skipped:", e)
            else:
                _time.sleep(2.0)


# ----------------------------------------------------------------------------
# fade the bottom edge to white so the soft ground shadow is not clipped
# hard at the panel trim (smoothstep over the lowest FADE_FRAC of the frame)
# ----------------------------------------------------------------------------
def fade_bottom(path, frac=0.08):
    import numpy as np
    img = bpy.data.images.load(path)
    try:
        w, h = img.size
        buf = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, -1)
        n = max(2, int(h * frac))           # rows 0..n-1 are the BOTTOM rows
        for i in range(n):
            s = (n - i) / n                 # 1 at bottom row -> 0 at band top
            a = s * s * (3.0 - 2.0 * s)     # smoothstep toward white
            buf[i, :, :3] = buf[i, :, :3] * (1.0 - a) + a
        img.pixels[:] = buf.reshape(-1)
        img.filepath_raw = path
        img.file_format = 'PNG'
        img.save()
        print(f"[fig1a v5] faded bottom {n} rows to white")
    finally:
        bpy.data.images.remove(img)


try:
    fade_bottom(out_png, 0.08)
except Exception as e:
    print("[fig1a v5] bottom fade skipped:", e)


# ----------------------------------------------------------------------------
# stamp the true physical size into the PNG (pHYs chunk) — pure python
# ----------------------------------------------------------------------------
def stamp_dpi(path, ppm):
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return
    pos = 8
    ihdr_end = None
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype = data[pos + 4:pos + 8]
        nextpos = pos + 12 + length
        if ctype == b"IHDR":
            ihdr_end = nextpos
        if ctype == b"pHYs":
            data = data[:pos] + data[nextpos:]
            continue
        pos = nextpos
    if ihdr_end is None:
        return
    body = struct.pack(">IIB", ppm, ppm, 1)
    chunk = struct.pack(">I", len(body)) + b"pHYs" + body
    chunk += struct.pack(">I", zlib.crc32(b"pHYs" + body) & 0xffffffff)
    data = data[:ihdr_end] + chunk + data[ihdr_end:]
    with open(path, "wb") as f:
        f.write(data)
    dpi = ppm * 0.0254
    print(f"[fig1a v5] stamped {dpi:.0f} dpi ({ppm} ppm)")


import time
PPM = int(round(RES_X / (PANEL_W_MM / 1000.0)))    # true pixels-per-metre
for _attempt in range(8):                # retry: Dropbox may briefly lock it
    try:
        stamp_dpi(out_png, PPM)
        break
    except Exception as e:
        if _attempt == 7:
            print("[fig1a v5] dpi stamp skipped:", e)
        else:
            time.sleep(1.0)
