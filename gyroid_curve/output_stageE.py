"""Stage E -- output & render.

Exports the single curve as .obj polyline, .ply and .csv, exports the Stage-A
block mesh for context, and renders a matplotlib turntable (semi-transparent
mesh + curve coloured by global arclength = the Hilbert sweep).
"""

from __future__ import annotations

import os
import numpy as np

from .mesh_stageA import BlockMesh


# --------------------------------------------------------------------------
# Curve exporters
# --------------------------------------------------------------------------
def write_obj_polyline(path: str, P: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write("# gyroid surface-filling curve (single polyline)\n")
        for p in P:
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        # one polyline element
        f.write("l " + " ".join(str(i + 1) for i in range(len(P))) + "\n")


def write_ply_polyline(path: str, P: np.ndarray) -> None:
    n = len(P)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element edge {n - 1}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("end_header\n")
        for p in P:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for i in range(n - 1):
            f.write(f"{i} {i + 1}\n")


def write_csv(path: str, P: np.ndarray) -> None:
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0),
                                                        axis=1))])
    with open(path, "w") as f:
        f.write("x,y,z,arclength\n")
        for p, a in zip(P, s):
            f.write(f"{p[0]:.6f},{p[1]:.6f},{p[2]:.6f},{a:.6f}\n")


def write_mesh_obj(path: str, mesh: BlockMesh) -> None:
    with open(path, "w") as f:
        f.write("# gyroid block mesh (Stage A)\n")
        for v in mesh.V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in mesh.F:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------
def render_png(path: str, P: np.ndarray, mesh: BlockMesh | None = None,
               n_frames: int = 1, title: str = "",
               show_mesh: bool = True, max_seg_draw: float = 1.5) -> None:
    """Turntable-style still: translucent block surface + curve coloured by
    global arclength (the Hilbert sweep).  Bridge chords longer than
    ``max_seg_draw`` are not drawn so the stripe texture reads cleanly."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    if mesh is not None and show_mesh:
        F = mesh.F
        step = max(1, len(F) // 12000)        # cap drawn triangles
        tris = mesh.V[F[::step]]
        pc = Poly3DCollection(tris, alpha=0.06, facecolor="#5577aa",
                              edgecolor="none")
        ax.add_collection3d(pc)

    # colour by global arclength; hide long bridge chords
    seg = np.stack([P[:-1], P[1:]], axis=1)
    seglen = np.linalg.norm(P[1:] - P[:-1], axis=1)
    keep = seglen <= max_seg_draw
    s = np.arange(len(P) - 1) / max(1, len(P) - 1)
    lc = Line3DCollection(seg[keep], cmap="turbo", linewidths=1.1)
    lc.set_array(s[keep])
    ax.add_collection3d(lc)

    lo = P.min(0)
    hi = P.max(0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    ax.view_init(elev=22, azim=35)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def write_html_viewer(path: str, P: np.ndarray, mesh: BlockMesh | None = None,
                      mesh_subsample: int = 1) -> None:
    """Self-contained three.js viewer with interactive orbit.

    Curve coloured by global arclength; optional translucent block mesh.
    Geometry is embedded as JSON.
    """
    import json

    s = np.linspace(0, 1, len(P))
    curve = {"pts": P.astype(np.float32).ravel().tolist(),
             "s": s.astype(np.float32).tolist()}
    mesh_json = None
    if mesh is not None:
        F = mesh.F[::mesh_subsample]
        mesh_json = {"v": mesh.V.astype(np.float32).ravel().tolist(),
                     "f": F.astype(np.int32).ravel().tolist()}
    center = P.mean(0).tolist()
    scale = float(np.linalg.norm(P.max(0) - P.min(0)))

    html = _HTML_TEMPLATE.replace("__CURVE__", json.dumps(curve)) \
        .replace("__MESH__", json.dumps(mesh_json)) \
        .replace("__CENTER__", json.dumps(center)) \
        .replace("__SCALE__", json.dumps(scale))
    with open(path, "w") as f:
        f.write(html)


_HTML_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<title>Gyroid surface-filling curve</title>
<style>body{margin:0;overflow:hidden;background:#0a0a12;color:#ccc;
font-family:monospace}#i{position:absolute;top:8px;left:8px;font-size:12px}</style>
</head><body><div id="i">gyroid surface-filling curve &mdash; drag to orbit, scroll to zoom</div>
<script type="importmap">{"imports":{
"three":"https://unpkg.com/three@0.160.0/build/three.module.js",
"three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const CURVE=__CURVE__, MESH=__MESH__, CENTER=__CENTER__, SCALE=__SCALE__;
const scene=new THREE.Scene();
const cam=new THREE.PerspectiveCamera(55,innerWidth/innerHeight,0.01,1e5);
cam.position.set(CENTER[0]+SCALE,CENTER[1]+SCALE,CENTER[2]+SCALE);
const rnd=new THREE.WebGLRenderer({antialias:true});
rnd.setSize(innerWidth,innerHeight);document.body.appendChild(rnd.domElement);
const ctl=new OrbitControls(cam,rnd.domElement);ctl.target.set(...CENTER);
scene.add(new THREE.AmbientLight(0xffffff,0.7));
const dl=new THREE.DirectionalLight(0xffffff,0.6);dl.position.set(1,1,1);scene.add(dl);
// curve coloured by global arclength (the Hilbert sweep)
const pts=CURVE.pts, sarr=CURVE.s;
const g=new THREE.BufferGeometry();
const pos=new Float32Array(pts); g.setAttribute('position',new THREE.BufferAttribute(pos,3));
const col=new Float32Array(pts.length);
for(let i=0;i<sarr.length;i++){const c=new THREE.Color().setHSL(0.7*(1-sarr[i]),0.9,0.55);
 col[3*i]=c.r;col[3*i+1]=c.g;col[3*i+2]=c.b;}
g.setAttribute('color',new THREE.BufferAttribute(col,3));
const line=new THREE.Line(g,new THREE.LineBasicMaterial({vertexColors:true}));
scene.add(line);
if(MESH){const mg=new THREE.BufferGeometry();
 mg.setAttribute('position',new THREE.BufferAttribute(new Float32Array(MESH.v),3));
 mg.setIndex(MESH.f); mg.computeVertexNormals();
 const mm=new THREE.MeshStandardMaterial({color:0x4466aa,transparent:true,
  opacity:0.12,side:THREE.DoubleSide,depthWrite:false});
 scene.add(new THREE.Mesh(mg,mm));}
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;
 cam.updateProjectionMatrix();rnd.setSize(innerWidth,innerHeight);});
(function loop(){requestAnimationFrame(loop);ctl.update();rnd.render(scene,cam);})();
</script></body></html>"""


def export_all(out_dir: str, name: str, P: np.ndarray,
               mesh: BlockMesh | None = None, render: bool = True,
               verbose: bool = True) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    base = os.path.join(out_dir, name)
    write_obj_polyline(base + ".obj", P);      paths["obj"] = base + ".obj"
    write_ply_polyline(base + ".ply", P);      paths["ply"] = base + ".ply"
    write_csv(base + ".csv", P);               paths["csv"] = base + ".csv"
    if mesh is not None:
        write_mesh_obj(base + "_mesh.obj", mesh)
        paths["mesh_obj"] = base + "_mesh.obj"
    write_html_viewer(base + ".html", P, mesh)
    paths["html"] = base + ".html"
    if render:
        render_png(base + ".png", P, mesh, title=name)
        paths["png"] = base + ".png"
    if verbose:
        print("[E] wrote: " + ", ".join(os.path.basename(p)
                                        for p in paths.values()))
    return paths
