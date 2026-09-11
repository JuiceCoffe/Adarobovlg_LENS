import os
from typing import Dict
import trimesh
import numpy as np
import open3d as o3d
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from scipy.spatial.transform import Rotation as R


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()
                )
            )
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh


def extract_colors_from_urdf(urdf_path):
    """Read URDF material colors for mesh conversion."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    global_materials = {}

    for material in root.findall("material"):
        name = material.attrib["name"]
        color_elem = material.find("color")
        if color_elem is not None and "rgba" in color_elem.attrib:
            rgba = [float(c) for c in color_elem.attrib["rgba"].split()]
            global_materials[name] = rgba

    link_colors = {}

    for link in root.iter("link"):
        link_name = link.attrib["name"]
        visual = link.find("./visual")
        if visual is not None:
            material = visual.find("./material")
            if material is not None:
                color = material.find("color")
                if color is not None and "rgba" in color.attrib:
                    rgba = [float(c) for c in color.attrib["rgba"].split()]
                    link_colors[link_name] = rgba
                elif "name" in material.attrib:
                    material_name = material.attrib["name"]
                    if material_name in global_materials:
                        link_colors[link_name] = global_materials[material_name]

    return link_colors


def parse_origin(element):
    """
    Parse the origin element for translation and rotation.
    """
    origin = element.find("origin")
    xyz = np.zeros(3)
    rotation = np.eye(3)
    if origin is not None:
        xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
        rotation = R.from_euler("xyz", rpy).as_matrix()
    return xyz, rotation


def apply_transform(mesh, translation, rotation):
    """
    Apply translation and rotation to a mesh.
    """
    # mesh.apply_translation(-mesh.centroid)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    mesh.apply_transform(transform)
    return mesh


def create_primitive_mesh(geometry, translation, rotation):
    """
    Create a trimesh object from primitive geometry definitions with transformations.
    """
    if geometry.tag.endswith("box"):
        size = np.fromstring(geometry.attrib["size"], sep=" ")
        mesh = trimesh.creation.box(extents=size)
    elif geometry.tag.endswith("sphere"):
        radius = float(geometry.attrib["radius"])
        mesh = trimesh.creation.icosphere(radius=radius)
    elif geometry.tag.endswith("cylinder"):
        radius = float(geometry.attrib["radius"])
        length = float(geometry.attrib["length"])
        mesh = trimesh.creation.cylinder(radius=radius, height=length)
    else:
        raise ValueError(f"Unsupported geometry type: {geometry.tag}")
    return apply_transform(mesh, translation, rotation)


def load_link_geometries(urdf_path, link_names, collision=False) -> Dict[str, trimesh.Trimesh]:
    """
    Load geometries (trimesh objects) for specified links from a URDF file, considering origins.
    """
    urdf_dir = os.path.dirname(urdf_path)
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    link_geometries = {}
    link_colors_from_urdf = extract_colors_from_urdf(urdf_path)

    for link in root.findall("link"):
        link_name = link.attrib["name"]
        link_color = link_colors_from_urdf.get(link_name, None)
        if link_name in link_names:
            geom_index = "collision" if collision else "visual"
            link_mesh = []
            for visual in link.findall(".//" + geom_index):
                geometry = visual.find("geometry")
                xyz, rotation = parse_origin(visual)
                try:
                    if geometry[0].tag.endswith("mesh"):
                        mesh_filename = geometry[0].attrib["filename"]
                        full_mesh_path = os.path.join(urdf_dir, mesh_filename)
                        mesh = trimesh.load(full_mesh_path, force="mesh")
                        scale = np.fromstring(geometry[0].attrib.get("scale", "1 1 1"), sep=" ")
                        mesh.apply_scale(scale)
                        mesh = apply_transform(mesh, xyz, rotation)
                        link_mesh.append(mesh)
                    else:  # Handle primitive shapes
                        mesh = create_primitive_mesh(geometry[0], xyz, rotation)
                        scale = np.fromstring(geometry[0].attrib.get("scale", "1 1 1"), sep=" ")
                        mesh.apply_scale(scale)
                        link_mesh.append(mesh)
                except Exception as e:
                    print(f"Failed to load geometry for {link_name}: {e}")
            if len(link_mesh) == 0:
                continue
            elif len(link_mesh) > 1:
                link_trimesh = as_mesh(trimesh.Scene(link_mesh))
            elif len(link_mesh) == 1:
                link_trimesh = link_mesh[0]

            if link_color is not None:
                link_trimesh.visual.face_colors = np.array(link_color)
            link_geometries[link_name] = link_trimesh

    return link_geometries


def trimesh_to_open3d(mesh_tm: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    # 1. Extract vertices and faces from Trimesh object
    """Convert a trimesh mesh and its available colors to an Open3D triangle mesh."""
    vertices = np.asarray(mesh_tm.vertices)
    faces = np.asarray(mesh_tm.faces)

    # Create an Open3D TriangleMesh object
    mesh_o3d = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces)
    )

    # 2. Handle vertex colors (if available)
    try:
        # Attempt to extract vertex colors
        if isinstance(mesh_tm.visual, trimesh.visual.texture.TextureVisuals):
            vc = mesh_tm.visual.to_color().vertex_colors
        elif isinstance(mesh_tm.visual, trimesh.visual.color.ColorVisuals):
            vc = mesh_tm.visual.vertex_colors
        else:
            vc = np.ones((len(mesh_tm.vertices), 3)) * 0.8  # default gray color
        vc = np.asarray(vc)

        # Validate the shape of the vertex colors
        if vc.ndim == 2 and vc.shape[0] == vertices.shape[0]:
            # If vertex colors have 4 channels, discard the alpha channel
            if vc.shape[1] == 4:
                vc_rgb = vc[:, :3]
            elif vc.shape[1] == 3:
                vc_rgb = vc
            else:
                print(f"[Warning] Unexpected vertex_colors shape: {vc.shape}, dropping colors.")
                vc_rgb = None
        else:
            print(
                f"[Warning] vertex_colors array shape mismatch: {vc.shape}, expected ({vertices.shape[0]}, 3 or 4)."
            )
            vc_rgb = None
    except Exception as e:
        # Handle case where vertex colors are not found or error occurs during extraction
        print(f"[Info] No vertex_colors found or error occurred: {e}")
        vc_rgb = None

    # 3. Normalize and assign colors if valid
    if vc_rgb is not None:
        # If the vertex colors are in uint8 format (0-255), convert them to float (0-1)
        if np.issubdtype(vc_rgb.dtype, np.integer):
            vc_norm = vc_rgb.astype(np.float64) / 255.0
        else:
            vc_norm = vc_rgb.astype(np.float64)
            # Optionally clamp colors to the [0, 1] range
            vc_norm = np.clip(vc_norm, 0.0, 1.0)

        # Validate the resulting color array shape
        if vc_norm.shape == (vertices.shape[0], 3):
            # Assign the normalized vertex colors to the mesh
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vc_norm)
        else:
            print(
                f"[Warning] Normalized colors shape {vc_norm.shape} mismatches vertices {vertices.shape[0]}. Dropping colors."
            )
    else:
        # If no valid vertex colors are found, use a fallback default color
        print("[Info] Using default color for mesh since no valid vertex_colors found.")
        mesh_o3d.paint_uniform_color(
            [0.7, 0.7, 0.7]
        )  # Optional: paint the mesh with a uniform color

    # 4. Compute normals for the mesh (recommended for proper visualization)
    mesh_o3d.compute_vertex_normals()

    return mesh_o3d


def extract_collision_spheres_from_urdf(urdf_path: str):
    """
    Extract all <sphere> collision geometries in a URDF.

    Returns a list of tuples:
    [
        (
            radius,
            {
                "link_name": [[x, y, z], ...],
                ...
            }
        ),
        ...
    ]

    No rounding or formatting is applied. All values are raw floats read from URDF.
    """
    urdf_path = Path(urdf_path)
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Mapping: radius -> { link_name -> list of xyz vectors }
    grouped = defaultdict(lambda: defaultdict(list))

    # Iterate through all <link> elements
    for link in root.findall("link"):
        link_name = link.get("name")

        # Iterate through all <collision> elements
        for coll in link.findall("collision"):
            geom = coll.find("geometry")
            if geom is None:
                continue

            # Look for <sphere> under <geometry>
            sphere = geom.find("sphere")
            if sphere is None:
                continue  # not a sphere collision

            radius_str = sphere.get("radius")
            if radius_str is None:
                continue

            radius = float(radius_str)

            # Read <origin xyz="..."> if present
            origin = coll.find("origin")
            if origin is not None and origin.get("xyz") is not None:
                xyz_str = origin.get("xyz")
                xyz = [float(v) for v in xyz_str.split()]
            else:
                # Default origin for collision if not specified
                xyz = [0.0, 0.0, 0.0]

            grouped[radius][link_name].append(xyz)

    # Convert mapping to list format sorted by radius
    spheres_list = []
    for r in sorted(grouped.keys()):
        spheres_list.append((r, dict(grouped[r])))

    return spheres_list
