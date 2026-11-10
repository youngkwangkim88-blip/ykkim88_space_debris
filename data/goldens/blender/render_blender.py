"""Blender-only orthographic reference for a declared outgoing-radiance law.

The custom law uses shader math plus an emission closure, not a substitute
microfacet sampler. For convex, distant-source targets this realizes the
specified directional radiance and independently integrates projected area.
"""
from pathlib import Path
import json
import math
import sys
import bpy
from mathutils import Vector, Quaternion
import numpy as np


def math_node(nodes, links, operation, a, b=0.0):
    node = nodes.new('ShaderNodeMath')
    node.operation = operation
    for index, value in enumerate((a, b)):
        if isinstance(value, (float, int)):
            node.inputs[index].default_value = value
        else:
            links.new(value, node.inputs[index])
    return node.outputs[0]


def dot(nodes, links, normal, vector):
    node = nodes.new('ShaderNodeVectorMath')
    node.operation = 'DOT_PRODUCT'
    links.new(normal, node.inputs[0])
    node.inputs[1].default_value = vector
    return math_node(nodes, links, 'MAXIMUM', node.outputs['Value'], 0.0)


def material(case):
    mat = bpy.data.materials.new('declared_' + case['law'])
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    out = nodes.new('ShaderNodeOutputMaterial')
    rho = case.get('rho_channels', [case['rho']] * 3)
    if case['law'] == 'cycles_lambert':
        shader = nodes.new('ShaderNodeBsdfDiffuse')
        shader.inputs['Color'].default_value = (*rho, 1.0)
        shader.inputs['Roughness'].default_value = 0.0
    elif case['law'] == 'cycles_ggx':
        shader = nodes.new('ShaderNodeBsdfPrincipled')
        shader.distribution = 'GGX'
        shader.inputs['Base Color'].default_value = (*rho, 1.0)
        shader.inputs['Metallic'].default_value = case.get('metallic', 0.0)
        shader.inputs['Roughness'].default_value = case['roughness']
        shader.inputs['IOR'].default_value = case.get('ior', 1.5)
        shader.inputs['Coat Weight'].default_value = 0.0
        shader.inputs['Transmission Weight'].default_value = 0.0
    elif case['law'] in ('declared_blinn', 'declared_lambert', 'unit_radiance'):
        shader = nodes.new('ShaderNodeEmission')
        shader.inputs['Color'].default_value = (*rho, 1.0)
        if case['law'] == 'unit_radiance':
            shader.inputs['Strength'].default_value = 1.0
        else:
            geometry = nodes.new('ShaderNodeNewGeometry')
            normal = geometry.outputs['True Normal']
            sun = Vector(case['sun']).normalized()
            mu_s = dot(nodes, links, normal, sun)
            if case['law'] == 'declared_blinn':
                half = (sun + Vector(case['view']).normalized()).normalized()
                mu_h = dot(nodes, links, normal, half)
                power = math_node(nodes, links, 'POWER', mu_h, case['m'])
                m, f = case['m'], case['f']
                c = (m + 2) * (m + 4) / (8 * (m + 2 ** (-m / 2)))
                mixed = math_node(nodes, links, 'ADD',
                    math_node(nodes, links, 'MULTIPLY', power, f * c), 1 - f)
                rate = math_node(nodes, links, 'MULTIPLY', mu_s, mixed)
            else:
                rate = mu_s
            rate = math_node(nodes, links, 'MULTIPLY', rate, 1 / math.pi)
            links.new(rate, shader.inputs['Strength'])
    else:
        raise ValueError('unknown explicit law: ' + case['law'])
    links.new(shader.outputs[0], out.inputs['Surface'])
    return mat


def orient(obj, direction, distance):
    obj.location = Vector(direction).normalized() * distance
    obj.rotation_euler = (-obj.location).to_track_quat('-Z', 'Y').to_euler()


def render(case, output):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = case.get('samples', 16)
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.seed = case.get('seed', 20260924)
    scene.cycles.max_bounces = 1
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 1
    res = case.get('resolution', 256)
    scene.render.resolution_x = scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '32'
    scene.render.film_transparent = False
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    world = bpy.data.worlds.new('black')
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.0
    scene.world = world
    dims = case['dimensions_m']
    if case['geometry'] == 'plane':
        bpy.ops.mesh.primitive_plane_add(size=1)
        obj = bpy.context.object
        obj.scale = (dims[0], dims[1], 1)
    elif case['geometry'] == 'cylinder':
        bpy.ops.mesh.primitive_cylinder_add(vertices=case.get('cylinder_sides', 512),
            radius=dims[0] / 2, depth=dims[2], end_fill_type='NGON')
        obj = bpy.context.object
    elif case['geometry'] == 'box':
        bpy.ops.mesh.primitive_cube_add(size=1)
        obj = bpy.context.object
        obj.dimensions = dims
    else:
        raise ValueError('unsupported reference geometry')
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = Quaternion(case['quaternion_wxyz'])
    obj.data.materials.append(material(case))
    bpy.ops.object.camera_add()
    cam = bpy.context.object
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = case.get('ortho_scale_m', .32)
    orient(cam, case['view'], 3.0)
    scene.camera = cam
    if case['law'].startswith('cycles_'):
        bpy.ops.object.light_add(type='SUN')
        light = bpy.context.object
        orient(light, case['sun'], 3)
        light.data.energy = 1.0
        light.data.angle = 0.0
    exr = output / (case['id'] + '.exr')
    scene.render.filepath = str(exr)
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(str(exr), check_existing=False)
    rgb = np.array(image.pixels[:], dtype=np.float32).reshape(res, res, 4)[..., :3]
    np.save(output / (case['id'] + '.npy'), rgb)
    pixel_area = (cam.data.ortho_scale / res) ** 2
    integrated = rgb.astype(np.float64).sum(axis=(0, 1)) * pixel_area
    bpy.data.images.remove(image)
    return dict(**case, blender_version=bpy.app.version_string,
        integrated_channels_m2_sr=integrated.tolist(), pixel_area_m2=pixel_area,
        image_domain='scene-linear relative radiance per unit incident channel irradiance',
        custom_law_scope='prescribed directional radiance, not a stock BSDF sampler',
        stock_law_scope='Cycles direct distant-light reflection', exr=exr.name)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Rerender the public linear Blender reference images")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--group", choices=("poses", "exposure"), required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    request = json.loads(Path(__file__).with_name("cases.json").read_text(encoding="utf-8"))
    cases = request["poses"] if args.group == "poses" else request["exposure"]["cases"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = [render(case, args.output_dir) for case in cases]
    (args.output_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
