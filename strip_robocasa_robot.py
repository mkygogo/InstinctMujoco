"""Strip the PandaOmron robot from exported RoboCasa kitchen XML.

Produces a scene-only XML that can be loaded and combined with our G1 robot.
"""
import re
import sys
from pathlib import Path
from lxml import etree


def strip_robot(input_xml: str, output_xml: str):
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(input_xml, parser)
    root = tree.getroot()

    # Remove robot-related bodies from worldbody
    worldbody = root.find("worldbody")
    bodies_to_remove = []
    for body in worldbody.findall("body"):
        name = body.get("name", "")
        if name in ("robot0_base", "left_eef_target", "right_eef_target"):
            bodies_to_remove.append(body)
    for b in bodies_to_remove:
        worldbody.remove(b)

    # Remove robot meshes from asset
    asset = root.find("asset")
    meshes_to_remove = []
    for mesh in asset.findall("mesh"):
        name = mesh.get("name", "")
        if name.startswith("robot0_") or name.startswith("mobilebase0_") or name.startswith("gripper0_"):
            meshes_to_remove.append(mesh)
    for m in meshes_to_remove:
        asset.remove(m)

    # Remove robot materials
    mats_to_remove = []
    for mat in asset.findall("material"):
        name = mat.get("name", "")
        if name.startswith("robot0_") or name.startswith("mobilebase0_") or name.startswith("gripper0_"):
            mats_to_remove.append(mat)
    for m in mats_to_remove:
        asset.remove(m)

    # Remove robot textures
    texs_to_remove = []
    for tex in asset.findall("texture"):
        name = tex.get("name", "")
        if name.startswith("robot0_") or name.startswith("mobilebase0_") or name.startswith("gripper0_"):
            texs_to_remove.append(tex)
    for t in texs_to_remove:
        asset.remove(t)

    # Remove actuator section entirely
    actuator = root.find("actuator")
    if actuator is not None:
        root.remove(actuator)

    # Remove sensor section entirely
    sensor = root.find("sensor")
    if sensor is not None:
        root.remove(sensor)

    # Remove equality section (often has robot constraints)
    equality = root.find("equality")
    if equality is not None:
        root.remove(equality)

    # Fix compiler: remove meshdir since all paths are absolute
    compiler = root.find("compiler")
    if compiler is not None:
        if "meshdir" in compiler.attrib:
            del compiler.attrib["meshdir"]

    # Write output
    tree.write(output_xml, xml_declaration=True, encoding="utf-8", pretty_print=True)
    print(f"Stripped scene saved to: {output_xml}")
    print(f"  Removed {len(bodies_to_remove)} robot bodies")
    print(f"  Removed {len(meshes_to_remove)} robot meshes")
    print(f"  Removed {len(mats_to_remove)} robot materials")


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "robocasa_kitchen.xml"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "robocasa_kitchen_scene.xml"
    strip_robot(input_path, output_path)
