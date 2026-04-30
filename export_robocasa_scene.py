"""Export a RoboCasa kitchen scene as MuJoCo XML for use in InstinctMujoco."""
import argparse
import os
import sys

import numpy as np
import robosuite
import robocasa
import robocasa.environments
from robosuite.controllers import load_composite_controller_config


def export_scene(layout_id=1, style_id=1, output_path="robocasa_kitchen.xml"):
    """Generate a RoboCasa kitchen scene and export it as XML."""
    config = {
        "env_name": "Kitchen",
        "robots": "PandaOmron",
        "controller_configs": load_composite_controller_config(robot="PandaOmron"),
        "translucent_robot": False,
    }

    print(f"Creating RoboCasa environment (layout={layout_id}, style={style_id})...")
    env = robosuite.make(
        **config,
        layout_ids=layout_id,
        style_ids=style_id,
        has_renderer=False,
        has_offscreen_renderer=False,
        render_camera=None,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )

    env.reset()

    # Get the MuJoCo model and save as XML
    model = env.sim.model
    xml_string = env.sim.model.get_xml()

    # Save to file
    output_path = os.path.abspath(output_path)
    with open(output_path, "w") as f:
        f.write(xml_string)
    print(f"Exported scene to: {output_path}")

    env.close()
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", type=int, default=1, help="Kitchen layout ID (1-60)")
    parser.add_argument("--style", type=int, default=1, help="Kitchen style ID (1-60)")
    parser.add_argument("--output", type=str, default="robocasa_kitchen.xml", help="Output XML path")
    args = parser.parse_args()

    export_scene(layout_id=args.layout, style_id=args.style, output_path=args.output)
