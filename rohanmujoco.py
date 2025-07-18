import mujoco
import mujoco_viewer
from robot_descriptions import op3_mj_description

model = mujoco.MjModel.from_xml_path(op3_mj_description.MJCF_PATH)
data = mujoco.MjData(model)

# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data)

# simulate and render
for _ in range(10000):
    if viewer.is_alive:
        mujoco.mj_step(model, data)
        viewer.render()
    else:
        break

# close
viewer.close()
