# skull_kinematics

Closed-form kinematics and the crawl gait for ServoSkull. **In progress.**

## Interface

The rest of the robot will import these, so the names and signatures are the contract.

### `params.py`
- `load()` → reads `robot_params.yaml`, returns a `RobotParams` with
  `.geometry` (link lengths in **metres**), `.limits`, `.legs`, `.gait`,
  `.actuator`, `.body`, `.meta`, and `.leg(name)`.
- Validates at load time.

### `leg_ik.py`
- `inverse_kinematics(target_xyz, geom) -> JointAngles(coxa, femur, tibia)`
- `forward_kinematics(angles, geom) -> np.ndarray`
- `neutral_stance(geom, body_height, splay=None) -> np.ndarray`
- `within_limits(angles, limits)`, `clamp_to_limits(angles, limits)`
- `UnreachableTarget(ValueError)`

Leg frame: origin on the coxa axis, +x outward along the leg's zero direction,
+z up. Conventions: `coxa=0` points along +x; `femur=0` is horizontal, positive
raised; `tibia=0` is a straight leg, positive folds the knee.

### `body_ik.py`
- `BodyPose(position, roll, pitch, yaw)` with `.rotation()`
- `world_to_leg(foot, pose, mount)` / `leg_to_world(foot, pose, mount)`
- `solve_body(pose, feet_world, params) -> dict[str, JointAngles]`

### `gait.py`
- `bezier_swing(start, end, height, s)`
- `support_polygon(feet, airborne)` / `stability_margin(com_xy, polygon)`
- `CrawlGait(params)` with `.phase()`, `.is_swinging()`, `.airborne_leg()`,
  `.foot_offset()`, `.neutral_feet()`, `.body_shift()`, `.feet_at()`, `.com_at()`

## Build order

1. `params.py`
2. `leg_ik.py`: the coxa axis is vertical, so it decouples and leaves a planar 2-link problem
3. `body_ik.py`: frame transforms
4. `gait.py`: the crawl, and the stability check that validates it
