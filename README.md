# ServoSkull

A robot skull that talks, sees and, eventually, walks on four spider legs — and the first build of
**[OpenHiveLab](https://youtube.com/@OpenHiveLab)**, a channel about the long
road to hive robotics.

The arc is: one robot → robots that talk to each other → drones → a hive.
This is robot number one. Everything is built from scratch and documented as it
happens, mistakes included.

**Fiction to Reality, Project 1.** Built in parts, one video each:

| Part | What it is | Status |
|---|---|---|
| **1** | **Talking skull**: a static workshop assistant. Voice, a moving jaw, decorative eyes, a camera | in progress |
| 2 | **Train to walk**: the leg kinematics and the crawl gait in MuJoCo | in progress |
| 3 | **Walking skull**: legs, bench tests, the full build, first steps | planned |

<!-- TODO: hero shot / demo gif once it walks -->

---

## What it is

| | |
|---|---|
| **Head (Part 1)** | a printed anatomical skull; jaw on an MG90S, decorative eyes on 5 g micro servos, fixed camera |
| **Brain** | Raspberry Pi 5 inside the skull, Claude API over WiFi |
| **Senses** | Pi Camera Module 3 NoIR, USB microphone; later an IMU and foot contact |
| **Voice** | wake word → Whisper → Claude → TTS, jaw synced to the audio |
| **Power** | 3S LiPo → separate 5.1 V (Pi) and 6.0 V (servo) rails |
| **Legs (Part 3)** | 4 × 3 DOF, spider/insect topology (coxa yaw → femur → tibia), 12 × MG996R via one PCA9685 |
| **Gait** | statically stable crawl, omnidirectional |

## Simulate before you draw

The link lengths in this repo were not guessed and then adjusted on the bench.
They were **swept in MuJoCo against stability and reachability gates before any
CAD existed**, and the winning numbers became the CAD specification.

Two things fell out of that which would have been expensive to discover in
plastic:

1. **A centred body cannot crawl.** Lifting a leg puts the centre of mass on or
   outside the edge of the remaining support triangle — two of the four
   triangles have a *negative* stability margin. The body has to sway toward
   the triangle it is about to stand on. This is why crawling robots waddle;
   it is load-bearing, not decorative.
2. **Reach, then torque.** With the servo first chosen (STS3215, 30 kg·cm) the
   design was reach-limited: peak femur torque ~12.8 kg·cm, but only ~24 mm of
   reach headroom. The build then switched to the affordable MG996R (~9 kg·cm
   usable), which makes torque a limit too. The leg geometry is being
   re-derived for it before any leg CAD — that is Part 2.

## Quick start

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

The kinematics, simulation harness and tests are being rewritten from scratch
and will land here as they're finished.

## Layout

```
robot_params.yaml     ← SINGLE SOURCE OF TRUTH for every dimension
software/
  skull_kinematics/   closed-form leg IK, body IK, crawl gait  (in progress)
  skull_core/         head + leg servo drivers, gait engine, IMU, safety
  skull_brain/        Claude tools, STT/TTS, wake word         (Part 1)
  skull_bridge/       pub/sub seam — local now, Zenoh later
sim/                  MJCF generator + physics harness      (in progress)
cad/                  Fusion sources, print files, sim meshes
hardware/             BOM, power budget, wiring
docs/                 CAD spec (generated)
tests/                the correctness gates                 (in progress)
```

### One source of truth

Every geometric constant lives in `robot_params.yaml`. The MuJoCo model, the
runtime IK, and the Fusion dimension table are all generated from or checked
against it. Nothing hardcodes a link length.

After any geometry change, the model, the CAD spec and the tests are
regenerated and re-run (the scripts land with the simulation harness).

## Why not ROS 2?

For one developer, one robot, and a deadline, ROS 2 is a large tax paid for
benefits that only arrive at multi-robot. Plain Python gets to walking in weeks.

The swarm is still the destination, so the *seam* is already in: messaging goes
through `skull_bridge` with Zenoh-shaped key expressions. The planned stack is
**Zenoh** for transport (it is an official ROS 2 middleware, so adopting ROS 2
later costs nothing, and `zenoh-pico` reaches microcontrollers for the drone
tier), **Buzz** for swarm coordination, and **ARGoS** for swarm-scale
simulation — MuJoCo does one robot's physics, ARGoS does a hundred robots'
behaviour.

## Status

- [x] **M0** — foundations: repo, BOM, servos and power parts bought
- [x] **M1** — simulation study: link lengths and gait swept against stability and reach (results in `docs/cad-spec.md`); code being rewritten for release

**Part 1 — talking skull**
- [ ] **M2** — skull bay and electronics
- [ ] **M3** — voice loop
- [ ] **M4** — face: jaw sync and eye movement
- [ ] **M5** — persona and workshop skills

**Part 2 — train to walk**
- [ ] **M7** — own simulation and leg geometry for the MG996R

**Part 3 — walking skull**
- [ ] **M9** — servo bench test and pot feedback
- [ ] **M10–M14** — leg CAD, one leg, full build with E-stop, standing, walking
- [ ] **M15** — Claude drives the legs

## Credits

- **Skull model:** [V3_updated_skull_face](https://www.printables.com/model/1507564-v3_updated_skull_face)
  by Gregoory P. Ennis on Printables (CC0), a remix of Printables models 293680 and 2770. It is linked,
  not redistributed here; only our own mounting parts live in `cad/`.
- **Servo feedback idea:** the buffered potentiometer tap on the MG996R from
  [TNY-360](https://github.com/TNY-Robotics/TNY-360) by TNY Robotics (CC BY-NC-SA 4.0). We credit the
  idea; none of their files are copied here.

## Licence

Software under **MIT**. Hardware, CAD and documentation under
**CERN-OHL-W v2** / **CC BY-SA 4.0**. See [LICENSE](LICENSE).
