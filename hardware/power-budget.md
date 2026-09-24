# Power Budget — MG996R build

> **Part 1 (talking skull, 2026-09-23):** the same tree, powered from the LiPo lying on the table next to the
> static skull. Only the head is on the servo rail (1× MG90S jaw + small 5 g eye servos); the 12 leg servos and
> the E-stop come with the walking build. Keep the LiPo alarm on, and never leave the pack charging or powered
> unattended. **Open:** the MG90S and 5 g servo voltage ratings; check them before putting them on the 6.0 V rail.

## Architecture

```
  3S LiPo (11.1 V nominal · 12.6 V full · 9.9 V cutoff)
      │ XT60
  PDB-XT60 (distribution, 25 A ×4 pads)
      │
      ├── pad 1 ── 20 A fuse ── 300 W / 20 A buck ── E-stop ── servo rail (14 AWG)
      │                         set 6.0 V, CC ~15 A              ├── 12 × MG996R (legs)
      │                                                          ├──  1 × MG90S (jaw) + 5 g eye servos
      │                                                          └── 2200–4700 µF bulk caps
      │
      ├── pad 2 ── XL4016, set 5.1 V ── USB-C pigtail ────────── Raspberry Pi 5
      │                                                          ├── 3.3 V → PCA9685 VCC (signal only)
      │                                                          ├── Pi Camera 3 (CSI)
      │                                                          ├── USB mic
      │                                                          └── I2S → MAX98357A
      │
      └── PDB 5 V / 2 A BEC ───────────────────────────────────── small loads only
                                                                 ├── MAX98357A supply
                                                                 └── buck cooling fan (if fitted)

  PDB 12 V output: unused — on 3S it is unregulated (pack − 1 V)
  Common ground: pack − · PDB − · buck − · XL4016 − · Pi GND · PCA9685 GND
```

**Why 3S and not 2S.** A buck needs input headroom above its output. A 2S pack
sits at ~6.6 V near cutoff, which leaves the buck almost nothing to regulate
6.0 V with — the servo rail would sag exactly when the pack is tired. 3S gives
~4 V of headroom all the way down.

**Why 6.0 V and not 7.2 V.** 6 V is where the datasheet torque is specified, and
it leaves margin for ripple and overshoot below the 7.2 V ceiling.

**Why two branches.** A servo stall sags the servo rail, not the Pi's. The Pi
browning out mid-step is the classic way a first quadruped falls and loses the
log that would explain why.

**Why 5.1 V for the Pi.** A little headroom for the drop across the pigtail
and connectors, so the Pi still sees ≥ 5.0 V under load.

**Why a USB-C pigtail, not the GPIO 5 V pins.** GPIO power bypasses the Pi's
input protection.

## Current draw

Assumptions: MG996R ~0.5 A holding under load, ~0.9 A moving, 2.5 A stall
(datasheet). Buck and XL4016 efficiency ~90%. Pi 5 1.2–2.5 A at 5 V.

| Mode | Servo rail @ 6 V | At the pack | Runtime* |
|---|---:|---:|---:|
| Lying down, servos unpowered | 0 A | 0.75 A | ~5 h |
| Standing still, holding pose | ~4.2 A | 3.4 A | ~70 min |
| **Crawling** (9 holding + 3 moving) | **~7.2 A** | **5.3 A** | **~45 min** |
| Transient peak | ~15 A | 10.7 A | — |
| All 12 stalled (fault) | ~30 A | 19.7 A | fuse / CC limit trips |

\* One 5000 mAh pack discharged to 80% = 4000 mAh usable.
The head servos (1 MG90S for the jaw, small 5 g servos for the eyes) add a few hundred mA on the servo rail while moving. *Unmeasured; M2 measures it.*

**Holding is not free.** Unlike the bus servos, an analog MG996R draws current
to hold position under load and hunts back and forth inside its deadband. The
standing-still row is most of the walking cost.

## Sizing the parts

| Part | Requirement | Chosen | Margin |
|---|---|---|---|
| Distribution | ≥ 11 A at the pack, 3S | **PDB-XT60**, 25 A ×4 pads, 9–18 V in | ✅ ample |
| Servo buck | ≥ 15 A peak, 6.0 V | **DC-DC 300 W / 20 A** (15 A recommended), 6–40 V in | ✅ crawling ~43 W of 300 W. Fan on heatsink only for sustained high load |
| Pi supply | 5 A continuous, 5.1 V | **XL4016**, 5 A continuous / 8 A with fan, 4–38 V in | ⚠️ at its continuous rating — mount with airflow, not sealed |
| Fuse | above peak, below fault | **20 A** on the servo branch | ✅ |
| Servo wiring | 15 A peak | **14 AWG** trunk, stock leads to each servo | ✅ |
| Bulk caps | four legs moving together | 2–3 × 2200–4700 µF, 16 V, near the servo connectors | ✅ |

**Rejected for the Pi:** PDB's own 5 V BEC (2 A), SkyRC UBEC (3 A continuous),
Matek UBEC Duo (4 A). 3 A-class supplies brown the Pi 5 out under STT load.

**Spares:** one extra 20 A buck and one extra XL4016. The servo buck is the most
critical part in the tree.

**CAD space:** buck 60 × 53 × 27 mm · XL4016 60 × 37 mm · PDB 36 × 50 mm
(30.5 mm mount pattern).

## ✅ Verify before power-on

- [ ] **Servo buck: CV pot 20+ turns counter-clockwise first** (at the top end
      output = input = 12.6 V), then set **6.0 V** on a multimeter with nothing
      connected.
- [ ] Servo buck CC limit set to ~15 A.
- [ ] **XL4016 set to 5.1 V** on a multimeter, nothing connected.
- [ ] **Both modules labelled** after setting. Spare buck labelled "unset".
- [ ] 20 A fuse fitted on the servo branch.
- [ ] Nothing connected to the PDB's 12 V output.
- [ ] Common ground across all branches.
- [ ] PCA9685 VCC on the Pi's **3.3 V**; servo power **not** through its V+ terminal.
- [ ] Bulk capacitors on the servo rail, polarity checked.
- [ ] LiPo low-voltage alarm fitted, set **10.2–10.5 V**. 3S never below 9.9 V.
- [ ] If USB devices misbehave on the Pi: it limits USB to 600 mA on a non-PD
      supply. Add `usb_max_current_enable=1` to `config.txt` — only after the
      XL4016 is confirmed to hold 5 V under load.

## E-stop

A physical switch cutting the **servo rail only** (between buck output and the
distribution rail), leaving the Pi alive and logging.

Note: when MG996Rs lose power they go limp, so **the robot drops**. That's still
the right failure mode, but test the E-stop with the skull supported, not
standing.
