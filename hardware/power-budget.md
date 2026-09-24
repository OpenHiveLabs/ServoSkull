# Power Budget — MG996R build

One 3S LiPo powers the whole robot. It rides on a **power sled** that sits in the Part 1 stand and later slides
into the disc body of the walking build. The skull clicks onto either base through its spine opening: one keyed
connector carries the **pack voltage** up and I²C down. The Pi's 5.1 V is made **inside the skull**.

**The skull is only moved between the stand and the disc when powered down**: say "power down" (or shut the Pi
down), wait for the activity LED to stop, unplug the LiPo at its XT60, move the skull, plug the LiPo back in.

## Architecture

```
  POWER SLED  (in the Part 1 stand now, in the disc body later)
  3S LiPo (11.1 V nominal · 12.6 V full · 9.9 V cutoff)   ← the only power source
      │ XT60  (the single disconnect for the whole robot)   + capacity gauge / alarm
  PDB-XT60 (distribution, 25 A ×4 pads)
      │
      ├── pad 1 ── fuse (value TBD) ── seated interlock (MOSFET) ── SPINE: VBAT ────────► SKULL
      │                                                                                    ├── XL4016, set 5.1 V ── USB-C pigtail ── Raspberry Pi 5
      │                                                                                    │     ├── 5 V fans (visible cooling)       ├── 3.3 V → both PCA9685 VCC (signal only)
      │                                                                                    │     └── MAX98357A supply (TBD)           ├── Pi Camera 3 NoIR (CSI)
      │                                                                                    │                                          ├── USB mic
      │                                                                                    │                                          └── I2S → MAX98357A
      │                                                                                    └── head servo rail ◄── SPINE: HEAD_SERVO
      │
      └── pad 2 ── 20 A fuse ── 300 W / 20 A buck, set 6.0 V, CC ~15 A
                                  ├── SPINE: HEAD_SERVO (6.0 V) ─► 1 × MG90S (jaw) + 5 g eye servos
                                  └── E-stop ── leg servo rail (14 AWG)            ← walking build only
                                                 ├── 12 × MG996R (legs)
                                                 └── 2200–4700 µF bulk caps

  Spine connector: VBAT · GND (several) · HEAD_SERVO · 3V3 · SDA · SCL · LEG_OE · ESTOP_SENSE · SEATED loop · spares
  PDB 12 V output and 5 V BEC: unused
  Common ground: pack − · PDB − · buck − · XL4016 − · Pi GND · both PCA9685 GND
```

**Seated interlock.** Two short contacts close only when the skull is fully seated, and switch VBAT on through a
MOSFET on the sled. Lifting the skull breaks the loop before the power pins part, so a forgotten shutdown doesn't
arc on the connector or put pack voltage on a logic pin. It does not protect the SD card, so shut down first.

**Leg safe state is limp.** The leg PCA9685's OE is pulled up (outputs off) in the disc; the skull must drive it low
to enable the legs. Unpowered or disabled, the legs go limp and the disc settles onto its skids.

**Why 3S and not 2S.** A buck needs input headroom above its output. A 2S pack
sits at ~6.6 V near cutoff, which leaves the buck almost nothing to regulate
6.0 V with — the servo rail would sag exactly when the pack is tired. 3S gives
~4 V of headroom all the way down.

**Why 6.0 V and not 7.2 V.** 6 V is where the datasheet torque is specified, and
it leaves margin for ripple and overshoot below the 7.2 V ceiling.

**Why two branches.** A servo stall sags the servo rail, not the Pi's. The Pi
browning out mid-step is the classic way a first quadruped falls and loses the
log that would explain why.

**Why 5.1 V for the Pi.** A little headroom for the drop across the pigtail,
so the Pi still sees ≥ 5.0 V under load.

**Why the XL4016 is in the skull.** Pack voltage crosses the spine connector instead of 5.1 V. At 5.1 V × 5 A the
whole Pi supply path had only ~20 mΩ to spare; at pack voltage the connector carries at most 2.86 A for the Pi
(9.9 V, 90 % efficiency) and the XL4016 regulates any drop away. It needs input headroom: 9.9 − 5.1 = 4.8 V at
cutoff; its minimum dropout is TBD from the datasheet. The cost is heat in the skull, handled by fans.

**Why a USB-C pigtail, not the GPIO 5 V pins.** GPIO power bypasses the Pi's
input protection.

## Current draw

### Part 1: the skull on the stand

The skull alone, from the numbers above (XL4016 at 5.1 V, ~90 % efficient, one 5000 mAh pack to 80 %, 4000 mAh
usable). Pi only; the fans, the amp and the head servos come on top and are **not measured yet**, so these runtimes
are upper bounds.

| Pi load at 5.1 V | At the pack (11.1 V) | Spine at 9.9 V | Runtime (Pi only) | Heat in the skull |
|---|---:|---:|---:|---:|
| 1.2 A (idle) | 0.61 A | 0.69 A | ~6.5 h | 6.8 W |
| 2.5 A (STT + vision) | 1.28 A | 1.43 A | ~3.1 h | 14.2 W |
| 5.0 A (rating) | 2.55 A | 2.86 A | ~1.6 h | 28.3 W |

Almost all of the heat is the Pi's; the XL4016 adds about 10 %. The skull is closed PLA, so the fans are
required, not optional. Pi: heatsink plus the skull fans.

### Walking build

Assumptions: MG996R ~0.5 A holding under load, ~0.9 A moving, 2.5 A stall
(datasheet). Buck and XL4016 efficiency ~90%. Pi 5 1.2–2.5 A at 5 V.

Estimates from the v1 plan, unchanged by moving the XL4016 into the skull (the same load, a different place).

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
| Pi supply | 5 A continuous, 5.1 V | **XL4016 in the skull**, 5 A continuous / 8 A with fan, 4–38 V in | ⚠️ at its continuous rating — in the fans' airflow, not sealed |
| Spine feed | ≤ 2.86 A for the Pi at 9.9 V, plus fans, amp | connector TBD (current rating from its datasheet) | TBD |
| Fuse | above peak, below fault | **20 A** on the servo branch | ✅ |
| Servo wiring | 15 A peak | **14 AWG** trunk, stock leads to each servo | ✅ |
| Bulk caps | four legs moving together | 2–3 × 2200–4700 µF, 16 V, near the servo connectors | ✅ |

**Rejected for the Pi:** PDB's own 5 V BEC (2 A), SkyRC UBEC (3 A continuous),
Matek UBEC Duo (4 A). 3 A-class supplies brown the Pi 5 out under STT load.

**Spares:** one extra 20 A buck and one extra XL4016. The sled uses one of each, so the spares stay spares.

**CAD space:** buck 60 × 53 × 27 mm · XL4016 60 × 37 mm · PDB 36 × 50 mm
(30.5 mm mount pattern).

## ✅ Verify before power-on

- [ ] **Servo buck: CV pot 20+ turns counter-clockwise first** (at the top end
      output = input = 12.6 V), then set **6.0 V** on a multimeter with nothing
      connected.
- [ ] Servo buck CC limit set to ~15 A.
- [ ] **XL4016 set to 5.1 V** on a multimeter, nothing connected.
- [ ] **Both modules labelled** after setting. Spare buck labelled "unset".
- [ ] 20 A fuse fitted on the servo branch; spine feed fused on the sled.
- [ ] Seated interlock: VBAT is off at the connector with the skull lifted (measure it).
- [ ] Nothing connected to the PDB's 12 V output.
- [ ] Common ground across all branches.
- [ ] PCA9685 VCC on the Pi's **3.3 V**; servo power **not** through its V+ terminal.
- [ ] Bulk capacitors on the servo rail, polarity checked.
- [ ] LiPo low-voltage alarm fitted, set **10.2–10.5 V**. 3S never below 9.9 V.
- [ ] If USB devices misbehave on the Pi: it limits USB to 600 mA on a non-PD
      supply. Add `usb_max_current_enable=1` to `config.txt` — only after the
      XL4016 is confirmed to hold 5 V under load.

## E-stop

A physical switch on the disc cutting the **leg servo rail only**, leaving the Pi and the head alive and
logging. The skull reads its state over ESTOP_SENSE.

Note: when MG996Rs lose power they go limp, so **the robot drops**. That's still
the right failure mode, but test the E-stop with the skull supported, not
standing.
