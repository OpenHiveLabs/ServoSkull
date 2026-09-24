# Bill of Materials — ServoSkull v1 (MG996R build)

Budget target **~12,000 TL**. Prices are Turkish market figures gathered
2026-09-11 and 2026-09-15, VAT included. Rows marked *est.* are estimates, not
quotes. **Confirm before ordering.**

## Actuators — the decision that cascades

| Option | Unit | ×12 | Feedback | Verdict |
|---|---:|---:|---|---|
| **MG996R** (metal gear, PWM) | ~200 TL | ~2,400 TL | ❌ (pot-tap moddable, stage 2) | **Chosen** — local stock, fits budget |
| Feetech STS3215 (7.4 V) | ~$27 | ~$324 | ✅ encoder, load, temp | Rejected — ~17k TL for 12 |
| Feetech STS3215 (12 V / 9 V variants) | ~$30 | ~$360 | ✅ | Rejected — same reason |
| DS3218MG (20 kg·cm) | ~990–1,285 TL | ~12–15k TL | ❌ | Badly priced in TR — skip |

Decision record: vault `ADR-007 MG996R Servos` (supersedes ADR-002).

### MG996R — datasheet (Handsontec EMH-1056)

| | |
|---|---|
| Stall torque | 9.4 kg·cm @ 4.8 V · **11 kg·cm @ 6 V** |
| Speed | 0.17 s/60° @ 4.8 V · **0.14 s/60° @ 6 V** (≈ 430 °/s) |
| Voltage | 4.8–7.2 V — **we run 6.0 V** |
| Current | ~500 mA running · **2.5 A stall @ 6 V** |
| Deadband | 5 µs (≈ 0.5° on paper — clones often worse, measure it) |
| Signal | 50 Hz, 1–2 ms nominal |
| Rotation | **~120° on 1–2 ms** per datasheet; clones often ~180° on 0.5–2.5 ms |
| Mass / size | 55 g · 40.7 × 19.7 × 42.9 mm |

**Buy the 180° (positional) version, not "360°".** A 360° MG996R is a
continuous-rotation servo: PWM sets speed, not angle, and it cannot hold a
pose. In Turkish listings avoid "360 derece" / "sürekli dönüş".

### ⚠️ What this servo means for the design

- **Torque.** Plan on **~9 kg·cm usable** (11 derated for clones and continuous
  holding). The current geometry in `robot_params.yaml` needs **12.8 kg·cm** peak
  on the femur. **It will not stand as specced.** Resolve the geometry *before*
  modelling the leg mounts in Fusion.
- **Travel.** `limits.femur` is ±90° = 180° of travel. The datasheet promises
  ~120°. Measure each servo's real end stops at M3 before trusting 180°.
- **Speed is fine.** ~430 °/s against the crawl's ~240 °/s need.
- **Mass.** 12 leg servos are **660 g — 44% of the 1500 g budget** before a
  single printed part.
- **No feedback.** The Pi commands an angle and cannot know if it got there, or
  if a servo stalled or died. Stage 2 adds it.

## Full list

**Stage** = the milestone that first needs it (v2 plan, 2026-09-23):
**M2** Part 1, the talking skull (head electronics, powered from the LiPo on the table),
**M9** servo bench test, **M12** full build of the walking skull.

| # | Item | Qty | TL | Stage | Notes |
|---|---|---:|---:|---|---|
| | **Power** | | | | |
| 1 | PDB-XT60 w/ BEC power distribution board | 1 | 279 | M2 | Motorobit. XT60 hub after the pack; 25 A ×4 pads. Its 5 V / 2 A BEC is **too small for the Pi** — small loads only |
| 2 | DC-DC 300 W / 20 A buck (CV/CC) | 2 | 1,069 | M2 | Motorobit, 534.60 each. **Servo rail, 6.0 V** + 1 spare. 15 A recommended continuous |
| 3 | XL4016 8 A step-down | 2 | 475 | M2 | Motorobit, 237.60 each. **Pi 5 supply, 5.1 V** + 1 spare. 5 A continuous |
| 4 | 3S 11.1 V 5000 mAh 100C LiPo (Profuse) | 1 | 4,786 | M2 | Motorobit, **bought** 2026-09-15 (the 65C at 2,039 was quoted, not bought). 3S, not 2S — see power budget |
| 5 | LiPo balance charger + fire bag | 1 | 600 | M2 | *est.* |
| 6 | LiPo low-voltage alarm | 1 | 100 | M2 | *est.* set 10.2–10.5 V |
| 7 | Bulk capacitors 2200–4700 µF, 16 V | 3 | 105 | M12 | *est.* on the servo rail |
| 8 | 20 A blade fuse + holder, XT60 pigtails, 14 AWG | — | 250 | M2 | *est.* PDB replaces terminal blocks |
| 9 | USB-C pigtail cable | 1 | 100 | M2 | *est.* XL4016 → Pi. Not the GPIO 5 V pins |
| 10 | E-stop switch, ≥ 15 A DC | 1 | 150 | M12 | *est.* cuts the servo rail only |
| | **Actuation** | | | | |
| 11 | MG996R servo, **180°** | 15 | 3,447 | M9 | **Bought** from Robotistan 2026-09-15, 229.78 each (Robot Sepeti quoted 182.65). 12 legs + 3 spares. **One batch, one seller** — clones vary |
| 12 | PCA9685 16-ch PWM (clone) | 1 | 240 | M2 | drives the head servos first, then all 15 PWM servos; Robotzade has it at ~130 |
| 13 | Metal 25T servo horns | 12 | 600 | M9 | *est.* plastic horns strip under load — would spoil the torque test |
| 14 | Servo extension leads, 30 cm | 12 | 435 | M12 | Motorobit, 36.24 each. Femur/tibia servos can't reach the skull on stock leads |
| | **Structure** | | | | |
| 15 | PETG (structural) + PLA (shell), ~1.5 kg | — | 900 | M2 | *est.* |
| 16 | M3 heat-set inserts, bearings, screws | — | 700 | M2 | *est.* |
| | **Sensing & head** | | | | |
| 17 | BNO055 IMU (GY clone) | 1 | 375 | M12 | genuine Adafruit ~1,980 |
| 18 | Foot microswitches | 4 | 200 | M12 | *est.* |
| 19 | Raspberry Pi Camera Module 3 **NoIR** | 1 | 1,782 | M2 | **Bought** from Robotistan 2026-09-15. Fixed camera in the skull; the eyes are decorative |
| 20 | MAX98357A I2S amp | 1 | 150 | M2 | *est.* Pi 5 has no analog audio |
| 21 | 3 W 4 Ω speaker | 1 | 120 | M2 | *est.* |
| 22 | USB microphone | 1 | 300 | M2 | *est.* |
| 23 | MG90S micro servo | 1 | 65 | M2 | *est.* jaw |
| 24 | 5 g micro servo | 2 | — | M2 | **owned** (10+). Decorative eyes; the count depends on the eye design |
| — | Raspberry Pi 5 (8 GB) | 1 | — | — | **owned** |
| — | Old BEC 5 V | 1 | — | — | **owned, untested — not used.** XL4016 (#3) replaces it |
| | **Total** | | **~17,228** | | purchases where marked **bought**, quotes and estimates otherwise (recomputed 2026-09-23) |

### Order now (M3) — ~10,200 TL *(historical: the v1 order plan of 2026-09-15)*

Rows 1–9, 11–13, 15–16. The Motorobit part of it (rows 1–4) is ~3,860 TL,
over their 2,500 TL free-shipping line on its own.

### Stage 2 — joint feedback (after it walks)

| Item | Qty | TL | Notes |
|---|---:|---:|---|
| MCP3008 8-ch 10-bit ADC | 2 | 252 | 16 channels over SPI; beats 3× ADS1115 |
| Thin hookup wire | — | — | one wire from each pot wiper |

The pot-tap mod: open each servo, solder a wire to the potentiometer wiper,
read it through the MCP3008. The Pi 5 has no ADC of its own.

## Getting under 12,000 TL *(historical, v1)*

The power spares and the separate Pi supply pushed the total over budget.

| Cut | Saves | Running total |
|---|---:|---:|
| USB webcam instead of Pi Camera 3 | −1,375 | 12,522 |
| Skip the spare buck + spare XL4016 | −773 | 11,749 |
| Foot switches later | −200 | 11,549 |

Webcam + no spares gets under budget. Keep the metal horns — a stripped
plastic horn invalidates the M3 torque test.

## ⚠️ Gotchas

- **Set the buck to 6.0 V with a multimeter *before* connecting any servo.**
  These modules ship with the pot at a random position — at the top end the
  output equals the input. Turn the CV pot **20+ turns counter-clockwise**
  first, then bring it up to 6.0 V. 12.6 V into the servo rail kills all
  fifteen at once.
- **Use the buck's constant-current pot as a soft fuse** — set ~15 A.
- **Label the two bucks after setting them.** They're identical modules; a mix-up
  puts 6 V into the Pi. Set the XL4016 to **5.1 V** before the Pi touches it.
- **The PDB's 12 V output is not regulated on 3S** (pack voltage − 1 V). Don't use it.
- **Don't feed servo power through the PCA9685's V+ terminal.** Its traces are
  not meant for 12 servos. Run servo power on a separate rail; the PCA9685 only
  carries signal.
- **Power the PCA9685's VCC from the Pi's 3.3 V, not 5 V.** Its I²C pull-ups go
  to VCC, so 5 V back-feeds the Pi's 3.3 V pins. MG996R triggers fine on 3.3 V.
- **Common ground** between the 6 V rail, the Pi, and the PCA9685.
- **The Pi 5 has no analog audio jack.** I2S (MAX98357A) or USB only.
- **Bring up tethered** to a 6 V bench supply if you can borrow one. Battery
  after the gait works.

## Suppliers

MG996R: Robot Sepeti (182.65) · Robotistan (~202–207). PDB, 20 A buck, XL4016,
LiPo, servo leads: Motorobit (XL4016 also F1Depo ~200). PCA9685: Robotistan ·
Robotzade. BNO055: Kartal Otomasyon · F1Depo. Pi Camera 3: Robotistan.
MCP3008: F1Depo.

Rejected for the Pi supply: SkyRC 3/5A UBEC (3 A continuous, 705 TL), Matek
UBEC Duo (4 A), Pololu D36V50F5 (ideal, but 2,194 TL and out of stock).

Everything is **in stock locally** — no import lead time, so servos are no
longer the critical path.
