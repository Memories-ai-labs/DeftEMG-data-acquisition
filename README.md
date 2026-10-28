# DeftEMG data acquisition

Tools for recording EMG with the WAVELETECH 18-channel EMG armband (2000 Hz EMG, IMU at about
217 Hz, USB serial). They prompt cued finger presses at two force levels, and check the recordings
afterwards. This is the data-collection side of DeftEMG, a project estimating finger force
from EMG.

This project is independent and is not affiliated with or endorsed by WAVELETECH.

## Command summary

| Command | What it does |
|---|---|
| `pixi run acquire` | Record trials from the keyboard (`s` start, `e` end, `q` quit) |
| `pixi run viewer` | Live EMG / IMU / spectrum viewer (or add `--viewer` to `acquire` / `prompter`) |
| `pixi run make-sequences` | Generate cued finger-press sequences (JSON files) |
| `pixi run prompter` | Run the scrolling finger-press task while recording |
| `pixi run check-timing` | Timing report for a recorded session |
| `pixi run filter-emg` | Offline EMG filtering into `data/processed` (20 Hz high-pass + 60 Hz notches at every harmonic) |
| `pixi run replay` | Replay a recorded trial: prompts, EMG viewer with prompt markers, camera (placeholder) |
| `pixi run make-synthetic` | Write synthetic prompter sessions, for dry runs |
| `pixi run test` / `lint` / `format` | Tests (no hardware needed), ruff check, ruff format |

Every command takes `--help`. `--simulate` (on `acquire` and `prompter`) uses a simulated
armband, so everything can be tried without hardware. `examples/S01` is a synthetic session
to try the offline tools on (see [Example data](#example-data)).

## Setup

Requires [pixi](https://pixi.sh). Runs on macOS (Apple silicon and Intel) and Linux.

```bash
pixi install
pixi run test
```

### Connecting the armband

1. Power on (hold the button until the status light alternates colours).
2. Plug the USB-C cable into the **communication** port. The other port is charge-only and
   carries no data.
3. Find the serial port: compare `ls /dev/cu.*` (macOS) or `ls /dev/ttyACM*` (Linux) before
   and after plugging in, or run `pixi run python -m serial.tools.list_ports -v`. On macOS use
   the `/dev/cu.*` name (e.g. `/dev/cu.usbmodem1101`). On Linux your user typically
   needs to be in the `dialout` group to open it. If ModemManager is installed, it may probe
   the port for a few seconds after you plug in; wait before starting a session.

`--port` can be left out when only one USB serial device is connected.

## 1. Recording trials from the keyboard

```bash
pixi run acquire --port /dev/cu.usbmodem1101 --subject S01 --viewer
pixi run acquire --simulate --viewer                  # no hardware
```

Keys in the terminal: `s` start trial, `e` end trial, `q` quit.

At start-up the program:
1. stops the armband;
2. turns its built-in filters off and checks that they are off (raw EMG, the default).
   `--device-filter` turns them on instead: power-line 60 Hz and high-pass 60 Hz, changed with
   `--powerline-hz` / `--highpass-hz` (see below);
3. starts streaming.

The armband then streams for the whole session, so the viewer works between trials. Each
trial sets the armband clock to the computer's time (`TIME…` command) and saves only the data
between `s` and `e`. `--per-trial-acq` instead sends `StartDataAcq` / `StopDataAcq` around
each trial.

**Raw EMG (the default; `--device-filter` for the armband's filters).** The armband has one
switch for all its filtering, so the power-line notch can't be turned off alone. The notch and
the high-pass go off together (`SetFilterEN-0`) and the armband sends raw 24-bit EMG. Sessions
with `--device-filter` turn the filters back on. `session.json` records which way each session
ran, and `select_trials` refuses to mix the two unless asked (`allow_mixed`). Filter raw
recordings offline with `filter-emg` (section 4).

The recorded files stay raw. The live stream, and so the viewer, is filtered in software,
causally, with the same filter as offline filtering (see "Offline EMG filtering" below):
- `--live-highpass {none,20,40,60}`: 4th-order Butterworth high-pass, default 20 Hz.
- `--live-notch {none,50,60,comb50,comb60,comb50all,comb60all}`: notches at the mains
  frequency (`--powerline-hz`, 60 Hz).
  - `comb60all` (the default): 60 Hz and every harmonic, odd and even (60, 120, 180, ...,
    960 Hz), as offline `--harmonics all`. Where mains is 50 Hz, pass `--powerline-hz 50` for
    `comb50all`.
  - `60` / `50`: the mains frequency alone.
  - `comb60` / `comb50`: add only the odd harmonics below 1 kHz (60, 180, 300, ..., 900 Hz;
    50, 150, 250, ..., 950 Hz). These are the strongest lines when a mains-powered device is
    attached or mains pickup is high (a loose or freshly placed band). Same as offline
    `--harmonics odd`.
  - Each notch removes about 2 Hz of EMG band.
- `none` for both shows the raw stream, for diagnostics.

The filter starts from the first sample's level (`sosfilt_zi`), so the raw DC offset doesn't
ring through the high-pass. It starts over the same way after a break in the stream (a stall,
or each trial with `--per-trial-acq`). After each start, mains hum takes about half a second to
fade out while the notch settles.

While a session is open on macOS, the computer is kept awake (`caffeinate`, system and
display). On Linux, turn off automatic suspend yourself. If the computer sleeps, USB is
suspended and the armband stops sending. The terminal warns when data stops for more than
0.5 s, and each trial records its stalls. Closing a laptop lid still sleeps it unless it has
power and an external display.

**Subject and donning.** At start-up the command asks for your user ID. If you have recorded
before, it also asks "Did you re-don the band since your last session?".
- **Yes:** the session gets the next donning id.
- **No:** it keeps the current donning id, and records which session of that donning it is
  (`session_in_donning`), when the donning started, and the previous session.
- `--subject S01` skips the first question.
- For scripted runs without a terminal, `--redon` / `--same-donning` answer the second (a real
  session then needs them). Don't keep them in a command you recall from shell history.
- With `--simulate` there are no questions (subject "sim").
- Only sessions with recorded trials count. Simulated and real sessions keep separate
  histories, and history is per `--data-dir`.

A donning is one wearing of the band, so models can be tested on a re-donned band. Prompter
sessions also start with a calibration block (see section 3).

Other options:
- `--arm`: arm wearing the band, default right. It is stored with the session and passed to
  the viewer.
- `--label`: label for every trial.
- `--session`: folder name.
- `--data-dir`.
- `--no-stream`: no live stream.
- `-v`: log serial traffic.

## 2. Live viewer

```bash
pixi run viewer                     # finds the running stream on its own
pixi run viewer --window 10         # seconds of data shown (default 5)
```

The viewer shows:
- **EMG traces:** 18 channels, one colour each (red → orange → yellow → green → blue → back
  to red).
- **Spectrum:** mean of channels, all channels, or a single channel, with lines at
  60/120/180 Hz to check line noise.
- **IMU:** gyro and accelerometer.
- **Band RMS:** a cross-section of the wrist with one disc per electrode, coloured by its EMG
  RMS over the last 200 ms (adjustable), plus an orange shape pointing toward the active side.
  Details below.
- **Status bar:** sample rates, battery, temperature, dropped packets, current trial.

Trial start/end lines appear on the traces; channels start 300 µV apart (Spacing). Layout
options (`prompter --viewer` uses the last two, section 3): `--compact` (the EMG traces and the
band RMS, with their controls), `--beside` (the side panels on the left of the EMG, the
controls on two rows), `--geometry X Y W H` (the window's place, frame included).

**Band RMS layout.** The wrist is drawn as seen from the elbow looking toward the hand:
- **Sides:** dorsal (back of the wrist) up, palm side down. For the right arm the thumb side
  is on the left and the little-finger side on the right; `--arm left` swaps those labels.
- **Assumed wearing:** puck on the back of the wrist, cable pointing up the arm.
- **Channel order:**
  - The puck sits between ch9 and ch10 (dorsal).
  - The band opening (red dash, between ch18 and ch1) is directly opposite, centred on the
    palm side. The opening is about one electrode wide, so the band has 19 positions (360/19°
    apart) with one empty, and ch18 and ch1 are two positions apart.
  - ch10 → ch18 run down the thumb side to the opening, then ch1 → ch9 come back up the
    little-finger side. That direction is an assumption. If the electrode check below shows it
    the other way round, swap `PUCK_NEIGHBOURS` in `src/deftemg_acquisition/viz/band.py`.

To check this on a subject:
- **Wrist extension** should light up the channels near the puck.
- **Wrist flexion** should light up the channels on the palm side.
- **Pressing one electrode** should light up its own disc.

RMS colour scale: a fixed `max`, 100 µV by default, or tick `auto` to follow the most active
channel.

The viewer is a separate process reading the Lab Streaming Layer (LSL) streams `Waveletech-EMG`,
`Waveletech-IMU` and `Waveletech-Markers`. Closing or crashing it never affects recording.
Other LSL tools (e.g. LabRecorder) can read the same streams.

## 3. Cued finger-press task (scrolling prompter)

Prompts scroll down five lanes, one per digit, onto a target row above an outline of the
hand. The hand is viewed from above, so for the right hand the thumb is on the left.
- **Capsule width** is the force level: thin = low, wide = high.
- **Capsule length** is the hold time.
- **Pressing:** press when the capsule reaches the target and hold until it has shrunk away.

### Make sequences

One JSON file per block; one block is recorded as one trial.

```bash
# single-digit set: each digit x force level appears --reps times, shuffled
pixi run make-sequences --set single --reps 3 --out protocols/S01

# multi-digit set: chords of 2 (and/or 3) digits, same or independent force per digit
pixi run make-sequences --set multi --chord-sizes 2 3 --chord-force independent --out protocols/S01

# pinch / squeeze: the thumb with one finger, every digit at the same force
pixi run make-sequences --set pinch --chord-sizes 2 --reps 3 --per-block 12 --out protocols/P01
pixi run make-sequences --set squeeze --chord-sizes 2 --reps 3 --per-block 12 --out protocols/Q01
```

`pinch` means fingertips pressed against the thumb tip, and `squeeze` means thumb and fingers
closed together. By default each prompts 5 chords: the thumb with each other finger, and the
whole hand. `--chord-sizes 2` keeps only the thumb with one finger. Each chord comes at each
force level `--reps` times. Two- and three-finger chords proved too hard; `--chord-sizes 2 3 4
5` (thumb included) brings them back. The prompter shows the task's instructions before each
block and leaves the thumb's lane empty (no capsules, no target rings), since the thumb joins
every press. The prompts, and so the labels, still include it at the chord's level.
Recordings keep `task_set: pinch` / `squeeze`.

| Option | Default | |
|---|---|---|
| `--hold MIN MAX` | 0.5 2.0 | press duration, s (random in range) |
| `--gap MIN MAX` | 0.8 1.5 | rest between presses, s |
| `--per-block` | 15 | presses per block |
| `--lead-in` | 6.5 | s before the first prompt; keep ≥ the prompter's `--lookahead` |
| `--reps` | 2 | repetitions of each combination |
| `--chord-sizes` | 2; 2 5 for pinch / squeeze | digits per press (chords), thumb included |
| `--prefix`, `--seed` | seq, random | file names, randomisation (the seed used is printed and saved in each file) |

Ready-made sets in `protocols/`:
- `demo`: a short tour, four blocks of 5 presses: single presses (each digit once), two-digit
  presses, pinches and squeezes (the thumb with each finger), at both levels (about 18 s each).
- `test`: short 5-press blocks.
- `sync`: quick presses for timing checks.

The recording protocols have one folder per session. Each folder is randomised separately,
and all use `--reps 3`:

| Folder | Task | Chords | Blocks |
|---|---|---|---|
| `single_press/SP04`–`SP08` | single | one digit | 2 × 15 |
| `multi_press/MPR03`–`MPR07` | multi | 2 digits | 4 × 15 |
| `single_pinch/SPI01`–`SPI05`, `single_squeeze/SSQ01`–`SSQ05` | pinch, squeeze | thumb + one finger (`--chord-sizes 2`) | 2 × 12 (`--per-block 12`) |

### Calibration block

Every prompter session starts with a calibration block of about 25 s. Another one comes up
about 10 minutes later, then every 20 minutes (`--recalibrate 10 20`: minutes to the second
block, then between later ones). A due block is put in front of the next task block, when a
block finishes or when you press Shift after a pause; the screen says how long it has been.
`--no-calibration` skips them all. The prompter shows the instructions, and each block is
saved as a trial with `task_set: calibration`:
1. Rest quietly until the first capsule reaches the target, 4 s in (1 s to settle, then 3 s
   of rest; the capsule is on screen from the start).
2. Give each digit one brief press (1.5 s) as hard as is comfortable ("max", the widest
   capsule), thumb to pinky. Near-maximal holds tire the shared forearm muscles quickly, so the
   presses are brief, with 3 s to relax in between. `--calibration-reps 2` presses each digit
   twice, which also measures fatigue, but was found too tiring.
3. Rest a minute before the first task block.

Pinching and squeezing use other postures than pressing, so the block matches the task blocks
it comes before:
- **Before pinch blocks:** step 2 is one brief maximum pinch of each fingertip against the
  thumb tip, index to pinky (about 19 s).
- **Before squeeze blocks:** one maximum squeeze of the thumb and each finger.
- In both, the thumb's lane stays empty, as in those blocks, and the block is saved as
  `calibration_pinch` / `calibration_squeeze` (`task` in its parameters).
- **Single and multi blocks** get the press block above.
- A session's first calibration matches its first task block, and a later one the block it is
  put in front of.

`deftemg_acquisition.data.calibration` turns a block into:
- **Rest noise level per channel:** how well each electrode is in contact this donning.
- **Each digit's maximum:** the EMG amplitude (RMS per channel, rest noise removed) of the
  best effort it takes part in (its press; in a pinch block the thumb's is its strongest
  pinch).
  - It is measured from past the ramp: a third of the hold after onset, at most 1 s, which is
    0.5 s here.
  - With two rounds, `rep_ratio` gives second / first of the same effort, with a warning below
    0.8 (fatigue).
  - A digit whose best effort is under 3x the rest noise on every channel is left out.
  - A block that can't be used (e.g. too little rest) is skipped in favour of the session's
    other blocks.
- **Low / high presses relative to the maximum:** `press_fraction` gives each press's EMG
  amplitude relative to the maximum of the same effort. A single-digit press is compared with
  that digit's maximum press, and a thumb + finger pinch with that pinch's maximum (squeezes
  alike). Chords, and efforts in another posture than the block's, have no such maximum and
  are left out. The result tells you whether the force levels were consistent. It is EMG
  amplitude, not force.
- **Normalisation factors:** `session_gain(cal, mode)`.
  - `max` divides a session's EMG by its mean maximum-press amplitude, so amplitudes are
    relative to maximum effort.
  - `rest` divides by the rest noise SD instead. It corrects for electrode contact but not for
    how hard the subject presses.
  - `rest_channel` divides each channel by its own rest noise SD.
  - `session_calibration(session, before_ns)` picks the block for a trial: the latest
    completed one recorded before it in its session, or the first one after if there is none
    before.

Measure raw recordings on filtered EMG: pass `emg_filter=EmgFilter()` (zero-phase 20 Hz
high-pass and 60 Hz notches at every harmonic). On one raw block, unfiltered EMG gave a rest
SD of 30 µV (mostly mains hum) and a false fatigue warning from movement artifacts at the
press onsets. Filtered, it gave 5 µV and no warnings.

### Run the task

```bash
pixi run prompter --port /dev/cu.usbmodem1101 --subject S01 --viewer \
    protocols/S01/seq_single_*.json protocols/S01/seq_multi_*.json
pixi run prompter --simulate protocols/test           # no hardware; folders are sorted by name
pixi run prompter --replay data/raw/<session> protocols/test   # a recording plays as the armband
```

With `--viewer` the two windows go side by side: the prompter on the left of the screen at full
height, the live viewer filling the right (its band RMS, spectrum and IMU next to the prompter,
the EMG traces beyond them, its controls on two rows). They stay separate windows and
processes, so the viewer can't slow the prompter's frames.

`--replay PATH...` (sessions or trials) feeds a recording through the acquisition path instead
of the armband, to try a setup without hardware: its exact bytes, in the chunks and at the pace
they arrived (`--replay-speed` for faster), with their original spacing; trials without their
`.bin` (synthetic sessions) are sent as packets made from their EMG. Re-recorded, a trial is the
same EMG, sample for sample, with the same breaks. Several trials are joined 0.5 s apart, so a
block running across a join shows a stall there. The recording starts playing when the prompter
opens and plays once, during and between blocks: a block started after it has run out records
no EMG (and that trial can't be loaded). The replayed EMG doesn't follow the prompts on screen.
The session is kept apart from real ones, as with `--simulate`.

Blocks run in the order given. Keys (the prompter window must have focus; click it first):

| Key | Action |
|---|---|
| Shift (or `s`) | Start the current block. It ends by itself 1.5 s after the last prompt and the next block becomes current |
| Enter (or `e`) | End the block early: saved, marked incomplete, block stays current to redo |
| ← / → | Choose a block (between blocks) |
| Esc (or `q`) | Quit |

Options:
- `--hand left`.
- `--lookahead`: seconds of prompts visible, default 6.
- `--sync-patch`: see section 4.
- The same subject / donning questions as `acquire` (`--subject`, `--redon` /
  `--same-donning`), `--session`, and the same filter and stream options.

### Effort meter

`--effort-meter` adds one bar under the target row, whatever digits are prompted. It helps
keep the force levels consistent: lows clearly above rest, highs clearly above lows.

```bash
pixi run prompter --effort-meter protocols/single_press/SP05
pixi run prompter --effort-meter --effort-zones 0.2 0.4 0.65 0.85 protocols/single_pinch/SPI02
```

- **What it shows:** the overall EMG amplitude above rest noise, over the last 0.15 s, as a
  fraction of the calibration maximum of what is prompted.
  - For a single digit, that maximum is the digit's maximum press. For pinches and squeezes,
    it is that pinch's maximum.
  - For other chords, it combines the digits' maxima by adding their EMG power, so each digit
    at 30 % reads 30 %.
  - Between presses the bar is scaled to the next prompt, so rest reads near 0.
- **Target zones:** low 20–40 % and high 65–85 % of the maximum (`--effort-zones`). During a
  press the prompted level's zone is outlined. The fill is green inside it, vermillion above
  it and grey below it. Keep it in the outlined zone while you hold the press.
- **When it is on:** from the session's first completed calibration block, which sets the
  100 %; each later calibration block replaces it.
  - It needs calibration blocks, so it can't be combined with `--no-calibration`.
  - With `--simulate` it shows how the bar looks, but its readings mean nothing, because the
    simulated armband's EMG doesn't follow the prompts.
- **What is saved:** each task block's trial file gets the trace (`effort`, `effort_host_ns`,
  `effort_max_uv`, `effort_level`: −1 between presses, else the index into `level_names`).
  Its `effort_meter` metadata records the zones, window, reference block and filter. Sessions
  with and without the meter can then be told apart.
- **Lag:** the bar runs about 0.1–0.15 s behind the press. Live EMG and the reference both go
  through the live stream's causal filter.

It measures EMG, not force: it keeps EMG levels consistent from press to press. Feedback
changes how people press, so also record some sessions without it.

## 4. Timing checks

```bash
pixi run check-timing                           # latest real session in data/raw
pixi run check-timing data/raw/<session>
```

Sessions recorded with `--simulate` are skipped. Per trial:

| Column | Meaning |
|---|---|
| EMG Hz / ppm | EMG sample rate measured on the computer's clock |
| dev ppm | EMG sample rate on the armband's own clock. Consistent clocks give ppm ≈ dev ppm − drift |
| drop | Lost packets |
| stalls | Times no data arrived for >0.1 s, with total time (e.g. the computer slept). Rates are then fitted per continuous stretch |
| jit99 | USB arrival jitter |
| USB ms | Armband → computer delay, with its p5–p95 range, after removing the drift trend |
| drift | Clock drift between armband and computer |
| onsets | EMG onsets detected / prompts |
| lag ms | EMG onset minus the frame that drew each prompt at the target |
| paint ms | Scheduled → drawn delay |

**Testing alignment end to end:**

1. Run a sync block with the corner flash:
   `pixi run prompter --port … --sync-patch protocols/sync/sync_single_b01.json`.
   Press sharply as each capsule reaches the target.
2. Run `pixi run check-timing`.
3. Measure the display's own delay once per computer and monitor: film the corner square
   (top right, white for 100 ms at each prompt onset) at 240 fps, or tape a photodiode to it.
   Compare against the logged `sync_flash_paint_ns`.

**Long-term clock drift:** record one long continuous trial, then check it:
`pixi run acquire --port … --label drift-test` (`s`, wait 15 min, `e`), then
`pixi run check-timing`. Expect `stalls` = 0.

The lag includes the subject's anticipation: EMG typically starts 100–150 ms *before* the
prompt reaches the target.

### Offline EMG filtering

Raw recordings (the default) carry a few-mV DC offset, drift and mains hum. The standard
offline filter is a zero-phase 4th-order Butterworth high-pass at 20 Hz and 2 Hz wide notches
at 60 Hz and every harmonic. It is applied to each continuous segment separately, so there is
no need to subtract the mean first. It is the live stream's default filter, run zero-phase
instead of causally:

```bash
pixi run filter-emg data/raw/<session>                    # 20 Hz high-pass + 60, 120, 180, ... Hz
pixi run filter-emg data/raw/<session> --harmonics none   # 20 Hz high-pass + 60 Hz only
pixi run filter-emg data/raw/<session> --harmonics odd    # 60, 180, 300, 420, ... Hz
pixi run filter-emg data/raw/<session> --harmonics 3,5,9  # 60, 180, 300, 540 Hz only
```

Mains harmonics are strongest when a monitor or other mains device is attached to the
computer: an HDMI monitor added 300 and 540 Hz lines. `--causal` filters forward only, as a
live pipeline would.

In code:
- `deftemg_acquisition.analysis.filtering.filter_emg(x_uv, 2000, FilterConfig(...), segments)`;
- or `filter_trial(load_trial(npz), FilterConfig(...))`.

`FilterConfig()` alone is the 20 Hz high-pass and the 60 Hz notch, without harmonics.
`EmgFilter()` (high-pass 20 Hz, `comb60all`) is the standard filter.

## 5. Replay

```bash
pixi run replay                                    # first trial of the latest session
pixi run replay data/raw/<session> --trial 2
pixi run replay data/raw/<session>/trial_001_<label>.npz --speed 0.5 --window 10
```

One timeline, played back together:
- **Prompter:** as the subject saw it (prompter trials only), with the effort bar for blocks
  recorded with `--effort-meter`.
- **EMG view:** traces, spectrum and IMU exactly as in the live viewer.
  - Replay opens at the start of the trial and plays, with new data entering at the right
    edge (`--paused` opens it paused).
  - When a press starts, a line in the digit's colour marks it, labelled e.g. `index H` or
    `thumb+pinky L/H`, and scrolls left with the EMG.
- **Band RMS:** as in the live viewer, for the arm recorded with the session (or `--arm`).
- **Camera panel:** empty until camera recording is added. To add it, implement `VideoSource`
  and return it from `find_video_sources` in `src/deftemg_acquisition/viz/replay.py`.

Timing:
- **Prompt lines** use the time each prompt was drawn at the target, falling back to the
  schedule for older recordings.
- **EMG times** come from `check-timing`'s `emg_time_ns` if it has been run, otherwise they
  are fitted on load.
- Recordings made with the earlier three force levels still replay (shown as L/M/H).

| Key | Action |
|---|---|
| Space | Play / pause |
| ← / → | Back / forward 1 s (Shift: 5 s) |
| , / . | Previous / next prompt |
| Home / End | Start / end |
| Esc | Quit |

The slider scrubs, and the speed menu goes from 0.1× to 4×. Window, spacing, offset and
spectrum options work as in the live viewer.

## 6. Using recordings in Python

The commands above run from a clone with pixi. Installed as a package (e.g. `pip install
git+<repo URL>`), it provides the `deftemg_acquisition` library, and the commands as `python -m
deftemg_acquisition.cli.<name>` (`viewer` and `replay`: `deftemg_acquisition.viz.viewer` and
`.viz.replay`). `deftemg_acquisition.data` loads sessions without Qt:

```python
from pathlib import Path

from deftemg_acquisition.analysis.filtering import EmgFilter
from deftemg_acquisition.data.calibration import session_calibration
from deftemg_acquisition.data.labels import LabelSpec, digit_targets
from deftemg_acquisition.data.trials import load_trial, scan_trials, select_trials

session = Path("examples/S01")
usable, skipped = select_trials(scan_trials([session]))  # skipped: (id, reason)
trial = load_trial(usable[0].npz_path)
trial.emg_uv, trial.emg_t_ns  # (n, 18) µV, and one fitted host time per sample
trial.segment_bounds()  # continuous stretches (split at stalls)
y = digit_targets(trial.emg_t_ns, trial.prompts, LabelSpec())  # (n, 5): 0 none, 1 low, 2 high
cal = session_calibration(session, emg_filter=EmgFilter())
```

- **Labels** (`data.labels`): a digit counts as pressed from 50 ms before the drawn prompt
  onset to 300 ms after its offset. This is the measured EMG lead and lag (in sync trials the
  EMG starts ~150–200 ms before the drawn onset and ends ~150–280 ms after the offset), plus
  a 100 ms delay, which gives a causal model some of the press before it has to report it.
  - `digit_targets(..., margin=True)` marks targets within 100 ms of a label edge as ignored
    (−100).
  - `aligned_prompts` can replace the fixed shifts with each trial's measured lags (`Alignment
    ("emg_trial_median")`) or each press's own measured edges (`"emg"`). It can also flag
    presses with no detectable EMG (`mask_undetected`), or build null controls (`null:
    permute / shift`).
- **Press timing** (`data.press_times`): EMG onset and offset of each prompted press, detected
  against the trial's rest noise.
- **Synthetic sessions** (`data.synthetic`, `make-synthetic`) are written in the recorded
  format, with known press timing and strength, for testing analysis code.

## 7. Extending the tools

Other packages can add to the tools without copying them, e.g. to show a model's output live:

- `tasks.prompter.PrompterOverlay`: draws in the prompter's lanes, under or over the prompts
  (`PrompterWidget.overlays`). `PrompterWidget.on_paint` is called after each frame is
  painted.
- `tasks.app.TaskPlugin`: hooks in the task window (`TaskWindow(plugins=...)`): text between
  blocks, metadata and arrays saved with each block, a call each frame tick, when a block ends
  and when the window closes. A plugin can't stop a recording: a hook that raises is logged
  and skipped.
- `viz.viewer.ViewerPanel` and `viz.replay.ReplayPanel`: more panels in the live viewer (under
  the band RMS) and in replay (under the EMG traces, with a caption under the prompter and a
  status in the transport bar). `parser()` and `run()` let a command add its own options.
- `cli.prompter.PrompterExtension`: more prompter options, task window plugins and live viewer
  options (`prompter.main(extension=...)`).

## Example data

`examples/S01` is one small synthetic session: a calibration block and two task blocks of
single-digit presses and two-digit chords. It was made with
`pixi run make-synthetic --out examples --sessions 1 --blocks 2 --per-block 8 --strength 2 --seed 1`.
Its EMG is simulated, not recorded, so it is only for trying the tools. Synthetic sessions
have no raw `.bin` files, though `trials.jsonl` names them.

```bash
pixi run replay examples/S01
pixi run check-timing examples/S01
pixi run filter-emg examples/S01        # writes data/processed/S01
```

## Data layout

Each session is a folder `data/raw/<YYYYMMDD-HHMMSS>/` (git-ignored):

| File | Contents |
|---|---|
| `session.json` | Port, confirmed filter settings, streaming mode, subject, donning (id, start, session within it, previous session), hand |
| `trials.jsonl` | One line per trial: label, times, packet counts, drops, file paths, metadata (for prompter trials: the full sequence, task set, hand, completed flag) |
| `trial_NNN_<label>.bin` | Exact bytes received from the armband (lossless backup) |
| `trial_NNN_<label>.npz` | Decoded data (below) |
| `trial_NNN_<label>_timing.npz`, `timing.json` | Written by `check-timing` |

`pixi run filter-emg` writes `data/processed/<session>/trial_NNN_<label>.npz` (`emg_uv`
filtered, `emg_t_ns`, `segment`, `filter` settings as JSON) and `filter.json`.

Key arrays in the trial `.npz`:

- **EMG:** `emg_uv` (n, 18) µV, `emg_counts`, `emg_seq`, `emg_host_time_ns` (computer time
  when each USB chunk arrived).
- **IMU:** `gyro_rad_s`, `acc_m_s2`, `temperature_c`, `battery_mv`, `battery_pct`,
  `device_time_ms` (armband clock, set from the computer at trial start), `imu_host_time_ns`.
- **Prompts** (prompter trials):
  - `prompt_onset_host_ns`, `prompt_offset_host_ns`: scheduled times.
  - `prompt_onset_paint_ns`, `prompt_offset_paint_ns`: when each prompt was actually drawn
    reaching the target / gone.
  - `prompt_digits` (n, 5) and `prompt_level` (n, 5; −1 = digit not pressed), coded by
    `digit_names` / `level_names` (0 = low, 1 = high).
  - `scroll_start_host_ns`.

**Clocks.**
- Every timestamp taken on the computer (`*_host_*`, `*_paint_ns`, `host_start_ns`, …) uses
  its monotonic clock, which counts steadily and is never adjusted. The wall clock isn't used
  for these because the operating system corrects it at any time; one such correction
  shifted a recording by ~10 ms over ~20 s.
- To convert to wall-clock (Unix) time, use the anchors saved with each trial: `wall = host +
  (anchor_wall_ns − anchor_host_ns)`. The same anchors are in `anchor_start` / `anchor_end` in
  `trials.jsonl` and `clock_anchor` in `session.json`; helper: `deftemg_acquisition.clock`.
- The armband's `device_time_ms` is set from the wall clock at trial start.

For analysis, use `emg_time_ns` from `_timing.npz`: one smooth time per sample fitted over the
trial. It reflects arrival time; subtract `usb_delay_ms_median` (~1–3 ms) for the time the
sample was taken.

## Armband notes (observed on hardware, differs from the manual)

- **Unknown commands:** the reply is `Unknown Command` (with a space). A stopped armband also
  answers `StopDataAcq` this way, slowly, and silently drops commands sent before that
  reply. The code waits for every reply before sending the next command.
- **`TIME…` while streaming:** applied, but not echoed (it is echoed while stopped).
- **IMU rate:** about 217 packets/s, not the 208 Hz in the manual. Use `device_time_ms` /
  `imu_host_time_ns` rather than assuming a fixed rate.
- **Temperature:** 0.1 °C per count (the manual's table says 0.01).
- **Clocks:** the armband's and the computer's clocks drift apart, so time EMG with
  `emg_time_ns`, not by counting samples against `device_time_ms`.
  - The armband's ms clock runs about 8 ppm fast against a Mac (≈0.5 ms per minute). It was
    steady over a 10-minute test, with no temperature effect.
  - EMG sampling runs about 10 ppm slow against the armband's own clock. They are separate
    oscillators.
- **Host sleep:** while the computer sleeps the armband sends nothing. Its sequence counter
  pauses and its clock keeps running, and a burst of packets is lost on resume.
- **Filters:** settings reset when the armband restarts. The software sets and verifies them
  on every start-up.

## Layout

- `src/deftemg_acquisition/acquisition/`: serial protocol, armband driver, recorder, session,
  subject / donning ids, LSL streaming, simulator, recordings replayed as the armband.
- `src/deftemg_acquisition/viz/`: live viewer, replay, the plot panels they share, and window
  placement.
- `src/deftemg_acquisition/tasks/`: prompt sequences, prompter drawing, task window, effort
  meter.
- `src/deftemg_acquisition/analysis/`: timing analysis, EMG filtering.
- `src/deftemg_acquisition/data/`: trial loading, labels, press timing, calibration,
  synthetic sessions.
- `src/deftemg_acquisition/cli/`: command-line entry points (`acquire`, `prompter`,
  `make_sequences`, `check_timing`, `filter_emg`, `make_synthetic`).
- `protocols/`: prompt sequence files.
- `examples/`: a synthetic example session.
- `tests/`: pytest tests (no hardware needed).
- `data/raw`, `data/processed`, `data/synthetic`: local data (git-ignored).

## License

Copyright (c) 2026 Memories.ai. MIT License; see [LICENSE](LICENSE).
