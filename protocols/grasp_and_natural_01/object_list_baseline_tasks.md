# Object list: baseline grasp and natural tasks

These are the objects for the set in `protocols/grasp_and_natural_01/`, made from
`protocols/grasp_natural/catalogue.toml` with seed 1495099208. The set has 9 sessions of about
25 to 29 min each: 590 grasp prompts and 31 natural-task trials, about 4 h in total.
`session_objects.txt` lists what to set out for each session.

## Task variations

### Grasp tasks

Each grasp prompt names an object, an approach, a grasp type and a variation. The approach is
"from the side" or "from the top" for most objects, and "off the edge of the table" or "flat"
for the book.

**Grasp types:** the fingertips to use are highlighted on the hand on screen.

| Grasp | Fingers |
|---|---|
| Thumb-index (TI) | thumb and index fingertips |
| Thumb-index-middle (TIM) | thumb, index and middle fingertips |
| All fingertips (AD) | all five fingertips |
| Power | the whole hand wrapped around the object |

**Variations:**

| Variation | What the subject does | Est. time |
|---|---|---|
| Light grasp | **Stay on table.** Touch the object lightly. | 4 s |
| Grasp and squeeze | **Stay on table.** Grasp and squeeze at less than maximum force. | 6 s |
| Pick up and place | **Pick up.** Place it about 30 cm away. | 8 s |
| Pick up, squeeze, place | **Pick up.** Squeeze while moving, then place it about 30 cm away. | 9 s |
| Rotate slowly | **Pick up.** Rotate slowly in all directions in the air, then put it back. | 12 s |
| Rotate quickly | **Pick up.** Rotate quickly in all directions in the air, then put it back. | 8 s |

Each object gets every allowed combination of approach, grasp and variation once. The two
rotations are done only once per grasp type, on the first approach that offers that grasp.

### Natural tasks

These are everyday tasks, written out one per prompt, and the subject does them as written.

| Group | Task | Trials |
|---|---|---|
| Pages | Open the book, then turn pages for 20 s | 1 |
| Containers | Open the Aquaphor cap; open, then close the cream jar's turn-lid; drink from each mug using the handle | 5 |
| Screw panel | Loosen each fastener all the way off, then tighten it again | 16 |
| USB | Insert, then remove each plug | 6 |
| Grip tools | Squeeze each grip-strength tool once | 3 |

### Grasp calibration

Grasp calibration comes at the start of every session and again about 10 min in, then every
20 min. It consists of one brief maximum grasp of each kind, held for 1.5 s:
- thumb-index on the hardcover book
- thumb-index-middle on the hardcover book
- all fingertips on the hardcover book
- power on the ball

## Objects

### Calibration (every session)

| Object | Notes |
|---|---|
| Hardcover book | Thumb-index, thumb-index-middle and all-fingertips maximum grasps |
| Ball | Power maximum grasp |

### Grasp objects

| Object | Session | Approaches | Grasps | Variations | Prompts | Notes |
|---|---|---|---|---|---|---|
| Tennis ball | S01 | one | all 4 | all 6 | 24 | |
| Baseball | S09 | one | all 4 | all 6 | 24 | |
| Lacrosse ball | S08 | one | all 4 | all 6 | 24 | |
| Water bottle (rubber) | S01 | side, top | all 4 | all 6 | 40 | Filled or empty: asked at session start |
| Water bottle (metal/plastic) | S02 | side, top | TIM, AD, Power | all 6 | 30 | Filled or empty: asked at session start |
| Tuna can | S06 | side, top | side: TI, TIM; top: all 4 | all 6 | 32 | |
| Tomato/soup can | S02 | side, top | all 4 | all 6 | 40 | |
| Dish soap | S09 | side, top | all 4 | all 6 | 40 | |
| Windex bottle | S06 | side, top | all 4 | all 6 | 40 | Grasp near the top of the stem |
| Peanut butter jar | S07 | side, top | all 4 | all 6 | 40 | |
| Apple | S08 | side, top | all 4 | all 6 | 40 | |
| Orange | S04 | side, top | all 4 | all 6 | 40 | |
| Box of crackers | S03 | side, top | all 4 | all 6 | 40 | |
| Bag of chips | S05 | one | all 4 | all 6 | 24 | |
| Potato chip | S09 | side, top | TI, TIM | no rotations | 16 | |
| Hammer | S08 | one | AD, Power | no rotations | 8 | |
| Small coffee mug | S03 | top (on the lip), side (around the body) | top: TI, TIM; side: AD, Power | all 6 | 24 | Also used in the S01 drinking task |
| Large coffee mug | S07 | top (on the lip), side (around the body) | top: TI, TIM; side: AD, Power | all 6 | 24 | Also used in the S01 drinking task |
| Book | S05 | off the edge of the table, flat | all 4 | all 6 | 40 | Also used in the S03 page-turning task |

### Natural-task objects

| Object | Session | Task |
|---|---|---|
| Book | S03 | Open it, then turn pages for 20 s |
| Aquaphor tube | S01 | Open the cap |
| Medium cream jar | S01 | Open, then close the turn-lid |
| Small coffee mug | S01 | Drink from it using the handle |
| Large coffee mug | S01 | Drink from it using the handle |
| Screw panel | S04 | Loosen, then tighten each fastener: wide Phillips screw, narrow Phillips screw, eye bolt, hand knob, Allen screw, hex bolt (with the ratchet), hex bolt (with the wrench) and wing nut |
| Phillips screwdriver | S04 | For the two Phillips screws |
| Allen key | S04 | For the Allen screw |
| Ratchet | S04 | For the hex bolt |
| Wrench | S04 | For the hex bolt |
| USB-C plug | S05 | Insert into its port, then remove; also insert into the A-to-C adapter, then remove |
| USB-A plug | S05 | Insert into its port, then remove |
| A-to-C adapter | S05 | Takes the USB-C plug |
| USB ports | S05 | Ports for the USB-C and USB-A plugs |
| Light grip-strength tool | S07 | Squeeze once |
| Medium grip-strength tool | S07 | Squeeze once |
| Heavy grip-strength tool | S07 | Squeeze once |

## Checklist

There are 35 distinct items: 2 calibration objects, 19 grasp objects and 14 natural-task
items. The book and both mugs appear in both tables. Session conditions to decide at each
session start: whether the rubber bottle and the metal/plastic bottle are filled.

- [ ] Hardcover book (calibration)
- [ ] Ball (calibration)
- [ ] Tennis ball
- [ ] Baseball
- [ ] Lacrosse ball
- [ ] Water bottle (rubber)
- [ ] Water bottle (metal/plastic)
- [ ] Tuna can
- [ ] Tomato/soup can
- [ ] Dish soap
- [ ] Windex bottle
- [ ] Peanut butter jar
- [ ] Apple
- [ ] Orange
- [ ] Box of crackers
- [ ] Bag of chips
- [ ] Potato chip(s)
- [ ] Hammer
- [ ] Small coffee mug
- [ ] Large coffee mug
- [ ] Book
- [ ] Aquaphor tube
- [ ] Medium cream jar (turn-lid)
- [ ] Screw panel with the 8 fasteners above
- [ ] Phillips screwdriver
- [ ] Allen key
- [ ] Ratchet
- [ ] Wrench
- [ ] USB-C plug
- [ ] USB-A plug
- [ ] A-to-C adapter
- [ ] USB ports
- [ ] Light grip-strength tool
- [ ] Medium grip-strength tool
- [ ] Heavy grip-strength tool
