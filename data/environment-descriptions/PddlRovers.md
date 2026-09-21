# Rovers

Two turtlebot rovers explore a 5m square arena. They must acquire **one stone sample and one soil sample**, **photograph every objective**, and **radio all of it to the Husky lander**, finishing with **both rovers back where they started** and **both sample stores empty**. This is PDDLStream's `rovers` benchmark.

Mounds, pillars and a wall down the middle of the arena block both driving and line of sight, so the difficulty is *where you have to stand*: within 2m of an objective with nothing in the way to photograph it, and within 4m of the lander with nothing in the way to radio anything at all.

## Variant

A variable number of objectives, standing on mounds; 3 stone samples and 3 soil samples scattered on the floor; 8 pillars. Objectives are reassigned to mounds every episode, so which ones are occluded changes.

## Action Space

`Box((8,), float32)` -- four entries per rover, rover0 then rover1:

| **Index** | **Description** |
| --- | --- |
| 0, 4 | delta base x (+/-0.2) |
| 1, 5 | delta base y (+/-0.2) |
| 2, 6 | delta base heading (+/-0.4) |
| 3, 7 | operator selector (+/-1) |

The selector splits [-1, 1] into six equal bands. Band *k* spans `-1 + k/3` to `-1 + (k+1)/3`, so its centre is `(k + 0.5) / 3 - 1`, and the bands select, in order: **sample**, **calibrate**, **image**, **noop**, **send**, **drop**. Noop is the band containing zero, so an all-zero action does nothing.

Each operator is refused silently -- the state simply does not change -- unless its precondition holds:

- **sample**: the rover's base is within 0.25m of a sample and its store is empty. Fills the store and records the analysis.
- **calibrate**: some objective is visible. Cameras must be calibrated immediately before each photograph.
- **image**: the camera is calibrated and an objective is visible. Photographs **one** objective -- the nearest one this rover still needs -- and spends the calibration, so N objectives cost N calibrate/image pairs.
- **send**: the lander is visible. Radios everything this rover is holding, images and analyses alike. Only the rover that took a result can send it.
- **drop**: empties the store. The analysis is already recorded, so dropping before or after sending both work.

Both rovers act on every step. Dynamics are kinematic: a motion that would put a rover in collision with a wall, a mound, a pillar, the lander or the other rover is rejected, leaving it where it was, and the operator on that same step still applies.

## Reward

-1 per step, so the return is the negated number of steps and a better policy is a shorter one.


## Variable Object Count

The number of objectives changes between episodes, so observations are object-centric rather than a fixed-length vector: each is an `ObjectCentricState` holding `rover0` and `rover1`, the `lander`, `objective0`..`objectiveN-1`, `sample0`.. (the stone samples first, then the soil ones, told apart by `is_soil`), and `obstacle0`.. for the mounds and pillars.

Only the imaging half of the goal grows with the count: one stone and one soil analysis are required whatever it is.

Feature names per type:

- `rover`: x, y, theta, store_full, calibrated, at_home
- `lander`: x, y, z
- `objective`: x, y, z, have_image_rover0, have_image_rover1, received_image
- `sample`: x, y, z, is_soil, analyzed_rover0, analyzed_rover1, received_analysis
- `obstacle`: x, y, z, half_x, half_y, half_z

Iterate the state's objects rather than indexing fixed offsets, so one program handles any number of objectives.
