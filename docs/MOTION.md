# PaperLab motion

Every animation in the Attention room (`src/attention/room.css`), with purpose, token and Calm behaviour (DIRECTION 3.5). Calm is `data-motion="calm"` on the root: every duration token becomes 160 ms, every spring and ease token becomes `--h-ease-out`, and `--h-press-scale` is neutral. Nothing loops and nothing runs longer than 600 ms. The house layer's own animations are in the hub's `house/MOTION.md`.

| Animation | Purpose | Tokens | Calm |
| --- | --- | --- | --- |
| Shell crossfade | The static hero fades out as the live app fades in, in one grid cell, so layout never moves. | `--h-dur-ui`, `--h-ease-out` | 160 ms fade. |
| Chip, rail button and switch press | Confirms a press. Scales to the press scale and back. | `--h-press-scale`, `--h-dur-quick`, `--h-spring-snappy` | No scale. Colour change only, 160 ms. |
| Chip, rail button and word hover | Shows the target is live. Background colour. | `--h-dur-quick`, `--h-ease-standard` | 160 ms. |
| Attention thread | A thread between words gets thicker or fainter as the weight changes. | `--h-dur-ui`, `--h-spring-snappy` | 160 ms, no spring. |
| Ghost thread | The dashed comparison thread changes opacity and width. | `--h-dur-ui`, `--h-ease-out` | 160 ms. |
| Word weight | The selected word changes font weight. | `--h-dur-ui`, `--h-spring-snappy` | 160 ms, no spring. |
| Switch thumb and track | Shows a Break the math switch changed. | `--h-dur-quick`, `--h-spring-snappy`, `--h-ease-standard` | The thumb still moves to show state, in 160 ms with no spring. |

The First version at `/paperlab/legacy/` keeps its own short transitions (150 to 200 ms, border colour, a stepper rule and a cell fade). They are gated by `prefers-reduced-motion` in `src/legacy/styles.css`, and they are not on house tokens.
