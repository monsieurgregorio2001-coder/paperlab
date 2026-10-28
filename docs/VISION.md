# PaperLab — vision

**One line.** A playground where you can see models think: from a line through a cloud of points to a real open-weight language model running in your browser.

**Visualizer first, educational second, accurate always.** Every lab is something you touch: drag points, paint classes, type a sentence, scrub a slider, and watch the maths respond in real time, in 3D where 3D earns it. The explanation sits beside the picture, never in front of it. Nothing on screen is faked: toy models say so in-scene, trained models run real weights, and the frontier lab instruments a real model's attention.

## The ladder

Ten labs, each a self-contained room with its own colour, one big interaction, a tour, and a "see the maths" drawer.

| # | Lab | You do | You see | Maths runs where |
| --- | --- | --- | --- | --- |
| 1 | **Line** — linear regression | Drag points on a scatter; choose closed-form or gradient descent; set the learning rate | The line chases the points; a 3D loss bowl with a ball rolling to the minimum | TS, exact |
| 2 | **Boundary** — logistic regression / perceptron | Paint two classes | A sigmoid sheet in 3D tilts and bends; the decision line swings into place | TS, exact |
| 3 | **Net** — small MLP | Paint data (spiral, moons, your own); add neurons and layers; play | A decision surface as 3D terrain morphing during training; neurons as glowing nodes with live weight edges | TS, exact (TF Playground lineage) |
| 4 | **Space** — embeddings | Search a word; drag to compare; type an analogy | A 3D cloud of real word vectors; analogies as arrows (king − man + woman) | Real pretrained vectors (small public set), PCA in TS |
| 5 | **Sight** — CNN | Draw a digit | Filters slide across your drawing; activation maps stack up in 3D; the prediction bars race | Real tiny CNN trained offline, run in-browser (ONNX) |
| 6 | **Memory** — RNN / LSTM | Type a sequence | The hidden state as a glowing cell that carries information forward; gates open and close | TS, exact, on a tiny trained net |
| 7 | **Attention** | Type a sentence; toggle heads; drop the causal curtain | Tokens on a ring; attention as arcs whose thickness is the weight; one colour per head | TS, exact (existing tested kernel) |
| 8 | **Block** — transformer | Type and step | Tokens flow through a 3D stack of blocks; residual stream as a river; next-token bars race | TS, exact (toy weights, labelled) |
| 9 | **Noise** — diffusion | Scrub a timeline | An image emerges from noise step by step; the predicted-noise field shown as a vector field | Real tiny model or an honestly-labelled precomputed trace |
| 10 | **Frontier** — a real open-weight LLM | Type a prompt; watch it think | Qwen3-0.6B runs live on your GPU; its 28 layers × 16 heads light up as it generates; `<think>` tokens shown as thinking. Beside it, DeepSeek-V3's architecture to scale: MLA squeezing 7168 to 512, the 256-expert router picking 8 | Real model via transformers.js + WebGPU; static diagram from published configs |

Labs 1–3 teach "fitting"; 4–6 teach "representations"; 7–8 teach "attention"; 9–10 show the frontier.

## Principles

1. **Touch before text.** The first thing in every lab is the thing you can move.
2. **Real maths, labelled toys.** Exact kernels in TypeScript with tests against reference values; trained weights where a toy would mislead; a visible "toy weights" badge whenever weights are synthetic.
3. **3D when it shows a shape that 2D cannot.** Loss bowls, sigmoid sheets, decision terrains, block stacks, embedding clouds. Never 3D for decoration.
4. **60 fps on a laptop, usable on a phone.** WebGPU with WebGL fallback; pixel-ratio caps; every lab has a reduced-motion and a no-WebGL state that still teaches.
5. **One voice.** Short captions, first person where it is the author speaking, plain words, numbers in mono.
6. **Credit the lineage.** TensorFlow Playground (Apache 2.0), Bycroft's LLM Visualization (MIT), Polo Club explainers (MIT) are cited in the About; borrowed code keeps its licence header.
