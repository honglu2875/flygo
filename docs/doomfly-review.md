# Doomfly wiring and the differences from FlyGo

Source review on 2026-09-14, pinned to
[nftechie/doomfly, 71ecf53](https://github.com/nftechie/doomfly/tree/71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33).
The current README selects experimental v6 learning; historical baseline-only
statements in older reports are not descriptions of that complete model.
This was a code/report review, not an independent execution of Doomfly.

| Boundary | Doomfly | Current FlyGo embedding pilot |
|---|---|---|
| Graph admission | 166,700 neuronal entries; 25,582,938 edges, including single-contact pairs | 165,122 Traced entries; 15,270,273 pairs with at least two contacts |
| Initial fast strength | Signed contact count × .275, in model current units | Signed log1p(count), normalized by total incoming magnitude |
| Visual geometry | Per-eye normalized hex coordinates projected into overlapping screen viewports | Common engineered spherical chart with local sampling; optical registration pending |
| History | Continuous frame sequence and persistent neural state | Four historical boards, left lags 0/2 and right 1/3; reset recurrent state per position |
| Input selection | Annotated R1–R6 and R8 types; no zero-in-degree criterion | Qualified inferred photoreceptors |
| Decoder | Four selected descending neurons drive three fixed controls | 2,129 individual candidates feed a representation loss and ordinal score |
| Learning | Local plasticity on 4,184 existing KC→MBON11 edges | Gradients through the recurrent graph and Adam, with declared frozen groups |

Graph policies are explicit in [their importer](https://github.com/nftechie/doomfly/blob/71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33/doom/connectome.py)
and [our importer](../scripts/prepare_graph.py). This is not an identical-graph
performance comparison. No FlyGo topology change is proposed by this review.

For R1–R6, Doomfly assigns the strongest contact-weighted L1/L2/L3 hex column.
The left viewport covers horizontal screen coordinates 0–.6, and the mirrored
right viewport .4–1. These are inferred display coordinates, not registered
optical viewing directions. [Projection code](https://github.com/nftechie/doomfly/blob/71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33/doom/prepare.py).

The newer adapter separately stimulates 330 R8p and 481 R8y cells using blue
and green display channels. It changes the sign of 390 existing R8→aMe12
connections while preserving contact magnitudes. These remain explicit
physiological and spectral assumptions. [Adapter](https://github.com/nftechie/doomfly/blob/71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33/doom_learning_v6/visual.py).
R8 cotransmission has experimental support, but that does not validate every
transferred subtype/sign assignment in MaleCNS. [Xiao et al.](https://www.nature.com/articles/s41586-023-06681-6).

The controller smooths spike rates for 100 ms. DNp20 right-minus-left activity
controls turning; the DNpe017 pair controls forward motion and firing. These
were selected after response observations. There is no fitted correlation
partition in this decoder. [Controller](https://github.com/nftechie/doomfly/blob/71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33/doom/engine.py).

Current v6 adds KC-specific rest/adaptation, separates annotated modulators
from ordinary fast excitation, and uses dopamine/KC rate traces to modify
existing KC→MBON11 weights. Nonfatal damage supplies a timed PPL101 stimulus;
game state does not directly choose actions. [Current protocol](https://github.com/nftechie/doomfly/blob/71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33/docs/doom-live-training.md),
[memory rule](https://github.com/nftechie/doomfly/blob/71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33/doom_learning_v6/rule.py).

The published v6 pilot has one training replica and two held-out starts.
Learned and shuffled-timing arms die at 3.657 s on both; frozen weights reach
the 8 s cap on one and die at 3.657 s on the other. Erasing memory reproduces
the frozen trace. These limited results show no learned survival advantage.
[Recorded analysis](https://github.com/nftechie/doomfly/blob/71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33/outputs/doom-learning/physiology-v6/survival-pilot/analysis.json).

## Concrete question for our physiology audit

Our graph has **312 R8→aMe12 edges, all initially negative**; 220 originate
from receptors illuminated by the current spherical montage. The count includes
uncertain and dorsal R8 subtypes, so it is not a blanket prescription to flip
312 signs. A subtype-qualified target-specific sign experiment is a candidate
within the existing propagation/physiology gate. It has not been run and is
not an established cause of our weak motor responses.
[Read-only graph audit](results/doomfly-comparison-r8-v1.json).

aMe12 is associated with broad visual input to Kenyon cells; increasing this
route need not preserve the board's spatial detail. Measure intermediate
responses, spatial sensitivity and recovery as well as overall activity.
[Visual-input anatomy](https://www.nature.com/articles/s41467-024-49616-z).

## Deferred input perturbation requested by the user

Keep the fixed graph. Compare a bilateral encoding of the two most recent
moves/states with previous move sequences, captures, komi and other context
attached through nonvisual inputs. The exact move-event versus board-state
representation remains to be specified at registration. Audit zero-in-degree
candidates against sensory annotations and the source graph: the current
two-contact filter can create apparent source nodes. Preserve matched total
information, input scale, exposure budgets, probe positions and frozen controls.
This is a follow-up hypothesis, not a launched run or a change to the completed
spherical pilot. Signal/optimizer calibration remains the active gate.
