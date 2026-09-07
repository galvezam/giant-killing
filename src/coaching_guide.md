# Giant Killings and Team Coordination — what the evidence actually says

**A one-page summary for coaches and analysts.**
Giant Killing Project · August 2026 · [manuscript](manuscript/manuscript.pdf) · [replication package](replication/REPRODUCE.md)

---

## What we did

We took 136 matches with player-tracking data — 30 giant killings and 106 matched
"expected result" matches between similarly mismatched sides — and measured, second by
second, how much of each team's *collective* shape (centre of mass, compactness, how
tightly players cluster together, how similarly they move) predicts that team's own
near future **beyond what the individual players predict on their own**.

That quantity is called *causal emergence*. Positive means the team is behaving
synergistically — the whole genuinely exceeds the sum of the parts. Negative means the
team is redundancy-dominated — players are synchronised and mutually predictable, which
is what a well-drilled defensive block looks like.

## The headline: three honest statements

### 1. There is no "togetherness surge."

The popular theory is that underdogs win by finding a burst of collective inspiration.
We tested that 54 different ways — four measures of team shape, three levels of
interaction (whole team, pairs, midfield triangles), with and without adjusting for the
scoreline — and **found nothing**. A machine-learning model given 260 summary measures
of every match could not tell a giant killing from an expected result any better than a
coin flip.

**What this means for you:** don't build a session, or a team talk, around producing a
measurable spike in collective coordination. Either it doesn't happen, or it is too
small for the best available measurement to see. This is a genuinely useful negative:
it removes a plausible-sounding idea from the table.

### 2. The one signal we did find points the *other* way — toward consolidation.

The single effect that survived our analysis is about **change over the match**, not
about the average. In giant killings, the underdog becomes progressively *more*
synchronised and structured relative to the favourite as the match wears on. In matches
that go to form, the reverse happens — the underdog's structure erodes relative to the
favourite's.

Both teams start indistinguishable. They diverge.

**What this means for you:** the question is not how organised your team is at kick-off.
It's whether that organisation **holds and tightens** through the phase of the match when
fatigue, substitutions and score pressure all work against it. That is a trainable,
measurable, coachable property in a way that "wanting it more" is not.

> **Caveat, stated up front:** this result was found *after* our planned tests came back
> empty. That makes it a hypothesis worth testing, not a fact to build a philosophy on.
> It needs confirming on matches we haven't seen. Treat it accordingly.

### 3. Where you are matters more than how you're moving.

Across every part of this study, the measures that carried any signal at all were about
**spatial configuration** — how close players are to one another, how compact the unit is.
The measure based on **velocity similarity** — players moving at the same speed in the
same direction — consistently showed nothing, or pointed the opposite way.

This matches the only prior study of causal emergence in football (Cheng et al., 2025),
which found the same thing on higher-quality data: positional measures correlated
strongly with possession; velocity-synchrony measures didn't correlate at all.

**What this means for you:** compactness and spacing appear to be where team-level
information lives. Coordinated *running* — everyone pressing at once, everyone dropping
at once — does not appear to be a carrier of collective advantage in its own right. It
may still be necessary; the evidence says it is not sufficient, and it is not what
distinguishes teams.

---

## What we cannot tell you

| Question | Status |
|---|---|
| Is there a tipping point where the underdog "takes over"? | **No.** In most matches the underdog's coordination never overtakes the favourite's at all. There is no crossover to detect. |
| Do underdogs disrupt the favourite's shape? | **Unanswerable with our data.** We would need matches where a favourite loses to an *equal* opponent, to separate "being beaten by an underdog" from "losing." Our corpus has none. |
| Does coordination in pairs or trios matter more than team-wide? | **No difference detected** (p = 0.995). |
| Which specific shape metric predicts an upset? | **None of them,** at this sample size. |

---

## How much weight to put on this

**Moderate, and mostly on the negatives.** Three things limit us.

- **Sample size.** 30 giant killings. We can only reliably detect a fairly large
  difference; a real but modest effect would be missed roughly two times in three.
- **The measurement itself.** We found that this particular coordination measure is
  badly behaved on football data — it is overwhelmingly driven by the arithmetic fact
  that a team's average position *is* the average of its players' positions. Part of the
  reason we found little may be that the instrument is too blunt, not that there is
  nothing there.
- **A data-processing bug we found late.** The match clock we used accidentally
  double-counted the first half, so about 29% of each match was filled with a synthetic
  straight-line "reconstruction" of players who were not actually being tracked. We have
  checked what this changes: the consolidation finding survives almost unchanged when
  those stretches are thrown away, and the negative findings should be unaffected because
  the error hit both teams equally. But the whole analysis is being re-run properly, and
  the numbers may shift slightly when it is.

So: the absence of a togetherness surge is a reasonably solid finding. The consolidation
effect is a promising lead. Neither should change what you do on Monday without
replication.

---

## The one thing to take away

If the consolidation result holds up, the coaching implication is about **trajectory, not
intensity**. An underdog doesn't need a moment of collective magic. It needs its shape to
be *better organised at minute 75 than at minute 15* — against a favourite whose own
structure is drifting the other way. Build the session plan around holding and tightening
structure under accumulating fatigue and score pressure, and you are training the only
thing in this study that separated the teams that pulled off a giant killing from the
teams that didn't.
