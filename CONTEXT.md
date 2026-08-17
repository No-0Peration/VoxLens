# VoxLens

VoxLens reads speech from the visible movement of a speaker's mouth, without using audio. This glossary fixes the vocabulary for that process — what goes in, what comes out, and how the system talks about the parts it cannot see.

## Language

### Input

**Clip**:
A bounded video recording with a known start and end, processed as a complete unit.
_Avoid_: video, footage, file, recording

**Stream**:
An unbounded sequence of frames arriving live, with no known end.
_Avoid_: feed, live video, camera

**Frame**:
A single image drawn from a Clip or a Stream.
_Avoid_: image, still, picture

**Speaker**:
The person whose mouth is being read. Exactly one per Clip or Stream.
_Avoid_: subject, user, talker, person

**Mouth Region**:
The area within a Frame that contains the Speaker's mouth, and the only part of the Frame that carries speech information.
_Avoid_: ROI, crop, bounding box, face

### Visibility

**Occlusion**:
A run of consecutive Frames in which the Mouth Region cannot be read — turned away, obstructed, out of focus, or outside the Frame. The cause is irrelevant; the unreadability is what matters.
_Avoid_: gap, dropout, blockage, missing frames, blind spot

### Output

**Transcript**:
The complete text VoxLens produces for one Clip, or for one continuous stretch of a Stream. A Transcript remains a single unit even when Occlusion interrupts it.
_Avoid_: output, result, caption, subtitle

**Read Text**:
Text grounded in visible mouth movement — the Speaker was legible when these words were spoken.
_Avoid_: actual, real, observed, confident

**Inferred Text**:
Text VoxLens reconstructs across an Occlusion from surrounding context rather than from visual evidence. Always presented as distinct from Read Text, so a reader can tell what was seen from what was filled in.
_Avoid_: predicted, guessed, filled, hallucinated, assumed

> Every word VoxLens emits is a model prediction, including Read Text. The distinction that matters is **evidence**, not certainty — which is why "predicted" is not the word for the inferred kind.
