# Read on screen, never aloud — and not sold as an accessibility aid

Live capture displays text. It never speaks the transcript with a synthetic voice, and it is not presented to deaf or hard-of-hearing users as an assistive tool.

At 48% word errors on real footage, roughly every other word is wrong, and the errors are fluent: *"of those seven have nuclear weapons"* was read as *"if you use seven if you use a weapon"*. Read aloud in a confident voice, that removes the only signal warning the listener — text on a screen can be doubted, a spoken sentence is simply believed.

## Consequences

- **The obvious demo is off the table.** Pointing a phone at someone and hearing their words spoken is a far better demo than watching text appear. It is also the version that puts invented words in someone's mouth in front of other people.
- **The clearest use case is explicitly not claimed.** A deaf person following a seminar speaker is the use that comes to mind first — and it is one this loses badly. The audio exists; the user simply cannot hear it. A phone microphone with iOS Live Captions transcribes it at 5–10% word errors against this pipeline's 48%.
- **The gap is where a real use would live**, and none is named yet: audio absent or unusable — behind glass, beyond microphone range but within sight, a muted call, extreme noise. Until such a scenario is named, this is a technical exercise, which is a legitimate thing to build and a different thing to claim.

## Considered options

**Speak the transcript.** Rejected on the error profile, not on principle. If accuracy reached a level where fluent errors were rare, this would be worth revisiting — and the same argument would need making again with the numbers of the day.

**Speak only high-confidence sentences.** Plausible later, but it depends on a confidence measure that is not yet validated ([ADR-0011](0011-confidence-from-decoder-disagreement.md)).
