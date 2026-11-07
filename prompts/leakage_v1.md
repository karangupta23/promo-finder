You are checking a promo clip for PAYOFF LEAKAGE. This is a separate editorial check that happens after the clip was already selected; you are allowed to see what comes right after the clip.

You receive:
- CLIP: the screenplay blocks of the promo exactly as it would be posted (+seconds from clip start).
- AFTER: the blocks that immediately follow the clip in the show (they are NOT in the promo).

Answer:
1. answer_revealed_in_clip - Does the clip itself already answer the question it ends on (so there is no real cliffhanger)? Cite clip block ids.
2. resolved_immediately_after - Does the AFTER material resolve the cliffhanger almost immediately (within a few seconds)? That is fine for the show, but tells you whether the question is strong or trivial. Estimate seconds_until_resolution if resolved.
3. stronger_earlier_end - Would ending the clip slightly EARLIER (on a block inside the clip, keeping it at least 20 s long) create a stronger unresolved moment? Give end_block_id or null.
4. recommendation: KEEP (fine as cut), TRIM (end earlier at end_block_id), or DROP (no real cliffhanger / it gives the payoff away).
Only cite block ids that appear in the input. Be concise.
