# DJ Mix Point Research Specification

## Summary
Research and validate optimal DJ mix transition points by analyzing real professional DJ sets and documented DJ mixing strategies.

## Acceptance Criteria

### Given: Access to professional DJ sets and mixing resources
**When:** Research is conducted on DJ mix point strategies
**Then:**
- At least 3 professional DJ sets are analyzed for mix point patterns
- Mix-in and mix-out timing patterns are documented with timestamps
- Phrase boundary alignment (8/16/32 bars) is verified
- Key and tempo transition strategies are documented
- Common mix point indicators are identified (energy drops, breakdowns, buildups)

### Given: Professional DJ mixing guides and tutorials
**When:** DJ mixing best practices are researched
**Then:**
- Documented rules for mix-in points (intro length, phrase boundaries)
- Documented rules for mix-out points (outro length, phrase boundaries)
- Energy matching strategies are identified
- Harmonic mixing rules (Camelot wheel, key compatibility) are validated
- Tempo matching tolerances are documented

### Given: Current AutoMix AI implementation
**When:** Implementation is compared against professional DJ strategies
**Then:**
- Current mix point detection algorithm is evaluated against real DJ behavior
- Gaps between current implementation and professional practice are identified
- Recommendations for algorithm improvements are documented
- Validation test cases are created based on real DJ set examples

## Input/Output Examples

### Example 1: DJ Set Analysis
```
DJ Set: Boiler Room - Carl Cox (2023)
Track 1 → Track 2 transition at 4:32
- Track 1 mix-out: 4:22 (32-bar phrase boundary, energy drop)
- Track 2 mix-in: 4:32 (16-bar intro, first kick drum)
- Mix duration: 10 seconds
- Key relationship: Am → Dm (perfect fourth)
- Tempo: 128 BPM → 130 BPM (+2 BPM)
```

### Example 2: Mix Point Pattern Documentation
```
Common Mix-In Points:
- After 8-16 bar intro (most common)
- At first full kick drum entry
- After vocal intro completes
- At phrase boundary (16/32 bars from track start)

Common Mix-Out Points:
- 16-32 bars before track end
- At energy breakdown/drop
- Before final chorus/outro
- At phrase boundary aligned with incoming track
```

### Example 3: Algorithm Validation
```
Current Implementation:
- Mix-in: 5 seconds from start (fixed offset)
- Mix-out: 10 seconds before end (fixed offset)

Professional DJ Behavior:
- Mix-in: 8-32 bars from start (phrase-aligned, varies by track structure)
- Mix-out: 16-64 bars before end (phrase-aligned, energy-aware)

Gap: Current implementation uses fixed time offsets instead of phrase boundaries
Recommendation: Implement phrase detection and align mix points to 16/32 bar boundaries
```

## Research Sources

### DJ Sets to Analyze
1. Boiler Room sets (Carl Cox, Nina Kraviz, Ben UFO)
2. BBC Radio 1 Essential Mix archives
3. Resident Advisor podcast mixes
4. Recorded club sets from professional DJs

### DJ Mixing Resources
1. "How to DJ Right" by Frank Broughton & Bill Brewster
2. Digital DJ Tips tutorials and courses
3. DJ TechTools mixing guides
4. Harmonic mixing guides (Camelot wheel, Mixed In Key)
5. Professional DJ interviews and technique breakdowns

### Technical Resources
1. Mixed In Key documentation (key detection standards)
2. Rekordbox/Serato beat grid documentation
3. Academic papers on music information retrieval (MIR)
4. DJ software documentation (Pioneer, Native Instruments)

## Deliverables

1. **Research Document** (`.agent/dj-mix-research.md`):
   - Analysis of 3+ professional DJ sets with timestamps
   - Documented mix point patterns and strategies
   - Key/tempo transition rules from professional practice
   - Common phrase lengths and structural patterns

2. **Validation Report** (`.agent/mix-point-validation.md`):
   - Comparison of current implementation vs. professional behavior
   - Identified gaps and improvement opportunities
   - Prioritized recommendations with rationale
   - Test cases based on real DJ examples

3. **Updated Memories** (`.agent/memories.md`):
   - Key learnings about DJ mixing strategies
   - Phrase boundary importance and detection methods
   - Energy-aware mix point selection
   - Professional DJ timing patterns

## Edge Cases

1. **Tracks without clear phrase structure**: Document how DJs handle experimental/ambient tracks
2. **Live recordings vs. studio tracks**: Note differences in mix point selection
3. **Genre-specific patterns**: Document variations across house, techno, drum & bass, etc.
4. **Quick cuts vs. long blends**: Document when DJs use different mix durations

## Non-Functional Requirements

- **Time**: Research should be completable within 2-3 hours
- **Depth**: Focus on actionable patterns, not exhaustive analysis
- **Practicality**: Prioritize findings that can improve current implementation
- **Evidence**: All claims should be backed by specific examples with timestamps

## Out of Scope

- Building a complete DJ mixing algorithm from scratch
- Real-time mixing or audio playback implementation
- Analyzing every possible DJ technique or style
- Creating a comprehensive DJ mixing tutorial
- Implementing automatic mixing/crossfading
