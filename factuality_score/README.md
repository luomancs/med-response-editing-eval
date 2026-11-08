# MedFactChecking Score

_Code coming soon._

This folder will contain the MedFactChecking Score. The score measures how well an
edited response *d′* keeps the medical facts of the original physician response *d*.
It is adapted from FactEHR:

- Both responses are broken down into atomic facts, *C* (from *d*) and *C′* (from *d′*).
- **Fact-Recall** = (1/|C|) Σ_{c∈C} 𝟙[d′ ⊨ c]: the share of original facts that the
  edited response still entails.
- **Fact-Precision** = (1/|C′|) Σ_{c′∈C′} 𝟙[d ⊨ c′]: the share of facts in the edited
  response that the original supports. Facts that are not supported were added by
  the edit.
- Micro and macro averages are reported.

The input uses the same CSV format as [`../data/README.md`](../data/README.md), so it
can be run on the outputs of `../empathy_score/generate_responses.py`.
