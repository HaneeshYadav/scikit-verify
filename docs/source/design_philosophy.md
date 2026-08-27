# Design Philosophy

`scikit-verify` bridges the gap between concrete Python execution and symbolic verification. 

## The Exact-or-Refuse Contract
If an operation has no faithful symbolic form, `scikit-verify` raises an error instead of guessing. The goal is to avoid silent mistranslations and maintain strict mathematical correctness.

## Lineage and Foundations
Our stance relies heavily on classic computer science foundations:
* **Concolic Style Execution**: Pairing concrete execution with symbolic pathways (King, 1976; Cadar & Sen).
* **Result Checking**: Verifying a routine's outcome against its core defining equations (Blum & Kannan, 1989).
* **Verified Lifting**: Translating operations upwards for correctness assurances.
