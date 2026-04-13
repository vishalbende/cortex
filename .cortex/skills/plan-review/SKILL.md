---
description: Reviews an execution plan for completeness, confidence, and parallelizability
argument-hint: <plan-description>
---

Review the following plan for:

1. **Completeness** — are all steps needed to fulfill the intent present?
2. **Dependencies** — are depends_on edges correct? Can any steps run in parallel?
3. **Confidence** — flag any step with confidence < 0.6
4. **Agent assignment** — is each step assigned to the right agent?
5. **Testability** — can each step's output be verified?

Plan: $ARGUMENTS
