---
paths:
  - "**/*test*.py"
  - "tests/**/*.py"
---

# Testing Rules

- Use descriptive test names: "test_should_[expected]_when_[condition]"
- Mock external dependencies, not internal modules
- Clean up side effects in teardown
- Every agent contract must ship with at least 3 test cases:
  happy path, empty/null input, and permission-denied path
- Mocks must be explicit — never rely on real external services in unit tests
- Test file naming: test_<module>_<feature>.py
