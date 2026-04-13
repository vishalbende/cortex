---
paths:
  - "src/api/**/*.py"
  - "**/api/**/*.py"
---

# API Design Rules

- All endpoints must validate input with Pydantic schemas
- Return shape: { "data": T } | { "error": str }
- Rate limit all public endpoints
- Log all requests with correlation IDs
