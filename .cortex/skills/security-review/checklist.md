# Security Review Checklist

## Input Validation
- [ ] All user input sanitized before DB queries
- [ ] File upload MIME types validated
- [ ] Path traversal prevented on file operations

## Authentication
- [ ] JWT tokens expire after 24 hours
- [ ] API keys stored in environment variables
- [ ] Passwords hashed with bcrypt or argon2

## Agent Security
- [ ] Intent is sanitized before passing to sub-agents
- [ ] Permission resolver checks all agent actions
- [ ] Destructive actions require explicit user confirmation
- [ ] Mistake records logged for all permission violations
