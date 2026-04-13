"""
Legacy entry point — redirects to cortex.cli.

Prefer using the `cortex` command directly:
  cortex run "Design a login component"
  cortex interactive
  cortex --help
"""

from cortex.cli import app as main  # noqa: F401

if __name__ == "__main__":
    main()
