"""
CortexInit — scaffolds the .cortex/ directory structure in any repo.

Mirrors the Claude Code .claude/ directory spec with Cortex naming:

  your-project/
  ├── CORTEX.md                         # committed — project instructions
  ├── .cortex/
  │   ├── .gitignore                    # ignores local-only files
  │   ├── settings.json                 # committed — permissions, hooks, config
  │   ├── settings.local.json           # gitignored — personal overrides
  │   ├── rules/                        # committed — topic-scoped instructions
  │   │   ├── testing.md                #   scoped to test files
  │   │   ├── agents.md                 #   scoped to agent code
  │   │   └── api-design.md             #   scoped to API code
  │   ├── skills/                       # committed — reusable prompts
  │   │   ├── security-review/
  │   │   │   ├── SKILL.md
  │   │   │   └── checklist.md
  │   │   └── plan-review/
  │   │       └── SKILL.md
  │   ├── commands/                     # committed — single-file commands
  │   │   ├── fix-issue.md
  │   │   └── status.md
  │   ├── agents/                       # committed — subagent definitions
  │   │   ├── code-reviewer.md
  │   │   └── plan-architect.md
  │   ├── agent-memory/                 # committed — subagent persistent memory
  │   │   └── README.md
  │   ├── agent-memory-local/           # gitignored — local-only agent memory
  │   └── output-styles/                # committed — custom output styles
  │       ├── teaching.md
  │       └── concise.md

Usage:
  cortex init                   # scaffold in current directory
  cortex init /path/to/repo     # scaffold in specific directory
  cortex init --minimal         # only CORTEX.md + settings.json
  cortex init --force           # overwrite existing files
"""

from __future__ import annotations

import logging
from pathlib import Path

from cortex.scaffold import templates as T

logger = logging.getLogger(__name__)


# File manifest: (relative_path, content_attr, badge)
# badge: "committed" | "gitignored" | "local"
FILE_MANIFEST: list[tuple[str, str, str]] = [
    # Root
    ("CORTEX.md",                                    "CORTEX_MD",                  "committed"),

    # .cortex/ core
    (".cortex/.gitignore",                           "DOTCORTEX_GITIGNORE",        "committed"),
    (".cortex/settings.json",                        "SETTINGS_JSON",              "committed"),
    (".cortex/settings.local.json",                  "SETTINGS_LOCAL_JSON",        "gitignored"),

    # .cortex/rules/
    (".cortex/rules/testing.md",                     "RULE_TESTING_MD",            "committed"),
    (".cortex/rules/agents.md",                      "RULE_AGENTS_MD",             "committed"),
    (".cortex/rules/api-design.md",                  "RULE_API_MD",               "committed"),

    # .cortex/skills/
    (".cortex/skills/security-review/SKILL.md",      "SKILL_SECURITY_REVIEW_MD",   "committed"),
    (".cortex/skills/security-review/checklist.md",  "SKILL_SECURITY_CHECKLIST_MD","committed"),
    (".cortex/skills/plan-review/SKILL.md",          "SKILL_PLAN_REVIEW_MD",       "committed"),

    # .cortex/commands/
    (".cortex/commands/fix-issue.md",                "COMMAND_FIX_ISSUE_MD",       "committed"),
    (".cortex/commands/status.md",                   "COMMAND_STATUS_MD",          "committed"),

    # .cortex/agents/
    (".cortex/agents/code-reviewer.md",              "AGENT_CODE_REVIEWER_MD",     "committed"),
    (".cortex/agents/plan-architect.md",              "AGENT_PLANNER_MD",           "committed"),

    # .cortex/agent-memory/
    (".cortex/agent-memory/README.md",               "AGENT_MEMORY_INDEX_MD",      "committed"),

    # .cortex/output-styles/
    (".cortex/output-styles/teaching.md",            "OUTPUT_STYLE_TEACHING_MD",   "committed"),
    (".cortex/output-styles/concise.md",             "OUTPUT_STYLE_CONCISE_MD",    "committed"),
]

# Directories to create even if empty (for the structure to exist)
EMPTY_DIRS: list[str] = [
    ".cortex/agent-memory-local",
]

# Minimal set (for --minimal flag)
MINIMAL_FILES: set[str] = {
    "CORTEX.md",
    ".cortex/.gitignore",
    ".cortex/settings.json",
    ".cortex/settings.local.json",
}


class CortexInit:
    """
    Scaffolds the .cortex/ directory structure in a target directory.
    """

    def __init__(
        self,
        target_dir: str | Path = ".",
        minimal: bool = False,
        force: bool = False,
        dry_run: bool = False,
    ) -> None:
        self.target = Path(target_dir).resolve()
        self.minimal = minimal
        self.force = force
        self.dry_run = dry_run
        self._created: list[str] = []
        self._skipped: list[str] = []
        self._overwritten: list[str] = []

    def run(self) -> dict[str, list[str]]:
        """
        Execute the scaffold. Returns a summary dict with
        created, skipped, and overwritten file lists.
        """
        logger.info("Initializing Cortex in: %s", self.target)

        # Create empty directories
        for dir_path in EMPTY_DIRS:
            full = self.target / dir_path
            if not self.dry_run:
                full.mkdir(parents=True, exist_ok=True)
            logger.debug("  dir: %s", dir_path)

        # Write files
        for rel_path, content_attr, badge in FILE_MANIFEST:
            if self.minimal and rel_path not in MINIMAL_FILES:
                continue
            self._write_file(rel_path, content_attr)

        # Update root .gitignore if it exists
        self._update_gitignore()

        # Summary
        summary = {
            "target": str(self.target),
            "created": self._created,
            "skipped": self._skipped,
            "overwritten": self._overwritten,
        }

        logger.info(
            "Done. Created: %d, Skipped: %d, Overwritten: %d",
            len(self._created), len(self._skipped), len(self._overwritten),
        )
        return summary

    def _write_file(self, rel_path: str, content_attr: str) -> None:
        """Write a single file from the template."""
        full_path = self.target / rel_path
        content = getattr(T, content_attr, "")

        if full_path.exists() and not self.force:
            self._skipped.append(rel_path)
            logger.debug("  skip (exists): %s", rel_path)
            return

        if full_path.exists():
            self._overwritten.append(rel_path)
            action = "overwrite"
        else:
            self._created.append(rel_path)
            action = "create"

        if self.dry_run:
            logger.info("  [dry-run] %s: %s", action, rel_path)
            return

        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        logger.info("  %s: %s", action, rel_path)

    def _update_gitignore(self) -> None:
        """Append Cortex ignores to root .gitignore if not already present."""
        gitignore = self.target / ".gitignore"
        marker = "# Cortex local files"

        if gitignore.exists():
            existing = gitignore.read_text(encoding="utf-8")
            if marker in existing:
                logger.debug("  .gitignore already has Cortex entries")
                return

            if not self.dry_run:
                with gitignore.open("a", encoding="utf-8") as f:
                    f.write(T.ROOT_GITIGNORE_ADDITIONS)
            logger.info("  updated: .gitignore (appended Cortex ignores)")
        else:
            # No .gitignore — create one with just the Cortex entries
            if not self.dry_run:
                gitignore.write_text(T.ROOT_GITIGNORE_ADDITIONS.lstrip(), encoding="utf-8")
            logger.info("  created: .gitignore")

    # ── Inspection ───────────────────────────────────────────────────

    @staticmethod
    def is_initialized(target_dir: str | Path = ".") -> bool:
        """Check if a directory already has a .cortex/ folder."""
        return (Path(target_dir).resolve() / ".cortex").is_dir()

    @staticmethod
    def tree(target_dir: str | Path = ".") -> str:
        """Return a visual tree of the .cortex/ directory."""
        root = Path(target_dir).resolve()
        lines = []

        def _walk(path: Path, prefix: str = "") -> None:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
            for i, entry in enumerate(entries):
                connector = "└── " if i == len(entries) - 1 else "├── "
                lines.append(f"{prefix}{connector}{entry.name}")
                if entry.is_dir():
                    extension = "    " if i == len(entries) - 1 else "│   "
                    _walk(entry, prefix + extension)

        cortex_dir = root / ".cortex"
        cortex_md = root / "CORTEX.md"

        if cortex_md.exists():
            lines.append("CORTEX.md")
        if cortex_dir.exists():
            lines.append(".cortex/")
            _walk(cortex_dir)

        return "\n".join(lines)
