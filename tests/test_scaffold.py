"""Tests for the cortex init scaffolder."""

import pytest
import tempfile
from pathlib import Path

from cortex.scaffold.init import CortexInit


class TestCortexInit:
    def test_full_scaffold(self, tmp_path):
        init = CortexInit(target_dir=tmp_path)
        summary = init.run()

        # CORTEX.md at root
        assert (tmp_path / "CORTEX.md").exists()
        assert "Project conventions" in (tmp_path / "CORTEX.md").read_text()

        # .cortex/ directory
        assert (tmp_path / ".cortex").is_dir()
        assert (tmp_path / ".cortex" / "settings.json").exists()
        assert (tmp_path / ".cortex" / "settings.local.json").exists()
        assert (tmp_path / ".cortex" / ".gitignore").exists()

        # Rules
        assert (tmp_path / ".cortex" / "rules" / "testing.md").exists()
        assert (tmp_path / ".cortex" / "rules" / "agents.md").exists()
        assert (tmp_path / ".cortex" / "rules" / "api-design.md").exists()

        # Skills
        assert (tmp_path / ".cortex" / "skills" / "security-review" / "SKILL.md").exists()
        assert (tmp_path / ".cortex" / "skills" / "security-review" / "checklist.md").exists()
        assert (tmp_path / ".cortex" / "skills" / "plan-review" / "SKILL.md").exists()

        # Commands
        assert (tmp_path / ".cortex" / "commands" / "fix-issue.md").exists()
        assert (tmp_path / ".cortex" / "commands" / "status.md").exists()

        # Agents
        assert (tmp_path / ".cortex" / "agents" / "code-reviewer.md").exists()
        assert (tmp_path / ".cortex" / "agents" / "plan-architect.md").exists()

        # Agent memory
        assert (tmp_path / ".cortex" / "agent-memory" / "README.md").exists()
        assert (tmp_path / ".cortex" / "agent-memory-local").is_dir()

        # Output styles
        assert (tmp_path / ".cortex" / "output-styles" / "teaching.md").exists()
        assert (tmp_path / ".cortex" / "output-styles" / "concise.md").exists()

        # Summary
        assert len(summary["created"]) > 0
        assert len(summary["skipped"]) == 0

    def test_minimal_scaffold(self, tmp_path):
        init = CortexInit(target_dir=tmp_path, minimal=True)
        summary = init.run()

        assert (tmp_path / "CORTEX.md").exists()
        assert (tmp_path / ".cortex" / "settings.json").exists()
        assert (tmp_path / ".cortex" / ".gitignore").exists()

        # Minimal should NOT create skills, agents, etc.
        assert not (tmp_path / ".cortex" / "skills").exists()
        assert not (tmp_path / ".cortex" / "agents").exists()
        assert not (tmp_path / ".cortex" / "commands").exists()

    def test_skip_existing_no_force(self, tmp_path):
        # First run
        CortexInit(target_dir=tmp_path).run()

        # Modify CORTEX.md
        (tmp_path / "CORTEX.md").write_text("CUSTOM CONTENT")

        # Second run without --force
        summary = CortexInit(target_dir=tmp_path).run()

        # Should skip existing files
        assert "CORTEX.md" in summary["skipped"]
        assert (tmp_path / "CORTEX.md").read_text() == "CUSTOM CONTENT"

    def test_force_overwrites(self, tmp_path):
        CortexInit(target_dir=tmp_path).run()
        (tmp_path / "CORTEX.md").write_text("CUSTOM CONTENT")

        summary = CortexInit(target_dir=tmp_path, force=True).run()
        assert "CORTEX.md" in summary["overwritten"]
        assert "Project conventions" in (tmp_path / "CORTEX.md").read_text()

    def test_dry_run(self, tmp_path):
        init = CortexInit(target_dir=tmp_path, dry_run=True)
        summary = init.run()

        # Nothing should actually be created
        assert not (tmp_path / "CORTEX.md").exists()
        assert not (tmp_path / ".cortex").exists()
        assert len(summary["created"]) > 0  # but summary shows what WOULD be created

    def test_is_initialized(self, tmp_path):
        assert not CortexInit.is_initialized(tmp_path)
        CortexInit(target_dir=tmp_path).run()
        assert CortexInit.is_initialized(tmp_path)

    def test_tree_output(self, tmp_path):
        CortexInit(target_dir=tmp_path).run()
        tree = CortexInit.tree(tmp_path)
        assert "CORTEX.md" in tree
        assert ".cortex/" in tree
        assert "settings.json" in tree

    def test_gitignore_created(self, tmp_path):
        CortexInit(target_dir=tmp_path).run()
        gitignore = (tmp_path / ".gitignore").read_text()
        assert "Cortex local files" in gitignore
        assert "settings.local.json" in gitignore

    def test_gitignore_not_duplicated(self, tmp_path):
        CortexInit(target_dir=tmp_path).run()
        CortexInit(target_dir=tmp_path, force=True).run()
        gitignore = (tmp_path / ".gitignore").read_text()
        # Should only appear once
        assert gitignore.count("Cortex local files") == 1

    def test_settings_json_has_permissions(self, tmp_path):
        CortexInit(target_dir=tmp_path).run()
        import json
        settings = json.loads((tmp_path / ".cortex" / "settings.json").read_text())
        assert "permissions" in settings
        assert "allow" in settings["permissions"]
        assert "deny" in settings["permissions"]

    def test_rules_have_paths_frontmatter(self, tmp_path):
        CortexInit(target_dir=tmp_path).run()
        testing_rule = (tmp_path / ".cortex" / "rules" / "testing.md").read_text()
        assert "paths:" in testing_rule
        assert "**/*test*.py" in testing_rule

    def test_skills_have_description_frontmatter(self, tmp_path):
        CortexInit(target_dir=tmp_path).run()
        skill = (tmp_path / ".cortex" / "skills" / "security-review" / "SKILL.md").read_text()
        assert "description:" in skill
        assert "disable-model-invocation: true" in skill
