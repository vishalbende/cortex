"""Tests for the PermissionResolver."""

import pytest
from cortex.permissions.resolver import PermissionResolver


class TestPermissionResolver:
    def test_granted_permission_passes(self):
        resolver = PermissionResolver(granted_permissions=["read", "write"])
        assert resolver.check("read", step_id="s1") is True

    def test_missing_permission_denied(self):
        resolver = PermissionResolver(granted_permissions=["read"])
        assert resolver.check("admin:delete_all", step_id="s1") is False
        assert len(resolver.denied) == 1

    def test_destructive_always_denied(self):
        resolver = PermissionResolver(granted_permissions=["delete"])
        # Even if "delete" is granted, destructive actions need explicit confirmation
        assert resolver.check("delete files", step_id="s1") is False

    def test_grant_and_revoke(self):
        resolver = PermissionResolver()
        resolver.grant("deploy")
        assert "deploy" in resolver.granted_permissions
        resolver.revoke("deploy")
        assert "deploy" not in resolver.granted_permissions

    def test_semantic_similarity_matching(self):
        resolver = PermissionResolver(granted_permissions=["design:read", "design:write"])
        # "design:read" should match "design:read" exactly
        assert resolver.check("design:read", step_id="s1") is True

    def test_check_agent_all_pass(self):
        resolver = PermissionResolver(granted_permissions=["read", "write"])
        assert resolver.check_agent("test", ["read", "write"], step_id="s1") is True

    def test_check_agent_partial_fail(self):
        resolver = PermissionResolver(granted_permissions=["read"])
        assert resolver.check_agent("test", ["read", "admin"], step_id="s1") is False
