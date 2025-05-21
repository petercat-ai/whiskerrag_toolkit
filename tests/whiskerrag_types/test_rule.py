from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from whiskerrag_types.model.rule import GlobalRule, Rule, SpaceRule


class TestRuleModels:
    def test_rule_base_creation(self) -> None:
        # 测试基础规则创建
        rule = Rule(content="test content", tenant_id="tenant1", space_id="space1")
        assert rule.content == "test content"
        assert rule.tenant_id == "tenant1"
        assert rule.space_id == "space1"
        assert rule.is_active == True  # 测试默认值

    def test_rule_base_update(self) -> None:
        # 测试更新方法
        rule = Rule(content="test content", tenant_id="tenant1", space_id="space1")
        original_updated_at = rule.updated_at

        import time

        time.sleep(0.001)

        updated_rule = rule.update(content="new content")

        # 确保两个时间戳都转换为 UTC 时区进行比较
        original_time = original_updated_at.astimezone(timezone.utc)
        updated_time = updated_rule.updated_at.astimezone(timezone.utc)

        assert updated_rule.content == "new content"
        assert updated_time > original_time

    def test_global_rule_valid_creation(self) -> None:
        # 测试有效的全局规则创建
        rule1 = GlobalRule(content="global rule", tenant_id="tenant1")
        assert rule1.space_id is None

        rule2 = GlobalRule(content="global rule", tenant_id="tenant1", space_id=None)
        assert rule2.space_id is None

    def test_global_rule_invalid_creation(self) -> None:
        # 测试无效的全局规则创建（space_id 不为 None）
        with pytest.raises(ValidationError) as exc_info:
            GlobalRule(content="global rule", tenant_id="tenant1", space_id="space1")
        assert "space_id must be None for GlobalRule" in str(exc_info.value)

    def test_space_rule_valid_creation(self) -> None:
        # 测试有效的空间规则创建
        rule = SpaceRule(content="space rule", tenant_id="tenant1", space_id="space1")
        assert rule.space_id == "space1"

    def test_space_rule_invalid_creation(self) -> None:
        # 测试无效的空间规则创建（缺少必需的 space_id）
        with pytest.raises(ValidationError):
            SpaceRule(content="space rule", tenant_id="tenant1")

    def test_required_fields(self) -> None:
        # 测试必填字段验证
        with pytest.raises(ValidationError):
            Rule(tenant_id="tenant1")  # 缺少 rule_content

        with pytest.raises(ValidationError):
            Rule(content="content")  # 缺少 tenant_id

    def test_optional_fields(self) -> None:
        # 测试可选字段
        rule = Rule(content="test content", tenant_id="tenant1")
        assert rule.space_id is None
        assert rule.is_active == True

    def test_is_active_toggle(self) -> None:
        # 测试 is_active 字段
        rule = Rule(content="test content", tenant_id="tenant1", is_active=False)
        assert rule.is_active == False

    def test_timestamps(self) -> None:
        # 测试时间戳字段
        rule = Rule(content="test content", tenant_id="tenant1")
        assert isinstance(rule.created_at, datetime)
        assert isinstance(rule.updated_at, datetime)

    def test_rule_models_from_dict(self) -> None:
        # 测试从字典创建规则模型
        test_data = {
            "content": "test content",
            "tenant_id": "tenant1",
            "space_id": "space1",
            "is_active": True,
            "created_at": "2023-01-01T12:00:00+00:00",
            "updated_at": "2023-01-01T12:00:00+00:00",
        }

        # 测试 RuleBase
        rule_base = Rule.model_validate(test_data)
        assert rule_base.content == "test content"
        assert rule_base.tenant_id == "tenant1"
        assert rule_base.space_id == "space1"
        assert rule_base.is_active == True
        assert rule_base.created_at.isoformat() == "2023-01-01T12:00:00+00:00"
        assert rule_base.updated_at.isoformat() == "2023-01-01T12:00:00+00:00"

        # 测试 SpaceRule
        space_rule = SpaceRule.model_validate(test_data)
        assert space_rule.space_id == "space1"

        # 测试缺少必填字段的情况
        invalid_data = {
            "rule_content": "test content"
            # 缺少 tenant_id
        }
        with pytest.raises(ValidationError):
            Rule.model_validate(invalid_data)

    def test_rule_models_from_complex_dict(self) -> None:
        # 测试从包含嵌套结构的字典创建规则模型
        test_data = {
            "content": "test content",
            "tenant_id": "tenant1",
            "space_id": "space1",
            "is_active": True,
            "created_at": "2023-01-01T12:00:00+00:00",
            "updated_at": "2023-01-01T12:00:00+00:00",
            # 添加一些额外的字段
            "metadata": {
                "version": "1.0",
                "tags": ["tag1", "tag2"],
                "config": {"priority": 1, "enabled": True},
            },
            "extra_field": "should be ignored",  # 额外的字段应该被忽略
        }

        # 测试 RuleBase（应该忽略额外的字段）
        rule_base = Rule.model_validate(test_data)
        assert rule_base.content == "test content"
        assert rule_base.tenant_id == "tenant1"
        assert rule_base.space_id == "space1"
        assert rule_base.is_active == True
        assert rule_base.created_at.isoformat() == "2023-01-01T12:00:00+00:00"
        assert rule_base.updated_at.isoformat() == "2023-01-01T12:00:00+00:00"

        # 确认额外字段被忽略
        with pytest.raises(AttributeError):
            rule_base.metadata
        with pytest.raises(AttributeError):
            rule_base.extra_field

        # 测试 GlobalRule（space_id 必须为 None）
        global_rule_data = test_data.copy()
        global_rule_data["space_id"] = None
        global_rule = GlobalRule.model_validate(global_rule_data)
        assert global_rule.space_id is None

        # 测试无效的 GlobalRule 数据
        with pytest.raises(ValidationError):
            GlobalRule.model_validate(test_data)  # space_id 不为 None 应该失败
