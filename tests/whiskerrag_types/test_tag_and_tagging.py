from uuid import UUID

import pytest
from pydantic import ValidationError

from whiskerrag_types.model.tag import Tag, TagCreate
from whiskerrag_types.model.tagging import Tagging, TaggingCreate, TagObjectType


def test_tag_name_normalization_and_blacklist():
    # ✅ 合法的
    t = TagCreate(name=" 标签123🚀_", object_type="space")
    assert t.name == "标签123🚀_"
    assert t.object_type == TagObjectType.SPACE

    # ❌ 包含黑名单特殊字符
    with pytest.raises(ValidationError) as exc_info:
        TagCreate(name="bad@tag", object_type="space")
    assert "不允许包含空格或常见特殊字符" in str(exc_info.value)

    # ✅ 允许中间空格
    t2 = TagCreate(name="bad tag", object_type="space")
    assert t2.name == "bad tag"  # 保留中间空格


def test_tag_object_type_normalization():
    t = TagCreate(name="abc", object_type="SPACE")  # 大写输入
    assert t.object_type == TagObjectType.SPACE


def test_tag_uuid_and_update_method():
    t = Tag(tenant_id="tenant1", name="abc", object_type=TagObjectType.SPACE)
    # 检查 UUID 格式
    UUID(t.tag_id, version=4)

    old_updated = t.updated_at
    t.update(description="new desc")

    assert t.description == "new desc"
    assert t.updated_at > old_updated


def test_tagging_create_normalization():
    tc = TaggingCreate(
        tag_name=" 标签A 🚀 ", object_type="SPACE", object_id="   obj_1   "
    )
    assert tc.tag_name.startswith("标签a")  # 小写化
    assert tc.object_id == "obj_1"
    assert tc.object_type == TagObjectType.SPACE


def test_tagging_uuid_and_update_method():
    tagging = Tagging(
        tenant_id="tenant1",
        tag_id=str(UUID(int=1)),  # mock UUID
        object_id="obj1",
        object_type=TagObjectType.SPACE,
    )
    # 检查 UUID 格式
    UUID(tagging.tagging_id, version=4)

    old_updated = tagging.updated_at
    tagging.update(object_id="obj2")

    assert tagging.object_id == "obj2"
    assert tagging.updated_at > old_updated


def test_tag_and_tagging_cross_check():
    """
    模拟绑定流程：创建 Tag，然后 Tagging 绑定该 tag_id
    """
    tag = Tag(tenant_id="tenant1", name="abc", object_type=TagObjectType.SPACE)

    tagging = Tagging(
        tenant_id=tag.tenant_id,
        tag_id=tag.tag_id,
        object_id="space_001",
        object_type=tag.object_type,
    )

    assert tagging.tag_id == tag.tag_id
    assert tagging.tenant_id == tag.tenant_id
    assert tagging.object_type == tag.object_type
