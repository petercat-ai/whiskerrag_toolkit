from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from whiskerrag_types.model.artifact_index import ArtifactIndex, ArtifactIndexCreate


def test_artifact_create_with_valid_data():
    """正常创建 ArtifactIndexCreate，ecosystem 应该标准化为小写"""
    create_data = ArtifactIndexCreate(
        ecosystem="PYPI", name="  Requests  ", version="1.0.0", space_id="my-space"
    )
    assert create_data.ecosystem == "pypi"
    assert create_data.name == "Requests"  # name 去掉首尾空格
    assert create_data.version == "1.0.0"
    assert create_data.space_id == "my-space"


def test_space_id_validation():
    """space_id 禁止包含 .. 和 //"""
    with pytest.raises(ValidationError) as exc_info:
        ArtifactIndexCreate(ecosystem="npm", name="left-pad", space_id="invalid..id")
    assert "space_id cannot contain consecutive dots" in str(exc_info.value)

    with pytest.raises(ValidationError):
        ArtifactIndexCreate(ecosystem="npm", name="left-pad", space_id="invalid//id")


def test_artifact_id_auto_uuid():
    """ArtifactIndex 自动生成 artifact_id"""
    art = ArtifactIndex(ecosystem="go", name="gin", space_id="my-space")
    # 检查 artifact_id 是否为 UUIDv4
    UUID(art.artifact_id, version=4)


def test_update_method_and_updated_at():
    """update 方法应该更新字段并刷新更新时间"""
    art = ArtifactIndex(ecosystem="php", name="laravel/laravel", space_id="my-space")
    old_updated_at = art.updated_at
    art.update(version="9.0.0")
    assert art.version == "9.0.0"
    assert art.updated_at > old_updated_at


def test_model_validator_uuid_conversion():
    """model_validator 应该把 UUID 类型的输入转成 str"""
    art = ArtifactIndex(
        artifact_id=uuid4(), ecosystem="pypi", name="requests", space_id="my-space"
    )
    assert isinstance(art.artifact_id, str)
    UUID(art.artifact_id, version=4)


def test_artifact_extra_field():
    art = ArtifactIndexCreate(
        ecosystem="pypi",
        name="requests",
        version="2.31.0",
        space_id="my-space",
        extra={"build_env": "linux", "tag": "release"},
    )
    assert art.extra["build_env"] == "linux"
    assert art.extra["tag"] == "release"


def test_extra_field_defaults_to_empty_dict():
    art = ArtifactIndexCreate(ecosystem="npm", name="lodash", space_id="npm-space")
    assert art.extra == {}
