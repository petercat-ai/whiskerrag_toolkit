from datetime import datetime

from whiskerrag_types.model.task import Task, TaskStatus


class TestTask:
    def test_dist_to_task(self) -> None:
        data = {
            "task_id": "8db89531-0a67-44e8-88f9-3616d9a49c8a",
            "status": "failed",
            "knowledge_id": "8cbfc366-cfb9-4c27-8273-494c97b7464a",
            "space_id": "ant-design/ant-design-charts",
            "user_id": None,
            "created_at": "2025-02-26T04:23:11.550164+00:00",
            "updated_at": "2025-03-06T13:02:44.383539+00:00",
            "tenant_id": "38fbd88b-e869-489c-9142-e4ea2c226e42",
            "error_message": "",
        }
        task = Task(**data)
        assert task.task_id == "8db89531-0a67-44e8-88f9-3616d9a49c8a"
        assert task.status == TaskStatus.FAILED.value
        assert task.knowledge_id == "8cbfc366-cfb9-4c27-8273-494c97b7464a"
        assert task.space_id == "ant-design/ant-design-charts"
        assert task.user_id is None
        assert task.created_at == datetime.fromisoformat(
            "2025-02-26T04:23:11.550164+00:00"
        )
        assert task.updated_at == datetime.fromisoformat(
            "2025-03-06T13:02:44.383539+00:00"
        )

    def test_json_alias_to_task(self) -> None:
        data = {
            "task_id": "8db89531-0a67-44e8-88f9-3616d9a49c8a",
            "status": "failed",
            "knowledge_id": "8cbfc366-cfb9-4c27-8273-494c97b7464a",
            "space_id": "ant-design/ant-design-charts",
            "user_id": None,
            "gmt_create": "2025-02-26T04:23:11.550164+00:00",
            "gmt_modified": "2025-03-06T13:02:44.383539+00:00",
            "tenant_id": "38fbd88b-e869-489c-9142-e4ea2c226e42",
            "error_message": "",
        }
        task = Task(**data)
        assert task.task_id == "8db89531-0a67-44e8-88f9-3616d9a49c8a"
        assert task.status == TaskStatus.FAILED.value
        assert task.knowledge_id == "8cbfc366-cfb9-4c27-8273-494c97b7464a"
        assert task.space_id == "ant-design/ant-design-charts"
        assert task.user_id is None
        assert task.created_at == datetime.fromisoformat(
            "2025-02-26T04:23:11.550164+00:00"
        )
        assert task.updated_at == datetime.fromisoformat(
            "2025-03-06T13:02:44.383539+00:00"
        )

    def test_update(self) -> None:
        now = datetime.now()
        data = {
            "task_id": "8db89531-0a67-44e8-88f9-3616d9a49c8a",
            "status": "failed",
            "knowledge_id": "8cbfc366-cfb9-4c27-8273-494c97b7464a",
            "space_id": "ant-design/ant-design-charts",
            "user_id": None,
            "gmt_create": "2025-02-26T04:23:11.550164+00:00",
            "gmt_modified": "2025-03-06T13:02:44.383539+00:00",
            "tenant_id": "38fbd88b-e869-489c-9142-e4ea2c226e42",
            "error_message": "",
        }
        task = Task(**data)
        task.update(status=TaskStatus.SUCCESS)
        assert task.task_id == "8db89531-0a67-44e8-88f9-3616d9a49c8a"
        assert task.status == TaskStatus.SUCCESS.value
        assert abs(task.updated_at.timestamp() - now.timestamp()) <= 1
