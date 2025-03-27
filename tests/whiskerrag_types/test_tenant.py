from uuid import UUID

from whiskerrag_types.model.tenant import Tenant
from whiskerrag_types.model.utils import parse_datetime

data = {
    "tenant_id": UUID("00a69557-995a-4ab4-99a7-c72a3048a3c0"),
    "tenant_name": "local_test",
    "email": "petercat.assistant@gmail.com",
    "secret_key": "sk-xxxxxxx",
    "is_active": True,
    "gmt_create": "2025-03-27T13:33:22.744889Z",
    "gmt_modified": "2025-03-27T13:33:23.625602Z",
}


class TestTenant:
    def test_dist_to_tenant(self) -> None:
        tenant = Tenant(**data)
        assert tenant.tenant_id == "00a69557-995a-4ab4-99a7-c72a3048a3c0"
        assert tenant.created_at == parse_datetime("2025-03-27T13:33:22.744889Z")
        assert tenant.updated_at == parse_datetime("2025-03-27T13:33:23.625602Z")

    def test_update_tenant(self) -> None:
        tenant = Tenant(**data)
        tenant.update(tenant_name="local_test_update")
        assert tenant.tenant_name == "local_test_update"
