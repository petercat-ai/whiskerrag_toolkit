from unittest.mock import Mock, patch

import httpx
import pytest
from pydantic import BaseModel

from whiskerrag_client import HttpClient


class TestModel(BaseModel):
    name: str
    value: int


@pytest.fixture
async def http_client():
    client = HttpClient(base_url="http://test.com", token="test_token", timeout=5.0)
    yield client
    await client.close()


@pytest.mark.asyncio
async def test_client_initialization(http_client):
    assert http_client.base_url == "http://test.com"
    assert http_client.headers == {
        "Authorization": "Bearer test_token",
        "Content-Type": "application/json",
    }
    assert http_client.timeout == 5.0


@pytest.mark.asyncio
async def test_request_with_dict_json(http_client):
    test_json = {"key": "value"}
    test_response = {"result": "success"}

    with patch.object(httpx.AsyncClient, "request") as mock_request:
        mock_response = Mock()
        mock_response.json.return_value = test_response
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response

        response = await http_client._request(
            method="POST", endpoint="/test", json=test_json
        )

        mock_request.assert_called_once()
        assert response == test_response


@pytest.mark.asyncio
async def test_request_with_pydantic_model(http_client):
    test_model = TestModel(name="test", value=123)
    test_response = {"result": "success"}

    with patch.object(httpx.AsyncClient, "request") as mock_request:
        mock_response = Mock()
        mock_response.json.return_value = test_response
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response

        response = await http_client._request(
            method="POST", endpoint="/test", json=test_model
        )

        mock_request.assert_called_once()
        assert response == test_response


@pytest.mark.asyncio
async def test_request_with_params(http_client):
    test_params = {"query": "test"}
    test_response = {"result": "success"}

    with patch.object(httpx.AsyncClient, "request") as mock_request:
        mock_response = Mock()
        mock_response.json.return_value = test_response
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response

        response = await http_client._request(
            method="GET", endpoint="/test", params=test_params
        )

        mock_request.assert_called_once()
        assert response == test_response


@pytest.mark.asyncio
async def test_request_with_invalid_json_type(http_client):
    with pytest.raises(ValueError) as exc_info:
        await http_client._request(
            method="POST", endpoint="/test", json=123  # Invalid JSON type
        )
    assert "Unsupported JSON type" in str(exc_info.value)


@pytest.mark.asyncio
async def test_request_http_error(http_client):
    with patch.object(httpx.AsyncClient, "request") as mock_request:
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPError("HTTP Error")
        mock_request.return_value = mock_response

        with pytest.raises(httpx.HTTPError):
            await http_client._request(method="GET", endpoint="/test")


@pytest.mark.asyncio
async def test_context_manager():
    async with HttpClient(base_url="http://test.com", token="test_token") as client:
        assert isinstance(client, HttpClient)


@pytest.mark.asyncio
async def test_custom_timeout(http_client):
    test_response = {"result": "success"}

    with patch.object(httpx.AsyncClient, "request") as mock_request:
        mock_response = Mock()
        mock_response.json.return_value = test_response
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response

        await http_client._request(method="GET", endpoint="/test")

        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["timeout"] == 5.0


@pytest.mark.asyncio
async def test_additional_request_kwargs(http_client):
    test_response = {"result": "success"}

    with patch.object(httpx.AsyncClient, "request") as mock_request:
        mock_response = Mock()
        mock_response.json.return_value = test_response
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response

        await http_client._request(
            method="GET", endpoint="/test", follow_redirects=True, verify=False
        )

        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["follow_redirects"] is True
        assert call_kwargs["verify"] is False
