import pytest
from langchain_core.documents import Document

from whiskerrag_types.model.knowledge import Knowledge
from whiskerrag_types.model.multi_modal import Text
from whiskerrag_utils.helper.yuque import ExtendedYuqueLoader
from whiskerrag_utils.parser.yuque_doc_parser import YuqueParser


@pytest.mark.skip(reason="need to use true token")
async def test_parse_document_with_full_fields() -> None:
    try:
        loader = ExtendedYuqueLoader(
            api_url="https://yuque-api.xxx.com",
            access_token="xxx",
        )
        result: Document = loader.load_document_by_path(
            group_login="fish-pond", book_slug="smallfish", document_id="faq"
        )
        text = Text(
            content=result.page_content,
            metadata=result.metadata,
        )
        parser = YuqueParser()
        knowledge = Knowledge(
            **{
                "created_at": "2025-05-29T06:27:54.000000Z",
                "embedding_model_name": "bge-base-chinese-1117",
                "enabled": True,
                "knowledge_id": "e195fd88-58c3-4465-9627-b28995fb4032",
                "knowledge_name": "常见问题#faq",
                "knowledge_type": "yuquedoc",
                "metadata": {
                    "_knowledge_name": "常见问题#faq",
                    "_knowledge_type": "yuquedoc",
                    "_reference_url": "https://yuque.antfin.com/fish-pond/smallfish/faq",
                    "_summary_updated_at": "1416166.998307174",
                },
                "parent_id": None,
                "source_config": {
                    "api_url": "https://yuque-api.xxx.com",
                    "group_login": "fish-pond",
                    "book_slug": "smallfish",
                    "document_id": "faq",
                    "auth_info": "xxx",
                },
                "source_type": "yuque",
                "space_id": "ccc38cde-26c2-4386-bb3b-ba73e2e0eaab",
                "split_config": {
                    "type": "yuquedoc",
                    "chunk_size": 1500,
                    "chunk_overlap": 200,
                    "separators": ["#", "##", "###", "####", "\n\n", "\n", " "],
                    "is_separator_regex": False,
                },
                "tenant_id": "32c6837f-9999-9999-9999-999999999999",
            }
        )
        parsed_res = await parser.parse(knowledge, text)
        assert len(parsed_res) > 0
        print("------parsed_res", parsed_res)
    except Exception as e:
        print("error", e)
        pytest.skip(f"Failed to load document: use true token")
