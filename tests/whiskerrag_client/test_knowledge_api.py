# coding: utf-8

"""
FastAPI

No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)

The version of the OpenAPI document: 0.1.0
Generated by OpenAPI Generator (https://openapi-generator.tech)

Do not edit the class manually.
"""  # noqa: E501


import unittest

from whiskerrag_client.api.knowledge_api import KnowledgeApi


class TestKnowledgeApi(unittest.TestCase):
    """KnowledgeApi unit test stubs"""

    def setUp(self) -> None:
        self.api = KnowledgeApi()

    def tearDown(self) -> None:
        pass

    def test_add_knowledge(self) -> None:
        """Test case for add_knowledge

        Add Knowledge
        """
        pass

    def test_get_knowledge_by_id(self) -> None:
        """Test case for get_knowledge_by_id

        Get Knowledge By Id
        """
        pass

    def test_get_knowledge_list(self) -> None:
        """Test case for get_knowledge_list

        Get Knowledge List
        """
        pass


if __name__ == "__main__":
    unittest.main()
