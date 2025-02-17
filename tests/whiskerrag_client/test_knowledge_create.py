# coding: utf-8

"""
FastAPI

No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)

The version of the OpenAPI document: 0.1.0
Generated by OpenAPI Generator (https://openapi-generator.tech)

Do not edit the class manually.
"""  # noqa: E501


import unittest

from whiskerrag_client.models.knowledge_create import KnowledgeCreate


class TestKnowledgeCreate(unittest.TestCase):
    """KnowledgeCreate unit test stubs"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def make_instance(self, include_optional) -> KnowledgeCreate:
        """Test KnowledgeCreate
        include_optional is a boolean, when False only required
        params are included, when True both required and
        optional params are included"""
        # uncomment below to create an instance of `KnowledgeCreate`
        """
        model = KnowledgeCreate()
        if include_optional:
            return KnowledgeCreate(
                source_type = 'github_repo',
                knowledge_type = 'text',
                space_id = '',
                knowledge_name = '',
                file_sha = '',
                file_size = 56,
                split_config = whiskerrag_client.models.knowledge_split_config.KnowledgeSplitConfig(
                    separators = [
                        ''
                        ], 
                    split_regex = '', 
                    chunk_size = 1.0, 
                    chunk_overlap = 0.0, ),
                source_data = '',
                source_url = 'ftp://PUx!u\'K}qz^sEC)lJ*=-jQ+\'6`%cClu,k\'',
                auth_info = '',
                embedding_model_name = 'openai',
                metadata = whiskerrag_client.models.metadata.Metadata()
            )
        else:
            return KnowledgeCreate(
                space_id = '',
                knowledge_name = '',
                split_config = whiskerrag_client.models.knowledge_split_config.KnowledgeSplitConfig(
                    separators = [
                        ''
                        ], 
                    split_regex = '', 
                    chunk_size = 1.0, 
                    chunk_overlap = 0.0, ),
        )
        """

    def testKnowledgeCreate(self):
        """Test KnowledgeCreate"""
        # inst_req_only = self.make_instance(include_optional=False)
        # inst_req_and_optional = self.make_instance(include_optional=True)


if __name__ == "__main__":
    unittest.main()
