# coding: utf-8

"""
FastAPI

No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)

The version of the OpenAPI document: 0.1.0
Generated by OpenAPI Generator (https://openapi-generator.tech)

Do not edit the class manually.
"""  # noqa: E501


import unittest

from whiskerrag_client.models.page_params_chunk import PageParamsChunk


class TestPageParamsChunk(unittest.TestCase):
    """PageParamsChunk unit test stubs"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def make_instance(self, include_optional) -> PageParamsChunk:
        """Test PageParamsChunk
        include_optional is a boolean, when False only required
        params are included, when True both required and
        optional params are included"""
        # uncomment below to create an instance of `PageParamsChunk`
        """
        model = PageParamsChunk()
        if include_optional:
            return PageParamsChunk(
                page = 1.0,
                page_size = 1.0,
                order_by = '',
                order_direction = '',
                eq_conditions = None
            )
        else:
            return PageParamsChunk(
        )
        """

    def testPageParamsChunk(self):
        """Test PageParamsChunk"""
        # inst_req_only = self.make_instance(include_optional=False)
        # inst_req_and_optional = self.make_instance(include_optional=True)


if __name__ == "__main__":
    unittest.main()
