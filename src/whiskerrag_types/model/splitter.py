from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


class BaseCharSplitConfig(BaseModel):
    """Base split configuration class"""

    chunk_size: int = Field(default=1500, ge=1, lt=5000)
    chunk_overlap: int = Field(default=150, ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> "BaseCharSplitConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self


class MarkdownSplitConfig(BaseCharSplitConfig):
    """Markdown document split configuration"""

    separators: Optional[List[str]] = Field(description="separator list")
    split_regex: Optional[str] = Field(description="split_regex")


class PDFSplitConfig(BaseCharSplitConfig):
    """PDF document split configuration"""

    split_by_page: bool = Field(default=False, description="Whether to split by pages")
    keep_layout: bool = Field(
        default=True, description="Whether to preserve the original layout"
    )
    extract_images: bool = Field(default=False, description="Whether to extract images")
    table_extract_mode: str = Field(
        default="text", description="Table extraction mode: 'text' or 'structure'"
    )


class TextSplitConfig(BaseCharSplitConfig):
    """Plain text split configuration"""

    separators: List[str] = Field(
        default=[
            "\n\n",
        ],
        description="""List of separators to split the text. If None, uses default separators""",
    )
    keep_separator: Optional[Union[bool, Literal["start", "end"]]] = Field(
        default=False,
        description="""Whether to keep the separator and where to place it in each corresponding chunk (True='start')""",
    )
    strip_whitespace: Optional[bool] = Field(
        default=False,
        description="""If `True`, strips whitespace from the start and end of every document""",
    )


class JSONSplitConfig(BaseCharSplitConfig):
    """JSON document split configuration"""

    split_level: int = Field(
        default=1, description="Depth level for JSON splitting", ge=1
    )
    preserve_structure: bool = Field(
        default=True, description="Whether to preserve JSON structure"
    )
    array_handling: str = Field(
        default="split",
        description="Array handling mode: 'split' or 'merge'",
    )
    key_filters: Optional[List[str]] = Field(
        default=None, description="List of keys to process; processes all keys if None"
    )

    @model_validator(mode="after")
    def validate_array_handling(self) -> "JSONSplitConfig":
        valid_handlers = ["split", "merge"]
        if self.array_handling not in valid_handlers:
            raise ValueError(f"array_handling must be one of {valid_handlers}")
        return self
