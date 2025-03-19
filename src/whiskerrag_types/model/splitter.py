from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


class BaseSplitConfig(BaseModel):
    """Base split configuration class"""

    chunk_size: int = Field(default=1500, ge=1)
    chunk_overlap: int = Field(default=150, ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> "BaseSplitConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self


class MarkdownSplitConfig(BaseSplitConfig):
    """Markdown document split configuration"""

    separators: Optional[List[str]] = Field(description="separator list")
    split_regex: Optional[str] = Field(description="split_regex")
    # split_by_heading: Optional[bool] = Field(description="Whether to split by headings")
    # remove_markdown_chars: Optional[bool] = Field(
    #     description="Whether to remove Markdown syntax characters"
    # )


class PDFSplitConfig(BaseSplitConfig):
    """PDF document split configuration"""

    split_by_page: bool = Field(default=False, description="Whether to split by pages")
    keep_layout: bool = Field(
        default=True, description="Whether to preserve the original layout"
    )
    extract_images: bool = Field(default=False, description="Whether to extract images")
    table_extract_mode: str = Field(
        default="text", description="Table extraction mode: 'text' or 'structure'"
    )


class TextSplitConfig(BaseSplitConfig):
    """Plain text split configuration"""

    keep_separator: Union[bool, Literal["start", "end"]] = Field(
        default=False,
        description="""Whether to keep the separator and where to place it in each corresponding chunk (True='start')""",
    )
    strip_whitespace: bool = Field(
        default=False,
        description="""If `True`, strips whitespace from the start and end of every document""",
    )


class JSONSplitConfig(BaseSplitConfig):
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
