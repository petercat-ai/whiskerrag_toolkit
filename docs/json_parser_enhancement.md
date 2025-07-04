# JSON Parser Enhancement

## Overview

The JSON parser has been enhanced to provide better compatibility with different JSON formats. Previously, the parser only supported dictionary (object) format JSON, but now it also supports array format JSON.

## Key Improvements

### 1. Enhanced JSON Content Parsing

A new helper function `parse_json_content()` has been added to handle JSON parsing with better error handling and format support:

```python
def parse_json_content(content: str) -> Union[Dict[str, Any], List[Any]]:
    """
    Parse JSON content with better compatibility.

    Args:
        content: JSON string content

    Returns:
        Parsed JSON object (dict or list)

    Raises:
        ValueError: If content is not valid JSON or is not a dict/list
    """
```

### 2. Support for Array Format

The parser now supports both dictionary and array JSON formats:

- **Dictionary format**: `{"key": "value", ...}`
- **Array format**: `[{"item": 1}, {"item": 2}, ...]`

When an array is provided, it's automatically converted to a dictionary format with a "data" key for processing by the `RecursiveJsonSplitter`.

### 3. Improved Error Handling

Better error messages for different types of JSON parsing issues:

- Invalid JSON syntax
- Non-dictionary/non-array JSON content
- General parsing errors

## Usage Examples

### Dictionary Format JSON

```python
json_content = """
{
    "user": {
        "name": "John Doe",
        "age": 30,
        "email": "john.doe@example.com"
    }
}
"""

# This works as before
result = await parser.parse(knowledge, Text(content=json_content, metadata={}))
```

### Array Format JSON

```python
json_content = """
[
    {"id": 1, "name": "Product A"},
    {"id": 2, "name": "Product B"},
    {"id": 3, "name": "Product C"}
]
"""

# This now works with the enhanced parser
result = await parser.parse(knowledge, Text(content=json_content, metadata={}))
```

### Error Handling

```python
# Invalid JSON
try:
    await parser.parse(knowledge, Text(content='{"invalid": json}', metadata={}))
except ValueError as e:
    print(f"Error: {e}")  # "Invalid JSON content provided for splitting: ..."

# Primitive JSON (not supported)
try:
    await parser.parse(knowledge, Text(content="42", metadata={}))
except ValueError as e:
    print(f"Error: {e}")  # "JSON content must be a dictionary or array."
```

## Backward Compatibility

This enhancement is fully backward compatible. Existing code that uses dictionary format JSON will continue to work without any changes.

## Testing

The enhancement includes comprehensive tests covering:

- Dictionary format JSON parsing
- Array format JSON parsing
- Error handling for invalid JSON
- Error handling for unsupported JSON types

Run the tests with:

```bash
python -m pytest tests/whiskerrag_utils/parser/test_json_splitter.py -v
```

## Example Script

A complete example demonstrating the new functionality is available at:

```
examples/json_parser_example.py
```

Run it with:

```bash
python examples/json_parser_example.py
```
