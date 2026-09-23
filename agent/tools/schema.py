"""极简 JSON Schema 校验器，用于工具参数校验。

不依赖 pydantic，只覆盖工具参数常见的子集：
type(string/integer/number/boolean/object/array)、required、properties、items、
enum、minimum、maximum、minLength、maxLength。
"""

from __future__ import annotations

from typing import Any

_TYPE_MAP: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def validate_params(params: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """校验 params 是否符合 schema，返回错误列表（空列表 = 通过）。"""
    if schema.get("type") != "object":
        return ["参数 schema 根类型必须是 object"]
    return _validate(params, schema, "")


def _validate(value: Any, schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    label = path or "参数"
    type_name = schema.get("type")

    if type_name in _TYPE_MAP and not isinstance(value, _TYPE_MAP[type_name]):
        return [f"{label} 应为 {type_name} 类型"]

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{label} 必须是 {schema['enum']} 之一")

    if type_name in ("integer", "number"):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{label} 不能小于 {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{label} 不能大于 {schema['maximum']}")

    if type_name == "string":
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{label} 长度不能少于 {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{label} 长度不能超过 {schema['maxLength']}")

    if type_name == "object":
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"缺少必填字段 {key}")
        for key, sub_value in value.items():
            if key in schema.get("properties", {}):
                errors.extend(
                    _validate(sub_value, schema["properties"][key], f"{path}.{key}" if path else key)
                )

    if type_name == "array" and "items" in schema:
        for i, item in enumerate(value):
            errors.extend(_validate(item, schema["items"], f"{path}[{i}]" if path else f"[{i}]"))

    return errors
