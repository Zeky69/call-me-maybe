from pydantic import BaseModel
from typing import Any

class ParameterDef(BaseModel):
    type: str

class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterDef]
    returns: ParameterDef


class FunctionCall(BaseModel):
    prompt : str
    name : str
    parameters : dict[str, Any]