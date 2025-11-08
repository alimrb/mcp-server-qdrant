"""
Hybrid HTTP transport for mcp-server-qdrant.

This module keeps the official Streamable HTTP endpoint (for MCP clients)
while also serving a lightweight JSON-RPC shim on the same `/mcp` path so
simple HTTP clients (like Codex's Mattermost bridge) can talk to Qdrant
without implementing the streaming contract.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from fastmcp.server.http import create_streamable_http_app
from qdrant_client import models
from starlette.applications import Starlette
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

from .server import mcp
from .qdrant import Entry


def _jsonrpc_result(id_value: Any, result: Dict[str, Any]) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": id_value, "result": result})


def _jsonrpc_error(id_value: Any, code: int, message: str) -> JSONResponse:
    return JSONResponse(
        {"jsonrpc": "2.0", "id": id_value, "error": {"code": code, "message": message}}
    )


async def _tool_payload() -> List[Dict[str, Any]]:
    tools = await mcp._tool_manager.get_tools()
    return [tool.to_mcp_tool().model_dump() for tool in tools.values()]


def _resolve_tool_name(raw_name: str | None) -> str:
    if not raw_name:
        raise ValueError("Missing tool name")
    name = raw_name
    if "." in raw_name:
        namespace, rest = raw_name.split(".", 1)
        if namespace in {"qdrant", "qdrant-rag", "qdrant-rag-mcp"}:
            name = rest
    elif raw_name.startswith("qdrant_"):
        name = raw_name[len("qdrant_") :]
    return name


def _serialize_content(items: Iterable[Any]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for item in items:
        if hasattr(item, "model_dump"):
            content.append(item.model_dump())  # pydantic BaseModel
        elif isinstance(item, dict):
            content.append(item)
        else:
            content.append({"type": "text", "text": str(item)})
    return content


async def _handle_jsonrpc(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"error": "Invalid JSON-RPC payload"}, status_code=400)

    method = payload.get("method")
    id_value = payload.get("id")

    if method == "initialize":
        result = {
            "protocolVersion": "0.1",
            "serverInfo": {"name": mcp.name, "version": "0.1.0"},
            "capabilities": {"tools": {"listChanged": True}},
        }
        return _jsonrpc_result(id_value, result)

    if method == "tools/list":
        return _jsonrpc_result(id_value, {"tools": await _tool_payload()})

    if method == "tools/list_changed":
        return _jsonrpc_result(id_value, {})

    if method == "tools/call":
        params = payload.get("params") or {}
        name = _resolve_tool_name(params.get("name"))
        arguments = params.get("arguments") or {}
        try:
            result_blocks = await _call_tool_direct(name, arguments)
        except Exception as exc:  # noqa: BLE001
            return _jsonrpc_error(id_value, -32000, str(exc))
        return _jsonrpc_result(
            id_value,
            {"content": _serialize_content(result_blocks)},
        )

    if method == "ping":
        return _jsonrpc_result(id_value, {"status": "ok"})

    return _jsonrpc_error(id_value, -32601, f"Method not implemented: {method}")


def _should_use_jsonrpc(scope, headers: Headers) -> bool:
    if scope["method"] != "POST":
        return False
    accept = headers.get("accept", "")
    if "application/json-seq" in accept or "text/event-stream" in accept:
        return False
    content_type = headers.get("content-type", "")
    return "application/json" in content_type.lower()


class HybridMCPApp:
    def __init__(self, stream_app):
        self.stream_app = stream_app

    async def __call__(self, scope, receive, send):  # type: ignore[override]
        if scope["type"] != "http":
            await self.stream_app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        path = scope.get("path", "")

        if path.startswith("/mcp") and _should_use_jsonrpc(scope, headers):
            request = Request(scope, receive=receive)
            response: Response = await _handle_jsonrpc(request)
            await response(scope, receive, send)
            return

        await self.stream_app(scope, receive, send)


def create_app() -> Starlette:
    stream_app = create_streamable_http_app(
        mcp,
        "/mcp",
        json_response=True,
        stateless_http=True,
    )

    async def health(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "server": mcp.name})

    hybrid = HybridMCPApp(stream_app)

    return Starlette(
        routes=[
            Route("/healthz", health, methods=["GET"]),
            Route("/", health, methods=["GET"]),
            Mount("/", hybrid),
        ],
        lifespan=stream_app.router.lifespan_context,
    )


# Uvicorn entrypoint convenience (used by docker-compose).
app = create_app()


async def _call_tool_direct(name: str, arguments: Dict[str, Any]) -> List[str]:
    """
    Minimal JSON-RPC handlers for qdrant-store and qdrant-find.
    Streamable HTTP requests still flow through FastMCP, so we only need to cover
    the simple JSON bridge used by Codex automation.
    """
    if name == "qdrant-store":
        information = arguments.get("information")
        if not information:
            raise ValueError("information is required")
        metadata = arguments.get("metadata")
        collection_name = arguments.get("collection_name")
        collection = collection_name or mcp.qdrant_settings.collection_name
        if not collection:
            raise ValueError("collection_name is required when no default is set")
        entry = Entry(content=information, metadata=metadata)
        await mcp.qdrant_connector.store(entry, collection_name=collection)
        return [f"Remembered: {information} in collection {collection}"]

    if name == "qdrant-find":
        query = arguments.get("query")
        if not query:
            raise ValueError("query is required")
        collection_name = arguments.get("collection_name")
        collection = collection_name or mcp.qdrant_settings.collection_name
        if not collection:
            raise ValueError("collection_name is required when no default is set")
        query_filter = arguments.get("query_filter")
        filter_obj = models.Filter(**query_filter) if query_filter else None
        entries = await mcp.qdrant_connector.search(
            query,
            collection_name=collection,
            limit=mcp.qdrant_settings.search_limit,
            query_filter=filter_obj,
        )
        if not entries:
            return ["No matching entries."]
        content = [f"Results for the query '{query}'"]
        for entry in entries:
            entry_meta = entry.metadata or {}
            content.append(
                f"<entry><content>{entry.content}</content><metadata>{entry_meta}</metadata></entry>"
            )
        return content

    raise ValueError(f"Unknown tool: {name}")
