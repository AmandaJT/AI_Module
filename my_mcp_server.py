#!/usr/bin/env python3

"""
My MCP Server

一个简化版 MCP（Model Context Protocol）服务器示例：
对外通过 MCP 协议暴露工具，对内通过 HTTP/SSE 调用后端业务 API。
整体分两层：
  - Client 类：封装后端 REST/SSE 调用（API 客户端层）
  - @mcp_tool 装饰的函数：把能力暴露给 LLM（协议适配层）
"""

import threading  # 用于 threading.local 实现线程本地存储
import json       # 解析 SSE 事件中的 JSON 数据
import os         # 读取环境变量（如鉴权 token）
import requests   # HTTP 客户端库

from requests.exceptions import RequestException
from upload_paths import set_active_transport  # 记录当前使用的传输方式（stdio/sse/http）


class Client:
    """后端 API 客户端：统一处理鉴权头、会话管理和请求发送"""

    def __init__(self, url: str, key: str):
        self.url = url  # 后端服务基础地址
        self.key = key  # API 密钥
        # 每个线程都有自己的会话 request session非线程安全
        # MCP 2.x 会把同步工具放到线程池中并发执行，
        # 而 requests.Session 不是线程安全的，
        # 所以用 threading.local 为每个线程维护独立的 Session
        self._session_local = threading.local()

    def _get_session(self) -> requests.Session:
        """创建一个新的 Session，并注入鉴权头和默认 Content-Type"""
        session = requests.Session()
        session.headers.update(
            {"API_Key": self.key,
            "Content-Type": "application/json"
        })
        return session

    @property
    def session(self) -> requests.Session:
        """线程本地 Session 的惰性初始化：
        当前线程第一次访问时才创建，之后复用同一会话（连接池）"""
        if not hasattr(self._session_local, "session"):
            self._session_local.session = self._get_session()
        return self._session_local.session

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """统一的普通 HTTP 请求入口：拼 URL、发请求、检查状态码、解析 JSON

        Args:
            method: HTTP 方法（GET/POST/PUT/DELETE）
            endpoint: 接口路径（不含基础地址）
            **kwargs: 透传给 requests 的参数（如 json、params 等）

        Returns:
            解析后的 JSON 响应（dict）
        """
        url = f"{self.url}/{endpoint}"
        try:
            response = self.session.request(method, url, **kwargs)
            # 4xx/5xx 状态码直接抛异常
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {e}")

    def queryForExample(self, query: str) -> Dict[str, Any]:
        # 查询示例：演示一个普通的 POST JSON 接口调用
        return self._request("POST", "queryForExample", json=query)

    def _sse_stream(self, url: str, body: Dict[str, Any]) -> Dict:
        """POST 请求并消费 SSE（Server-Sent Events）流式响应，组装出完整结果

        SSE 协议要点：
          - 每行格式为 "字段名: 值"，本实现只关心 data 字段
          - 一个事件可包含多行 data，多行值用 \n 拼接
          - 空行（blank line）表示一个事件结束，此时才解析派发
        """
        try:
            # stream=True：边下边读，不等整个响应体
            # timeout=(连接超时, 读取间隔超时)，读超时设长一些以容忍 LLM 慢响应
            response = self.session.post(url, json=body, stream=True,
            timeout=(10, CHAT_TIMEOUT),
            )
            response.raise_for_status()

            asw_chunks: list = []        # 累积所有 answer 片段，最后拼接成完整回答
            ref: list = []               # 引用列表
            debug: list = []             # 调试事件轨迹
            content_line: list[str] = [] # 当前事件尚未结束的所有 data 行
            content_bytes = 0            # 当前事件累计字节数（用于超限防护）

            # with 确保 break/异常时连接也能归还连接池
            with response:
                for line in response.iter_lines():
                    # iter_lines 按 \n 或 \r\n 切分并去掉行尾换行符
                    if isinstance(line, bytes):
                        line = line.decode("utf-8")
                    if line:
                        # 非空行：按 "字段名: 值" 拆分，只处理 data 字段
                        field, _, value = line.partition(": ")
                        if field == "data":
                            if value.startswith(" "):
                                # 去掉第一个空格（SSE 规范：冒号后至多去掉一个前导空格）
                                value = value[1:]
                            # 统计当前事件字节数，超过上限直接抛异常，防止内存被撑爆
                            content_bytes += len(value.encode("utf-8"))+1
                            if content_bytes > MAX_BYTES:
                                raise RequestException(
                                    f"SSE事件超过最大字节数: {MAX_BYTES}"
                                )
                            content_line.append(value)
                        # 非 data 字段（event/id/retry/注释等）直接忽略
                        continue
                    # 空行 = 一个事件结束：没有累积 data 的空帧直接跳过
                    if not content_line:
                        continue
                    # 多行 data 用 \n 拼接成完整 JSON 再解析
                    payload = "\n".join(content_lines)
                    content_lines = []
                    content_bytes = 0
                    try:
                        event_data = json.loads(payload)
                    except json.JSONDecodeError:
                        # 非 JSON 帧静默跳过（容错）
                        continue
                    response_type = event_data.get("response_type", "")
                    # 记录调试轨迹（content 截断到 80 字符防止日志过长）
                    debug.append({"type": response_type, "content": event_data.get("content", "")[:80]})

                    # 按事件类型分发处理
                    if response_type == "answer":
                        # 回答片段：追加累积
                        chunk = event_data.get("content", "")
                        if chunk:
                            asw_chunks.append(chunk)
                    elif response_type == "reference":
                        # 引用：整体覆盖（取最新一次）
                        ref = event_data.get("reference", "") or []
                    elif response_type == "error":
                        # 服务端报错：立即抛异常终止
                        raise RequestException(
                            f"错误: {event_data.get('content','unknown error')}"
                        )
                    elif response_type == "complete":
                        # 流结束：提前退出，后续数据不再读取
                        break

            # EOF 处没有空行收尾的残余数据按 SSE 规范直接丢弃
            return {
                "answer": "".join(asw_chunks),  # 拼接所有片段得到完整回答
                "reference": ref,
                "debug": debug,
            }
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {e}")

    def chat(self, session_id: str, query: str) -> Dict[str, Any]:
        """聊天接口：向指定会话发送问题，消费 SSE 流并返回组装结果"""
        body = Dict[str, Any] = {"query": query}
        result = self._sse_stream(
            f"{self.url}/chat/{session_id}",
            body
        )
        # 把会话 ID 一并返回，方便调用方关联上下文
        result["session_id"] = session_id
        return result

# ===================== MCP SERVER =====================
# 创建 MCP 服务器实例：根据函数类型注解自动生成工具的 JSON Schema，
# docstring 作为工具描述展示给 LLM
mcp = MCPServer("server_name", version="v1")
# 初始化后端 API 客户端（全局单例，供各工具函数使用）
client = Client(BASE_URL, "key")


# 注册工具：@mcp_tool() 把普通函数暴露为 LLM 可调用的 MCP 工具
@mcp_tool()
def queryForExample(query: str) -> Dict[str, Any]:
    return client.queryForExample(query)


# ===================== 三种传输方式 =====================

async def run_stdio():
    """stdio 传输：通过标准输入/输出通信，供本地桌面 AI 客户端使用，无需鉴权"""
    set_active_transport("stdio")
    await mcp.run_stdio_async()

async def run_sse(host: str, port: int):
    """SSE 传输：HTTP 长连接方式（旧版 MCP 客户端），需 uvicorn 提供 ASGI 服务"""
    set_active_transport("sse")
    # 网络传输强制要求配置共享密钥，防止未授权访问
    auth_token = os.getenv("MCP_SERVER_AUTH_TOKEN", "").strip()
    if not auth_token:
        raise Exception("MCP_SERVER_AUTH_TOKEN is required for sse transport.")
    try:
        import uvicorn
    except ImportError as e:
        raise Exception(f"uvicorn is required: {e}")

    # MCPAuth 是 ASGI 鉴权中间件：包裹在 MCP 的 Starlette 应用外面，
    # 校验请求头中的 token，不通过直接返回 401，通过则转交内层应用处理
    app = MCPAuth(mcp.sse_app(host=host, message_path=SSE_MESSAGE_PATH),
    auth_token,
    )
    config = uvicorn.Config(app, host=host, port=port)
    server = uvicorn.Server(config)
    await server.serve()

async def run_http(host: str, port: int):
    """Streamable HTTP 传输：新版 MCP 规范（2025-03-26）的无状态 HTTP 方式"""
    set_active_transport("http")
    # 同样强制要求鉴权 token
    auth_token = os.getenv("MCP_SERVER_AUTH_TOKEN", "").strip()
    if not auth_token:
        raise Exception("MCP_SERVER_AUTH_TOKEN is required for http transport.")
    try:
        import uvicorn
    except ImportError as e:
        raise Exception(f"uvicorn is required: {e}")

    app = MCPAuth(mcp.streamable_http_app(host=host, stateless_http=STREAMABLE_HTTP_STATELESS),
    auth_token,
    )
    config = uvicorn.Config(app, host=host, port=port)
    server = uvicorn.Server(config)
    await server.serve()


# 以下为 set_active_transport 的参考实现（已迁移到 upload_paths 模块，此处保留注释备忘）
# _active_transport: Optional[str] = None


# def set_active_transport(transport: str) -> None:
#     """Record the transport selected at server startup (CLI or run_* entry)."""
#     global _active_transport
#     _active_transport = transport.strip().lower()
