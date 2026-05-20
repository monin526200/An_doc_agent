"""
项目统一配置加载器

从 config.yaml 加载所有配置，替代原有的环境变量方式。
提供类型化的配置访问和自动默认值处理。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent


def get_config_path() -> Path:
    """获取配置文件路径，优先使用环境变量覆盖（仅用于指定配置文件位置）"""
    env_path = os.environ.get("MOFL_CONFIG_PATH")
    if env_path:
        return Path(env_path).resolve()
    return ROOT / "config.yaml"


@dataclass
class PortsConfig:
    api: int = 8080
    ui: int = 7860
    mcp: int = 8000
    redis_local: int = 6380


@dataclass
class EndpointsConfig:
    mcp_arxiv_url: str = ""  # 空则自动生成
    api_base_url: str = ""   # 空则自动生成


@dataclass
class McpConfig:
    host: str = "0.0.0.0"


@dataclass
class RedisConfig:
    connection_mode: str = "params"  # "url" 或 "params"
    url: str = "redis://127.0.0.1:6380/0"
    host: str = "127.0.0.1"
    port: int = 6380
    db: int = 0
    username: str = ""
    password: str = ""
    local_bind: str = "127.0.0.1"
    local_data_dir: str = ""
    server_bin: str = ""
    session_ttl: int = 604800  # 7天
    max_summary_chars: int = 12000

    def get_connection_url(self) -> str:
        """获取Redis连接URL"""
        if self.connection_mode == "url" and self.url:
            return self.url
        
        # 从参数构建URL
        auth = ""
        if self.username and self.password:
            from urllib.parse import quote
            auth = f"{quote(self.username)}:{quote(self.password)}@"
        elif self.password:
            from urllib.parse import quote
            auth = f":{quote(self.password)}@"
        
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


@dataclass
class CorsConfig:
    allow_origins: str = "*"
    
    def get_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allow_origins.split(",") if o.strip()]


@dataclass
class PathsConfig:
    source_file: str = "rag/data/source_file"
    mcp_downloads: str = "mcp_downloads"
    
    def get_source_file_dir(self) -> Path:
        p = Path(self.source_file)
        if p.is_absolute():
            return p
        return ROOT / p
    
    def get_mcp_downloads_dir(self) -> Path:
        p = Path(self.mcp_downloads)
        if p.is_absolute():
            return p
        return ROOT / p


@dataclass
class TimeoutsConfig:
    workflow: float = 300.0
    ingest: float = 3600.0
    upload: float = 600.0


@dataclass
class LLMApiConfig:
    base_url: str = ""
    key: str = ""
    model: str = ""


@dataclass
class LLMLocalConfig:
    path: str = ""


@dataclass
class LLMConfig:
    mode: str = "api"  # "api" 或 "local"
    api: LLMApiConfig = field(default_factory=LLMApiConfig)
    local: LLMLocalConfig = field(default_factory=LLMLocalConfig)


@dataclass
class RuntimeConfig:
    chat_thread_id: str = ""


@dataclass
class ProjectConfig:
    ports: PortsConfig = field(default_factory=PortsConfig)
    endpoints: EndpointsConfig = field(default_factory=EndpointsConfig)
    mcp: McpConfig = field(default_factory=McpConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    cors: CorsConfig = field(default_factory=CorsConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    timeouts: TimeoutsConfig = field(default_factory=TimeoutsConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    
    def get_mcp_arxiv_url(self) -> str:
        if self.endpoints.mcp_arxiv_url:
            return self.endpoints.mcp_arxiv_url
        return f"http://127.0.0.1:{self.ports.mcp}/arxiv/mcp"
    
    def get_api_base_url(self) -> str:
        if self.endpoints.api_base_url:
            return self.endpoints.api_base_url
        return f"http://127.0.0.1:{self.ports.api}"


_config_instance: ProjectConfig | None = None


def load_config(force_reload: bool = False) -> ProjectConfig:
    global _config_instance
    
    if _config_instance is not None and not force_reload:
        return _config_instance
    
    config_path = get_config_path()
    config = ProjectConfig()
    
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            
            if "ports" in data:
                config.ports = PortsConfig(**data["ports"])
            if "endpoints" in data:
                config.endpoints = EndpointsConfig(**data["endpoints"])
            if "mcp" in data:
                config.mcp = McpConfig(**data["mcp"])
            if "redis" in data:
                config.redis = RedisConfig(**data["redis"])
            if "cors" in data:
                config.cors = CorsConfig(**data["cors"])
            if "paths" in data:
                config.paths = PathsConfig(**data["paths"])
            if "timeouts" in data:
                config.timeouts = TimeoutsConfig(**data["timeouts"])
            if "llm" in data:
                llm_data = data["llm"]
                config.llm = LLMConfig(
                    mode=llm_data.get("mode", "api"),
                    api=LLMApiConfig(**llm_data.get("api", {})),
                    local=LLMLocalConfig(**llm_data.get("local", {})),
                )
            if "runtime" in data:
                config.runtime = RuntimeConfig(**data["runtime"])
                
        except Exception as e:
            print(f"[warn] 配置加载失败 {config_path}: {e}，使用默认配置")
    else:
        print(f"[warn] 配置文件不存在 {config_path}，使用默认配置")
    
    _config_instance = config
    return config


def get_config() -> ProjectConfig:
    return load_config()


def reload_config() -> ProjectConfig:
    return load_config(force_reload=True)


def sync_to_environ(config: ProjectConfig | None = None) -> None:
    """将配置同步到环境变量（兼容旧代码）"""
    if config is None:
        config = load_config()
    
    # Redis
    os.environ["REDIS_URL"] = config.redis.get_connection_url()
    os.environ["REDIS_HOST"] = config.redis.host
    os.environ["REDIS_PORT"] = str(config.redis.port)
    os.environ["REDIS_DB"] = str(config.redis.db)
    if config.redis.username:
        os.environ["REDIS_USERNAME"] = config.redis.username
    if config.redis.password:
        os.environ["REDIS_PASSWORD"] = config.redis.password
    os.environ["REDIS_SESSION_TTL"] = str(config.redis.session_ttl)
    os.environ["REDIS_MAX_SUMMARY_CHARS"] = str(config.redis.max_summary_chars)
    os.environ["REDIS_LOCAL_PORT"] = str(config.redis.port)
    os.environ["REDIS_LOCAL_BIND"] = config.redis.local_bind
    if config.redis.local_data_dir:
        os.environ["REDIS_LOCAL_DATA_DIR"] = config.redis.local_data_dir
    if config.redis.server_bin:
        os.environ["REDIS_SERVER_BIN"] = config.redis.server_bin
    
    # MCP
    os.environ["MCP_PORT"] = str(config.ports.mcp)
    os.environ["MCP_HOST"] = config.mcp.host
    os.environ["MCP_ARXIV_URL"] = config.get_mcp_arxiv_url()
    
    # API/UI
    os.environ["API_BASE_URL"] = config.get_api_base_url()
    os.environ["CORS_ALLOW_ORIGINS"] = config.cors.allow_origins
    
    # 路径
    os.environ["API_SOURCE_FILE_DIR"] = str(config.paths.get_source_file_dir())
    
    # 超时
    os.environ["UI_HTTP_TIMEOUT_WORKFLOW"] = str(config.timeouts.workflow)
    os.environ["UI_HTTP_TIMEOUT_INGEST"] = str(config.timeouts.ingest)
    os.environ["UI_HTTP_TIMEOUT_UPLOAD"] = str(config.timeouts.upload)
    
    # LLM配置同步（供llm_server等使用）
    os.environ["LLM_MODE"] = config.llm.mode
    os.environ["LLM_API_BASE_URL"] = config.llm.api.base_url
    os.environ["LLM_API_KEY"] = config.llm.api.key
    os.environ["LLM_API_MODEL"] = config.llm.api.model
    os.environ["LLM_LOCAL_PATH"] = config.llm.local.path


if __name__ == "__main__":
    # 测试配置加载
    cfg = load_config()
    print(f"配置文件路径: {get_config_path()}")
    print(f"API端口: {cfg.ports.api}")
    print(f"UI端口: {cfg.ports.ui}")
    print(f"MCP端口: {cfg.ports.mcp}")
    print(f"Redis端口: {cfg.ports.redis_local}")
    print(f"MCP arXiv URL: {cfg.get_mcp_arxiv_url()}")
    print(f"API基础URL: {cfg.get_api_base_url()}")
    print(f"Redis连接URL: {cfg.redis.get_connection_url()}")
    print(f"源文件目录: {cfg.paths.get_source_file_dir()}")
