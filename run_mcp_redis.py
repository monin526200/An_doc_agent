from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 保证从项目根导入 mcp_tool、history_redis
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loader import load_config
from history_redis.run_local_redis import (  # noqa: E402
    _data_dir,
    _redis_server_binary,
    write_local_redis_conf,
)

# 加载配置
_config = load_config()


def _terminate(proc: subprocess.Popen | None, *, name: str, grace: float = 3.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    sys.stderr.write(f"已停止：{name}\n")


@dataclass
class LocalServicesProcs:
    """后台 Redis + MCP 子进程句柄及连接信息（供 FastAPI 等复用）。"""

    redis_proc: subprocess.Popen | None
    mcp_proc: subprocess.Popen | None
    redis_url: str
    mcp_arxiv_url: str


def start_background_services(
    *,
    skip_redis: bool = False,
    redis_port: int | None = None,
    redis_bind: str | None = None,
    mcp_host: str | None = None,
    mcp_port: int | None = None,
    mcp_startup_sleep: float = 0.5,
) -> LocalServicesProcs:
    """
    非阻塞启动 Redis（可选）与 MCP（uvicorn），并设置 REDIS_URL、MCP_ARXIV_URL。
    mcp_host 默认 127.0.0.1，便于本机 FastMCP Client 连接；若需局域网访问可改为 0.0.0.0。
    参数默认从 config.yaml 读取，可通过传参覆盖。
    """
    os.chdir(ROOT)
    
    # 从配置读取默认值
    port = redis_port if redis_port is not None else _config.ports.redis_local
    bind = redis_bind if redis_bind is not None else _config.redis.local_bind
    mcp_host_val = mcp_host if mcp_host is not None else _config.mcp.host
    mcp_port_val = mcp_port if mcp_port is not None else _config.ports.mcp
    
    redis_url = f"redis://{bind}:{port}/0"
    connect_host = "127.0.0.1" if mcp_host_val in ("0.0.0.0", "::") else mcp_host_val
    mcp_arxiv_url = f"http://{connect_host}:{mcp_port_val}/arxiv/mcp"

    redis_proc: subprocess.Popen | None = None

    if not skip_redis:
        binary = _redis_server_binary()
        if not binary:
            raise RuntimeError(
                "未找到 redis-server。请安装 Redis，或设置 skip_redis=True。"
            )
        data_dir = _data_dir()
        conf = write_local_redis_conf(data_dir, port=port, bind=bind)
        sys.stderr.write(
            f"[Redis] {binary} {conf}\n"
            f"       数据目录：{data_dir}\n"
            f"       {redis_url}\n"
        )
        redis_proc = subprocess.Popen([binary, str(conf)])
        time.sleep(0.4)
        if redis_proc.poll() is not None:
            raise RuntimeError(
                f"Redis 进程已退出，可能端口 {port} 被占用。请检查配置或使用已有 Redis（skip_redis）。"
            )
    else:
        sys.stderr.write("[Redis] skip_redis=True，未启动本脚本管理的 redis-server\n")

    os.environ["REDIS_URL"] = redis_url
    os.environ["MCP_ARXIV_URL"] = mcp_arxiv_url

    sys.stderr.write(
        f"[MCP] uvicorn mcp_tool.mcp_server:app --host {mcp_host_val} --port {mcp_port_val}\n"
        f"      MCP 客户端 URL：{mcp_arxiv_url}\n"
    )
    mcp_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "mcp_tool.mcp_server:app",
            "--host",
            mcp_host_val,
            "--port",
            str(mcp_port_val),
        ],
        cwd=str(ROOT),
        env={**os.environ},
    )
    time.sleep(mcp_startup_sleep)
    if mcp_proc.poll() is not None:
        _terminate(redis_proc, name="Redis")
        raise RuntimeError("MCP (uvicorn) 进程启动失败，可能端口被占用。")

    return LocalServicesProcs(
        redis_proc=redis_proc,
        mcp_proc=mcp_proc,
        redis_url=redis_url,
        mcp_arxiv_url=mcp_arxiv_url,
    )


def stop_background_services(svc: LocalServicesProcs | None) -> None:
    if svc is None:
        return
    _terminate(svc.mcp_proc, name="MCP (uvicorn)")
    _terminate(svc.redis_proc, name="Redis")


def main(argv: list[str] | None = None) -> int:
    os.chdir(ROOT)

    p = argparse.ArgumentParser(description="本地启动 Redis + MCP（arXiv）服务")
    p.add_argument(
        "--skip-redis",
        action="store_true",
        help="不启动本脚本管理的 redis-server（使用已有实例）",
    )
    p.add_argument("--redis-port", type=int, default=None, help=f"默认 {_config.ports.redis_local}")
    p.add_argument("--redis-bind", default=None, help=f"默认 {_config.redis.local_bind}")
    p.add_argument("--mcp-host", default=None, help=f"默认 {_config.mcp.host}")
    p.add_argument("--mcp-port", type=int, default=None, help=f"默认 {_config.ports.mcp}")
    args = p.parse_args(argv)
    
    # 如果命令行未提供，使用配置文件值
    redis_bind = args.redis_bind if args.redis_bind is not None else _config.redis.local_bind
    mcp_host = args.mcp_host if args.mcp_host is not None else _config.mcp.host

    try:
        svc = start_background_services(
            skip_redis=args.skip_redis,
            redis_port=args.redis_port,
            redis_bind=redis_bind,
            mcp_host=mcp_host,
            mcp_port=args.mcp_port,
        )
    except RuntimeError as e:
        sys.stderr.write(f"{e}\n")
        return 1

    exit_code = 0
    try:
        assert svc.mcp_proc is not None
        exit_code = svc.mcp_proc.wait()
        if exit_code is None:
            exit_code = 1
    except KeyboardInterrupt:
        sys.stderr.write("\n正在停止子进程…\n")
        exit_code = 130
    finally:
        stop_background_services(svc)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
