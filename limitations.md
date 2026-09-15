# Limitations

Confirmed known limitations of the F1 platform.

- 单进程 / 单 lifecycle owner，不支持多 ASGI worker 跨进程协调。
- 无 distributed durable queue / Redis / Celery。
- CPU / GPU 并发槽位为静态配置，不支持按显存负载动态调度。
- 无任务失败自动 retry 或 priority 抢占。
- 无 WebSocket logs。
- 服务重启后未完成任务不续跑。
- API 无认证 / RBAC。
- Worker 无 per-job OS sandbox，继承宿主进程权限。
- `gradio`、`fastapi`、`pydantic`、`uvicorn` 等关键依赖尚未完整锁定版本上限。
- Windows 无 symlink 权限时相关测试会 skip；Final HEAD `0e7a8f83b53c97abc8eb3caedc532f5779fbb086` 的 Ubuntu CI 已实际执行对应 symlink 边界用例。
- OBB / classification 结果表支持仍不完整，列为 future scope。
- ONNX legacy exporter deprecation warning 为已知 P3。
