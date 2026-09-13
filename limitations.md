# Limitations

Confirmed known limitations of the F1 platform.

- 单进程 / 单 lifecycle owner，不支持多 ASGI worker 跨进程协调。
- 无 distributed durable queue / Redis / Celery。
- 无 WebSocket logs。
- 服务重启后未完成任务不续跑。
- Windows 无 symlink 权限时相关测试会 skip，Linux CI 应实际执行。
- OBB / classification 结果表支持仍不完整，列为 future scope。
- ONNX legacy exporter deprecation warning 为已知 P3。