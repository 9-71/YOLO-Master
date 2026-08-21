# 腾讯犀牛鸟开源人才计划 · YOLO-Master 课题筹备检查

## 检查结论

- 原稿共 30 页，主线、课题卡片、时间表、依赖关系和验收红线均已形成，适合作为筹备会底稿。
- 课题内容与当前仓库 HEAD 的主要切入点基本一致：`MultiTaskHead`、routing protocol、Foundation distillation、F11 router KD、V-PEFT、export preflight 和 routing diagnostics 均能在仓库中找到对应实现。
- 原稿存在发布风险：中文字体声明为 `Noto Sans CJK SC`，在部分演示环境可能缺字；主题 D 的第三课题编号使用 `F11`，与其他主题的编号体系不一致。

## 已完成的优化

1. 以原稿为基线生成独立文件：`tencent_rhino_bird_yolo_master_topics_20260820_optimized.pptx`。
2. 将演示文稿中的中文字体声明统一为 `Heiti SC`，降低 macOS 演示环境缺字风险。
3. 将主题 D 第三个课题统一命名为 `D3 路由蒸馏`；正文保留 `F11` 作为仓库实现路径标识，例如 `foundation_distill_model.py` 的 F11 路径。
4. 将跨组依赖中的 `D2 → F11` 更新为 `D2 → D3`，使目录、课题页、选题表和依赖页的编号一致。
5. 使用 PPTX 打包校验和 `markitdown` 复核，输出文件结构有效、30 页内容仍然完整。

## 今晚筹备会建议优先确认

- 每个课题的负责人、备选负责人和算力额度；先锁定 P0，不把 P2 目标当成硬承诺。
- A1/A2/A3、D1/D2/D3、E1/E2/E3 的公共脚手架与共享 GPU 排期，避免重复搭建。
- 统一实验协议：数据版本、训练预算、seed 数、延迟 batch、评测脚本和 PR 验收模板。
- 需要对外投屏前，在实际会议电脑上打开优化版并检查字体；若仍有缺字，直接在该电脑上替换为已安装的中文字体后再保存。

## 技术表述边界

- `OBB` 在 `MultiTaskHead` 中仍是 compatibility-only；课题 C2 应继续表述为“补齐 unified OBB 训练分支”，不要说成仓库已经完成统一 OBB。
- F11 是 Foundation Teacher Router 的实现路径，不是主题 D 的课题编号；对外统一使用 D3，代码和配置中保留 F11。
- 导出课题应区分 eager sparse dispatch 与导出后的 dense fallback，不能把“导出支持”直接等同于“稀疏路由在端侧保持不变”。

