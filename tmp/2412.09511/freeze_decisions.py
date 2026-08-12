import json
from collections import Counter
from pathlib import Path

path = Path(r"D:\study\deep-learning\paper\tmp\2412.09511\2412.09511_figure_table_decisions.json")
data = json.loads(path.read_text(encoding="utf-8"))
inserts = {"Figure 2", "Figure 7"}
defects = {"Table 9", "Table 10"}
notes = {
    "Figure 2": "裁剪覆盖完整双分支总览与右侧一致性对齐模块，分辨率 2096×724，可作为方法机制总图。",
    "Figure 7": "裁剪覆盖 PIAD-C 的 Clean 与七类腐化定性对比，包含 GT、GEAL、LASO，适合呈现鲁棒性。",
}

for item in data["decisions"]:
    if item.get("decision") != "review_pending":
        continue
    source_id = item["source_id"]
    if source_id in inserts:
        decision = "insert"
        skip_reason = ""
        quality = "usable_reviewed"
        review_status = "pass"
        failure_reason = ""
    elif source_id in defects:
        decision = "visual_defect"
        skip_reason = "identity_ambiguous_shared_crop"
        quality = "reject_visual_quality"
        review_status = "fail"
        failure_reason = "ambiguous_visual_body"
    else:
        decision = "skip"
        skip_reason = "reproduced_as_markdown_or_redundant"
        quality = "usable_but_not_inserted"
        review_status = "pass"
        failure_reason = ""

    if source_id == "Figure 2":
        preserved = ["完整框架模块", "2D/3D 分支", "GAFM", "Consistency Alignment"]
    elif source_id == "Figure 7":
        preserved = ["GT/GEAL/LASO", "Clean 与七类 corruption"]
    else:
        preserved = ["表头", "方法行", "指标数值"]

    omitted = ["候选裁剪同时包含 Table 9 与 Table 10，无法单表绑定"] if source_id in defects else []
    item.update(
        {
            "decision": decision,
            "reason": notes.get(source_id, "数值表将在正文中重写为可检索 Markdown，或与已选图内容重复。"),
            "skip_reason": skip_reason,
            "visual_quality_status": quality,
            "visual_review": {
                "status": review_status,
                "reviewed_asset_sha256": item.get("source_image_sha256", ""),
                "preserved_scientific_elements": preserved,
                "omitted_scientific_elements": omitted,
                "notes": notes.get(source_id, "图表主体尺寸与 PDF 区域匹配；为减少冗余不插图，关键数值转写为 Markdown。"),
                "failure_reason": failure_reason,
                "repair_attempts": 0,
                "revised_bbox": [],
            },
        }
    )

counts = Counter(item["decision"] for item in data["decisions"])
data["summary"]["by_decision"] = {
    key: counts.get(key, 0)
    for key in ["insert", "low_priority", "placeholder", "review_pending", "skip", "visual_defect"]
}
path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(data["summary"])
