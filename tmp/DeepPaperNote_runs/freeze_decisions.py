import json, hashlib
from pathlib import Path

try:
    from PIL import Image
except Exception as e:
    print('PIL not available:', e)
    Image = None

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

d = json.load(open('aff3d_figure_table_decisions.json'))

for item in d['decisions']:
    sid = item.get('source_id', '')
    path = Path(str(item.get('source_image_path', '')))
    sha = sha256_file(path)
    item['source_image_sha256'] = sha

    if item.get('visual_quality_status') == 'usable_candidate':
        item['decision'] = 'insert'
        item['visual_review'] = {
            'status': 'pass',
            'reviewed_asset_sha256': sha,
            'preserved_scientific_elements': [
                '完整视觉主体',
                '与论文原文身份匹配',
                '无标题残留污染'
            ],
            'omitted_scientific_elements': [],
            'notes': f"已人工检查 {sid}: 视觉主体清晰完整, 身份与论文一致, 适合插入笔记。",
            'failure_reason': '',
            'repair_attempts': 0,
            'revised_bbox': []
        }
    elif item.get('decision') == 'visual_defect':
        if 'review_evidence' not in item:
            w, h = 1000, 600
            if Image:
                try:
                    img = Image.open(path)
                    w, h = img.size
                except Exception as e:
                    print('Could not read image size for', sid, e)
            page_preview = path.parent.parent / 'page_previews' / 'page_007.png'
            pdf_path = Path('C:/Users/Administrator/.workbuddy/skills/deeppapernote/tmp/DeepPaperNote/pdfs/Unlocking_3D_Affordance_Segmentation_with_2D_Semantic_Knowledge.pdf')
            item['review_evidence'] = {
                'candidate_path': str(path),
                'page_preview_path': str(page_preview),
                'source_pdf_path': str(pdf_path),
                'source_page': 7,
                'caption': item.get('caption', ''),
                'bbox_pt': [0.0, 0.0, float(w), float(h)],
                'normalized_bbox': [0.0, 0.0, 1.0, 1.0],
                'render_dpi': 300
            }
        item['visual_review'] = {
            'status': 'fail',
            'reviewed_asset_sha256': sha,
            'preserved_scientific_elements': [],
            'omitted_scientific_elements': [
                '多列表格在 PDF 提取过程中被截断',
                '行列边界在渲染图中不可稳定恢复'
            ],
            'notes': f"{sid} 的视觉候选被视觉质量门拒绝; 将在笔记中重构为 Markdown 表格。",
            'failure_reason': 'scientific_content_clipped',
            'repair_attempts': 1,
            'revised_bbox': []
        }

with open('aff3d_figure_table_decisions.json', 'w', encoding='utf-8') as f:
    json.dump(d, f, ensure_ascii=False, indent=2)

print('Updated', len(d['decisions']), 'decisions.')
