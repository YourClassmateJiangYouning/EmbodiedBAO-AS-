"""Write the conclusions + Stage 2 document as a real .docx, with no dependencies.

python-docx is not installed on the machines this runs on and pandoc is not either, so
the OOXML package is written directly: [Content_Types].xml, the relationships, a styles
part with the headings and body font, and word/document.xml.  Word, LibreOffice and
WPS all open the result; the file is a zip, which is why a broken one is obvious.

Content lives in BLOCKS at the bottom: every paragraph, bullet, heading and table is a
tuple.  "**bold**" inside a paragraph becomes a bold run.

Run from the repository root:  python lab_logs/make_conclusions_docx.py
"""

from __future__ import annotations

import os
import re
import zipfile

OUT = "结论与Stage2设计.docx"
BODY_FONT = "Times New Roman"
BODY_CJK = "宋体"
HEAD_FONT = "Arial"
HEAD_CJK = "黑体"

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
</Types>"""

ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
</Relationships>"""

DOC_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

CORE = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:title>EmbodiedBAO — 结论与 Stage 2 设计</dc:title>
<dc:creator>EmbodiedBAO</dc:creator>
</cp:coreProperties>"""


def style(style_id: str, name: str, size_half_pt: int, bold: bool, font: str, cjk: str,
          before: int, after: int, outline: int = None) -> str:
    outline_xml = f'<w:outlineLvl w:val="{outline}"/>' if outline is not None else ""
    return f"""<w:style w:type="paragraph" w:styleId="{style_id}">
<w:name w:val="{name}"/>
<w:pPr><w:spacing w:before="{before}" w:after="{after}" w:line="300" w:lineRule="auto"/>{outline_xml}</w:pPr>
<w:rPr><w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:eastAsia="{cjk}"/>
<w:b w:val="{'true' if bold else 'false'}"/><w:sz w:val="{size_half_pt}"/>
<w:szCs w:val="{size_half_pt}"/></w:rPr>
</w:style>"""


STYLES = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:docDefaults><w:rPrDefault><w:rPr>
<w:rFonts w:ascii="{BODY_FONT}" w:hAnsi="{BODY_FONT}" w:eastAsia="{BODY_CJK}"/><w:sz w:val="21"/>
</w:rPr></w:rPrDefault></w:docDefaults>
{style("Normal", "Normal", 21, False, BODY_FONT, BODY_CJK, 0, 120)}
{style("Title", "Title", 32, True, HEAD_FONT, HEAD_CJK, 0, 240)}
{style("Heading1", "heading 1", 28, True, HEAD_FONT, HEAD_CJK, 280, 140, 0)}
{style("Heading2", "heading 2", 24, True, HEAD_FONT, HEAD_CJK, 220, 120, 1)}
{style("Heading3", "heading 3", 22, True, HEAD_FONT, HEAD_CJK, 180, 100, 2)}
{style("ListParagraph", "List Paragraph", 21, False, BODY_FONT, BODY_CJK, 0, 60)}
</w:styles>"""


def escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def runs(text: str, bold_all: bool = False) -> str:
    """Turn **bold** markers into bold runs; everything else is one normal run.

    The span pattern is deliberately non-greedy and allows a lone asterisk inside, so a
    bold span may contain notation like the optimal angle 2*theta* -- the first version
    of this rejected that and left two literal ** markers in the document.
    """
    pieces = re.split(r"(\*\*.+?\*\*)", text, flags=re.S)
    out = []
    for piece in pieces:
        if not piece:
            continue
        bold = bold_all
        if piece.startswith("**") and piece.endswith("**") and len(piece) > 4:
            piece, bold = piece[2:-2], True
        out.append(
            '<w:r><w:rPr><w:b w:val="%s"/></w:rPr><w:t xml:space="preserve">%s</w:t></w:r>'
            % ("true" if bold else "false", escape(piece))
        )
    return "".join(out)


def paragraph(text: str, style_id: str = "Normal", bullet: bool = False) -> str:
    prefix = "•  " if bullet else ""
    return (
        '<w:p><w:pPr><w:pStyle w:val="%s"/><w:ind w:left="%d" w:hanging="%d"/></w:pPr>%s</w:p>'
        % (style_id, 360 if bullet else 0, 200 if bullet else 0, runs(prefix + text))
    )


def table(rows, widths=None) -> str:
    columns = len(rows[0])
    widths = widths or [int(9360 / columns)] * columns
    borders = (
        '<w:tblBorders>'
        + "".join(
            '<w:%s w:val="single" w:sz="4" w:space="0" w:color="808080"/>' % edge
            for edge in ("top", "left", "bottom", "right", "insideH", "insideV")
        )
        + "</w:tblBorders>"
    )
    xml = ['<w:tbl><w:tblPr><w:tblW w:w="9360" w:type="dxa"/>%s</w:tblPr>' % borders]
    xml.append("<w:tblGrid>" + "".join('<w:gridCol w:w="%d"/>' % w for w in widths) + "</w:tblGrid>")
    for row_index, row in enumerate(rows):
        xml.append("<w:tr>")
        for column, cell in enumerate(row):
            shade = '<w:shd w:val="clear" w:fill="EFEFEF"/>' if row_index == 0 else ""
            xml.append(
                '<w:tc><w:tcPr><w:tcW w:w="%d" w:type="dxa"/>%s</w:tcPr>%s</w:tc>'
                % (widths[column], shade, paragraph(str(cell), "Normal"))
            )
        xml.append("</w:tr>")
    xml.append("</w:tbl>")
    return "".join(xml)


def build(blocks, path: str) -> None:
    body = []
    for kind, payload in blocks:
        if kind == "title":
            body.append(paragraph(payload, "Title"))
        elif kind == "h1":
            body.append(paragraph(payload, "Heading1"))
        elif kind == "h2":
            body.append(paragraph(payload, "Heading2"))
        elif kind == "h3":
            body.append(paragraph(payload, "Heading3"))
        elif kind == "p":
            body.append(paragraph(payload))
        elif kind == "b":
            body.append(paragraph(payload, "ListParagraph", bullet=True))
        elif kind == "table":
            body.append(table(payload))
        elif kind == "gap":
            body.append(paragraph(""))
        else:
            raise ValueError(kind)
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>" + "".join(body)
        + '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
          '<w:pgMar w:top="1418" w:right="1418" w:bottom="1418" w:left="1418" '
          'w:header="851" w:footer="992" w:gutter="0"/></w:sectPr>'
        "</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("[Content_Types].xml", CONTENT_TYPES)
        package.writestr("_rels/.rels", ROOT_RELS)
        package.writestr("word/_rels/document.xml.rels", DOC_RELS)
        package.writestr("word/styles.xml", STYLES)
        package.writestr("word/document.xml", document)
        package.writestr("docProps/core.xml", CORE)
    print("wrote %s (%d blocks, %.1f KB)" % (
        path, len(blocks), os.path.getsize(path) / 1024.0))


BLOCKS = [
    ("title", "EmbodiedBAO：Stage 1 结论汇总与 Stage 2 设计"),
    ("p", "**数据来源**：本文所有数字均由仓库内脚本从归档 `lab_logs/bao_v7_all.tgz`（660 集）现算，"
          "未手工转录。对应脚本：`lab_logs/stage1_results.py`（结果表）、"
          "`lab_logs/stage1_reasoning_and_positions.py`（推理与转身位置）、"
          "`lab_logs/check_passage_yaws.py`（可穿越角度）、`lab_logs/model_report.py`（逐模型分类）。"
          "图件见 `lab_logs/figures/`。参考文献编号见文末；其中 [2] 与 [13] 的数值主张已联网核对原文。"),
    ("gap", ""),

    ("h1", "一、结论"),

    ("h2", "结论 1｜模型直到余量归零（A/S 1.0）才开始改变行为，0.9 处断崖至 14.5%"),
    ("p", "**数据**：660 集 = 11 个多模态模型 × 12 档开口（A/S 2.00 → 0.90，步长 0.1）× 每档 5 次。"
          "A/S ≥ 1.10 的 9 档通过率稳定在 91–98%（A/S 2.00 = 52/55，1.50 = 51/55，1.10 = 50/55）；"
          "**A/S 1.00（0.570 m，恰等于肩宽）降至 80%（44/55）**；"
          "**A/S 0.90（0.513 m）断崖式跌到 14.5%（8/55）**。"
          "A/S 0.90 要求至少 60° 的躯干旋转才能通过（实测可穿越偏航角为 60°–90°）。"),
    ("p", "**人类对照（三种口径）**：① Warren & Whang (1987) 及其后续（Higuchi et al. 2006）报告人类"
          "在开口宽为肩宽 **1.2–1.3 倍**时开始侧身 [1][13]；② Keizer et al. (2013) 的健康对照组为"
          "**1.25 倍**（该组患者为 1.40 倍）[2]；③ Franchak et al. (2012) 用 0.5 cm 分辨率的装置"
          "实测仅 **1.10 倍**，并指出早期研究因开口步长较粗（2.5–13 cm）而系统性偏高 [13]。"),
    ("p", "**推论**：模型在**几何上仍可通过、但余量已归零**（A/S 1.00）时才开始显著改变行为，"
          "到几何上不可能（0.90）才失败——即**缺少提前预防的安全余量**。"
          "幅度上，模型阈值 ≈ 1.00 比人类的 1.10/1.25/1.30 分别低约 **9%/20%/23%**，"
          "因此本文统一表述为“**低 9–23%，取决于所比较的人类口径**”，不写作 25–30%。"
          "此外须注明口径差异：人类文献测的是**旋转起始阈值**（肩部旋转达 20° 的开口宽），"
          "本文测的是**通过率曲线**，两者并非同一口径的量；本基准的结论是“模型的门槛更低”，"
          "而非“模型的旋转起始角等于 1.00”。"),
    ("p", "**参考文献**：[1] Warren & Whang (1987)；[2] Keizer et al. (2013)；[13] Franchak et al. (2012)；"
          "[3] Gibson (1979)。"),

    ("h2", "结论 2｜转身是早期决定，而不是被挡之后的反应"),
    ("p", "**数据**：281 个至少转身一次的集中，首次转身位置的中位数为 **x = 2.00 m**（平均 2.17 m），"
          "即离墙（x = 8.0 m）还有 **5.8 m**；**仅 3%（21/660）**的集贴到墙前（x ≥ 7.0 m）才转身；"
          "在 x < 4 m 就转身的占 38%（253/660）。"),
    ("p", "**推论**：这**推翻了“模型撞墙才反应”的直觉**。既然决定发生得很早，"
          "阈值偏低的成因就不在时机，而在（a）决定是否与开口宽度绑定（结论 3）与（b）转身幅度是否足够（结论 4）。"),
    ("p", "**参考文献**：[3] Gibson (1979)；[4] Fajen & Warren (2003)。"),

    ("h2", "结论 3｜“要不要转身”与开口宽度只是弱相关：39% 的无谓转身"),
    ("p", "**数据**：在**不需要任何转身**的 9 档（A/S ≥ 1.10，最少转身次数 = 0）共 605 集中，"
          "**39%（235/605）的集仍然转身**；这些集的通过率为 86.0%，而未转身的集为 96.8%。"
          "逐模型看，宽档转身率跨越 0%–98%（gpt-4o-mini 与 gemini-2.5-flash 为 0%，"
          "gemini-2.5-pro 为 98%）。"),
    ("p", "**推论**：相当一部分模型把“转身”当作**与宽度无关的固定动作**，并未把 A/S 作为决策变量。"
          "（通过率之差应读作相关性：乱转与判断混乱可能同源，不足以断言因果。）"),
    ("p", "**参考文献**：[5] Luchins (1942)；[6] Bilalić, McLeod & Gobet (2008)。"),

    ("h2", "结论 4｜最窄档的失败主因是幅度与对位，而不是“不知道要转”"),
    ("p", "**数据**：A/S 0.90 的 47 次失败中——完全不转 **9 次（19%）**、转 1–3 次 13 次、"
          "转 4–6 次 12 次、转 ≥7 次 13 次；即 **25/47（53%）已达到几何所需的最少转身次数却仍然失败**。"
          "**33/47（70%）从未达到 45°**（该档需 ≥60°）；另有 14 次进入过侧身带（45°–135°）仍失败，"
          "其中 5 次最大转角 ≥60° 仍未通过，指向**横向对位是独立的失败维度**。"),
    ("p", "**推论**：失败集中在“转多少”与“转完停在哪里”，而非“要不要转”。"
          "对训练与提示设计有直接含义：仅告知“需要转身”不足以解决问题。"),
    ("p", "**参考文献**：[1] Warren & Whang (1987)。"),

    ("h2", "结论 5｜失败不是知识问题，也不是不肯行动"),
    ("p", "**数据**：解析 660 份完整 agent 日志中模型的自述推理——**80%（528/660）的集主动提到开口/间隙**；"
          "在最窄档（A/S 0.90）该比例达 **87%（48/55）**，且该档 **93% 提到转身、80% 提到宽度/肩宽**。"
          "行动层面：**0/660 从未发出 forward**，仅 **8/660（1.2%）**未接近到 x ≥ 7.0 m。"),
    ("p", "**推论**：模型知道有开口，也知道可能需要转身（该知识亦写在动作说明中），"
          "但判断不出该转多少、在何处转、何时停。因此本基准测的是**身体尺度可供性判断**，"
          "而不是“知不知道转身有用”。"),
    ("p", "**参考文献**：[3] Gibson (1979)；[1] Warren & Whang (1987)。"),

    ("h2", "结论 6｜模型可分为四种行为类型，“按宽度判断”尚未解决"),
    ("p", "**数据**：整体通过率 58%–95%；**仅看两个需要转身的档（A/S 1.0 与 0.9）即可分层**"
          "（claude-sonnet-4-6 7/10、gemini-2.5-flash 6/10 … qwen3-vl-32b-instruct 2/10）。"
          "按可复核阈值（宽档转身率 ≥ 60% → 固定策略；全部集转身率 ≤ 10% → 直立派；"
          "介于两者且宽档 ≤ 40% → 按需转身；其余 → 半按需）分为："),
    ("table", [
        ["类型", "模型（宽档转身率）", "行为含义"],
        ["固定策略型（3）", "gemini-2.5-pro 98%、gpt-4.1 75%、qwen3-vl-235b 75%",
         "不需要转身的宽度也照转；平均最大转角最高 51°，多花 4–7 步"],
        ["直立派（3）", "gemini-2.5-flash 0%、gpt-4o-mini 0%、glm-4.6v 4%",
         "几乎不转身；分主要来自 A/S 1.0 的零容差正面通过，到 0.9 基本全败"],
        ["按需转身型（3）", "deepseek-v4.1-flash 11%、claude-sonnet-4-6 35%、gpt-4o 36%",
         "转与不转随宽度变化，最接近“判断”而非“固定策略”"],
        ["半按需型（2）", "qwen-vl-max 49%、qwen3-vl-32b-instruct 45%",
         "与宽度的关系较弱，介于两者之间"],
    ]),
    ("p", "**推论**：**模型间差异（58–95%）大于任何单一模型随宽度的变化**，"
          "说明“按宽度连续判断”是尚未解决的能力维度；同时存在两种截然不同的失败路径"
          "（从不转身 vs 一直转身），这决定了 Stage 2 必须按模型分别报告。"),
    ("p", "**参考文献**：[1] Warren & Whang (1987)；[5] Luchins (1942)。"),

    ("h2", "结论 7｜几何：最优躯干角是 21.1°，且“半转”比“不转”更占地方"),
    ("p", "**数据（解析 + 实测）**：所需宽度 needed(θ) = 0.570·|cosθ| + 0.220·|sinθ| "
          "= **0.611·cos(θ − 21.1°)**，峰值在 **21.1°（0.611 m）**；并且可解析证明 "
          "**needed(2θ*) = 肩宽 w 精确成立**，因此在 **0 < θ < 42.2°** 区间内，"
          "转身比不转身**更占地方**（θ = 15° → 0.608 m、30° → 0.604 m，而 0° → 0.570 m）；"
          "θ = 90° 时为厚度 0.220 m。**数据验证**：A/S 1.00（0.570 m）的 44 次成功中，"
          "**37 次穿越角为 0°、7 次 |θ| ≥ 45°、12°–42° 零次**；11 次失败中 9 次恰为 "
          "θ = 15°（4 次）或 30°（5 次）。对照 A/S 1.10（0.627 m）时 15°/30° 均可通过（成功集 15 次）。"),
    ("p", "**推论**：本任务的“正确答案”是反直觉的——不是“转到 45° 侧身”；"
          "任何固定的“永远转 90°/45°”策略都会在宽档暴露（结论 3）。"
          "这为指标设计提供了依据：必须测量**连续转角**，而不是“是否侧身”这一布尔量。"),
    ("p", "**参考文献**：[1] Warren & Whang (1987)；[4] Fajen & Warren (2003)。"),

    ("h2", "结论 8｜行动效率：即使通过也伴随显著冗余"),
    ("p", "**数据**：660 集平均 15.3 步（该宽度最优步数平均 11.3）→ 平均浪费 4.0 步；"
          "宽档通过者为 12.4–13.0 步（最优 11）；**最窄档通过者平均 22.1 步（最优 15，多 47%）**。"),
    ("p", "**推论**：成功往往伴随大量反复调整。因此在 Stage 2 中，"
          "“浪费步数（excess）是否下降”可作为“是否真正掌握策略”的辅助判据——"
          "区别于“碰巧通过”。"),
    ("p", "**参考文献**：（无外部引用，数据自证）"),

    ("h2", "结论 9｜主动自检（低头看自己）不带来成功"),
    ("p", "**数据**：**52%（346/660）的集至少低头看过自己一次**，平均 1.35 次感知动作/集。"
          "**同档位内**比较（排除“低头集中在窄档”的混淆）：11 个模型平均 **−9.1 个百分点**"
          "（即低头过的集通过率反而更低），仅 1 个模型为正；极端例：qwen3-vl-32b-instruct "
          "**−37.0 pp**（9 档中仅 1 档为正）；gpt-4o-mini 全程一次未低头，通过率仍达 92%。"),
    ("p", "**推论**：自检在本任务中更像“不确定时的补救动作”，而非有效策略；"
          "它既不构成成功的充分条件，也不足以弥补判断缺陷。"
          "Stage 2 因此把 look_down 作为**“会不会主动收集信息以推翻已学规则”**的指标，而不是成功指标。"),
    ("p", "**参考文献**：[3] Gibson (1979)。"),
    ("gap", ""),

    ("h1", "二、Stage 2：设计、依据与预期"),

    ("h2", "2.1 从 Stage 1 的结论到 Stage 2 的问题"),
    ("b", "结论 2 与 3 说明“转不转”是一个**早期且与宽度弱相关**的决定。"
          "若把模型放进一个**几何上注定失败**的通道里反复尝试、并让它每回合自己写一条笔记，"
          "这个决定会被更新吗？"),
    ("b", "结论 4 说明失败在**幅度与对位**。因此观测量必须是**连续的 gap**"
          "（离“塞得进去”还差多少），而不是“是否通过”——后者只有 0/1，学不出“在逼近”还是“卡住”；"
          "“是否通过”恒为否，没有信息量。"),
    ("b", "结论 5 说明**知识不是瓶颈**。因此任何变化只能来自**判断的更新**，"
          "这正是“顿悟 vs 渐悟”所要测量。"),
    ("b", "结论 6 说明存在“从不转”与“一直转”两种失败路径。"
          "因此两种记忆模式（累积 / 滚动）可能对不同类型模型效果不同，结果必须按模型分别报告。"),

    ("h2", "2.2 如何参考 Stage 1 设计（逐项对应）"),
    ("table", [
        ["Stage 1 的结论或参数", "Stage 2 的设计", "依据"],
        ["几何：A/S 0.90 需 ≥60° 旋转、最优 15 步；A/S 0.80 需 5 次转身、16 步",
         "学习段使用 **A/S 0.80（0.456 m）**",
         "比 Stage 1 最窄档再窄一档：任何姿态都过不去，唯一出路是改变策略而非运气"],
        ["A/S 1.10（0.627 m）：最少转身 0 次、最优 11 步、通过率 91%",
         "probe 段使用 **A/S 1.10**",
         "该档“转身不是任务要求”，因此出现的任何转身都是学来的——习惯的干净测量"],
        ["同一套场景、8 个动作、行走坐标系、512 px、成功判据",
         "完全沿用", "保证两阶段数据可比、可拼表"],
        ["Stage 1 字段（turned / total_rotation / final_torso_rotation / "
         "first_sideways_step / passed_sideways / step_success）",
         "Stage 2 全部保留并补齐",
         "两批数据可放进同一张表；已有测试直接以 Stage 1 的 CSV 表头做断言"],
        ["结论 4：失败在幅度",
         "主指标 gap = min(needed(θ) − W)（仅取 x ≥ 7.0 m 的步）",
         "连续量才能区分“改进”与“没动”；无值即空，不做插补"],
        ["结论 5：知识不是瓶颈",
         "判据只问“策略是否变化 + gap 是否跃变”",
         "避免把“说了要转”当成学到"],
    ]),

    ("h2", "2.3 协议与预注册判据"),
    ("b", "**两段式**：学习段 12 回合（A/S 0.80，每回合结束由模型自己写一条笔记，"
          "下一回合连同历史注入提示词）→ probe 段 5 回合（A/S 1.10，记忆继续注入，不再写笔记）。"),
    ("b", "**两种记忆模式**：累积（注入全部历史结果行与全部笔记，4 个 run）；"
          "滚动（只保留一条自写笔记、不带回合编号与结果，2 个 run）。"
          "用于分离“练了多少次”与“它知道自己练了多少次”。"),
    ("b", "**规模**：每模型 6 个 run × 17 回合 = 102 回合；11 个模型共 1,122 回合。"),
    ("b", "**预注册判据**（先于数据）：S = D/I（突变占比）、rho（gap 对回合的 Spearman 秩相关）、"
          "SetIndex（策略集中度）、improving_rounds、approach_latency。"
          "顿悟：S ≥ 0.60 且突变后连续 ≥ 2 回合 gap ≤ 0 且策略标签质变；"
          "渐悟：S ≤ 0.35 且改善回合 ≥ 3 且 rho ≤ −0.70；"
          "定势：I ≤ 0 或（SetIndex ≥ 0.60 且 rho > −0.70）；否则为振荡。"),
    ("b", "**实现要点**（均经实测）：笔记调用使用**独立的系统提示词**"
          "（沿用动作提示词会返回 JSON 而非文字）；帧尺寸 512 px；"
          "逐模型请求参数（deepseek 关闭推理、智谱关闭 thinking）；"
          "断点续跑；每个 tag 的 summary.json 每回合刷新；run_progress.txt 记录时间线。"),

    ("h2", "2.4 预期结果与可证伪推断"),
    ("b", "**H1（顿悟多于渐悟）**：若失败源于“少一个关键判断”，"
          "则笔记中出现“要转更多”之后应见 gap 跃变（S ≥ 0.6）。"),
    ("b", "**H2（习惯会残留）**：probe 第 1 回合的转角应显著高于 Stage 1 在 A/S 1.10 的同时期基线"
          "（Stage 1 该档仅 44% 的集转身、平均最大转角 15.0°），并随 probe 回合下降。"),
    ("b", "**H3（滚动记忆更接近人的“经验总结”）**：若压缩成一条笔记时顿悟比例更高，"
          "说明信息压缩有利于抽象出规则；反之说明模型需要原始结果行才能推理。"),
    ("b", "**证伪条件**：若 probe 阶段的转角与 Stage 1 基线无差异，则“习惯残留”不成立"
          "（学到的是任务特定的姿势调整）；若判据全部落在“定势”，"
          "则说明**反复失败加上自写笔记不足以更新判断**——这本身是关于现有智能体"
          "自我改进行为边界的结论。"),
    ("b", "**成本（实测依据）**：面板实测 Stage 1 两轮扫描共 17,813 次调用、记账 $42.07"
          "（≈ 1.7 分/次）；Stage 2 全量预估 18,300–28,600 次调用 ≈ **¥310–490**；"
          "工具 `tools/usage_delta.py --before/--after` 可给出每一轮的真实花费。"),
    ("p", "**参考文献（Stage 2 相关）**：[7] Shinn et al. (2023)；[8] Park et al. (2023)；"
          "[9] Wang et al. (2023)；[10] Packer et al. (2023)；[11] Metcalfe & Wiebe (1987)；"
          "[12] Wood & Rünger (2016)；[5] Luchins (1942)；[6] Bilalić et al. (2008)。"),
    ("gap", ""),

    ("h1", "三、参考文献"),
    ("p", "[1] Warren, W. H., & Whang, S. (1987). Visual guidance of walking through apertures: "
          "Body-scaled information for affordances. Journal of Experimental Psychology: Human "
          "Perception and Performance, 13(3), 371–383."),
    ("p", "[2] Keizer, A., Smeets, M. A. M., Dijkerman, H. C., Uzunbajakau, S. A., van Elburg, A., & "
          "Postma, A. (2013). Too fat to fit through the door: First evidence for disturbed "
          "body-scaled action in anorexia nervosa during locomotion. PLoS ONE, 8(5), e64602. "
          "doi:10.1371/journal.pone.0064602. **（已核实原文）**：36 次试验 = 12 档开口宽度 × 每档 3 次，"
          "A/S 自 0.9 至 2.0、步长 0.1；健康对照组在比肩宽宽 25% 时开始侧身（A/S_crit = 1.25），"
          "患者组为 40%（1.40）。本基准的 12 档阶梯与每档 5 次重复即沿用该设计。"),
    ("p", "[3] Gibson, J. J. (1979). The Ecological Approach to Visual Perception. Houghton Mifflin."),
    ("p", "[4] Fajen, B. R., & Warren, W. H. (2003). Behavioral dynamics of steering, obstacle "
          "avoidance, and route selection. Journal of Experimental Psychology: Human Perception "
          "and Performance, 29(2), 343–362."),
    ("p", "[5] Luchins, A. S. (1942). Mechanization in problem solving: The effect of Einstellung. "
          "Psychological Monographs, 54(6), 1–95."),
    ("p", "[6] Bilalić, M., McLeod, P., & Gobet, F. (2008). Why good thoughts block better ones: "
          "The mechanism of the pernicious Einstellung (set) effect. Cognition, 108(3), 652–661."),
    ("p", "[7] Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. (2023). Reflexion: "
          "Language agents with verbal reinforcement learning. Advances in Neural Information "
          "Processing Systems (NeurIPS), 36."),
    ("p", "[8] Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. "
          "(2023). Generative agents: Interactive simulacra of human behavior. Proceedings of "
          "UIST '23."),
    ("p", "[9] Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., Fan, L., & "
          "Anandkumar, A. (2023). Voyager: An open-ended embodied agent with large language "
          "models. arXiv:2305.16291."),
    ("p", "[10] Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S. G., Stoica, I., & "
          "Gonzalez, J. E. (2023). MemGPT: Towards LLMs as operating systems. arXiv:2310.08560."),
    ("p", "[11] Metcalfe, J., & Wiebe, D. (1987). Intuition in insight and noninsight problem "
          "solving. Memory & Cognition, 15(3), 238–246."),
    ("p", "[12] Wood, W., & Rünger, D. (2016). Psychology of habit. Annual Review of Psychology, "
          "67, 289–314."),
    ("p", "[13] Franchak, J. M., Celano, E. C., & Adolph, K. E. (2012). Perception of passage "
          "through openings depends on the size of the body in motion. Experimental Brain "
          "Research, 223(2), 301–310. doi:10.1007/s00221-012-3261-y. **（用于人类口径的第三种估计）**："
          "该文综述 Warren & Whang 与 Higuchi 等的旋转起始比为 1.2–1.3，并以其 0.5 cm 分辨率装置"
          "实测得 1.10，指出早期研究的开口步长较粗（2.5–13 cm）会系统性抬高阈值。"),
    ("p", "**交稿前须处理**：[5][6][11][12][13] 请按目标期刊格式核对卷期页码；"
          "所有引用的数值主张（1.2–1.3、1.25、1.10、12 ratios × 3 trials）均已逐条对照原文，"
          "其中 [2][13] 由本项目联网核实（原文摘要与方法节）。"),
]


if __name__ == "__main__":
    build(BLOCKS, OUT)
