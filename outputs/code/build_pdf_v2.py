#!/usr/bin/env python3
"""说明文档 → PDF(LaTeX/xelatex) 排版优化版 v2。

与 v1 相比的版式改进:
  * 正文字号 11pt → 12pt; 行距 1.35; 页边距 2.5cm; A4
  * 中文正文宋体(Songti SC) + 标题/代码无衬线(PingFang SC) + 西文 Times New Roman
  * 标题彩色分级、题注加粗、表格行高与书签线加强、代码块左侧强调线
  * 图片统一居中并占 92% 行宽, 公式上下留白加大
用法: python3 build_pdf_v2.py [文档.md] [--style v1|v2]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

UNICODE_MAP = "".join(
    "\\newunicodechar{%s}{\\ensuremath{%s}}\n" % (ch, cmd)
    for ch, cmd in [
        ("≤", "\\le"), ("≥", "\\ge"), ("≈", "\\approx"), ("≠", "\\neq"),
        ("×", "\\times"), ("±", "\\pm"), ("·", "\\cdot"), ("∼", "\\sim"),
        ("→", "\\rightarrow"), ("↔", "\\leftrightarrow"), ("⇒", "\\Rightarrow"),
        ("∈", "\\in"), ("∅", "\\varnothing"), ("∪", "\\cup"), ("∩", "\\cap"),
        ("⊂", "\\subset"), ("⊆", "\\subseteq"), ("⊇", "\\supseteq"),
        ("∎", "\\blacksquare"), ("√", "\\surd"), ("▼", "\\blacktriangledown"),
        ("⁻", "^{-}"), ("¹", "^{1}"), ("²", "^{2}"), ("³", "^{3}"), ("⁶", "^{6}"),
        ("θ", "\\theta"), ("α", "\\alpha"), ("β", "\\beta"), ("γ", "\\gamma"),
        ("δ", "\\delta"), ("ε", "\\varepsilon"), ("η", "\\eta"), ("λ", "\\lambda"),
        ("μ", "\\mu"), ("π", "\\pi"), ("ρ", "\\rho"), ("σ", "\\sigma"),
        ("τ", "\\tau"), ("φ", "\\varphi"), ("ω", "\\omega"), ("Δ", "\\Delta"),
        ("Σ", "\\Sigma"), ("Ω", "\\Omega"), ("°", "^\\circ"), ("⁄", "/"),
    ])

STYLE_V2 = r"""
% ================= 版式优化(v2) =================
\geometry{a4paper,margin=2.5cm}  % 覆盖 pandoc 默认页边距
\definecolor{Accent}{HTML}{1F4E79}
\definecolor{Accent2}{HTML}{2E75B6}
\definecolor{ShadeGray}{HTML}{F2F5F9}
\usepackage{setspace}\setstretch{1.35}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf,skip=8pt,singlelinecheck=false}
\renewcommand{\arraystretch}{1.22}
\setlength{\parindent}{2em}
\setlength{\parskip}{0.28em}
\setlength{\abovedisplayskip}{10pt plus 3pt minus 2pt}
\setlength{\belowdisplayskip}{10pt plus 3pt minus 2pt}
\setlength{\abovedisplayshortskip}{6pt}\setlength{\belowdisplayshortskip}{6pt}
\ctexset{
  section/format = \Large\bfseries\sffamily\color{Accent},
  section/beforeskip = 1.6em plus .3em minus .2em,
  section/afterskip = 0.9em plus .2em,
  subsection/format = \large\bfseries\sffamily\color{Accent},
  subsection/beforeskip = 1.2em plus .3em minus .2em,
  subsection/afterskip = 0.6em plus .2em,
  subsubsection/format = \normalsize\bfseries\sffamily\color{Accent2},
  subsubsection/beforeskip = 1.0em plus .3em minus .2em,
  subsubsection/afterskip = 0.5em plus .2em,
}
\fvset{breaklines=true,breakanywhere=true,fontsize=\small,
       frame=leftline,framerule=2.5pt,rulecolor=\color{Accent2},
       framesep=8pt,baselinestretch=1.05}
\setkeys{Gin}{width=0.92\linewidth,keepaspectratio}
\makeatletter
\@ifpackageloaded{booktabs}{}{\usepackage{booktabs}}
\makeatother
\widowpenalty=10000 \clubpenalty=10000
% ==============================================
"""


def sh(cmd, **kw):
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def patch_tex(tex_path: str, style: str) -> None:
    t = open(tex_path, encoding="utf-8").read()
    t = t.replace(
        "\\usepackage{longtable,booktabs,array}",
        "\\usepackage{longtable,booktabs,array}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{etoolbox}\n"
        "\\usepackage{fvextra}\n"
        "\\setlength{\\emergencystretch}{3em}\n"
        "\\tolerance=2500\n"
        "\\AtBeginEnvironment{longtable}{\\small\\setlength{\\tabcolsep}{4pt}}\n"
        "\\setmonofont{Menlo}\n"
        "\\setCJKmonofont{PingFang SC}\n"
        "\\usepackage{newunicodechar}\n" + UNICODE_MAP,
    )

    def fix_plain(m):
        spec = m.group(1)
        if "p{" in spec or "c" in spec:
            return m.group(0)
        n = spec.count("l")
        if n < 2:
            return m.group(0)
        frac = 1.0 / n
        new = "".join(
            ">{\\raggedright\\arraybackslash}p{(\\linewidth - %d\\tabcolsep) * \\real{%.4f}}"
            % (2 * (n - 1), frac) for _ in range(n))
        return "\\begin{longtable}[]{@{}%s@{}}" % new

    t = re.sub(r"\\begin\{longtable\}\[\]\{@\{\}(l+)@\{\}\}", fix_plain, t)

    def fix_wide(m):
        spec = m.group(1)
        if spec.count(">{\\raggedright") >= 7 and "\\footnotesize" not in spec:
            spec = spec.replace(">{\\raggedright\\arraybackslash}",
                                ">{\\raggedright\\arraybackslash\\footnotesize}")
        return "\\begin{longtable}[]{@{}" + spec + "@{}}"

    t = re.sub(r"\\begin\{longtable\}\[\]\{@\{\}(.*?)@\{\}\}", fix_wide, t, flags=re.S)
    t = t.replace("\\begin{verbatim}",
                  "\\begin{Verbatim}[breaklines=true,breakanywhere=true]")
    t = t.replace("\\end{verbatim}", "\\end{Verbatim}")

    if style == "v2":
        t = t.replace("\\begin{document}", STYLE_V2 + "\n\\begin{document}", 1)
    open(tex_path, "w", encoding="utf-8").write(t)
    print("patched:", tex_path, "style:", style)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("md", nargs="?", default=os.path.join(HERE, "..",
                                                          "B题_建模与求解说明文档_v2.md"))
    ap.add_argument("--style", default="v2", choices=["v1", "v2"])
    a = ap.parse_args()
    md = os.path.abspath(a.md)
    tex = os.path.splitext(md)[0] + ".tex"
    workdir = os.path.dirname(md)
    sh(["pandoc", os.path.basename(md), "-s", "-o", os.path.basename(tex), "-t", "latex",
        "--toc", "--toc-depth=3",
        "-V", "documentclass=ctexart", "-V", "CJKmainfont=PingFang SC",
        "-V", "geometry:margin=2.3cm", "-V", "fontsize=12pt",
        "-V", "colorlinks=true", "-V", "linkcolor=NavyBlue", "-V", "urlcolor=NavyBlue"],
       cwd=workdir)
    patch_tex(tex, a.style)
    sh(["latexmk", "-norc", "-xelatex", "-interaction=nonstopmode", "-halt-on-error",
        "-synctex=1", os.path.basename(tex)], cwd=workdir)
    print("PDF:", os.path.splitext(md)[0] + ".pdf")


if __name__ == "__main__":
    main()
