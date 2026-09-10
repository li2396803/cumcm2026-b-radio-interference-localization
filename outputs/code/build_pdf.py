#!/usr/bin/env python3
"""把说明文档 Markdown 编译为 PDF(LaTeX/xelatex)。

流程: pandoc(Markdown -> LaTeX, ctexart + xeCJK) -> 表格/代码块排版补丁 -> latexmk -xelatex
用法: python3 build_pdf.py [文档.md 的路径]   (默认取同级上一级目录的同名 .md)
依赖: pandoc, MacTeX/TeX Live(xelatex, latexmk, ctex, fvextra)
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# 文本中出现的 Unicode 符号 -> LaTeX 数学命令(避免字体缺字)
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


def sh(cmd, **kw):
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def patch_tex(tex_path: str) -> None:
    t = open(tex_path, encoding="utf-8").read()
    # 1) 表格与代码块排版: 表内 \small + 缩小列距; 代码块自动折行
    t = t.replace(
        "\\usepackage{longtable,booktabs,array}",
        "\\usepackage{longtable,booktabs,array}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{etoolbox}\n"
        "\\usepackage{fvextra}\n"
        "\\fvset{breaklines=true,breakanywhere=true,fontsize=\\footnotesize}\n"
        "\\setlength{\\emergencystretch}{3em}\n"
        "\\tolerance=2500\n"
        "\\AtBeginEnvironment{longtable}{\\small\\setlength{\\tabcolsep}{4pt}}\n"
        "\\setmonofont{Menlo}\n"
        "\\setCJKmonofont{PingFang SC}\n"
        "\\usepackage{newunicodechar}\n" + UNICODE_MAP,
    )
    # 2) A4 纸张
    t = t.replace("\\usepackage[margin=2.3cm]{geometry}",
                  "\\usepackage[a4paper,margin=2.2cm]{geometry}")
    # 3) 自然宽度(l)列改为按比例分配的 p 列, 保证中文表格不越界
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
    # 4) 宽表(>=7 列)使用 \footnotesize
    def fix_wide(m):
        spec = m.group(1)
        if spec.count(">{\\raggedright") >= 7 and "\\footnotesize" not in spec:
            spec = spec.replace(">{\\raggedright\\arraybackslash}",
                                ">{\\raggedright\\arraybackslash\\footnotesize}")
        return "\\begin{longtable}[]{@{}" + spec + "@{}}"

    t = re.sub(r"\\begin\{longtable\}\[\]\{@\{\}(.*?)@\{\}\}", fix_wide, t, flags=re.S)
    # 5) verbatim 代码块改为可断行的 Verbatim
    t = t.replace("\\begin{verbatim}",
                  "\\begin{Verbatim}[breaklines=true,breakanywhere=true,fontsize=\\small]")
    t = t.replace("\\end{verbatim}", "\\end{Verbatim}")
    open(tex_path, "w", encoding="utf-8").write(t)
    print("patched:", tex_path)


def main():
    if len(sys.argv) > 1:
        md = os.path.abspath(sys.argv[1])
    else:
        md = os.path.abspath(os.path.join(HERE, "..", "B题_建模与求解说明文档.md"))
    tex = os.path.splitext(md)[0] + ".tex"
    workdir = os.path.dirname(md)
    sh(["pandoc", os.path.basename(md), "-s", "-o", os.path.basename(tex), "-t", "latex",
        "--toc", "--toc-depth=3",
        "-V", "documentclass=ctexart", "-V", "CJKmainfont=PingFang SC",
        "-V", "geometry:margin=2.3cm", "-V", "fontsize=11pt",
        "-V", "colorlinks=true", "-V", "linkcolor=blue", "-V", "urlcolor=blue"],
       cwd=workdir)
    patch_tex(tex)
    sh(["latexmk", "-norc", "-xelatex", "-interaction=nonstopmode", "-halt-on-error",
        "-synctex=1", os.path.basename(tex)], cwd=workdir)
    print("PDF:", os.path.splitext(md)[0] + ".pdf")


if __name__ == "__main__":
    main()
