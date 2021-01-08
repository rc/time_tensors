fragments = {
    'begin-document' : r"""
\documentclass[a4paper]{article}

\usepackage{graphicx}
\usepackage{siunitx}
\usepackage{booktabs}

\begin{document}
    """,

    'end-document' : r"""
\end{document}
    """,

    'section' : r"""
\{level}section{{{name}}}
\label{{{label}}}
    """,

    'center' : r"""
\begin{{center}}
{text}
\end{{center}}
    """,

    'figure' : r"""
\begin{{figure}}[htp!]
  \centering
    \includegraphics[width={width}\linewidth]{{{path}}}
  \caption{{{caption}}}
  \label{{{label}}}
\end{{figure}}
    """,

    'newline' : r"""
\\
""",

    'newpage' : r"""
\clearpage
""",
}
