import os
import shutil
import subprocess

# Paths
notebook_path = '/home/captc/Devwork/Msc/Ethics/AI_G&E_Term_Paper.ipynb'
temp_notebook_path = '/home/captc/Devwork/Msc/Ethics/temp_notebook.ipynb'
temp_html_path = '/home/captc/Devwork/Msc/Ethics/temp_notebook.html'
pdf_path = '/home/captc/Devwork/Msc/Ethics/AI_G&E_Term_Paper.pdf'
jupyter_bin = '/home/captc/Devwork/Msc/.venv/bin/jupyter'

def main():
    print("Step 1: Copying notebook to temporary file...")
    # We copy the entire notebook to temp_notebook.ipynb without deleting any cells
    # This preserves all sections, including the Title and Introduction.
    shutil.copy(notebook_path, temp_notebook_path)

    print("Step 2: Converting notebook to HTML via nbconvert...")
    nbconvert_cmd = [
        jupyter_bin,
        'nbconvert',
        '--to', 'html',
        temp_notebook_path,
        '--output', temp_html_path
    ]
    subprocess.run(nbconvert_cmd, check=True)

    print("Step 3: Patching HTML (MathJax config and custom CSS)...")
    with open(temp_html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace MathJax configuration from CHTML-full to SVG-full for vector rendering
    content = content.replace('config=TeX-AMS_CHTML-full,Safe', 'config=TeX-AMS_SVG-full,Safe')

    # Add custom styles for print layout: margins, centered equations, and readable Mermaid diagrams
    custom_css = """
<style>
@page {
    size: letter;
    margin-top: 0.5in !important;
    margin-bottom: 0.5in !important;
    margin-left: 0.25in !important;
    margin-right: 0.5in !important;
}
body {
    margin: 0 !important;
    padding: 0 !important;
    background-color: transparent !important;
}
#notebook, #notebook-container, .container, .jp-Notebook {
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    box-shadow: none !important;
}
.cell, .jp-Cell {
    width: 100% !important;
    margin: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
}
.input_area, .jp-InputArea {
    width: 100% !important;
}
.output_area, .jp-OutputArea {
    width: 100% !important;
}
.MathJax_SVG_Display {
    text-align: center !important;
    margin: 1em 0 !important;
}
.MathJax_SVG, .MathJax_SVG *, mjx-container, mjx-container *, .MathJax, .MathJax * {
    font-weight: normal !important;
}
/* Style the rendered Mermaid diagram */
.jp-RenderedMermaid {
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
    display: flex !important;
}
.jp-RenderedMermaid figure {
    width: 100% !important;
    max-width: 850px !important;
    margin: 1.5em auto !important;
    display: block !important;
}
.jp-RenderedMermaid svg {
    width: 100% !important;
    max-width: 850px !important;
    height: auto !important;
    display: block !important;
    margin: 0 auto !important;
}
</style>
"""
    content = content.replace('</head>', custom_css + '\n</head>')

    old_segment = """svg = cleanMermaidSvg(svg);
        results = makeMermaidImage(svg);
        output = document.createElement("figure");
        results.map(output.appendChild, output);"""
    new_segment = """output = document.createElement("figure");
        output.innerHTML = svg;"""
    content = content.replace(old_segment, new_segment)

    # Rename class in HTML and JS to prevent Mermaid auto-rendering race conditions
    content = content.replace('<pre class="mermaid">', '<pre class="mermaid-raw">')
    content = content.replace('document.querySelectorAll(".jp-Mermaid > pre.mermaid")', 'document.querySelectorAll(".jp-Mermaid > pre.mermaid-raw")')

    with open(temp_html_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("Step 4: Compiling PDF via Headless Chrome...")
    chrome_cmd = [
        'google-chrome',
        '--headless',
        '--disable-gpu',
        '--no-sandbox',
        '--no-pdf-header-footer',
        '--virtual-time-budget=15000',
        f'--print-to-pdf={pdf_path}',
        temp_html_path
    ]
    subprocess.run(chrome_cmd, check=True)
    print("PDF successfully compiled!")

    print("Step 5: Cleaning up temporary files...")
    if os.path.exists(temp_notebook_path):
        os.remove(temp_notebook_path)
    if os.path.exists(temp_html_path):
        os.remove(temp_html_path)
    
    # Also remove any test_notebook.html or test files if they exist
    test_html = '/home/captc/Devwork/Msc/Ethics/test_notebook.html'
    if os.path.exists(test_html):
        os.remove(test_html)

    print("Cleanup completed successfully!")

if __name__ == '__main__':
    main()
