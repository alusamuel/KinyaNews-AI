"""Render the report source, tables and measured figure to an A4 PDF."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/kinyanews-matplotlib')
from pathlib import Path
import re
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent
source = (ROOT / 'FINAL_REPORT.md').read_text()
output = ROOT / 'KinyaNews_Final_Research_Report.pdf'
figures = []
fig = None
y = 0

def new_page():
    global fig, y
    fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
    fig.text(.085, .963, 'KINYANEWS AI  /  NLP CAPSTONE RESEARCH REPORT',
             fontsize=8, color='#17665e', fontweight='bold')
    fig.add_artist(plt.Line2D([.085,.915],[.948,.948],color='#c5d8d4',linewidth=.7))
    figures.append(fig)
    y = .923


def space(height):
    if y-height < .075:
        new_page()


def paragraph(text, size=10, bold=False, color='#263238', width=96):
    global y
    lines = textwrap.wrap(text, width=width, break_long_words=True, break_on_hyphens=False)
    leading = .0168 if size <= 10 else .022
    space(len(lines)*leading+.008)
    urls = re.findall(r'https?://[^\s]+', text)
    for line in lines:
        kwargs={'url': urls[0]} if len(urls)==1 else {}
        fig.text(.085, y, line, fontsize=size, fontweight='bold' if bold else 'normal',
                 color=color, va='top', **kwargs)
        y -= leading
    y -= .008


def table(lines):
    global y
    rows = [[c.strip() for c in line.strip().strip('|').split('|')] for line in lines]
    rows = [r for r in rows if not all(re.fullmatch(r'[:\-\s]+',c) for c in r)]
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    if len(widths)==4 and widths[0]>30:
        proportions=[.52,.17,.17,.14]
    elif len(widths)==3:
        proportions=[.62,.20,.18]
    elif len(widths)==5:
        proportions=[.33,.18,.18,.16,.15]
    else:
        proportions=[.35,.15,.35,.15]
    height = len(rows)*.027
    space(height+.025)
    ax=fig.add_axes([.085,y-height,.83,height]);ax.axis('off')
    t=ax.table(cellText=rows[1:],colLabels=rows[0],colWidths=proportions,cellLoc='left',bbox=[0,0,1,1])
    t.auto_set_font_size(False);t.set_fontsize(8.7)
    for (r,c), cell in t.get_celld().items():
        cell.set_edgecolor('#d3dfdd');cell.set_linewidth(.5)
        if r==0:
            cell.set_facecolor('#17665e');cell.set_text_props(color='white',fontweight='bold')
        else:cell.set_facecolor('#f0f6f5' if r%2 else 'white')
    y-=height+.025

for page in source.split('<!-- page -->'):
    new_page()
    lines=page.strip().splitlines();i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:
            i+=1;continue
        if line.startswith('|'):
            block=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                block.append(lines[i]);i+=1
            table(block);continue
        if line.startswith('!['):
            image_path=re.search(r'\]\((.*?)\)',line).group(1)
            height=.55;space(height+.015)
            ax=fig.add_axes([.085,y-height,.83,height]);ax.imshow(plt.imread(ROOT/image_path));ax.axis('off')
            y-=height+.015;i+=1;continue
        if line.startswith('# '):
            paragraph(line[2:],size=18,bold=True,color='#153744',width=58)
            y-=.007
        elif line.startswith('## '):
            paragraph(line[3:],size=11,bold=True,color='#17665e',width=84)
        else:paragraph(line)
        i+=1

with PdfPages(output,metadata={'Title':'KinyaNews AI: Kinyarwanda News Topic Classification',
                              'Author':'Samuel RURANGAMIRWA',
                              'Subject':'NLP capstone research report; measured classification results'}) as pdf:
    for i,f in enumerate(figures,1):
        f.text(.085,.035,'KINNEWS classification study | 18 October 2026',fontsize=7,color='#66727a')
        f.text(.915,.035,f'{i} / {len(figures)}',fontsize=8,ha='right',color='#66727a')
        pdf.savefig(f);plt.close(f)
print(f'Created {output}: {len(figures)} pages')
