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
    leading = .0150 if size < 10 else (.0168 if size == 10 else .022)
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

def cover_page():
    """Dedicated vector cover; all links remain clickable in the PDF."""
    global fig
    from matplotlib.patches import Rectangle, Circle
    fig = plt.figure(figsize=(8.27, 11.69), facecolor='#f8faf8')
    figures.append(fig)
    navy, teal, muted = '#163441', '#14786e', '#5b7077'
    fig.add_artist(Rectangle((0, .972), 1, .028, transform=fig.transFigure,
                             facecolor=navy, edgecolor='none'))
    fig.add_artist(Rectangle((.085, .915), .09, .005, transform=fig.transFigure,
                             facecolor=teal, edgecolor='none'))
    fig.text(.085, .883, 'NLP CAPSTONE  /  FINAL RESEARCH REPORT', fontsize=9,
             color=muted, fontweight='bold')
    # A restrained document/feature motif, drawn as vector shapes.
    for index in range(5):
        xx = .735 + index * .027
        fig.add_artist(Rectangle((xx, .68), .012, .04 + index * .015,
                                 transform=fig.transFigure, facecolor='#d6e6e1', edgecolor='none'))
    fig.add_artist(Circle((.865, .79), .018, transform=fig.transFigure,
                          facecolor='#d1b568', edgecolor='none'))
    fig.text(.08, .785, 'KinyaNews', fontsize=47, fontweight='bold', color=navy)
    fig.text(.085, .71, 'AI', fontsize=47, fontweight='bold', color=teal)
    fig.text(.085, .638, 'Kinyarwanda News', fontsize=24, color=navy, fontweight='bold')
    fig.text(.085, .597, 'Topic Classification', fontsize=24, color=navy, fontweight='bold')
    fig.text(.085, .546, 'TF-IDF and Linear Models', fontsize=13, color=muted)
    fig.add_artist(plt.Line2D([.085, .915], [.488, .488], color='#c7d8d3', linewidth=.9))
    fig.text(.085, .446, 'PREPARED BY', fontsize=8.5, color=muted, fontweight='bold')
    fig.text(.085, .410, 'Samuel RURANGAMIRWA', fontsize=18, color=navy, fontweight='bold')
    fig.text(.085, .363, 'Text Classification for an African Language', fontsize=11, color=muted)
    fig.text(.085, .334, '18 October 2026', fontsize=11, color=muted)
    fig.text(.085, .255, 'PROJECT RESOURCES', fontsize=8.5, color=muted, fontweight='bold')
    links = [
        ('LIVE APPLICATION', 'kinya-news-ai.vercel.app', 'https://kinya-news-ai.vercel.app/'),
        ('SOURCE CODE', 'github.com/alusamuel/KinyaNews-AI', 'https://github.com/alusamuel/KinyaNews-AI'),
        ('DEMONSTRATION', 'canva.link/hhol2d33m4wq1ui', 'https://canva.link/hhol2d33m4wq1ui'),
    ]
    for index, (label, display, url) in enumerate(links):
        yy = .207 - index * .043
        fig.text(.085, yy, label, fontsize=8, color=muted, fontweight='bold')
        fig.text(.355, yy, display, fontsize=10, color=teal, url=url)
    fig.add_artist(plt.Line2D([.085,.915],[.067,.067], color='#c7d8d3', linewidth=.7))
    fig.text(.085, .043, 'KINYARWANDA  /  NATURAL LANGUAGE PROCESSING', fontsize=7.5, color=muted)


for page in source.split('<!-- page -->'):
    if page.strip().startswith('# KinyaNews AI\n'):
        cover_page()
        continue
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
        else:paragraph(line, size=9.2 if page.strip().startswith('# 11. References') else 10)
        i+=1

with PdfPages(output,metadata={'Title':'KinyaNews AI: Kinyarwanda News Topic Classification',
                              'Author':'Samuel RURANGAMIRWA',
                              'Subject':'NLP capstone research report; measured classification results'}) as pdf:
    for i,f in enumerate(figures,1):
        if i == 1:
            pdf.savefig(f); plt.close(f)
            continue
        f.text(.085,.035,'KINNEWS classification study | 18 October 2026',fontsize=7,color='#66727a')
        f.text(.915,.035,f'{i-1} / {len(figures)-1}',fontsize=8,ha='right',color='#66727a')
        pdf.savefig(f);plt.close(f)
print(f'Created {output}: {len(figures)} pages')
