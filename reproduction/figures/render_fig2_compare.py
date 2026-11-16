import warnings; warnings.filterwarnings('ignore')
import fitz, os
base = r'D:\EmInAI\EmbodiedBAO(AS)\analysis\_v67_tex\figtest'
for n in ['where10.pdf', 'Fig2_revised.pdf']:
    d = fitz.open(os.path.join(base, n))
    pg = d[0]
    print(n, 'pages', d.page_count, 'rect', pg.rect, 'text:', repr(pg.get_text()[:400]))
    pix = pg.get_pixmap(dpi=220)
    out = os.path.join(r'D:\EmInAI\EmbodiedBAO(AS)\analysis', 'figtest_' + n.replace('.pdf', '') + '.png')
    pix.save(out)
    print('saved', out, pix.width, pix.height)
    d.close()
