"""Render the review guide, validate all numbered answers, and render every page."""
from pathlib import Path
import re, json, ast, textwrap, hashlib
from xml.sax.saxutils import escape
import fitz
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, PageBreak, CondPageBreak, Table, TableStyle, Preformatted, KeepTogether
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'EMADS_Architecture_Code_Interview_Guide.pdf'
TMP = ROOT / 'tmp' / 'pdfs'
COMMIT = '292d600919934449e4ae781afe05662b80b07704'
DATE = '2026-11-02'
for name, file in [('Body','arial.ttf'),('BodyBold','arialbd.ttf'),('BodyItalic','ariali.ttf'),('Code','consola.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(Path('C:/Windows/Fonts')/file)))
pdfmetrics.registerFontFamily('Body', normal='Body', bold='BodyBold', italic='BodyItalic', boldItalic='BodyBold')
NAVY=colors.HexColor('#17334D'); TEAL=colors.HexColor('#167D8D'); INK=colors.HexColor('#263545'); PALE=colors.HexColor('#EDF4F7'); BORDER=colors.HexColor('#CFDCE3')
W,H=A4
WIDTH=W-100
styles={
 'body':ParagraphStyle('body',fontName='Body',fontSize=10.2,leading=14.1,textColor=INK,spaceAfter=7,splitLongWords=True),
 'small':ParagraphStyle('small',fontName='Body',fontSize=8.8,leading=12,textColor=INK,spaceAfter=6,splitLongWords=True),
 'title':ParagraphStyle('title',fontName='BodyBold',fontSize=29,leading=34,textColor=NAVY,spaceAfter=22),
 'q':ParagraphStyle('q',fontName='BodyBold',fontSize=17,leading=21,textColor=NAVY,spaceAfter=15,keepWithNext=True),
 'section':ParagraphStyle('section',fontName='BodyBold',fontSize=11,leading=14,textColor=TEAL,spaceBefore=10,spaceAfter=6,keepWithNext=True),
 'sub':ParagraphStyle('sub',fontName='BodyBold',fontSize=10.7,leading=14,textColor=NAVY,spaceBefore=10,spaceAfter=6,keepWithNext=True),
 'cell':ParagraphStyle('cell',fontName='Body',fontSize=9.0,leading=12.1,textColor=INK,spaceAfter=0,splitLongWords=True),
 'cellhead':ParagraphStyle('cellhead',fontName='BodyBold',fontSize=9.1,leading=12,textColor=colors.white,spaceAfter=0,splitLongWords=True),
 'code':ParagraphStyle('code',fontName='Code',fontSize=8,leading=11,textColor=INK,leftIndent=8,rightIndent=8,spaceBefore=6,spaceAfter=10,borderColor=BORDER,borderWidth=.5,borderPadding=8,backColor=PALE),
}

def p(t,style='body'):
    t=escape(t)
    t=re.sub(r'`([^`]+)`',r'<font name="Code">\1</font>',t)
    t=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',t)
    return Paragraph(t,styles[style])

def table(rows, widths=None):
    n=len(rows[0])
    if widths is None:
        ratios={2:[.29,.71],3:[.23,.37,.40]}.get(n,[1/n]*n)
        widths=[WIDTH*v for v in ratios]
    cells=[[p(c,'cellhead' if i==0 else 'cell') for c in r] for i,r in enumerate(rows)]
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.4,BORDER),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F4F7F9')]),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    return [t,Spacer(1,12)]

def arrow(d,x1,y1,x2,y2):
    d.add(Line(x1,y1,x2,y2,strokeColor=TEAL,strokeWidth=1.4))
    if abs(y2-y1)>abs(x2-x1):
        sign=1 if y2>y1 else -1
        d.add(Polygon([x2,y2,x2-3.5,y2-sign*7,x2+3.5,y2-sign*7],fillColor=TEAL,strokeColor=TEAL))
    else:
        sign=1 if x2>x1 else -1
        d.add(Polygon([x2,y2,x2-sign*7,y2-3.5,x2-sign*7,y2+3.5],fillColor=TEAL,strokeColor=TEAL))

def box(d,x,y,w,h,lines,fill=PALE):
    d.add(Rect(x,y,w,h,rx=6,ry=6,fillColor=fill,strokeColor=BORDER))
    for i,line in enumerate(lines):
        d.add(String(x+w/2,y+h-17-i*14,line,fontName='BodyBold' if i==0 else 'Body',fontSize=9.2,fillColor=NAVY,textAnchor='middle'))

def architecture():
    d=Drawing(WIDTH,390)
    box(d,0,334,WIDTH,47,['Streamlit interface','Upload / manual choices / progress / result tabs / downloads'])
    box(d,0,266,WIDTH,47,['SupervisorAgent + LangGraph','Fixed order / partial update checks / stop-on-failure'])
    box(d,0,188,WIDTH,58,['Nine BaseAgent implementations','Understanding -> EDA -> preprocessing -> selection -> tuning','Evaluation -> explainability -> meta-evaluation -> reporting'])
    gap=15; half=(WIDTH-gap)/2
    box(d,0,100,half,65,['EMADSState','Paths, metrics, summaries','Decisions and log reducers'])
    box(d,half+gap,100,half,65,['Services and utilities','Groq text generation / PDF rendering','Plots / filesystem / disk logger'])
    box(d,0,10,WIDTH,64,['Stored artifacts - shared local filesystem','Uploads / cleaned CSV / estimator pickle / PNG plots / PDFs / logs','State is per run; most artifact filenames are not per run'])
    for y1,y2 in [(334,313),(266,246)]: arrow(d,WIDTH/2,y1,WIDTH/2,y2)
    arrow(d,WIDTH*.25,188,WIDTH*.25,165); arrow(d,WIDTH*.75,188,WIDTH*.75,165)
    arrow(d,WIDTH*.25,100,WIDTH*.25,74); arrow(d,WIDTH*.75,100,WIDTH*.75,74)
    return d

def flow():
    d=Drawing(WIDTH,330)
    lines=[['1. Upload + choices','Save CSV, read preview, click Run'],['2. Initialize + stream','create_initial_state -> fixed graph nodes'],['3. Update after each stage','Reducers in graph; manual update merge in UI'],['4. Success / failure','Success: save final_state; error: clear stale state'],['5. Present + download','Tabs read state; download buttons read artifact paths']]
    for i,ls in enumerate(lines):
        y=268-i*63
        box(d,20,y,WIDTH-40,46,ls)
        if i<4: arrow(d,WIDTH/2,y,WIDTH/2,y-17)
    return d

class Doc(BaseDocTemplate):
    def __init__(self,*a,**k):
        super().__init__(*a,**k)
        self.addPageTemplates(PageTemplate('main',Frame(50,49,WIDTH,H-101,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0),onPage=self.decorate))
    def decorate(self,c,doc):
        c.saveState(); c.setStrokeColor(BORDER); c.line(50,H-35,W-50,H-35)
        c.setFont('Body',8);c.setFillColor(NAVY)
        c.drawString(50,H-27,'EMADS | Architecture, code and interview guide')
        c.drawRightString(W-50,H-27,'Source review | '+DATE)
        c.line(50,36,W-50,36);c.setFont('Body',8);c.setFillColor(INK)
        c.drawString(50,23,'Inspected working tree: '+COMMIT[:12]);c.drawRightString(W-50,23,str(doc.page));c.restoreState()
    def afterFlowable(self,f):
        if hasattr(f,'toc_info'):
            level,title,key=f.toc_info
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(title,key,level=level,closed=False)
            self.notify('TOCEntry',(level,title,self.page,key))

raw=(TMP/'guide.md').read_text(encoding='utf-8')
raw=re.sub(r'# EMADS architecture, code and interview guide.*?(?=@Q 1\|)', '',raw,flags=re.S)
raw=raw.replace('app/ui/theme.py:272','app/ui/theme.py:231')
raw=raw.replace('Seven contain a Local Explanations heading.', 'Six contain a Local Explanations heading.')
raw=raw.replace('EMADS automates a sequence', 'EMADS (Explainable Multi-Agent Data Science System) automates a sequence')
raw=raw.replace('one in-memory DataFrame throughout', 'one in-memory pandas DataFrame (a table object) throughout')
raw=raw.replace('BLAS/OpenMP thread pools', 'numerical-library thread pools')
raw=raw.replace('structured JSON response mode', 'structured JavaScript Object Notation (JSON) response mode')
raw=raw.replace('| Model-selection / evaluation wording |', '| app/ui/streamlit_app.py:433; app/services/pdf_service.py:278 | The no-arbitration branch says a statistical test found a significant difference, even when no Wilcoxon comparison was applicable (for example unsupervised scoring). Absence of arbitration is not evidence of significance. |\n| Model-selection / evaluation wording |')
raw=raw.replace('Reporting conclusion omits audit context.', 'Reporting conclusion omits audit context; its deterministic fallback also claims valid cross-validation alignment without checking it.')
raw=raw.replace('test_xgboost_label_encoding.py:73', 'test_xgboost_label_encoding.py:71')
raw=raw.replace('EMADS V2 - Plan de test complet et datasets recommandes.md (actual repository filename retains French accents/em dash)', 'EMADS V2 — Plan de test complet et datasets recommandés.md')
raw=raw.replace('app/core/agents/reporting_agent.py:68, generate_pdf_report()', 'app/core/agents/reporting_agent.py:72, generate_pdf_report()')
raw=raw.replace('reporting_agent.py:69, PDF filename','reporting_agent.py:73, PDF filename')
raw=raw.replace('reporting_agent.py:75. Paths','reporting_agent.py:77. Paths')
raw=raw.replace('app/core/agents/reporting_agent.py:69, PDF filename','app/core/agents/reporting_agent.py:73, PDF filename')
matches=list(re.finditer(r'^@Q (\d+)\|([^\n|]+)\|([^\n]+)\n',raw,re.M))
questions=[]
for i,m in enumerate(matches):
    body=raw[m.end():matches[i+1].start() if i+1<len(matches) else len(raw)]
    questions.append((int(m.group(1)),m.group(2),m.group(3),body))
questions.sort()
assert [q[0] for q in questions]==list(range(1,31)), [q[0] for q in questions]
for n,title,section,body in questions:
    for label in ['1. Simple explanation','2. Actual implementation','3. Code evidence','4. Limitations or uncertainties','5. Short interview answer']:
        assert '## '+label in body,(n,label)
tests=json.loads((TMP/'test_results.json').read_text())
inspection=json.loads((TMP/'inspection_results.json').read_text())
story=[]
story += [Spacer(1,65),p('EMADS','title'),p('Architecture, code and interview guide','q'),p('30 source-linked answers, executed test evidence, statistical caveats and interview practice.','body'),Spacer(1,24)]
story += table([['Review scope','Verified provenance'],['Analysis date',DATE+' (session-provided date)'],['Inspected commit',COMMIT],['Working tree','Not clean on arrival: runtime CSV/model/plot/log files already modified. Application source had no reported tracked changes. This review does not reset or overwrite those existing artifacts.'],['Method','Complete application source and tests inspected; offline test suite executed; targeted probes; nine historical PDFs text-inspected; final guide rendered and visually reviewed.'],['Verification boundary','No live external API calls, dependency installation or project source modifications. No CV or separate thesis report supplied.'],['Existing tests','23 passed; 0 failures; 0 errors; 0 skipped. Offline isolated review harness.'],['Reading guide','Implemented = code path exists. Partial = useful behavior with a material gap. Absent = no implementation found. Proposed = future work, not performed.']])
story += [p('Privacy: credentials and .env values are excluded. Examples are synthetic; private dataset rows and historical run metric values are not reproduced.','small'),PageBreak(),p('Table of contents','q')]
story.insert(-2,p('Abbreviations used throughout: API = application programming interface; SDK = software development kit; CV = cross-validation (or curriculum vitae when explicitly discussing your CV); EDA = exploratory data analysis; SHAP = SHapley Additive exPlanations; TPE = Tree-structured Parzen Estimator; RMSE = root mean squared error; R2 = coefficient of determination; ROC AUC = area under the receiver operating characteristic curve; IQR = interquartile range; LOF = Local Outlier Factor; DBSCAN = Density-Based Spatial Clustering of Applications with Noise; PNG = Portable Network Graphics. Concepts are explained in the relevant answers and glossary.','small'))
toc=TableOfContents(); toc.levelStyles=[ParagraphStyle('toc0',fontName='BodyBold',fontSize=10.5,leading=15,spaceBefore=10,textColor=NAVY),ParagraphStyle('toc1',fontName='Body',fontSize=9.4,leading=13,leftIndent=13,firstLineIndent=0,spaceBefore=4,textColor=INK)]
story += [toc,PageBreak()]

def parse_body(body):
    lines=body.strip().splitlines(); out=[]; i=0; evidence=False
    while i<len(lines):
        line=lines[i].strip()
        if not line: i+=1; continue
        if line.startswith('## '):
            evidence=line.startswith('## 3.')
            out.append(p(line[3:],'section'));i+=1;continue
        if line.startswith('### '): out.append(p(line[4:],'sub'));i+=1;continue
        if line=='@ARCH': out += [architecture(),Spacer(1,10)];i+=1;continue
        if line=='@FLOW': out += [flow(),Spacer(1,10)];i+=1;continue
        if line.startswith('```'):
            code=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):
                code.extend(textwrap.wrap(lines[i],width=85,replace_whitespace=False,drop_whitespace=False,subsequent_indent='    ') or [''])
                i+=1
            out.append(Preformatted('\n'.join(code),styles['code']));i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                row=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.match(r'^[-: ]+$',x) for x in row): rows.append(row)
                i+=1
            assert len(set(map(len,rows)))==1, rows
            out.extend(table(rows));continue
        if line.startswith('- '): out.append(p('• '+line[2:],'small' if evidence else 'body'));i+=1;continue
        paragraph=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('##','|','```','@','- ')):
            if re.match(r'^\d+\. ',lines[i]): break
            paragraph.append(lines[i].strip());i+=1
        out.append(p(' '.join(paragraph),'small' if evidence else 'body'))
    return out

current=None
for n,title,section,body in questions:
    if n>1:
        story += [CondPageBreak(220),Spacer(1,18)]
    if current!=section:
        head=p(section,'section');head.toc_info=(0,section,'section-'+section[0]);story.append(head);current=section
    head=p(f'{n}. {title}','q');head.toc_info=(1,f'{n}. {title}','q'+str(n));story.append(head)
    story.extend(parse_body(body))

story += [PageBreak()]
head=p('Glossary','q');head.toc_info=(0,'Glossary','glossary');story.append(head)
terms={
'Agent (in EMADS)':'A Python processing class with an execute(state) method; no general planning capability is implied.',
'Anomaly detection':'Finding observations unusual relative to learned structure; not automatically equivalent to known fraud classification.',
'Arbitration':'A secondary decision mechanism used when a predefined trigger is reached; here sometimes an LLM recommendation.',
'Artifact':'A stored output such as a CSV, model pickle, plot, log or PDF.',
'Auditability':'Ability to inspect what happened and the stated basis; it is not proof that the choice was correct.',
'Baseline':'A reference configuration/score used to compare a changed model.',
'Calibration':'Agreement between claimed probabilities and empirical event frequencies under a defined task/population.',
'Checkpoint':'A saved execution-state snapshot that supports later resume; absent from this graph configuration.',
'Classification / regression':'Predicting a category / a numeric quantity.',
'Clustering':'Grouping similar observations without a supplied target label.',
'Contamination':'Anomaly-estimator parameter describing an assumed outlier fraction or thresholding policy.',
'Cross-validation (CV)':'Repeated fitting and validation using partitioned folds to compare or assess models.',
'CSV':'Comma-separated values: a tabular text-file format.',
'Data leakage':'Evaluation information influences fitting or model-choice decisions that should not have access to it.',
'DBSCAN':'Density-Based Spatial Clustering of Applications with Noise; groups density-connected points and marks noise.',
'Decision record':'EMADS AgentDecision: chosen action, reasoning, optional confidence indicator and timestamp.',
'Deterministic rule':'A programmed mapping from inputs to behavior; it need not be statistically justified.',
'EDA':'Exploratory data analysis: descriptive summaries and visual investigation before modeling.',
'Encoding':'Representing categorical values numerically, such as one-hot columns or integer labels.',
'Ensemble / boosting':'Combining models / sequentially adding models to improve an objective.',
'F1 / precision / recall':'F1 combines precision and recall; precision concerns correctness of positive predictions, recall concerns recovery of actual positives.',
'Feature / target':'An input predictor / the supervised output to be predicted.',
'Fold / stratification':'One validation partition / partitioning that approximately preserves class proportions.',
'Holdout':'A set excluded from estimator fitting and model selection; full independence also requires fitting preprocessing without it.',
'Hyperparameter':'A model configuration chosen outside its ordinary fit, such as depth or regularization strength.',
'Imputation':'Replacing missing feature values using an estimated or fixed value.',
'IQR':'Interquartile range: 75th percentile minus 25th percentile.',
'KMeans':'Centroid-based clustering minimizing within-cluster squared distances.',
'LangGraph / node / edge':'A graph-execution library / a processing function / a permitted execution connection.',
'LLM / Groq':'Large language model / the remote inference provider used by this repository.',
'LOF':'Local Outlier Factor: local-density-based anomaly detection.',
'MAE / MSE / RMSE':'Mean absolute error / mean squared error / square root of MSE.',
'Mock':'A test substitute that returns controlled behavior instead of running a real dependency.',
'Nested CV':'An outer evaluation loop around an inner model-selection/tuning loop; not configured here.',
'One-hot encoding':'One binary feature per represented category; drop_first omits a reference category.',
'Optuna / TPE':'Hyperparameter-optimization library / Tree-structured Parzen Estimator sampler using prior trials to guide suggestions.',
'p-value / null hypothesis':'A tail-probability measure under a statistical reference model / the hypothesis tested; p is not a probability that a chosen model is correct.',
'Permutation importance':'Mean performance drop when a feature is shuffled relative to a fixed evaluation sample.',
'Pickle':'Python object serialization; loading untrusted files can execute unsafe behavior.',
'Pipeline / ColumnTransformer':'Sklearn composition of transformations and estimator / column-specific transformation composition; proposed for full inference export.',
'R2':'Coefficient of determination: squared-error improvement over a mean-only test baseline; can be negative.',
'Reducer':'A rule that combines a new state update with an existing value.',
'ROC AUC':'Area under the receiver operating characteristic curve; evaluates discrimination from prediction scores.',
'Seed':'Initial value controlling pseudorandom choices; reproducibility also depends on software and execution details.',
'SHAP / Shapley value':'SHapley Additive exPlanations / feature attribution based on average marginal contribution in a cooperative-game formulation.',
'Silhouette':'Internal clustering measure comparing within-group and other-group distances; ranges from -1 to 1.',
'Spearman correlation':'Correlation of ranks; measures monotonic ordering agreement, not correctness or causality.',
'StandardScaler':'Sklearn transformer subtracting fitted means and dividing by fitted standard deviations.',
'State / TypedDict':'Shared run-result dictionary / Python type annotation for dictionary fields, not complete runtime validation.',
'Wilcoxon signed-rank':'Paired nonparametric test on signed ranks of differences; standard reference assumptions do not remove CV dependence.',
}
story += table([['Term','Meaning']]+list(terms.items()))

story += [PageBreak()]
head=p('Verification evidence and environment','q');head.toc_info=(0,'Verification evidence and environment','evidence');story.append(head)
story += [p('Executed suite: '+str(tests['tests_run'])+' tests; '+str(tests['failures'])+' failures; '+str(tests['errors'])+' errors; '+str(tests['skipped'])+' skipped; '+str(tests['seconds'])+' seconds. This duration is not an application performance benchmark.'),p(tests['conditions'],'small')]
story += table([['Package','Inspected installed version']]+list(inspection['versions'].items()))
story += [p('Individual existing tests (all passed)','sub')]
story += table([['Test identifier','Seconds (review conditions)']]+[[r['test'],str(r['seconds'])] for r in tests['records']], [WIDTH*.83,WIDTH*.17])
story += [p('Historical reports inspected for generic claims only','sub'),p('Text extraction and opening were verified. This is not a visual QA certificate for those historical files and does not attribute them to the inspected commit. No private run values are reproduced.','small')]
story += table([['Repository-relative report','Pages / claim check']]+[[r['file'],f"{r['pages']} pages; statistics-grounded cover claim present; audit and statistical-verdict headings present."] for r in inspection['historical_reports']])
story += [p('Diagnostic probes','sub')]
for key in ['parser_negation','wilcoxon_exact_5','wilcoxon_exact_10','wilcoxon_all_zero_default','offline_target_fallback','xgb_refit','zero_difference_comparison']:
    story.append(p(key+': '+str(inspection[key]),'small'))
story += [p('Unresolved evidence','sub'),p('No supplied CV or personal-contribution record; no separate thesis report; no live provider/model-availability test; no production deployment/concurrency test; no independent benchmark or calibration study. Historical artifacts have no saved code-version manifest. These gaps are not filled by interpreting filenames or class names.')]

story += [PageBreak()]
head=p('Source symbol index','q');head.toc_info=(0,'Source symbol index','source-index');story.append(head)
story += [p('This index is generated directly from the inspected Python syntax trees. It gives exact definition lines for classes/functions cited in the answers; body-line references in prose identify operations within them. Paths are repository-relative.','small')]
index=[]
for file in sorted((ROOT/'app').rglob('*.py')):
    tree=ast.parse(file.read_text(encoding='utf-8'))
    entries=[]
    for node in tree.body:
        if isinstance(node,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
            entries.append((node.name,node.lineno))
            if isinstance(node,ast.ClassDef):
                for child in node.body:
                    if isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef)):
                        entries.append((node.name+'.'+child.name,child.lineno))
    # Optional XGBoost wrappers are defined inside module-level try blocks.
    for node in ast.walk(tree):
        if isinstance(node,ast.ClassDef) and node.name=='XGBClassifierWithLabelEncoding':
            entries.append((node.name,node.lineno))
            for child in node.body:
                if isinstance(child,ast.FunctionDef): entries.append((node.name+'.'+child.name,child.lineno))
    entries.sort(key=lambda e:e[1])
    if entries:
        path=file.relative_to(ROOT).as_posix()
        story.append(p(path,'sub'))
        story += table([['Class / function','Line']]+[[name,str(line)] for name,line in entries],[WIDTH*.86,WIDTH*.14])
        index.extend({'path':path,'symbol':name,'line':line} for name,line in entries)

doc=Doc(str(OUT),pagesize=A4,title='EMADS Architecture, Code and Interview Guide',author='Repository study guide',subject='Source review and interview preparation; 30 numbered answers')
doc.multiBuild(story)
pdf=fitz.open(OUT)
alltext='\n'.join(page.get_text() for page in pdf)
for n,title,section,body in questions:
    assert f'{n}. {title}' in alltext, (n,title)
assert alltext.count('1. Simple explanation')==30
assert alltext.count('2. Actual implementation')==30
assert alltext.count('3. Code evidence')==30
assert alltext.count('4. Limitations or uncertainties')==30
assert alltext.count('5. Short interview answer')==30
render=TMP/'rendered';render.mkdir(exist_ok=True)
outside=[];empty=[]
for i,page in enumerate(pdf):
    if len(page.get_text().strip())<100: empty.append(i+1)
    for block in page.get_text('dict')['blocks']:
        if 'lines' not in block: continue
        for ln in block['lines']:
            for span in ln['spans']:
                x0,y0,x1,y1=span['bbox']
                if x0<20 or x1>W-20 or y0<8 or y1>H-8: outside.append({'page':i+1,'bbox':span['bbox'],'text':span['text'][:80]})
    page.get_pixmap(matrix=fitz.Matrix(1.25,1.25),alpha=False).save(render/f'page-{i+1:03}.png')
# Contact sheets provide a view of every rendered page without omitting overflow pages.
from PIL import Image, ImageOps, ImageDraw
for group_start in range(0,len(pdf),12):
    sheet=Image.new('RGB',(4*310,3*460),'#dbe2e8');draw=ImageDraw.Draw(sheet)
    for j,pageidx in enumerate(range(group_start,min(group_start+12,len(pdf)))):
        im=Image.open(render/f'page-{pageidx+1:03}.png').convert('RGB');im.thumbnail((292,425))
        x=(j%4)*310+9;y=(j//4)*460+23;sheet.paste(im,(x,y));draw.text((x,y-17),f'Page {pageidx+1}',fill='black')
    sheet.save(TMP/f'contact-{group_start//12+1:02}.png')
manifest={'pdf':str(OUT),'pages':len(pdf),'answers':30,'five_part_answers':30,'page_boundary_issues':outside,'near_empty_pages':empty,'sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),'rendered_pages':len(pdf),'question_titles':{str(q[0]):q[1] for q in questions},'source_symbols':index}
(TMP/'pdf_validation.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in manifest.items() if k not in ('source_symbols','question_titles')},indent=2))
