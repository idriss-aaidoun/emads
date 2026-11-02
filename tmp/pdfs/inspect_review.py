"""Read-only metadata, historical report inspection, and small isolated probes."""
import os, sys, json, ast, importlib.metadata as md
sys.stderr = sys.stdout
from pathlib import Path
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
WORK = ROOT/'tmp'/'pdfs'/'probe_run'
WORK.mkdir(exist_ok=True)
os.chdir(WORK)
out = {'versions': {}}
for name in ['numpy','pandas','scipy','scikit-learn','xgboost','lightgbm','optuna','shap','langgraph','streamlit','groq','reportlab','PyMuPDF']:
    try: out['versions'][name] = md.version(name)
    except md.PackageNotFoundError: out['versions'][name] = 'not installed'
import fitz
out['historical_reports'] = []
for p in sorted((ROOT/'reports').glob('*.pdf')):
    doc = fitz.open(p)
    txt = '\n'.join(page.get_text() for page in doc)
    out['historical_reports'].append({'file':p.relative_to(ROOT).as_posix(), 'pages':len(doc), 'has_statistical_verdict':'Statistical Verdict' in txt, 'has_confidence_audit':'System Confidence Audit' in txt, 'claims_statistics_grounded':'grounded in real statistics' in txt, 'has_local_explanations':'Local Explanations' in txt})
from joblib import parallel_backend
with patch('dotenv.load_dotenv', return_value=False), parallel_backend('threading'):
    from app.services.llm_service import LLMService
    with patch.object(LLMService, '_resolve_api_key', return_value=None):
        from app.core.agents.model_selection_agent import ModelSelectionAgent, XGBClassifierWithLabelEncoding
        from app.core.agents.data_understanding_agent import DataUnderstandingAgent
        from scipy.stats import wilcoxon
        import numpy as np, pandas as pd
        a = ModelSelectionAgent()
        out['parser_negation'] = a._parse_arbitration_recommendation('I do not recommend RandomForest; choose LogisticRegression.', 'RandomForest','LogisticRegression')
        for n in [5,10]:
            out[f'wilcoxon_exact_{n}'] = float(wilcoxon(np.arange(n)+1., np.zeros(n), method='exact').pvalue)
        out['wilcoxon_all_zero_default'] = str(wilcoxon([.8]*10,[.8]*10).pvalue)
        df = pd.DataFrame({'feature':range(40),'rating':[1,2]*20})
        d = DataUnderstandingAgent()
        with patch.object(d.llm, 'generate_summary', side_effect=lambda **kw: kw['fallback_message']):
            col, decision, log = d._arbitrate_target_column(df,'rating',.05,.53)
        out['offline_target_fallback'] = {'column':col,'confidence':decision.confidence,'claims_llm_arbitration':'(LLM arbitration)' in decision.decision,'log_entries':int(log is not None)}
        model = XGBClassifierWithLabelEncoding(n_estimators=2,max_depth=1,n_jobs=1)
        X = np.arange(60).reshape(30,2)
        y = np.array([1,2,3]*10)
        model.fit(X,y)
        try:
            model.fit(X,y)
            out['xgb_refit'] = 'succeeded'
        except Exception as e: out['xgb_refit'] = type(e).__name__+': '+str(e)
        rows=[{'model_name':'A','mean_score':.8,'std_score':0.,'scoring_metric':'accuracy','fold_scores':[.8]*10},{'model_name':'B','mean_score':.8,'std_score':0.,'scoring_metric':'accuracy','fold_scores':[.8]*10}]
        with patch.object(a.llm,'generate_summary', return_value='recommend A') as mocked:
            p,entry,winner=a._compare_top_two(rows)
        out['zero_difference_comparison']={'p_value':str(p),'arbitration':entry is not None,'llm_calls':mocked.call_count}
(ROOT/'tmp'/'pdfs'/'inspection_results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
