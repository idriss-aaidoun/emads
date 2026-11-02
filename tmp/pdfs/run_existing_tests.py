"""Review-only harness: isolate outputs and force offline execution."""
import os
import sys
import json
import socket
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'tmp' / 'pdfs' / 'test_run'
WORK.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
os.chdir(WORK)
os.environ['MPLBACKEND'] = 'Agg'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

def deny_network(*args, **kwargs):
    raise RuntimeError('Network disabled by review harness')

class Result(unittest.TextTestResult):
    records = []
    def startTest(self, test):
        self.started = time.monotonic()
        super().startTest(test)
    def stopTest(self, test):
        self.records.append({'test': test.id(), 'seconds': round(time.monotonic()-self.started, 3)})
        super().stopTest(test)

if __name__ == '__main__':
    from joblib import parallel_backend
    from threadpoolctl import threadpool_limits
    with patch('dotenv.load_dotenv', return_value=False), patch('socket.socket.connect', deny_network), patch('socket.create_connection', deny_network):
        from app.services.llm_service import LLMService
        with patch.object(LLMService, '_resolve_api_key', return_value=None), parallel_backend('threading'), threadpool_limits(limits=1):
            suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), top_level_dir=str(ROOT))
            start = time.monotonic()
            result = unittest.TextTestRunner(verbosity=2, resultclass=Result).run(suite)
            payload = {'command': r'.\venv\Scripts\python.exe tmp/pdfs/run_existing_tests.py', 'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped), 'seconds': round(time.monotonic()-start, 3), 'records': result.records, 'failure_details': [(t.id(), e) for t,e in result.failures], 'error_details': [(t.id(), e) for t,e in result.errors], 'conditions': 'Offline LLM; dotenv loading suppressed; socket connections blocked; joblib threading backend; native thread pools limited to one; cwd isolated.'}
            (ROOT / 'tmp' / 'pdfs' / 'test_results.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
            print(json.dumps({k:v for k,v in payload.items() if k not in ('records','failure_details','error_details')}))
            sys.exit(not result.wasSuccessful())
