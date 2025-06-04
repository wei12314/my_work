from pathlib import Path
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from data_handle.coze import CozeWorkFlow
import json

def coze_request(sample):
    coze = CozeWorkFlow("pat_nPuQ8A1quO2oXV3xgyhkUbnmrzm215lrGTGjSp5QPpNxVGehkwAzZNWGhDp76aE8", "7510065004298403878")
    dialogue = str(sample["conversation"])
    parameters = {"dialogue": f"{dialogue}"}
    resp = coze.requestWorkFlow(parameters)
    return json.loads(resp['data'])