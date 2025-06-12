import requests

class CozeWorkFlow:
    def __init__(self, access_token:str, workflow_id:str):
        self.access_token = access_token
        self.workflow_id = workflow_id
    
    def requestWorkFlow(self, parameters:dict) -> dict:
        url = "https://api.coze.cn/v1/workflow/run"

        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }

        data = {
            "workflow_id": self.workflow_id,
            "parameters": parameters
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            resp = response.json()
            return resp
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            return "fail in workflow"
        except requests.exceptions.RequestException as req_err:
            print(f"Request error occurred: {req_err}")
            return "fail in workflow"

# if __name__ == "__main__":
#     coze = CozeWorkFlow("pat_nPuQ8A1quO2oXV3xgyhkUbnmrzm215lrGTGjSp5QPpNxVGehkwAzZNWGhDp76aE8", "7510065004298403878")
#     parameters = {"dialogue": "[{'role': 'user', 'content': '举而不坚，坚而不久。有时无法完成性生活。今年上半年，感觉很优秀。后来因为咳凑诊所给配的中药（12付其中有川贝粉30克）然后就出现举而不坚，坚而不久。不知道是巧合还是吃中药吃的。后来也在江阴中医院和江阴人民医院看过，医生没有做任何检查，只是配了些药。中医院配的还少胶囊和金水宝胶囊。吃了没有效果。我该怎么办？我这还有希望能治好吗？自我感觉身体挺好的，可以一口气跑8公里。身高1.64米，体重126斤。不吸烟、不喝酒。没有手淫史。平时爱锻炼自我觉得很结实。没有糖尿病和其它毛病。今年上半年还勃起特别有力，勃起后如没有性生活会翘好久'}, {'role': 'assistant', 'content': '根据你的描述，可能是之前的疾病和中药导致的。停药后应该会好转，但是失败的经历可能导致你的心理自信不足。我建议你使用西药来增强自信，自然就会好起来。'}, {'role': 'user', 'content': '中药停了3个月，后来江阴中医院和江阴人民医院配的药停了也有两个月了。现在完全找不到以前的感觉了。举而不坚，坚而不久。今晚做着做着就软了。根本没法完成性生活。'}, {'role': 'assistant', 'content': '不要有心理压力，我给你开了一盒药。你可以尝试在和妻子做爱时换个地点，这可能会有所帮助。在房事前两个小时吃药，这样效果可能会更好。'}, {'role': 'user', 'content': '赵医师，你说的这两盒药是激素药吗？我这种情况能恢复到以前的状态吗？以后是不是就得靠药物维持性生活？请问这些药是什么药，我应该去哪里购买？'}, {'role': 'assistant', 'content': '这些药不是激素药物，是一种辅助治疗的药物。你的情况是可以恢复的，不需要长期依赖药物来维持性生活。我已经给你开了一盒药，明天药师审核后你就能看到了。你可以到药店购买这些药物。'}, {'role': 'user', 'content': '谢谢医生的建议，我会按照您的指导去尝试的。'}, {'role': 'assistant', 'content': '不客气，希望你能早日康复。如果还有其他问题，请随时向我咨询。祝你身体健康！'}]"}
#     resp = coze.requestWorkFlow(parameters)
#     print(resp)