import evaluate

# 分别加载 F1 和 accuracy 指标
metric = evaluate.combine(["f1", "accuracy", "precision"])
f1_metric = evaluate.load("f1")
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
# 示例预测和参考标签
predictions = [0, 1]
references = [0, 1]

# 分别计算指标
f1_result = f1_metric.compute(predictions=predictions, references=references)
accuracy_result = accuracy_metric.compute(predictions=predictions, references=references)
precision_result = precision_metric.compute(predictions=predictions, references=references)
metric_result = metric.compute(predictions=predictions, references=references)
# 输出结果
print("F1:", f1_result)
print("Accuracy:", accuracy_result)
print("Precision:", precision_result)
print("Metric:", metric_result)