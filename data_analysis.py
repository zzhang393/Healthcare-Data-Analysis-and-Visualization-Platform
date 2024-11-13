from pyspark import SparkContext
import os


# 创建Spark上下文
sc = SparkContext(appName="FraminghamRDD")

# 读取数据集为RDD
data_rdd = sc.textFile("framingham.csv").filter(lambda line: "age" not in line)  # 跳过表头


# 解析数据
def parse_line(line):
    parts = line.split(",")
    # 处理缺失值（'NA'）
    age = float(parts[0]) if parts[0] != 'NA' else 0.0
    totChol = float(parts[1]) if parts[1] != 'NA' else 0.0
    blood_pressure = float(parts[2]) if parts[2] != 'NA' else 0.0
    TenYearCHD = 1 if parts[3] == 'Y' else 0

    return {
        "age": age,
        "totChol": totChol,
        "blood_pressure": blood_pressure,
        "TenYearCHD": TenYearCHD
    }

# 转换为RDD字典
parsed_rdd = data_rdd.map(parse_line)

# 计算描述性统计
age_stats = parsed_rdd.map(lambda x: x['age']).stats()
print(f"Age statistics: {age_stats.mean}, {age_stats.stdev}, {age_stats.min}, {age_stats.max}")



from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

# 创建Spark会话
spark = SparkSession.builder.appName("FraminghamDataFrame").getOrCreate()

# 读取数据集为DataFrame
data_df = spark.read.csv("framingham.csv", header=True, inferSchema=True)

# 数据清洗：删除缺失值
data_df = data_df.na.fill(0)  # 用0填充所有缺失值

# 转换目标变量为0和1
data_df = data_df.withColumn("TenYearCHD", when(col("TenYearCHD") == 1, 1).otherwise(0))

# 将特定的列从string转换为float
columns_to_cast = ["totChol", "BPMeds", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
for col_name in columns_to_cast:
    data_df = data_df.withColumn(col_name, col(col_name).cast("float"))

# 确保没有null值
data_df = data_df.na.fill(0)  # 再次填充空值，确保所有列都为数值型

# 显示DataFrame的结构
data_df.printSchema()

# 特征选择
from pyspark.ml.feature import VectorAssembler
feature_columns = ["age", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data_transformed = assembler.transform(data_df)

# 划分训练集和测试集
train_data, test_data = data_transformed.randomSplit([0.7, 0.3])

# 训练逻辑回归模型
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(labelCol="TenYearCHD", featuresCol="features")
model = lr.fit(train_data)

# 进行预测
predictions = model.transform(test_data)

# 评估模型
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier


evaluator = MulticlassClassificationEvaluator(labelCol="TenYearCHD", predictionCol="prediction", metricName="accuracy")
# accuracy = evaluator.evaluate(predictions)
# print(f"Model accuracy: {accuracy}")

# 1. 逻辑回归模型
lr = LogisticRegression(labelCol="TenYearCHD", featuresCol="features")
lr_model = lr.fit(train_data)
lr_predictions = lr_model.transform(test_data)
lr_accuracy = evaluator.evaluate(lr_predictions)
print(f"Logistic Regression accuracy: {lr_accuracy}")

# 2. 决策树模型
dt = DecisionTreeClassifier(labelCol="TenYearCHD", featuresCol="features")
dt_model = dt.fit(train_data)
dt_predictions = dt_model.transform(test_data)
dt_accuracy = evaluator.evaluate(dt_predictions)
print(f"Decision Tree accuracy: {dt_accuracy}")

# 3. 随机森林模型
rf = RandomForestClassifier(labelCol="TenYearCHD", featuresCol="features", numTrees=10)
rf_model = rf.fit(train_data)
rf_predictions = rf_model.transform(test_data)
rf_accuracy = evaluator.evaluate(rf_predictions)
print(f"Random Forest accuracy: {rf_accuracy}")

def main():
    """
    运行数据分析、模型训练和评估的主要流程。
    返回每个模型的准确率。
    """
   
    lr_accuracy = evaluator.evaluate(lr_predictions)
    dt_accuracy = evaluator.evaluate(dt_predictions)
    rf_accuracy = evaluator.evaluate(rf_predictions)

    return {
        "Logistic Regression accuracy": lr_accuracy,
        "Decision Tree accuracy": dt_accuracy,
        "Random Forest accuracy": rf_accuracy
    }

# 确保只有直接运行时才调用 main 函数
if __name__ == "__main__":
    results = main()
    print(results)
