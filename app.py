from flask import Flask, request, jsonify
from gpt import get_answer  # 导入 gpt.py 中的 get_answer 函数
import data_analysis  # 导入 data_analysis 模块

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    user_query = data.get('query')
    
    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    try:
        # 使用 GPT 模型获取查询的初步答案
        gpt_response = get_answer(user_query)


        # 假设 gpt_response 是有效的输入数据
        

        # 调用 data_analysis 中的 main 函数进行数据分析和模型评估
        analysis_result = data_analysis.main()

        # 将 GPT 返回的响应和分析结果一起返回给前端
        return jsonify({
            "gpt_response": gpt_response,
            "analysis_results": analysis_result
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
