from flask import render_template,redirect,jsonify,request,session,url_for
from flask import Flask
from params_req import BaseResponseParams
from datetime import datetime
import time,os
from run_tf import run_tf

# render_template渲染模版 redirect重定向 jsonify返回json格式字符串 
# request 主要使用模块的上传的功能 ression  url_for简化url更改(view转url)

# 初始化app
app = Flask(
    __name__,
)

# flask配置
app.secret_key = "a new web"
app.jinja_env.auto_reload = True

# 如果这个配置项被 True （默认值）， 
# 如果不是 XMLHttpRequest 请求的话（由 X-Requested-With 标头控制） 
# json 字符串的返回值会被打印出来。
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


# 添加路由
# (GET请求)
@app.route("/",methods=["GET"])
def home():
    requests = BaseResponseParams(request)

    params = {
        "requests":requests,
    }

    # **转换为关键字参数
    return render_template("index.html",**params)


# (POST请求)
@app.route("/upload",methods=["POST"])
def upload():
    print("upload .....")
    try:
        file = request.files.get("file")
        save_path = "./static/upload"
        out_path = "./static/outpath"
        os.makedirs(save_path,exist_ok=True)
        os.makedirs(out_path,exist_ok=True)
        file_path = os.path.join(save_path,str((time.time()*1000))+".jpg")
        out_path = os.path.join(out_path,str((time.time()*1000))+".jpg")
        file.save(file_path)
        
        res = run_tf(file_path,out_path)
        print("res: ", res)
        assert res
        assert res["code"] != 100

        # jsonify是flask提供的返回json格式字符串的高效方法
        return jsonify(**{
            "code":200,
            "msg": "upload success",
            "out_path" : res["out_path"],
            "res" : res["prediction"],
        })
    except Exception as e:
        return jsonify(**{
            "code" : 101,
            "msg": "upload failed"
        })

if __name__ == "__main__":
    # 运行
    app.run(debug=True,port=8000,host="0.0.0.0")