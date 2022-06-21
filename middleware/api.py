from flask import Flask, request, jsonify, Response
from ChatbotUtils import *

from KBQA.modules import gpt_robot
from KBQA.modules import gossip_robot
from KBQA.modules import classifier,medical_robot
app = Flask(__name__)


@app.route('/')
def index():
    return '<h1>Hello World</h1>'

@app.route('/img/<imgName>')
def getImage(imgName):
    img_path = os.path.join(IMGPath, imgName)
    mdict = {
        'jpeg': 'image/jpeg',
        'jpg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif'
    }
    mime = mdict[imgName.split('.')[-1]]
    with open(img_path, 'rb') as f:
        image = f.read()
    return Response(image, mimetype=mime)


@app.route('/LoginService', methods=['POST'])
def loginService():
    code = request.json.get('code')
    print(code)
    user_id, session_id = getUserLoginStatus(code=code)
    if user_id is None:
        user_id=''
        session_id=''
    return jsonify(userID=user_id, sessionID=session_id)

#这里得到输入以后就可以进行分析了
@app.route('/SendMessageService', methods=['POST'])
def sendMessageService():
    sessionID = request.json.get('sessionID')

    content = request.json.get('content')
    # 判断用户是想闲聊还是任务型
    user_intent = classifier(content)

    print(user_intent)
    #闲聊
    if user_intent in ["greet","goodbye","deny","isbot"]:
    # if user_intent is "isbot":
        reply = gossip_robot(content)

    # if user_intent in ["创作"]:
    #     reply = gpt_robot(content)
    #医疗
    else:
        reply = medical_robot(content,user=None)
        reply = reply.get("replay_answer")

    return jsonify(text=reply)

# def text_replay(msg):
#     user_intent = classifier(msg['Text'])
#     print(user_intent)
#     if user_intent in ["greet","goodbye","deny","isbot"]:
#         reply = gossip_robot(user_intent)
#     elif user_intent == "accept":
#         reply = load_user_dialogue_context(msg.User['NickName'])
#         reply = reply.get("choice_answer")
#     else:
#         reply = medical_robot(msg['Text'],msg.User['NickName'])
#         if reply["slot_values"]:
#             dump_user_dialogue_context(msg.User['NickName'],reply)
#         reply = reply.get("replay_answer")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)