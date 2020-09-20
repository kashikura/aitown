import os
from flask import Flask, request, redirect, url_for, render_template, flash, send_from_directory, session
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import time
import numpy as np
import cv2
import math

UPLOAD_FOLDER = "static"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.secret_key = 'aitown'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
   if not session.get('logged_in'):
       return render_template('login.html')
   else:
       return render_template('mainpage.html')
# ------------------------------------------------------------------
@app.route('/login', methods=['POST'])
def do_admin_login():
   if request.form['username'] == 'aitown' \
       and request.form['password'] == 'password':
       session['logged_in'] = True
   else:
       flash("名前か合言葉が違うようだ...","failed")
   return home()
# ------------------------------------------------------------------
@app.route("/logout")
def logout():
   session['logged_in'] = False
   return home()

#------------------------------------------------------------------------
@app.route('/dog_vs_cat', methods=['GET', 'POST'])
def upload_file():
    try :
      classes = ["イヌ","ネコ"]
      num_classes = len(classes)
      image_size = 50
      model = load_model('./dog_vs_cat.h5')#学習済みモデルをロードする
      if request.method == 'POST':
          if 'file' not in request.files:
              flash('ファイルがありません')
              return redirect(request.url)
          file = request.files['file']
          if file.filename == '':
              flash('ファイルがありません')
              return redirect(request.url)
          if file and allowed_file(file.filename):
              filename = secure_filename(file.filename)
              file.save(os.path.join(UPLOAD_FOLDER, filename))
              filepath = os.path.join(UPLOAD_FOLDER, filename)

              #受け取った画像を読み込み、np形式に変換
              img = image.load_img(filepath, target_size=(image_size,image_size))
              img = image.img_to_array(img)
              data = np.array([img])
              #変換したデータをモデルに渡して予測する
              result = model.predict(data)[0]
              predicted = result.argmax()
              max_prob = result[predicted] * 100# predictedに相当するidの確率を取得

              pred_answer = classes[predicted]

              return render_template("dog_vs_cat_result.html",answer=pred_answer,raw_img_url=filepath,max_prob=math.floor(max_prob))

      return render_template("dog_vs_cat.html",answer="")
    except:
      return render_template("dog_vs_cat.html",answer="")

@app.route('/what_fruits', methods=['GET', 'POST'])
def upload_file2():
    try :
      classes = ["リンゴ","ブドウ","オレンジ"]
      num_classes = len(classes)
      image_size = 50
      model = load_model('./what_fruits.h5')#学習済みモデルをロードする
      if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            #受け取った画像を読み込み、np形式に変換
            img = image.load_img(filepath, target_size=(image_size,image_size))
            img = image.img_to_array(img)
            data = np.array([img])
            #変換したデータをモデルに渡して予測する
            result = model.predict(data)[0]
            predicted = result.argmax()
            max_prob = result[predicted] * 100# predictedに相当するidの確率を取得

            pred_answer = classes[predicted]

            return render_template("what_fruits_result.html",answer=pred_answer,raw_img_url=filepath,max_prob=math.floor(max_prob))

      return render_template("what_fruits.html",answer="")
    except:
       return render_template("what_fruits.html",answer="")
if __name__ == "__main__":
    app.run()