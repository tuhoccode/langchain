from flask import Flask, Blueprint, render_template, redirect, url_for

# Định nghĩa blueprint

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/21 Bài Học Cho Thế Kỷ 21 (Tái Bản)')
def Bai_hoc_tk21():
    return render_template('21 Bài Học Cho Thế Kỷ 21 (Tái Bản).html')
if __name__ == '__main__':
    app.run(debug=True)
