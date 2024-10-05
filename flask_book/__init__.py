from flask import Flask
from datetime import timedelta
from flask_sqlalchemy import SQLAlchemy
from os import path

db = SQLAlchemy()

def Create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] ="anhanh"
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////media/anh/428916C82C800CE5/langchain_final/flask_book/user.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.permanent_session_lifetime = timedelta(minutes=1)

    from .users import user
    from .data_user import User
    db.init_app(app)
    app.register_blueprint(user,url_prefix='/')
    return app
