from flask import Flask
from datetime import timedelta
from flask_sqlalchemy import SQLAlchemy
import os

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "anhanh"
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'users.db')}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.permanent_session_lifetime = timedelta(minutes=1)

    from .users import user
    from .data_user import User

    db.init_app(app)
    app.register_blueprint(user, url_prefix='/')

    return app