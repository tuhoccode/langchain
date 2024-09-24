from flask import Flask


def Create_app():
    app = Flask(__name__)
    from .users import user
    app.register_blueprint(user,url_prefix='/')
    return app
