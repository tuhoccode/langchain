from flask_book import db

class User(db.Model):
    __tablename__ = 'user'  
    user_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100))
    password = db.Column(db.String(100))
    again_password = db.Column(db.String(100))

    def __init__(self,name,email,password, again_password):
        self.name = name
        self.email = email
        self.password = password
        self.again_password = again_password
