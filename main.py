from flask_book import Create_app, db
from os import path
from flask_migrate import Migrate

app = Create_app()
db_path = "sqlite:////media/anh/428916C82C800CE5/langchain_final/flask_book/user.db"
migrate = Migrate(app,db)
if __name__ == '__main__':
    if not path.exists(db_path):
        with app.app_context(): #cho phesp truy cap current_app, request, và session
            db.create_all() 
            print("created database")   
    app.run(debug = True)
