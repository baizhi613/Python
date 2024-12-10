from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('hello.html', title='Home', message='Welcome to the homepage!')


if __name__ == '__main__':
    app.run(debug=True)
