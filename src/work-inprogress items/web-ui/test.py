from flask import Flask
app = Flask(__name__)

@app.route('/flask/')
def hello_flask():
   return 'Hello Flask'

@app.route('/python/')
def hello_python():
   return 'Hello Python'

@app.route('/hello/<name>/')
def yello_name(name):
   return 'Hello Mr.%s!' % name

if __name__ == '__main__':
   app.run(debug = True)