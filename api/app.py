import os

from flask import Flask, request, flash, redirect, url_for, render_template
from werkzeug.utils import secure_filename

from src.model_inference import run_one_image
from config import ALLOWED_EXTENSIONS, UPLOAD_FOLDER, INFERENCE_RESULTS_FOLDER, MAX_CONTENT_LENGTH

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            return redirect(url_for('display_result',
                                    filename=filename))
    return render_template('transform_image.html')


@app.route('/display_result/<filename>')
def display_result(filename):
    run_one_image(
        os.path.join(UPLOAD_FOLDER, filename),
        os.path.join(INFERENCE_RESULTS_FOLDER, filename),
        "../model/checkpoints/CycleGAN-apple2orange.pth.tar",
        "mps"
    )
    return render_template('display_result.html', image_name=filename)


def main():
    app.run()


if __name__ == "__main__":
    main()