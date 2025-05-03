from flask import Flask, request, render_template, jsonify
from model import *

app = Flask(__name__)


# Route for the index page
@app.route("/")
def index():
    return render_template("index.html")


# Route to handle image upload and model processing
@app.route("/upload", methods=["POST"])
def upload_image():
    # Check if 'image' is part of the request
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    # Get the image from the request
    file = request.files["image"]

    # If no file is selected, return an error
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Open and process the image
    try:
        image_file = Image.open(file.stream)

        confidence = predict(image_file)

        if confidence < 50:
            text = f"No macular hole detected with confidence: {confidence:.2f}%"
        else:
            text = f"Macular hole detected with confidence: {confidence:.2f}%"

        return render_template(
            "result.html",
            result_text=text,
        )

    except Exception as e:
        print(e)
        return jsonify({"error": "Error processing image"}), 500


if __name__ == "__main__":
    app.run(debug=True)
