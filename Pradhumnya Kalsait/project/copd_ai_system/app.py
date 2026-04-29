from flask import Flask, render_template
from routes.stage1_routes import stage1_bp
from routes.stage2_routes import stage2_bp
from routes.pipeline_routes import pipeline_bp

def create_app():
    app = Flask(__name__)

    # Register Blueprints
    app.register_blueprint(stage1_bp)
    app.register_blueprint(stage2_bp)
    app.register_blueprint(pipeline_bp)

    @app.route("/")
    def home():
        return render_template("index.html")

    @app.route("/health")
    def health():
        return {"status": "running"}

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)