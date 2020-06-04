"""
Function that launches a Flask service and provides a predict function that can be called from the browser.
"""
import sys

sys.path.insert(0, ".")
import traceback
import argparse
import pandas as pd
from flasgger import Swagger
from flask import Flask
from flask import jsonify
from flask import request
from src.service.Service import FirstResponderService

service = FirstResponderService(end_date_path='./data/datasets/end_date.parquet',
                                model_path='./data/models/xgb-model.pkl')

# Creating the Flask app
app = Flask(__name__)
app.config['SWAGGER'] = {
    'title': 'Prédiction risque premiers répondants',
    'uiversion': 2
}

Swagger(app)


@app.route('/first-resp-risk/predict', methods=['GET'])
def predict():
    """
    API premiers répondants
    Appeler cet API pour appeler la fonction "predict" du modèle entraîné. Le modèle prédit le nombre d'appels sur un
    certain nombre de jours pour chaque secteur (voir les paramètres définis dans les notebooks).
    
    Pour plus d'informations sur les données météos, voir le notebook météo sous notebooks_exploration/.
    
    Avant d'appeler cet API, il est important de s'assurer d'avoir mis la base de donnée à jour et d'avoir réentraîné
    le modèle. La date de prédiction est un jour après la dernière date de la base de données d'incidents.
    ---
    tags:
      - API premiers répondants
    responses:
      500:
        description: Erreur
      200:
        description: Prédiction du nombre d'appels par secteur pour les X (selon le paramètre utilisé) prochains jours.
        schema:
          type: object
          properties:
            pred:
              type: array
              items:
                type: float
                description: prédiction pour un secteur
              description: Pandas Series au format JSON qui contient la prédiction pour chacun des secteurs
    """ 
    try:
        # Call service predict, which builds X_test and calls model.predict(X_test)
        pred = service.predict()
        # Convert Pandas Series to JSON
        response_data = pred.to_json()
        return response_data, 200
    except:
        # Error
        return jsonify(exception=traceback.format_exc()), 500


if __name__ == '__main__':
    print('Starting service  ...')

    parser = argparse.ArgumentParser(description='run flask api')
    parser.add_argument('--port', default=5000)
    args = parser.parse_args()
    print('Launching api ...')
    app.run(host='0.0.0.0', port=int(args.port), extra_files=service.dependency_files, debug=True)
