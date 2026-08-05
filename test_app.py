import unittest
import json
import pandas as pd
from app import app

class DiabetesAppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'DiaPredict', response.data)

    def test_samples_api(self):
        response = self.app.get('/api/samples')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertGreater(len(data['samples']), 0)

    def test_metrics_api(self):
        response = self.app.get('/api/metrics')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('metrics', data)

    def test_predict_low_risk(self):
        payload = {
            'Pregnancies': 1,
            'Glucose': 85,
            'BloodPressure': 68,
            'SkinThickness': 18,
            'Insulin': 60,
            'BMI': 21.5,
            'DiabetesPedigreeFunction': 0.15,
            'Age': 25
        }
        response = self.app.post('/api/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['label'], 'Non-Diabetic')

    def test_predict_colab_sample(self):
        # Exact patient input from Colab notebook CELL 35
        payload = {
            'Pregnancies': 5,
            'Glucose': 166,
            'BloodPressure': 72,
            'SkinThickness': 19,
            'Insulin': 175,
            'BMI': 25.8,
            'DiabetesPedigreeFunction': 0.587,
            'Age': 51
        }
        response = self.app.post('/api/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['label'], 'Diabetic')
        self.assertGreater(data['risk_percentage'], 50)

if __name__ == '__main__':
    unittest.main()
