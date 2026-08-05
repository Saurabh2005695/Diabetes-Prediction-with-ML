import sys
from app import app

if __name__ == '__main__':
    print("=" * 60)
    print("  Starting Diabetes ML Prediction Web Application...")
    print("  Server URL: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
