import requests
import json

BASE_URL = 'http://localhost:5000'

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f'{BASE_URL}/api/health')
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_stats():
    """Test stats endpoint"""
    print("Testing stats endpoint...")
    response = requests.get(f'{BASE_URL}/api/stats')
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_single_classification():
    """Test single email classification"""
    print("Testing single classification...")
    
    spam_email = """
    WINNER! You have won $1,000,000 in our lottery!
    Click here NOW to claim your prize!
    This offer expires in 24 hours! Act fast!
    FREE MONEY! Limited time offer!
    """
    
    response = requests.post(
        f'{BASE_URL}/api/classify',
        json={'text': spam_email}
    )
    
    print(f"Status: {response.status_code}")
    print("Spam Email Result:")
    print(json.dumps(response.json(), indent=2))
    print()
    
    ham_email = """
    Hi John,
    Just confirming our meeting tomorrow at 3 PM.
    Please bring the quarterly reports.
    Thanks, Sarah
    """
    
    response = requests.post(
        f'{BASE_URL}/api/classify',
        json={'text': ham_email}
    )
    
    print("Ham Email Result:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_batch_classification():
    """Test batch classification"""
    print("Testing batch classification...")
    
    emails = [
        "Hey, want to grab lunch tomorrow?",
        "URGENT! Your account will be closed! Click here NOW!",
        "Meeting at 3 PM in conference room B",
        "Congratulations! You won a FREE iPhone! Claim now!"
    ]
    
    response = requests.post(
        f'{BASE_URL}/api/classify-batch',
        json={'texts': emails}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_top_spam_words():
    """Test top spam words endpoint"""
    print("Testing top spam words...")
    response = requests.get(f'{BASE_URL}/api/top-spam-words?n=10')
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

if __name__ == '__main__':
    print("="*60)
    print("API TESTING")
    print("="*60)
    print("\nMake sure the Flask server is running on http://localhost:5000\n")
    
    try:
        test_health()
        test_stats()
        test_single_classification()
        test_batch_classification()
        test_top_spam_words()
        
        print("="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to server")
        print("Please start the Flask server first using: python app.py")
    except Exception as e:
        print(f"ERROR: {e}")