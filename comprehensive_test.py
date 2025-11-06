import requests
import json

BASE_URL = 'http://localhost:5000'

def print_section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def test_health():
    print_section("TEST 1: Health Check")
    try:
        response = requests.get(f'{BASE_URL}/api/health')
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        assert response.status_code == 200
        assert response.json()['status'] == 'healthy'
        print("PASSED")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def test_stats():
    print_section("TEST 2: Model Statistics")
    try:
        response = requests.get(f'{BASE_URL}/api/stats')
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Accuracy: {data['accuracy']:.2f}%")
        print(f"Precision: {data['precision']:.2f}%")
        print(f"Recall: {data['recall']:.2f}%")
        print(f"F1-Score: {data['f1_score']:.2f}%")
        print(f"Algorithm: {data['algorithm']}")
        print(f"Total Emails Trained: {data['total_emails_trained']}")
        assert response.status_code == 200
        assert data['accuracy'] > 0
        print("PASSED")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def test_spam_detection():
    print_section("TEST 3: Spam Email Detection")
    
    spam_emails = [
        "WINNER! You've won $1,000,000! Click NOW! FREE MONEY!",
        "Congratulations! Claim your FREE prize today! Limited offer!",
        "URGENT! Your account needs verification. Click here now!",
        "Get cheap viagra and pills online! Special discount!",
        "You won the lottery! Call now to claim your cash prize!"
    ]
    
    passed = 0
    failed = 0
    
    for i, email in enumerate(spam_emails, 1):
        try:
            response = requests.post(
                f'{BASE_URL}/api/classify',
                json={'text': email}
            )
            data = response.json()
            
            print(f"\nSpam Test {i}:")
            print(f"Text: {email[:60]}...")
            print(f"Prediction: {data['prediction'].upper()}")
            print(f"Confidence: {data['confidence']:.2f}%")
            
            if data['prediction'] == 'spam':
                print("PASSED")
                passed += 1
            else:
                print("FAILED - Should be spam")
                failed += 1
                
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1
    
    print(f"\nSpam Detection: {passed}/{len(spam_emails)} passed")
    return failed == 0

def test_ham_detection():
    print_section("TEST 4: Ham (Legitimate) Email Detection")
    
    ham_emails = [
        "Hi John, can we reschedule our meeting to tomorrow at 3 PM?",
        "The quarterly report is ready for review. Please check your inbox.",
        "Thank you for your purchase. Your order will arrive in 3-5 days.",
        "Meeting reminder: Team standup at 10 AM in conference room B.",
        "Could you please send me the latest project documentation?"
    ]
    
    passed = 0
    failed = 0
    
    for i, email in enumerate(ham_emails, 1):
        try:
            response = requests.post(
                f'{BASE_URL}/api/classify',
                json={'text': email}
            )
            data = response.json()
            
            print(f"\nHam Test {i}:")
            print(f"Text: {email[:60]}...")
            print(f"Prediction: {data['prediction'].upper()}")
            print(f"Confidence: {data['confidence']:.2f}%")
            
            if data['prediction'] == 'ham':
                print("PASSED")
                passed += 1
            else:
                print("FAILED - Should be ham")
                failed += 1
                
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1
    
    print(f"\nHam Detection: {passed}/{len(ham_emails)} passed")
    return failed == 0

def test_batch_classification():
    print_section("TEST 5: Batch Classification")
    
    emails = [
        "Meeting at 3 PM tomorrow",
        "WIN FREE MONEY NOW!!!",
        "Can you review the document?",
        "CLICK HERE for amazing prizes!"
    ]
    
    try:
        response = requests.post(
            f'{BASE_URL}/api/classify-batch',
            json={'texts': emails}
        )
        data = response.json()
        
        print(f"Status Code: {response.status_code}")
        print(f"Total Emails Classified: {data['count']}")
        
        for i, result in enumerate(data['results'], 1):
            print(f"\nEmail {i}: {emails[i-1][:50]}...")
            print(f"  Prediction: {result['prediction'].upper()}")
            print(f"  Confidence: {result['confidence']:.2f}%")
        
        assert response.status_code == 200
        assert data['count'] == len(emails)
        print("\nPASSED")
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def test_top_spam_words():
    print_section("TEST 6: Top Spam Indicator Words")
    
    try:
        response = requests.get(f'{BASE_URL}/api/top-spam-words?n=15')
        data = response.json()
        
        print(f"Status Code: {response.status_code}")
        print(f"Total Words: {data['count']}")
        print("\nTop Spam Words:")
        
        for i, word in enumerate(data['words'], 1):
            print(f"  {i:2d}. {word}")
        
        assert response.status_code == 200
        assert len(data['words']) > 0
        print("\nPASSED")
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def test_error_handling():
    print_section("TEST 7: Error Handling")
    
    tests_passed = 0
    
    print("\n7a. Empty text test:")
    try:
        response = requests.post(
            f'{BASE_URL}/api/classify',
            json={'text': ''}
        )
        if response.status_code == 400:
            print("PASSED - Correctly rejected empty text")
            tests_passed += 1
        else:
            print("FAILED - Should reject empty text")
    except Exception as e:
        print(f"FAILED: {e}")
    
    print("\n7b. Missing text field test:")
    try:
        response = requests.post(
            f'{BASE_URL}/api/classify',
            json={}
        )
        if response.status_code == 400:
            print("PASSED - Correctly rejected missing field")
            tests_passed += 1
        else:
            print("FAILED - Should reject missing field")
    except Exception as e:
        print(f"FAILED: {e}")
    
    print("\n7c. Invalid endpoint test:")
    try:
        response = requests.get(f'{BASE_URL}/api/invalid')
        if response.status_code == 404:
            print("PASSED - Correctly returned 404")
            tests_passed += 1
        else:
            print("FAILED - Should return 404")
    except Exception as e:
        print(f"FAILED: {e}")
    
    return tests_passed == 3

def test_edge_cases():
    print_section("TEST 8: Edge Cases")
    
    edge_cases = [
        ("Very short text", "ok"),
        ("Numbers only", "123456789"),
        ("Special chars", "!@#$%^&*()"),
        ("Mixed case", "HeLLo WoRLd"),
        ("Long text", "This is a very long email " * 50)
    ]
    
    passed = 0
    
    for name, text in edge_cases:
        try:
            response = requests.post(
                f'{BASE_URL}/api/classify',
                json={'text': text}
            )
            
            print(f"\n{name}:")
            if response.status_code == 200:
                data = response.json()
                print(f"  Prediction: {data['prediction'].upper()}")
                print(f"  Confidence: {data['confidence']:.2f}%")
                print("  PASSED")
                passed += 1
            else:
                print(f"  FAILED - Status: {response.status_code}")
                
        except Exception as e:
            print(f"  FAILED: {e}")
    
    print(f"\nEdge Cases: {passed}/{len(edge_cases)} passed")
    return passed == len(edge_cases)

def run_all_tests():
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  SPAM EMAIL CLASSIFIER - COMPREHENSIVE TEST SUITE".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    print("\nMake sure the Flask server is running on http://localhost:5000")
    input("Press Enter to start testing...")
    
    results = []
    
    results.append(("Health Check", test_health()))
    results.append(("Model Statistics", test_stats()))
    results.append(("Spam Detection", test_spam_detection()))
    results.append(("Ham Detection", test_ham_detection()))
    results.append(("Batch Classification", test_batch_classification()))
    results.append(("Top Spam Words", test_top_spam_words()))
    results.append(("Error Handling", test_error_handling()))
    results.append(("Edge Cases", test_edge_cases()))
    
    print_section("TEST SUMMARY")
    
    total_tests = len(results)
    passed_tests = sum(1 for _, result in results if result)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    print("\n" + "-"*70)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print("-"*70)
    
    if passed_tests == total_tests:
        print("\nCONGRATULATIONS! All tests passed!")
    else:
        print("\nSome tests failed. Please review the output above.")
    
    print("\n" + "#"*70)

if __name__ == '__main__':
    try:
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot connect to server")
        print("Please start the Flask server first using: python app.py")
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")