'''
This is the client script for testing the ZeroRPC connection
between NUC and the Desktop.
'''

import zerorpc

def test_client():
    server_ip = "10.53.21.79"
    
    client = zerorpc.Client()
    client.connect(f"tcp://{server_ip}:4242")
    
    try:
        response = client.say_hello("Python")
        print("Server response:", response)
    except Exception as e:
        print("Error:", e)
    finally:
        client.close()

if __name__ == "__main__":
    test_client()

