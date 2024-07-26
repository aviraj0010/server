import requests


api_url = "https://python-project-lf06.onrender.com/api/predict" 


file_path = 'C:/Users/avu/Desktop/image.jpeg' 


with open(file_path, 'rb') as f:
    # Prepare the files dictionary
    files = {'file': f}
    
    # Send the POST request to the API
    response = requests.post(api_url, files=files)
    
    # Print the raw response text
    print("Response Status Code:", response.status_code)
    print("Response Text:", response.text)
