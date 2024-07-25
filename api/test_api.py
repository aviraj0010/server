import requests

# Define the API URL
api_url = "https://python-project-lf06.onrender.com/api/predict"  # Update with the correct URL if testing remotely

# Path to the image file you want to test
file_path = 'C:/Users/chuda/Desktop/image.jpeg'  # Update with the correct path

# Open the image file in binary mode
with open(file_path, 'rb') as f:
    # Prepare the files dictionary
    files = {'file': f}
    
    # Send the POST request to the API
    response = requests.post(api_url, files=files)
    
    # Print the raw response text
    print("Response Status Code:", response.status_code)
    print("Response Text:", response.text)
