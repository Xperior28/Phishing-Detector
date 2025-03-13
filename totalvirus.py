import requests
import json
import pandas as pd
import time

# VirusTotal API details
VT_URL = "https://www.virustotal.com/api/v3/urls"
API_KEY = "df82d2283efde4dede6dbf9e96382ff531d1a6a2308303f13b08fc926a874897"



# Input and Output file names
INPUT_CSV = "urls.csv" 
OUTPUT_EXCEL = "dataset.xlsx"

# Read URLs from CSV
df = pd.read_csv(INPUT_CSV)

results_list = []

# Process each URL
for test_url in df["url"]:
    payload = {"url": test_url}
    HEADERS = {
        "Accept": "application/json",
        "X-Apikey": API_KEY,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    response = requests.post(VT_URL, data=payload, headers=HEADERS)
    data = response.json()
    print(data)
    analysis_id = data["data"]["id"]

    analysis_url = f"https://www.virustotal.com/api/v3/analyses/{analysis_id}"
    time.sleep(15)   # Wait to allow the scan to complete

    HEADERS = {
        "Accept": "application/json",
        "X-Apikey": API_KEY,
    }
    response = requests.get(analysis_url, headers=HEADERS)
    data = response.json()
    print(data)

    flag = 0 
    results = data["data"]["attributes"]["results"]
    for engine, details in results.items():
        if details.get("result") == "phishing":
            flag = 1
            break

    label = "phishing" if flag == 1 else "clean"
    results_list.append({"url": test_url, "label": label})
    print(results_list)


output_df = pd.DataFrame(results_list)
output_df.to_excel(OUTPUT_EXCEL, index=False)

print(f"Results saved to {OUTPUT_EXCEL}")


