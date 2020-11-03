import requests

API_URL = 'http://localhost:5000/api/v1/predict'
data = {
    "text_narration": "pusat vulkanologi dan mitigasi bencana geologi (pvmbg), badan geologi, kementerian energi dan sumber daya mineral (esdm) menegaskan bahwa peningkatan aktivitas vulkanik gunung soputan di sulawesi utara bukan akibat gempa bumi di palu."}

post_req = requests.post(API_URL, json=data)

print(post_req.text)
