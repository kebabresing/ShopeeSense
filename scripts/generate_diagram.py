import base64
import urllib.request
import json

mermaid_code = """
flowchart TD
    classDef dataFill fill:#fce4ec,stroke:#d81b60,stroke-width:2px,color:#000
    classDef infoFill fill:#e3f2fd,stroke:#1e88e5,stroke-width:2px,color:#000
    classDef knowFill fill:#e8f5e9,stroke:#43a047,stroke-width:2px,color:#000

    subgraph S1 ["1. TAHAP DATA (Sumber & Persiapan)"]
        direction TB
        A["1. Pengumpulan Data (Scraping)"]
        B["2. Preprocessing (Cleansing)"]
        C["3. Pelabelan (Rating 1-5)"]
        A --> B --> C
    end

    subgraph S2 ["2. TAHAP INFORMASI (Pemrosesan Inti)"]
        direction TB
        D["4. Ekstraksi Fitur (TF-IDF)"]
        E["5. Pemodelan & Evaluasi"]
        D --> E
    end

    subgraph S3 ["3. TAHAP PENGETAHUAN (Insight)"]
        direction TB
        F["6. Analisis Hasil"]
        G["7. Dokumentasi"]
        F --> G
    end

    S1 ===> S2 ===> S3

    class A,B,C dataFill
    class D,E infoFill
    class F,G knowFill
"""

state = {'code': mermaid_code, 'mermaid': {'theme': 'default'}}
json_str = json.dumps(state)
b64_str = base64.urlsafe_b64encode(json_str.encode('utf-8')).decode('utf-8')
url = 'https://mermaid.ink/img/' + b64_str

req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/100.0'})
try:
    with urllib.request.urlopen(req) as response:
        with open(r'd:\TUGASSSSSSSSSSSSSSSSSSS\DIP\assets\figures\diagram_metodologi.png', 'wb') as f:
            f.write(response.read())
    print("Success")
except Exception as e:
    print(f"Error: {e}")
