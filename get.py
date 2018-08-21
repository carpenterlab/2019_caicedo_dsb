import json
import os.path
import pickle
import shutil

import requests
import tqdm

headers = {
    "__RequestVerificationToken": "B2dbpMcDzRJf29dzsQZO_dzjfStLg_bHDUxQILFgpWBvRbhDIirpil-1KyCWhScvo5erwLb-VIR0KORg3mMxcl4i7Fk1",
    "Accept": "application/json", 
    "Accept-Encoding": "br, gzip, deflate", 
    "Accept-Language": "en-us", 
    "Connection": "keep-alive", 
    "Content-Type": "application/json", 
    "Cookie": "_ga=GA1.2.1910633702.1527101007; _gid=GA1.2.871612136.1527101007; intercom-lou-koj6gxx6=1; intercom-session-koj6gxx6=eWt3dmRZbi9RNm4ydWNuTDNYcW4vLy9pQkdIV1FQTmk5bTZxRVAvNzNCOXpVZHp5a1BjRHdSYzZ6SnJSbzQvVi0tZkw5NStlQ1dQdXhvUFdjZERLbm1DUT09--60237e8067058c5ac212d74694f4ee235b085d8a; ai_session=he8R+|1527101007411|1527102347219; .ASPXAUTH=FE609CF6940B0779D6F440AA0722C410765EDDC4067C07F2EE9190394BF270EC118D57B9CF7E60197E454A5766137093FBB2EE986700998972D52151C056C9038873A8E79BE5DE791381EFCBDAE23FF5EAD9C081; intercom-id-koj6gxx6=6d076101-644c-4bac-a3fa-c40bee72e10b; ka_sessionid=e24726f7bbbde2b906451bedc6801763eb909e39; ai_user=RWt4l|2018-05-23T14:38:22.528Z; ARRAffinity=b07887e32b4005f086f79c24964bb3a226fc1c1919f52825961efddcaba2c27e; __RequestVerificationToken=aanYSoPw-N8JphPIYwRBlRRHz5LHMQIXM9gdfSGGubJH42AT2s_awegsZNI_JjRnNaLfkftoPuIdDGUVKZDnlQ0hamM1", 
    "Host": "www.kaggle.com", 
    "Referer": "https://www.kaggle.com/c/data-science-bowl-2018/host", 
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_3) AppleWebKit/604.5.6 (KHTML, like Gecko) Version/11.0.3 Safari/604.5.6"
}

def download(resource, headers={}):    
    pathname = os.path.join("/storage/data/DSB2018/submissions", resource.split("/")[-1])

    if not os.path.exists(pathname):
        with requests.get(resource, headers=headers, stream=True) as response:
            with open(pathname, "wb") as stream:
                shutil.copyfileobj(response.raw, stream)

    return pathname

with open("team-submissions.pickle", "rb") as stream:
    teams = pickle.load(stream)

for team in tqdm.tqdm(teams):
    for submission in team["submissions"]:
        resource = "https://www.kaggle.com{}".format(submission["url"])

        pathname = download(resource, headers=headers)

        with open("log", "a") as stream:
            stream.write("{}\n".format(pathname))
