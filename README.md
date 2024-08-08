# ad_filtering website
Review Analyst Website는 광고성 블로그와 내돈내산 블로그를 실시간으로 필터링하여 각 분류의 포스팅을 요약한 뒤, 요약본 간의 의미적 유사도를 산출해 보여줍니다.
<p align="center">
<img width="722" alt="스크린샷 2024-08-08 16 32 29" src="https://github.com/user-attachments/assets/a663bbc4-78e3-4589-bc7d-bdf5c3127798">
</p>

# Filtering
검색하고 싶은 ‘소형 전자제품’을 입력하면
Naver Developers에서 제공하는 Open API가
키워드와 관련도가 높은 순서대로 정한 개수만큼 블로그 정보를 불러옵니다.
불러오는 정보는 포스팅 제목, 작성일, 해당 글의 링크이며,
우리는 가져온 링크를 이용하여 beaurifulSoup를 사용해 블로그 텍스트를 크롤링하였습니다.
이때 async를 이용하여 병렬로 처리함으로써 크롤링 속도를 향상하였습니다.
크롤링한 텍스트 데이터는 정규식을 이용한 광고성 리뷰와 정보성 리뷰 필터를 거치게 됩니다.
<p align="center">
<img width="720" alt="스크린샷 2024-08-08 16 38 29" src="https://github.com/user-attachments/assets/1ada2a24-96f5-4e9b-9690-b3c968259dcf">
</p>

# Keyword Extraction
키워드 추출단계에서는 sklearn.feature_extraction.text 을 이용하여
최종 요약본 이전 요약본을 합한 텍스트에서 중요도가 높은 키워드를 상위 8개 키워드를 추출합니다.
<p align="center">
<img width="708" alt="스크린샷 2024-08-08 16 39 15" src="https://github.com/user-attachments/assets/a32b667f-d16c-47e7-9d09-f304fd55606d">
</p>

# Summary
필터링 결과, 화면에서 보이는 텍스트는 광고성 텍스트와 정보성 텍스트로 분류되고,
나뉜 텍스트는 각각 LlaMa 모델을 이용하여 요약이 됩니다.
다만, LlaMa 모델은 한번에 사용 가능한 토큰의 제한이 있어
텍스트의 길이가 일정 토큰수를 넘어서면 본문이 여러 단위로 나뉘어 요약이 진행되고,
마지막으로 한 번 더 요약을 진행하여 최종 요약본을 생성합니다.
<p align="center">
<img width="723" alt="스크린샷 2024-08-08 16 32 46" src="https://github.com/user-attachments/assets/eebb412b-8aba-4de6-a764-664ad5c007de">
</p>

# Calculate Similarity
마지막으로 보여주는 텍스트의 유사도 또한 최종 요약본이 아닌 요약본을 합한 텍스트에서
광고성 리뷰 요약본과 정보성 리뷰 요약본을 비교하여 텍스트의 유사도를 추출합니다.
코사인 유사도의 분류에 따라 유사도가 70% 이상이면 아주 유사한 정도,
유사도가 50% 이상이면 보통으로 유사한 정도,
유사도가 그 이하이면 낮은 유사도를 보입니다.
<p align="center">
<img width="716" alt="스크린샷 2024-08-08 16 33 02" src="https://github.com/user-attachments/assets/36055e3f-b5ae-44b7-b65b-cc751c999f12">
</p>

