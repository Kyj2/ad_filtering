import streamlit as st
import urllib.request
import urllib.parse
import json
import requests
from bs4 import BeautifulSoup
import re
import asyncio
import aiohttp
import time
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
import os




async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def crawling_async(link_list):
    content = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in link_list:
            mobile_url = f"{i}".replace('://blog','://m.blog')
            task = asyncio.ensure_future(fetch(session, mobile_url))
            tasks.append(task)

        responses = await asyncio.gather(*tasks)
        for response in responses:
            soup = BeautifulSoup(response, 'html.parser')
            se_main_container = soup.select_one('.se-main-container')
            if se_main_container:
                t = se_main_container.get_text(strip=True)
                t1=t.encode('euc-kr', 'ignore').decode('euc-kr')
                content.append(t1.replace('\u200b','').replace('ㅋ','').replace('ㅎ','').replace('ㅠ','').replace('~','').replace('^',''))
            else:
                content.append("")

    return content

def ad_filter(text):
    st.caption('⏳광고 필터링을 시작합니다...⌛️')
    ad_content = []
    
    patterns1 = [
        r'원고료',
        r'제공 받',
        r'체험단',
        r'제휴 활동',
        r'소정의',
        r'해당 업체',
        r'해당 브랜드',
        r'무상으로',
        r'협찬 받',
        r'파트너스',
        r'광고 활동',
        r'광고활동',
        r'광고제휴활동'
        
    ]

    patterns2 = [
        r'내돈 내산',
        r'내돈내산',
        r'내 돈 내산',
        r'노 협찬',
        r'노 광고',
        r'내 돈 내 산',
        r'내 돈으로',
        r'내돈으로'
    ]

    for t in text:
        matched = False
        for pattern in patterns1:
            if re.search(pattern, t):
                ad_content.append(t)
                
                matched = True
                break

    filtered_ad_content = []
    for i in ad_content:
        matched = False
        for pattern in patterns2:
            if re.search(pattern, i):
                matched = True
                break
        if not matched:
            filtered_ad_content.append(i)

    return filtered_ad_content if filtered_ad_content else []

def real_filter(text, titles):
    #st.write('실제 후기 필터링을 시작합니다...')
    real_content = []
   
    patterns1 = [
        r'원고료',
        r'제공 받',
        r'체험단',
        r'제휴 활동',
        r'소정의',
        r'해당 업체',
        r'해당 브랜드',
        r'무상으로',
        r'협찬 받',
        r'파트너스',
        r'광고 활동',
        r'광고활동',
        r'광고제휴활동'
        
        
    ]
    patterns2 = [
        r'내돈 내산',
        r'내돈내산',
        r'내 돈 내산',
        r'노 협찬',
        r'노 광고',
        r'내 돈 내 산',
        r'내 돈으로',
        r'내돈으로'
        
        
    ]
    
    for t, title in zip(text, titles):
        if any(re.search(pattern, t) or re.search(pattern, title) for pattern in patterns2):
            real_content.append(t)

    real_content = [i for i in real_content if not any(re.search(pattern, i) for pattern in patterns1)]

    return real_content if real_content else []


def extract(extract_text):
    stop=['가','가까스로','가령','각','각각','각자','각종','갖고말하자면','같다','같이','개의치않고',
          '거니와','거바','거의','것','것과 같이','것들','게다가','게우다','겨우','견지에서',
         '결과에 이르다','결국','결론을 낼 수 있다','겸사겸사','고려하면','고로','곧','공동으로','과','과연',
         '관계가 있다','관계없이','관련이 있다','관하여','관한','관해서는','구','구체적으로','구토하다','그',
         '그들','그때','그래','그래도','그래서','그러나','그러니','그러니까','그러면','그러므로','그러한즉','그런 까닭에','그런데','그런즉','그럼',
        '그럼에도 불구하고','그렇게 함으로써','그렇지','그렇지 않다면','그렇지 않으면','그렇지만','그렇지않으면','그리고','그리하여','그만이다','그에 따르는',
        '그위에','그저','그중에서','그치지 않다','근거로','근거하여','기대여','기점으로','기준으로','기타','까닭으로','까악','까지','까지 미치다','까지도','꽈당',
        '끙끙','끼익','나','나머지는','남들','남짓','너','너무','너희','너희들','네','넷','년','논하지 않다','놀라다','누가 알겠는가',
       '누구','다른','다른 방면으로','다만','다섯','다소','다수','다시 말하자면','다시말하면','다음','다음에','다음으로','단지','답다','당신','당장','대로 하다',
       '대하면','대하여','대해 말하자면','대해서','댕그','더구나','더군다나','더라도','더불어','더욱더','더욱이는','도달하다','도착하다','동시에','동안','된바에야',
        '된이상','두번째로','둘','둥둥','뒤따라','뒤이어','든간에','들','등','등등','딩동','따라','따라서','따위','따지지 않다','딱','때','때가 되어',
      '때문에','또','또한','뚝뚝','라 해도','령','로','로 인하여','로부터','로써','륙','를','마음대로','마저','마저도','마치','막론하고','만 못하다','만약','만약에',
      '만은 아니다','만이 아니다','만일','만큼','말하자면','말할것도 없고','매','매번','메쓰겁다','몇','모','모두','무렵','무릎쓰고','무슨','무엇','무엇때문에','물론','및',
      '바꾸어말하면','바꾸어말하자면','바꾸어서 말하면','바꾸어서 한다면','바꿔 말하면','바로','바와같이','밖에 안된다','반대로','반대로 말하자면','반드시','버금',
      '보는데서','보다더','보드득','본대로','봐','봐라','부류의 사람들','부터','불구하고','불문하고','붕붕','비걱거리다','비교적','비길수 없다','비로소','비록','비슷하다',
     '비추어 보아','비하면','뿐만 아니라','뿐만아니라','뿐이다','삐걱','삐걱거리다','사','삼','상대적으로 말하자면','생각한대로','설령','설마','설사','셋','소생','소인','솨',
     '쉿','습니까','습니다','시각','시간','시작하여','시초에','시키다','실로','심지어','아','아니','아니나다를가','아니라면',
     '아니면','아니었다면','아래윗','아무거나','아무도','아야','아울러','아이','아이고','아이구','아이야','아이쿠','아하','아홉','안 그러면','않기 위하여','않기 위해서',
     '알 수 있다','알았어','압력밥솥은', '앗','앞에서','앞의것','야','약간','양자','어','어기여차','어느','어느 년도','어느것','어느곳','어느때','어느쪽','어느해','어디','어때','어떠한','어떤',
     '어떤것','어떤것들','어떻게','어떻해','어이','어째서','어쨋든','어쩔수 없다','어찌','어찌됏든','어찌됏어','어찌하든지','어찌하여','언제','언젠가','얼마','얼마 안 되는 것','얼마간',
     '얼마나','얼마든지','얼마만큼','얼마큼','엉엉','에','에 가서','에 달려 있다','에 대해','에 있다', '에게','에서','여','여기','여덟','여러분','여보시오','여부','여섯','여전히',
     '여차','연관되다','연이서','영','영차','옆사람','예','예를 들면','예를 들자면','예컨대','예하면','오','오로지','오르다','오자마자','오직','오호','오히려','와','와 같은 사람들','와르르',
     '와아','왜','왜냐하면','외에도','요만큼','요만한 것','요만한걸','요컨대','우르르','우리','우리들','우선','우에 종합한것과같이','운운','월','위에서 서술한바와같이','위하여','위해서',
     '윙윙','육','으로','으로 인하여','으로서','으로써','을','응','응당','의','의거하여','의지하여','의해','의해되다','의해서','이','이 되다','이 때문에','이 밖에',
     '이 외에','이 정도의','이것','이곳','이때','이라면','이래','이러이러하다','이러한','이런',
     '이럴정도로','이렇게 많은 것','이렇게되면','이렇게말하자면','이렇구나','이로 인하여','이르기까지','이리하여','이만큼','이번','이봐','이상',
     '이어서','이었다','이와 같다','이와 같은','이와 반대로','이와같다면','이외에도','이용하여','이유만으로','이젠','이지만','이쪽','이천구','이천육',
     '이천칠','이천팔','인 듯하다','인젠','일','일것이다','일곱','일단','일때','일반적으로','일지라도','임에 틀림없다','입각하여','입장에서','잇따라','있다','자','자기','자기집',
     '자마자','자신','잠깐','잠시','저','저는','저것','저것만큼','저기','저쪽','저희','전부','전자','전후','점에서 보아','정도에 이르다','제','제각기','제외하고','조금','조차','조차도',
     '졸졸','좀','좋아','좍좍','주룩주룩','주저하지 않고','줄은 몰랏다','줄은모른다','중에서','중의하나','즈음하여','즉','즉시''지든지','지만','지말고',
     '진짜로','진짜','쪽으로','차라리','참','참나','첫번째로','쳇','총적으로','총적으로 말하면','총적으로 보면','칠''콸콸','쾅쾅','쿵','타다','타인',
     '탕탕','토하다','통하여','툭','퉤','틈타','팍','팔','퍽','펄렁','하','하게될것이다','하게하다','하겠는가','하고 있다','하고있었다','하곤하였다','하구나','하기 때문에','하기 위하여',
     '하기는한데','하기만 하면','하기보다는','하기에','하나','하느니','하는 김에','하는 편이 낫다','하는것도','하는것만 못하다','하는것이 낫다','하는바','하더라도','하도다','하도록시키다',
     '하도록하다','하든지','하려고하다','하마터면','하면 할수록','하면된다','하면서','하물며','하여금','하여야','하자마자','하지 않는다면','하지 않도록','하지마',
     '하지마라','하지만','하하','한 까닭에','한 이유는','한 후','한다면','한다면 몰라도','한데','한마디','한적이있다','한켠으로는','한항목','할 따름이다',
     '할 생각이다','할 줄 안다','할 지경이다','할 힘이 있다','할때','할만하다','할망정','할뿐','할수있다','할수있어','할줄알다','할지라도','할지언정','함께',
     '해도된다','해도좋다','해봐요','해서는 안된다','해야한다','해요','했어요','향하다','향하여','향해서','허','허걱','허허','헉','헉헉','헐떡헐떡',
     '형식으로 쓰여','혹시','혹은','혼자','훨씬','휘익','휴','흐흐','흥','힘입어','ㅎㅎ', '이렇게','있는','없는','있습니다','종합','요약',
     '먹다', '커피', '제품', '음식물', '자다', '맛있다', '하고', '머신', '정도', '식기세척기', '토스트', '처리기', '해주다', '에는',
        '에어', '오븐', '너무', '가지', '때문', '재료', '주다', '프라이어', '전자레인지', '감자', '싶다', '준비', '믹서기', '생각', '정말',
        '이렇게', '주방', '드리다', '그렇다', '나오다', '없이', '세제', '해보다', '이나', '줍다', '쉬다', '이라', '인데', '레시피', '커피포트',
        '식빵', '두다', '구매', '완성', '치즈', '에도', '계란', '가다', '누르다', '전기밥솥', '오늘', '이에요', '즐기다', '경우', '요즘', '올리다',
        '간식', '옥수수', '소금', '돌리다', '오다', '확인', '단호박', '블렌더', '나다', '처럼', '한번', '밥솥', '전기포트', '시작', '원두', '치킨', 
        '말다', '가루', '그릇', '토스터', '위해', '돼다', '메뉴', '이제', '그냥', '내다', '아침', '보이다', '처음', '아주', '캡슐', '먹기', '써다', 
        '소개', '주스', '더욱', '남다', '스푼', '버터', '가전', '간장', '소스', '살짝', '진행', '여름', '지다', '특히', '우유', '물이', '알아보다',
        '뿌리다', '숟가락', '에서도', '포트', '예요', '감자전', '먼저', '생기다', '에요', '아래', '원하다', '다시', '설탕', '상품', '덥다', '쓸다', 
        '진짜', '피자', '참고', '양념', '기르다', '이유', '사실', '에스프레소', '마다', '모습', '마늘', '라면', '찾다', '고소하다', '계란찜', '안녕하다',
        '사람', '기다', '통해', '라고', '취향', '고구마', '지금', '콩나물', '그대로', '인용', '느끼다', '소용', '포스팅', '넘다', '기준', '보고', '하루',
        '반죽', '들이다', '챙기다', '덕분', '식초', '싱크대', '이고', '건강', '주전자', '역시', '라서', '가족', '돌려주다', '야채', '해드리다', '중간',
        '반찬', '느껴지다', '으로도', '당근', '기도', '냉장고', '이라고', '니까', '냄비', '식기', '토마토', '쿠쿠', '양념장', '업체', '잡다', '기계',
        '작업', '판매', '더하다', '에서는', '불다', '발생', '양파', '모델', '스무디', '작성', '호박', '구연산', '해도', '늘다', '마시버', '세이버',
        '체크', '보내다', '여행', '수도', '계속', '미리', '베이글', '아메리카노', '압력밥솥', '대해', '찍다', '방문', '해결', '필립스', '확실하다', 
        '끓다', '사다', '만들어지다', '국내', '본체', '또는', '뒤지다', '보통', '채소', '데우다', '대한', '불리다', '일이', '보기', '신경', '건강하다', 
        '참기름', '기분', '달걀', '바르다', '만원', '대파', '정보', '분유', '엄청', '하니', '포장', '창업', '키친', '용하다', '랍니', '이기', '궁금하다',
        '고기', '가득', '찹쌀', '닫다', '누룽지', '볶다', '매장', '나서다', '음료', '영상', '계시다', '바라다', '아기', '이랑', '제일', '수가', '이라서', 
        '내리다', '완전', '닌자', '라는', '깔다', '남편', '특징', '요거트', '양배추', '보시', '아무래도', '기간', '매우', '이벤트', '일리', '마루', '바쁘다',
        '접시', '항상', '오픈', '크림', '공식', '상담', '후추', '구울', '아이스', '드립', '라떼', '날씨', '전체', '붙다', '위치', '만나다', '운영', '박스',
        '무료', '튀김', '진하다', '전문', '블로그', '근데', '문의', '나가다', '해봤다', '대신', '형태', '발라', '딸기', '개월', '달콤하다', '대로', '부치다',
        '입맛', '다니다', '이라는', '덮다', '마무리', '로켓', '크게', '식다', '맛보다', '기름기', '샐러드', '인지', '식히다', '마지막', '기다리다', '리뷰',
        '분간', '저녁', '들어오다', '장소', '클리너', '보다는', '믿다', '엄마', '푸드', '무치다', '풍부하다', '살다', '맛집', '그렇게', '도로', '올라오다',
        '인하다', '제철', '주시', '거품', '이건', '유라', '서다', '지나다', '잡곡', '네스프레소', '사과', '오랜', '카라', '쿠첸', '영양', '디저트', '타블렛',
        '친구', '핸드', '부스러기', '이네', '카스테라', '물질', '주말', '생선', '블렌딩', '장비', '사이', '전혀', '감사하다', '분들', '치다', '오일', '블랙', 
        '부어', '분쇄기', '주변', '회사', '예전', '수박', '차이', '스토어', '이야기', '현미', '비리다', '담그다', '프로', '녹다', '맥주', '결정', '파슬리',
        '전자렌지', '표시', '네이버', '레이', '린스', '상황', '거나', '위즈', '식당', '백미', '이전', '남기다', '쌓이다', '전분', '가게', '경험', '벗기다', 
        '식용유', '묻다', '펴다', '가지다', '고춧가루', '달달', '여서', '듬뿍', '찾아보다', '살펴보다', '출시', '감다', '쿠키', '액체', '갖추다', '달라지다',
        '샌드위치', '그릴', '마요네즈', '비주', '찰옥수수', '레몬', '국물', '고추', '조각', '돈까스', '대다', '나누다', '으로는', '테팔', '포인트', '부다', 
        '센터', '모으다', '휴롬', '발효', '삼성', '삼겹살', '정수기', '이후', '안주', '깨다', '거리', '아직', '멀다', '약밥', '스럽게', '라이스', '개다',
        '모짜렐라', '거기', '지원', '중이', '기적', '이니', '만두', '여름철', '성품', '닿다', '몰다', '녹이다', '케어', '야하다', '최소', '취소', '집다',
        '돼지고기', '살리다', '튀기다', '단맛', '원래', '인분', '브런치', '직원', '고려', '걸다', '믹서', '비율', '펭수', '가스', '아이템', '요렇게', '켜다',
        '곁들이다', '빨리', '이국주', '링크', '최근', '킹소', '쫀득', '이지', '안내', '트레이', '비닐', '뭔가', '만의', '최화정', '미닉스', '어리다', '이유식',
        '손님', '해먹', '이미', '이며', '꼬집다', '올리브오일', '드롱기', '가전제품', '생산', '믹스', '건조기', '넣다', '없다', '아니다', '받다', '찌다', '버리다',
        '따르다', '고민', '차다', '느낌', '사진', '삶다', '과일', '방식', '과정', '일반', '밉다', '모르다', '끝나다', '밖에', '바꾸다', '줄어들다', '받고', '기능으로', 
        '다음과', '특징은', '소형가전밥솥은', '소형전기밥솥은', '평가를', '가능', '같습니다', '에그밥솥', '요리','있고','기능','기능은','위면']
    # 예시 한국어 텍스트
    example_text = extract_text
    
    # 한국어 불용어 리스트 정의
    stop_words_list = stop
    
    # TF-IDF 벡터화 객체 생성
    vectorizer = TfidfVectorizer(max_features=8, stop_words=stop_words_list)  # 최대 10개의 키워드 추출, stopword 적용
    X = vectorizer.fit_transform([example_text])
    
    # TF-IDF 가중치가 가장 높은 단어들 추출
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = X.toarray()[0]
    
    # 추출된 키워드 출력
    keywords = [feature_names[i] for i in np.argsort(-tfidf_scores)]
    
    return keywords

load_dotenv()
client = OpenAI(
    base_url="http://sionic.chat:8001/v1",
    api_key=os.getenv("LLAMA_API")
)

async def fetch_summary(content):
    try:
        response = await asyncio.get_event_loop().run_in_executor(None, lambda: client.chat.completions.create(
            model="xionic-ko-llama-3-70b",
            messages=[
                {
                    "role": "system",
                    "content": """You are an AI assistant.
                        You will be given a task.
                        I'll send you several blog posts about a specific product. Please summarize the top three mentioned features of the product in the following format. 
                        The format below is for reference, so there's no need to summarize it again.
                        Please summarize the following content based on performance criteria in 4 to 7 lines.
                        Just summarize what's on the blog
                        and answer in korean.
                        < 참고 형식 >
                        이 믹서기는 ① 빠르고 효율적인 마늘 까기와 다지기 기능으로 많은 호평을 받고 있으며, ② 디자인과 ③ 사용 편의성에서도 긍정적인 평가를 받고 있습니다. 
                        하지만 ① 포장 상태 문제로 유리 용기가 깨져 오는 경우가 있으며,
                        ② 소음과 진동이 커서 불편하다는 의견도 있습니다. 또한, 일부 리뷰에서는 ③ 제품의 내구성과 ④ 설명서 부족, 그리고 ⑤ 부가 기능에 대한 아쉬움이 지적되었습니다. 
                        """
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=200,  # 예시로 200을 설정해 보세요. 필요에 따라 조정 가능합니다.
            temperature=0
        ))
        return response.choices[0].message.content
    except Exception as e:
        return f"Error occurred: {str(e)}"

async def fetch_summaries_batch(text_chunks):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_summary(chunk) for chunk in text_chunks]
        summaries = await asyncio.gather(*tasks)
        return summaries

def calculate_cosine_similarity(text1, text2, model):
    # 문장 임베딩 생성
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    
    # 의미적 유사도 계산
    similarity = util.pytorch_cos_sim(embedding1, embedding2)[0][0].item()
    return similarity

st.title('Filtering Web Page')

client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')

key_word = st.text_input('검색할 제품명을 정확히 입력하세요')

link_list = []
titles = []

if key_word:
    encText = urllib.parse.quote(key_word)
    url = f"https://openapi.naver.com/v1/search/blog?query={encText}&display=20&start=1&sort=sim"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)

    try:
        response = urllib.request.urlopen(request)
        rescode = response.getcode()

        if rescode == 200:
            response_body = response.read().decode('utf-8')
            items = json.loads(response_body)['items']

            for item in items:
                link_list.append(item['link'])
                titles.append(item['title'].replace('<b>', ' ').replace('</b>', ''))

    except urllib.error.HTTPError as e:
        st.write(f"HTTP Error: {e.code}")

    except Exception as e:
        st.write(f"Error occurred: {e}")

    # CSS 스타일 추가
    st.markdown("""
        <style>
        .keyword-container {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .keyword-box {
            display: inline-block;
            padding: 5px 10px;
            margin: 5px;
            border-radius: 25px;
            background-color: #ADD8E6;
            color: #333;
            font-size: 14px;
        }
        </style>
        """, unsafe_allow_html=True)

    async def async_main():
        final_ad_summary = None
        final_real_summary = None
        
        # 크롤링, 필터링 등의 작업이 여기서 수행됩니다.
        content = await crawling_async(link_list)
        ad_content = ad_filter(content)
        real_content = real_filter(content, titles)

        col1, col2 = st.columns(2)
        col1.subheader(f'{len(ad_content)}개의 광고성 후기 요약')
        col2.subheader(f'{len(real_content)}개의 내돈내산 후기 요약')

        if ad_content:
            ad_text_chunks = [ad_content[i:i + 3] for i in range(0, len(ad_content), 3)]
            ad_text_chunks = [' '.join(chunk) for chunk in ad_text_chunks]

            ad_summaries = await fetch_summaries_batch(ad_text_chunks)
            combined_ad_summaries = ' '.join(ad_summaries)
            key = list(set(extract(combined_ad_summaries)))
            col1.markdown('<div class="keyword-container">' + ' '.join([f'<div class="keyword-box">{k}</div>' for k in key]) + '</div>', unsafe_allow_html=True)
            col1.markdown('<br>', unsafe_allow_html=True)  # Add empty line
            col1.markdown('종합 요약 진행중..')
            final_ad_summary = await fetch_summary(combined_ad_summaries)
            col1.success(final_ad_summary)
        else:
            col1.write('후기가 없습니다')

        
        if real_content:
            real_text_chunks = [real_content[i:i + 3] for i in range(0, len(real_content), 3)]
            real_text_chunks = [' '.join(chunk) for chunk in real_text_chunks]

            real_summaries = await fetch_summaries_batch(real_text_chunks)
            combined_real_summaries = ' '.join(real_summaries)
            key2 = list(set(extract(combined_real_summaries)))
            col2.markdown('<div class="keyword-container">' + ' '.join([f'<div class="keyword-box">{k}</div>' for k in key2]) + '</div>', unsafe_allow_html=True)
            col2.markdown('<br>', unsafe_allow_html=True)  # Add empty line
            col2.markdown('종합 요약 진행중..')
            final_real_summary = await fetch_summary(combined_real_summaries)
            col2.success(final_real_summary)
        else:
            col2.write('후기가 없습니다')

        st.subheader('광고성 후기 요약과 정보성 후기 요약 간의 의미적 유사도')
        st.caption('⏳유사도 측정을 시작합니다...⌛️')

        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        
        if final_ad_summary is not None and final_real_summary is not None:
            if final_ad_summary and final_real_summary:
                similarity = calculate_cosine_similarity(final_ad_summary, final_real_summary, model)
                if similarity >=0.70 :
                    message= " 의미적 유사도: <span style='font-size: 24px;'>{:.2f}</span>".format(similarity)
                    recommendation = "👍유사도가 70% 이상인 제품으로, 리뷰 요약이 실 사용자의 리뷰와 상당히 일치하여 추천하는 제품입니다👍" 
                elif similarity >=0.50 :
                    message = " 의미적 유사도: <span style='font-size: 24px;'>{:.2f}</span>".format(similarity)
                    recommendation = "👍유사도가 50% 이상인 제품으로, 광고가 다수의 핵심 사항을 포함하지만 정보성 리뷰를 읽어 포괄적으로 제품 정보를 이해할 것을 권장합니다👍" 
                else:
                    message=" 의미적 유사도: <span style='font-size: 24px;'>{:.2f}</span>".format(similarity)
                    recommendation = "👍유사도가 50% 미만인 제품으로, 광고에서 포착하지 못한 실 사용자들의 의견이 많으니 자세한 정보성 리뷰를 읽어 구매에 참고하시길 바랍니다👍" 
                
                
                st.markdown(f"""
                <div style="background-color: #FFFACD; padding: 5px; border-radius: 5px; ">
                <p>{message}</p>
                <p>{recommendation}</p>
                </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color: #FDD; padding: 5px; border-radius: 5px;">
                <p >의미적 유사도:0.00</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: #FDD; padding: 5px; border-radius: 5px;">
            <p>의미적 유사도:0.00</p>
            </div>
            """, unsafe_allow_html=True)


         
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(async_main())
    




