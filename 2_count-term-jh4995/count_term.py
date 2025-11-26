import os
import json
import re
from collections import Counter

# 기본 설정
DATA_PATH = r"C:\Users\82104\OneDrive\바탕 화면\충남대 자료\2025-2\알고리즘\167.과학기술표준분류 대응 특허 데이터\01-1.정식개방데이터\Training\01.원천데이터\unzipped"

# 정규식 패턴
KOREAN_PATTERN = re.compile(r'[가-힣]+')
ENGLISH_PATTERN = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]*')
NUMBERS_ONLY_PATTERN = re.compile(r'^\d+$')

def read_json_file(file_path):
    """JSON 파일을 읽어서 데이터를 반환"""
    with open(file_path, encoding='utf8') as f:
        return json.load(f)

def extract_text_from_document(data):
    """invention_title, abstract, claims를 추출하여 concatenate"""
    dataset = data['dataset']
    text_parts = []
    
    if 'invention_title' in dataset:
        text_parts.append(dataset['invention_title'])
    if 'abstract' in dataset:
        text_parts.append(dataset['abstract'])
    if 'claims' in dataset:
        text_parts.append(dataset['claims'])
    
    return " ".join(text_parts)

def extract_terms(text):
    """조건에 맞는 term들을 추출"""
    terms = []
    
    for token in text.split():
        # 한글 term 추출
        terms.extend(KOREAN_PATTERN.findall(token))
        
        # 영문/숫자 term 추출 (숫자만인 것 제외)
        english_terms = ENGLISH_PATTERN.findall(token)
        for term in english_terms:
            if not NUMBERS_ONLY_PATTERN.match(term):
                terms.append(term)
    
    return terms

def calculate_tf_df(documents_terms):
    """TF와 DF를 계산"""
    term_freq = Counter()  # TF: 전체 문서 집합에서의 출현 빈도
    doc_freq = Counter()   # DF: 몇 개의 문서에서 사용되었는가
    
    for doc_terms in documents_terms:
        # TF 계산
        term_freq.update(doc_terms)
        
        # DF 계산 (문서당 unique terms만)
        unique_terms = set(doc_terms)
        doc_freq.update(unique_terms)
    
    return term_freq, doc_freq

def sort_and_format_results(term_freq, doc_freq):
    """term_freq 기준으로 내림차순 정렬하여 JSON 형태로 포맷팅"""
    result = {}
    for term, tf in term_freq.most_common():
        result[term] = {
            "doc_freq": doc_freq[term],
            "term_freq": tf
        }
    return result

def save_to_json(data, output_filename):
    """데이터를 JSON 파일로 저장"""
    with open(output_filename, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main():
    """메인 실행 함수"""
    all_documents_terms = []
    
    processed_files = 0
    
    # JSON 파일들 처리
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                
                # JSON 파일 읽기
                json_data = read_json_file(file_path)
                
                # 텍스트 추출
                document_text = extract_text_from_document(json_data)
                
                # term 추출
                document_terms = extract_terms(document_text)
                if document_terms:
                    all_documents_terms.append(document_terms)
                
                processed_files += 1
                if processed_files % 10000 == 0:
                    print(f"처리된 파일: {processed_files:,}개")
    
    # TF/DF 계산
    term_freq, doc_freq = calculate_tf_df(all_documents_terms)
    
    # 결과 정렬 및 포맷팅
    formatted_results = sort_and_format_results(term_freq, doc_freq)
    
    # JSON 파일로 저장
    save_to_json(formatted_results, "term_dict.json")

if __name__ == "__main__":
    main()