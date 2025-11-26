from src.indexer import Indexer
from src.searcher import Searcher

# 설정
DATA_DIR = r"C:\Users\82104\OneDrive\바탕 화면\충남대 자료\2025-2\알고리즘\167.과학기술표준분류 대응 특허 데이터\01-1.정식개방데이터\Training\01.원천데이터\unzipped"  # data path
INDEX_DIR = "index"  # indexer 출력 dir
DOC_TABLE_FILE = "doc_table.json"
TERM_DICT_FILE = "term_dict.json"
POSTINGS_FILE = "postings.bin"

if __name__ == "__main__":
    task = input("작업을 선택하세요 (index/search): ").strip().lower()
    
    if task in ("index", "i"):
        indexer = Indexer(DATA_DIR, INDEX_DIR, DOC_TABLE_FILE, 
                         TERM_DICT_FILE, POSTINGS_FILE)
        indexer.build_index()
    
    elif task in ("search", "s"):
        searcher = Searcher(INDEX_DIR, DOC_TABLE_FILE, TERM_DICT_FILE, 
                           POSTINGS_FILE)
        while True:
            input_query = input("검색어를 입력하세요: ").strip()
            if not input_query:
                break
            searcher.process_query(input_query)