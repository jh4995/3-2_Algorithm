# 3-2_Algorithm

202101976 유종호

수행 가능한 QUERY: BM25F, AND, FIELD, PHRASE, VERBOSE

특이사항:
1. PHRASE와 AND 동시 사용시 에러 메시지 출력
2. PHRASE와 FIELD=A 동시 사용시 에러 메시지 출력
3. PHRASE와 FIELD=C 동시 사용시 에러 메시지 출력
4. AND, FIELD, PHRASE, VERBOSE를 제외한 잘못된 query가 입력된 경우, 잘못된 형식이 입력되었다는 알림 메시지 출력
5. 동일한 내용의 파일들이 중복출력되는 문제를 해결하기 위해서, indexer.py에서 seen_filenames라는 set을 정의 후 활용
6. searcher.py의 get_postings에서 기존에는 매번 파일을 읽고 List로 반환하던 방식을, 최초로 파일을 읽는 경우 Dictionary형태로 postings_cache에 저장하여서 캐시된 Dictionary를 반환하도록 수정 -> 파일 I/O횟수와 tf조회 복잡도 감소


## 구현 내용

### 1. BM25F 랭킹 모델 (기본 기능)
- **BM25F 스코어링 공식** 적용
  - `k1 = 1.1`
  - 필드별 가중치(`w_f`): Title 2.5, Abstract 1.5, Claims 1.1
  - 필드별 길이 정규화 세기(`b_f`): Title 0.3, Abstract 0.75, Claims 0.8
- **global df** 사용 (T, A, C 공통)
- IDF 계산: `log((N - df + 0.5) / (df + 0.5) + 1)`

### 2. AND Query
- `[AND]` 또는 `[A]` prefix로 활성화
- Query term이 **모두 존재하는 문서만** 추출
- 랭킹은 BM25F 사용
- FIELD 조건과 결합 가능 (예: `[AND][FIELD=T][FIELD=A]`)

### 3. PHRASE Query
- `[PHRASE]` 또는 `[P]` prefix로 활성화
- **Title 필드에서만** exact matching 검색
- 쿼리 문자열이 Title에 정확히 포함된 문서만 추출
- 후보 문서 필터링 후 문자열 매칭으로 검색 속도 최적화

### 4. Field-specific 검색
- `[FIELD=T]`, `[FIELD=A]`, `[FIELD=C]` prefix로 활성화
- 지정된 필드에서만 검색 수행
- 복수 필드 지정 가능 (예: `[FIELD=T][FIELD=C]`)
- 필드 미지정 시 전체 필드(T, A, C) 검색

### 5. VERBOSE (검색 결과 Highlighting)
- `[VERBOSE]` 또는 `[V]` prefix로 활성화
- 상위 5개 문서에 대해 검색어 하이라이팅 출력
- 검색어를 `<<...>>`로 표시
- Window 최대 크기: 80자
- 출력 필드명 명시: [TITLE], [ABSTRACT], [CLAIMS]

#### VERBOSE 세부 동작:
| 검색 모드 | 하이라이팅 방식 |
|----------|---------------|
| OR (기본) | 서로 다른 검색어가 가장 많이 등장하는 한 부분만 출력 |
| AND | 모든 term이 등장할 때까지 여러 부분(여러 필드) 출력 |
| PHRASE | Title에서 query와 정확히 일치하는 한 곳만 출력 |

---

## 특이사항

### 에러 처리
1. **PHRASE와 AND 동시 사용 시** 에러 메시지 출력
2. **PHRASE와 FIELD=A 동시 사용 시** 에러 메시지 출력
3. **PHRASE와 FIELD=C 동시 사용 시** 에러 메시지 출력
4. AND, FIELD, PHRASE, VERBOSE를 제외한 **잘못된 prefix 입력 시** 알림 메시지 출력

### 성능 최적화
5. 동일한 내용의 파일들이 중복 출력되는 문제를 해결하기 위해, `indexer.py`에서 `seen_filenames` set을 정의하여 중복 파일명 체크
6. `searcher.py`의 `get_postings`에서 기존에는 매번 파일을 읽고 List로 반환하던 방식을, **최초로 파일을 읽는 경우 Dictionary 형태로 `postings_cache`에 저장**하여 캐시된 Dictionary를 반환하도록 수정
   - 파일 I/O 횟수 감소
   - tf 조회 복잡도: O(n) → O(1)
7. 쿼리 처리 시작 시 `clear_cache()` 호출로 메모리 관리

### 쿼리 파싱
- 대소문자 구분 없이 prefix 인식 (예: `[verbose]`, `[VERBOSE]`, `[V]` 모두 동일)
- 약어 지원: `[V]`=VERBOSE, `[A]`=AND, `[P]`=PHRASE

---

## 사용 예시

```
# 기본 OR 검색
데이터 보안

# AND 검색
[AND] 위성 서버 컴퓨터 디바이스

# PHRASE 검색 (Title에서 exact matching)
[PHRASE] 바운딩 박스를 검출

# FIELD 제한 검색
[FIELD=T] 낙뢰 활동
[FIELD=A][FIELD=C] 위성 서버 디바이스

# VERBOSE 출력
[VERBOSE] 무선 유선 통신

# 복합 쿼리
[VERBOSE][AND] 호스트 디바이스 스트로브
[VERBOSE][PHRASE] 어뢰 대응
[VERBOSE][AND][FIELD=A] 터빈 증기 폭발
```

---

## 파일 구조

```
TextRetrieval/
├── index/
│   ├── doc_table.json      # 문서 테이블 (메타데이터 + 문서 정보)
│   ├── term_dict.json      # 용어 사전 (df, 필드별 포스팅 위치)
│   └── postings.bin        # 포스팅 리스트 (바이너리)
├── src/
│   ├── __init__.py
│   ├── tokenizer.py        # 한글 형태소 분석 (Komoran)
│   ├── indexer.py          # 인덱서
│   └── searcher.py         # 검색기
├── main.py
├── requirements.txt
└── README.md
```

---

## 인덱스 파일 구조

### term_dict.json
```json
{
  "위성": {
    "df": 1234,
    "T": {"start": 0, "length": 120},
    "A": {"start": 960, "length": 880},
    "C": {"start": 16640, "length": 700}
  }
}
```

### doc_table.json
```json
{
  "metadata": {
    "avgdl_T": 12.5,
    "avgdl_A": 185.3,
    "avgdl_C": 310.7
  },
  "documents": {
    "0": {
      "doc_id": 0,
      "filename": "xxx.json",
      "path": "...",
      "len_T": 12,
      "len_A": 185,
      "len_C": 310,
      "T_text": "발명의 제목..."
    }
  }
}
```