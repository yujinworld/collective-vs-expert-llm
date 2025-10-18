import pandas as pd
import re

# CSV 파일 불러오기
input_file = "../data/articles/csv/news_with_article_body_test_v1.csv"
output_file = "../data/articles/csv/news_with_article_body_cleaned.csv"

df = pd.read_csv(input_file)

def clean_text(text):
    """따옴표 및 특수문자 제거"""
    if pd.isna(text):
        return text

    # 문자열로 변환
    text = str(text)

    # 모든 종류의 따옴표 제거 (큰따옴표, 작은따옴표, 스마트 따옴표)
    text = text.replace('"', '')
    text = text.replace("'", '')
    text = text.replace('"', '')
    text = text.replace('"', '')
    text = text.replace('”', '')
    text = text.replace('“', '')
    

    return text

# 모든 텍스트 컬럼에 적용
text_columns = ['title', 'description', 'article_body']
for col in text_columns:
    if col in df.columns:
        print(f"Processing column: {col}")
        df[col] = df[col].apply(clean_text)

# 결과 저장
df.to_csv(output_file, index=False, encoding='utf-8')
print(f"\n✅ 정리 완료!")
print(f"입력: {input_file}")
print(f"출력: {output_file}")
print(f"총 {len(df)}개 행 처리됨")
