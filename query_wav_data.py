import os
import pandas as pd
from sqlalchemy import create_engine
import argparse
from dotenv import load_dotenv

# 加載 .env 文件中的環境變數
load_dotenv()


def export_query_results(
        db_url,
        table_name,
        serial_number=None,
        date_range=None,
        ai_result=None
):
    """
    查詢指定資料庫的數據，並將結果匯出至 CSV 檔案。

    參數:
    db_url (str): 資料庫連線 URL。
    table_name (str): 目標資料表名稱。
    serial_number (str, optional): 要篩選的序列號。
    date_range (str or tuple, optional): 單一天數 (YYYY-MM-DD) 或範圍 (YYYY-MM-DD,YYYY-MM-DD)。
    ai_result (int or tuple, optional): 單一數值過濾條件，或區間範圍 (低值,高值)。
    """
    # 建立資料庫連線
    engine = create_engine(db_url)

    # 構建 SQL 查詢
    query = f"SELECT serial_number, date, ai_result, wav_data FROM {table_name} WHERE 1=1"

    if serial_number:
        query += f" AND serial_number = '{serial_number}'"

    if date_range:
        if isinstance(date_range, str):
            query += f" AND date = '{date_range}'"
        elif isinstance(date_range, tuple) and len(date_range) == 2:
            query += f" AND date BETWEEN '{date_range[0]}' AND '{date_range[1]}'"

    if ai_result:
        if isinstance(ai_result, int):
            query += f" AND ai_result >= {ai_result}"
        elif isinstance(ai_result, tuple) and len(ai_result) == 2:
            query += f" AND ai_result BETWEEN {ai_result[0]} AND {ai_result[1]}"

    # 執行查詢並轉成 DataFrame
    with engine.connect() as connection:
        df = pd.read_sql(query, connection)

    # 設定輸出檔案名稱
    ai_result_str = f"{ai_result[0]}_{ai_result[1]}" if ai_result else "0"
    ai_result_str = 'range_' + ai_result_str
    if serial_number and date_range:
        output_file = f"query_results_{serial_number}_{date_range}_{ai_result_str}.csv"
    elif serial_number:
        output_file = f"query_results_{serial_number}_{ai_result_str}.csv"
    elif date_range:
        output_file = f"query_results_{date_range}_{ai_result_str}.csv"
    else:
        output_file = f"query_results_default_{ai_result_str}.csv"

    # 匯出 CSV
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"資料已成功匯出至 {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="查詢資料並匯出至 CSV")
    parser.add_argument("--serial_number", type=str, default=None, help="篩選特定 serial_number")
    parser.add_argument("--date", type=str, default=None,
                        help="指定日期 (格式: YYYY-MM-DD) 或範圍 (YYYY-MM-DD,YYYY-MM-DD)")
    parser.add_argument("--ai_result", type=str, default=0,
                        help="篩選 ai_result，單一數值或區間 (格式: 數值 或 數值,數值)")

    args = parser.parse_args()

    db_url = os.getenv("DB_URL")
    if not db_url:
        raise ValueError("請在 .env 文件中設定 DB_URL")

    date_range = args.date
    if date_range:
        date_parts = date_range.split(",")
        if len(date_parts) == 1:
            date_range = date_parts[0]
        elif len(date_parts) == 2:
            date_range = tuple(date_parts)
        else:
            raise ValueError("日期參數格式錯誤，請使用 YYYY-MM-DD 或 YYYY-MM-DD,YYYY-MM-DD")

    ai_result = args.ai_result
    if ai_result:
        ai_result_parts = ai_result.split(",")
        if len(ai_result_parts) == 1:
            ai_result = int(ai_result_parts[0])
        elif len(ai_result_parts) == 2:
            ai_result = tuple(map(int, ai_result_parts))
        else:
            raise ValueError("ai_result 參數格式錯誤，請使用 數值 或 數值,數值")

    export_query_results(
        db_url=db_url,
        table_name='map_location_web',
        serial_number=args.serial_number,
        date_range=date_range,
        ai_result=ai_result
    )

# Man Page 說明文件
"""
NAME:
    export_db_to_csv.py - 從資料庫查詢數據並匯出為 CSV 檔案

SYNOPSIS:
    python export_db_to_csv.py [OPTIONS]

DESCRIPTION:
    這個腳本會根據提供的參數，從 MariaDB 查詢數據，並匯出為 CSV。

OPTIONS:
    --table <str>          指定查詢的資料表名稱 (預設: map_location_web)
    --serial_number <str>  指定 serial_number 來篩選數據
    --date <str>           指定日期 (YYYY-MM-DD) 或範圍 (YYYY-MM-DD,YYYY-MM-DD)
    --ai_result <str>      指定 AI 結果過濾值 (格式: 單一數值 或 數值,數值)

EXAMPLES:
    1. 查詢特定日期的數據並匯出：
       python export_db_to_csv.py --date 2024-11-13

    2. 查詢 AI 結果大於等於 5 的數據：
       python export_db_to_csv.py --ai_result 5

    3. 查詢 AI 結果範圍 5 到 10：
       python export_db_to_csv.py --ai_result 5,10

AUTHOR:
    這個腳本由 AI 工程師開發，專為數據提取與處理設計。
"""
