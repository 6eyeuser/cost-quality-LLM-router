import sqlite3

DB_PATH = "data/router_v2.db"

def main():
    con = sqlite3.connect(DB_PATH)

    # 1. Check finish_reason values and purge truncated rows
    print("finish_reason counts:", con.execute(
        "SELECT finish_reason, COUNT(*) FROM evaluation_cache GROUP BY finish_reason"
    ).fetchall())
    cur = con.execute("DELETE FROM evaluation_cache WHERE finish_reason = 'length'")
    print(f"Deleted truncated rows: {cur.rowcount}")
    con.commit()

    # 2. Inspect 10 "wrong" allam-2-7b GSM8K answers to check for Arabic grading artifacts
    rows = con.execute("""
        SELECT query_id, response_text
        FROM evaluation_cache
        WHERE model_name = 'allam-2-7b'
          AND query_id LIKE 'gsm8k_%'
          AND is_correct = 0
        LIMIT 10
    """).fetchall()

    print(f"\nFound {len(rows)} wrong allam-2-7b GSM8K samples:")
    for qid, resp in rows:
        print(f"\n--- {qid} ---")
        print((resp or "")[:300])

    con.close()

if __name__ == "__main__":
    main()