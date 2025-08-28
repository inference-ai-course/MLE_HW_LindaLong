import sqlite3

DB_PATH = "storage.db"
conn = None


def create_table():

    # cursor.execute("DROP TABLE IF EXISTS documents;")
    # cursor.execute("DROP TABLE IF EXISTS doc_chunks;")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""       
        CREATE TABLE IF NOT EXISTS documents (
            doc_id    INTEGER PRIMARY KEY,
            title     TEXT,
            author    TEXT,
            year      INTEGER,
            keywords  TEXT
        )
    """)
  
    cursor.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks USING fts5(
        chunk,
        chunk_id UNINDEXED,
        doc_id UNINDEXED,
        title
    );
    """)
    conn.commit()
    conn.close()


def insert_data(doc_id: int, title: str, author: str, year: int, keywords: str, chunk_text:list):

    conn = sqlite3.connect(DB_PATH)

    # Insert document metadata
    conn.execute("INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
                (doc_id, title, author, year, keywords))
    
    # Insert chunk text into FTS table, linking by rowid
    for chunk in chunk_text:
        conn.execute("INSERT INTO doc_chunks (doc_id, chunk, chunk_id, title) VALUES (?, ?, ?, ?)",
                     (doc_id, chunk['chunk'], chunk['chunk_id'], title))
    conn.commit()
    conn.close()


def search_docs(search_terms: str, k: int) -> list:

    if not search_terms or str(search_terms).strip() == "":
        return []

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    search_terms = search_terms.replace("'", "\'")

    query = f"""
        SELECT d.doc_id, d.title, d.author, d.year, d.keywords, c.chunk_id, c.chunk
        FROM documents d, doc_chunks c
        WHERE d.doc_id = c.doc_id AND c.chunk MATCH '{search_terms}'
        LIMIT {k};
    """
    cursor.execute(query)
    results = cursor.fetchall()

    scored_results = []
    for result in results:
       scored_results.append({
            "doc_id": result[0], #doc_id
            "chunk_id": result[5], #chunk_id
            "chunk": result[6][:20] + "...", #chunk
            "title": result[1] #title
        })

    conn.close()

    return scored_results


def get_max_id() -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(doc_id) FROM documents")
    result = cursor.fetchone()
    if result and result[0] is not None:
        print(f"Max ID in documents table: {result[0]}")
    conn.close()
    return result[0] if result and result[0] is not None else 0



if __name__ == "__main__":

    create_table()
    #search_docs("neural network", 5)